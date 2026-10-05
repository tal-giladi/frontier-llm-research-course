"""Module 10: provenance, cross-split near-duplicates, leakage, document-masked packing, the mixture sampler
(determinism, resume, exact accounting), the training wrapper, the quality classifier, the rephrasing
accounting and the mixture regressions. No downloads: every corpus here is synthetic."""

import json

import numpy as np
import pytest
import torch

from frontierlab.datax import leakage, mixture, neardup, packing, provenance, quality, regmix, rephrase
from frontierlab.datax.mixture import MixtureSampler, MixtureSpec, SourceRef, window_counts
from frontierlab.datax.packing import document_segments, segment_ids, segments_from_eot
from frontierlab.model import LM, toy
from frontierlab.testing import cache_agreement, causal_check, equivalence, grad_check

EOT = 0


def make_source(root, n_docs=40, seed=0, vocab=64, lo=5, hi=60, name="syn", with_prov=True):
    """A Data-v0-layout folder of random documents, each ending with EOT (id 0)."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "vocab_size": vocab, "eot_id": EOT, "tokenizer_sha256": "t" * 64}
    if with_prov:
        meta["provenance"] = {"dataset": "synthetic/test", "config": None, "revision": "a" * 40,
                              "license": "ODC-By 1.0", "command": "test"}
    from frontierlab.data.prepare import sha256_file
    for split, n in (("train", n_docs), ("val", max(4, n_docs // 4)), ("test", max(4, n_docs // 4))):
        docs = [np.append(rng.integers(1, vocab, size=int(rng.integers(lo, hi))), EOT).astype(np.uint16) for _ in range(n)]
        toks = np.concatenate(docs)
        toks.tofile(root / f"{split}.bin")
        starts = np.concatenate([[0], np.cumsum([d.size for d in docs])[:-1]]).astype(np.int64)
        np.save(root / f"{split}_docs.npy", starts)
        meta[split] = {"documents": n, "tokens": int(toks.size), "bin_sha256": sha256_file(root / f"{split}.bin")}
    (root / "meta.json").write_text(json.dumps(meta))
    return root


# ------------------------------------------------------------------------------------------ packing


def test_segment_ids_worked_example():
    assert segment_ids(np.array([0, 5, 7]), 3, 6).tolist() == [0, 0, 1, 1, 2, 2]
    x = torch.tensor([[5, 6, EOT, 7, EOT, 8]])
    assert segments_from_eot(x, EOT).tolist() == [[0, 0, 0, 1, 1, 2]]


def test_useful_pair_fraction():
    assert packing.useful_pair_fraction(np.zeros(8)) == 1.0
    seg = np.repeat(np.arange(4), 4)                   # four documents of 4 tokens in 16
    assert abs(packing.useful_pair_fraction(seg) - (4 * 4 * 5) / (16 * 17)) < 1e-12


def docmask_model(vocab=61):
    torch.manual_seed(0)
    return LM(toy(vocab_size=vocab).with_(attention="gqa-docmask"))


def test_docmask_without_segments_is_gqa():
    torch.manual_seed(0)
    a = LM(toy(vocab_size=61))
    b = LM(toy(vocab_size=61).with_(attention="gqa-docmask"))
    b.load_state_dict(a.state_dict(), strict=True)        # same parameters: a gqa checkpoint loads as is
    x = torch.randint(0, 61, (2, 20))
    assert torch.equal(a(x).logits, b(x).logits)


def test_packed_masked_equals_separate_documents():
    """RoPE is relative, so a masked packed window gives each document exactly its stand-alone logits."""
    m = docmask_model().double().eval()
    g = torch.Generator().manual_seed(1)
    d1, d2, d3 = (torch.randint(1, 61, (n,), generator=g) for n in (7, 11, 6))
    packed = torch.cat([d1, d2, d3])[None]
    seg = torch.tensor([[0] * 7 + [1] * 11 + [2] * 6])
    with torch.no_grad(), document_segments(seg):
        out = m(packed).logits[0]
    with torch.no_grad():
        alone = [m(d[None]).logits[0] for d in (d1, d2, d3)]
    equivalence(out, torch.cat(alone), atol=1e-10, what="packed+masked vs separate")
    with torch.no_grad():                                  # without the mask the later documents differ
        nomask = m(packed).logits[0]
    assert (nomask[7:] - torch.cat(alone)[7:]).abs().max() > 1e-3


def test_docmask_passes_the_correctness_suite():
    m = docmask_model()
    g = torch.Generator().manual_seed(3)
    seg = torch.sort(torch.randint(0, 4, (2, 24), generator=g), dim=1).values   # 4 documents per row
    with document_segments(seg):
        assert causal_check(m, 61, T=16, split=9) < 1e-9
        assert cache_agreement(m, 61, T=24, prefix=8, chunk=1) < 1e-9
        assert cache_agreement(m, 61, T=24, prefix=8, chunk=5) < 1e-9


def test_docmask_gradient():
    torch.manual_seed(0)
    attn = packing.DocMaskGQAttention(toy(vocab_size=61).with_(hidden_size=16, num_attention_heads=2,
                                                               num_key_value_heads=1, head_dim=8)).double()
    seg = torch.tensor([[0, 0, 0, 1, 1, 2]])
    pos = torch.arange(6)
    with document_segments(seg):
        assert grad_check(lambda x: attn(x, pos), torch.randn(1, 6, 16))


def test_docmask_rejects_wrong_segments():
    m = docmask_model()
    with document_segments(torch.zeros(3, 8, dtype=torch.long)), pytest.raises(ValueError):
        m(torch.randint(0, 61, (2, 8)))


# ------------------------------------------------------------------------------------------ mixture


def test_window_counts_largest_remainder():
    assert window_counts([0.5, 0.3, 0.2], 7).tolist() == [4, 2, 1]
    assert window_counts([1, 0, 1], 10).tolist() == [5, 0, 5]
    assert window_counts([0.7, 0.3], 100).sum() == 100


def spec_for(tmp_path, block=10, seed=0):
    a = make_source(tmp_path / "a", n_docs=30, seed=1, name="a")
    b = make_source(tmp_path / "b", n_docs=12, seed=2, name="b")
    return MixtureSpec([SourceRef("a", str(a), 0.7), SourceRef("b", str(b), 0.3)], seed=seed, block=block)


def test_mixture_deterministic_and_resumable(tmp_path):
    spec = spec_for(tmp_path)
    T, B = 16, 3
    s1 = MixtureSampler(spec)
    straight = [s1.next_windows(B, T) for _ in range(30)]
    s2 = MixtureSampler(spec)
    first = [s2.next_windows(B, T) for _ in range(11)]
    s3 = MixtureSampler(MixtureSpec.from_dict(json.loads(json.dumps(spec.to_dict()))))   # a fresh process
    s3.load_state_dict(s2.state_dict())
    rest = [s3.next_windows(B, T) for _ in range(19)]
    for (x, sg, src), (y, sh, srd) in zip(straight, first + rest):
        assert torch.equal(x, y) and torch.equal(sg, sh) and src == srd


def test_mixture_accounting_is_exact(tmp_path):
    spec = spec_for(tmp_path)
    T = 16
    s = MixtureSampler(spec)
    seen = np.zeros(2, dtype=np.int64)
    for k in range(57):                                   # not a multiple of the block
        _, _, src = s.window(k, T)
        seen[src] += 1
    acc = s.accounting(57, T)
    assert [acc["sources"][n]["windows"] for n in ("a", "b")] == seen.tolist()
    assert acc["sources"]["a"]["tokens"] + acc["sources"]["b"]["tokens"] == 57 * T
    full = s.accounting(50, T)                            # after whole blocks: exactly the block counts
    assert full["sources"]["a"]["windows"] == 35 and full["sources"]["b"]["windows"] == 15


def test_mixture_segments_match_eot_and_cover_every_document(tmp_path):
    spec = spec_for(tmp_path)
    s = MixtureSampler(spec)
    T = 16
    src0 = s.sources[0]
    one_epoch_windows = []
    for k in range(400):
        toks, seg, src = s.window(k, T)
        assert segments_from_eot(torch.from_numpy(toks.astype(np.int64))[None], EOT)[0].tolist() == seg.tolist()
        if src == 0:
            one_epoch_windows.append(toks)
    stream = np.concatenate(one_epoch_windows)[:src0.total]
    # the first epoch of source a is a permutation of its documents: same multiset of tokens
    assert np.array_equal(np.sort(stream), np.sort(np.asarray(src0.data.tokens)))
    assert (stream == EOT).sum() == src0.lengths.size


def test_mixture_epochs_reported(tmp_path):
    spec = spec_for(tmp_path)
    s = MixtureSampler(spec)
    acc = s.accounting(2000, 16)
    b = acc["sources"]["b"]
    assert abs(b["epochs"] - b["tokens"] / b["unique_tokens"]) < 1e-12 and b["repeat_warning"]


def test_make_source_subset(tmp_path):
    a = make_source(tmp_path / "a", n_docs=20)
    meta = mixture.make_source_subset(a, tmp_path / "sub", [3, 1, 7], note="test")
    from frontierlab.data.loader import TokenData
    full, sub = TokenData("train", a), TokenData("train", tmp_path / "sub")
    ends = np.append(full.doc_starts[1:], len(full.tokens))
    want = np.concatenate([full.tokens[full.doc_starts[i]:ends[i]] for i in (3, 1, 7)])
    assert np.array_equal(np.asarray(sub.tokens), want) and meta["selection"]["documents"] == 3


# ------------------------------------------------------------------------------------------ wrapper


def run_wrapper(tmp_path, run, spec_path, *extra):
    from frontierlab.datax import train as dtrain
    return dtrain.main(["--mixture", str(spec_path), *extra, "--run", str(run), "--preset", "toy", "--steps", "6",
                        "--batch", "2", "--seq", "32", "--warmup", "2", "--log-every", "2", "--eval-every", "6",
                        "--eval-windows", "2", "--ckpt-every", "100", "--data", str(tmp_path / "a")])


def test_wrapper_resume_is_exact_and_card_has_accounting(tmp_path):
    spec = spec_for(tmp_path)
    p = tmp_path / "mix.json"
    spec.save(p)
    m1 = run_wrapper(tmp_path, tmp_path / "straight", p, "--doc-mask")
    run_wrapper(tmp_path, tmp_path / "resumed", p, "--doc-mask", "--stop-after", "3")
    m2 = run_wrapper(tmp_path, tmp_path / "resumed", p, "--doc-mask")
    for (n, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), n
    import yaml
    card = yaml.safe_load((tmp_path / "straight" / "run_card.yaml").read_text())
    planned = card["datax"]["planned_accounting"]
    assert planned["tokens"] == 6 * 2 * 32 and card["config"]["attention"] == "gqa-docmask"
    used = json.loads((tmp_path / "resumed" / "mixture_accounting.json").read_text())
    assert used["sources"] == planned["sources"]


def test_wrapper_init_from_and_anneal(tmp_path):
    from frontierlab.datax import train as dtrain
    spec = spec_for(tmp_path)
    p = tmp_path / "mix.json"
    spec.save(p)
    run_wrapper(tmp_path, tmp_path / "base", p)
    run_wrapper(tmp_path, tmp_path / "ann", p, "--init-from", str(tmp_path / "base" / "checkpoint.pt"), "--anneal")
    import yaml
    card = yaml.safe_load((tmp_path / "ann" / "run_card.yaml").read_text())
    assert card["parent_run"] == "base" and card["datax"]["init_step"] == 6
    assert dtrain.anneal_lr(1, 10, 1.0, 2) == 1.0 and dtrain.anneal_lr(6, 10, 1.0, 2) == 0.5
    assert dtrain.anneal_lr(10, 10, 1.0, 2) == 0.0


# ------------------------------------------------------------------------------------------ near-dups, leakage


def lorem(rng, n):
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "north", "seven", "table", "green",
             "paper", "storm", "glass", "night", "honey", "metal", "cloud", "field", "dream"]
    return " ".join(rng.choice(words, size=n)) + " " + " ".join(f"w{rng.integers(1e6)}" for _ in range(n // 2))


def test_cross_split_near_dups_found_and_verified():
    rng = np.random.default_rng(0)
    train = [lorem(rng, 120) for _ in range(60)]
    val = [lorem(rng, 120) for _ in range(10)]
    twin = train[5].split()
    twin[10], twin[50] = "changed", "edited"               # two-word edit: Jaccard well above 0.8
    val.append(" ".join(twin))
    res = neardup.cross_split_near_dups({"train": train, "val": val}, threshold=0.8)
    assert len(res["pairs"]) == 1
    p = res["pairs"][0]
    assert (p["a"], p["a_index"], p["b"], p["b_index"]) == ("train", 5, "val", 10) and p["jaccard"] > 0.8
    assert abs(p["jaccard_est"] - p["jaccard"]) < 0.15
    assert res["with_cross_split_near_dup"] == {"train": 1, "val": 1}


def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(1)
    a = neardup.shingles(neardup.doc_words(lorem(rng, 400)), 3)
    b = np.unique(np.concatenate([a[:300], neardup.shingles(neardup.doc_words(lorem(rng, 200)), 3)]))
    h = neardup.MinHasher(256, 0)
    est = neardup.estimate_jaccard(h.signature(a), h.signature(b))
    assert abs(est - neardup.jaccard(a, b)) < 0.08
    assert abs(neardup.candidate_probability(0.8, 16, 8) - (1 - (1 - 0.8 ** 8) ** 16)) < 1e-15


def test_leakage_planted_items_flagged():
    rng = np.random.default_rng(2)
    train = [lorem(rng, 200) for _ in range(100)]
    items = [lorem(rng, 40) for _ in range(20)]
    clean = leakage.NgramIndex(13).add_texts(train).finish()
    rep = leakage.leakage_report(clean, items, threshold=0.5)
    assert rep["flagged"] == 0
    leaked = leakage.NgramIndex(13).add_texts(leakage.plant(train, items[:5])).finish()
    rep = leakage.leakage_report(leaked, items, threshold=0.5)
    assert rep["flagged"] == 5 and rep["flagged_indices"] == [0, 1, 2, 3, 4]
    assert leakage.leakage_report(clean, ["too short"], threshold=0.5)["too_short"] == 1


# ------------------------------------------------------------------------------------------ provenance


def test_provenance_findings(tmp_path):
    good = make_source(tmp_path / "good")
    f = provenance.check_source(good)
    assert not [x for x in f if x.level == "BLOCK"]
    bad = make_source(tmp_path / "bad", with_prov=False)
    meta = json.loads((bad / "meta.json").read_text())
    meta["revision"] = "main"
    (bad / "meta.json").write_text(json.dumps(meta))
    levels = {x.message.split()[0]: x.level for x in provenance.check_source(bad)}
    assert levels["revision"] == "BLOCK" and levels["no"] == "BLOCK"          # unpinned, no licence
    with open(good / "train.bin", "r+b") as fh:                                 # tamper with the data
        fh.write(b"\x01\x00")
    assert any(x.level == "BLOCK" and "DIFFERS" in x.message for x in provenance.check_source(good))
    sa = make_source(tmp_path / "sa")
    meta = json.loads((sa / "meta.json").read_text())
    meta["provenance"]["license"] = "CC BY-SA 3.0 and GFDL"
    (sa / "meta.json").write_text(json.dumps(meta))
    assert any(x.level == "WARN" and "share-alike" in x.message for x in provenance.check_source(sa))
    assert any(x.level == "WARN" and "held-out" in x.message for x in provenance.check_source(good, split="val"))


def test_manifest(tmp_path):
    spec = spec_for(tmp_path)
    man = provenance.build_manifest(spec)
    assert man["mixture_digest"] == spec.digest() and len(man["sources"]) == 2
    assert abs(sum(s["weight"] for s in man["sources"]) - 1) < 1e-12


# ------------------------------------------------------------------------------------------ quality, rephrase, regmix


def test_quality_classifier_learns_a_signal():
    rng = np.random.default_rng(0)
    good_w, bad_w = ["theorem", "energy", "cell", "history", "equation"], ["click", "sale", "login", "cheap", "buy"]
    texts, scores = [], []
    for _ in range(600):
        s = rng.uniform(0, 5)
        k = int(round(s * 4))
        words = list(rng.choice(good_w, size=k)) + list(rng.choice(bad_w, size=20 - k)) + ["the", "and"]
        rng.shuffle(words)
        texts.append(" ".join(words))
        scores.append(s)
    m = quality.fit(texts[:500], scores[:500], epochs=20, lr=0.2)
    pred = quality.predict(m, texts[500:])
    assert quality.spearman(pred, scores[500:]) > 0.8
    met = quality.binary_metrics(pred >= 3, np.asarray(scores[500:]) >= 3)
    assert met["f1"] > 0.7
    assert quality.top_fraction(np.array([0.1, 0.9, 0.5, 0.9]), 0.5).tolist() == [1, 3]


def test_spearman_matches_scipy():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 50), rng.normal(size=50)
    assert abs(quality.spearman(a, b) - scipy.spearmanr(a, b).statistic) < 1e-12


def test_generation_flops_by_hand():
    # N=10, L=1, d_attn=2, P=3, G=2: 2·10·5 + 2·1·2·9 + 4·1·2·(3+4) = 100 + 36 + 56 = 192
    assert rephrase.generation_flops(10, 1, 2, 3, 2) == 192
    f = rephrase.fidelity("In 1999 the 3 rivers flooded Springfield badly.", "Springfield flooded in 1999, and 4 rivers rose.")
    assert f["numbers_kept"] == 0.5 and f["invented_numbers"] == 1 and not f["meta_text"]
    assert rephrase.fidelity("x", "Here is the rewritten text: x")["meta_text"]


def test_regmix_fits_recover_a_known_law():
    X = regmix.sample_mixtures(40, [0.4, 0.3, 0.2, 0.1], seed=0)
    assert np.allclose(X.sum(1), 1)
    t = np.array([-1.0, 0.5, -0.3, 0.8])
    y = 2.0 + 0.7 * np.exp(X @ t)
    law = regmix.fit_mixing_law(X, y)
    assert np.abs(regmix.predict(law, X) - y).max() < 1e-4
    lin = regmix.fit_ridge(X, y, l2=1e-8)
    assert regmix.rank_corr(regmix.loo_predictions("ridge", X, y, l2=1e-8), y) > 0.9
    best = regmix.best_mixture(law, 4, n=5000, top=10)
    assert np.isclose(best["mixture"].sum(), 1) and best["mixture"][0] > 0.4    # most weight on the -1 source
    assert lin["w"].shape == (5,)


def test_seed_level_and_decide():
    from frontierlab.datax import evaluate as ev
    r = ev.seed_level([3.0, 3.1, 3.2], [2.9, 3.05, 3.1])
    assert abs(r["mean_diff"] + 0.25 / 3) < 1e-12 and r["ci"][0] < r["ci"][1]
    assert ev.decide((-0.05, -0.03), 0.02) == "adopt" and ev.decide((-0.01, 0.01), 0.02) == "reject"
    assert ev.decide((-0.05, 0.01), 0.02) == "inconclusive"


# ------------------------------------------------------------------------------------------ groups (10.6)


def test_groups_helpers():
    from frontierlab.datax import groups
    f = groups.fertility(lambda s: list(s.encode("utf-8")), ["ab cd", "é"])
    assert f["tokens_per_byte"] == 1.0 and f["words"] == 3
    ref = np.arange(100.0)                                   # threshold 10 removes 10% of the reference
    tgt = np.arange(1000.0, 2000.0)
    thr = groups.quantile_threshold(ref, 10.0, tgt)
    assert abs(groups.removal_rate(tgt, lo=thr) - 0.10) < 0.002
    cs = np.array([1] * 10 + [2] * 10 + [3] * 10)
    rm = np.array([1] * 6 + [0] * 4 + [1] * 1 + [0] * 9 + [1] * 5 + [0] * 5, dtype=bool)   # rates .6 .1 .5, global .4
    w = groups.rehydration_weights(cs, rm)["weights"]
    assert w[2] == 10.0 and w[1] == 1.0 and w[3] == 1.0
    keys = ["repoA"] * 5 + ["repoB"] * 5
    sp = groups.group_split(keys, 500, 0)
    assert len(set(sp[:5])) == 1 and len(set(sp[5:])) == 1
