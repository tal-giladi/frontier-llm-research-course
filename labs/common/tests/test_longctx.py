"""Module 4: RoPE scaling rules, long-context attention kinds, long-document data, the extension
wrapper and Eval Suite v1. Reference values for the RoPE rules come from Hugging Face Transformers
5.18.0 (``transformers.modeling_rope_utils``), which computes them in float32: tolerance 1e-6 relative.
"""

import json
import math

import numpy as np
import pytest
import torch

import frontierlab.longctx  # noqa: F401  (registers the attention kinds)
from frontierlab.attention.base import RotaryEmbedding, apply_rope
from frontierlab.data.loader import TokenData
from frontierlab.evals import suite_v1 as v1
from frontierlab.longctx import rope as R
from frontierlab.longctx.attention import convert, nope_temperature, rope_extra
from frontierlab.longctx.data import LongDocData, doc_lengths, doc_start_windows, same_doc_context
from frontierlab.model import LM, toy
from frontierlab.testing import cache_agreement, causal_check, grad_check

# ----------------------------------------------------------------------------------- RoPE rules vs Transformers


def hf_inv_freq(rope_parameters, head_dim=64, max_pos=8192, seq_len=None):
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = transformers.Qwen3Config(hidden_size=4 * head_dim, num_attention_heads=4, head_dim=head_dim,
                                   max_position_embeddings=max_pos, rope_parameters=dict(rope_parameters))
    fn = ROPE_INIT_FUNCTIONS[rope_parameters["rope_type"]]
    inv, factor = fn(cfg, "cpu", seq_len) if seq_len else fn(cfg, "cpu")
    return inv.double(), factor


def rel_err(a, b):
    return ((a - b) / b).abs().max().item()


@pytest.mark.parametrize("hd,base,s,orig,extra", [
    (64, 1e4, 8.0, 1024, {}),
    (32, 1e4, 4.0, 256, {}),                                            # the course's toy shapes
    (64, 150000.0, 32.0, 4096, {"truncate": False}),                    # gpt-oss-20b's rope_scaling
    (128, 1e6, 4.0, 32768, {}),                                         # Qwen3's documented YaRN setting
    (64, 1e4, 16.0, 2048, {"beta_fast": 16.0, "beta_slow": 2.0}),
])
def test_yarn_matches_transformers(hd, base, s, orig, extra):
    hf, hf_factor = hf_inv_freq({"rope_type": "yarn", "rope_theta": base, "factor": s,
                                 "original_max_position_embeddings": orig, **extra}, head_dim=hd, max_pos=int(s * orig))
    ours = R.yarn_inv_freq(hd, base, s, orig, extra.get("beta_fast", 32.0), extra.get("beta_slow", 1.0),
                           truncate=extra.get("truncate", True))
    assert rel_err(ours, hf) < 1e-6
    assert abs(R.yarn_attention_factor(s) - hf_factor) < 1e-12


def test_yarn_partial_rope_matches_transformers():
    hf, _ = hf_inv_freq({"rope_type": "yarn", "rope_theta": 1e4, "factor": 4.0, "original_max_position_embeddings": 256,
                         "partial_rotary_factor": 0.5}, head_dim=32, max_pos=1024)
    spec = R.RopeScaling(type="yarn", factor=4.0, original_max_position_embeddings=256, partial_rotary_factor=0.5)
    assert hf.numel() == 8 and rel_err(spec.inv_freq(32, 1e4), hf) < 1e-6


def test_pi_ntk_llama3_match_transformers():
    hf, _ = hf_inv_freq({"rope_type": "linear", "rope_theta": 1e4, "factor": 4.0})
    assert rel_err(R.pi_inv_freq(64, 1e4, 4.0), hf) < 1e-6
    # static NTK-aware at scale s equals Transformers' dynamic NTK with factor 1 evaluated at seq_len = s * max_pos
    hf, _ = hf_inv_freq({"rope_type": "dynamic", "rope_theta": 1e4, "factor": 1.0}, max_pos=1024, seq_len=4096)
    assert rel_err(R.ntk_inv_freq(64, 1e4, 4.0), hf) < 1e-6
    hf, _ = hf_inv_freq({"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}, head_dim=128, max_pos=131072)
    assert rel_err(R.llama3_inv_freq(128, 5e5, 8.0, 8192), hf) < 1e-6


def test_proportional_matches_transformers():
    hf, _ = hf_inv_freq({"rope_type": "proportional", "rope_theta": 1e4, "partial_rotary_factor": 0.25}, head_dim=32)
    spec = R.RopeScaling(type="proportional", partial_rotary_factor=0.25)
    ours = spec.inv_freq(32, 1e4)
    assert spec.rot_dim(32) == 32 and ours.shape == hf.shape == (16,)
    assert torch.equal(ours[4:], torch.zeros(12, dtype=torch.float64)) and rel_err(ours[:4], hf[:4]) < 1e-6
    partial = R.RopeScaling(partial_rotary_factor=0.25).inv_freq(32, 1e4)   # NeoX/Qwen3-Next convention
    assert partial.shape == (4,) and abs(partial[-1] - 1e4 ** (-6 / 8)) < 1e-12   # still reaches slow frequencies


def test_rule_endpoints_by_hand():
    theta = R.default_inv_freq(32, 1e4)
    ntk = R.ntk_inv_freq(32, 1e4, 4.0)
    assert ntk[0] == theta[0] and abs(ntk[-1] / (theta[-1] / 4.0) - 1) < 1e-12   # highest kept, lowest / s
    for ramp in ("code", "paper"):
        y = R.yarn_inv_freq(32, 1e4, 4.0, 256, ramp=ramp)
        assert y[0] == theta[0] and abs(y[-1] / (theta[-1] / 4.0) - 1) < 1e-12
    # the two ramps differ in between (linear in the pair index vs linear in rotations)
    assert rel_err(R.yarn_inv_freq(32, 1e4, 4.0, 256, ramp="paper"), R.yarn_inv_freq(32, 1e4, 4.0, 256)) > 0.05
    # correction dims: rotations at the computed index equal the requested count
    i = R.yarn_correction_dim(32.0, 32, 1e4, 256)
    assert abs(256 * 1e4 ** (-2 * i / 32) / (2 * math.pi) - 32.0) < 1e-9


def test_scaled_rotary_default_equals_base_rotary():
    pos = torch.arange(50)
    a = RotaryEmbedding(32, 1e4)(pos)
    b = R.RopeScaling().rotary(32, 1e4)(pos)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("spec", [R.RopeScaling(), R.RopeScaling(type="pi", factor=4),
                                  R.RopeScaling(type="yarn", factor=4, original_max_position_embeddings=64)])
def test_scores_depend_on_relative_position_only(spec):
    """For any frequencies, <R_m q, R_n k> depends on m - n (times the attention factor squared)."""
    torch.manual_seed(0)
    rope = spec.rotary(32, 1e4).double()
    q, k = torch.randn(1, 1, 1, 32, dtype=torch.float64), torch.randn(1, 1, 1, 32, dtype=torch.float64)

    def score(m, n):
        cm, sm = rope(torch.tensor([m]))
        cn, sn = rope(torch.tensor([n]))
        return (apply_rope(q, cm.double(), sm.double()) * apply_rope(k, cn.double(), sn.double())).sum().item()
    a2 = spec.get_attention_factor() ** 2
    assert abs(score(10, 3) - score(1007, 1000)) < 1e-6
    assert abs(score(5, 5) - a2 * (q * k).sum().item()) < 1e-6        # distance 0: plain dot product x a^2


# ----------------------------------------------------------------------------------- attention kinds

KINDS = [
    ("gqa-rope-scaled", rope_extra("yarn", 4.0, 16)),
    ("gqa-rope-scaled", rope_extra("pi", 2.0)),
    ("gqa-rope-scaled", rope_extra("ntk", 4.0)),
    ("gqa-rope-scaled", {"rope": {"type": "default", "partial_rotary_factor": 0.5}}),
    ("gqa-rope-scaled", {"rope": {"type": "proportional", "partial_rotary_factor": 0.5}}),
    ("gqa-irope", {"irope": {"nope_every": 2}}),
    ("gqa-irope", {"irope": {"nope_every": 2, "temperature": True, "floor_scale": 4, "attn_scale": 0.5},
                   **rope_extra("yarn", 4.0, 16)}),
]


@pytest.mark.parametrize("kind,extra", KINDS)
def test_kinds_pass_causal_and_cache(kind, extra):
    torch.manual_seed(0)
    m = LM(toy(vocab_size=61).with_(attention=kind, extra=extra))
    assert causal_check(m, 61) < 1e-9
    assert cache_agreement(m, 61, chunk=1) < 1e-9
    assert cache_agreement(m, 61, chunk=5) < 1e-9


def test_default_scaled_kind_equals_baseline():
    torch.manual_seed(0)
    base = LM(toy(vocab_size=61))
    same = convert(base, "gqa-rope-scaled", **rope_extra("default"))
    x = torch.randint(0, 61, (2, 40))
    assert torch.equal(base(x).logits, same(x).logits)
    yarn = convert(base, "gqa-rope-scaled", **rope_extra("yarn", 4.0, 16))
    assert not torch.allclose(base(x).logits, yarn(x).logits)        # the rule actually changes the model


def test_irope_layout_and_temperature():
    m = LM(toy(vocab_size=61).with_(num_hidden_layers=8, attention="gqa-irope", extra={"irope": {"nope_every": 4}}))
    assert [layer.self_attn.use_rope for layer in m.model.layers] == [True, True, True, False] * 2
    t = nope_temperature(torch.tensor([0, 7, 8, 23, 24]), floor_scale=8, attn_scale=0.1)
    assert torch.allclose(t, 1 + 0.1 * torch.log1p(torch.tensor([0., 1, 1, 3, 3], dtype=torch.float64)))


@pytest.mark.parametrize("kind,extra", [("gqa-rope-scaled", rope_extra("yarn", 4.0, 8)),
                                         ("gqa-irope", {"irope": {"nope_every": 1, "temperature": True, "floor_scale": 2}})])
def test_scaled_attention_gradients(kind, extra):
    """float64 gradient check w.r.t. the input. qk_norm is off because the shared RMSNorm computes in
    float32 even for float64 input (Module 3 proposes the shared fix)."""
    torch.manual_seed(0)
    cfg = toy(vocab_size=61).with_(hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                   qk_norm=False, attention=kind, extra=extra)
    attn = LM(cfg).model.layers[0].self_attn.double()
    pos = torch.arange(6)
    assert grad_check(lambda x: attn(x, pos), torch.randn(1, 6, 16))


def test_partial_rope_on_trained_model_is_refused(tmp_path):
    from frontierlab.longctx.extend import build_model, build_parser
    m = LM(toy(vocab_size=64))
    torch.save({"model": m.state_dict(), "config": m.config.to_dict(), "step": 7}, tmp_path / "ck.pt")
    ok = build_parser().parse_args(["--init-from", str(tmp_path / "ck.pt"), "--rope", "yarn", "--factor", "4",
                                    "--original", "32"])
    cfg, new, init = build_model(ok, "toy", 64, 128)
    assert cfg.attention == "gqa-rope-scaled" and init["step"] == 7
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), new.state_dict().values()))
    with pytest.raises(ValueError):
        build_model(build_parser().parse_args(["--init-from", str(tmp_path / "ck.pt"), "--partial", "0.5"]), "toy", 64, 128)
    with pytest.raises(ValueError):
        build_model(ok, "pilot-10m", 64, 128)


# ----------------------------------------------------------------------------------- long-document data

@pytest.fixture
def doc_data(tmp_path):
    """Documents of lengths 5..200 tokens, each ending with the end-of-text id 0."""
    rng = np.random.default_rng(0)
    meta = {"name": "synthetic-docs", "vocab_size": 64, "eot_id": 0}
    for split, n_docs in (("train", 300), ("val", 60), ("test", 10)):
        lens = rng.integers(5, 200, size=n_docs)
        toks, starts = [], []
        for L in lens:
            starts.append(sum(len(t) for t in toks))
            toks.append(np.append(rng.integers(1, 64, size=L - 1), 0))
        np.concatenate(toks).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
        np.save(tmp_path / f"{split}_docs.npy", np.asarray(starts, dtype=np.int64))
        meta[split] = {"tokens": int(sum(lens))}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_within_doc_windows_never_cross_a_boundary(doc_data):
    d = LongDocData("train", doc_data, long_fraction=1.0)
    g = torch.Generator().manual_seed(0)
    x = d.batch(64, 100, g)
    assert (x[:, :-1] == 0).sum() == 0                     # an end-of-text token can only be the last one
    st = d.stats(100)
    assert st["documents"] == int((doc_lengths(d) >= 100).sum())
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(d.batch(8, 50, g1), d.batch(8, 50, g2))   # same generator state, same batch
    mixed = LongDocData("train", doc_data, long_fraction=0.0).batch(64, 100, torch.Generator().manual_seed(0))
    assert (mixed[:, :-1] == 0).sum() > 0                  # ordinary windows do cross boundaries


def test_same_doc_context_by_hand(doc_data):
    d = TokenData("train", doc_data)
    ctx = same_doc_context(d, 64, n_windows=32)
    assert ctx.shape == (32, 64) and (ctx[:, 0] == 0).all()
    assert ((ctx[:, 1:] == ctx[:, :-1] + 1) | (ctx[:, 1:] == 0)).all()   # grows by one, resets at a new document
    starts = doc_start_windows(d, 150)
    assert all(d.tokens[s - 1] == 0 for s in starts if s > 0)


def test_extend_runs_from_checkpoint_and_resumes(tiny_data, tmp_path):
    from frontierlab.longctx.extend import main as extend
    from frontierlab.metrics import read_jsonl
    from frontierlab.runcard import read_run_card
    from frontierlab.train.loop import main as train
    common = ["--preset", "toy", "--batch", "2", "--lr", "1e-3", "--warmup", "2", "--log-every", "2",
              "--eval-every", "100", "--eval-windows", "4", "--device", "cpu", "--data", str(tiny_data)]
    train(["--run", str(tmp_path / "base"), "--seq", "32", "--steps", "6", *common])
    args = ["--init-from", str(tmp_path / "base" / "checkpoint.pt"), "--rope", "yarn", "--factor", "4",
            "--original", "32", "--long-fraction", "1.0", "--seq", "128", "--steps", "6", *common]
    a = extend(["--run", str(tmp_path / "a"), *args])
    extend(["--run", str(tmp_path / "b"), *args, "--stop-after", "3"])
    b = extend(["--run", str(tmp_path / "b"), *args])
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    card = read_run_card(tmp_path / "a")
    assert card["config"]["attention"] == "gqa-rope-scaled" and card["parent_run"] == "base"
    assert card["longctx"]["init_step"] == 6 and card["longctx"]["rope"]["type"] == "yarn"
    assert len([r for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if r["split"] == "train"]) == 3


# ----------------------------------------------------------------------------------- Eval Suite v1

@pytest.fixture
def task_vocab():
    rng = np.random.default_rng(1)
    return v1.TaskVocab(keys=list(range(100, 140)), values=list(range(200, 240)), remember=[10, 11], ignore=[12, 11],
                        period=[13], filler=rng.integers(20, 60, size=20000))


@pytest.mark.parametrize("hops,depth,hard", [(1, 0.0, 0), (1, 0.5, 2), (1, 1.0, 1), (2, None, 0), (3, None, 0)])
def test_items_are_well_formed(task_vocab, hops, depth, hard):
    items = v1.make_items(task_vocab, 300, 20, hops=hops, depth=depth, distractors=3, hard=hard, seed=5)
    assert items == v1.make_items(task_vocab, 300, 20, hops=hops, depth=depth, distractors=3, hard=hard, seed=5)
    for it in items:
        assert len(it["ids"]) == len(it["ids_ablated"]) == 300
        assert it["ids"][-3:] == [10, 11, it["ids"][-1]] and it["ids"][-1] in task_vocab.keys
        assert len(it["candidates"]) == 4 + hard and it["answer"] in it["candidates"]
        assert v1.oracle_answer(it, task_vocab) == it["answer"]            # solvable from the context
        assert v1.oracle_answer(it, task_vocab, "ids_ablated") is None     # not solvable without the evidence
        if depth is not None:
            assert abs(it["depth"] - depth) < 0.1


def test_score_items_and_chance(task_vocab):
    torch.manual_seed(0)
    model = LM(toy(vocab_size=256))
    items = v1.make_items(task_vocab, 128, 30, hops=1, depth=0.5, distractors=3, seed=0)
    sc = v1.score_items(model, items, batch=7)
    assert len(sc) == 30 and all(s["k"] == 4 and s["logp_cand"] <= 0 for s in sc)
    summ = v1.summarize_scores(sc)
    assert summ["chance_acc"] == 0.25 and abs(summ["chance_logp_cand"] - math.log(0.25)) < 1e-12
    # an untrained model's candidate log-probability is close to uniform
    assert abs(summ["logp_cand"][0] - math.log(0.25)) < 0.05


def test_doc_position_losses_and_context_gain(doc_data):
    torch.manual_seed(0)
    model = LM(toy(vocab_size=64))
    val = TokenData("val", doc_data)
    L = v1.doc_position_losses(model, val, 120)
    n = len(doc_start_windows(val, 120))
    assert L.shape == (n, 119)
    # with W = lo, the truncated window starts at the document start: gain is exactly zero
    g = v1.context_gain(model, val, 120, W=40, lo=40, hi=120)
    assert g.shape == (n,) and np.abs(g).max() < 1e-5
    # the full-context term equals the matching slice of doc_position_losses
    model.eval()
    with torch.no_grad():
        s = doc_start_windows(val, 120)[0]
        x = val.window(s, 120)[None]
        ref = model(x, labels=x).per_token_loss[0, 79:119].mean()
        cut = val.window(s + 40, 80)[None]
        trunc = model(cut, labels=cut).per_token_loss[0, 39:].mean()
    g2 = v1.context_gain(model, val, 120, W=40, lo=80, hi=120)
    assert abs(g2[0] - float(trunc - ref)) < 1e-5
    b = v1.bucket_summary(L, [0, 30, 60, 119])
    assert [c["lo"] for c in b] == [0, 30, 60] and all(c["ci"][0] <= c["mean"] <= c["ci"][1] for c in b)


def test_effective_length_rule():
    cells = [{"length": 256, "acc": (0.9, 0.8, 0.95)}, {"length": 512, "acc": (0.8, 0.7, 0.9)},
             {"length": 1024, "acc": (0.5, 0.3, 0.6)}, {"length": 2048, "acc": (0.9, 0.75, 0.95)}]
    assert v1.effective_length(cells, 0.65) == 512        # 2048 passing again does not count
    assert v1.effective_length(cells, 0.85) is None


def test_short_context_regression_requires_same_items():
    base = {"heldout": {"windows": 3, "T": 8, "window_seed": 1234, "split": "val", "losses": [1.0, 2.0, 3.0]},
            "lambada": {"sha256": "x", "n": 2, "items": [{"logprob": -1.0}, {"logprob": -2.0}]}}
    new = json.loads(json.dumps(base))
    new["heldout"]["losses"] = [1.1, 2.1, 3.1]
    r = v1.short_context_regression(base, new, n_boot=200)
    assert abs(r["heldout_loss_diff"]["mean_diff"] - 0.1) < 1e-12 and r["lambada_logprob_diff"]["mean_diff"] == 0
    new["heldout"]["T"] = 16
    with pytest.raises(ValueError):
        v1.short_context_regression(base, new)
