"""Lab 17.1: superposition, and three kinds of sparse autoencoder judged by what the model loses when they are
spliced in.

    python labs/module-17/lesson-01/sae_lab.py                      # free CPU: about 3 minutes, plus ~8 the first time (model training)
    python labs/module-17/lesson-01/sae_lab.py --variant main --print   # main-path commands (rented GPU)

1. Your functions against the course's.
2. The toy model of superposition: represented features and dimensions per feature as sparsity grows.
3. The Module 17 language model: Baseline-0's architecture at toy size trained on Data-v0 (600 steps; trained
   here if ``runs/m17/lm`` has no checkpoint, about 8 minutes on a laptop).
4. ReLU (L1 coefficient 1 and 4), TopK (k = 16, 32) and JumpReLU (L0 coefficient 0.5 and 2) SAEs with 1,024 latents on
   the residual stream after layer 1, trained on 64K tokens; FVU, L0 and dead latents on held-out tokens;
   the splice check; next-token loss with each SAE spliced in (paired bootstrap over 64 validation windows);
   loss recovered against mean ablation.
5. What a few latents read: the contexts where they fire most.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.interp import hooks as HK
from frontierlab.interp import sae as S
from frontierlab.interp import superposition as SP
from frontierlab.interp import tasks as T
from frontierlab.labkit import load_path
from frontierlab.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
RUN = Path("runs/m17/lm")
SITE = "resid_post.1"

MAIN = """# Main path (1x L40S/A100/H100; not run in this build, part of the Module 17 pilot). Env A (see frontierlab/interp/hf.py).
# 1) The published Qwen-Scope TopK SAE (32K latents, k=50) on Qwen3-1.7B-Base, three layers: FVU, L0, delta loss, splice check.
for L in 7 14 21; do python -m frontierlab.interp.hf sae-eval --layer $L --out runs/m17/qwen-scope-l$L.json; done
# 2) Your own TopK SAE on the same site, trained with the course code (16K latents, 4M tokens), compared with the published one.
#    4M tokens x 2048 x 2 bytes (bf16) = 16 GB of host RAM; larger runs need a streaming buffer (SAELens's ActivationsStore).
python labs/module-17/lesson-01/sae_lab.py --variant main --layer 14 --tokens 4000000 --d-sae 16384 --k 50
# PROJECTED (pending pilot): collection 4M tokens x 2 x 1.4e9 non-embedding params = 1.1e16 FLOPs (seconds of H100 compute;
# the Python loop and tokenisation dominate, ~10-20 min); SAE training ~4 epochs x 4M tokens x 6 x 2 x 2048 x 16384 FLOPs
# = 6.4e16 FLOPs (minutes); evaluation minutes. Total under 1 GPU-hour.
"""


def ensure_lm():
    """The Module 17 language model (``frontierlab.interp.tasks.m17_model``): trained if needed."""
    return T.m17_model(RUN)


def windows(split: str, n: int, T_: int, seed: int) -> torch.Tensor:
    return T.data_windows(split, n, T_, seed)


def check_learner(lab):
    from frontierlab.interp.sae import SAE
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 32, generator=g)
    sae = SAE(32, 64, "topk", k=4)
    ok = torch.equal(lab.topk_code(x, 4), sae.code(x))
    print(f"  topk_code matches the course: {ok}")
    with torch.no_grad():
        xh = sae(x)
    print(f"  fvu: yours {lab.fvu(x, xh):.6f}, course {S.evaluate(sae, x)['fvu']:.6f}")


def part_superposition():
    print("\n2. Toy model of superposition: n = 20 features, m = 5 dimensions, importance 0.9^i")
    print(f"  {'sparsity':>8} {'represented':>11} {'dims/feature':>12}  feature dimensionality of the represented features")
    for s in (0.0, 0.7, 0.9, 0.97):
        st = SP.stats(SP.train(20, 5, s, steps=2000))
        dims = [round(d, 2) for d, n in zip(st["feature_dimensionality"], st["norms"]) if n > 0.5]
        print(f"  {s:8.2f} {st['represented']:11d} {st['dims_per_feature']:12.3f}  {dims}")


def part_saes(model, d_sae=1024, n_train=512, T=128, steps=2000):
    print(f"\n4. SAEs on {SITE} (d = {model.config.hidden_size}, {d_sae} latents)")
    tr = S.collect(model, windows("train", n_train, T, 0), SITE)
    va = windows("val", 64, T, 1234)
    held = S.collect(model, va, SITE)
    print(f"  training activations {tuple(tr.shape)}, held-out activations {tuple(held.shape)}")
    clean = S.spliced_losses(model, va, SITE, mode="clean")
    mean_abl = S.spliced_losses(model, va, SITE, mode="mean", mean_act=tr.mean(0))
    zero_abl = S.spliced_losses(model, va, SITE, mode="zero")
    print(f"  clean loss {clean.mean():.4f}; mean-ablated {mean_abl.mean():.4f}; zero-ablated {zero_abl.mean():.4f}")
    configs = [("relu", {"coeff": 1.0}), ("relu", {"coeff": 4.0}), ("topk", {"k": 16}), ("topk", {"k": 32}),
               ("jumprelu", {"coeff": 0.5}), ("jumprelu", {"coeff": 2.0})]
    rows, saes = [], {}
    print(f"  {'kind':9} {'setting':>12} {'FVU':>6} {'L0':>6} {'dead':>5} {'d loss':>8} {'95% CI':>18} {'recov.':>7} {'check':>8} {'s':>4}")
    for kind, kw in configs:
        t = time.time()
        sae, _ = S.train_sae(tr, kind, d_sae, steps=steps, **kw)
        ev = S.evaluate(sae, held)
        chk = S.spliced_losses(model, va, SITE, sae, "sae+error")
        spl = S.spliced_losses(model, va, SITE, sae, "sae")
        pb = paired_bootstrap(spl, clean)
        rec = S.loss_recovered(clean.mean(), spl.mean(), mean_abl.mean())
        setting = f"k={kw['k']}" if kind == "topk" else f"lam={kw['coeff']:g}"
        print(f"  {kind:9} {setting:>12} {ev['fvu']:6.3f} {ev['l0']:6.1f} {ev['dead']:5.2f} {pb['mean_diff']:+8.4f} "
              f"[{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}] {rec:7.3f} {np.abs(chk - clean).max():8.1e} {time.time() - t:4.0f}")
        rows.append((kind, setting, ev, pb, rec))
        saes[(kind, setting)] = sae
    return saes, va


def part_contexts(model, sae, va):
    try:
        from tokenizers import Tokenizer
        from frontierlab.data.prepare import DEFAULT_OUT
        tok = Tokenizer.from_file(str(Path(DEFAULT_OUT) / "tokenizer.json"))
        dec = lambda ids: tok.decode(ids)
    except Exception:                                    # pragma: no cover
        dec = lambda ids: str(ids)
    print("\n5. Top contexts of three frequently used TopK latents (k = 32); the last token is where it fires")
    z = sae.encode(S.collect(model, va, SITE))
    freq = (z > 0).float().mean(0)
    for lat in freq.argsort(descending=True)[[5, 50, 200]].tolist():
        print(f"  latent {lat} (fires on {100 * freq[lat]:.1f}% of tokens):")
        for v, left, at in S.top_contexts(sae, model, va, SITE, lat, k=4):
            print(f"     {v:6.2f}  ...{dec(left)!r} -> {dec([at])!r}")


def main_variant(a):
    """Main path: activations of Qwen3-1.7B-Base at resid_post.layer on Data-v0 text, a course TopK SAE, and the
    published Qwen-Scope SAE evaluated with the same code. Not run in this build."""
    from frontierlab.interp import hf
    dev = "cuda"
    model, tok = hf.load("qwen3-1.7b-base", dtype=torch.bfloat16, device=dev)
    T = 256
    n = a.tokens // T
    rows = []
    for i in range(0, n, 64):
        w = hf.text_windows(tok, 64, T, seed=i, split="train").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows.append(S.collect(model, w, f"resid_post.{a.layer}").to(torch.bfloat16).cpu())
    acts = torch.cat(rows)
    torch.cuda.synchronize()
    sae, _ = S.train_sae(acts.float().to(dev), "topk", a.d_sae, k=a.k, steps=max(1000, 4 * len(acts) // 4096), batch=4096)
    va = hf.text_windows(tok, 32, 128, seed=999).to(dev)
    print("course SAE:", hf.sae_eval(model, tok, sae, a.layer, windows=va))
    print("Qwen-Scope SAE:", hf.sae_eval(model, tok, hf.qwen_scope_sae(a.layer, dev).to(dev), a.layer, windows=va))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true", help="print the main-path commands and exit")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--tokens", type=int, default=4_000_000)
    ap.add_argument("--d-sae", type=int, default=16384)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    if a.print:
        print(MAIN)
        return
    if a.variant == "main":
        main_variant(a)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Windows consoles: decoded tokens
    t0 = time.time()
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    print("1. Your functions")
    try:
        check_learner(lab)
    except NotImplementedError as e:
        print(f"  {e} - finish the TODOs first (the rest of the script uses the course's code)")
    part_superposition()
    print("\n3. The Module 17 language model")
    model = ensure_lm()
    saes, va = part_saes(model)
    part_contexts(model, saes[("topk", "k=32")], va)
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
