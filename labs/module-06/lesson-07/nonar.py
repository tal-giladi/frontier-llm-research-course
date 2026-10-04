"""Lab 06.7: a tiny masked-diffusion LM against the autoregressive Baseline-0, and the cost of latent loops.

    python labs/module-06/lesson-07/nonar.py                       # free CPU, about 5 minutes (one training run)
    python labs/module-06/lesson-07/nonar.py --variant t4 --device cuda

1. Train the toy model with LLaDA's objective (``--objective diffusion``: bidirectional attention, a mask token,
   t ~ U(0, 1], loss Eq. 3) for the same 200 steps as Baseline-0 (whose run lesson 06.1 made).
2. On the same 256 held-out windows: Baseline-0's next-token loss (nats per predicted token, tokens 2..T) and the
   diffusion model's Eq. 6 upper bound on the NLL (nats per token, all T tokens, 8 Monte-Carlo draws). Both are
   per-token numbers in nats; the AR one is an exact NLL of T-1 tokens given the first, the diffusion one a bound
   on all T. Report them as such, not as a like-for-like comparison.
3. Sample 24 tokens after a 16-token prompt with low-confidence remasking (8 steps), and greedily from Baseline-0.
4. Cost arithmetic: decode steps for AR vs diffusion sampling, continuous-thought steps (Coconut), and the
   effective depth and FLOPs of a recurrent-depth model as r grows.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402
from frontierlab.blocks import train as blocks_train  # noqa: E402
from frontierlab.blocks.diffusion import nll_bound, sample  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.data.prepare import DEFAULT_OUT  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402

lab = load_path(str(Path(__file__).parent / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    for arm in ("b0", "diffusion"):
        m06.train_arm(a.variant, arm, a.seed, question=f"lesson 06.7: masked diffusion vs next-token, {arm}",
                      device=a.device)
    n, T = v["eval_n"], v["eval_T"]
    ar_loss = float(np.mean(m06.eval_losses(m06.run_dir(a.variant, "b0", a.seed), n=n, T=T, device=a.device)))
    dm = blocks_train.load_model(m06.run_dir(a.variant, "diffusion", a.seed) / "checkpoint.pt", a.device).to(a.device).eval()
    val = TokenData("val")
    starts = val.eval_windows(n, T)
    bounds = []
    for i in range(0, n, 16):
        x = torch.stack([val.window(s, T) for s in starts[i:i + 16]]).to(a.device)
        bounds.append(nll_bound(dm, x, samples=8, seed=i))
    bound = float(torch.cat(bounds).mean())
    print(f"held-out, {n} windows of {T} tokens:")
    print(f"  Baseline-0 (autoregressive) next-token loss      {ar_loss:.4f} nats/token  (exact NLL of tokens 2..T)")
    print(f"  masked diffusion, Eq. 6 upper bound on the NLL   {bound:.4f} nats/token  (bound on all T tokens, 8 draws)")
    tok = Tokenizer.from_file(str(DEFAULT_OUT / "tokenizer.json"))
    prompt = val.window(starts[3], 16).view(1, -1).to(a.device)
    out = sample(dm, prompt, gen_len=24, steps=8)
    ar = blocks_train.load_model(m06.run_dir(a.variant, "b0", a.seed) / "checkpoint.pt", a.device).to(a.device).eval()
    greedy = ar.generate(prompt, 24)
    print(f"\nprompt:     {tok.decode(prompt[0].tolist())!r}")
    print(f"diffusion:  {tok.decode(out[0, 16:].tolist())!r}   (24 tokens in 8 steps)")
    print(f"AR greedy:  {tok.decode(greedy[0, 16:].tolist())!r}   (24 tokens in 24 steps)")
    print(f"low-confidence remasking commits per step: "
          f"{[lab.commit_count(24 - 3 * k, 8 - k) for k in range(8)]} (24 masks, 8 steps)")
    print("\ncost arithmetic (forward passes over the sequence per generated block of G tokens):")
    print("  autoregressive with a KV cache: G decode steps, each over 1 new token")
    print("  masked diffusion (LLaDA): `steps` passes, each over the whole sequence (no KV cache, section 2.2)")
    print("  Coconut, k continuous thoughts: k extra decode steps before the answer; training needs k + 1 sequential passes")
    print("  recurrent depth, (l_P, l_R, l_C) = (2, 4, 2):")
    for r in (1, 4, 8, 32, 64):
        print(f"    r = {r:3d}: effective depth {lab.effective_depth(2, 4, 2, r):4d} layers; with 1.5B parameters outside "
              f"the core and 1.5B in it (paper section 4), about 2 x {(1.5 + 1.5 * r):.1f}B FLOPs per token")


if __name__ == "__main__":
    main()
