"""Lab 14.3: rollout systems and staleness.

    python labs/module-14/lesson-03/async_lab.py                    # free CPU, about 10 minutes
    python labs/module-14/lesson-03/async_lab.py --part schedule    # seconds
    python labs/module-14/lesson-03/async_lab.py --variant main --print

Part ``schedule`` — your ``simulate`` on two rollout/trainer balances, with response lengths drawn from a
heavy-tailed distribution: generation-bound (one engine, long responses) and trainer-bound (four times the
inference capacity). For each staleness bound k: speed-up over synchronous RL, mean and maximum lag, and how
busy each side is. The timing constants are assumptions for illustration (PROJECTED, pending the Module 14
pilot, which measures them): 0.03 s per batched decode step, 25 s per update, 256 responses per batch.

Part ``staleness`` — the course loop with a bounded-staleness sampler whose lags come from your schedule
(bound 8, a trainer-bound balance, so most batches are 6-8 updates old). Arms, 2 seeds x 100 steps from the
SFT start: GRPO on-policy; GRPO, CISPO and GSPO at bound 8 with the behaviour policy's log-probabilities as
pi_old; GRPO at bound 8 with pi_old recomputed by the current policy (staleness ignored).

Part ``mismatch`` — (a) how much one row of a matmul changes with its batch on this CPU, with the default
kernel and with your fixed-tile matmul; (b) the trainer/sampler gap of the toy policy when the sampler runs in
bfloat16, per token and per sequence (length-normalised, as GSPO's ratio sees it).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from frontierlab.labkit import load_path
from frontierlab.posttrain import arms as AR
from frontierlab.posttrain import rl
from frontierlab.posttrain.policy import sample, token_logprobs
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import encode_prompts, make_problems
from frontierlab.rlscale import asyncsim, runner
from frontierlab.rlscale.stability import run_stability

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m14/l143")


def schedule(lab):
    rng = np.random.default_rng(0)
    L = asyncsim.lognormal_lengths(rng, 100, 256, 300, 0.8, 2048)
    for title, scale in (("generation-bound: one engine", 1.0), ("balanced: 2.5x inference capacity", 0.4),
                         ("trainer-bound: 4x inference capacity", 0.25)):
        g = [scale * asyncsim.batch_gen_time(l, 0.03) for l in L]
        base = lab.simulate(g, 25.0, 0)["total_s"]
        print(f"\n{title}: mean generation {np.mean(g):.1f} s per batch (longest response decides), update 25 s")
        print(f"  {'bound k':>7s} {'speed-up':>9s} {'mean lag':>9s} {'max lag':>8s} {'engine busy':>12s} {'trainer busy':>13s}")
        for k in (0, 1, 2, 4, 8):
            r = lab.simulate(g, 25.0, k)
            print(f"  {k:7d} {base / r['total_s']:9.2f} {r['mean_lag']:9.2f} {r['max_lag']:8d} "
                  f"{sum(g) / r['total_s']:12.2f} {len(g) * 25.0 / r['total_s']:13.2f}")
    frac = (L >= 2048).mean()
    print(f"\n  responses at the 2,048-token cap: {frac:.3%}; batches containing one: "
          f"{(L >= 2048).any(1).mean():.0%} (each such batch takes the full 61 s)")


def staleness(lab):
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    rng = np.random.default_rng(1)
    lags = lab.simulate(rng.lognormal(0, 0.8, 200), 2.0, 8)["lags"]
    print(f"schedule lags (bound 8): mean {np.mean(lags):.2f}, histogram {np.bincount(lags).tolist()}")
    base = rl.RLConfig(init=str(sft), steps=100, minibatches=2, eval_every=50, eval_n=300)
    arms = {"grpo-sync": {"objective": "grpo"},
            "grpo-k8": {"objective": "grpo", "over": {"staleness": 8}, "lags": lags},
            "cispo-k8": {"objective": "cispo", "over": {"staleness": 8}, "lags": lags},
            "gspo-k8": {"objective": "gspo", "over": {"staleness": 8}, "lags": lags},
            "grpo-k8-recompute": {"objective": "grpo", "over": {"staleness": 8, "stale_correction": "recompute"},
                                  "lags": lags}}
    res = runner.run_arms(base, arms, [0, 1], ROOT / "staleness")
    print(AR.table(res, "final_sampled", "grpo-sync"))
    print(f"\n{'arm':18s} {'clip_frac':>9s} {'ratio_max':>9s} {'entropy':>8s} {'KL':>6s} {'drawdown':>9s}")
    for name in arms:
        S = [run_stability(ROOT / "staleness" / f"{name}-s{s}") for s in (0, 1)]
        print(f"{name:18s} " + " ".join(f"{np.mean([x[k] for x in S]):{w}.3f}" for k, w in
                                       (("clip_frac", 9), ("ratio_max", 9), ("entropy_min", 8), ("kl_final", 6),
                                        ("eval_drawdown", 9))))


@torch.no_grad()
def mismatch_part(lab):
    torch.manual_seed(0)
    print("\n(a) row 0 of x @ W computed alone vs in a batch: max |difference| (float32, this CPU)")
    for K in (512, 2048):
        W, x = torch.randn(K, K), torch.randn(256, K)
        full, tiled = x @ W, lab.fixed_tile_matmul(x, W)
        d = {b: float(((x[:b] @ W)[0] - full[0]).abs().max()) for b in (1, 3, 16, 64)}
        t = {b: float((lab.fixed_tile_matmul(x[:b], W)[0] - tiled[0]).abs().max()) for b in (1, 3, 16, 64)}
        print(f"  K={K:5d}  default {d}   fixed-tile {t}")
    print("\n(b) toy policy, sampler in bfloat16, trainer in float32 (same weights), 1,024 responses")
    sft = ensure_sft("runs/m12/sft")
    trainer = load_policy(sft).eval()
    for dt, name in ((torch.float32, "float32"), (torch.bfloat16, "bfloat16")):
        sampler = load_policy(sft).eval().to(dt)
        probs = make_problems(1024, seed=5)
        ro = sample(sampler, encode_prompts(probs), 8, 1.0, torch.Generator().manual_seed(0))
        tl = token_logprobs(trainer, ro.tokens, ro.prompt_len)
        st = lab.mismatch_stats(tl, ro.sampler_logp, ro.mask)
        d = (tl - ro.sampler_logp).double() * ro.mask
        seq_norm = (d.sum(-1) / ro.mask.sum(-1)).abs().max()
        print(f"  sampler {name:8s}: mean |d| {st['mean_abs']:.2e}, max |d| {st['max_abs']:.2e}, k3 {st['k3']:.2e}, "
              f"max token ratio {st['max_ratio']:.4f}, max |sum d| {st['seq_logratio_absmax']:.2e}, "
              f"max |mean d| per sequence {float(seq_norm):.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "schedule", "staleness", "mismatch"], default="all")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        if a.variant == "t4":
            print("# vLLM 0.30.0 colocated with training does not fit a T4 for this lab; run the HF rollout arms:")
            print("python -m frontierlab.rlscale.hf_rl --objective grpo --model Qwen/Qwen3-0.6B-Base "
                  "--revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8 --max-new 256 --steps 50 "
                  "--run runs/m14/l143-t4/hf-sync")
            return
        root = "runs/m14/l143-main"
        for name, flags in (("hf-sync", "--rollout hf"), ("vllm-sync", "--rollout vllm --staleness 0"),
                            ("vllm-sync-bi", "--rollout vllm --staleness 0 --batch-invariant"),
                            ("vllm-k1-grpo", "--rollout vllm --staleness 1 --objective grpo"),
                            ("vllm-k1-cispo", "--rollout vllm --staleness 1 --objective cispo")):
            print(f"python -m frontierlab.rlscale.hf_rl {flags} --steps 100 --eval-every 50 --run {root}/{name}")
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.part in ("all", "schedule"):
        schedule(lab)
    if a.part in ("all", "mismatch"):
        mismatch_part(lab)
    if a.part in ("all", "staleness"):
        staleness(lab)


if __name__ == "__main__":
    main()
