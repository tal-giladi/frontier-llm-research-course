"""Lab 12.3: the details that change RL results, one controlled comparison each.

    python labs/module-12/lesson-03/details_lab.py                    # free CPU, about 20 minutes
    python labs/module-12/lesson-03/details_lab.py --part staleness   # one part only
    python labs/module-12/lesson-03/details_lab.py --variant main --print

Every run uses *your* response mask and *your* policy loss (built from your ``aggregate``,
``token_ratio``/``sequence_ratio`` and ``clipped_surrogate``), passed to the course loop as hooks. 2 seeds per
arm, 120 steps, GRPO advantages, everything not named in an arm fixed.

* ``aggregation`` — GRPO's per-sequence mean, DAPO's token mean, Dr. GRPO's constant normaliser, ScaleRL's
  prompt mean. Watch ``len_wrong`` and ``len_correct`` (do wrong answers get longer?).
* ``truncation`` — a budget of 3 new tokens: a 3-digit sum plus EOS needs 4, so every response to a
  problem with a sum >= 100 is truncated. Truncated responses kept with reward 0 (``none``) or removed
  from the loss (``filter``); or DAPO's design (``soft``): an "expected maximum" of 3 tokens plus a
  1-token punishment cache, so the budget is 4 and DAPO's Eq. 13 gives length 4 a penalty of -1.
* ``masking`` — the correct mask; the mask with PAD positions after EOS included; the mask without EOS.
* ``staleness`` — on-policy; rollouts 4 steps stale with the behaviour policy's log-probabilities as
  ``old_logp`` (corrected); 4 steps stale but treated as on-policy (``recompute``); a bfloat16 sampler
  without and with truncated importance sampling (cap 2).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from frontierlab.labkit import load_path
from frontierlab.posttrain import arms as AR
from frontierlab.posttrain import rl
from frontierlab.posttrain.policy import sample
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import encode_prompts, problems_from, response_text, split_problems

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m12/l123")

PARTS = {
    "aggregation": {"grpo-seqmean": dict(aggregation="seq_mean_token_mean"), "dapo-token": dict(aggregation="token_mean"),
                    "drgrpo-const": dict(aggregation="seq_mean_token_sum_norm"),
                    "scalerl-prompt": dict(aggregation="prompt_mean")},
    "truncation": {"trunc-none": dict(max_new=3), "trunc-filter": dict(max_new=3, overlong="filter"),
                   "trunc-soft": dict(max_new=4, overlong="soft", l_cache=1)},
    "masking": {"mask-correct": dict(), "mask-with-pad": dict(), "mask-no-eos": dict()},
    "staleness": {"onpolicy": dict(), "stale4-behaviour": dict(staleness=4),
                  "stale4-recompute": dict(staleness=4, stale_correction="recompute"),
                  "bf16-sampler": dict(sampler_dtype="bf16"), "bf16-sampler-tis": dict(sampler_dtype="bf16", tis_cap=2.0)},
}
BASELINE = {"aggregation": "grpo-seqmean", "truncation": "trunc-none", "masking": "mask-correct",
            "staleness": "onpolicy"}


def make_hooks(lab, arm: str):
    def policy_loss(logp, old_logp, adv, mask, *, aggregation, group_size, eps_low, eps_high, ratio, norm_len,
                    is_weight=None):
        rho = lab.sequence_ratio(logp, old_logp, mask) if ratio == "sequence" else lab.token_ratio(logp, old_logp)
        obj = lab.clipped_surrogate(rho, adv, eps_low, eps_high)
        if is_weight is not None:
            obj = obj * is_weight
        loss = -lab.aggregate(obj, mask, aggregation, group_size, norm_len)
        with torch.no_grad():
            m = mask.float()
            a2 = adv[:, None].expand_as(rho) if adv.dim() == 1 else adv
            clipped = (torch.clamp(rho, 1 - eps_low, 1 + eps_high) * a2 < rho * a2).float()
            diag = {"clip_frac": float((clipped * m).sum() / m.sum().clamp_min(1)),
                    "ratio_mean": float((rho * m).sum() / m.sum().clamp_min(1)),
                    "ratio_max": float((rho * m).max()) if m.sum() > 0 else 1.0}
        return loss, diag

    def loss_mask(mask, ro):
        keep = (mask.sum(-1, keepdim=True) > 0).float()          # respects overlong filtering
        m, _ = lab.response_mask(ro.response)
        m = m.float()
        if arm == "mask-with-pad":
            m = torch.ones_like(m)
        elif arm == "mask-no-eos":
            m = m * (ro.response != lab.EOS).float()
        return m * keep

    return {"policy_loss": policy_loss, "loss_mask": loss_mask}


@torch.no_grad()
def truncation_breakdown(policy_path, max_new: int = 3) -> dict:
    """Held-out addition at the arm's budget: accuracy on small and large sums, and how often a large sum
    gets a finished answer that is too short to be right (2 digits or fewer)."""
    _, held = split_problems(2)
    probs = problems_from(held, ".", 2, "+")[:300]
    ro = sample(load_policy(policy_path), encode_prompts(probs), max_new, 1.0, torch.Generator().manual_seed(7))
    big = small = big_short = small_ok = big_ok = 0
    for row, p in zip(ro.response, probs):
        text, fin = response_text(row)
        ok = int(fin and text == p.target)
        if p.result >= 100:
            big += 1
            big_ok += ok
            big_short += int(fin and len(text) <= 2)   # finished, but too short to be right
        else:
            small += 1
            small_ok += ok
    return {"acc_sum_lt_100": small_ok / max(1, small), "acc_sum_ge_100": big_ok / max(1, big),
            "short_wrong_sum_ge_100": big_short / max(1, big)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", *PARTS], default="all")
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant == "main":
        cmd = "python -m frontierlab.posttrain.hf --run runs/m12/l123-main/{arm}-s{s} --seed {s} --steps 100 {flags}"
        main_arms = {"grpo-seqmean": "--aggregation seq_mean_token_mean", "dapo-token": "--aggregation token_mean",
                     "drgrpo-const": "--aggregation seq_mean_token_sum_norm --scale none",
                     "trunc-none": "--max-new 256", "trunc-filter": "--max-new 256 --overlong filter",
                     "trunc-soft": "--max-new 256 --overlong soft --l-cache 64"}
        for arm, flags in main_arms.items():
            for s in (0, 1):
                print(cmd.format(arm=arm, s=s, flags=flags))
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sft = ensure_sft("runs/m12/sft")
    base = rl.RLConfig(init=str(sft), steps=120, eval_every=40, eval_n=300)
    original = rl.train
    out = {}
    for part, arms in PARTS.items():
        if a.part not in ("all", part):
            continue
        res = {}
        for arm, over in arms.items():
            AR.train = lambda cfg, arm=arm: original(cfg, make_hooks(lab, arm))   # noqa: E731
            res.update(AR.run_arms(base, {arm: over}, [0, 1], ROOT / part))
        out[part] = res
        b = BASELINE[part]
        print(f"\n=== {part}")
        for metric in ("final_sampled", "len", "len_wrong", "trunc", "entropy", "clip_frac", "ratio_max",
                       "sampler_gap"):
            print(AR.table(res, metric, b))
        if part == "truncation":
            for arm in arms:
                for s in (0, 1):
                    bd = truncation_breakdown(ROOT / part / f"{arm}-s{s}" / "policy.pt", arms[arm]["max_new"])
                    res[arm][s]["breakdown"] = bd
                    print(f"  {arm:14s} s{s}: " + "  ".join(f"{k} {v:.3f}" for k, v in bd.items()))
            for budget in (3, 4):
                sb = truncation_breakdown(sft, budget)
                print(f"  SFT start, budget {budget}: " + "  ".join(f"{k} {v:.3f}" for k, v in sb.items()))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"results-{a.part}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
