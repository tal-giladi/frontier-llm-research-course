"""Lab 06.1, step 5: the MTP modules as a draft — acceptance rate, measured; latency gain, projected.

    python labs/module-06/lesson-01/acceptance.py                  # runs/m06/cpu, seed 0
    python labs/module-06/lesson-01/acceptance.py --variant main --device cuda

For the mtp-ds and mtp-meta checkpoints:
1. teacher-forced first-draft agreement on 64 held-out windows (``frontierlab.blocks.mtp.teacher_forced_acceptance``):
   at positions where the main head's greedy token equals the real next token this *is* the greedy acceptance test;
2. real self-speculative greedy decoding (``speculative_greedy``) from 8 held-out prompts of 32 tokens, 32 new
   tokens each: the output must equal plain greedy decoding (checked), and the script counts proposed and
   accepted drafts and tokens per verification round;
3. the projected decode speed-up from the measured acceptance p (formula in the lesson; PROJECTED — Module 15
   measures latency with a serving engine).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402
from frontierlab.blocks import train as blocks_train  # noqa: E402
from frontierlab.blocks.mtp import speculative_greedy, teacher_forced_acceptance  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402


def projected_speedup(p: float, draft_cost: float, depth: int = 1) -> float:
    """Expected tokens per verification round over the cost of a round, relative to plain decoding.

    With D chained drafts each accepted with probability p (given the previous one was), a round yields
    1 + p + p² + ... + p^D tokens and costs one main-model step plus D draft steps of relative cost ``draft_cost``
    (verification of the D extra tokens assumed free: memory-bound decode, where reading the weights dominates)."""
    tokens = sum(p ** j for j in range(depth + 1))
    return tokens / (1 + depth * draft_cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompts", type=int, default=8)
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    val = TokenData("val")
    T = v["eval_T"]
    starts = val.eval_windows(64, T)
    windows = torch.stack([val.window(s, T) for s in starts]).to(a.device)
    for arm in ("mtp-ds", "mtp-meta"):
        run = m06.run_dir(a.variant, arm, a.seed)
        if not (run / "checkpoint.pt").exists():
            print(f"{run}: no checkpoint, skipping")
            continue
        model = blocks_train.load_model(run / "checkpoint.pt", map_location=a.device).to(a.device).eval()
        tf = {"positions": 0, "agree": 0.0, "exact_n": 0, "exact_acc": 0.0}
        for i in range(0, 64, 16):
            st = teacher_forced_acceptance(model, windows[i:i + 16])
            tf["positions"] += st["positions"]
            tf["agree"] += st["agreement"] * st["positions"]
            tf["exact_n"] += st["exact_positions"]
            tf["exact_acc"] += st["exact_acceptance"] * st["exact_positions"]
        p_tf = tf["exact_acc"] / max(1, tf["exact_n"])
        print(f"\n{arm}: teacher-forced first-draft agreement {tf['agree'] / tf['positions']:.3f} over {tf['positions']} "
              f"positions; where the main head was right ({tf['exact_n']} positions) acceptance {p_tf:.3f}")
        prop = acc = rounds = new = 0
        lossless = True
        for j in range(a.prompts):
            prompt = windows[j:j + 1, :32]
            ref = model.generate(prompt, 32)
            out, st = speculative_greedy(model, prompt, 32)
            lossless &= bool(torch.equal(out, ref))
            prop, acc, rounds, new = prop + st["proposed"], acc + st["accepted"], rounds + st["rounds"], new + 32
        p = acc / prop if prop else float("nan")
        print(f"{arm}: speculative greedy on {a.prompts} prompts: output identical to greedy: {lossless}; drafts "
              f"proposed {prop}, accepted {acc} (acceptance {p:.3f}); {new / rounds:.2f} tokens per verification round")
        L = model.config.num_hidden_layers
        c = 1 / L        # one extra block against L blocks (head and eh_proj ignored): a rough relative cost
        print(f"{arm}: PROJECTED decode speed-up with D = 1, p = {p:.2f}, draft cost {c:.2f} of a main step: "
              f"{projected_speedup(p, c):.2f}x  (DeepSeek-V3 reports p = 0.85-0.90 and 1.8x TPS, section 5.4.3)")


if __name__ == "__main__":
    main()
