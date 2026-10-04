"""Lab 06.5: one MatFormer run, its nested submodels and Mix'n'Match, against independently trained models.

    python labs/module-06/lesson-05/elastic.py                    # free CPU, about 6 minutes of training
    python labs/module-06/lesson-05/elastic.py --variant t4 --device cuda

Arms (toy preset, 200 steps, seed 0):
  matformer   g = 4 nested FFN widths {48, 96, 192, 384}, one width sampled per step (paper section 3.2)
  b0          Baseline-0 at full width 384 (shared with lesson 06.1)
  ffn48       Baseline-0 with SwiGLU width 48, trained on its own (the "S" baseline)
Evaluates held-out loss of every MatFormer granularity and of two Mix'n'Match configurations, inference FLOPs per
token of each, and consistency: the share of positions where the small model's greedy next token equals the
large model's (MatFormer S vs its own XL; independent ffn48 vs b0). MatFormer reports its submodels more
consistent with the universal model than independently trained ones (section 4.1, Figure 2c).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m06  # noqa: E402
from frontierlab.blocks import accounting  # noqa: E402
from frontierlab.blocks import train as blocks_train  # noqa: E402
from frontierlab.data.loader import TokenData  # noqa: E402
from frontierlab.evals.heldout import window_losses  # noqa: E402
from frontierlab.labkit import load_path  # noqa: E402

lab = load_path(str(Path(__file__).parent / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))


@torch.no_grad()
def greedy(model, x):
    return model(x).logits.argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(m06.VARIANTS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    v = m06.VARIANTS[a.variant]
    vocab = m06.vocab_size()
    I = m06.arm_config("b0", a.variant, vocab).intermediate_size
    widths = [I // 8, I // 4, I // 2, I]
    small = f"ffn{widths[0]}"
    for arm in ("matformer", "b0", small):
        m06.train_arm(a.variant, arm, a.seed, question=f"lesson 06.5: elastic FFN, arm {arm}", device=a.device)
    val = TokenData("val")
    n, T = v["eval_n"], v["eval_T"]
    mat = blocks_train.load_model(m06.run_dir(a.variant, "matformer", a.seed) / "checkpoint.pt", a.device).to(a.device).eval()
    L = mat.config.num_hidden_layers
    base_cfg = m06.arm_config("b0", a.variant, vocab)
    print(f"{'model':28s} {'held-out':>9s} {'inference MFLOP/token':>22s}")
    configs = [("MatFormer g=%d (width %d)" % (i + 1, w), [w] * L) for i, w in enumerate(widths)]
    for budget in ((widths[0] + widths[1]) / 2 + 12, (widths[2] + widths[3]) / 2 - 24):
        mm = lab.mix_n_match(widths, L, budget)
        configs.append((f"Mix'n'Match {mm}", mm))
    for name, ws in configs:
        mat.set_widths(ws)
        loss = float(np.mean(window_losses(mat, val, n, T, device=a.device)))
        fl = sum(accounting.flops_per_token(base_cfg.with_(intermediate_size=w), T, training=False) for w in ws) / L
        print(f"{name:28s} {loss:9.4f} {fl / 1e6:22.3f}")
    for arm in ("b0", small):
        loss = float(np.mean(m06.eval_losses(m06.run_dir(a.variant, arm, a.seed), n=n, T=T, device=a.device)))
        fl = accounting.flops_per_token(m06.arm_config(arm, a.variant, vocab), T, training=False)
        print(f"{'independent ' + arm:28s} {loss:9.4f} {fl / 1e6:22.3f}")
    x = torch.stack([val.window(s, T) for s in val.eval_windows(64, T)]).to(a.device)
    mat.set_widths(widths[-1])
    xl = greedy(mat, x)
    mat.set_widths(widths[0])
    s_mat = greedy(mat, x)
    b0 = blocks_train.load_model(m06.run_dir(a.variant, "b0", a.seed) / "checkpoint.pt", a.device).to(a.device).eval()
    sm = blocks_train.load_model(m06.run_dir(a.variant, small, a.seed) / "checkpoint.pt", a.device).to(a.device).eval()
    print(f"\nconsistency (greedy next-token agreement on 64 windows): MatFormer S vs its XL "
          f"{lab.consistency(s_mat, xl):.3f}; independent {small} vs b0 {lab.consistency(greedy(sm, x), greedy(b0, x)):.3f}")
    print(f"parameters: one MatFormer checkpoint holds every granularity; storing b0 and {small} separately costs "
          f"{accounting.param_counts(base_cfg)['total'] + accounting.param_counts(m06.arm_config(small, a.variant, vocab))['total']:,} "
          f"parameters vs {accounting.param_counts(m06.arm_config('matformer', a.variant, vocab))['total']:,}")


if __name__ == "__main__":
    main()
