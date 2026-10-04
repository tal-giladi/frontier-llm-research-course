"""Lab 07.2, step 2: measure the update RMS of every matrix of the toy model under AdamW and three Muon scalings.

    python labs/module-07/lesson-02/rms_check.py           # free CPU, about a minute

Trains four copies of the toy model from the same initialisation for 40 steps on the same Data-v0 batches
(AdamW; Muon with adjust none / original / match_rms, all at learning rate 3e-3 and weight decay 0.1),
and prints, per weight shape, the median RMS of the update divided by the learning rate over steps 21-40.
Moonlight (section 2.2) predicts ~1/sqrt(max(A, B)) for "none", and ~0.2 for "match_rms" — AdamW's range.
"""

import copy
import statistics
from collections import defaultdict

import torch

from frontierlab.data.loader import TokenData
from frontierlab.model import LM, toy
from frontierlab.optim.muon import make_optimizer, update_scale


def main(steps=40, lr=3e-3):
    data = TokenData("train")
    torch.manual_seed(0)
    base = LM(toy(vocab_size=data.meta["vocab_size"]))
    arms = {"adamw": dict(optimizer="adamw"), "muon/none": dict(optimizer="muon", adjust="none"),
            "muon/original": dict(optimizer="muon", adjust="original"),
            "muon/match_rms": dict(optimizer="muon", adjust="match_rms")}
    shapes = {n: tuple(p.shape) for n, p in base.named_parameters()}
    table = {}
    for arm, kw in arms.items():
        model = copy.deepcopy(base)
        opt = make_optimizer(model, lr=lr, weight_decay=0.0, **kw)    # no decay: measure the gradient part only
        gen = torch.Generator().manual_seed(0)
        per = defaultdict(list)
        for s in range(steps):
            x = data.batch(16, 128, gen)
            opt.zero_grad(set_to_none=True)
            model(x, labels=x).loss.backward()
            opt.collect_stats = s >= steps // 2
            opt.step()
            for n, st in opt.last_stats.items() if opt.collect_stats else ():
                if len(shapes[n]) == 2 and "embed" not in n:
                    per[shapes[n]].append(st["update_rms"] / lr)
        table[arm] = {sh: statistics.median(v) for sh, v in per.items()}
    print("median RMS(update) / lr over steps 21-40, per matrix shape (A, B) = (fan_out, fan_in)")
    print(f"{'shape':>12s} {'1/sqrt(max)':>12s} " + " ".join(f"{a:>15s}" for a in arms))
    for sh in sorted(table["adamw"]):
        theory = 1 / max(sh) ** 0.5
        print(f"{str(sh):>12s} {theory:12.4f} " + " ".join(f"{table[a][sh]:15.4f}" for a in arms)
              + f"   (original scale {update_scale(sh, 'original'):.2f}, match_rms scale {update_scale(sh, 'match_rms'):.2f})")


if __name__ == "__main__":
    main()
