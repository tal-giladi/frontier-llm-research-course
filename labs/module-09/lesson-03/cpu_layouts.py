"""Free CPU variant of lesson 09.3: three layouts of the course model on 4 gloo processes, measured.

    python labs/module-09/lesson-03/cpu_layouts.py                       # DDP, FSDP2, FSDP2 x TP (2 x 2)
    python labs/module-09/lesson-03/cpu_layouts.py --rounds 3 --steps 10

For each layout (alternating rounds, so machine drift hits all of them): the slowest rank's step time with
gradient synchronisation, the exposed communication (sync minus no-sync, 95% interval), tokens per second, and
the bytes of parameters + gradients + optimizer state each rank holds, next to your ``state_bytes_per_rank``.

Same model, same per-rank batch, same data per DP rank in every layout; the FSDP2 x TP layout has 2 DP ranks, so
it processes half the tokens per step — compare tokens per second, not step times, across it. These are real
collectives between CPU processes; the times say nothing about NVLink or InfiniBand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontierlab.dist.measure import layout_worker
from frontierlab.labkit import load_target
from frontierlab.perf.dist import spawn

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def diff_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    boots = (np.median(a[rng.integers(0, a.size, (n_boot, a.size))], 1)
             - np.median(b[rng.integers(0, b.size, (n_boot, b.size))], 1))
    return float(np.median(a) - np.median(b)), *np.quantile(boots, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--preset", default="toy")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    modes = ["ddp", "fsdp2", "fsdp2_tp"]
    acc = {m: {"sync": [], "nosync": [], "res": None} for m in modes}
    for _ in range(a.rounds):
        for mode in modes:
            res = spawn(layout_worker, a.world, mode=mode, preset=a.preset, vocab=a.vocab, batch=a.batch, seq=a.seq,
                        steps=a.steps)
            acc[mode]["sync"] += np.max([r["sync"] for r in res], axis=0).tolist()
            acc[mode]["nosync"] += np.max([r["nosync"] for r in res], axis=0).tolist()
            acc[mode]["res"] = res
    from frontierlab.model import LM, PRESETS
    n = sum(p.numel() for p in LM(PRESETS[a.preset](vocab_size=a.vocab)).parameters())
    print(f"{a.preset} model, {n / 1e6:.2f}M parameters, {a.world} ranks, {a.batch} x {a.seq} tokens per DP rank, "
          f"{a.rounds} rounds x {a.steps} steps per layout")
    print(f"{'layout':10s} {'step ms [95% CI]':>24s} {'exposed comm ms [95% CI]':>28s} {'tokens/s':>9s} "
          f"{'held MB per rank (measured)':>30s} {'predicted MB':>13s}")
    for mode in modes:
        s = lab.summarize(acc[mode]["sync"], 0)
        e, lo, hi = diff_ci(acc[mode]["sync"], acc[mode]["nosync"])
        res = acc[mode]["res"]
        tok = res[0]["tokens_per_step"] / s["median"]
        held = [r["held"]["total"] / 1e6 for r in res]
        pred = ""
        if mode in ("ddp", "fsdp2"):
            pred = f"{lab.state_bytes_per_rank(n, a.world, mode) / 1e6:.2f}"
        print(f"{mode:10s} {s['median'] * 1e3:8.1f} [{s['lo'] * 1e3:6.1f}, {s['hi'] * 1e3:6.1f}] "
              f"{e * 1e3:10.1f} [{lo * 1e3:6.1f}, {hi * 1e3:6.1f}] {tok:9.0f} "
              f"{' / '.join(f'{h:.2f}' for h in held):>30s} {pred:>13s}")
    print("\nFSDP2's no-sync arm still all-gathers parameters, so its 'exposed' number is the reduce-scatter part only"
          " (lesson 02.3). FSDP2 x TP also runs TP collectives in both arms.")


if __name__ == "__main__":
    main()
