"""Restore a checkpoint into a different layout with DCP resharding (lab 09.4).

    python labs/module-09/lesson-04/reshard.py                 # free CPU: FSDP2 on 2 ranks -> FSDP2 on 4, DDP on 1
    python labs/module-09/lesson-04/reshard.py --device cuda   # main path: same on GPUs (8 -> 4 GPUs, etc.)

1. Train 6 steps with FSDP2 on 2 ranks and save a checkpoint at step 3 and 6 (DCP: each rank writes its shards).
2. Load the step-3 checkpoint into (a) FSDP2 on 2 ranks, (b) FSDP2 on 4 ranks, (c) DDP on 1 rank; compare the full
   gathered model, optimizer, scheduler and data position after loading: they must be bit-identical.
3. Continue each layout from step 3 to step 6 and compare losses with the original run, once with stateful
   per-rank RNG and once with counter-based RNG. The expected result: with counter RNG, losses agree to rounding
   (the gradient reduction sums in a different order); with stateful RNG the noise streams cannot be mapped onto a
   different number of ranks, the runs diverge, and the trainer says so in its start record.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from frontierlab.dist import recovery as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/m09/reshard"))
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    base = dict(steps=6, ckpt_every=3, global_batch=8, seq=32, hidden=64, device=a.device)
    for rng in ("stateful", "counter"):
        print(f"RNG mode: {rng}")
        src = a.out / rng / "fsdp2-2"
        assert R.launch(2, run_dir=str(src), rng_mode=rng, **base)["ok"]
        ckpt = src / "ckpt" / "step_000003"
        loaded, cont = {}, {}
        for world, mode in ((2, "fsdp2"), (4, "fsdp2"), (1, "ddp")):
            tag = f"{mode}-{world}"
            run = a.out / rng / f"load-{tag}"
            assert R.launch(world, run_dir=str(run), resume_from=str(ckpt), mode=mode, rng_mode=rng,
                            **{**base, "steps": 3})["ok"]
            loaded[tag] = R.load_final(run)
            run2 = a.out / rng / f"cont-{tag}"
            assert R.launch(world, run_dir=str(run2), resume_from=str(ckpt), mode=mode, rng_mode=rng, **base)["ok"]
            notes = [x["notes"] for x in R.read_metrics(run2) if x["event"] == "start"][0]
            cont[tag] = (R.losses_by_step(run2), notes)
        ref_loss = R.losses_by_step(src)
        for tag in loaded:
            cmp = R.compare_states(loaded["fsdp2-2"], loaded[tag])
            eq = ", ".join(f"{k} {'yes' if cmp[k]['bitwise_equal'] else 'NO'}" for k in ("model", "optim", "scheduler", "data", "step"))
            dl = max(abs(cont[tag][0][s] - ref_loss[s]) for s in (4, 5, 6))
            print(f"   {tag:8s} loaded state bit-identical to the 2-rank load: {eq}")
            print(f"   {'':8s} steps 4-6 vs the original run: max |loss diff| {dl:.2e}"
                  + (f"  [{cont[tag][1][0][:90]}...]" if cont[tag][1] else ""))


if __name__ == "__main__":
    main()
