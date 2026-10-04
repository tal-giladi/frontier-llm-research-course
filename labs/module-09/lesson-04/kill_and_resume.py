"""Kill a distributed run, restart it, and prove the restart is the same run (lab 09.4).

    python labs/module-09/lesson-04/kill_and_resume.py                       # free CPU: 2 gloo ranks, FSDP2
    python labs/module-09/lesson-04/kill_and_resume.py --mode ddp --world 4
    python labs/module-09/lesson-04/kill_and_resume.py --device cuda --world 8 --hidden 1024 --layers 8   # main path

Part 1 — exactness (float64, small model): an uninterrupted run, a run whose rank 1 dies hard (``os._exit``)
after step ``--die-at`` and is relaunched with the same command, and a run that dies *during* a checkpoint (after
the shards are written, before the commit). After each relaunch the final model, optimizer, scheduler, data-stream
position, step and every rank's RNG must be bit-identical to the uninterrupted run, and so must every logged loss.

Part 2 — cost (float32, larger model): checkpoint size, the blocking time of ``dcp.save`` vs ``dcp.async_save``,
restart cost (process start + load), and the work lost per restart (your ``lost_work``).

All runs go to ``runs/m09/recovery/``. CPU processes write to the local disk; on a cluster the checkpoint goes to a
shared file system whose bandwidth decides delta (lesson 09.4 projects it).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from frontierlab.dist import recovery as R
from frontierlab.labkit import load_target

lab = load_target(str(Path(__file__).parent / "test_lab.py"))


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--mode", default="fsdp2", choices=["fsdp2", "ddp"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--die-at", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=512, help="part 2 model width")
    ap.add_argument("--layers", type=int, default=4, help="part 2 model depth")
    ap.add_argument("--out", type=Path, default=Path("runs/m09/recovery"))
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    base = dict(steps=12, ckpt_every=4, mode=a.mode, device=a.device, global_batch=8, seq=32, hidden=64)

    print(f"1. exactness: {a.world} ranks, {a.mode}, float64, 12 steps, checkpoint every 4, rank 1 dies after step {a.die_at}")
    straight = a.out / "straight"
    r = R.launch(a.world, run_dir=str(straight), **base)
    assert r["ok"], r
    for name, extra in (("crash", {"die_at": a.die_at}), ("crash-in-save", {"die_at": 8, "die_in_save": True})):
        run = a.out / name
        first = R.launch(a.world, run_dir=str(run), **base, **extra)
        uncommitted = sorted(p.name for p in (run / "ckpt").glob("step_*") if not (p / "COMMITTED").exists())
        resume_from = lab.latest_committed(run / "ckpt")
        second = R.launch(a.world, run_dir=str(run), **base)
        cmp = R.compare_states(R.load_final(straight), R.load_final(run))
        la, lb = R.losses_by_step(straight), R.losses_by_step(run)
        same_losses = la == lb
        lost = lab.lost_work(R.read_metrics(run))
        print(f"   {name:14s} first launch: {'ok' if first['ok'] else first['error']}; uncommitted folders left: "
              f"{uncommitted or 'none'}; resumed from {resume_from.name if resume_from else None}; second launch ok: {second['ok']}")
        print(f"   {'':14s} bit-identical to the uninterrupted run: " +
              ", ".join(f"{k} {'yes' if v['bitwise_equal'] else 'NO'}" for k, v in cmp.items()) +
              f"; all 12 losses identical: {'yes' if same_losses else 'NO'}; steps recomputed: {lost}")
        assert all(v["bitwise_equal"] for v in cmp.values()) and same_losses, "recovery is not exact"

    print(f"\n2. cost: float32, hidden {a.hidden}, {a.layers} layers, {a.world} ranks, {a.mode}")
    cost = dict(base, dtype="float32", hidden=a.hidden, layers=a.layers, seq=128, global_batch=8, steps=12,
                ckpt_every=3, vocab=8192, final_state=False)
    rows = {}
    for name, extra in (("sync", {}), ("async", {"async_save": True}), ("crash", {"die_at": 8})):
        run = a.out / f"cost-{name}"
        R.launch(a.world, run_dir=str(run), **cost, **extra)
        if name == "crash":
            R.launch(a.world, run_dir=str(run), **cost)
        recs = R.read_metrics(run)
        steps = [x for x in recs if x["event"] == "step"]
        ck = [x["ckpt_s"] for x in steps if x.get("ckpt_s") is not None]
        st = [x["step_s"] for x in steps]
        starts = [x for x in recs if x["event"] == "start"]
        rows[name] = dict(ck=ck, step=st, starts=starts, lost=lab.lost_work(recs),
                          size=dir_bytes(R.latest_committed(run / "ckpt")) if R.latest_committed(run / "ckpt") else 0)
    size = rows["sync"]["size"]
    print(f"   checkpoint size on disk: {size / 1e6:.1f} MB (model + AdamW state, fp32)")
    for name in ("sync", "async"):
        ck = np.array(rows[name]["ck"])
        print(f"   {name:5s} blocking time per checkpoint: median {np.median(ck) * 1e3:.0f} ms (n = {ck.size}); "
              f"median step {np.median(rows[name]['step']) * 1e3:.0f} ms; write bandwidth {size / np.median(ck) / 1e9:.2f} GB/s"
              + (" (the blocking part only; the write continues in the background)" if name == "async" else ""))
    s = rows["crash"]["starts"]
    print(f"   restart: process start {s[-1]['startup_s'] - s[-1]['load_s']:.1f} s + checkpoint load {s[-1]['load_s']:.2f} s; "
          f"steps recomputed {rows['crash']['lost']} (crash after step 8, last commit at step 6)")
    print("\n   Use the sync blocking time as delta and process start + load as a lower bound for R in goodput_calc.py.")


if __name__ == "__main__":
    main()
