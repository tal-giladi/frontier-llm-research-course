"""Is a CPU training run bitwise reproducible? Three runs of the same command (lab 01.5, step 3).

    python labs/module-01/lesson-05/rerun.py                  # ~1 min on a 16-thread laptop
    python labs/module-01/lesson-05/rerun.py --steps 40       # quicker

Runs ``frontierlab.train.loop`` three times with identical arguments:

  A  with T CPU threads (T = what torch uses by default on this machine)
  B  with T threads again                       -> expected: identical to A, bit for bit
  C  with T // 2 threads                        -> same code, same seed, same data order

then compares the final checkpoints tensor by tensor (``frontierlab.record.state_diff``, bitwise) and
the logged losses, and runs the run-card diff on A vs C. Each run gets its own process-wide thread
setting; nothing else changes.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from frontierlab.metrics import read_jsonl
from frontierlab.record import diff_cards, state_diff
from frontierlab.train import loop


def run(path: Path, threads: int, a) -> dict:
    if path.exists():
        shutil.rmtree(path)
    torch.set_num_threads(threads)
    loop.main(["--run", str(path), "--preset", "toy", "--steps", str(a.steps), "--batch", "8", "--seq", "128",
               "--warmup", "10", "--seed", "0", "--log-every", "10", "--eval-every", str(a.steps),
               "--eval-windows", "16", "--ckpt-every", str(10**9), "--device", "cpu",
               "--question", "lesson 01.5: bitwise rerun"])
    ck = torch.load(path / "checkpoint.pt", weights_only=False)
    losses = [r["loss"] for r in read_jsonl(path / "metrics.jsonl") if r["split"] == "train"]
    return {"model": ck["model"], "optimizer": ck["optimizer"]["state"], "losses": losses}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("runs/l15-rerun"))
    a = ap.parse_args()
    T = torch.get_num_threads()
    res = {name: run(a.out / name, th, a) for name, th in (("A", T), ("B", T), ("C", max(1, T // 2)))}
    torch.set_num_threads(T)
    report = {}
    for other in ("B", "C"):
        model_diff = state_diff(res["A"]["model"], res[other]["model"])
        opt_diff = state_diff(res["A"]["optimizer"], res[other]["optimizer"])
        la, lo = res["A"]["losses"], res[other]["losses"]
        first = next((i for i, (x, y) in enumerate(zip(la, lo)) if x != y), None)
        max_loss = max(abs(x - y) for x, y in zip(la, lo))
        report[other] = {"tensors_differing": len(model_diff), "optimizer_entries_differing": len(opt_diff),
                         "max_loss_diff": max_loss, "first_differing_log_row": first}
        threads = T if other == "B" else max(1, T // 2)
        print(f"A ({T} threads) vs {other} ({threads} threads): {len(model_diff)} of {len(res['A']['model'])} "
              f"weight tensors differ, {len(opt_diff)} optimizer entries differ, largest logged loss "
              f"difference {max_loss:.3e}" + ("" if first is None else f" (first at log row {first})"))
    print("\nrun-card diff A vs C (seeds are the same, so any INVALIDATES line is a real difference):")
    for f in diff_cards(a.out / "A", a.out / "C"):
        if f.severity != "ignore":
            print("  ", f)
    print("   (cpu_threads is recorded but classified as bookkeeping; is that the right rule for this result?)")
    (a.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
