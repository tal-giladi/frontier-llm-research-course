"""Lab 07.2, step 4: read the arms' stability logs and held-out losses, and apply the contract's rules.

    python labs/module-07/lesson-02/compare_arms.py                       # runs/m07/l72/cpu
    python labs/module-07/lesson-02/compare_arms.py runs/m07/l72/main --device cuda

For every arm: the maximum attention logit at steps 50, 150 and the end, its maximum over the run, how many
head-updates QK-Clip clipped, the median update/weight RMS ratio of the Muon matrices, the final training
loss, and the held-out loss on 256 fixed validation windows of 128 tokens, paired against ``muon-noqk``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m07  # noqa: E402


def at(rows, key, step):
    pts = [r for r in rows if r["step"] <= step and r.get(key) is not None]
    return pts[-1][key] if pts else float("nan")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path("runs/m07/l72/cpu")
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    arms = [p for p in sorted(root.iterdir()) if (p / "checkpoint.pt").exists()]
    ref = root / "muon-noqk"
    ref_losses = m07.eval_losses(ref, device=device) if (ref / "checkpoint.pt").exists() else None
    print(f"{'arm':14s} {'logit@50':>9s} {'@150':>8s} {'end':>8s} {'max':>8s} {'clipped':>8s} {'ratio':>7s} "
          f"{'train':>7s} {'held-out':>9s}  vs muon-noqk (paired, 256 windows)")
    out = {}
    for run in arms:
        st = m07.stability(run)
        end = st[-1]["step"] if st else 0
        mx = max((r["max_logit"] for r in st if r.get("max_logit") is not None), default=float("nan"))
        clipped = sum(r.get("clipped_heads") or 0 for r in st)
        ratio = at(st, "ratio_muon_median", end) if any("ratio_muon_median" in r for r in st) else at(st, "ratio_adamw_median", end)
        tr = m07.train_rows(run)
        final_train = m07.mean([r["loss"] for r in tr[-20:]])
        losses = m07.eval_losses(run, device=device)
        cmp = m07.compare(losses, ref_losses) if ref_losses is not None and run != ref else ""
        out[run.name] = {"max": mx, "end": at(st, "max_logit", end), "heldout": m07.mean(losses)}
        print(f"{run.name:14s} {at(st, 'max_logit', 50):9.2f} {at(st, 'max_logit', 150):8.2f} {at(st, 'max_logit', end):8.2f} "
              f"{mx:8.2f} {clipped:8d} {ratio:7.4f} {final_train:7.3f} {m07.mean(losses):9.4f}  {cmp}")
    print("\nRules (stated in the lesson's contract):")
    if "muon-noqk" in out and "muon-qknorm" in out:
        r = out["muon-noqk"]["max"] / out["muon-qknorm"]["max"]
        print(f"  H1 logit growth: max logit without QK-norm / with QK-norm = {r:.1f}  (observed if >= 3)")
    for arm, tau in (("muon-clip", 100.0), ("muon-clip-lo", 15.0), ("mla-clip-lo", 15.0)):
        if arm in out:
            binding = out.get("muon-noqk" if "mla" not in arm else "mla-muon", {}).get("max", float("nan")) > tau
            print(f"  H2 {arm}: unclipped arm passed tau={tau:g}: {binding}; clipped arm's max over the run "
                  f"{out[arm]['max']:.2f} (pre-clip, per step); end {out[arm]['end']:.2f}")


if __name__ == "__main__":
    main()
