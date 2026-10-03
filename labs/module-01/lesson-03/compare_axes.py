"""Same two models, two comparison axes: does the winner change? (lab 01.3, step 4)

    python labs/module-01/lesson-03/compare_axes.py                       # ~25 min on a 16-thread laptop
    python labs/module-01/lesson-03/compare_axes.py --steps 150           # ~11 min, noisier
    python labs/module-01/lesson-03/compare_axes.py --device cuda --batch 64 --seq 512 --steps 2000   # GPU

Arms (both are Baseline-0's architecture, only width and depth differ):
  small = the ``toy`` preset (C=128, 4 layers)          wide = ``toy-wide`` (C=256, 6 layers)

1. Calibrate: 20 untimed steps then 30 timed steps of each model, measuring tokens/s on THIS machine.
2. Equal tokens: both train ``--steps`` steps (same batch, sequence, schedule shape).
3. Equal wall-clock: the wide run from step 2 sets the time budget; the small model gets
   ``steps × (tok/s small ÷ tok/s wide)`` steps, i.e. the same seconds.
4. Every run is scored on the same 256 fixed validation windows; the script prints the mean loss and
   the measured wall-clock of each run, and writes ``runs/l13-axes/summary.json``.

Equal training FLOPs is printed too (from frontierlab.flops), so you can see how close it is to
equal wall-clock on your hardware. One seed per arm: this step shows the *axis* effect, not a
significance test (lesson 01.4 adds seeds).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.evals.heldout import window_losses
from frontierlab.flops import flops_per_token
from frontierlab.model.config import PRESETS, ModelConfig
from frontierlab.train import loop


def toy_wide(vocab_size: int = 8192) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, hidden_size=256, num_hidden_layers=6, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=64, intermediate_size=768, max_position_embeddings=512)


PRESETS.setdefault("toy-wide", toy_wide)       # registered for this process only; the loop is unchanged


def train(run: Path, preset: str, steps: int, a) -> tuple[torch.nn.Module, float]:
    args = ["--run", str(run), "--preset", preset, "--steps", str(steps), "--batch", str(a.batch),
            "--seq", str(a.seq), "--lr", str(a.lr), "--warmup", str(max(1, steps // 10)), "--seed", str(a.seed),
            "--eval-every", str(10**9), "--log-every", "25", "--ckpt-every", str(10**9), "--device", a.device,
            "--question", "lesson 01.3: does the winner depend on the comparison axis?"]
    if a.device.startswith("cuda"):
        args += ["--dtype", "bf16"]
    t0 = time.perf_counter()
    model = loop.main(args)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def throughput(preset: str, a, vocab: int) -> float:
    """Tokens/s of training steps only (no eval, no checkpoint), after a warm-up."""
    from frontierlab.model import LM
    torch.manual_seed(0)
    model = LM(PRESETS[preset](vocab_size=vocab)).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data, g = TokenData("train"), torch.Generator().manual_seed(0)
    for i in range(50):
        if i == 20:
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        x = data.batch(a.batch, a.seq, g, a.device)
        loss = model(x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    return 30 * a.batch * a.seq / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("runs/l13-axes"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    val = TokenData("val")
    vocab = val.meta["vocab_size"]
    tps = {p: throughput(p, a, vocab) for p in ("toy", "toy-wide")}
    fpt = {p: flops_per_token(PRESETS[p](vocab_size=vocab), a.seq) for p in tps}
    ratio = tps["toy"] / tps["toy-wide"]
    plan = {"small @ equal tokens": ("toy", a.steps), "wide (sets both budgets)": ("toy-wide", a.steps),
            "small @ equal wall-clock": ("toy", round(a.steps * ratio))}
    print(f"tokens/s: small {tps['toy']:,.0f}, wide {tps['toy-wide']:,.0f} (ratio {ratio:.2f}); "
          f"FLOPs/token ratio {fpt['toy-wide'] / fpt['toy']:.2f} -> equal-FLOPs steps for small: "
          f"{round(a.steps * fpt['toy-wide'] / fpt['toy'])}")
    rows = {}
    for label, (preset, steps) in plan.items():
        run = a.out / f"{preset}-{steps}"
        if (run / "checkpoint.pt").exists():
            raise SystemExit(f"{run} already exists (the loop would resume it); delete {a.out} to rerun")
        model, secs = train(run, preset, steps, a)
        losses = window_losses(model, val, a.windows, a.seq, device=a.device)
        rows[label] = {"preset": preset, "steps": steps, "tokens": steps * a.batch * a.seq,
                       "wall_s": round(secs, 1), "val_loss": sum(losses) / len(losses), "run": str(run)}
        print(f"{label:26s} steps {steps:5d}  tokens {rows[label]['tokens']:>9,d}  wall {secs:7.1f}s"
              f"  val loss {rows[label]['val_loss']:.4f}")
        (run / "window_losses.json").write_text(json.dumps(losses))
    s, w, sw = (rows["small @ equal tokens"], rows["wide (sets both budgets)"], rows["small @ equal wall-clock"])
    print(f"\nequal tokens:     {'wide' if w['val_loss'] < s['val_loss'] else 'small'} wins "
          f"({s['val_loss'] - w['val_loss']:+.4f} small - wide)")
    print(f"equal wall-clock: {'wide' if w['val_loss'] < sw['val_loss'] else 'small'} wins "
          f"({sw['val_loss'] - w['val_loss']:+.4f} small - wide)")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps({"tokens_per_s": tps, "flops_per_token": fpt, "runs": rows},
                                                   indent=2))


if __name__ == "__main__":
    main()
