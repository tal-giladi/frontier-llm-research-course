"""The free-CPU models of lessons 15.2 and 15.4, trained once on Data-v0 and reused.

* **target** — the ``toy`` preset (4 layers, 128 wide, vocabulary 8192) with one DeepSeek-V3-style MTP module
  (``--mtp deepseek --mtp-depth 1``, lambda 0.3), trained with ``python -m frontierlab.blocks.train`` (Module 6).
  Its main path is an ordinary Baseline-0-layout LM; its MTP module is the "MTP as a draft" arm.
* **draft** — an independent 1-layer, 64-wide LM with the same tokenizer, trained the same way without MTP.
* **eagle** — an EAGLE-style head trained on the frozen target (:mod:`frontierlab.ttc.eagle`).

Each ``ensure_*`` trains only if its checkpoint is missing (exact resume through the course loop, so an
interrupted run continues when the command is repeated).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

ROOT = Path("runs/m15/models")
TARGET_ARGS = ["--mtp", "deepseek", "--mtp-depth", "1", "--preset", "toy", "--batch", "16", "--seq", "128",
               "--lr", "3e-3", "--eval-every", "100000", "--log-every", "100"]
DRAFT_ARGS = ["--preset", "toy", "--cfg", "hidden_size=64", "num_hidden_layers=1", "num_attention_heads=2",
              "num_key_value_heads=1", "head_dim=32", "intermediate_size=192", "--batch", "16", "--seq", "128",
              "--lr", "3e-3", "--eval-every", "100000", "--log-every", "100"]


def _train(run: Path, args: list[str], steps: int):
    from frontierlab.blocks import train as BT
    ck = run / "checkpoint.pt"
    done = ck.exists() and torch.load(ck, map_location="cpu", weights_only=False)["step"] >= steps
    if not done:
        BT.main(["--run", str(run), "--steps", str(steps), *args])
    return BT.load_model(ck).eval()


def ensure_target(steps: int = 800, root: Path = ROOT):
    return _train(root / "target", TARGET_ARGS, steps)


def ensure_draft(steps: int = 800, root: Path = ROOT):
    return _train(root / "draft", DRAFT_ARGS, steps)


def ensure_eagle(target, steps: int = 400, root: Path = ROOT):
    from frontierlab.data.loader import TokenData
    from frontierlab.ttc.eagle import new_head, train_head
    path = root / "eagle.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        head = new_head(target.config)
        head.load_state_dict(ck["model"])
        return head.eval(), ck["history"]
    t0 = time.perf_counter()
    head, hist = train_head(target, TokenData("train"), steps=steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": head.state_dict(), "history": hist, "seconds": round(time.perf_counter() - t0, 1)}, path)
    return head, hist


def val_prompts(n: int = 16, length: int = 32, seed: int = 7) -> list[torch.Tensor]:
    """``n`` fixed prompts of ``length`` tokens from Data-v0's validation split (fixed windows)."""
    from frontierlab.data.loader import TokenData
    val = TokenData("val")
    return [val.window(s, length)[None] for s in val.eval_windows(n, 256, seed)]
