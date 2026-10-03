"""Run cards: the experiment record of every run (Module 1, lesson 01.5).

A run card is ``<run>/run_card.yaml``. It records what must be identical for two runs to be
comparable and what is needed to reproduce one: code version, configuration, seeds, data hashes,
hardware, software versions, budget and the parent run it branches from.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def git_state(cwd: str | Path | None = None) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 - git missing or not a repo
            return ""
    return {"commit": run("rev-parse", "HEAD") or "unknown", "dirty": bool(run("status", "--porcelain"))}


def hardware() -> dict:
    hw = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
          "cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        hw.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
                  cuda=torch.version.cuda)
    return hw


def write_run_card(run_dir: str | Path, *, question: str = "", parent: str | None = None, config: dict,
                   args: dict, data_meta: dict | None = None, budget: dict | None = None,
                   extra: dict | None = None) -> Path:
    import yaml
    run_dir = Path(run_dir)
    card = {"run": run_dir.name, "question": question, "parent_run": parent, "git": git_state(),
            "hardware": hardware(), "config": config,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in args.items()},
            "data": {k: data_meta.get(k) for k in ("name", "dataset", "revision", "vocab_size",
                                                   "tokenizer_sha256")} if data_meta else None,
            "data_files": {s: data_meta[s] for s in ("train", "val", "test") if data_meta and s in data_meta},
            "budget": budget or {}, **(extra or {})}
    path = run_dir / "run_card.yaml"
    path.write_text(yaml.safe_dump(json.loads(json.dumps(card, default=str)), sort_keys=False))
    return path


def read_run_card(run_dir: str | Path) -> dict:
    import yaml
    return yaml.safe_load((Path(run_dir) / "run_card.yaml").read_text())
