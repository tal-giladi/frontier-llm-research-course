"""Write the EXAMPLE run cards of lab 19.3 (already committed; rerun only to regenerate them).

    python labs/module-19/lesson-03/make_cards.py

The cards follow the real schema of ``frontierlab.optim.train`` (copied from a toy run of lesson 19.2 on the build
laptop, 2026-10-07, with the hardware block shortened). The runs themselves are fictional: these cards back the
fictional write-ups in ``flawed/`` and ``fixed/``, and every number in those write-ups is invented.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

BASE = {
    "run": None, "question": "QK-Clip vs QK-norm under Muon at lr 1e-2 (EXAMPLE card for lab 19.3)", "parent_run": None,
    "git": {"commit": "EXAMPLE-a6f293e", "dirty": False},
    "hardware": {"python": "3.12.13", "torch": "2.14.1+cpu", "cpu_threads": 16},
    "config": {"vocab_size": 8192, "hidden_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
               "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 384, "tie_word_embeddings": True,
               "qk_norm": True, "attention": "gqa", "extra": {}},
    "args": {"run": None, "preset": "toy", "steps": 300, "batch": 16, "grad_accum": 1, "seq": 128, "lr": 0.01,
             "schedule": "cosine", "warmup": 50, "weight_decay": 0.1, "clip": 1.0, "seed": 0, "data_seed": None,
             "dtype": "fp32", "eval_every": 50, "eval_windows": 256, "device": "cpu"},
    "data": {"name": "Data-v0", "dataset": "HuggingFaceFW/fineweb-edu",
             "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "vocab_size": 8192,
             "tokenizer_sha256": "c725a9cef6c589518599c5d1099ddec4306b5714790a7b1b1dc7a7564b36a94d"},
    "data_files": {"train": {"tokens": 24061465, "bin_sha256": "8ac79bf10d436c5db9f62c1596390821c519511af3ce96f19004b8834e9ff0ad"},
                   "val": {"tokens": 214882, "bin_sha256": "3ca73e6bbe3825231bf22abc6989f2f938a66edafceedb3d26b2fd1b83fc31f3"}},
    "budget": {"steps": 300, "tokens": 614400, "train_flops": 7011355852800.0, "optimizer_flops": 0},
    "params": {"total": 1836416, "non_embedding": 787840},
    "optim": {"optimizer": "muon", "qk_clip": None, "qk_norm": "on", "stability_log": True},
}


def card(run: str, seed: int, steps: int = 300, clip: bool = False) -> dict:
    c = copy.deepcopy(BASE)
    c["run"], c["args"]["run"], c["args"]["seed"] = run, f"runs/m19/l193/{run}", seed
    if clip:
        c["config"]["qk_norm"] = False
        c["optim"].update(qk_clip=15.0, qk_norm="off")
        c["params"] = {"total": 1836160, "non_embedding": 787584}        # 4 layers x 2 x 32 norm gains fewer
    c["args"]["steps"] = c["budget"]["steps"] = steps
    c["budget"]["tokens"] = steps * 16 * 128
    c["budget"]["train_flops"] = BASE["budget"]["train_flops"] * steps / 300
    return c


def write(folder: Path, cards: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for c in cards:
        (folder / f"{c['run']}.yaml").write_text("# EXAMPLE run card for lab 19.3 (fictional run)\n"
                                                 + yaml.safe_dump(c, sort_keys=False), encoding="utf-8")


def main():
    write(HERE / "flawed" / "cards", [card("clip-s0", 0, steps=400, clip=True), card("norm-s0", 0)])
    write(HERE / "fixed" / "cards", [card(f"clip-s{s}", s, clip=True) for s in range(3)]
          + [card(f"norm-s{s}", s) for s in range(3)])


if __name__ == "__main__":
    main()
