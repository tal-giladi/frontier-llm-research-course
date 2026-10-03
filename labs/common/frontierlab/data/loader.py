"""Token batches from Data-v0 files, with a resumable sampler and fixed evaluation windows.

Training batches are random windows drawn with a ``torch.Generator``; saving the generator state in
the checkpoint makes the data stream resume exactly where it stopped (parent course lesson 07.3).
Evaluation uses :meth:`TokenData.eval_windows`, a fixed list of window starts derived from a seed,
so every run in an experiment is scored on exactly the same tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from frontierlab.data.prepare import DEFAULT_OUT


class TokenData:
    def __init__(self, split: str = "train", root: str | Path = DEFAULT_OUT):
        root = Path(root)
        path = root / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run: python -m frontierlab.data.prepare")
        self.split = split
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.meta = json.loads((root / "meta.json").read_text())
        docs = root / f"{split}_docs.npy"
        self.doc_starts = np.load(docs) if docs.exists() else None

    def __len__(self) -> int:
        return len(self.tokens)

    def window(self, start: int, T: int) -> torch.Tensor:
        return torch.from_numpy(self.tokens[start:start + T].astype(np.int64))

    def batch(self, B: int, T: int, generator: torch.Generator, device="cpu") -> torch.Tensor:
        """(B, T) long tensor of random windows; the model shifts labels inside."""
        starts = torch.randint(0, len(self.tokens) - T, (B,), generator=generator).tolist()
        return torch.stack([self.window(s, T) for s in starts]).to(device)

    def eval_windows(self, n: int, T: int, seed: int = 1234) -> list[int]:
        """``n`` fixed, non-overlapping window starts (sorted), the same for every run."""
        slots = (len(self.tokens) - 1) // T
        if n > slots:
            raise ValueError(f"only {slots} windows of length {T} in {self.split}")
        g = torch.Generator().manual_seed(seed)
        return sorted((torch.randperm(slots, generator=g)[:n] * T).tolist())
