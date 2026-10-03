"""Local metrics: one JSON object per line. No account, no server; plot with ``frontierlab.metrics.plot``."""

from __future__ import annotations

import json
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, **row):
        row.setdefault("time", round(time.time(), 3))
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
