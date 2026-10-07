"""Reference solution for lab 16.2 — software-engineering tasks and group-level splits."""

from __future__ import annotations

import random
import re


def split_groups(keys: list[str], held_frac: float = 0.25, seed: int = 0) -> list[str]:
    groups = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[:max(1, round(held_frac * len(groups)))])
    return ["held" if k in held else "train" for k in keys]


def sibling_leak(splits: list[str], keys: list[str]) -> float:
    train = {k for k, s in zip(keys, splits) if s == "train"}
    held = [k for k, s in zip(keys, splits) if s == "held"]
    return sum(k in train for k in held) / len(held) if held else float("nan")


def patch_files(patch: str) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", patch or "", re.M))


def instance_keys(instance_id: str) -> dict:
    repo, _commit, rest = instance_id.split(".", 2)
    return {"instance": instance_id, "repository": repo, "family": rest.split("__")[0]}
