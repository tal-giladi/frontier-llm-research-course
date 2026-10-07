"""Lab 16.2 — software-engineering tasks and group-level splits. Fill in the TODOs; run
`pytest labs/module-16/lesson-02` to check. ``swe_lab.py`` uses these functions.
"""

from __future__ import annotations

import random
import re


def split_groups(keys: list[str], held_frac: float = 0.25, seed: int = 0) -> list[str]:
    """Assign every item to "train" or "held" by its group key: sort the distinct keys, shuffle them with
    ``random.Random(seed)``, hold out the first max(1, round(held_frac * number of groups)) groups. Every item
    of a group gets its group's split."""
    raise NotImplementedError("TODO 1: split by group")


def sibling_leak(splits: list[str], keys: list[str]) -> float:
    """Fraction of held-out items whose ``keys`` value (for example the source function or repository) also
    occurs among training items. NaN if nothing is held out."""
    raise NotImplementedError("TODO 2: share of held-out items with a sibling in train")


def patch_files(patch: str) -> set[str]:
    """Paths a unified git diff modifies: the ``a/<path>`` of every ``diff --git a/<path> b/<path>`` header line."""
    raise NotImplementedError("TODO 3: files touched by a patch")


def instance_keys(instance_id: str) -> dict:
    """SWE-smith instance id -> {"instance", "repository", "family"}.

    'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc' -> repository 'oauthlib__oauthlib', family 'combine_file'
    (the id is <repository>.<commit>.<family>__<hash>)."""
    raise NotImplementedError("TODO 4: SWE-smith grouping keys")
