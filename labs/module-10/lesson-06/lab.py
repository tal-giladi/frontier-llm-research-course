"""Lab 10.6 (extension) — multilingual and code data. Fill in the TODOs; run `pytest labs/module-10/lesson-06`."""

from __future__ import annotations

import hashlib

import numpy as np


def tokens_per_byte(encode, texts: list[str]) -> float:
    """Total tokens of ``encode(text)`` over all texts divided by their total UTF-8 bytes."""
    raise NotImplementedError("TODO 1: tokenizer fertility per byte")


def quantile_threshold(reference_values, reference_threshold: float, target_values) -> float:
    """FineWeb2's Quantile method for a "remove values below the threshold" filter: the fraction q of the
    reference (English) documents below ``reference_threshold``, then the q-quantile of the target language's
    values (np.quantile)."""
    raise NotImplementedError("TODO 2: per-language threshold")


def rehydration_weight(rate: float, global_rate: float, best_rate: float, max_weight: float = 10.0) -> float:
    """FineWeb2's upsampling weight of one MinHash cluster size from its filter removal ``rate``: 1 if
    rate >= global_rate (or if global_rate <= best_rate); ``max_weight`` at rate == best_rate; linear in
    between: 1 + (max_weight − 1)·(global_rate − rate)/(global_rate − best_rate)."""
    raise NotImplementedError("TODO 3: rehydration")


def group_split(key, val_per_mille: int = 100, test_per_mille: int = 100, salt: str = "") -> str:
    """Data-v0's hash rule on a group key: b = int(sha1(salt + str(key)).hexdigest()[:8], 16) mod 1000;
    b < val -> "val", b < val + test -> "test", else "train"."""
    raise NotImplementedError("TODO 4: split by group")


def swesmith_keys(instance_id: str) -> tuple[str, str]:
    """(repository, bug-generation family) of a SWE-smith instance id:
    'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc' -> ('oauthlib__oauthlib', 'combine_file');
    'pydicom__pydicom.7d361b3d.func_pm_op_change__x1' -> ('pydicom__pydicom', 'func_pm_op_change')."""
    raise NotImplementedError("TODO 5: task-family keys")
