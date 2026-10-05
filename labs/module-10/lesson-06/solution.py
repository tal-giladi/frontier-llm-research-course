"""Reference solution for lab 10.6 — multilingual and code data (extension)."""

from __future__ import annotations

import hashlib

import numpy as np


def tokens_per_byte(encode, texts):
    return sum(len(encode(t)) for t in texts) / max(1, sum(len(t.encode("utf-8")) for t in texts))


def quantile_threshold(reference_values, reference_threshold, target_values):
    q = float((np.asarray(reference_values, dtype=np.float64) < reference_threshold).mean())
    return float(np.quantile(np.asarray(target_values, dtype=np.float64), q))


def rehydration_weight(rate, global_rate, best_rate, max_weight=10.0):
    if rate >= global_rate or global_rate <= best_rate:
        return 1.0
    return 1.0 + (max_weight - 1.0) * (global_rate - rate) / (global_rate - best_rate)


def group_split(key, val_per_mille=100, test_per_mille=100, salt=""):
    b = int(hashlib.sha1((salt + str(key)).encode("utf-8")).hexdigest()[:8], 16) % 1000
    return "val" if b < val_per_mille else "test" if b < val_per_mille + test_per_mille else "train"


def swesmith_keys(instance_id):
    """'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc' -> ('oauthlib__oauthlib', 'combine_file')."""
    repo, _commit, rest = instance_id.split(".", 2)
    return repo, rest.split("__")[0]
