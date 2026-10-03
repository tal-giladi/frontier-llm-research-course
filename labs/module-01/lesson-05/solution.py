"""Reference solution for lab 01.5."""

from __future__ import annotations

import torch

BOOKKEEPING = ("run", "question", "parent_run", "args.run", "args.question", "args.parent", "args.log_every",
               "args.ckpt_every", "args.eval_every", "args.stop_after", "args.max_minutes", "args.device")
DATA_AND_EVAL = ("data", "data_files", "args.eval_windows", "args.seq")
SOFTWARE = ("git.commit", "hardware.torch", "hardware.python", "hardware.cuda", "hardware.gpu")


def flatten(card, prefix=""):
    out = {}
    for k, v in (card or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _covered(key, prefixes):
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def classify(key, changed=(), seeds_are_replicates=False, axis="tokens"):
    if _covered(key, BOOKKEEPING):
        return "ignore"
    if _covered(key, changed):
        return "changed"
    if key in ("args.seed", "args.data_seed"):
        return "replicate" if seeds_are_replicates else "invalidates"
    if _covered(key, DATA_AND_EVAL):
        return "invalidates"
    if key in SOFTWARE:
        return "invalidates" if axis == "wallclock" else "warn"
    if key.startswith("params."):
        return "changed" if any(c.split(".")[0] == "config" for c in changed) else "invalidates"
    if key in ("budget.tokens", "args.steps"):
        return "invalidates" if axis == "tokens" else "changed"
    if key.startswith("config.") or key.startswith("args."):
        return "invalidates"
    return "warn"


def same_bits(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.bool:
        return torch.equal(a, b)
    return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
