"""Run-card diff: which differences between two runs invalidate a comparison? (lesson 01.5)

A run card (``frontierlab.runcard``) is nested YAML. :func:`flatten` turns it into dotted keys
(``args.lr``, ``data_files.train.bin_sha256``); :func:`classify` decides what each difference means,
given the variable the experiment contract says it changes and the comparison axis it uses.

The rules, in order:

1. Bookkeeping never matters: the run name, the question, the parent, logging and checkpoint cadence,
   ``--stop-after`` / ``--max-minutes`` (exact resume makes them invisible), the data path, the device
   string, the CPU thread count and the OS string.
2. The declared changed variable (``changed``) is expected.
3. Seeds are ``replicate`` when ``seeds_are_replicates`` (comparing seed sets), else they invalidate.
4. Data, tokenizer and evaluation settings always invalidate: different data or different evaluation
   windows means the numbers are not on the same footing.
5. Parameter counts follow from the config: expected if a ``config.*`` change is declared, otherwise a
   sign of an undeclared change. On the equal-parameters axis they must match within ``params_tol``.
6. Budget fields are checked against the axis: equal tokens needs equal steps and tokens; equal FLOPs
   needs ``budget.train_flops`` within ``flops_tol``; equal wall-clock needs the same GPU and software and
   the measured hours within ``time_tol``. A budget field the axis does not hold equal is expected.
7. Software versions and the git commit are ``warn`` (record them; they invalidate a wall-clock
   comparison). A dirty working tree is ``warn`` even when both runs are dirty.
8. Every other ``config.*`` or ``args.*`` difference invalidates: it is a second changed variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

IGNORE_PREFIXES = ("run", "question", "parent_run", "notes", "args.run", "args.question", "args.parent",
                   "args.log_every", "args.ckpt_every", "args.eval_every", "args.stop_after", "args.max_minutes",
                   "args.data", "args.device", "args.peak", "hardware.platform", "hardware.cpu_threads",
                   "measured.cost_usd", "measured.final_val_loss")
SEED_KEYS = ("args.seed", "args.data_seed")
INVALIDATING_PREFIXES = ("data", "data_files", "args.eval_windows", "args.seq")
SOFTWARE_KEYS = ("hardware.torch", "hardware.python", "hardware.cuda", "git.commit")
AXES = ("tokens", "flops", "wallclock", "params")
ORDER = {"invalidates": 0, "warn": 1, "changed": 2, "replicate": 3, "ignore": 4}


@dataclass
class Finding:
    key: str
    a: object
    b: object
    severity: str          # invalidates | warn | changed | replicate | ignore
    reason: str

    def __str__(self) -> str:
        return f"{self.severity.upper():11s} {self.key}: {self.a!r} -> {self.b!r}  ({self.reason})"


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> {"a.b.c": value}. Empty dicts are kept as values."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _under(key: str, prefixes) -> bool:
    """True if ``key`` is one of ``prefixes`` or lies below one of them (``data`` covers ``data.name``)."""
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def _rel(a, b) -> float:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return float("inf")
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def classify(key: str, a, b, *, changed=(), axis: str = "tokens", seeds_are_replicates: bool = False,
             flops_tol: float = 0.01, time_tol: float = 0.05, params_tol: float = 0.01) -> tuple[str, str]:
    """Severity and reason for one differing key (rules in the module docstring)."""
    if _under(key, IGNORE_PREFIXES):
        return "ignore", "bookkeeping"
    if _under(key, changed):
        return "changed", "declared changed variable"
    if key in SEED_KEYS:
        if seeds_are_replicates:
            return "replicate", "seed is the replicate axis"
        return "invalidates", "different seed in a comparison that should share seeds"
    if _under(key, INVALIDATING_PREFIXES):
        return "invalidates", "data or evaluation differs"
    if key.startswith("params."):
        if axis == "params" and key == "params.total" and _rel(a, b) > params_tol:
            return "invalidates", f"equal-parameters axis, totals differ by {_rel(a, b):.1%}"
        if any(c.split(".")[0] == "config" for c in changed):
            return "changed", "follows from the changed config"
        return "invalidates", "parameter count differs but no config change is declared"
    if key.startswith("budget.") or key == "args.steps":
        if axis == "tokens" and key in ("budget.tokens", "budget.steps", "args.steps"):
            return "invalidates", "equal-tokens axis but the token budget differs"
        if axis == "flops" and key == "budget.train_flops" and _rel(a, b) > flops_tol:
            return "invalidates", f"equal-FLOPs axis, training FLOPs differ by {_rel(a, b):.1%}"
        if axis == "params" and key in ("budget.tokens", "budget.steps", "args.steps"):
            return "invalidates", "equal parameters does not set the run length; tokens must match"
        return "changed", f"budget differs as the {axis} axis allows"
    if key in ("measured.wall_clock_h", "measured.gpu_hours"):
        if axis == "wallclock" and _rel(a, b) > time_tol:
            return "invalidates", f"equal-wall-clock axis, time differs by {_rel(a, b):.1%}"
        return "ignore", "measured outcome"
    if key.startswith("hardware.gpu"):
        if axis == "wallclock":
            return "invalidates", "wall-clock comparison on different hardware"
        return "warn", "different hardware (fine unless time is compared)"
    if key in SOFTWARE_KEYS:
        if axis == "wallclock":
            return "invalidates", "wall-clock comparison with different software"
        return "warn", "different software or code version; record it"
    if key == "git.dirty":
        return "warn", "uncommitted changes: the code that ran is not recoverable from the commit"
    if key.startswith("config.") or key.startswith("args."):
        return "invalidates", "a second changed variable"
    if key.startswith("measured."):
        return "ignore", "measured outcome"
    return "warn", "unclassified difference; check it by hand"


def _load(x) -> dict:
    if isinstance(x, dict):
        return x
    p = Path(x)
    if p.is_dir():
        p = p / "run_card.yaml"
    return yaml.safe_load(p.read_text())


def diff_cards(a, b, *, changed=(), axis: str = "tokens", seeds_are_replicates: bool = False,
               **tol) -> list[Finding]:
    """All differences between two run cards (dicts, run folders or YAML paths), most severe first."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    fa, fb = flatten(_load(a)), flatten(_load(b))
    out = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, "<missing>"), fb.get(key, "<missing>")
        if va == vb:
            if key == "git.dirty" and va is True:
                out.append(Finding(key, va, vb, "warn", "both runs from uncommitted code"))
            continue
        sev, why = classify(key, va, vb, changed=tuple(changed), axis=axis,
                            seeds_are_replicates=seeds_are_replicates, **tol)
        out.append(Finding(key, va, vb, sev, why))
    return sorted(out, key=lambda f: ORDER[f.severity])


def comparable(findings: list[Finding]) -> bool:
    return not any(f.severity == "invalidates" for f in findings)


def _same_bits(x, y) -> bool:
    """Bit-for-bit equality. ``torch.equal`` is value equality: it calls -0.0 and 0.0 equal and NaN unequal."""
    import torch
    x, y = x.detach().cpu().contiguous(), y.detach().cpu().contiguous()
    if x.dtype == torch.bool:
        return torch.equal(x, y)
    return torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8))


def state_diff(sd_a: dict, sd_b: dict) -> list[str]:
    """Names of entries that differ in any bit (or exist in only one) between two state dicts."""
    import torch
    names = []
    for k in sorted(set(sd_a) | set(sd_b), key=str):
        if k not in sd_a or k not in sd_b:
            names.append(str(k))
            continue
        x, y = sd_a[k], sd_b[k]
        if torch.is_tensor(x):
            if not torch.is_tensor(y) or x.shape != y.shape or x.dtype != y.dtype or not _same_bits(x, y):
                names.append(str(k))
        elif isinstance(x, dict) and isinstance(y, dict):
            names += [f"{k}.{n}" for n in state_diff(x, y)]
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and len(x) == len(y):
            names += [f"{k}.{n}" for n in state_diff(dict(enumerate(x)), dict(enumerate(y)))]
        elif x != y:
            names.append(str(k))
    return names
