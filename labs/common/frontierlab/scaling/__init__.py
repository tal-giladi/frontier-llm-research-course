"""Module 11: scaling laws, downstream prediction and de-risking a run.

* :mod:`frontierlab.scaling.laws` — Chinchilla, data-constrained and inference-aware formulas with published constants;
* :mod:`frontierlab.scaling.fit` — iso-FLOP minima, the parametric fit, power laws with an offset, bootstrap, hold-out;
* :mod:`frontierlab.scaling.ladder` — ladder sizes (registered as presets), run plans, reading results;
* :mod:`frontierlab.scaling.train` — the training wrapper (any course wrapper, plus a unique-data cap);
* :mod:`frontierlab.scaling.downstream` — a cloze multiple-choice task, metrics, two-step and observational fits;
* :mod:`frontierlab.scaling.derisk` — pre-registration, prediction bands for a whole loss curve, go/no-go checks.
"""

from frontierlab.scaling import ladder  # noqa: F401  (registers the ladder presets)
