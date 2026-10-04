"""Module 7 — which optimizer and parametrization?

* :mod:`~frontierlab.optim.muon` — Newton-Schulz orthogonalisation (quintic, cubic, DeepSeek-V4 hybrid),
  Muon, and :class:`MuonAdamW`, one optimizer with Muon and AdamW parameter groups (lesson 07.1).
* :mod:`~frontierlab.optim.cost` — Newton-Schulz FLOPs, overhead per step, equal-wall-clock step counts and
  a timing helper built on ``frontierlab.perf`` (07.1, project).
* :mod:`~frontierlab.optim.qkclip` — per-head maximum attention logits during training and QK-Clip for the
  GQA family and MLA (07.2).
* :mod:`~frontierlab.optim.mup` — µP relative to a base width: init, readout multiplier, per-group learning
  rates, width ladder, coordinate check (07.3).
* :mod:`~frontierlab.optim.schedules` — cosine, constant and WSD with explicit decay branches (07.4).
* :mod:`~frontierlab.optim.stability` — the stability-statistics logger, spike detection and diagnosis (07.5).
* :mod:`~frontierlab.optim.stabilizers` — z-loss, final-logit soft-cap, and the ``"gqa-softcap"`` attention
  kind with attention soft-capping and QKV clamping (07.5). Importing this package registers that kind.
* :mod:`~frontierlab.optim.train` — the training wrapper around ``frontierlab.train.loop``.
"""

from frontierlab.optim import stabilizers  # noqa: F401  (registers "gqa-softcap")
from frontierlab.optim.muon import (CUBIC, NS_SCHEDULES, QUINTIC, V4_FINAL, MuonAdamW, make_optimizer, newton_schulz,
                                    ns_polynomial, orthogonal_polar, param_groups, split_params, update_scale)
from frontierlab.optim.qkclip import LogitMonitor, QKClip, head_max_logits
from frontierlab.optim.stability import StabilityLogger, detect_spikes, diagnose

__all__ = ["CUBIC", "LogitMonitor", "MuonAdamW", "NS_SCHEDULES", "QKClip", "QUINTIC", "StabilityLogger", "V4_FINAL",
           "detect_spikes", "diagnose", "head_max_logits", "make_optimizer", "newton_schulz", "ns_polynomial",
           "orthogonal_polar", "param_groups", "split_params", "update_scale"]
