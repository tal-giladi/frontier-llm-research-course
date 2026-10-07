"""Lab 17.3 — transcoders and attribution graphs. Fill in the TODOs; run `pytest labs/module-17/lesson-03`.

``graph_lab.py`` checks your functions against the course's (``frontierlab.interp.graphs``), trains a transcoder
per MLP of the Module 17 model, builds and prunes attribution graphs, and tests the graphs' predictions with
interventions in the original model against random-feature controls.
"""

from __future__ import annotations

import numpy as np
import torch


def normalise_rows(A: np.ndarray) -> np.ndarray:
    """|A| with every row divided by its sum (rows that sum to 0 stay 0). A[u, s] is the edge from s into u."""
    raise NotImplementedError("TODO 1: normalised absolute adjacency")


def total_influence(A_hat: np.ndarray) -> np.ndarray:
    """Sum of all paths of length >= 1: Â + Â² + Â³ + ... = (I − Â)⁻¹ − I (the series converges because the
    graph is acyclic and rows sum to at most 1)."""
    raise NotImplementedError("TODO 2: total influence matrix")


def prune_nodes(influence_on_logits: np.ndarray, is_logit: np.ndarray, threshold: float = 0.8) -> list[int]:
    """Indices to keep: every logit node, plus non-logit nodes in decreasing order of influence until the kept
    ones hold at least ``threshold`` of the total influence of all nodes (the node that crosses the threshold
    is not added: stop as soon as the running share has reached it). Return them sorted."""
    raise NotImplementedError("TODO 3: prune by cumulative influence")


def feature_write(W_dec: torch.Tensor, out_scale: float, feature: int, activation: float) -> torch.Tensor:
    """What one transcoder feature adds to its MLP output: activation · W_dec[feature] · out_scale, shape (d,).
    Removing a feature in the real model subtracts exactly this from the MLP output at its position."""
    raise NotImplementedError("TODO 4: one feature's write")
