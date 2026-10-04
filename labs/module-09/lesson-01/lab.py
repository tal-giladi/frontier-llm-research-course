"""Lab 09.1 — parallelism layouts and ring attention. Fill in the TODOs; run `pytest labs/module-09/lesson-01`."""

from __future__ import annotations

import torch  # noqa: F401  (TODO 4)


def group_nodes(order: list[str], sizes: dict[str, int], dims: list[str], gpus_per_node: int) -> int:
    """Number of nodes touched by the group of rank 0 that varies along ``dims``.

    ``order`` lists the parallel dimensions from innermost (consecutive ranks) to outermost, e.g.
    ["tp", "cp", "pp", "dp"]; ``sizes`` gives each dimension's size. The stride of a dimension is the product of
    the sizes of the dimensions inside it. The group's ranks are all sums ``Σ i_d · stride_d`` over d in ``dims``
    with ``0 <= i_d < sizes[d]``; a rank r lives on node ``r // gpus_per_node``.
    """
    raise NotImplementedError("TODO 1: rank placement")


def state_bytes_per_device(n_params: float, tp: int, pp: int, replicas: int, zero: int,
                           w: float = 2, g: float = 4, master: float = 4, optim: float = 8) -> float:
    """Weights + gradients + master weights + optimizer state per device, for a model of ``n_params`` split evenly.

    Each device holds n_params / (tp · pp) parameters. ZeRO stage ``zero`` shards over the ``replicas`` data-parallel
    copies: stage >= 1 divides master + optimizer by ``replicas``, stage >= 2 also the gradients, stage 3 also the
    weights. Bytes per parameter: ``w`` weights, ``g`` gradients, ``master`` and ``optim`` as named.
    """
    raise NotImplementedError("TODO 2: memory per device")


def tp_cp_bytes_per_layer(s: int, b: int, h: int, kv_width: int, e: int, tp: int, cp: int) -> dict:
    """Bytes one device sends per layer and micro-batch, forward + backward (ring algorithms).

    TP with sequence parallelism: 2 all-gathers and 2 reduce-scatters forward, the same backward, each over an
    activation of s_l·b·h elements (s_l = s / cp), each moving (tp-1)/tp of it: ``8·(tp-1)/tp · s_l·b·h·e``.
    CP (ring): forward passes this rank's K and V (s_l·b·kv_width elements, kv_width = KV heads per TP rank × (d_k + d_v))
    to the next rank cp-1 times; backward passes K, V and their gradients: ``3·(cp-1) · s_l·b·kv_width·e``.
    Return {"tp": ..., "cp": ...} (0 for a dimension of size 1).
    """
    raise NotImplementedError("TODO 3: TP and CP bytes")


def merge(o_a: torch.Tensor, lse_a: torch.Tensor, o_b: torch.Tensor, lse_b: torch.Tensor):
    """Combine two partial attention results over disjoint key blocks (the online-softmax rule).

    o_* (B, H, T, d) are softmax-weighted values over each block; lse_* (B, H, T) the log-sum-exp of each block's
    scores (-inf for a query that saw no key in that block). Return (o, lse) for the union of the two blocks:
    lse = log(exp(lse_a) + exp(lse_b)); o = exp(lse_a - lse)·o_a + exp(lse_b - lse)·o_b. A row where both are -inf
    must come out as o = 0 and lse = -inf (no NaN).
    """
    raise NotImplementedError("TODO 4: online-softmax merge")
