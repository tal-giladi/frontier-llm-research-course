"""Lab 09.5 (extension) — torus topology and sharding notation. Fill in the TODOs; run `pytest labs/module-09/lesson-05`."""

from __future__ import annotations


def torus_hops(a: tuple[int, ...], b: tuple[int, ...], dims: tuple[int, ...], wrap: bool = True) -> int:
    """Fewest link hops between chips ``a`` and ``b`` (coordinates) in a mesh of shape ``dims``.

    Along each axis of size n the distance is |a_i - b_i|, or, with wraparound links (a torus),
    min(|a_i - b_i|, n - |a_i - b_i|). Hops add over axes (dimension-ordered routing).
    """
    raise NotImplementedError("TODO 1: hops in a torus")


def bisection_links(dims: tuple[int, ...], wrap: bool = True) -> int:
    """Links cut when a mesh of shape ``dims`` is split into two equal halves across its largest axis.

    Cutting an axis of size n (even) crosses one link per row of the other axes, i.e. prod(dims)/n links — and twice
    that with wraparound, because each ring along that axis is cut in two places.
    """
    raise NotImplementedError("TODO 2: bisection links")


def collective_time(kind: str, nbytes: float, axis_size: int, link_bw: float) -> float:
    """Time of a collective over one ring axis of ``axis_size`` chips, bandwidth-bound (the Scaling Book's model).

    AllGather and ReduceScatter: each chip sends (n-1)/n of the full array over its links in both directions of the
    ring, so the time is ``nbytes · (n-1)/n / (2 · link_bw)`` with ``link_bw`` the one-way bandwidth of one link.
    AllReduce is a ReduceScatter followed by an AllGather: twice that. ``kind`` is "allgather", "reducescatter" or
    "allreduce"; ``nbytes`` is the size of the full (unsharded) array.
    """
    raise NotImplementedError("TODO 3: collective time on a ring axis")


def matmul_comm(lhs: tuple[str, str], rhs: tuple[str, str]) -> str:
    """Which collective a sharded matmul C[I, K] = A[I, J] · B[J, K] needs, in the Scaling Book's notation.

    ``lhs`` and ``rhs`` give the mesh axis sharding each dimension ("" = replicated): A[I_X, J] is ("X", ""),
    B[J_X, K] is ("X", ""). Return:
      "none"        — the contracting dimension J is sharded in neither operand;
      "allgather"   — J is sharded in exactly one operand (gather it first);
      "allreduce"   — J is sharded the same way in both (multiply locally, then sum the partial results);
      "invalid"     — both operands use the same mesh axis on a non-contracting dimension (A[I_X] · B[K_X]).
    """
    raise NotImplementedError("TODO 4: which collective")
