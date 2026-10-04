"""Distributed training at scale (Module 9): what the cluster costs and how it fails.

* :mod:`~frontierlab.dist.layout` — parallelism layout planner: memory and communication per device for
  TP / CP / PP / DP / EP layouts, rank placement and which groups leave the node (lesson 09.1).
* :mod:`~frontierlab.dist.ring_attention` — ring attention (context parallelism) over ``torch.distributed``
  with an exact backward pass; contiguous and load-balanced sequence sharding (lesson 09.1).
* :mod:`~frontierlab.dist.schedule` — pipeline schedule simulator: GPipe, 1F1B, interleaved 1F1B, a
  zero-bubble variant, DualPipe and DualPipeV; bubble and activation memory per device (lesson 09.2).
* :mod:`~frontierlab.dist.pipeline` — a small real pipeline over processes (gloo or NCCL) running GPipe or
  1F1B on the course model, with measured busy time per stage (lesson 09.2).
* :mod:`~frontierlab.dist.capability` — the tested-capability matrix: which features compose for the course
  model on this machine, probed, not assumed (lesson 09.3).
* :mod:`~frontierlab.dist.recovery` — fault injection and recovery: a data-parallel trainer whose full state
  (model, optimizer, scheduler, RNG, data position) is saved with ``torch.distributed.checkpoint`` and can be
  restored into another layout (lesson 09.4).
* :mod:`~frontierlab.dist.goodput` — failure rates, checkpoint-interval math (Young, Daly) and goodput (09.4).

CPU runs use real ``gloo`` collectives between processes on one machine. They show the semantics, the
order of operations and how to measure; their times say nothing about NVLink or InfiniBand.
"""
