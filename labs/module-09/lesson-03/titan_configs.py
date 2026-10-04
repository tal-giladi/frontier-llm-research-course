"""torchtitan 0.3.0 run configurations for lesson 09.3 (main path, one 8-GPU node). NOT RUN IN THIS BUILD.

torchtitan 0.3.0 describes a run as a Python function that returns a complete ``Trainer.Config``
(torchtitan/config/README.md at tag v0.3.0); ``--module`` names an importable module and ``--config`` the
function. The ``--section.option`` command-line flags still exist at 0.3.0 but are deprecated, so the layouts are
written here as functions. Field names checked against the tag v0.3.0 sources on 2026-10-04:
``torchtitan/config/configs.py`` (ParallelismConfig, TrainingConfig, CompileConfig),
``torchtitan/components/checkpointer/base.py`` (checkpoint fields), ``torchtitan/tools/profiler.py``
(Profiler.Config), ``torchtitan/models/llama3/config_registry.py`` (``llama3_8b``).

Run from a torchtitan 0.3.0 checkout with this folder on PYTHONPATH (see run_titan.sh):

    NGPU=8 MODULE=titan_configs CONFIG=m09_fsdp8      ./run_train.sh
    NGPU=8 MODULE=titan_configs CONFIG=m09_fsdp4_tp2  ./run_train.sh

Both layouts train Llama 3.1 8B at the same global batch (16 sequences × 8,192 tokens per step), the same data,
seed, precision (BF16 parameters, FP32 reductions), selective activation checkpointing and ``torch.compile``; only
the parallel layout differs. Each run is 60 steps; metrics are logged every step and a profiler trace is written at
step 50 for rank 0..7 (``outputs/<run>/profiling/traces/iteration_50/rank{r}_trace.json.gz``).
"""

from __future__ import annotations

from torchtitan.models.llama3.config_registry import llama3_8b
from torchtitan.trainer import Trainer

STEPS = 60
SEQ = 8192
GLOBAL_SEQS = 16


def _base(name: str) -> Trainer.Config:
    c = llama3_8b()
    c.dump_folder = f"./outputs/{name}"
    c.training.steps = STEPS
    c.training.seq_len = SEQ
    c.training.mixed_precision_param = "bfloat16"
    c.training.mixed_precision_reduce = "float32"
    c.compile.enable = True
    c.metrics.log_freq = 1
    c.profiler.enable_profiling = True
    c.profiler.profile_freq = 50
    c.profiler.enable_memory_snapshot = False
    c.checkpoint.enable = False                     # lesson 09.4 measures checkpointing separately
    c.debug.seed = 0
    c.debug.deterministic = False
    return c


def m09_fsdp8() -> Trainer.Config:
    """Layout A: FSDP2 over all 8 GPUs (ZeRO-3 style), 2 sequences per GPU."""
    c = _base("m09_fsdp8")
    c.parallelism.data_parallel_shard_degree = 8
    c.parallelism.tensor_parallel_degree = 1
    c.training.local_batch_size = GLOBAL_SEQS // 8
    return c


def m09_fsdp4_tp2() -> Trainer.Config:
    """Layout B: TP over pairs of GPUs (inside the node), FSDP2 over 4 such pairs, 4 sequences per DP rank."""
    c = _base("m09_fsdp4_tp2")
    c.parallelism.data_parallel_shard_degree = 4
    c.parallelism.tensor_parallel_degree = 2
    c.parallelism.enable_sequence_parallel = True
    c.training.local_batch_size = GLOBAL_SEQS // 4
    return c


def m09_fsdp4_pp2() -> Trainer.Config:
    """Optional layout C: 2 pipeline stages (1F1B, 8 micro-batches of 1 sequence) x FSDP2 over 4."""
    c = _base("m09_fsdp4_pp2")
    c.parallelism.data_parallel_shard_degree = 4
    c.parallelism.pipeline_parallel_degree = 2
    c.parallelism.pipeline_parallel_schedule = "1F1B"
    c.parallelism.pipeline_parallel_microbatch_size = 1
    c.training.local_batch_size = GLOBAL_SEQS // 4
    return c


def m09_fsdp4_cp2_32k() -> Trainer.Config:
    """Optional layout D (lesson 09.1's ring attention at scale): 32K tokens per sequence, CP over pairs of GPUs."""
    c = _base("m09_fsdp4_cp2_32k")
    c.training.seq_len = 32768
    c.parallelism.data_parallel_shard_degree = 4
    c.parallelism.context_parallel_degree = 2
    c.training.local_batch_size = 1
    return c
