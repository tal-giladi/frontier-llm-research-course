# Module 9 — proposed changes to shared files (for the main session)

Module 9 adds a new subpackage, `labs/common/frontierlab/dist/` (`layout.py`, `ring_attention.py`, `schedule.py`,
`pipeline.py`, `capability.py`, `measure.py`, `recovery.py`, `goodput.py`), its tests
`labs/common/tests/test_dist.py`, `labs/module-09/`, `lessons/module-09/`, `assessments/module-09-quiz.*` and
`projects/module-09-infrastructure-plan.md`. It edits no existing shared file.

## 1. BLOCKING: `.gitignore` ignores the new subpackage

Line 7 of `.gitignore` is `dist/`, which matches **any** directory named `dist`, including
`labs/common/frontierlab/dist/` (`git check-ignore -v labs/common/frontierlab/dist/layout.py` →
`.gitignore:7:dist/`). Without a change the subpackage will not be committed and every Module 9 lab and test fails
on a fresh clone. Either change the line to anchor it at the root:

```gitignore
/dist/
```

or keep it and add, right after it:

```gitignore
!labs/common/frontierlab/dist/
```

(The first is cleaner: the intent of `dist/` is the packaging output folder at the repository root.) After the change,
`git status` must list `labs/common/frontierlab/dist/` as untracked.

## 2. `references/versions.md`: torchtitan API note

Add under "Notes on reference implementations":

```markdown
- Module 9: torchtitan 0.3.0 checked 2026-10-04 at tag v0.3.0: runs are Python functions returning `Trainer.Config`
  (`MODULE=<module> CONFIG=<function> ./run_train.sh`, i.e. `torchtitan.train --module --config`); `--section.option`
  CLI overrides still work but are deprecated; `COMM_MODE="fake_backend"` dry-runs a config on one GPU. Field names used
  by the course (`parallelism.data_parallel_shard_degree`, `tensor_parallel_degree`, `pipeline_parallel_degree`,
  `pipeline_parallel_schedule`, `context_parallel_degree`, `training.local_batch_size`, `training.seq_len`,
  `checkpoint.enable/interval`, `profiler.enable_profiling/profile_freq`, `metrics.log_freq`) are in
  `torchtitan/config/configs.py`, `components/checkpointer/base.py`, `tools/profiler.py`. `tps` in the metrics line is per
  device. PyTorch 2.14.1 ships `ScheduleDualPipeV` and `ScheduleZBVZeroBubble` in `torch.distributed.pipelining`.
```

## 3. Optional: a DCP checkpoint mode for `train/loop.py`

The course loop saves one `torch.save` file and draws batches from a single-process generator, which is right for
one GPU. Module 9's recovery harness (`frontierlab/dist/recovery.py`) shows the multi-process version. If a later
module trains the course loop with FSDP2, the loop would need: `--ckpt-format dcp` (save `get_state_dict(model, opt)`
plus the scheduler step and generators with `dcp.save` into `ckpt/step_NNNNNN/`, commit marker written by rank 0 after a
barrier), a global data stream (every rank draws the global batch from one generator and keeps its slice, as
`recovery.GlobalStream` does), and loading the newest committed folder on start. Not needed by any current lab.

## 4. Optional: `frontierlab/perf/dist.py` `spawn()`

`spawn()` raises `torch.multiprocessing.ProcessExitedException` / `ProcessRaisedException` when a rank dies, which is what
lesson 09.4 relies on (`recovery.launch` catches it). Two small additions would help later modules: a `timeout` argument
for `init_process_group` (a hung collective currently waits for the default 30 minutes), and writing a rank's
traceback into the result folder so the parent can print it. Neither is needed now.

## 5. Sidebar lines (running numbers to be fixed by the main session)

```markdown
- **Module 9 — What does the cluster cost, and how does it fail?**
  - [NN · Parallelism layouts at scale](lessons/module-09/lesson-01.md)
  - [NN · Pipeline schedules and overlap](lessons/module-09/lesson-02.md)
  - [NN · A measured multi-GPU investigation](lessons/module-09/lesson-03.md)
  - [NN · Failure and recovery](lessons/module-09/lesson-04.md)
  - [NN · TPUs, JAX and hardware co-design](lessons/module-09/lesson-05.md)
  - [Module 9 quiz](assessments/module-09-quiz.md)
```
