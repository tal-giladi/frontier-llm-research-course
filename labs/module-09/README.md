# Module 9 labs — What does the cluster cost, and how does it fail?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. Shared code is in
`labs/common/frontierlab/dist/` (layout planner, ring attention, pipeline schedule simulator, a real pipeline over
processes, the capability probes, layout measurements, the recovery harness with DCP, goodput); its tests are
`labs/common/tests/test_dist.py` (multi-process, `gloo`, 1–4 ranks; about 4–5 minutes on the build laptop).

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/module-09/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-09             # all five reference solutions
pytest labs/common/tests/test_dist.py                 # the shared Module 9 code
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Outputs go to
`runs/m09/` (ignored by git).

> [!IMPORTANT]
> The CPU variants run **real collectives** (`gloo`) between processes on one machine: the same semantics, ordering
> and correctness as NCCL, and the right way to measure. Their *times* are those of processes sharing one CPU's cores,
> memory and loopback sockets; they say nothing about NVLink, InfiniBand or GPU kernels. **No multi-GPU command in this
> module was run in this build** — the course's compute is one single-GPU Colab account, so multi-GPU is not piloted
> (plan section 12.1). Every GPU figure in the lessons is PROJECTED from formulas or PUBLISHED, and labelled.

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 09.1 Parallelism layouts at scale | `plan_layouts.py`, `ring_cp.py` | memory and communication per GPU for Llama 3 405B's Table 4 layouts (and the reversed rank order), DeepSeek-V3's layout (BF16 and FP8 activations), four Baseline-0 layouts on one node; ring attention vs single-device attention in float64 (output and gradients, 2 and 4 ranks), then contiguous vs load-balanced sharding timed in alternating rounds |
| `lesson-02/` | 09.2 Pipeline schedules and overlap | `simulate.py`, `real_pipeline.py` | GPipe, 1F1B, interleaved 1F1B, ZB-1P, DualPipe, DualPipeV: simulated idle time per device vs the published formulas, peak activations, ASCII Gantt charts; a real 4-stage pipeline of the course model over processes (gradients checked against one process), GPipe vs 1F1B bubbles and memory, compared with the simulator fed with measured F and B |
| `lesson-03/` | 09.3 A measured multi-GPU investigation | `titan_configs.py`, `run_titan.sh`, `titan_report.py`, `cpu_layouts.py`, `matrix.py`, `sample_titan.log` | torchtitan 0.3.0 configuration functions for FSDP 8 vs FSDP 4 × TP 2 (and optional PP and CP layouts) on Llama 3.1 8B; the main-path runner; the report (tokens/s, MFU, memory per rank, exposed communication from traces); the free CPU variant (DDP, FSDP2, FSDP2 × TP on 4 processes); the tested-capability matrix; a **synthetic** log in torchtitan's format for the parser test |
| `lesson-04/` | 09.4 Failure and recovery | `kill_and_resume.py`, `reshard.py`, `goodput_calc.py` | hard crash and crash-during-save with bitwise comparison against an uninterrupted run; checkpoint size, sync vs async blocking time, restart cost, lost work; FSDP2 on 2 ranks restored into FSDP2 on 4 and DDP on 1, with stateful vs counter-based RNG; Llama 3's interruption record turned into checkpoint intervals and goodput, with a Monte Carlo check |
| `lesson-05/` | 09.5 TPUs, JAX and hardware co-design (extension) | `scaling_book.py` | torus hops and bisection for TPU v4/v5p slices, all-gather times from the Scaling Book's table, the four sharded-matmul cases, checkpoint read times at published storage rates |
| `project/` | Module project | `plan_1t.py`, `buggy_plan.py` | the infrastructure plan for M9-1T (1.03T total, 40.8B active) on 2,048 H100s from labelled inputs; the debugging task |

## Hardware and time per variant

Free CPU times were measured on 2026-10-04 on a 16-thread Windows 11 laptop (torch 2.14.1+cpu, Python 3.12) with other
jobs running; expect variation. GPU commands are **not run in this build**; they are part of the Module 9 pilot.

| Lab | Main path (rented GPUs) | Free GPU | Free CPU (measured) |
|---|---|---|---|
| 09.1 | 1 node, 4–8 GPUs, ~10 GPU-min: `ring_cp.py --device cuda --world 8 --T 32768 --heads 32 --dim 128` | Kaggle 2× T4: `ring_cp.py --device cuda --world 2 --T 16384` (untested) | `ring_cp.py --rounds 3` 3 min 19 s; `plan_layouts.py` seconds |
| 09.2 | 4–8 GPUs, ~15 GPU-min: `torch.distributed.pipelining` schedules on Baseline-0 (see lesson) | Kaggle 2× T4, 2 stages (untested) | `real_pipeline.py` 3 min 26 s; `simulate.py` < 1 s |
| 09.3 | 1 node, 8× H100 SXM, 4–6 GPU-hours (PROJECTED), USD 10–20: `bash run_titan.sh`, then `titan_report.py WORK --skip 10` | Kaggle 2× T4: `layout_worker(..., device="cuda")` on 2 GPUs and `python -m frontierlab.dist.capability --world 2` (untested; torchtitan 8B does not fit) | `cpu_layouts.py` 5 min 24 s; `matrix.py` 53 s |
| 09.4 | 8 GPUs, ~30 GPU-min: `kill_and_resume.py --device cuda --world 8 --hidden 2048 --layers 16`, `reshard.py --device cuda`, and a torchtitan kill/restart with checkpointing on | Colab 1 GPU or Kaggle 2× T4: `kill_and_resume.py --device cuda --world 1` / `--world 2` (untested) | `kill_and_resume.py` about 6 min; `reshard.py` about 7 min; `goodput_calc.py` < 1 s |
| 09.5 | none needed (optional Cloud TPU + JAX, not run) | none needed | `scaling_book.py` < 1 s |
| project | uses the 09.3/09.4 main-path outputs | — | `plan_1t.py`, `buggy_plan.py` < 1 s each |

Notes:

- `frontierlab.perf.dist.spawn` starts the processes with the `spawn` method (Windows, macOS, Linux); `device="cuda"`
  selects NCCL and one GPU per rank.
- On the CPU, `torch.compile` needs a C++ compiler (MSVC `cl.exe` on Windows, gcc on Linux); the capability matrix
  records it as untestable when there is none.
- torchtitan 0.3.0 configures runs with Python functions (`MODULE=... CONFIG=... ./run_train.sh`); the
  `--section.option` flags are deprecated at that version. `run_titan.sh` dry-runs each configuration with
  `COMM_MODE="fake_backend"` before the real runs.
- Asynchronous DCP saves use their own `gloo` process group in `frontierlab.dist.recovery`: sharing the training
  group between the background writer and the training loop crashed the processes on the build machine.
