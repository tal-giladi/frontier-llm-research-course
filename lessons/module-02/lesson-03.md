---
id: "02.3"
module: 2
minutes: 35
practice_minutes: 70
prerequisites: ["02.2"]
objectives:
  - Compute the bytes each rank moves for DDP's all-reduce and FSDP2's all-gathers and reduce-scatter, and convert a measured collective time into bus bandwidth.
  - Explain how DDP's gradient buckets and FSDP2's prefetching overlap communication with the backward pass, and what bucket size changes.
  - Measure exposed communication as the difference between synchronised and unsynchronised steps, with an interval, on real collectives.
  - Read a profiler timeline of a data-parallel step and identify which communication was hidden and which was exposed.
volatility: implementation
sources:
  - title: "Li et al. — PyTorch Distributed: Experiences on Accelerating Data Parallel Training (VLDB 2020; bucketing, overlap, skipping synchronisation)"
    url: https://arxiv.org/abs/2006.15704
  - title: "Zhao et al. — PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"
    url: https://arxiv.org/abs/2304.11277
  - title: "PyTorch 2.14 — torch.distributed.fsdp.fully_shard (FSDP2)"
    url: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
  - title: "PyTorch 2.14 — DistributedDataParallel (bucket_cap_mb, no_sync)"
    url: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
  - title: "NVIDIA nccl-tests — Performance reported by NCCL tests (algorithm vs bus bandwidth)"
    url: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
  - title: "NVIDIA H100 specifications (H100 SXM NVLink 900 GB/s, PCIe Gen5 128 GB/s)"
    url: https://www.nvidia.com/en-us/data-center/h100/
last_verified: "2026-10-03"
---

# 02.3 · Communication and overlap

With more than one GPU, a training step also has to move gradients or parameters between devices. This lesson measures how much of that communication the computation hides and how much it adds to the step — the exposed communication — for DDP and FSDP2, and shows how bucket sizes and prefetching change the answer.

## Why this matters at a frontier lab

Almost every run in this course from Module 7 on is data-parallel, and Module 9 builds whole parallel layouts on top of it. A comparison between two designs on 8 GPUs is only fair if both pay the same communication, and a throughput claim is only honest if it says how much of the step is communication. The question is never "how fast is the network?" in isolation; it is "how much longer is my step because of the network?". That number — exposed communication — is what you report, what you optimise, and what changes when the model, the batch per GPU or the interconnect changes.

## The idea

### DDP: replicate, then all-reduce gradients

`DistributedDataParallel` keeps a full copy of the model on every rank. Each rank computes gradients on its own micro-batch; then every gradient is **all-reduced** (summed across ranks and divided by their number) so all replicas take the same optimizer step. Three mechanisms (Li et al., 2020) decide how much of the all-reduce is hidden:

- **Buckets.** DDP packs gradients into buckets of `bucket_cap_mb` (default 25 MiB, with a 1 MiB first bucket when the default is used; torch 2.14.1 source), in *reverse* parameter order, because that is the order in which backward produces them.
- **Overlap.** As soon as every gradient in a bucket is ready, DDP launches that bucket's all-reduce asynchronously while backward keeps computing earlier layers' gradients. Only the communication still running when backward ends is exposed.
- **Skipping synchronisation.** Inside `with model.no_sync():` no all-reduce is launched; gradients accumulate locally. With gradient accumulation you synchronise only on the last micro-batch.

Bucket size is a trade: tiny buckets start early but pay a fixed per-collective latency many times; one giant bucket cannot start until the whole backward is done, so nothing overlaps.

### FSDP2: shard, gather when needed, reduce-scatter

`fully_shard(module)` (FSDP2, `torch.distributed.fsdp`) splits each parameter across ranks along dimension 0 (stored as a `DTensor`). Applied to each transformer block and then to the root model, it makes each block a communication unit:

- **forward:** all-gather the block's parameters, compute, free the gathered copy (`reshard_after_forward=True`, the default for non-root modules);
- **backward:** all-gather again, compute gradients, then **reduce-scatter** them so each rank keeps only its shard of the summed gradient;
- the optimizer updates the local shards.

FSDP2 prefetches the next block's all-gather while the current block computes, which is how it overlaps. `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)` all-gathers BF16 parameters (half the bytes) and reduces gradients in FP32. `set_requires_gradient_sync(False)` skips the reduce-scatter, the FSDP2 counterpart of `no_sync()` — but the all-gathers still run, because no rank holds the full parameters.

### Bytes per rank, and bus bandwidth

For a buffer of $S$ bytes on $n$ ranks with the ring algorithm, each rank sends (and receives):

$$\text{all-reduce: } 2\,\tfrac{n-1}{n}\,S \qquad \text{reduce-scatter: } \tfrac{n-1}{n}\,S \qquad \text{all-gather: } \tfrac{n-1}{n}\,S$$

**Algorithm bandwidth** is $S / t$. **Bus bandwidth** multiplies it by the factor above ($2(n-1)/n$ for all-reduce), which makes numbers comparable across rank counts and with the link's peak (nccl-tests, PERFORMANCE.md). Use bus bandwidth when you report a collective.

### Exposed communication

$$t_{\text{exposed}} = t_{\text{step}}(\text{sync}) - t_{\text{step}}(\text{no sync})$$

Both arms do identical computation; only the synchronisation differs. If all communication were hidden the difference would be zero. The **overlap fraction** is $1 - t_{\text{exposed}} / t_{\text{comm}}$, where $t_{\text{comm}}$ is how long the same communication takes on its own (bytes per rank ÷ measured bus bandwidth). Both medians are noisy, so the difference gets a bootstrap interval (lesson 01.4's method, applied to times).

### What the profiler shows

On GPUs, NCCL collectives run as kernels (names start with `nccl`) on their own CUDA stream. In a Perfetto timeline of a DDP step you see the backward's GEMM kernels on the compute stream and the bucket all-reduces on the communication stream underneath them; the exposed part is the stretch at the end where the communication stream is busy and the compute stream is empty (the optimizer step waits for it). For FSDP2 you see an all-gather before each block's forward and backward and a reduce-scatter after each block's backward. On the CPU variant there are no streams: gloo runs the collectives on the CPU, and the profiler shows them as operators and record ranges named after the collective, interleaved with the backward's operators.

## Worked example

### Bus bandwidth, tiny numbers

Four ranks all-reduce an 8 MB buffer in 2 ms. Algorithm bandwidth $= 8 / 0.002 = 4$ GB/s. Each rank moved $2 \cdot \tfrac{3}{4} \cdot 8 = 12$ MB, so bus bandwidth $= 12 / 0.002 = 6$ GB/s.

### Baseline-0 on one 8× H100 SXM node

Baseline-0 has 121.9M parameters. DDP all-reduces FP32 gradients: $S = 121.9\text{M} \cdot 4 = 488$ MB, so each rank moves $2 \cdot \tfrac{7}{8} \cdot 488 = 853$ MB per synchronised step. NVIDIA lists 900 GB/s of NVLink bandwidth per H100 SXM (company claim; both directions combined). Achieved all-reduce bus bandwidth is lower and must be measured; with an *assumed* 300 GB/s (INFERENCE, to be replaced by your part-1 measurement):

$$t_{\text{comm}} = 853\ \text{MB} / 300\ \text{GB/s} = 2.8\ \text{ms}$$

Computation per rank for a 32 × 1024 micro-batch: $32{,}768 \cdot 788.1\text{M} = 2.58 \times 10^{13}$ FLOPs, which at 40% MFU on 989 TFLOP/s is 65 ms. Communication is about 4% of the computation and has the whole backward (about two thirds of the 65 ms) to hide behind, so the PROJECTED exposed communication is close to zero. With gradient accumulation over 8 micro-batches and `no_sync()` on the first 7, it is one all-reduce per 8 × 65 ms.

FSDP2 with BF16 parameters and FP32 gradient reduction on the same node: two all-gathers of $244$ MB (forward and backward) and one reduce-scatter of 488 MB, so $\tfrac{7}{8}(2 \cdot 244 + 488) = 854$ MB per rank per micro-batch — the same bytes as DDP here, but paid on *every* micro-batch, because the parameters must be gathered for each forward. Exposed communication becomes visible when per-GPU compute shrinks (smaller micro-batch, shorter sequences) or the link slows (PCIe at 128 GB/s, or between nodes). All figures PROJECTED with the formulas above; multi-GPU is not piloted on the course's single-GPU Colab account.

## Shapes and cost

| Object (Baseline-0, n ranks) | Shape | dtype | Where | Bytes per rank |
|---|---|---|---|---|
| DDP replica | full model | fp32 | every GPU | 488 MB weights + 488 MB grads + 976 MB AdamW state |
| DDP bucket | flat buffer ≤ 25 MiB | fp32 | GPU, comm stream | all-reduce: $2\tfrac{n-1}{n}$ × bucket |
| FSDP2 shard of a (2816, 768) weight | (2816/n, 768) as a `DTensor` | fp32 | each GPU | 1/n of the weight |
| FSDP2 gathered block weights | full block | bf16 with the mixed-precision policy | GPU, freed after use | all-gather: $\tfrac{n-1}{n}$ × block |
| FSDP2 gradient shard | (2816/n, 768) | fp32 | each GPU | reduce-scatter: $\tfrac{n-1}{n}$ × block grads |

FSDP2's memory win is the point: weights, gradients and AdamW state per rank shrink by $n$ (about 1.95 GB → 0.24 GB at $n = 8$ for Baseline-0), at the cost of gathering parameters every micro-batch.

## Build it

`frontierlab/perf/dist.py` has the pieces:

```python
from frontierlab.perf import dist

print(dist.bus_bandwidth(8e6, 0.002, 4))                      # 6e9 bytes/s
print(dist.ddp_buckets([10 * 2**20] * 6, 25 * 2**20, 2**20))  # bucket sizes DDP would form
res = dist.spawn(dist.train_step_worker, 2, mode="ddp", bucket_cap_mb=1.0)   # 2 gloo processes
```

`spawn` starts the processes with the `spawn` method (so it works on Windows and macOS), initialises a process group through a file store — `gloo` on CPU, `nccl` with `device="cuda"` — and returns each rank's results. `train_step_worker` wraps the toy model in DDP (or applies `fully_shard` to every block and the root), then times steps with and without gradient synchronisation. Each timed step starts after a `dist.barrier()` so ranks start together, and on GPU synchronises the device before reading the clock. The step time used is the slowest rank's, because the next step cannot start before it.

The gloo variant runs **real collectives** with the same semantics as NCCL: the same buckets, the same all-gathers and reduce-scatters, the same ordering, and the same way of measuring exposure. Its *times* are those of processes on one machine moving data through loopback sockets and shared memory; they say nothing about NVLink or InfiniBand.

## What the evidence says

- **ESTABLISHED.** Gradient bucketing with overlap and skipped synchronisation (Li et al., 2020, who report near-linear scaling to 256 GPUs with appropriate configuration; company/team claim for their setup). Fully sharded data parallelism (Zhao et al., 2023; ZeRO-style sharding). The bus-bandwidth convention (nccl-tests).
- **REASONABLE INDUSTRY PRACTICE.** Reporting exposed communication as sync minus no-sync step time; small per-GPU batches and slower links make it grow.
- **Measured here, CPU only:** the numbers in the lab below. GPU figures are PROJECTED; multi-GPU is not piloted course-side (the course's Colab account has one GPU), so the main-path figures come from the formulas above until a learner's own measurement replaces them.

## Lab

### Experiment contract

- **Question:** how much does gradient communication add to a data-parallel step (exposed communication), and how does DDP bucket size change it; how does FSDP2 compare?
- **Hypothesis and status:** with one bucket the all-reduce cannot overlap, so exposed ≈ the full communication time; smaller buckets hide part of it; established mechanism, magnitude depends on hardware.
- **Baseline:** DDP with one bucket larger than all gradients (no overlap possible).
- **Changed variable:** bucket size (1 MiB vs 25 MiB on the toy model, 5/25/100 MiB on the main path); separately, DDP vs FSDP2. **Controlled:** model, init seed, per-rank batch and inputs, world size, steps, dtype, torch version.
- **Comparison axis:** equal work (same tokens per rank per step, same model).
- **Budget:** main path 1 node with 2–8 GPUs, about 15 GPU-minutes per GPU; free GPU Kaggle 2× T4, about 15 minutes; free CPU about 3.5 minutes.
- **Metrics and decision rule:** exposed communication = median(sync) − median(no sync), slowest rank per step, 95% bootstrap interval; overlap fraction. Decision rule: call a bucket size "better" only if its exposed-communication interval lies entirely below the other's; otherwise report "not distinguishable at this number of steps".
- **Correctness checks:** both arms of each run complete on every rank; for DDP, all ranks hold identical parameters after the synchronised steps (`run_dp.py` checks a parameter checksum per rank); lab tests pass.
- **Fallback evidence:** published scaling numbers, labelled as such, if you have no multi-GPU machine.
- **Limits:** one model size, one node; CPU gloo times say nothing about GPU interconnects.

**Folder:** [`labs/module-02/lesson-03/`](../../labs/module-02/) · **Time:** about 70 minutes · **Pass check:** `pytest labs/module-02/lesson-03` passes; your table reports exposed communication with intervals.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1 node, 2–8× H100 SXM (or A100), about 15 minutes | `run_dp.py --device cuda --world 8 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --dtype bf16 --buckets 5 25 100 --steps 30`, plus a profiler trace of rank 0 (not run in this build; multi-GPU is not piloted) |
| Free GPU (Kaggle "GPU T4 ×2") | 2× T4 over PCIe | `run_dp.py --device cuda --world 2 --preset pilot-10m --batch 8 --seq 512 --buckets 1 25 --steps 30`; you will see real NCCL over PCIe, not NVLink, and no BF16 |
| Free CPU | laptop, 2 gloo processes; measured at about 3.5 minutes with `--steps 30` on a 16-thread laptop (2026-10-03) | `run_dp.py --steps 30` |

1. **Implement** `bus_bandwidth`, `exposed_comm` and `overlap_fraction` in `lab.py`; `pytest labs/module-02/lesson-03` checks them.
2. **Predict.** Before running, compute for your variant the gradient bytes, the bytes per rank of one all-reduce, and the DDP bucket count at 1 MiB and 25 MiB (`frontierlab.perf.dist.ddp_buckets`). Write down which bucket size you expect to show less exposed communication.
3. **Run.**

   ```bash
   python labs/module-02/lesson-03/run_dp.py --steps 30
   ```

   Part 1 prints all-reduce times and bus bandwidth at 1–64 MiB. Part 2 prints, for DDP at each bucket size and for FSDP2, the synchronised and unsynchronised step times, the exposed communication with its interval, and the overlap fraction.
4. **Read the evidence.** Does your decision rule separate the bucket sizes? If the intervals overlap, how many steps would you need? (The interval width shrinks roughly with $1/\sqrt{\text{steps}}$.)
5. **Main path only: the timeline.** Profile two DDP steps on rank 0 with `frontierlab.perf.profiling.profile_steps(..., device="cuda", trace_path=...)`, open the trace in Perfetto and mark where the `nccl` kernels overlap backward GEMMs and where they run alone.

Measured in this build (free CPU, Windows 11, 2 gloo ranks with 8 threads each, torch 2.14.1+cpu, toy model, B = 8, T = 128 per rank; other processes were running): all-reduce bus bandwidth 0.08–0.48 GB/s (loopback sockets, not an interconnect); one all-reduce of the 7.3 MB of gradients would take about 51 ms at the 64 MiB bandwidth. Two runs of the same script:

| Setting | 10 timed steps per arm: exposed ms [95% CI] | 30 timed steps per arm: exposed ms [95% CI] |
|---|---|---|
| DDP, 25 MiB buckets (1 bucket, no overlap possible) | 172 [93, 239] | 282 [201, 377] (32% of the step) |
| DDP, 1 MiB buckets (3 buckets) | 35 [−65, 141] | 62 [8, 130] (10% of the step) |
| FSDP2 (reduce-scatter part) | 153 [58, 224] | 229 [175, 274] |

At 10 steps the two bucket sizes' intervals overlap, so the decision rule says "not distinguishable"; at 30 steps they do not (130 < 201), and smaller buckets expose less communication, as the mechanism predicts. The absolute numbers moved between the runs (machine load changed), which is why both arms are always measured in the same run. That the one-bucket exposure exceeds the 51 ms bandwidth estimate says gloo's per-collective costs on this machine are larger than its large-buffer bandwidth suggests. Runtime: about 3.5 minutes at 30 steps. Every DDP run also passed the replica checksum check.

<details>
<summary>Hint for step 2</summary>

Gradient bytes are the number of parameters times 4 (FP32 gradients; tied embeddings are one parameter). `ddp_buckets` walks parameters in reverse order and closes a bucket when it reaches the cap; when you pass `bucket_cap_mb` explicitly, the first bucket uses the same cap.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-02/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-02/lesson-03`.

</details>

## Common mistakes

- **Timing rank 0 only.** The step ends when the slowest rank ends. Take the maximum across ranks per step.
- **Using algorithm bandwidth to compare 2 and 8 GPUs.** It falls with $n$ for the same hardware; bus bandwidth does not.
- **Reading FSDP2's no-sync arm as "no communication".** `set_requires_gradient_sync(False)` removes only the reduce-scatter; all-gathers remain. Its "exposed" number is the reduce-scatter part.
- **Calling one bucket "efficient" because it launches fewer collectives.** It cannot start until backward is over; nothing overlaps.
- **Synchronising inside every micro-batch of gradient accumulation.** Wrap the first micro-batches in `no_sync()` (DDP) or `set_requires_gradient_sync(False)` (FSDP2).
- **In-place ops on an FSDP2 module's outputs.** PyTorch warns that an in-place op on a returned view can skip the pre-backward all-gather; use out-of-place ops on outputs.

## References

- S. Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*, VLDB 2020, abstract and section 3. https://arxiv.org/abs/2006.15704
- Y. Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*. https://arxiv.org/abs/2304.11277
- PyTorch 2.14 documentation: [fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html); signatures and the 25 MiB / 1 MiB bucket defaults checked against the installed torch 2.14.1.
- NVIDIA, *nccl-tests performance documentation*. https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- NVIDIA H100 specifications (interconnect). https://www.nvidia.com/en-us/data-center/h100/
- Shared code: `labs/common/frontierlab/perf/dist.py`; versions in [references/versions.md](../../references/versions.md).

## Next

[02.4 · Validating a performance claim](lesson-04.md)
