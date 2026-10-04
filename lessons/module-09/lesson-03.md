---
id: "09.3"
module: 9
minutes: 35
practice_minutes: 120
prerequisites: ["09.1", "09.2", "02.3", "02.4"]
objectives:
  - Run torchtitan 0.3.0 on one 8-GPU node in two layouts and report throughput, MFU, memory per rank and exposed communication with intervals, under an experiment contract.
  - Predict the same quantities with the lesson 09.1 planner before running, and explain the gap between prediction and measurement.
  - Build a tested-capability matrix that records which framework features composed for the course architecture, as opposed to the documented feature list.
  - Diagnose a composition failure (here: tensor parallelism with QK-norm) from a correctness probe and fix it.
volatility: implementation
sources:
  - title: "Liang et al. — TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training"
    url: https://arxiv.org/abs/2410.06511
  - title: "pytorch/torchtitan at tag v0.3.0 (README, run_train.sh, torchtitan/config/README.md, configs.py, models/llama3/config_registry.py, components/metrics.py, tools/profiler.py)"
    url: https://github.com/pytorch/torchtitan/tree/v0.3.0
  - title: "torchtitan releases (v0.3.0, 2026-09-03: typed Python configuration)"
    url: https://github.com/pytorch/torchtitan/releases
  - title: "PyTorch 2.14 — torch.distributed.tensor.parallel (ColwiseParallel, RowwiseParallel, parallelize_module)"
    url: https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
  - title: "PyTorch 2.14 — torch.distributed.fsdp.fully_shard (FSDP2)"
    url: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
  - title: "NVIDIA nccl-tests — performance documentation (bus bandwidth)"
    url: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
last_verified: "2026-10-04"
---

# 09.3 · A measured multi-GPU investigation

Lessons 09.1 and 09.2 predicted what a layout should cost. This lesson measures it: two parallel layouts of an 8B model on one 8-GPU node with torchtitan, with throughput, MFU, memory per rank and exposed communication reported under an experiment contract, next to the planner's prediction. Alongside it you build the tested-capability matrix — which framework features actually composed for the course architecture — because a feature list says what exists, not what works together for your model.

> [!IMPORTANT]
> The main path of this lesson needs one node with 8 GPUs. The course's own compute is a single-GPU Colab account, so **the multi-GPU main path has not been piloted**: every GPU number on this page is PROJECTED from the planner's formulas and labelled so. The free CPU variant runs the same mechanisms on 4 processes and its numbers are measured, but CPU collectives say nothing about NVLink.

## Why this matters at a frontier lab

A layout decision ends in a measurement, not a calculation. The planner of lesson 09.1 assumes bandwidths, an MFU and zero overlap; a real step overlaps some communication, exposes some, and spends time in places no formula has (allocator retries, a slow rank, a kernel that does not overlap). A research engineer is expected to produce a short, trustworthy report: same model, same tokens, two layouts, numbers with intervals, a trace that shows where the time went, and the decision the numbers support. The capability matrix is the other half of the job. Frameworks advertise FSDP, TP, PP, CP, Float8 and compile; for a model with an unusual attention (QK-norm, MLA, a hybrid) some of these combinations fail silently — the run trains, but the gradients are wrong. The course model is a small example of exactly that.

## The idea

### What to measure, and how

| Quantity | Where it comes from | Pitfall |
|---|---|---|
| tokens/s per GPU and MFU | torchtitan's per-step metrics line (`tps`, `tflops`, `mfu`); at v0.3.0 `tps` is per device (tokens of a DP rank ÷ time ÷ the non-DP degree) and `mfu` uses torchtitan's own FLOPs-per-token count against the BF16 peak | the first steps include `torch.compile` and allocator warm-up: drop them (`--skip`); compare torchtitan's FLOPs-per-token with the course's (lesson 01.2) before comparing MFUs |
| memory per rank | the same line, `memory: X GiB (Y%)` = peak *reserved* by the caching allocator | reserved ≥ allocated; compare like with like; ranks can differ (embedding, last layer) |
| exposed communication | a profiler trace per rank: time when an NCCL kernel runs and no compute kernel does | summing NCCL kernel time counts the overlapped part too |
| interconnect baseline | NCCL all-reduce bus bandwidth (lesson 02.3's script with `--device cuda`) | algorithm bandwidth falls with the rank count; report bus bandwidth |

Exposed communication from a trace is interval arithmetic: with $\mathcal{C}$ the union of the NCCL kernels' intervals on one GPU and $\mathcal{K}$ the union of all other kernels' intervals,

$$t_{\text{exposed}} = |\mathcal{C}| - |\mathcal{C} \cap \mathcal{K}|.$$

It is a per-rank number; the step is as slow as its slowest rank, so report the range over ranks. It complements lesson 02.3's sync-minus-no-sync method: the trace needs no second run, but sees only one profiled step.

### The two layouts

Llama 3.1 8B, 32 layers, sequence 8,192, global batch 16 sequences (131,072 tokens) per step, BF16 parameters with FP32 reductions, selective activation checkpointing, `torch.compile`:

- **A — FSDP2 over 8** (`m09_fsdp8`): every GPU holds 1/8 of the weights, gradients and AdamW state and processes 2 sequences.
- **B — FSDP2 over 4 × TP 2** (`m09_fsdp4_tp2`): GPU pairs split every matrix (with sequence parallelism); FSDP2 shards over the 4 pairs; each pair processes 4 sequences.

Both process the same tokens per GPU, so their tokens/s are directly comparable. On one node every collective uses NVLink; what differs is *which* collectives: B adds TP's per-layer activation all-gathers and reduce-scatters, which sit on the critical path unless they are overlapped (torchtitan has async TP for that, a separate feature), and halves the FSDP traffic.

### torchtitan 0.3.0 in one paragraph

torchtitan (Liang et al., 2024) is PyTorch's reference pre-training codebase: FSDP2, TP (including async TP), PP schedules, CP, selective and full activation checkpointing, distributed checkpointing (DCP, including async), Float8 and MXFP8, and `torch.compile`, composed in one trainer (README at tag v0.3.0). Version 0.3.0 (released 2026-09-03, requires PyTorch 2.14) describes a run as **a Python function returning a `Trainer.Config`**: `run_train.sh` reads `MODULE` (an importable module) and `CONFIG` (the function), and launches `torchrun -m torchtitan.train --module $MODULE --config $CONFIG`. The older `--section.option` flags still work and take precedence but are deprecated "and will be deleted" (`torchtitan/config/README.md`); the world size must equal the product of the parallel degrees. `COMM_MODE="fake_backend"` validates a configuration on one GPU without NCCL — run it before renting the node for long. The course's two layouts are functions in `labs/module-09/lesson-03/titan_configs.py`, built on the shipped `llama3_8b()` config; every field name was checked against the tag's sources.

### The tested-capability matrix

For each feature the framework documents, record what *you* observed for *your* architecture: composed (ran and passed a correctness check), failed (with the error or the mismatch), or untestable here (and why). "Ran without an error" is not "composed": a feature that trains with wrong gradients is the dangerous case. The course's CPU probes (`frontierlab.dist.capability`) check each combination against a single-process reference in float64.

## Worked example

### Predicting the two layouts

The planner of lesson 09.1 with the two layouts (`Layout(dp=8, zero=3)` and `Layout(tp=2, dp=4, zero=3)`):

- **FLOPs and time.** Training FLOPs per token at $T = 8192$: $5.15 \times 10^{10}$ (`frontierlab.calc.flops_per_token`, lesson 01.2). Each GPU processes 16,384 tokens per step: $8.43 \times 10^{14}$ FLOPs. At an *assumed* 40% MFU of 989 TFLOP/s that is 2.13 s per step, or $989 \times 10^{12} \cdot 0.40 / 5.15 \times 10^{10} = 7{,}690$ tokens/s per GPU (PROJECTED).
- **Model state per GPU** (identical in both): the planner's convention is 18 bytes per parameter (2 BF16 weights + 4 gradients + 12 master and moments), sharded 8 ways in effect (A shards over 8, B splits by 2 and shards over 4): $8.03\text{B} \times 18 / 8 = 18$ GB. FSDP2's mixed-precision policy keeps only the FP32 shards and casts to BF16 on the fly, so torchtitan should report about $8.03\text{B} \times 16 / 8 = 16$ GB of state plus the gathered blocks.
- **Activations per GPU:** 88 GB without recomputation, 4.3 GB with full recomputation; torchtitan's selective checkpointing lands in between, which only the measurement shows. Both layouts hold the same activations per GPU (B's pairs process twice the sequences, each GPU half of each).
- **Communication per GPU per step** (ring arithmetic of 09.1): A, ZeRO-3 over 8: $\tfrac78 (8.03\text{B} \cdot 4 + 2 \cdot 8.03\text{B} \cdot 2) = 56.2$ GB. B: TP $8 \cdot \tfrac12 \cdot 8192 \cdot 4 \cdot 4096 \cdot 2 \cdot 32 = 34.4$ GB plus FSDP over 4 of half the model, 24.1 GB.
- **Upper bound on exposed communication** at an *assumed* 300 GB/s bus bandwidth: A 187 ms (8.8% of the step), B $115 + 80 = 195$ ms (9.1%).

The prediction: the two are within noise of each other in bytes; FSDP's all-gathers prefetch and overlap well, while TP's collectives sit between dependent matmuls, so the *measured* exposed time should favour A unless async TP is on. Hypothesis: A ≥ B in tokens/s on one node. Every number here is PROJECTED.

### Interval arithmetic, by hand

Compute kernels at [0, 4] and [5, 9] µs; an NCCL kernel at [3, 6]. Union of NCCL: 3 µs. Overlap with compute: [3, 4] and [5, 6], 2 µs. Exposed: 1 µs — the gap [4, 5], where the GPU waited on the network.

## Shapes and cost

| Per GPU, Llama 3.1 8B | Layout A (FSDP 8) | Layout B (FSDP 4 × TP 2) | dtype |
|---|---|---|---|
| `q_proj` weight shard at rest | (4096/8, 4096) as a DTensor `Shard(0)` | (4096/2/4, 4096) | fp32 |
| `q_proj` while computing | (4096, 4096) gathered | (2048, 4096): TP shard, FSDP-gathered | bf16 |
| hidden states between blocks | (2, 8192, 4096) | (4, 4096, 4096): SP splits the sequence | bf16 |
| gradient reduce-scatter per block | whole block over 8 | half block over 4 | fp32 |
| TP collectives per layer | — | 2 all-gathers + 2 reduce-scatters of (4, 8192, 4096) bf16, forward and backward | bf16 |

The main-path budget: 2 layouts × 2 repetitions × 60 steps at about 2–3 s per step, plus compilation, the optional layouts and the bandwidth test: about 0.5–0.75 node-hours, 4–6 GPU-hours (PROJECTED; plan section 12.1's row). At USD 2–3 per H100-hour that is roughly USD 10–20.

## Build it

**Main path.** `labs/module-09/lesson-03/run_titan.sh` clones torchtitan at `v0.3.0`, installs it with torch 2.14.1, downloads the Llama 3.1 tokenizer (`scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer`; gated, needs `HF_TOKEN`), records `nvidia-smi topo -m` and the versions, measures NCCL all-reduce bus bandwidth, dry-runs both configurations with `COMM_MODE="fake_backend"`, then runs A, B, A, B so drift affects both, with a profiler trace at step 50 (`profiler.enable_profiling`, `profile_freq = 50`). `titan_report.py` parses every rank's metrics lines (`step: … memory: …GiB(…%) tps: … tflops: … mfu: …%`, the format in `components/metrics.py` at v0.3.0), drops the warm-up steps, reports medians with bootstrap intervals, and computes exposed communication per rank from the traces (`rank{r}_trace.json.gz`, CUDA kernel events whose names contain `nccl` against all other kernels).

**Free CPU path.** `frontierlab/dist/measure.py` runs the course model on 4 `gloo` processes under DDP, FSDP2 and FSDP2 × TP (a 2 × 2 mesh), times steps with and without gradient synchronisation (exposed communication as in 02.3), and counts the bytes of parameters, gradients and optimizer state each rank holds from the tensors themselves (local shards of DTensors). `frontierlab/dist/capability.py` probes each feature against a single-process float64 reference.

**What the probes found.** Applying the standard Megatron-style TP plan (`ColwiseParallel` on q, k, v, gate, up; `RowwiseParallel` on o, down) to the course model fails in two ways, both caught only because each probe compares with a reference:

1. The attention module reshapes with its *global* head counts (`view(B, T, self.H, hd)`), so after the column split the reshape raises. torchtitan's own `parallelize` functions change the per-rank head counts for their models; the course model needs the same (`layer.self_attn.H //= tp`).
2. With that fixed, the run trains and the loss matches — but the gradients of the **QK-norm weights** differ from the reference by about $4 \times 10^{-3}$. The norm weights are replicated on every TP rank, yet each rank only sees its own heads, so each holds a *partial* gradient. Summing them over the TP group (`_allreduce_qk_norm_grads`) makes TP agree with one process to $10^{-17}$. Without the fix the replicas of a "replicated" parameter drift apart, silently.

The same probe at 4 ranks also records that TP = 4 cannot split the toy model's 2 KV heads without KV replication. These are findings about the course model, torch 2.14.1 and DTensor — exactly the kind of row the matrix exists for.

## What the evidence says

- **PUBLICLY DOCUMENTED:** torchtitan's feature list and configuration format at v0.3.0 (repository at the tag); the torchtitan paper's stacked speed-ups for Llama 3.1 8B/70B/405B on 128–512 H100s (abstract: 65.08% with 1D parallelism at 128 GPUs for 8B, a further 12.59% with 2D at 256 GPUs for 70B, a further 30% with 3D at 512 GPUs for 405B; the authors' claim against their own baselines).
- **REASONABLE INDUSTRY PRACTICE:** FSDP alone up to the point where the per-GPU batch gets too small or the all-gathers stop overlapping; TP inside a node after that; measure before adding a dimension. A tested-capability matrix per architecture and framework version.
- **MEASURED here (CPU only):** the capability matrix and the CPU layout comparison below. **Not measured:** every GPU throughput, MFU, memory and exposed-communication figure; they are PROJECTED until the main-path run.
- **ESTABLISHED / PROMISING tags:** FSDP2 and TP — ESTABLISHED. Async TP, Float8 + FSDP2 + compile together — PROMISING (documented and used in torchtitan, few independent reports); test them for your model before relying on them.

## Lab

### Experiment contract

- **Question:** on one 8× H100 node, does FSDP2 × TP 2 train Llama 3.1 8B faster than FSDP2 over 8 at equal tokens per step? Decision informed: the node-level layout for the module project's plan.
- **Hypothesis and status:** A ≥ B in tokens/s (prediction above); reported effect direction in general (torchtitan uses TP for larger models and multi-node scale), may not hold for every model size.
- **Baseline:** layout A, `m09_fsdp8`, as shipped (no extra tuning; both layouts get the same zero tuning budget).
- **Changed variable:** the layout. **Controlled:** model, tokenizer, data (`c4` through torchtitan's loader), seed, 16 × 8,192 tokens per step, BF16/FP32 policy, selective activation checkpointing, `torch.compile`, torch and torchtitan versions, the node.
- **Comparison axis:** equal tokens per step (and per GPU).
- **Budget:** 4–6 GPU-hours on the main path (PROJECTED); free CPU about 6 minutes.
- **Metrics and decision rule:** tokens/s per GPU and MFU, median over steps 11–60 of both repetitions with 95% bootstrap intervals; peak reserved memory per rank; exposed communication per rank from the step-50 trace. Adopt B only if its tokens/s interval lies entirely above A's; otherwise keep A (fewer moving parts).
- **Correctness checks:** both layouts' losses over the first 20 steps agree within the run-to-run spread of A's two repetitions (same seed and data); the CPU capability probes for FSDP2 and for TP (with the QK-norm fix if your model has QK-norm) pass; the fake-backend dry run passes.
- **Fallback evidence:** the torchtitan paper's published numbers and the planner's projection, labelled.
- **Limits:** one node (no InfiniBand), one model size, one sequence length; selective checkpointing policy fixed; async TP off.

**Folder:** [`labs/module-09/lesson-03/`](../../labs/module-09/) · **Time:** about 2 hours attended (main path) or 1 hour (CPU) · **Pass check:** `pytest labs/module-09/lesson-03` passes; your report has both layouts with intervals and the decision; your capability matrix has every row filled with evidence.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1 node, 8× H100 SXM 80 GB (NVLink), about 4–6 GPU-hours, USD 10–20 | `bash labs/module-09/lesson-03/run_titan.sh`, then `titan_report.py WORK --skip 10` and `matrix.py --gpu-results gpu_matrix.json`. **Not run in this build; multi-GPU is not piloted course-side.** |
| Free GPU (Kaggle "GPU T4 ×2") | 2× T4, PCIe, no BF16 tensor cores | `cpu_layouts.py`'s worker on CUDA (`layout_worker(..., device="cuda")`, `--world 2`, `fsdp2` vs `ddp`) and `python -m frontierlab.dist.capability --world 2` for the NCCL column; torchtitan with an 8B model does not fit; untested |
| Free CPU | laptop, 4 `gloo` processes; measured 5 min 24 s for `cpu_layouts.py`, 53 s for `matrix.py` | `cpu_layouts.py` and `matrix.py` below |

1. **Implement** `parse_titan_line`, `exposed_comm`, `state_bytes_per_rank` and `summarize` in `lab.py`; `pytest labs/module-09/lesson-03` checks them (the parser on `sample_titan.log`, a **synthetic** sample in the v0.3.0 format, numbers invented).
2. **Predict** both layouts with the planner (the worked example's numbers) and write your prediction into the contract before running.
3. **Run** the main path (`run_titan.sh`) or the CPU variant:

   ```bash
   python labs/module-09/lesson-03/cpu_layouts.py
   python labs/module-09/lesson-03/matrix.py
   ```

4. **Report** (one page): the table of both layouts with intervals, the exposed communication per rank, the memory per rank, the prediction next to the measurement and the reason for the gap, the decision, and the capability matrix.

<details>
<summary>Hint for TODO 2</summary>

Merge each list into a sorted union of disjoint intervals first. Then the overlap is the sum, over pairs of a communication interval and a compute interval, of `max(0, min(b, y) - max(a, x))`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04 on the build laptop (Windows 11, 16 threads, torch 2.14.1+cpu, 4 `gloo` processes, another agent's jobs running). **CPU only; no GPU figure on this page is measured.**

`cpu_layouts.py` (toy model, 1.84M parameters, 8 × 128 tokens per DP rank, 2 rounds × 8 steps per layout, 5 min 24 s):

| Layout | step, slowest rank | exposed communication | tokens/s | held per rank (params + grads + AdamW) | your prediction |
|---|---|---|---|---|---|
| DDP (DP 4) | 1,053 ms [1,001, 1,131] | 216 ms [95, 312] | 3,889 | 29.38 MB | 29.38 MB |
| FSDP2 (DP 4) | 1,750 ms [1,392, 2,362] | 594 ms [215, 1,322] (reduce-scatter part) | 2,340 | 7.35 MB | 7.35 MB |
| FSDP2 × TP (2 × 2) | 1,765 ms [1,602, 1,870] | 410 ms [241, 558] | 1,160 (2 DP ranks) | 11.55 MB | — |

What a write-up should say. Memory per rank is exactly what the formula predicts (16 bytes per parameter, divided by 4 under FSDP2). FSDP × TP holds more than a quarter because the tied embedding is not tensor-parallel in the course's plan and is only sharded over the 2 DP ranks — a plan choice worth a row of its own. On CPU processes DDP is the fastest layout by far: the toy model's collectives are latency-bound (`gloo` over loopback), and FSDP2 and TP issue many more, smaller collectives per step. That says nothing about H100s, where FSDP's all-gathers overlap with compute and NVLink latency is microseconds; it is why the GPU decision needs the GPU run.

`matrix.py` (4 processes, float64, 53 s) — the CPU column of the matrix:

| Feature (torchtitan 0.3.0 documents it) | CPU probe result for the course model |
|---|---|
| FSDP2 | composed (parameters after one step equal one process to $7 \times 10^{-18}$) |
| Activation checkpointing (with FSDP2) | composed |
| TP | standard plan **failed** (head-count reshape); with per-rank heads **failed** (QK-norm gradients off by $3.9 \times 10^{-3}$); with the QK-norm all-reduce composed ($1.4 \times 10^{-16}$); FSDP2 × TP composed |
| PP (`torch.distributed.pipelining`, 1F1B) | composed (loss only checked) |
| CP | course ring attention composed ($1.3 \times 10^{-15}$); torchtitan's CP not testable here |
| DCP (FSDP2 → unsharded load; async save) | composed (exact) |
| torch.compile | untestable here: Inductor needs a C++ compiler (no `cl.exe` on the build laptop) |
| Float8, MXFP8, async TP, NCCL, torchtitan itself | untestable on CPU |

</details>

<details>
<summary>Reference solution</summary>

`labs/module-09/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-09/lesson-03`.

</details>

## Common mistakes

- **Comparing step times across layouts with different tokens per step.** Compare tokens/s per GPU at equal tokens per step.
- **Keeping the compile steps in the median.** The first steps of a compiled run are many times slower; skip them and say how many.
- **Summing NCCL kernel time and calling it "communication overhead".** Most of it may overlap compute; report the exposed part.
- **Reading "memory: X GiB" as what the model needs.** It is the allocator's peak reservation, which includes fragmentation and cached blocks.
- **Filling the capability matrix from the README.** "Documented" and "composed for my model" are different columns. A feature that runs is not composed until it passes a correctness check.
- **Trusting a loss curve to validate TP.** The QK-norm bug leaves the loss correct at the first step and corrupts only some parameters' gradients.
- **Using deprecated CLI overrides in new scripts.** At 0.3.0 they still work but are slated for removal; write the layout as a config function.

## References

- W. Liang et al., *TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training*, abstract. https://arxiv.org/abs/2410.06511
- pytorch/torchtitan at tag v0.3.0: README, `run_train.sh`, `torchtitan/config/README.md`, `torchtitan/config/configs.py`, `torchtitan/models/llama3/config_registry.py`, `torchtitan/components/metrics.py`, `torchtitan/tools/profiler.py`, `scripts/download_hf_assets.py` (read 2026-10-04). https://github.com/pytorch/torchtitan/tree/v0.3.0
- torchtitan release notes. https://github.com/pytorch/torchtitan/releases
- PyTorch 2.14 documentation: [tensor parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html), [fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).
- NVIDIA, *nccl-tests performance documentation*. https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- Shared code: `labs/common/frontierlab/dist/measure.py`, `capability.py`; versions in [references/versions.md](../../references/versions.md).

## Next

[09.4 · Failure and recovery](lesson-04.md)
