---
id: "09.2"
module: 9
minutes: 40
practice_minutes: 80
prerequisites: ["09.1", "02.3"]
objectives:
  - Derive the bubble and the activation memory of GPipe, 1F1B and interleaved 1F1B for p stages and m micro-batches, and check them against a simulator.
  - Explain how splitting the backward pass into input and weight gradients (zero-bubble schedules) and feeding micro-batches from both ends (DualPipe, DualPipeV) shrink the bubble, and what each costs in memory and parameters.
  - Run a real pipeline across processes, verify it computes the same gradients as one process, and measure its bubble against the simulator fed with measured forward and backward times.
  - Choose a schedule for a stated model and memory budget from the bubble and memory formulas.
volatility: concept
sources:
  - title: "Huang et al. — GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (bubble O((K-1)/(M+K-1)))"
    url: https://arxiv.org/abs/1811.06965
  - title: "Narayanan et al. — PipeDream: Fast and Efficient Pipeline Parallel DNN Training (1F1B)"
    url: https://arxiv.org/abs/1806.03377
  - title: "Narayanan et al. — Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (section 2.2: interleaved schedule)"
    url: https://arxiv.org/abs/2104.04473
  - title: "Qi et al. — Zero Bubble Pipeline Parallelism"
    url: https://arxiv.org/abs/2401.10241
  - title: "DeepSeek-V3 Technical Report (section 3.2.1: DualPipe and computation-communication overlap; Table 2)"
    url: https://arxiv.org/abs/2412.19437
  - title: "deepseek-ai/DualPipe (README: DualPipe and DualPipeV, bubble and memory table; dualpipe.py, dualpipev.py)"
    url: https://github.com/deepseek-ai/DualPipe
  - title: "The Llama 3 Herd of Models (section 3.3.2: pipeline schedule with flexible micro-batch count, balanced first and last stages)"
    url: https://arxiv.org/abs/2407.21783
  - title: "PyTorch 2.14 — torch.distributed.pipelining (Schedule1F1B, ScheduleInterleaved1F1B, ScheduleZBVZeroBubble, ScheduleDualPipeV)"
    url: https://docs.pytorch.org/docs/stable/distributed.pipelining.html
last_verified: "2026-10-04"
---

# 09.2 · Pipeline schedules and overlap

Pipeline parallelism splits the layers over devices, so a device can only work on a micro-batch once the previous stage has finished it; the time devices spend waiting is the bubble. This lesson derives the bubble and the activation memory of the standard schedules, builds a simulator that reproduces the published formulas — including the zero-bubble split of the backward pass and DeepSeek's DualPipe — and then runs a real pipeline across processes to measure the bubble and compare it with the simulator.

## Why this matters at a frontier lab

Every published layout in lesson 09.1 used pipeline parallelism: PP = 16 in both Llama 3 405B and DeepSeek-V3. With 16 stages and a schedule that leaves each device idle for 15 micro-batch slots per step, a run with 16 micro-batches spends almost half its time waiting. The schedule is therefore not an implementation detail: it decides the fraction of a 10,000-GPU cluster that does nothing, how many micro-batches the global batch must contain (which feeds back into the learning-rate and batch-size choices of Module 7), and how much activation memory the first stage needs (which decides recomputation, lesson 09.1). DeepSeek wrote a new schedule, DualPipe, specifically to hide the cross-node expert all-to-all of DeepSeek-V3.

## The idea

### GPipe and 1F1B: the same bubble, different memory

Write $p$ for the number of stages, $m$ for the micro-batches per optimizer step, $F$ for one stage's forward time per micro-batch and $B$ for its backward time.

- **GPipe** (Huang et al., 2019) runs all $m$ forwards, then all $m$ backwards. The first micro-batch reaches the last stage after $(p-1)F$, and the last backward leaves the first stage $(p-1)B$ after the last stage finished. Each device is busy $m(F+B)$ and idle $(p-1)(F+B)$: the bubble fraction is
$$\beta = \frac{p-1}{m+p-1}.$$
  Every stage keeps the activations of all $m$ micro-batches until their backward.
- **1F1B** (PipeDream-Flush, Narayanan et al.) lets stage $s$ run $p-s-1$ warm-up forwards, then alternate one forward and one backward, then drain the remaining backwards. The idle time is the same, $(p-1)(F+B)$ — the dependency chain from the first forward to the last backward has not changed — but stage $s$ holds at most $p - s$ micro-batches: **memory independent of $m$**. That is why 1F1B, not GPipe, is the default.

Megatron's papers state the bubble as idle time over *ideal* time, $(p-1)/m$; GPipe's is idle over *total* time, $(p-1)/(m+p-1)$. Both describe the same schedule; this course reports idle over total, as the simulator measures it.

### Interleaving: smaller chunks, smaller bubble

In **interleaved 1F1B** (Megatron-LM, section 2.2) each device holds $v$ non-adjacent chunks of $L/(pv)$ layers (device $s$ holds chunks $s, s+p, \dots$). A micro-batch passes each device $v$ times, each pass $1/v$ as long, so the warm-up and drain shrink by $v$: idle time $(p-1)(F+B)/v$, bubble $(p-1)/(vm + p - 1)$. The price: $v$ times more point-to-point messages, $m$ must be a multiple of $p$ in the standard schedule, and the first stage holds $p\,(1 + \tfrac{p-1}{pv})$ chunks' worth of activations, more than 1F1B's $p$. Llama 3 uses an interleaved schedule and relaxes the "multiple of $p$" constraint so that the number of micro-batches can be set freely (section 3.3.2).

### Zero bubble: split the backward

A linear layer's backward has two independent parts (lesson 08.2's GEMMs): the **input gradient** $\partial L/\partial x$ (Dgrad), which the previous stage is waiting for, and the **weight gradient** $\partial L/\partial W$ (Wgrad), which nobody needs until the optimizer step. Qi et al. (2024) call them $B$ and $W$ and schedule them separately: send $B$ upstream at once, run $W$ later, in the gaps that would otherwise be bubble. With $B_{\text{full}} = B + W$, the DualPipe README gives ZB1P's bubble as $(p-1)(F + B_{\text{full}} - 2W)$ at the same activation memory as 1F1B's worst stage; the zero-bubble paper reports up to 23% higher throughput than 1F1B at similar memory and 31% with relaxed memory (abstract; company/team claim for their setup), and removes the remaining synchronisation at the optimizer step.

### DualPipe and DualPipeV

DeepSeek-V3 (section 3.2.1) faces a different problem: cross-node expert parallelism gives "an inefficient computation-to-communication ratio of approximately 1:1". DualPipe's answer:

- every chunk is split into attention, all-to-all dispatch, MLP and all-to-all combine, and the backward of attention and MLP into input and weight parts as in ZeroBubble;
- a forward chunk of one micro-batch and a backward chunk of another run *together* on a device ("F&B"), arranged so that one's communication runs while the other computes, with the share of GPU SMs given to communication tuned by hand;
- micro-batches are fed **from both ends of the pipeline at once**, so each device holds stage $s$ of one direction and stage $p-1-s$ of the other: two copies of its parameters.

From the DualPipe README (PUBLICLY DOCUMENTED, the same table as V3's Table 2), for PP stages:

| Method | Bubble | Parameters per device | Activations per device | Devices |
|---|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | 1× | PP | PP |
| ZB1P | $(PP-1)(F+B-2W)$ | 1× | PP | PP |
| DualPipe | $(PP/2-1)(F\&B + B - 3W)$ | 2× | PP + 1 | PP |
| DualPipeV | $(PP/2-1)(F\&B + B - 3W)$ | 2× | PP + 1 | PP/2 |

($B$ here is the full backward.) **DualPipeV** is the "cut-in-half" variant (the README credits Sea AI Lab): the PP stages are laid out in a V over PP/2 devices, device $d$ holding stages $d$ and $PP-1-d$, with micro-batches fed from one end only. The V3 report adds that "neither the bubbles nor activation memory will increase as the number of micro-batches grows", and that the doubled parameters cost little because the EP degree is large. PyTorch 2.14 ships `ScheduleDualPipeV` and `ScheduleZBVZeroBubble` in `torch.distributed.pipelining`.

### What a simulator can and cannot tell you

A schedule is a list of operations per device plus dependencies: a chunk's forward needs the previous stage's forward of the same micro-batch, its backward needs its own forward and the next stage's backward, a $W$ needs its $B$. Running each device's list in order, each op as soon as its inputs exist, is the longest path through that graph — exact for the model. It does not model communication bandwidth (only a fixed latency per transfer), kernel overlap inside an F&B pair, or stages of unequal cost; those are what the real pipeline and the GPU traces of lesson 09.3 measure.

## Worked example

$p = 4$, $m = 8$, $F = 1$, $B = 2$ (time units).

- **1F1B / GPipe.** Busy per device $8 \cdot 3 = 24$, idle $(p-1)(F+B) = 9$, makespan 33, $\beta = 9/33 = 3/11 = 0.273$. Peak activations: stage 0 holds 4 micro-batches under 1F1B, 8 under GPipe.
- **Interleaved, $v = 2$.** Idle $9/2 = 4.5$, makespan $24 + 4.5 = 28.5$, $\beta = 4.5/28.5 = 0.158 = 3/(2 \cdot 8 + 3)$. Stage 0 holds $4 (1 + 3/8) = 5.5$ stages' worth of activations (11 chunks of half a stage each), against 4 for plain 1F1B.
- **Zero bubble**, splitting $B = 2$ into $B = 1$ and $W = 1$: idle $(p-1)(F + B_{\text{full}} - 2W) = 3 \cdot (1 + 2 - 2) = 3$, makespan 27, $\beta = 0.111$.
- **DualPipe** on 4 devices with $F\&B = F + B_{\text{full}} = 3$ (no compute gain from overlap): idle $(4/2 - 1)(3 + 2 - 3) = 2$, $\beta = 2/26 = 0.077$; each device holds $PP + 1 = 5$ chunk-units of activations and two stages' parameters.

All four agree exactly with the simulator (`simulate.py --p 4 --m 8`). One caution found while building it: the README's DualPipe formula matches the simulated op lists of `dualpipe.py` exactly when $F = B = W$, but not for every ratio — at $F = 2, B = 1, W = 0.5$ the simulated idle time is 13.5 against the formula's 10.5 for $PP = 8$. The README gives the formula without a derivation, so treat it as an approximation and simulate your own ratios (INFERENCE about its scope).

## Shapes and cost

| Item (stage $s$, micro-batch $i$) | Shape | dtype | Device | Lifetime |
|---|---|---|---|---|
| activation sent to stage $s+1$ | $(b, T, h)$ — $(b, T/t, h)$ with TP + SP | bf16 (fp32 / fp64 on CPU here) | GPU $s$ → GPU $s+1$ | until received |
| gradient sent to stage $s-1$ | same | same | GPU $s$ → GPU $s-1$ | until received |
| saved activations of the stage's layers | about $L_s \cdot A_{\text{layer}}$ (lesson 09.1) | bf16 | GPU $s$ | from F($i$) to B($i$), or to W($i$) when split |
| weight-gradient inputs kept for W | the linears' inputs and output gradients | bf16 | GPU $s$ | from B($i$) to W($i$) |

Cost per micro-batch and boundary: $2 \cdot b\,T h e$ bytes point-to-point (activation and gradient), small next to TP's per-layer collectives, so PP is the dimension that tolerates the slow links. The bubble costs $\beta$ of the whole cluster; activation memory on the first stage is $p$ micro-batches' worth under 1F1B and ZB1P, $p+1$ chunks under DualPipe, $m$ under GPipe.

## Build it

`frontierlab/dist/schedule.py` generates op lists — `gpipe`, `one_f_one_b`, `interleaved`, `zb1p`, `dualpipe`, `dualpipev` — and `simulate()` runs them. DualPipe and DualPipeV are generated from the eight-step structure of `dualpipe.py` and `dualpipev.py` in the DualPipe repository (main branch, read 2026-10-04), with the same op counts per rank (step 1 `nF0` runs $(\text{ranks}/2 - \text{rank} - 1) \cdot 2$ forwards, and so on). The zero-bubble schedule is 1F1B with every backward split and its $W$ deferred; a device runs a deferred $W$ when its next op is not ready yet, or before a forward that would hold more than $p$ micro-batches — a greedy placement, not the paper's hand-made one, which reproduces the $(p-1)(F+B-2W)$ bubble at 1F1B's peak memory in every case the tests cover.

`frontierlab/dist/pipeline.py` is a real pipeline over processes. Rank $s$ keeps layers `stage_layers(L, p)[s]` of the course model; rank 0 the embedding, the last rank the norm and head. The tied embedding is a problem a pipeline creates: the input embedding and the output head share one matrix but now live on different ranks, so each keeps a copy and their gradients are all-reduced between the first and last rank after the backward (Megatron-LM does the same). Forward activations go to the next rank with `isend`; gradients come back with `recv`; each rank measures its compute time and the time blocked in `recv`. `check=True` compares the loss and every parameter's gradient with the same model run in one process: float64 differences of $1.4 \times 10^{-17}$ on 4 stages (measured).

## What the evidence says

- **ESTABLISHED.** 1F1B and interleaved 1F1B (Megatron-LM, used by Llama 3); the bubble formulas follow from the dependency structure and the simulator reproduces them exactly.
- **PROMISING.** Zero-bubble scheduling (Qi et al.) and its descendants (ZBV in PyTorch); DeepSeek-V3 builds on it. Independent large-scale reports are still few.
- **MODEL-SPECIFIC.** DualPipe, designed for DeepSeek-V3's cross-node expert all-to-all; its value depends on there being communication to hide. Its bubble formula is the README's; the overlap benefit is the company's claim, and this course's simulator, which has no communication, cannot test it.
- **Measured here (CPU):** the real pipeline's bubble and its agreement with the simulator, below. On GPUs, communication and unequal stages (the last stage has the output head; Llama 3 removes a layer from the first and last stages for this reason) change the numbers; the schedule structure does not.

## Lab

### Experiment contract

- **Question:** at equal micro-batches, do GPipe and 1F1B differ in bubble, and by how much in activation memory? Does the simulator, fed with measured F and B, predict the measured bubble? Decision informed: whether the simulator is good enough to choose schedules for the module project.
- **Hypothesis and status:** same bubble, $(p-1)/(m+p-1) = 0.27$ for $p = 4, m = 8$; activation memory $m = 8$ micro-batches per stage for GPipe vs $p - s$ for 1F1B. Established (dependency arithmetic).
- **Baseline:** GPipe.
- **Changed variable:** the schedule. **Controlled:** model (4 stages × 2 layers of the course model, hidden 256), micro-batches (8 of 4 × 128 tokens), data, seed, float32, threads per process, number of steps.
- **Comparison axis:** equal work per step.
- **Budget:** free CPU about 4 minutes; main path a few GPU-minutes.
- **Metrics and decision rule:** per-rank bubble = $1 - \text{busy}/\text{step time}$ (step time = slowest rank's wall time), median over 10 steps with a bootstrap interval; peak live micro-batches per rank. The schedules "differ in bubble" only if their intervals do not overlap on the same rank; the simulator "predicts" the bubble if its value falls inside the measured interval on most ranks.
- **Correctness checks:** loss and gradients equal one process in float64 ($< 10^{-12}$) before any timing.
- **Fallback evidence:** the simulator and the published formulas.
- **Limits:** CPU processes share cores and memory bandwidth; stages are not perfectly equal (the last has the head); point-to-point times on `gloo` are loopback copies.

**Folder:** [`labs/module-09/lesson-02/`](../../labs/module-09/) · **Time:** about 80 minutes · **Pass check:** `pytest labs/module-09/lesson-02` passes; `real_pipeline.py` part 1 passes; your write-up applies both decision rules.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 4–8 GPUs on one node, about 15 GPU-minutes | `simulate.py` as below, then a GPU pipeline with `torch.distributed.pipelining` (`Schedule1F1B` vs `ScheduleGPipe`, then `ScheduleInterleaved1F1B` and `ScheduleZBVZeroBubble`) on Baseline-0 split into 4 stages, timed with `torch.cuda.synchronize()` and profiled (lesson 02.2); or torchtitan's `pipeline_parallel_schedule` in lesson 09.3. Not run in this build; part of the Module 9 pilot |
| Free GPU (Kaggle "GPU T4 ×2") | 2× T4 | the GPU pipeline above with 2 stages; untested |
| Free CPU | laptop, 4 `gloo` processes | the steps below; `real_pipeline.py` measured at 3 min 26 s, `simulate.py` under a second |

1. **Implement** `bubble_ratio`, `one_f_one_b_order`, `peak_in_flight` and `measured_bubble` in `lab.py`; `pytest labs/module-09/lesson-02` checks them against the simulator.
2. **Simulate.**

   ```bash
   python labs/module-09/lesson-02/simulate.py --p 4 --m 8 --gantt
   python labs/module-09/lesson-02/simulate.py --p 8 --m 16
   python labs/module-09/lesson-02/simulate.py --p 8 --m 16 --F 1 --B 1 --W 1 --fb 2.4
   ```

   Read the Gantt rows: where is 1F1B idle, and which idle slots does ZB-1P fill with `W`? With `--fb 2.4` (F&B 20% faster than F + B because communication hides under it), how much does DualPipe gain?
3. **Measure.**

   ```bash
   python labs/module-09/lesson-02/real_pipeline.py
   ```

4. **Decide** with both rules, and explain any rank whose bubble differs from the others.

<details>
<summary>Hint for TODO 2</summary>

Warm-up is `min(p - s - 1, m)` forwards. Then for `i` in `range(m - warm)`: forward `warm + i`, backward `i`. Then the backwards that are left.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04 on the build laptop (Windows 11, 16 threads, torch 2.14.1+cpu, 4 `gloo` processes with 4 threads each, other jobs running), `real_pipeline.py`, 3 min 26 s. Part 1: largest gradient difference $1.4 \times 10^{-17}$, loss difference 0.

| | GPipe | 1F1B |
|---|---|---|
| bubble per rank, median [95% CI] | 0.37 [0.29, 0.46], 0.28 [0.21, 0.36], 0.31 [0.23, 0.38], 0.40 [0.28, 0.59] | 0.32 [0.30, 0.37], 0.32 [0.26, 0.36], 0.28 [0.24, 0.34], 0.30 [0.25, 0.33] |
| simulator with the measured F and B | 0.27 | 0.27 |
| step time (slowest rank) | 3,984 ms (simulated 4,132) | 2,820 ms (simulated 2,742) |
| mean F / B per micro-batch | 112.6 / 263.1 ms | 89.7 / 159.5 ms |
| peak live micro-batches per rank | 8, 8, 8, 8 | 4, 3, 2, 1 |

What a write-up should say. Bubble: every rank's intervals overlap between the schedules, so by the rule they do not differ — as the dependency arithmetic says. The measured bubbles sit a few points above the simulator's 0.27 (communication and unequal stages are not in the model; the last rank also computes the head and the loss), but the simulator fed with measured F and B predicts the *step time* within 4% for both schedules, so it is good enough to rank schedules. Memory: exactly the counts the formulas give. And one thing the bubble formula does not show: 1F1B's step is 1.4× shorter, because every one of its forward and backward ops ran faster (F 90 vs 113 ms, B 160 vs 263 ms). Holding 8 micro-batches of activations instead of at most 4 makes GPipe's working set larger than the CPU caches can serve (INFERENCE from the per-op times; on a GPU the analogue is allocator pressure and recomputation). Measured on CPU processes; not a GPU result.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-09/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-09/lesson-02`.

</details>

## Common mistakes

- **Choosing GPipe to "keep it simple".** Same bubble as 1F1B, $m/p$ times the activation memory.
- **Raising $m$ to shrink the bubble without accounting for the batch.** $m$ micro-batches of size $b$ per DP rank fix the global batch; a bigger $m$ may force a smaller $b$ (worse GEMM efficiency) or a larger global batch (a training-recipe change, Module 7).
- **Comparing bubbles defined differently.** $(p-1)/m$ (idle over ideal) and $(p-1)/(m+p-1)$ (idle over total) describe the same schedule.
- **Measuring a rank's bubble against its own wall time.** The step ends when the slowest rank ends; use that time for every rank, as the simulator does.
- **Forgetting the tied embedding.** Splitting a model with tied input and output embeddings puts the two uses on different stages; without the all-reduce of their gradients the copies drift apart.
- **Expecting DualPipe to help a dense model on one node.** Its gain is hidden communication; with little to hide it costs twice the parameters for a bubble that interleaving or ZB can also shrink.

## References

- Y. Huang et al., *GPipe*, abstract and section 2. https://arxiv.org/abs/1811.06965
- D. Narayanan et al., *PipeDream*. https://arxiv.org/abs/1806.03377
- D. Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, section 2.2. https://arxiv.org/abs/2104.04473
- P. Qi et al., *Zero Bubble Pipeline Parallelism*. https://arxiv.org/abs/2401.10241
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, section 3.2.1 and Table 2. https://arxiv.org/abs/2412.19437
- deepseek-ai, *DualPipe* repository (README, `dualpipe/dualpipe.py`, `dualpipe/dualpipev.py`, main branch, read 2026-10-04). https://github.com/deepseek-ai/DualPipe
- Llama Team, *The Llama 3 Herd of Models*, section 3.3.2. https://arxiv.org/abs/2407.21783
- PyTorch 2.14, `torch.distributed.pipelining`. https://docs.pytorch.org/docs/stable/distributed.pipelining.html
- Shared code: `labs/common/frontierlab/dist/schedule.py`, `pipeline.py`.

## Next

[09.3 · A measured multi-GPU investigation](lesson-03.md)
