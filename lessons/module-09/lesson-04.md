---
id: "09.4"
module: 9
minutes: 40
practice_minutes: 90
prerequisites: ["09.1", "01.5", "02.3"]
objectives:
  - Kill a distributed training run and restart it so that model, optimizer, scheduler, RNG and data-stream position are bit-identical to an uninterrupted run, and prove it.
  - Save a sharded checkpoint with torch.distributed.checkpoint and restore it into a different layout, and say which parts of the state survive a change of rank count exactly.
  - Measure checkpoint overhead (blocking and asynchronous), restart cost and lost work, and turn them into a goodput estimate.
  - Choose a checkpoint interval with Young's formula and with the exact Poisson model, using Llama 3's interruption record, and explain how elasticity (Gemini 2.5) changes the answer.
volatility: concept
sources:
  - title: "The Llama 3 Herd of Models (section 3.3.4: 466 interruptions in 54 days, 419 unexpected, ~78% of unexpected attributed to hardware; Table 5)"
    url: https://arxiv.org/abs/2407.21783
  - title: "Gemini 2.5 technical report (section 2.3: slice-granularity elasticity, split-phase SDC detection with deterministic replay)"
    url: https://arxiv.org/abs/2507.06261
  - title: "Jiang et al. — MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs (sections 4.4, 6)"
    url: https://arxiv.org/abs/2402.15627
  - title: "Wan et al. — ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development"
    url: https://arxiv.org/abs/2407.20143
  - title: "Young — A first order approximation to the optimum checkpoint interval (Communications of the ACM 17(9), 1974)"
    url: https://doi.org/10.1145/361147.361115
  - title: "Daly — A higher order estimate of the optimum checkpoint interval for restart dumps (Future Generation Computer Systems 22(3), 2006)"
    url: https://doi.org/10.1016/j.future.2004.11.016
  - title: "PyTorch 2.14 — torch.distributed.checkpoint (save, async_save, load, resharding)"
    url: https://pytorch.org/docs/stable/distributed.checkpoint.html
last_verified: "2026-10-04"
---

# 09.4 · Failure and recovery

At thousands of GPUs a training job is interrupted every few hours, so the question is not whether a run will fail but how much each failure costs and whether the restarted run is still the same run. This lesson kills a distributed run, restarts it and proves the restart bit-identical; saves sharded checkpoints with `torch.distributed.checkpoint` and restores them into a different layout; measures checkpoint overhead, restart cost and lost work; and turns Llama 3's interruption record into a goodput model with Young's checkpoint-interval rule and its exact counterpart.

## Why this matters at a frontier lab

Llama 3's report gives the most detailed public record: during a 54-day snapshot of 405B pre-training there were **466 job interruptions, 47 planned and 419 unexpected, and about 78% of the unexpected ones were attributed to confirmed or suspected hardware issues** (section 3.3.4, Table 5; faulty GPUs and HBM3 memory are the two largest causes). That is one unexpected interruption every 3.1 hours. Meta still reports "higher than 90% effective training time". Getting there takes three things this lesson builds: checkpoints that capture the *whole* state, so a restart continues the same run (lesson 01.1's exact resume, now across processes and shards); checkpoints that are cheap enough to take often; and an interval chosen from measured costs, not habit. A fourth, elasticity — continuing on fewer machines instead of waiting for a replacement — is what Gemini 2.5's report describes, and it needs checkpoints that can be loaded into a different layout.

## The idea

### What "the whole state" is

| Component | Why it matters | Where it lives in a sharded run |
|---|---|---|
| model weights | obviously | sharded (FSDP2 DTensors) or replicated (DDP) |
| optimizer state (AdamW moments, step counts) | without it the first steps after a restart are a different optimizer | sharded like the weights |
| LR scheduler and step | the schedule continues at the right point | one value, the same on every rank |
| data-stream position | the same batches in the same order; nothing repeated or skipped | depends on the sampler; here one generator, identical on every rank |
| RNG states | dropout, token dropping, stochastic rounding (lesson 08.3) draw from them | one per rank (CPU and CUDA) |

The test of a resume is the strongest one available: an interrupted-and-restarted run must end with every one of these **bit-identical** to an uninterrupted run with the same seed, and every logged loss must be equal. "The loss curve looks continuous" is not a test: lesson 01.1 showed a resume with the wrong data RNG that looks fine and is a different run.

### Sharded checkpoints with DCP

`torch.distributed.checkpoint` (DCP) saves a state dict of possibly sharded tensors: each rank writes only the pieces it owns (`__0_0.distcp`, `__1_0.distcp`, …) and rank 0 writes a `.metadata` file that records every tensor's global shape and the offsets of each saved piece. Loading is the reverse with a twist: each rank asks for the pieces *its current sharding* needs, and DCP reads them from whichever files contain them. That is **resharding on load**: save from FSDP2 on 2 ranks, load into FSDP2 on 4, or into an unsharded model on 1, with no conversion step. `get_state_dict` / `set_state_dict` (`torch.distributed.checkpoint.state_dict`) produce and consume the sharded model and optimizer state dicts with fully qualified parameter names, so the optimizer state also follows the parameters to their new owners. Non-tensor values (the scheduler's dict, step counters, generator states) are pickled into the checkpoint.

What does *not* reshard: per-rank state such as RNG streams. A stream that belongs to rank 3 of 4 has no meaning when there are 2 ranks. The fix is to make randomness a function of *global* quantities — a **counter-based** generator seeded from (seed, step, global sample index) — so that the same sample gets the same noise whatever rank processes it. The data stream needs the same property; the course's `GlobalStream` draws the whole global batch from one generator on every rank and lets each rank keep its slice.

**Commit protocol.** A crash during a save leaves a folder of partial shards. Write each checkpoint to its own folder, barrier, then let rank 0 write a `COMMITTED` marker; a restart loads the newest folder that has one. **Asynchronous save** (`dcp.async_save`) copies the state to CPU memory (the blocking part) and writes the files in a background thread; the checkpoint counts as committed only when the write has finished on every rank.

### Case studies

- **MegaScale** (ByteDance, PUBLICLY DOCUMENTED with company claims, sections 4.4 and 6): checkpointing in two phases — each GPU writes its state to host memory "and then continues the training process" (several seconds), and a background process copies it to HDFS. On recovery, one worker per data-parallel group reads the shared shard from HDFS and broadcasts it to the others, so the file system is not hit by every rank. They report over 90% effective training time, more than 100 recoveries during a multi-week run, and failure detection plus diagnostics in under 10 minutes on average.
- **ByteCheckpoint** (company claim): a parallelism-agnostic checkpoint representation with load-time resharding across training frameworks and storage back-ends, reporting an average 54.20× reduction in checkpoint stalls and up to 9.96× faster saving and 8.80× faster loading against the open-source baselines they compare with.
- **Gemini 2.5** (PUBLICLY DOCUMENTED, section 2.3): trained with synchronous data parallelism over multiple 8,960-chip TPUv5p pods in several data centres. *Slice-granularity elasticity*: on a localised failure the job "automatically continues training with fewer 'slices'", losing "tens of seconds" per interruption instead of 10 minutes or more, at about 97% throughput while the failed slice recovers. *Split-phase SDC detection*: a step with suspicious metrics is replayed deterministically and per-device intermediate checksums are compared to locate the faulty chip; about 0.25% of steps were replayed and 6% of those replays turned out to be genuine hardware corruption. Overall 93.4% of the run's time was spent in TPU computation, the rest roughly half in elastic reconfiguration and half in rare cases where elasticity failed. Deterministic replay is only possible when the step is reproducible — the same requirement as the bit-identical resume above.
- **Silent data corruption** also appears in Llama 3's Table 5 (6 of the unexpected interruptions, 1.4%). It is the failure no checkpoint protects against unless something checks the numbers.

### Goodput and the checkpoint interval

Symbols: $M$ the job's mean time between interruptions, $\delta$ the blocking time of one checkpoint, $R$ the restart cost (detect, replace or exclude the node, relaunch, load), $\tau$ the compute time between checkpoints. Failures are a Poisson process. With $N$ devices failing independently with mean time $m$ each, $M = m / N$: the job's failure rate grows with the cluster.

To first order, the fraction of time wasted is

$$w(\tau) \approx \underbrace{\frac{\delta}{\tau}}_{\text{writing}} + \underbrace{\frac{\tau/2 + R}{M}}_{\text{lost work and restart per failure}},$$

minimised at **Young's interval** $\tau^* = \sqrt{2 \delta M}$ (Young, 1974). Daly (2006) refines it; the form most often quoted is $\tau^* \approx \sqrt{2\delta M} - \delta$ for $\delta < M/2$. The exact expectation under the same Poisson model, where each segment of $\tau + \delta$ is retried until it completes and every failure costs a restart that can itself be interrupted, is

$$E[\text{segment}] = M\, e^{R/M}\left(e^{(\tau+\delta)/M} - 1\right), \qquad \text{goodput} = \frac{\tau}{E[\text{segment}]}.$$

`frontierlab.dist.goodput` computes it, finds its optimum numerically, and checks the formula against a Monte Carlo simulation of the same process.

## Worked example

**Llama 3's rate.** $54 \cdot 24 = 1{,}296$ hours, 419 unexpected interruptions: $M = 3.09$ h. If the job used 16,384 GPUs throughout (the largest Table 4 configuration; the report does not state the snapshot's GPU count, so this is an ASSUMPTION), each GPU's mean time between job-stopping failures is $3.09 \cdot 16{,}384 = 50{,}700$ h, about 5.8 years. At 100,000 GPUs the same hardware gives $M = 0.51$ h.

**The interval**, with an *assumed* $\delta = 1$ minute and $R = 10$ minutes: Young's $\tau^* = \sqrt{2 \cdot (1/60) \cdot 3.09} = 0.321$ h $= 19.3$ min. First-order waste: $0.0167/0.321 + (0.161 + 0.167)/3.09 = 0.052 + 0.106 = 0.158$, goodput 84.2%; the exact formula gives 85.2% at Young's interval and 85.3% at its own optimum (18.6 min): the curve is flat near the optimum, so Young is good enough here.

**What it takes to reach Meta's ">90%".** The same model, exact formula at the optimal interval (PROJECTED, assumptions as stated):

| | $\delta$ = 10 s | 30 s | 60 s |
|---|---|---|---|
| $R$ = 2 min | 94.8% | 91.8% | 89.0% |
| $R$ = 5 min | 93.3% | 90.4% | 87.6% |
| $R$ = 10 min | 90.8% | 88.0% | 85.3% |

So above 90% at one interruption per 3.1 hours needs checkpoints that block for tens of seconds at most *and* restarts of a few minutes — consistent with Meta's description of automated recovery, though the report gives neither number (INFERENCE). Elasticity attacks $R$: with $R = 30$ s and $\delta = 60$ s the goodput is 89.7%; the sensitivity table in `goodput_calc.py` shows that at 100,000 GPUs the difference between $R = 10$ min and $R = 30$ s is 55% against 75%.

**Checkpoint size.** A 405B dense model with FP32 master weights and two FP32 AdamW moments is $405 \cdot 10^9 \cdot 12 = 4.9$ TB, plus BF16 weights if stored. Written by all ranks in parallel at Llama 3's Tectonic sustained 2 TB/s (section 3.3.1) that is about 2.4 s of write bandwidth; the blocking time is set by the copy out of HBM and the slowest rank, not by the file system alone (PROJECTED).

## Shapes and cost

| In a checkpoint (FSDP2, $n$ ranks, AdamW, FP32) | Shape per rank | dtype | Written by |
|---|---|---|---|
| a weight `W` of shape $(d_0, d_1)$ | $(\lceil d_0/n \rceil, d_1)$ | fp32 | every rank, its shard |
| `exp_avg`, `exp_avg_sq` of `W` | same | fp32 | every rank |
| optimizer `step` | scalar | fp32 / int | deduplicated |
| scheduler, data generator, step | Python objects | pickled | deduplicated (identical on every rank) |
| per-rank RNG | `torch.ByteTensor` state | uint8 | each rank under its own key |
| `.metadata` | global shapes, offsets, keys | — | rank 0 |

Cost: bytes $\approx 12$–$16$ per parameter; blocking time = copy to host (async) or the whole write (sync); load = read + scatter to the new owners. In the free CPU lab the model is small enough that process start-up dominates the restart.

## Build it

`frontierlab/dist/recovery.py` is a small trainer built for this lesson (the course loop `train/loop.py` is single-process; its exact resume is the single-GPU version of the same idea). `train_worker` trains the course model under FSDP2 or DDP; every `ckpt_every` steps it builds the application state — `get_state_dict(model, optimizer)`, the scheduler, the data stream, the step, and each rank's RNG under its own key — and saves it with `dcp.save` (or `dcp.async_save`), then commits. On start it loads the newest committed checkpoint; if the saved rank count differs from the current one, it reshards the model and optimizer through DCP and, in `stateful` RNG mode, records in the run's log that the per-rank streams could not be carried over. `die_at` makes one rank call `os._exit(17)` after a step (a hard crash; `torch.multiprocessing.spawn` then stops the others), `die_in_save` between writing the shards and the commit. `compare_states` checks every tensor and value of two runs' gathered final states bit for bit.

One composition problem appeared while building it: with `dcp.async_save` on the training process group, the background writer's collectives and the training loop's all-reduces ran on the same `gloo` group from two threads, and the processes died with a native error at the second checkpoint (Windows exit code `0xC0000409`). Giving the asynchronous save its own process group (`dist.new_group(backend="gloo")`, passed as `process_group=`) fixed it. The general rule: anything that runs collectives in the background needs its own communicator.

`labs/common/tests/test_dist.py` checks, on 2–4 CPU processes: a run killed after step 5 and relaunched ends bit-identical to an uninterrupted run (model, optimizer, scheduler, data position, step, RNG) with identical losses; a crash during a save leaves an uncommitted folder that the restart skips; a checkpoint from FSDP2 on 2 ranks loads bit-identically into FSDP2 on 4 ranks and into DDP on 1; with counter-based RNG the resharded run continues with losses equal to the original to $10^{-9}$.

## What the evidence says

- **ESTABLISHED.** Sharded, asynchronous checkpointing and resharding on load (DCP, ByteCheckpoint, MegaScale's two-phase scheme; torchtitan uses DCP). Young-style interval choice from measured $\delta$ and $M$.
- **PUBLICLY DOCUMENTED.** Llama 3's interruption counts and causes (section 3.3.4, Table 5), with the 78% scoped to *unexpected* interruptions; Gemini 2.5's elasticity, SDC replay rate and 93.4% compute time (section 2.3); MegaScale's and ByteCheckpoint's figures are the companies' own measurements of their systems (company claim).
- **PROMISING / MODEL-SPECIFIC.** Slice-granularity elasticity is documented for Gemini on TPU pods with synchronous data parallelism; doing the same on GPU clusters needs a framework that can drop a data-parallel replica group mid-run (torchtitan's fault-tolerance integration with torchft is the open-source attempt; not tested in this course).
- **Measured here (CPU):** exactness of recovery and of resharding, and the CPU-scale costs below. Every cluster-scale number is PROJECTED from the formulas with the stated assumptions.

## Lab

### Experiment contract

- **Question:** (1) Does the trainer restart into the same run after a hard crash and after a crash during a save? (2) How much training does a blocking (`dcp.save`) vs asynchronous (`dcp.async_save`) checkpoint stall, and what checkpoint interval and goodput follow for a 16K-GPU job? Decision informed: the checkpointing scheme and interval of the module project's plan.
- **Hypothesis and status:** (1) bit-identical, an established property of a correct implementation; (2) async blocks for less time than sync, an established mechanism whose size on CPU may be small.
- **Baseline:** the uninterrupted run (1); synchronous save (2).
- **Changed variable:** the crash (1); the save mode (2). **Controlled:** model, seeds, data stream, steps, checkpoint interval, world size, layout.
- **Comparison axis:** equal training steps.
- **Budget:** free CPU about 6 minutes for `kill_and_resume.py`, 7 minutes for `reshard.py`; main path about 30 GPU-minutes on 8 GPUs.
- **Metrics and decision rule:** (1) bitwise equality of every state component and every logged loss — any difference fails; (2) median blocking time per checkpoint for each mode (report all samples; with 4 checkpoints per run no interval is meaningful, so the rule is: prefer async if its slowest checkpoint blocks for less than sync's fastest). Then goodput at the optimal interval for $M = 3.09$ h with your measured $\delta$ scaled to the target as the project says.
- **Correctness checks:** `pytest labs/common/tests/test_dist.py -k "crash or reshard"` passes; part 1 of `kill_and_resume.py` passes before part 2's numbers count.
- **Fallback evidence:** Llama 3, MegaScale and Gemini 2.5's published figures, labelled.
- **Limits:** local disk, not a parallel file system; one small model; process start-up on Windows dominates the CPU restart cost; failures are simulated with `os._exit`, not real hardware faults (no hung NCCL collectives, no corrupted memory).

**Folder:** [`labs/module-09/lesson-04/`](../../labs/module-09/) · **Time:** about 90 minutes · **Pass check:** `pytest labs/module-09/lesson-04` passes; `kill_and_resume.py` part 1 prints "yes" for every component; your goodput table uses your measured numbers.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1 node, 8 GPUs, about 30 GPU-minutes | `kill_and_resume.py --device cuda --world 8 --hidden 2048 --layers 16` and `reshard.py --device cuda`; then kill a torchtitan run (lesson 09.3's `m09_fsdp8` with `checkpoint.enable = True`, `checkpoint.interval = 20`) with `kill -9` on one rank's process after step 30, relaunch the same command, and compare its loss log with an uninterrupted run. Not run in this build; part of the Module 9 pilot (the single-GPU part of the recovery check is piloted on Colab per plan section 12.1) |
| Free GPU (Colab or Kaggle) | 1–2 GPUs | `kill_and_resume.py --device cuda --world 1` (Colab) or `--world 2` (Kaggle 2× T4); untested |
| Free CPU | laptop, 2–4 `gloo` processes; measured about 6 minutes for `kill_and_resume.py` and 7 minutes for `reshard.py` | the steps below |

1. **Implement** `young`, `goodput`, `latest_committed` and `lost_work` in `lab.py`; `pytest labs/module-09/lesson-04`.
2. **Kill and resume.**

   ```bash
   python labs/module-09/lesson-04/kill_and_resume.py
   ```

   Part 1 must print "yes" for model, optim, scheduler, data, step and rng, for both crash kinds. Then break it on purpose: in a copy of `frontierlab/dist/recovery.py`, drop `"data"` from the saved `extra` dict, rerun, and see which components and which losses now differ.
3. **Reshard.**

   ```bash
   python labs/module-09/lesson-04/reshard.py
   ```

   Explain why the loaded states are bit-identical in every layout, why training on from them is not, and why the stateful-RNG run says it was re-seeded.
4. **Goodput.** Run `python labs/module-09/lesson-04/goodput_calc.py`, then again with `--delta-s` and `--restart-min` set to your measurements scaled to a 16K-GPU job (write down how you scaled them), and with `--gpus 100000`.

<details>
<summary>Hint for TODO 4</summary>

Walk the records once, remembering the last `step` seen since the most recent `start`. At each `start` after the first, the lost work is that last step minus the new `from_step` (or 0 if no step was logged in between); then reset.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04 on the build laptop (Windows 11, 16 threads, torch 2.14.1+cpu, `gloo`, local SSD, other jobs running). `kill_and_resume.py` took 6 min 7 s, `reshard.py` 6 min 40 s.

**Exactness** (2 ranks, FSDP2, float64, 12 steps, checkpoints at 4, 8, 12):

| Run | First launch | Left behind | Resumed from | Bit-identical to the uninterrupted run | Steps recomputed |
|---|---|---|---|---|---|
| rank 1 killed after step 7 | exit code 17 | nothing uncommitted | `step_000004` | model, optimizer, scheduler, data, step, RNG: yes; all 12 losses equal | 3 |
| rank 1 killed during the step-8 save | exit code 17 | `step_000008` without `COMMITTED` | `step_000004` | all yes; all 12 losses equal | 3 |

**Cost** (float32, width 512, 4 layers, 2 ranks, 12 steps, a checkpoint every 3):

| | per checkpoint (4 each) | median step |
|---|---|---|
| checkpoint size | 218.8 MB (weights + AdamW, FP32) | — |
| `dcp.save` blocking | 784, 616, 699, 602 ms (median 657; 0.33 GB/s) | 3,082 ms |
| `dcp.async_save` blocking | 183, 152, 180, 197 ms (median 181) | 2,856 ms |
| restart | process start 10.6 s + load 1.79 s | — |

By the rule, async's slowest checkpoint (197 ms) blocks less than sync's fastest (602 ms): prefer async. Process start dominates the CPU restart: on Windows each of the spawned processes imports torch; on a cluster, detection and node replacement dominate instead, so 12 s is only a lower bound for $R$. With these CPU numbers as if they held at 16,384 GPUs ($\delta = 0.66$ s, $R = 0.2$ min) the optimal interval would be 2 minutes and the goodput 98.8% — which shows what the formula does with small costs, not what a cluster would see: the checkpoint there is about 25,000 times larger and the restart includes finding the failure.

**Resharding** (`reshard.py`; FSDP2 on 2 ranks → checkpoint at step 3 → three layouts):

| Loaded into | State after load vs the 2-rank load | Steps 4–6, max loss difference vs the original: stateful RNG | counter RNG |
|---|---|---|---|
| FSDP2, 2 ranks | bit-identical | 0 | 0 |
| FSDP2, 4 ranks | bit-identical (model, optimizer, scheduler, data, step) | 0.045 (re-seeded, the log says so) | 0 |
| DDP, 1 rank | bit-identical | 0.043 (re-seeded) | $8.9 \times 10^{-16}$ |

What a write-up should say. Recovery is exact when every component is saved and the restart loads the newest *committed* checkpoint; the crash during the save proves the commit marker matters. Resharding through DCP moves every tensor exactly. Whether the continued run stays the same run is decided by how randomness was designed: per-rank streams diverge by 0.04 nats within three steps; counter-based noise keeps the losses equal to the last bit at 2 → 4 ranks (each rank's reduction happened to give the same sums here) and to $10^{-15}$ at 2 → 1. A change after results: the first build of `recovery.py` crashed with `async_save`; see "Build it" for the separate process group that fixed it.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-09/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-09/lesson-04`.

</details>

## Common mistakes

- **Saving model and optimizer only.** The scheduler restarts its warm-up, the data stream restarts from its seed, and the run is a different run that looks continuous.
- **Per-rank data samplers whose state depends on the rank count.** The checkpoint then cannot be loaded into another layout without repeating or skipping data.
- **Loading the newest folder instead of the newest *committed* one.** A crash during a save leaves partial shards that load and then fail, or worse, load with missing pieces.
- **Counting an async checkpoint as done when `async_save` returns.** It returns after the copy to host memory; the files are complete only when the future resolves on every rank.
- **Using a fixed checkpoint interval when the cluster grows.** $M$ falls as $1/N$; Young's interval falls as $1/\sqrt{N}$.
- **Quoting "78% of interruptions were hardware".** It is about 78% of the 419 *unexpected* interruptions, not of all 466.
- **Expecting bit-identical training after resharding.** The loaded state is identical; the next gradient reduction sums in a different order.

## References

- Llama Team, *The Llama 3 Herd of Models*, sections 3.3.1 and 3.3.4, Table 5. https://arxiv.org/abs/2407.21783
- Gemini Team, *Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities*, section 2.3. https://arxiv.org/abs/2507.06261
- Z. Jiang et al., *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs*, sections 4.4, 6.2, 6.3. https://arxiv.org/abs/2402.15627
- B. Wan et al., *ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development*, abstract. https://arxiv.org/abs/2407.20143
- J. W. Young, *A first order approximation to the optimum checkpoint interval*, Communications of the ACM 17(9), 1974. https://doi.org/10.1145/361147.361115
- J. T. Daly, *A higher order estimate of the optimum checkpoint interval for restart dumps*, Future Generation Computer Systems 22(3), 2006. https://doi.org/10.1016/j.future.2004.11.016
- PyTorch 2.14, *torch.distributed.checkpoint*. https://pytorch.org/docs/stable/distributed.checkpoint.html
- Shared code: `labs/common/frontierlab/dist/recovery.py`, `goodput.py`.

## Next

[09.5 · TPUs, JAX and hardware co-design](lesson-05.md) (extension), or the [Module 9 project](../../projects/module-09-infrastructure-plan.md).
