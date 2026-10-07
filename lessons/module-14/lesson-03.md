---
id: "14.3"
module: 14
minutes: 45
practice_minutes: 110
prerequisites: ["14.1", "12.3"]
objectives:
  - Draw the timeline of synchronous, one-step off-policy and bounded-staleness RL, compute speed-up and lag from generation and update times, and say when a larger staleness bound buys nothing.
  - Choose the behaviour log-probabilities for a stale batch and explain what goes wrong when the current policy's are used instead.
  - Measure the trainer/sampler log-probability mismatch per token and per sequence, and explain why batch-dependent kernels make nominally on-policy RL off-policy.
  - Implement a batch-invariant matrix multiply and show that a row's result no longer depends on its batch.
  - Configure a vLLM 0.30.0 rollout server colocated with the trainer (weight transfer, pause and resume, batch-invariant mode) and state what each setting costs.
volatility: implementation
sources:
  - title: "Fu et al., AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning (sections 5.1-5.2, Eq. 5; Table 1)"
    url: https://arxiv.org/abs/2505.24298
  - title: "Piché et al., PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation (section 4: in-flight weight updates)"
    url: https://arxiv.org/abs/2509.19128
  - title: "Noukhovitch et al., Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models (sections 3.3, 5)"
    url: https://arxiv.org/abs/2410.18252
  - title: "He and Thinking Machines Lab, Defeating Nondeterminism in LLM Inference (2025-09-10)"
    url: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  - title: "Yao et al., Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (blog, 2025)"
    url: https://fengyao.notion.site/off-policy-rl
  - title: "Khatri et al., The Art of Scaling Reinforcement Learning Compute for LLMs (section 3.1: PipelineRL-8; section 3.2: FP32 logits, Figure 5b)"
    url: https://arxiv.org/abs/2510.13786
  - title: "MiniMax, MiniMax-M1 (section 3.2: training/inference probability mismatch and the FP32 LM head)"
    url: https://arxiv.org/abs/2506.13585
  - title: "Zheng et al., Group Sequence Policy Optimization (section 5.4: benefit for RL infrastructure)"
    url: https://arxiv.org/abs/2507.18071
  - title: "vLLM v0.30.0 (examples/rl/rlhf_http_ipc.py; docs/features/batch_invariance.md; docs/features/sleep_mode.md)"
    url: https://github.com/vllm-project/vllm/tree/v0.30.0
  - title: "Sheng et al., HybridFlow: A Flexible and Efficient RLHF Framework (verl)"
    url: https://arxiv.org/abs/2409.19256
last_verified: "2026-10-07"
---

# 14.3 · Rollout systems and staleness

In reasoning RL most of the wall-clock goes into generating rollouts, not into gradient steps, so production systems put generation on a separate inference engine and let it run ahead of the trainer. That buys throughput and costs two things this lesson measures: rollouts come from a policy that is several updates old (staleness), and even fresh rollouts come from a different program with other kernels and precision (train/inference mismatch). You model the schedule, run bounded-staleness RL with each objective from lesson 14.1, measure the mismatch, and build the simplest batch-invariant kernel.

## Why this matters at a frontier lab

A step of RL on long reasoning traces is mostly decoding: one batch waits for its longest response while the trainer sits idle, then generation waits for the trainer. Every large open recipe separates the two and overlaps them. ScaleRL's recipe uses PipelineRL with up to 8 steps of off-policyness (section 3.1). AReaL bounds staleness at $\eta = 4$ for code and 8 for math (section 7.1) and reports up to 2.77× over synchronous training (abstract). verl v0.9.1 ships one-step off-policy and fully asynchronous trainers. The speed-up is real, but each design choice also changes the objective the run optimises. Which log-probabilities go into the ratio, how stale a batch may be, which kernels the sampler uses, in what precision the logits are computed: when one of these is wrong, the run keeps training and the reward still climbs, only more slowly or towards a worse place. Thinking Machines reports that, against a bitwise-identical sampler and trainer, RL without importance correction collapsed in their experiment (Defeating Nondeterminism, "true on-policy RL"). This is infrastructure that changes results, so it is research engineering, not operations.

## The idea

### Trainer and rollout engine as two programs

The trainer holds float32 master weights, optimizer state and activations, and runs teacher-forced forward and backward passes. The rollout engine (vLLM, SGLang) holds a bfloat16 copy of the weights and a paged KV cache, and runs continuous-batching decode with its own fused kernels. After every update the trainer's new weights have to reach the engine, either by copying GPU memory to GPU memory when both share the GPU (colocated; vLLM's CUDA-IPC weight transfer) or over NCCL to separate rollout GPUs (disaggregated). In vLLM 0.30.0 the trainer side is `WeightTransferTrainerFactory.trainer_init(...)` followed by `send_weights()` between `POST /pause` and `POST /resume` on the server (`examples/rl/rlhf_http_ipc.py` at the tag). verl's HybridFlow design colocates the actor, rollout and reference models and switches GPU memory between them.

### Synchronous, one-step off-policy, bounded staleness

Number the updates $j = 0, 1, \dots$ and let batch $j$ feed update $j$. Batch $j$ is generated with weights version $v_j$, the number of updates finished when its generation started. Its **lag** (staleness) is $j - v_j$. A staleness bound $k$ forbids generating batch $j$ before version $j - k$ exists. With $g_j$ the generation time of batch $j$ and $T$ the update time:

$$s_j = \max\big(e_{j-1},\ u_{j-k-1}\big), \qquad e_j = s_j + g_j, \qquad u_j = \max(e_j, u_{j-1}) + T,$$

where $s_j$ is when generation of batch $j$ starts, $e_j$ when it ends, $u_j$ when update $j$ ends (with $u_{j-k-1} = 0$ when $j - k \le 0$).

- $k = 0$ is **synchronous on-policy** RL: generate, update, generate with the new weights. Each step costs $g_j + T$.
- $k = 1$ is **one-step off-policy**: batch $j+1$ is generated with weights $v_j$ while update $j$ runs. Each step costs about $\max(g_j, T)$, and every batch is exactly one update old.
- $k > 1$ lets the engine run further ahead. That helps only when generation times vary so much that a buffer of finished batches keeps the trainer busy through a slow one. AReaL's rule is the same bound written for a stream of trajectories: with $N_r$ trajectories generated so far, batch size $B$ and current version $i$, generation may continue only while $\lfloor (N_r - 1)/B \rfloor \le i + \eta$ (section 5.1).

Why generation time varies so much: a batch decoded together finishes when its **longest** response does. With heavy-tailed response lengths, nearly every batch of 256 contains one long response. PipelineRL attacks this with **in-flight weight updates**. The engine pauses, loads new weights and continues the unfinished sequences without recomputing their KV cache (section 4), so one response can mix tokens from several policy versions. ScaleRL compares PipelineRL against plain off-policy PPO and reports a similar asymptote with better efficiency (its Figure 4a).

### Which log-probabilities go into the ratio

For a stale batch the importance ratio must compare the current policy with the policy that **generated** the batch, $\pi_{\text{behav}}$. The lesson 12.3 loop does this with the snapshot that sampled it. Recomputing $\pi_{\text{old}}$ with the current weights makes every ratio 1, so clipping never fires and the update treats data from an older policy as if it were its own: the gradient is computed for a policy that did not produce the data. AReaL makes the two roles explicit (decoupled PPO, Eq. 5):

$$J(\theta) = \mathbb{E}_{a_t \sim \pi_{\text{behav}}}\Big[\sum_t \frac{\pi_{\text{prox}}(a_t \mid s_t)}{\pi_{\text{behav}}(a_t \mid s_t)} \min\big(u^{\text{prox}}_t A_t,\ \mathrm{clip}(u^{\text{prox}}_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big], \qquad u^{\text{prox}}_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{prox}}(a_t \mid s_t)}.$$

Here $\pi_{\text{behav}}$ is the stale policy that sampled the data, $\pi_{\text{prox}}$ a recent policy that anchors the trust region, and the first factor an importance weight that corrects for the gap between them. A system that does not keep old weights on the trainer can use the **engine's own** log-probabilities as $\pi_{\text{behav}}$, which is what the course's main-path loop does for stale batches. That folds the train/inference mismatch into the correction.

Lesson 14.1's objectives react to staleness differently, because it is exactly the regime where their gradients differ. GRPO and DAPO silence tokens whose ratio left the trust region. GSPO's length-normalised sequence ratio moves less than any of its tokens. CISPO keeps every token at a capped weight.

### Train/inference mismatch

Even at lag 0 the engine's probabilities are not the trainer's. The engine uses other kernels, bfloat16 weights, and attention and matmul kernels whose reduction order depends on the batch they run in. He and Thinking Machines Lab sampled 1,000 completions of one prompt at temperature 0 from Qwen3-235B-A22B-Instruct-2507 and got 80 distinct outputs. The main cause was that server load changes the batch size and the kernels are not **batch-invariant**: the same row computed in a different batch gives a slightly different result. With batch-invariant RMSNorm, matmul and attention all 1,000 were identical, at a cost: 26 s for the default vLLM run, 42 s for the deterministic one with their improved attention kernel. vLLM 0.30.0 has this as a beta mode, `VLLM_BATCH_INVARIANT=1` (`docs/features/batch_invariance.md`).

Two other remedies attack the size of the gap rather than its variability. MiniMax-M1 found the language-model head's precision to be the main source of the gap, and computing it in FP32 raised the correlation between training and inference probabilities from about 0.9x to 0.99x (section 3.2). ScaleRL adopted FP32 logits in both generator and trainer, which raised the fitted asymptote from 0.52 to 0.61 in its ablation (Figure 5b). Truncated importance sampling (Yao et al.; lesson 12.3) corrects what remains with the weight $\min(\pi_{\text{trainer}}/\pi_{\text{sampler}}, C)$. GSPO notes that sequence-level likelihoods tolerate the precision gap better than token-level ones, because per-token errors average out in $s_i$ (section 5.4).

### A batch-invariant kernel in one line of reasoning

A matmul's result for one row depends on how the kernel splits the reduction over $K$ and the rows over tiles. Libraries choose the split by problem shape, so a row computed alone (an $M = 1$ matrix-vector product) goes through a different code path than the same row inside $M = 64$. The simplest invariant scheme always runs the same tile shape: pad every block to a fixed number of rows, run the same kernel, drop the padding. Every row then sees the same reduction order whatever the batch, and pays for the padding. Production batch-invariant kernels get the same property without the waste, by fixing the reduction strategy rather than the shape.

## Worked example

**Schedule.** Three batches, each taking $g = 2$ s to generate, and updates of $T = 1$ s.

- $k = 0$: generate 0–2, update 2–3; generate 3–5, update 5–6; generate 6–8, update 8–9. Total 9 s, every lag 0.
- $k = 1$: batch 0 is generated 0–2 and updated 2–3. Batch 1 may start once version $1 - 1 = 0$ exists (from the start), so it is generated 2–4 with version 0 and has lag 1, updated 4–5. Batch 2 needs version 1, which exists at 3 s; the engine is free at 4 s, so it is generated 4–6 with version 1 (lag 1), updated 6–7. Total 7 s, a speed-up of $9/7 = 1.29$. Generation was busy $6/7$ of the time and the trainer $3/7$: generation is the bottleneck, and no larger $k$ can help.

**AReaL's bound.** $B = 512$ trajectories per update, $\eta = 8$, current version $i = 10$. Trajectory number $N_r$ belongs to batch $\lfloor (N_r - 1)/512 \rfloor$, and generation may continue while that index is at most $i + \eta = 18$: up to $N_r = 19 \cdot 512 = 9{,}728$ trajectories, batches 0–18. Batch 18 is generated with version 10 and used by update 18, a lag of 8, the bound.

**A stale batch treated as fresh.** A token had probability 0.20 under the behaviour policy and 0.30 under the current one, with $A = +1$. Corrected: $r = 1.5 > 1.2$, so GRPO clips it (no further push), and CISPO weights it $\min(1.5, 1.28) = 1.28$. Recomputed with the current policy: $r = 0.30/0.30 = 1$, unclipped, so the token is pushed up again with full weight although it already rose 50% since the data was generated. Repeat that for 8 stale updates and the trust region is gone.

**Mismatch, token vs sequence.** One response of 8 tokens with trainer-minus-sampler log-probability differences $(+0.20, -0.05, +0.01, -0.12, +0.02, -0.03, +0.04, -0.01)$. The largest token ratio is $e^{0.20} = 1.22$, so under GRPO's $\epsilon = 0.2$ that token would be clipped by mismatch alone. The sum is $+0.06$ and the length-normalised mean is $0.0075$, so the sequence ratio is $e^{0.0075} = 1.0075$. The errors of opposite sign cancel in $s_i$, which is GSPO's section 5.4 point, though $1.0075$ is still outside GSPO's $4 \times 10^{-4}$ range.

**Weight sync volume.** Qwen3-1.7B-Base has 1,720,574,976 parameters, so a bfloat16 copy is $1.72 \times 10^9 \cdot 2 = 3.44$ GB per update. Over CUDA IPC on the same GPU this is a memory copy. Over a 25 GB/s link to a separate rollout node it takes about 0.14 s, small next to a 30–50 s step.

## Shapes and cost

| Object | Shape, dtype | Where |
|---|---|---|
| trainer weights + AdamW state (1.7B) | $1.72 \times 10^9 \times (4 + 4 + 4 + 4)$ B = 27.5 GB float32 | trainer GPU |
| engine weights | $1.72 \times 10^9 \times 2$ B = 3.44 GB bfloat16 | engine GPU (same GPU when colocated) |
| engine KV cache | gpu_memory_utilization × 80 GB minus weights; Qwen3-1.7B: 28 layers × 8 KV heads × 128 × 2 (K, V) × 2 B = 114,688 B per token | engine GPU |
| sampler log-probs | (B, R) float32, returned per sampled token (`logprobs=0`) | engine → trainer |
| behaviour snapshots (toy loop) | bound + 1 state dicts, 1.2 MB each | CPU |
| lag schedule | (steps,) int | — |

With `--gpu-memory-utilization 0.35` the engine takes 28 GB of an 80 GB H100: 3.4 GB of weights and about 24 GB of KV cache, about 210,000 tokens at 114,688 B per token. That fits 256 responses of 800 tokens with room to spare, while the trainer keeps its 27.5 GB plus activations. These are PROJECTED from the formulas, pending the Module 14 pilot.

Time is the reason for all of this. With the lab's illustrative constants (0.03 s per batched decode step, 25 s per update, 256 responses per batch with log-normal lengths around 300 tokens, capped at 2,048), 88% of batches contain at least one response that hits the cap, so generation averages 61 s per batch, about 2.4 times the update. The pilot measures the real constants.

## Build it

```python
from frontierlab.rlscale import asyncsim, mismatch, runner

r = asyncsim.simulate([2, 2, 2], train_time=1.0, bound=1)       # total 7 s, lags [0, 1, 1]
runner.train_objective(cfg, "cispo", lags=r["lags"])             # course loop, bounded-staleness sampler
mismatch.mismatch_stats(trainer_logp, sampler_logp, mask)        # mean/max |d|, k3, max ratio, sequence sums
```

- `frontierlab/rlscale/asyncsim.py`: the schedule above, plus heavy-tailed batch generation times.
- `runner.BoundedStalenessSampler`: replaces the loop's fixed-lag sampler (lesson 12.3's `--staleness k`) with one whose lag follows a schedule, never above the bound. The behaviour log-probabilities come from the same snapshot, so the ratio stays honest.
- `frontierlab/rlscale/mismatch.py`: mismatch statistics, `fixed_tile_matmul` and `batch_dependence`.
- `frontierlab/rlscale/vllm_rollout.py` + `hf_rl.py --rollout vllm --staleness 0|1 [--batch-invariant]`: the main path. A colocated `vllm serve` with `--weight-transfer-config '{"backend": "ipc"}'`, sampled-token log-probabilities returned with token ids, and weight sync after every update. With staleness 1, batch $j+1$ is generated on a worker thread with $v_j$ while update $j$ runs; the loop waits for it before syncing, so its lag is exactly 1. A stale batch uses the engine's log-probabilities as $\pi_{\text{behav}}$. A fresh batch is recomputed by the trainer, and the mismatch is logged every step (`mismatch_mean_abs`, `mismatch_k3`, `mismatch_max_ratio`). Not run in this build.

Correctness checks (`test_rlscale.py`): the schedule against the hand timeline; no lag above the bound for $k \in \{0, 1, 2, 4, 8\}$ on random times; speed-up never decreasing in $k$; the bounded sampler uses exactly $\min(\text{want}, k, \text{snapshots} - 1)$; `fixed_tile_matmul` agrees with `@` and gives bitwise-identical rows for every batch size.

**Framework mapping.** verl v0.9.1: `verl/experimental/one_step_off_policy` ($k = 1$) and `verl/experimental/fully_async_policy` (bounded staleness), documented in `docs/advance/one_step_off.md` and `fully_async.md`. vLLM 0.30.0: `VLLM_BATCH_INVARIANT=1`, `LLM(enable_sleep_mode=True)` with `sleep(level=1|2)` / `wake_up()` for colocated memory sharing, and the weight-transfer API above.

## What the evidence says

- **Separating rollout from training and overlapping them: ESTABLISHED** (verl, OpenRLHF, AReaL, PipelineRL, slime and the ScaleRL recipe all do it; PUBLICLY DOCUMENTED).
- **Bounded staleness with importance correction keeps learning: PROMISING.** AReaL reports 2.57× (1.5B, 14.8 h vs 33.6 h) and 2.24× (14B) over synchronous training at matched final quality (Table 1). Asynchronous RLHF reports 40% faster training for LLaMA 3.1 8B with online DPO "most robust to off-policyness" (sections 3.3 and 5). How much staleness a run tolerates depends on the objective, learning rate and clip range (INFERENCE).
- **In-flight weight updates (PipelineRL): PROMISING.** About 2× faster learning on 128 H100s for Qwen2.5-7B math RL (section 5; one group's runs).
- **Train/inference mismatch is real and matters: PUBLICLY DOCUMENTED** (MiniMax-M1 section 3.2; Yao et al.; Thinking Machines). **FP32 LM head: PROMISING** (MiniMax-M1; ScaleRL Figure 5b, A from 0.52 to 0.61). **Batch-invariant inference: PROMISING**, now a beta vLLM feature with a measured 1.6× slowdown in the blog's example (26 s vs 42 s).
- **Course measurement (free CPU, 2026-10-07, 2 seeds):** corrected stale arms at bound 8 stayed within noise of on-policy training (all slightly below it). Ignoring staleness collapsed one seed to 0.03 while its ratio and clip metrics looked healthy. A bf16 sampler alone pushed a token ratio to 1.23. In the schedule model, $k = 1$ gave all the speed-up. None of this says how a 1.7B model on vLLM behaves. That is the pilot's question.

## Lab

**Folder:** [`labs/module-14/lesson-03/`](../../labs/module-14/) · **Time:** about 110 minutes (about 25 minutes unattended) · **Pass check:** `pytest labs/module-14/lesson-03` passes; `async_lab.py` prints the three parts; your write-up answers the contract's question with the intervals and names the metric that exposed the uncorrected arm.

### Experiment contract

- **Question:** with rollouts up to 8 updates stale (a bounded-staleness schedule), does training with the behaviour policy's log-probabilities match on-policy training, does that depend on the objective (GRPO, CISPO, GSPO), and what happens when staleness is ignored? Decision informed: the staleness bound and the objective for the asynchronous main-path loop.
- **Hypothesis:** corrected GRPO, CISPO and GSPO at bound 8 stay within noise of on-policy GRPO over 100 steps. Their clip fractions rise (GSPO's most, CISPO counting capped weights), and the uncorrected arm falls behind with healthy-looking ratios. Status: reported (AReaL, ScaleRL's PipelineRL-8, lesson 12.3's fixed-lag result); objective dependence at this scale unknown.
- **Baseline:** `grpo-sync` (on-policy, lr $3 \times 10^{-4}$, untuned).
- **Changed variable:** staleness schedule and objective, as named per arm. **Controlled:** SFT start, 16 prompts × 8 responses, 100 steps, 1 epoch × 2 minibatches, temperature 1, KL off, seeds 0–1, the lag schedule (identical for all stale arms), the 300 evaluation problems and sampling seed.
- **Control arm:** lesson 14.2's random-reward arm (same SFT start, task and evaluation; GRPO, 120 steps, 3 seeds) is the reference for what a reward without information does. An arm that ends near it has learned nothing from the verifier.
- **Comparison axis:** equal updates and equal samples. Wall-clock is what asynchrony buys and this CPU run cannot show it. The schedule part models it.
- **Budget:** free CPU, measured runtime in the results box.
- **Metrics and decision rule:** held-out sampled accuracy at step 100, paired by seed against `grpo-sync`, 95% t-interval; "staleness hurt" only if the interval lies below 0. Secondary: `clip_frac`, `ratio_max`, minimum entropy, KL to the start, drawdown.
- **Correctness checks:** your TODO tests; `test_rlscale.py` (the bounded sampler never exceeds the bound); the lag histogram printed by the script.
- **Fallback evidence:** lesson 12.3's fixed-lag arms; AReaL's Table 1, labelled as published.
- **Limits:** 2 seeds (t-multiplier 12.7), 100 steps, a 308k-parameter policy with 2–4-token responses, emulated asynchrony (the lags are imposed, not produced by real concurrency).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB, vLLM 0.30.0 colocated. Not run in this build; part of the Module 14 pilot | `python labs/module-14/lesson-03/async_lab.py --variant main --print` prints 5 runs of 100 steps: HF rollouts; vLLM synchronous; vLLM synchronous with `--batch-invariant`; vLLM one-step off-policy with GRPO and with CISPO. Record s/step, generation share, `mismatch_*` and `max_mem_gb`. **PROJECTED:** 5 × 100 × 20–50 s = 2.8–7 GPU-hours, USD 6–21 |
| Free GPU (Colab/Kaggle T4) | T4 | vLLM plus training does not fit next to each other on 16 GB at useful sizes; run the HF arm on Qwen3-0.6B-Base (`--variant t4 --print`) and the CPU parts |
| Free CPU | laptop; measured 19 minutes in all (other jobs running) | the steps below |

### Steps

1. **Implement** the three TODOs in `lab.py` (`simulate`, `mismatch_stats`, `fixed_tile_matmul`) and run `pytest labs/module-14/lesson-03`.
2. **Schedule:** `python labs/module-14/lesson-03/async_lab.py --part schedule`. For each balance, which side is the bottleneck, and why does going from $k = 1$ to $k = 8$ change the lag but not the speed-up in this model? What real-system feature would make larger $k$ pay off?
3. **Mismatch:** `--part mismatch`. Report the default kernel's batch dependence and your kernel's. For the bf16 sampler, compare the largest token ratio with GRPO's clip range and the largest length-normalised sequence ratio with GSPO's.
4. **Staleness:** `--part staleness` (about 25 minutes). Apply the decision rule. Which log metric would have told you, without held-out evaluation, that `grpo-k8-recompute` was ignoring staleness?
5. **Write up** the contract's answer and one paragraph on what you would measure first in the main-path pilot.

<details>
<summary>Hint for TODO 1</summary>

Keep an array `upd_end` of update end times. The version for batch $j$ is `(upd_end[:j] <= s_j).sum()`: updates that ended at or before generation started. With `bound = 0` your schedule must reduce to generate-then-train.

</details>

<details>
<summary>Hint for TODO 3</summary>

Loop over `range(0, M, tile)`, concatenate zero rows when the block is short, multiply, and slice back with `[: tile - pad]`. Check with `torch.equal`, not `allclose`: invariance means bitwise equality.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads; Module 13's jobs and a second Module 14 script shared the CPU). `--part schedule` and `--part mismatch` take under a minute; `--part staleness` took 18 minutes for its 10 runs.

**Schedule model** (illustrative constants, so the speed-ups are properties of the model, not measurements of a system):

| Balance | mean generation per batch | speed-up at k = 1 | at k = 8 | mean lag at k = 8 |
|---|---|---|---|---|
| generation-bound (one engine) | 60.9 s | 1.40 | 1.40 | 0.99 |
| balanced (2.5× inference capacity) | 24.3 s | 1.95 | 1.95 | 2.81 |
| trainer-bound (4× inference capacity) | 15.2 s | 1.60 | 1.60 | 7.34 |

One-step off-policy captured all of the overlap in every balance. Beyond $k = 1$ the bound changed only the lag, by up to 7 extra updates when the trainer was the bottleneck, and bought nothing. With one engine that decodes whole batches, each batch has to finish before the next starts, so a buffer of finished batches never forms. Larger bounds pay off in systems this model leaves out: many engines finishing at different times, or partial rollouts that stream finished responses. 0.8% of responses hit the 2,048-token cap, yet 88% of batches contained one.

**Mismatch.** Row 0 of a float32 matmul computed alone or in a batch of 3 differed from the same row in a batch of 256 by $4 \times 10^{-5}$ (K = 512) and $7 \times 10^{-5}$ (K = 2,048), and by exactly 0 in batches of 16 and 64: the CPU library takes a different path for small $M$. The fixed-tile matmul gave bitwise-identical rows for every batch size. Toy policy, 1,024 responses: with a float32 sampler the trainer/sampler gap is rounding only (mean $|d| = 3.4 \times 10^{-7}$, cached decoding vs one forward pass). With a bfloat16 sampler, mean $|d| = 8.8 \times 10^{-3}$, max $|d| = 0.21$, so the largest token ratio is 1.23, outside GRPO's 1.2 bound from precision alone. The largest length-normalised sequence log-ratio is 0.072, about a third of the worst token's.

**Staleness** (bound 8; the imposed schedule's lags averaged 7.5, 166 of 200 at the bound). Held-out sampled accuracy at step 100 (the SFT start is 0.21):

| Arm | seed 0 | seed 1 | vs `grpo-sync` (95% t) | clip fraction | max ratio |
|---|---|---|---|---|---|
| grpo-sync | 0.363 | 0.327 | — | 0.001 | 1.6 |
| grpo-k8 (behaviour log-probs) | 0.330 | 0.273 | −0.043 [−0.170, +0.084] | 0.083 | 54 |
| cispo-k8 | 0.260 | 0.313 | −0.058 [−0.630, +0.513] | 0.070 (capped) | 94 |
| gspo-k8 | 0.300 | 0.220 | −0.085 [−0.360, +0.190] | 0.461 | 2.5 |
| grpo-k8-recompute (staleness ignored) | 0.223 | **0.030** | −0.218 [−1.214, +0.777] | 0.019 | 1.9 |

No interval excludes 0 with two seeds ($t_{0.975,1} = 12.7$), so under the rule nothing is shown to hurt. Every corrected stale arm was below on-policy in both seeds (by 0.01–0.11), consistent with a real but small cost of 7–8 updates of lag at this learning rate. Every arm except one seed of the uncorrected arm stayed far above lesson 14.2's random-reward control (0.07). Ignoring staleness was the only arm that broke: seed 1 collapsed to 0.03, far below the SFT start, and seed 0 ended where it began. Its logs looked healthier than the corrected arms': a clip fraction of 0.019 and a maximum ratio of 1.9, against 0.083 and 54 for corrected GRPO. Of the training metrics, the KL to the start (0.27, the highest) and the drawdown (0.05) pointed at it. The objectives did not separate. GSPO again clipped almost half of all tokens, and CISPO's raw ratios reached 94 while its weights stayed capped at 1.28.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-14/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-14/lesson-03`.

</details>

## Common mistakes

- **Recomputing $\pi_{\text{old}}$ for a stale batch with the current weights.** Every ratio becomes 1 and the logs look healthier than ever. Store the behaviour log-probabilities with the batch.
- **Raising the staleness bound to fix a generation bottleneck.** If the engine is busy 100% of the time, more lag buys nothing. Add inference capacity, shorten the tail (interruption, length control) or use partial rollouts.
- **Trusting the engine's log-probabilities without checking what they are.** Raw or processed (temperature, top-k)? From bf16 logits or FP32? Check one batch against the trainer's recomputation.
- **Expecting determinism from a fixed seed in a serving engine.** Without batch-invariant kernels the same request gives different results under different load.
- **Syncing weights while requests are in flight without meaning to.** Then a response mixes policies, which PipelineRL does on purpose. Pause, sync, resume, unless your correction accounts for it.
- **Colocating without a memory budget.** The engine's `gpu_memory_utilization` is a fraction of the whole GPU, and the trainer's 27.5 GB of state does not shrink.

## References

- W. Fu et al., *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning*, 2025, sections 5.1–5.2, 7.1, Table 1. https://arxiv.org/abs/2505.24298
- A. Piché et al., *PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation*, 2025, sections 4–5. https://arxiv.org/abs/2509.19128
- M. Noukhovitch et al., *Asynchronous RLHF*, 2024, sections 3.3 and 5. https://arxiv.org/abs/2410.18252
- H. He and Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- F. Yao et al., *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*, 2025. https://fengyao.notion.site/off-policy-rl
- D. Khatri et al., *The Art of Scaling Reinforcement Learning Compute for LLMs*, 2025, sections 3.1–3.2. https://arxiv.org/abs/2510.13786
- MiniMax, *MiniMax-M1*, 2025, section 3.2. https://arxiv.org/abs/2506.13585
- C. Zheng et al., *Group Sequence Policy Optimization*, 2025, section 5.4. https://arxiv.org/abs/2507.18071
- G. Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework*, 2024. https://arxiv.org/abs/2409.19256
- vLLM v0.30.0: `examples/rl/rlhf_http_ipc.py`, `docs/features/batch_invariance.md`, `docs/features/sleep_mode.md`. https://github.com/vllm-project/vllm/tree/v0.30.0
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[14.4 · RL scaling and the capability debate](lesson-04.md)
