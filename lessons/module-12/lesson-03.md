---
id: "12.3"
module: 12
minutes: 45
practice_minutes: 120
prerequisites: ["12.2"]
objectives:
  - Compute the weight each token gets under per-sequence, token-level, constant and prompt-level loss aggregation, and explain the length bias each one creates.
  - Choose how to treat truncated responses (keep, filter, soft overlong penalty) and predict what each choice teaches the policy when the length budget is too tight.
  - Build the response mask so that prompt tokens and padding are excluded and EOS is included, and show with a test that padding cannot change the loss.
  - Implement token and sequence importance ratios with asymmetric clipping, and check that the on-policy ratio is exactly 1 on the first minibatch.
  - Measure the effect of stale rollouts and of a lower-precision sampler, with and without importance correction, in controlled runs.
volatility: concept
sources:
  - title: "Liu et al., Understanding R1-Zero-Like Training (section 3.1: response-length bias; section 3.2: Dr. GRPO)"
    url: https://arxiv.org/abs/2503.20783
  - title: "Yu et al., DAPO (section 3.1 clip-higher; 3.3 token-level loss, Eq. 12; 3.4 overlong shaping, Eq. 13)"
    url: https://arxiv.org/abs/2503.14476
  - title: "Zheng et al., Group Sequence Policy Optimization (section 4.1; section 5.1 clipping ranges)"
    url: https://arxiv.org/abs/2507.18071
  - title: "Khatri et al., The Art of Scaling Reinforcement Learning Compute for LLMs (section 3.2)"
    url: https://arxiv.org/abs/2510.13786
  - title: "Schulman et al., Proximal Policy Optimization Algorithms (section 3, Eq. 7)"
    url: https://arxiv.org/abs/1707.06347
  - title: "Yao et al., Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (blog, 2025)"
    url: https://fengyao.notion.site/off-policy-rl
  - title: "He and Thinking Machines Lab, Defeating Nondeterminism in LLM Inference (2025-09-10)"
    url: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  - title: "Noukhovitch et al., Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models"
    url: https://arxiv.org/abs/2410.18252
last_verified: "2026-10-07"
---

# 12.3 · Details that change results

Two RL runs with the same algorithm name can disagree because one divides each response's loss by its length and the other does not, because one silently trains on padding, or because its rollouts came from a policy four updates old. This lesson takes the loss apart into the decisions that published recipes argue about — aggregation, truncation, masking, importance ratios and staleness — gives each one a unit test, and runs one controlled comparison per decision.

## Why this matters at a frontier lab

Dr. GRPO, DAPO, GSPO and ScaleRL are largely papers about these details, and each reports a visible change in length, stability or final score from changing them. At scale the stakes grow: rollouts run on a separate inference engine in another precision with other kernels, they are generated asynchronously while the trainer updates, and responses run to tens of thousands of tokens so truncation and length weighting decide much of the gradient. A detail done wrong rarely crashes anything. It shifts the objective, and the run optimises the shifted objective faithfully. The only defence is to know what each line is supposed to do and to test that it does.

## The idea

### Aggregation and length bias

A batch holds $B = PG$ responses; response $i$ has $|o_i|$ tokens with per-token objectives $\ell_{i,t}$ (the clipped surrogate times the advantage). The loss is some weighted sum $\sum_{i,t} w_{i,t}\,\ell_{i,t}$, and the aggregation decides the weights:

| Mode | $w_{i,t}$ | Used by |
|---|---|---|
| per-sequence mean | $\frac{1}{B}\cdot\frac{1}{\lvert o_i\rvert}$ | GRPO (DeepSeekMath Eq. 3) |
| token mean | $\frac{1}{\sum_j \lvert o_j\rvert}$ | DAPO (Eq. 12) |
| constant normaliser | $\frac{1}{B \cdot L_{\text{norm}}}$, $L_{\text{norm}}$ fixed (the generation budget) | Dr. GRPO (section 3.2) |
| prompt mean | $\frac{1}{P}\cdot\frac{1}{\sum_{j \in \text{group}(i)} \lvert o_j\rvert}$ | ScaleRL "prompt average" (section 3.2; our reading: each prompt counts equally, tokens within it equally) |

Under the per-sequence mean, every response has the same *total* weight $1/B$ whatever its length, so each token of a long response weighs less. With a negative advantage that is a discount on being long and wrong: Dr. GRPO's "response-level length bias", in which for incorrect responses "longer responses are penalized less" (section 3.1), and short correct responses are favoured for the same reason. Their Figure 1 reports that removing it "prevents the model from generating progressively longer incorrect responses". The token mean gives every token the same weight, so a response's total weight grows with its length (long responses dominate). The constant normaliser does the same with a fixed denominator, which also makes the loss scale independent of how long this batch's responses happen to be. Each choice is a statement about what a token of a long answer is worth; none is neutral.

### Truncation

Every rollout has a token budget $L_{\max}$. A response that hits it has no EOS: it is **truncated**, and its reward says nothing about whether the reasoning would have succeeded. Three treatments:

- **Keep it** with whatever the verifier says (usually 0). The policy learns "long is bad" even where the long answer was on its way to being right.
- **Overlong filtering**: drop truncated responses from the loss (DAPO section 3.4). No false signal, but no pressure against babbling either.
- **Soft overlong punishment** (DAPO Eq. 13): add a length penalty that starts $L_{\text{cache}}$ tokens before the limit,

$$R_{\text{length}}(y) = \begin{cases} 0 & |y| \le L_{\max} - L_{\text{cache}} \\ \dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}} & L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\ -1 & |y| > L_{\max} \end{cases}$$

Here $L_{\max}$ is the generation budget. DAPO sets an "expected maximum length" of 16,384 tokens and a 4,096-token punishment cache, so $L_{\max} = 20{,}480$ and the penalty starts at $L_{\max} - L_{\text{cache}} = 16{,}384$ (section 4.1). ScaleRL instead interrupts generation at a forced length ("interruption length control"). The truncation rate belongs in every training log.

### Loss masking

The policy gradient sums over **response** tokens. Prompt tokens were given, not sampled; tokens after EOS are padding the sampler wrote to keep the tensor rectangular. The mask is 1 from the first response token up to *and including* EOS: stopping is an action the policy took and must get credit or blame for. Two classic bugs: masking EOS out (the policy is never taught when to stop, and truncation creeps up), and leaving padding in (the loss trains the policy to predict PAD after EOS, and under token-mean aggregation every response looks equally long). Module 16 adds a third: tool outputs inside a multi-turn trajectory are observations, not actions, and must be masked like prompts.

### Importance ratios and clipping

One batch of rollouts is usually used for several gradient steps (minibatches, epochs). After the first step the policy $\pi_\theta$ differs from the one that sampled the batch, $\pi_{\text{old}}$, so each token is reweighted by

$$\rho_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})},$$

and PPO clips it: per token, $\min\big(\rho_t A,\ \mathrm{clip}(\rho_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\,A\big)$ (PPO Eq. 7, with "say, $\epsilon = 0.2$"). When the clipped branch is the minimum, the token's gradient is zero: the update stops pushing a token whose probability already moved far enough in the advantage's direction. DAPO's **clip-higher** uses $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$ so that unlikely tokens with positive advantage can grow more, which it reports keeps entropy from collapsing (section 3.1). **GSPO** replaces per-token ratios with one length-normalised sequence ratio $s_i = (\pi_\theta(y_i)/\pi_{\text{old}}(y_i))^{1/|y_i|}$, the geometric mean of the token ratios, clipped with much smaller ranges ($3 \times 10^{-4}$ and $4 \times 10^{-4}$, against 0.2 and 0.27 for its GRPO baseline; section 5.1). Module 14 derives GSPO and CISPO properly; here they are a flag and a test.

The check that catches most ratio bugs: **on-policy, on the first minibatch, $\rho = 1$ exactly** (old and current log-probabilities come from the same weights on the same tokens). If it is not, `old_logp` came from the wrong model, the wrong tokens, the wrong temperature or the wrong precision.

### Staleness and sampler mismatch

Asynchronous RL generates the next batch while the trainer updates on the current one (Noukhovitch et al.; ScaleRL's PipelineRL with up to 8 steps of off-policyness). The batch was then sampled by a policy $k$ updates old. Using the *behaviour* policy's log-probabilities as $\pi_{\text{old}}$ keeps the ratio an honest importance weight and lets clipping bound the update. Recomputing $\pi_{\text{old}}$ with the current weights (treating the batch as on-policy) makes every ratio 1 at the start of the update and silently drops the correction: the gradient is then for a different policy than the one that produced the data.

Even with $k = 0$, the sampler and the trainer are different programs. An inference engine uses other kernels, other batch shapes and often lower precision, so its probabilities differ from the trainer's for the same weights; inference itself is not batch-invariant (He and Thinking Machines Lab: 1,000 identical requests gave 80 distinct completions until the kernels were made batch-invariant). Yao et al. show that this mismatch makes "on-policy" RL silently off-policy, and correct it with **truncated importance sampling**: weight each token by $\min(\pi_{\text{old,trainer}}/\pi_{\text{sampler}}, C)$, comparing $C = 2$ and $C = 8$ (their Figure 4; TRL v1.14.1 defaults to a cap of 3.0, verl v0.9.1's `rollout_is_threshold` to 2.0). ScaleRL computes the LM head's logits in FP32 in both generator and trainer to shrink the mismatch at its source (section 3.2).

## Worked example

**Token weights for one group** of $G = 2$: a correct response of 2 tokens and a wrong response of 8 tokens, advantages $+1$ and $-1$.

| Mode | weight per token (short, long) | total weight (short, long) |
|---|---|---|
| per-sequence mean | $\frac{1}{2}\cdot\frac{1}{2} = 0.25$, $\frac{1}{2}\cdot\frac{1}{8} = 0.0625$ | 0.5, 0.5 |
| token mean | $\frac{1}{10} = 0.1$, $0.1$ | 0.2, 0.8 |
| constant, $L_{\text{norm}} = 8$ | $\frac{1}{2 \cdot 8} = 0.0625$, $0.0625$ | 0.125, 0.5 |

Under the per-sequence mean, each token of the long wrong answer is pushed down 4 times less hard than each token of the short right answer is pushed up. Under the token mean, the long wrong answer carries 80% of the group's weight.

**Soft overlong penalty** with DAPO's numbers ($L_{\max} = 20{,}480$, $L_{\text{cache}} = 4{,}096$): $|y| = 16{,}384$ gives 0; $|y| = 18{,}432$ gives $(16{,}384 - 18{,}432)/4{,}096 = -0.5$; $|y| = 20{,}480$ gives $-1$.

**Clipping.** $\rho = 1.5$, $A = +1$, $\epsilon_{\text{high}} = 0.28$: $\min(1.5, 1.28) = 1.28$, the clipped branch, so no gradient. $A = -1$: $\min(-1.5, -1.28) = -1.5$, the unclipped branch, so the gradient flows and pushes the probability back down. Clipping only stops updates that would move further in the advantage's direction.

**Sequence ratio.** Token ratios $2$ and $0.5$: their product is 1, so $s = 1^{1/2} = 1$ although each token moved a lot; per-token clipping would have clipped both.

**Truncated IS.** Trainer $\pi_{\text{old}} = 0.5$, sampler $0.1$: $\min(5, 2) = 2$. Trainer $0.2$, sampler $0.4$: $0.5$.

**A tight budget.** In the toy task a sum below 100 needs 2 digits plus EOS (3 tokens); a sum of 100 or more needs 4. With a 3-token budget every response to a large sum is truncated and none can be right: keeping them with reward 0 makes those groups all-wrong (zero variance); filtering removes them. DAPO's design with an expected maximum of 3 and a 1-token cache gives a budget of 4, so a correct 3-digit answer now fits, but Eq. 13 gives length 4 a penalty of $(3 - 4)/1 = -1$: a correct large sum scores $1 - 1 = 0$, the same as a wrong one. An expected maximum set below the length the task needs turns the length penalty into a penalty on correct answers.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| response tokens | (B, R) = (128, 8) | int64 | sampler's |
| mask, token weights | (128, 8) | float32 (float64 in `token_weights`) | same |
| ratio, clipped flag | (128, 8) | float32 | trainer's |
| sampler log-probs | (128, 8) | float32 (bf16 sampler: computed from bf16 logits cast to float32) | sampler's |
| behaviour log-probs (trainer precision) | (128, 8) | float32 | trainer's |
| policy snapshots for staleness $k$ | $k + 1$ state dicts | float32 | CPU memory: $(k + 1) \times 1.2$ MB here, $(k+1) \times 6.9$ GB for a float32 1.7B model |

Aggregation, masking and ratios cost nothing measurable. The importance correction for a separate sampler needs one extra forward of the trainer over the batch (the behaviour log-probabilities, $2N$ FLOPs per token), which the course loop always computes. Staleness costs memory for the snapshots (an asynchronous system keeps them on the rollout workers instead).

## Build it

```python
import torch
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain.policy import response_mask

resp = torch.tensor([[5, 6, 2, 0, 0, 0, 0, 0],      # EOS (id 2) at position 2, then PAD (id 0)
                     [5, 6, 7, 8, 9, 10, 11, 12]])   # truncated: no EOS
mask, finished = response_mask(resp)                 # [[1,1,1,0,0,0,0,0], [1]*8], [True, False]
print(Lo.token_weights(mask, "seq_mean_token_mean"))  # 1/6 per short token, 1/16 per long token
print(Lo.soft_overlong_penalty(torch.tensor([18432.]), 20480, 4096))   # -0.5 (DAPO's budget and cache)
```

The loop flags: `--aggregation`, `--overlong none|filter|soft --l-cache`, `--ratio token|sequence|none`, `--eps-low/--eps-high`, `--staleness k --stale-correction behaviour|recompute`, `--sampler-dtype fp32|bf16 --tis-cap C`. It logs `trunc`, `len_correct`, `len_wrong`, `clip_frac`, `ratio_max` and `sampler_gap` (mean absolute difference between the trainer's and the sampler's log-probabilities). Correctness checks (`labs/common/tests/test_posttrain.py`): aggregation and token weights against the hand values; clipping zeroes the gradient exactly where it should; the sequence ratio is the geometric mean; DAPO's penalty at the paper's numbers; sampler and teacher-forced log-probabilities agree to $10^{-5}$ in float32; left padding does not change a Hugging Face model's log-probabilities (the main path). The project's `test_loop.py` adds the integration checks: padding cannot change the loss; the first on-policy ratio is 1.

## What the evidence says

- **Response-length bias of per-sequence normalisation: ESTABLISHED as a mechanism** (it follows from the weights), with the empirical consequence (incorrect responses growing longer) reported by Dr. GRPO (PUBLICLY DOCUMENTED). Which aggregation is best is **debated**: DAPO uses token-level, Dr. GRPO a constant, ScaleRL prompt-level, each with its own ablation.
- **Overlong handling matters for long-response RL: PROMISING** (DAPO's ablation; ScaleRL's interruption control). The right $L_{\text{cache}}$ is setup-specific (MODEL-SPECIFIC).
- **Clip-higher: PROMISING** (DAPO section 3.1; adopted in several later recipes; TRL's help text recommends 0.28). **Sequence-level ratios (GSPO): PROMISING**, reported to stabilise MoE training in particular (Module 14).
- **Masking prompts and padding and including EOS: ESTABLISHED** (every framework does it; REASONABLE INDUSTRY PRACTICE to unit-test it).
- **Rollout-trainer mismatch makes on-policy RL off-policy: PUBLICLY DOCUMENTED** (Yao et al.; He and Thinking Machines Lab on batch invariance); **truncated IS as the fix: PROMISING and now a framework default** (TRL v1.14.1 `vllm_importance_sampling_correction=True`).
- **Bounded staleness with importance correction is workable: PROMISING** (Asynchronous RLHF; ScaleRL's PipelineRL-8). How much staleness a recipe tolerates depends on the learning rate and the clip range (INFERENCE).
- **Course measurement (free CPU, 2026-10-07, 2 seeds):** aggregation made no difference at 2–4-token responses; a soft overlong penalty with a threshold below the needed length taught the policy to answer every unsolvable problem with a short wrong number; a mask that includes padding cost 0.087 accuracy and raised entropy by 0.12; stale rollouts with behaviour log-probabilities trained as well as on-policy ones, while the same rollouts treated as on-policy fell below the SFT start in both seeds. Details in the lab's results box.

## Lab

**Folder:** [`labs/module-12/lesson-03/`](../../labs/module-12/) · **Time:** about 2 hours (about 20 minutes of it unattended) · **Pass check:** `pytest labs/module-12/lesson-03` passes; `details_lab.py` prints the four comparisons; the write-up explains each with the mechanism, and says for each detail whether the unit test or the training curve was the better detector.

### Experiment contract

- **Question:** for each of four details (aggregation, truncation handling, masking, staleness and sampler precision), does changing it from the course default change held-out accuracy, length or stability on the toy task, and which metric shows the change first? Decision informed: the defaults and the logged metrics of the Module 12 project loop.
- **Hypothesis:** aggregation changes little at 2–4-token responses (the length range is too small for length bias to matter: may not appear at this scale); with a 3-token budget, filtering and keeping truncated responses both leave large sums unsolved, and the soft penalty (expected maximum 3, cache 1) stops the policy from improving on large sums even though they now fit; the padded mask and the EOS-less mask train nearly normally over 120 steps but fail their unit tests; stale rollouts treated as on-policy learn worse than corrected ones, and the bf16 sampler's gap is small enough that TIS changes little. Status: mechanisms established; effect sizes at this scale unknown.
- **Baseline per part:** `grpo-seqmean`, `trunc-none`, `mask-correct`, `onpolicy` (all GRPO advantages, lr $3 \times 10^{-4}$, untuned).
- **Changed variable:** one detail per arm, as named. **Controlled:** the SFT checkpoint, 16 prompts × 8 samples, 120 steps, temperature 1, 2 minibatches, clip 0.2, seeds 0–1, the 300 held-out evaluation problems and sampling seed. Your `response_mask` and your policy loss (built from your `aggregate`, ratios and `clipped_surrogate`) drive every run.
- **Comparison axis:** equal samples.
- **Budget:** free CPU, measured 21 minutes for the 30 runs.
- **Metrics and decision rule:** primary: held-out sampled accuracy at step 120, paired by seed against the part's baseline, 95% t-interval; "changed the result" only if the interval excludes 0. Secondary: `len`, `len_wrong`, `trunc`, `entropy`, `clip_frac`, `ratio_max`, `sampler_gap`; for truncation, the accuracy on sums below 100 and the fraction of large-sum responses that finish (necessarily wrong).
- **Correctness checks:** your TODO tests; `test_posttrain.py`; `labs/module-12/project/test_loop.py` (padding invariance, on-policy ratio 1).
- **Fallback evidence:** a null with the seed spread; Dr. GRPO's and DAPO's ablations labelled as published.
- **Limits:** 2 seeds, 120 steps, responses of at most 8 tokens, one task; staleness up to 4 steps; bf16 emulated on CPU.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 12 pilot | `python labs/module-12/lesson-03/details_lab.py --variant main --print` prints 12 runs (aggregation and truncation arms × 2 seeds, 100 steps of `frontierlab.posttrain.hf`, where responses are hundreds of tokens and length bias can show). **PROJECTED:** 12 × 1–1.5 GPU-hours = 12–18 GPU-hours, USD 24–54 |
| Free GPU (Colab/Kaggle T4) | T4 | the same commands with Qwen3-0.6B-Base (`--model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd`), `--prompts 8`, fp32 |
| Free CPU | laptop; measured 21 minutes for the 30 runs (other jobs running) | the steps below |

### Steps

1. **Implement** the six TODOs in `lab.py` and run `pytest labs/module-12/lesson-03`.
2. **Predict** each part's outcome in one sentence.
3. **Run** `python labs/module-12/lesson-03/details_lab.py` (or one `--part` at a time).
4. **Run the integration checks** on a buggy mask: `LOOP_HOOKS=buggy pytest labs/module-12/project -k padding` and compare with the `mask-with-pad` arm's training curve. Which detector found the bug?
5. **Write up** each part: result with interval, mechanism, and the metric that showed it.

<details>
<summary>Hint for TODO 1</summary>

`torch.cumsum((response == EOS).long(), -1)` counts EOS tokens up to and including each position; subtract the EOS indicator itself to count those strictly *before* it. A position is in the mask exactly when that count is 0.

</details>

<details>
<summary>Hint for TODO 2</summary>

For `"prompt_mean"`, `x.view(P, -1)` puts each group's $G \times R$ values in one row, because groups are consecutive rows.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, another module's jobs sharing the CPU): 30 runs of 120 steps, 21 minutes in all. With 2 seeds the 95% t-interval uses $t_{0.975,1} = 12.7$, so it is very wide; where both seeds moved the same way by a lot, say so, but the rule's verdict is what it is.

**Aggregation.** Held-out accuracy: per-sequence mean 0.342, token mean 0.327, constant 0.362, prompt mean 0.330; every difference inconclusive, and lengths of wrong answers identical to 0.03 tokens. As predicted, length bias cannot show when every response is 2–4 tokens long: the per-token weights differ by at most a factor of 2 and every arm saw the same lengths. This part needs the main path's hundreds of tokens.

**Truncation (3-token budget).** At the start, 5% of large sums got a finished (necessarily wrong) 2-digit answer.

| Arm | accuracy, sums < 100 | finished-but-short on sums ≥ 100 | truncation rate | entropy |
|---|---|---|---|---|
| keep (reward 0) | 0.368, 0.318 | 0.65, 0.75 | 0.38 | 0.460 |
| filter | 0.354, 0.393 | 0.50, 0.40 | 0.45 | 0.463 |
| soft (expected maximum 3, cache 1) | 0.307, 0.371 | **1.00, 1.00** | **0.00** | 0.387 |

No arm solved a single large sum (none could at budget 3, and under the soft arm's budget 4 a correct answer scores 0). Keeping truncated responses taught the policy to write a short wrong answer most of the time on problems it could not finish; filtering did so less; the soft penalty made it universal: every large sum got a finished 2-digit answer, truncation fell to 0 and entropy dropped. The training curves look healthy in all three arms (accuracy on small sums 0.31–0.39, within noise of each other); only the breakdown by problem type shows that the soft arm learned to give up. A length penalty whose threshold sits below the length a correct answer needs trains the model to be short and wrong.

**Masking.** Including PAD after EOS cost accuracy, −0.087 [−0.171, −0.002], and raised entropy by +0.121 [+0.098, +0.144]; the clip fraction rose from 0.001 to 0.015 and the maximum ratio to about 1.9 (gradients flowing into PAD logits move the whole output distribution). Masking out EOS changed nothing visible in 120 steps (+0.028, inconclusive; lengths identical), because the SFT start already stops reliably: that bug is caught by the unit test, not by the curve.

**Staleness and sampler precision.**

| Arm | accuracy | vs on-policy | clip fraction | max ratio | sampler gap (nats/token) |
|---|---|---|---|---|---|
| on-policy | 0.343, 0.310 | — | 0.001 | 1.17 | 0 |
| 4 steps stale, behaviour log-probs | 0.353, 0.317 | +0.008 [−0.013, +0.030] | 0.055 | 2.6 | 0 |
| 4 steps stale, recomputed (no correction) | 0.110, 0.127 | −0.208 [−0.526, +0.109] | 0.017 | 1.22 | 0 |
| bf16 sampler | 0.387, 0.327 | +0.030 [−0.139, +0.199] | 0.001 | 1.18 | 0.011 |
| bf16 sampler + TIS (cap 2) | 0.380, 0.310 | +0.018 [−0.215, +0.251] | 0.001 | 1.18 | 0.010 |

Stale data with the behaviour policy's log-probabilities trained as well as on-policy data; the correction is visible in the logs as a clip fraction 50 times higher and ratios up to 2.6 (clipping doing its job). Treating the same stale data as on-policy made accuracy fall *below the SFT start* (0.21) in both seeds (to 0.11 and 0.13, a drop of about 0.21 per seed), with ratios that look perfectly healthy (max 1.22): the bug hides from the ratio metrics because the ratio is computed against the wrong policy. The rule calls it inconclusive only because two seeds give a 12.7× t-multiplier; this is the arm to rerun with more seeds. The bf16 sampler's gap was 0.011 nats per token, too small at this scale for TIS to change anything measurable.


</details>

<details>
<summary>Reference solution</summary>

`labs/module-12/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-12/lesson-03`.

</details>

## Common mistakes

- **Comparing aggregation modes at different learning rates without saying so.** The modes change the loss scale (the constant normaliser is several times smaller than the token mean); part of any difference is an effective learning-rate change.
- **Filtering truncated responses and declaring the length problem solved.** Filtering removes the false signal and the pressure; watch the truncation rate.
- **Computing `old_logp` after the first optimizer step**, or from the reference model, or at another temperature. Test that the first on-policy ratio is 1.
- **Treating a stale batch as on-policy.** Store the behaviour policy's log-probabilities with the rollout.
- **Trusting `sampler_logp` as `old_logp`.** It comes from another program; use the trainer's numbers for the ratio and the sampler's only for the mismatch weight.
- **Masking EOS.** The policy then never learns to stop on purpose.

## References

- Z. Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, 2025, sections 3.1–3.2. https://arxiv.org/abs/2503.20783
- Q. Yu et al., *DAPO*, 2025, sections 3.1, 3.3, 3.4. https://arxiv.org/abs/2503.14476
- C. Zheng et al., *Group Sequence Policy Optimization*, 2025, sections 4.1 and 5.1. https://arxiv.org/abs/2507.18071
- D. Khatri et al., *The Art of Scaling Reinforcement Learning Compute for LLMs*, 2025, section 3.2. https://arxiv.org/abs/2510.13786
- J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, section 3. https://arxiv.org/abs/1707.06347
- F. Yao et al., *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*, 2025. https://fengyao.notion.site/off-policy-rl
- H. He and Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- M. Noukhovitch et al., *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models*, 2024. https://arxiv.org/abs/2410.18252
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[12.4 · Eval Suite v2](lesson-04.md)
