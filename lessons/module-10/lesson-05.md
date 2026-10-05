---
id: "10.5"
module: 10
minutes: 30
practice_minutes: 100
prerequisites: ["10.1", "10.4", "04.3", "07.4"]
objectives:
  - Define the target gain and the forgetting of a continued-training run against an equal-token control, and explain why measuring either against the starting checkpoint alone is misleading.
  - Run a targeted mid-training study from a finished checkpoint with the --init-from pattern (fresh optimizer, re-warm, re-decay) and replay fractions of 0%, 50% and 90% general data.
  - Track forgetting during training from the loop's own validation log and summarise the gain–forgetting trade-off as a Pareto frontier.
  - Choose a replay fraction with a pre-stated rule and report the limits of a result obtained on one small model.
volatility: concept
sources:
  - title: "Ibrahim et al., Simple and Scalable Strategies to Continually Pre-train Large Language Models (abstract: re-warming, re-decaying, replay)"
    url: https://arxiv.org/abs/2403.08763
  - title: "OLMo Team, 2 OLMo 2 Furious (Table 9: pretraining vs mid-training; section 4.1)"
    url: https://arxiv.org/abs/2501.00656
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.1.3: annealing on code and math; section 3.4.3)"
    url: https://arxiv.org/abs/2407.21783
last_verified: "2026-10-04"
---

# 10.5 · Continued training and forgetting

Continued training takes a finished model and trains it further on data chosen for one purpose: a domain, a language, a skill, a longer context (Module 4 did the last one). It nearly always helps the target, and it nearly always costs something elsewhere. This lesson measures both numbers properly, the gain on the target and the forgetting on everything else, each against a control that saw the same number of tokens, and uses them to choose how much general data to replay.

## Why this matters at a frontier lab

Mid-training is now a standard stage of open recipes, and it is where many headline capability jumps come from: OLMo 2 7B's GSM8K score went from 24.1 after pretraining to 67.5 after mid-training on 50B tokens, with MMLU from 59.8 to 63.7 (Table 9; PUBLICLY DOCUMENTED). The same mechanism is how a lab adapts a released model to a customer's domain, a new language or a code base. The decisions are always the same three: how many tokens, how much replay of the original data, and what learning rate schedule. They are made on two numbers that are easy to measure wrongly.

## The idea

### Gain and forgetting, against the right reference

Let $\theta_0$ be the finished model and $\theta_A$ the model after continued training arm $A$ on $D$ tokens. On a target held-out set $T$ (here FineMath) and a general held-out set $G$ (here Eval v0's Data-v0 validation windows), the tempting definitions compare with $\theta_0$:

$$\Delta^{\text{naive}}_T = L_T(\theta_0) - L_T(\theta_A), \qquad \Delta^{\text{naive}}_G = L_G(\theta_A) - L_G(\theta_0).$$

Both mix two effects: what the target data did, and what $D$ more tokens of *any* training with a re-warmed and re-decayed learning rate did. The **equal-token control** $\theta_C$ continues $\theta_0$ on the general data only, for the same $D$ tokens with the same schedule. The course definitions:

$$\text{gain} = L_T(\theta_C) - L_T(\theta_A), \qquad \text{forgetting} = L_G(\theta_A) - L_G(\theta_C).$$

Positive gain means the target data helped the target; positive forgetting means the arm is worse on general text than it would have been with the same training on general data. If $\theta_0$ is not fully converged, the control improves $L_G$ on its own and the naive forgetting understates the cost; if the re-warm disturbs $\theta_0$, the naive gain overstates the benefit. Module 4's project hit exactly this: a control at the original length improved short-context loss by 0.098 nats on its own.

### Replay

The simplest protection against forgetting is **replay**: keep a fraction $1 - f$ of the continued-training tokens from the original distribution. Ibrahim et al. report that "a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data is sufficient to match the performance of fully re-training from scratch", at 405M and 10B parameters, for a weak shift (English to English) and a strong one (English to German) (abstract; PUBLICLY DOCUMENTED). Published mid-training mixes keep most of their tokens general: OLMo 2's Dolmino mixes are about half filtered web data (Table 13), and Llama 3's dataset-scoring anneals put 70% weight on the default mix (section 3.1.3).

With the counter-based sampler of lesson 10.1 the split is exact: an arm with target fraction $f$ sees $f \cdot D$ target tokens and $(1 - f) D$ replay tokens, written into its run card before the run starts.

### The schedule

`frontierlab.datax.train --init-from CKPT` loads only the weights; the optimizer starts with fresh moments and the loop's warmup and cosine decay start from step 0: re-warming and re-decaying in Ibrahim et al.'s terms, and the same pattern `frontierlab.longctx.extend` used in lesson 04.3 (lesson 07.4 covers why the moments of the old run do not belong to the new data). The peak learning rate is a variable of its own: higher re-warm peaks adapt faster and forget more; the lab fixes it at a third of the base run's.

### The trade-off as a frontier

Each arm gives a point (gain, forgetting). An arm is **dominated** if another arm has at least its gain and at most its forgetting, strictly better in one. The non-dominated arms form the Pareto frontier; the pre-stated rule picks one point on it: the largest mean gain whose gain interval is above zero and whose forgetting interval's upper bound is within the budget (0.02 nats in the lab).

## Worked example

Two seeds, per-seed mean losses:

| | control $C$ | arm $A$ |
|---|---|---|
| target $L_T$ | 4.50, 4.40 | 4.00, 4.10 |
| general $L_G$ | 6.00, 6.05 | 6.10, 6.20 |

Gain per seed: $4.50 - 4.00 = 0.50$, $4.40 - 4.10 = 0.30$; mean 0.40. Forgetting per seed: $6.10 - 6.00 = 0.10$, $6.20 - 6.05 = 0.15$; mean 0.125. With the base model at $L_G(\theta_0) = 6.08$, the naive forgetting would be $6.15 - 6.08 = 0.07$: about half the real cost, because the control itself improved general loss by 0.055.

Target tokens: 300 steps × 16 × 128 = 614,400 tokens per run; at $f = 0.5$, exactly 307,200 target tokens and 307,200 replay tokens.

Frontier: arms with (gain, forgetting) = (0.50, 0.20), (0.40, 0.05), (0.30, 0.10), (0.00, 0.00). The third is dominated by the second (less gain, more forgetting); the frontier is the other three. With a forgetting budget of 0.02 on the upper bound, only arms whose interval clears it are eligible.

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| initial checkpoint | the base run's `checkpoint.pt` (model, optimizer, step, RNG) | only `model` and `config` are read; SHA-256 and step go into the run card |
| training batch | (16, 128) int64 | windows from the arm's mixture |
| forgetting curve | `metrics.jsonl` rows with `split: val` every 50 steps | Data-v0 validation, 32 windows: cheap, noisier than the final 256-window score |

Cost per arm equals its tokens times the model's training FLOPs per token, the same for every arm; the control is one more arm. On the main path the base is your Module 1 Baseline-0 run (2.49B tokens) and each arm continues for 131M tokens (5%), PROJECTED below.

## Build it

```bash
python -m frontierlab.datax.train --init-from runs/m10/l105/base/checkpoint.pt --mixture runs/m10/l105/math50-s0/mixture.json \
    --run runs/m10/l105/math50-s0 --preset toy --steps 300 --batch 16 --seq 128 --lr 1e-3 --warmup 30 --seed 0 --eval-every 50
```

The run card's `datax` block records `init_from`, `init_sha256`, `init_step`, the mixture and the planned target and replay tokens; `parent_run` is the base run. Rerunning the command after an interruption resumes the *continued* run exactly (the run's own checkpoint wins over `--init-from`). Correctness checks: `pytest labs/common/tests/test_datax.py -k init_from` (parent and step recorded, anneal schedule), and the wrapper's exact-resume test of lesson 10.1.

## What the evidence says

- **Mid-training / continued pretraining on targeted data improves the target: ESTABLISHED** (OLMo 2 Table 9, Llama 3 section 3.1.3, many others; PUBLICLY DOCUMENTED). Llama 3 also reports that the same annealing gave large gains on GSM8K and MATH validation for the 8B model and negligible ones for the 405B model (section 3.1.3): the size of the gain depends on scale.
- **Replay plus re-warming and re-decaying the learning rate controls forgetting well enough to match re-training: PROMISING** (Ibrahim et al., one group, two sizes, two shifts).
- **The best replay fraction and peak learning rate: MODEL-SPECIFIC**; they depend on the shift, the model and the token budget, which is why the lab measures them.
- **Course measurement (free CPU, 2026-10-05, other jobs running):** from a 600-step Data-v0 base, 300 continued steps (614,400 tokens) per arm, 3 seeds. Every FineMath share gave a clear target gain (100%: +1.26 nats, 95% CI [+1.13, +1.39]; 50%: +1.12 [+1.03, +1.22]; 10%: +0.70 [+0.54, +0.85]) and forgetting on Data-v0 grew with the share (+0.45 [+0.40, +0.51]; +0.12 [+0.09, +0.15]; +0.017 [−0.006, +0.039]). All three arms are on the Pareto frontier, and none meets the pre-stated rule (forgetting upper bound at most 0.02): the rule keeps the base. 10% FineMath misses the budget by 0.019 on the upper bound with 3 seeds, a case for a new pre-stated arm (5%) or more seeds decided in advance, not for relaxing the budget after the fact.

## Lab

**Folder:** [`labs/module-10/lesson-05/`](../../labs/module-10/) · **Time:** about 100 minutes (about 60 of them unattended) · **Pass check:** `pytest labs/module-10/lesson-05` passes; `continued_training.py` prints the table, the gain and forgetting with intervals, the frontier and the choice; your write-up includes the forgetting curves.

### Experiment contract

- **Question:** which share of FineMath in a 300-step continued-training run gives the largest gain on FineMath held-out loss with at most 0.02 nats of forgetting on Data-v0 held-out loss, against an equal-token control? Decision informed: the target/replay split of Data-v1's domain mid-training stage.
- **Hypothesis:** gain grows with the FineMath share; forgetting grows faster, so 100% FineMath exceeds the budget and 10–50% stays within it. Status: reported effects (Ibrahim et al.; OLMo 2); sizes unknown at this scale.
- **Baseline:** `ctrl`: the base continued on Data-v0 only, same tokens, schedule and seeds.
- **Changed variable:** FineMath share $f \in \{1.0, 0.5, 0.1\}$. **Controlled:** the base checkpoint (600 steps of Data-v0, SHA-256 in every run card), 300 continued steps of 16 × 128 tokens, peak learning rate $10^{-3}$, 30 warmup steps, cosine, fresh optimizer, seeds 0–2 (data order; the starting weights are the base's in every run).
- **Comparison axis:** equal training tokens.
- **Budget:** free CPU: base run plus 12 continued runs plus scoring (measured below).
- **Metrics and decision rule:** gain on FineMath validation and forgetting on Data-v0 validation (256 windows of 128 tokens), each per seed against `ctrl` and summarised by a seed-level 95% t-interval; choose the arm with the largest mean gain among those whose gain lower bound is above 0 and whose forgetting upper bound is at most 0.02 nats; if none qualifies, keep the base. Secondary: Wikipedia and web validation, LAMBADA log-probability (forgetting elsewhere).
- **Correctness checks:** every continued run's card has `datax.init_sha256` equal to the base checkpoint's SHA-256 and `parent_run: base`; `mixture_accounting.json` shows the planned target tokens.
- **Fallback evidence:** none; a null result is valid.
- **Limits:** one base, one target domain, one peak learning rate, 0.6M continued tokens, 1.8M parameters; Llama 3's 8B-vs-405B result says the size of the gain itself changes with scale.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100. Not run in this build; part of the Module 10 pilot | `continued_training.py --variant main --base runs/m01/main/<your Baseline-0 seed-0 run>` (FineMath prepared with the vocabulary-32,768 tokenizer; 12 runs of 4,000 steps × 32 × 1,024 = 131M tokens). PROJECTED: 12 × 1.31e8 × 0.788 GFLOP/token ÷ (989e12 × 0.3) = 1.16 GPU-hours |
| Free GPU (Colab/Kaggle T4) | T4 | `continued_training.py --variant t4` (pilot-10m base of 4,000 steps, then 1,500-step arms) |
| Free CPU | laptop; measured: base 165 s, 12 continued runs of 217–285 s each (about 50 minutes), plus scoring | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py`: gain and forgetting against the control, target tokens, the Pareto frontier, the choice rule. Run `pytest labs/module-10/lesson-05`.
2. **Run** (unattended; reruns continue): `python labs/module-10/lesson-05/continued_training.py`.
3. **Check the record:** `python -m frontierlab.record runs/m10/l105/ctrl-s0 runs/m10/l105/math50-s0 --changed datax.mixture datax.mixture_digest datax.planned_accounting` and confirm the base hash matches in both cards.
4. **Write up:** the table of gain and forgetting with intervals; the naive (against the base) and the controlled numbers side by side for `math100`; the frontier and the choice; the forgetting curves (does Data-v0 loss recover during the decay?); what you would vary next (the peak learning rate is the obvious one).

<details>
<summary>Hint for TODO 3</summary>

Loop over pairs; arm `i` is dominated if some `j != i` has `gain_j >= gain_i and forg_j <= forg_i` and at least one of the two strictly.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-05 on the build laptop (torch 2.14.1 CPU, 16 threads shared with another job's runs). Base: 600 steps of Data-v0, 165 s. Continued arms: 300 steps of 16 × 128 tokens at peak 1e-3, 30 warmup steps, cosine, fresh optimizer; 217–285 s each (one run's wall time includes a stall of about 1.8 hours while the machine was busy or asleep; its losses are unaffected, since resume is exact). Every continued run card has `datax.init_sha256` `68c34871…6bea`, the base checkpoint's hash. Scoring: 256 windows of 128 tokens per set, 1,000 LAMBADA passages.

Mean loss, base and control, and each arm against the control (seed-level 95% intervals; for LAMBADA, log-probability, higher is better):

| Arm | FineMath val (target) | Data-v0 val (guard) | Wikipedia val | web val | LAMBADA |
|---|---|---|---|---|---|
| base (start) | 6.744 | 6.268 | 6.434 | 6.368 | −17.08 |
| ctrl (Data-v0 only) | 6.732 | 6.197 | 6.352 | 6.315 | −17.08 |
| math100 − ctrl | −1.259 [−1.393, −1.125] | +0.454 [+0.396, +0.512] | +0.477 [+0.398, +0.556] | +0.402 [+0.322, +0.481] | −0.92 [−1.13, −0.72] |
| math50 − ctrl | −1.121 [−1.216, −1.026] | +0.119 [+0.089, +0.150] | +0.097 [+0.090, +0.103] | +0.106 [+0.083, +0.129] | −0.13 [−0.19, −0.07] |
| math10 − ctrl | −0.696 [−0.853, −0.540] | +0.017 [−0.006, +0.039] | −0.015 [−0.024, −0.006] | +0.014 [−0.002, +0.031] | +0.01 [−0.17, +0.18] |

Target tokens: 614,400 (math100), 307,200 (math50), 61,440 (math10), exactly as planned in the run cards. Pareto frontier: all three arms. Choice under the rule (gain lower bound above 0, forgetting upper bound at most 0.02): **none, keep the base**.

What it shows. (1) The control matters. Against the base, math100's forgetting looks like 6.651 − 6.268 = 0.383 nats; against the control it is 0.454, because 300 more steps of Data-v0 improved the base by 0.071 on its own. The naive gain (6.744 − 5.473 = 1.270) is close to the controlled one here only because extra Data-v0 barely changed FineMath loss. (2) Gain saturates and forgetting does not: 10% of the target tokens bought 55% of the full-share gain for 4% of its forgetting, the shape Ibrahim et al. and OLMo 2's 10/90 anneal suggest. (3) The forgetting curves (Data-v0 validation, 32 windows, seed 0): ctrl 6.298, 6.435, 6.292, 6.232, 6.190, 6.175 at steps 50–300; math10 6.311, 6.400, 6.298, 6.242, 6.215, 6.202; math50 6.343, 6.401, 6.407, 6.358, 6.299, 6.282; math100 6.444, 6.724, 6.644, 6.587, 6.605, 6.588. Every arm, the control included, gets worse right after the re-warm (step 100, peak learning rate) and recovers during the decay; replay arms recover most of it, math100 does not. (4) The rule's null for math10 is a measurement, not a failure: with a seed-level interval half-width of about 0.023 nats, 3 seeds cannot certify a forgetting below 0.02 unless the true value is near zero. For Data-v1 the project's example recipe keeps FineMath in the main mixture at its natural token share instead of adding a separate mid-training stage, and says why.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-10/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-10/lesson-05`.

</details>

## Common mistakes

- **Measuring forgetting against the starting checkpoint.** Extra training changes general loss by itself; use the equal-token control.
- **Continuing with the old optimizer state and schedule.** The moments describe the old data, and a schedule that already decayed to its minimum gives no re-warm; start fresh and say so in the run card.
- **Pointing `--init-from` at the run folder you are writing to.** The second launch loads the run's own checkpoint (exact resume) and ignores `--init-from`; a new arm needs a new folder.
- **Judging forgetting on the target-adjacent set only.** Check several general sets (here Data-v0, Wikipedia, web, LAMBADA); forgetting is uneven.
- **Reading the 32-window validation curve as the result.** It is for watching the run; the decision uses the 256-window score with seeds.

## References

- A. Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models*, 2024, abstract. https://arxiv.org/abs/2403.08763
- OLMo Team, *2 OLMo 2 Furious*, 2024, section 4.1 and Table 9. https://arxiv.org/abs/2501.00656
- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, sections 3.1.3 and 3.4.3. https://arxiv.org/abs/2407.21783
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[10.6 · Multilingual and code data](lesson-06.md) is an extension. The module project then assembles Data-v1 from this module's decisions: [Module 10 project](../../projects/module-10-data-v1.md).
