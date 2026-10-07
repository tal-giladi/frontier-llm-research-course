# Module 14 project · A reasoning-RL run you can defend

This project turns the module into one result someone could act on. You train the Stage D base model with reasoning RL using the objective your lesson 14.2 comparison chose, next to a random-reward control. You report the run's stability by metrics fixed in advance, check capability retention with Eval Suite v2, and say with a pass@k analysis what the run bought: higher pass@1, wider coverage, or neither beyond what the control got. You also say what the run's own curve does and does not tell you about how far the recipe would go. Before trusting your objective code, you debug a colleague's objective library with three planted bugs.

**Time:** 6–8 attended hours plus unattended runs. **Folder:** [`labs/module-14/project/`](../labs/module-14/) (`run_project.py`, `test_objectives.py`, `buggy_objectives.py`, `buggy_run.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model and task | Hardware | Cost |
|---|---|---|---|
| Main path | Qwen3-1.7B-Base (revision `ea980cb`), GSM8K (revision `740312a`), `python -m frontierlab.rlscale.hf_rl`: your objective and its random-reward control, 2 seeds × 300 steps, 64 samples per question on 200 test questions every 50 steps; Eval v2 with `labs/module-12/lesson-04/eval_main.py` | 1× H100 80 GB | **PROJECTED, pending the Module 14 pilot:** 4 runs × 300 steps × 30–50 s = 10–17 GPU-hours, plus 7 evaluations per run × 0.5–1 GPU-hours (pass@k sampling) and 4 Eval v2 runs × 0.3–0.5: 25–47 GPU-hours, USD 50–141 at USD 2–3 per H100-hour. `python labs/module-14/project/run_project.py --variant main --objective <yours> --print` prints the commands |
| Free GPU (Colab/Kaggle T4) | Qwen3-0.6B-Base (revision `da87bfb`), `--prompts 8 --max-new 256 --eval-samples 16`, fp32 | T4 | PROJECTED 4–6 hours per run; the loop's checkpoints let you resume after a disconnect (rerun the same command) |
| Free CPU | the toy policy and task, `python labs/module-14/project/run_project.py --objective <yours>` (3 seeds × 200 steps per arm) | laptop | measured 18.7 minutes (reference box below); `buggy_run.py` 4.5 minutes |

## Deliverables

1. **The experiment record:** your filled contract, the run cards (written by the loop), the raw `metrics.jsonl` of every run, the printed tables.
2. **A stability report** per seed, using lesson 14.2's metrics and the thresholds you stated in your contract: entropy drop, gradient spikes and 99th-percentile norm, maximum ratio, clip fraction (with what it means for your objective), KL to the start, drawdown, and the flags. Then one paragraph: would you trust this objective for a run ten times longer, and which metric would you watch?
3. **Eval v2 retention:** the guard verdict per seed and per component against the SFT start (CPU) or Qwen3-1.7B-Base (main path), with the guards stated in advance.
4. **A pass@k analysis:** pass@k for $k$ from 1 to 256 (CPU) or 64 (main path) for the start, your policy and the control; paired differences with intervals at the smallest and largest $k$; the crossover if any; the solved-ever sets. State what each says about "RL added capability".
5. **The asymptote paragraph:** the profile interval for $A$ from your run's own curve, and the sentence you can defend about it.
6. **The debugging report** (below) and **the written defence** (below).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running. Fixed by the project:

- **Question:** does RL with your chosen objective raise held-out pass@1 beyond a random-reward control, without failing Eval v2's retention guards and without breaking the stability thresholds, and does it change pass@k at the largest $k$ you measure? Decision informed: whether this objective and learning rate become the default for Modules 15–16.
- **Hypothesis:** pass@1 rises beyond the control in every seed; Eval v2 holds; no stability flag; pass@k at the largest $k$ changes less than pass@1 (possibly a crossover). Status: pass@1 gains and the pass@k pattern are reported for larger models (Yue et al.); the control result is model-dependent (Shao et al.: large spurious gains on Qwen2.5-Math, none on Llama 3 or OLMo 2).
- **Baseline:** the random-reward control arm (same objective, settings, seeds), plus the start checkpoint for Eval v2 and pass@k.
- **Changed variable:** the reward (verifier vs random). **Controlled:** objective and learning rate (from your lesson 14.2 decision), start checkpoint, prompt stream per seed, rollouts per step, data reuse, KL setting, steps, every evaluation pin.
- **Comparison axis:** equal sampled responses and equal updates.
- **Metrics and decision rule:** final held-out pass@1, paired by seed against the control, 95% t-interval, which must lie above 0. Eval v2: no guarded component regressed in any seed. Stability: no flag in any seed. pass@k: the bootstrap interval at the largest $k$, against both the start and the control.
- **Correctness checks:** `pytest labs/module-14/project` (your objective library passes the derivation tests); lesson 14.1's tests; the run resumes exactly (stop at half, restart, compare logs); the control arm's evaluation used the strict verifier.
- **Fallback evidence:** lesson 14.2's comparison, Yue et al.'s Figure 2, ScaleRL's Figure 5, labelled as published.
- **Limits:** model size, steps, seeds; intervals over problems within a seed; GSM8K's contamination risk for Qwen models (Module 18 returns to this; OLMo-2-0425-1B is the documented open-data alternative).

<details>
<summary>What the build's free-CPU run gave (compare after your own report)</summary>

Measured 2026-10-07 on the build laptop (CPU, other jobs running): `run_project.py --objective cispo` with lr $3 \times 10^{-4}$, 6 runs × 200 steps (2 epochs × 4 minibatches), plus Eval v2 and 256 samples on 200 problems per final policy, 18.7 minutes. SFT start: held-out accuracy 0.220, pass@1 0.206, pass@256 0.990.

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| CISPO final accuracy | 0.493 | 0.363 | 0.367 |
| random control final accuracy | 0.000 | 0.000 | 0.000 |
| CISPO stability flags | collapse (drawdown 0.14) | collapse, ratio blow-up (74) | collapse, entropy collapse, ratio blow-up (101) |
| CISPO Eval v2 | FAIL: subtraction −0.15, SFT loss −0.42, format −0.02 | FAIL: subtraction −0.05, SFT loss −0.29 | FAIL: subtraction −0.08, SFT loss −0.35, format-and-correct −0.06 |
| CISPO pass@1 − SFT | +0.258 [+0.216, +0.300] | +0.146 [+0.107, +0.188] | +0.143 [+0.102, +0.182] |
| CISPO pass@256 − SFT | −0.035 [−0.070, −0.005] | −0.080 [−0.120, −0.045] | −0.060 [−0.100, −0.025] |
| crossover; solved only by SFT / only by CISPO | k = 16; 9 / 2 | k = 16; 16 / 0 | k = 16; 13 / 1 |
| profile interval for A | [0.437, 1.000] | [0.342, 1.000] | [0.300, 1.000] |

CISPO minus control, paired by seed: +0.408 [+0.224, +0.592]. The verifier's signal is needed here, because the random reward drove every control seed to 0 (KL to the start above 8 nats, every Eval v2 component regressed). The decision rule's other parts fail. Reusing each batch for 2 epochs × 4 minibatches at this learning rate broke the stability thresholds in every seed: held-out accuracy peaked and fell by 0.11–0.14, and raw ratios reached 74–101 (CISPO's weights stay capped, but the policy moves anyway). Retention failed the guards in every seed. pass@256 fell in all three seeds with intervals below 0, with a crossover at $k = 16$, unlike lesson 14.4's gentler one-epoch GRPO run. The defensible conclusion: this configuration raises pass@1, narrows coverage and costs retention, and it should not become a default without less data reuse or a KL term. That is a well-run negative result, and it scores as well as a positive one would.

</details>

## Debugging task

`labs/module-14/project/buggy_objectives.py` is a colleague's own GRPO, GSPO and CISPO. Their message is at the top of the file: all three "train, the losses go down", but GSPO "barely learns on the long-answer task", CISPO "learns slower than the paper says" and GRPO's "ratios get huge after a few minibatches". They ask you to tune hyperparameters. There is one bug per objective. For each one, name the derivation it breaks, the test that isolates it and the training metric that shows it, and say why tuning would not have fixed it.

```bash
OBJ_HOOKS=buggy pytest labs/module-14/project        # the derivation tests against their library
python labs/module-14/project/buggy_run.py           # their objectives in the loop, next to lesson 14.1's runs
pytest labs/module-14/project                        # the course library: all pass
```

<details>
<summary>Hint</summary>

Write the gradient weight each of their functions produces for one token, the way lesson 14.1's worked example does. For GSPO, compute the sequence ratio of a 2-token and a 6-token response whose tokens all have ratio 1.0002. For CISPO, ask which tensor the gradient flows through.

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07 on the build laptop (CPU, other jobs running; `buggy_run.py` 4.5 minutes; the course runs are reused from lesson 14.1). Same settings for every row: SFT start (0.21), seed 0, 80 steps, 2 epochs × 4 minibatches.

| Objective | held-out accuracy | clip fraction | max ratio | KL to start |
|---|---|---|---|---|
| GRPO, course | 0.357 | 0.046 | 18.3 | 0.277 |
| GRPO, colleague | 0.233 | 0.046 | 30.8 | 0.569 |
| GSPO, course | 0.353 | 0.453 | 1.8 | 0.239 |
| GSPO, colleague | 0.400 | 0.458 | 4.1 | 0.239 |
| CISPO, course | 0.367 | 0.025 | 13.6 | 0.285 |
| CISPO, colleague | 0.277 | 0.029 | 14.8 | 0.291 |

`OBJ_HOOKS=buggy pytest labs/module-14/project` fails exactly four tests.

**GRPO: `torch.maximum` instead of `torch.minimum`.** PPO's min is the pessimistic bound. With the max, a token whose ratio passed $1 + \epsilon$ with $A > 0$ takes the *unclipped* branch and keeps being pushed, and one below $1 - \epsilon$ with $A < 0$ keeps being pushed down: the trust region works backwards. Isolating test: `test_grpo_clipping_is_pessimistic`. In training: the KL to the start doubled (0.57 against 0.28), the maximum ratio grew, and accuracy barely moved from the start (0.23). The clip fraction looks normal because it is computed from the same flag in both versions. No learning rate fixes this: it changes how far the policy runs, not whether the bound holds.

**GSPO: the sequence ratio is not length-normalised.** `exp(sum of log-ratios)` is the product of token ratios, so $\log s_i$ is $|y_i|$ times too large. Every token's weight is $s_i A_i$, missing the $1/|y_i|$. Isolating tests: `test_gspo_ratio_is_length_normalised` (two responses of 2 and 6 tokens with all token ratios 1.0002 give sequence ratios 1.0004 and 1.0012 instead of 1.0002 for both: the short one sits on the $4 \times 10^{-4}$ bound and the long one is three times past it) and the on-policy REINFORCE test. In the toy run this bug is invisible and even scored higher (0.400 with one seed): responses of 2–4 tokens make it a 2–4× larger step with almost the same clipping. On the colleague's long-answer task $|y|$ is in the hundreds, so nearly every sequence is clipped and the rest get hundreds of times the intended weight, hence "barely learns". Only the unit test finds this bug at toy scale. A training curve on short answers would reward it.

**CISPO: the weight is not detached and multiplies $A$ instead of $A \log\pi$.** `clamp(exp(logp - old), max=1.28) * A` is a surrogate whose gradient is $r A \nabla\log\pi$ below the cap and **zero** above it. That is PPO clipping without the min, and exactly what CISPO exists to avoid. Isolating test: `test_cispo_keeps_gradient_past_the_cap`. In training: accuracy 0.28 against 0.37 with an almost identical "clip fraction". The diagnostic counts capped weights in both versions, but in the buggy one capped means silenced. Tuning `eps_high` changes how many tokens are silenced, not the fact that they are.

Which detector found what: GRPO's bug shows in the KL and ratio columns once you know a healthy run's values. CISPO's shows only as slower learning with normal-looking diagnostics. GSPO's is visible only through its derivation test on short-answer data. All three survive a "the loss goes down" check.

</details>

## Written defence

One to two pages, answering:

1. Why this objective? Cite your lesson 14.2 result with its interval, and say what a reviewer could still attribute the choice to.
2. Your control arm: how large was its pass@1 gain, and what does that say about how much of your RL gain the verifier's signal explains?
3. Which stability metric came closest to its threshold, and what would you change for a run ten times longer?
4. Did any Eval v2 component regress or come close to its guard? What is the smallest regression your comparison could have detected?
5. What does your pass@k analysis say about capability, and what does it not say? Name the evaluation change (benchmark, $n$, temperature) that could change your conclusion.
6. Your run's profile interval for $A$: what would it take, in compute and evaluation, to identify it?
7. Each planted bug: which test caught it, which metric showed it, and why tuning could not have fixed it.

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the control question, guards, stability thresholds and pass@k rule written before any run |
| 2 | Controls | run cards identical except the reward; the control scored with the strict verifier |
| 3 | Axis and budget parity | equal samples and updates verified from the logs; evaluation compute counted |
| 4 | Correctness | `pytest labs/module-14/project` output included; exact resume shown; the three planted bugs found with their isolating tests |
| 5 | Uncertainty | seed-level interval vs the control; bootstrap intervals for pass@k; profile interval for $A$ |
| 6 | Conclusion matches evidence | no claim of capability gain without the large-$k$ interval; no asymptote claim from an unbounded profile |
| 7 | Limits | scale, steps, seeds, the Qwen spurious-reward caveat, GSM8K contamination risk |
