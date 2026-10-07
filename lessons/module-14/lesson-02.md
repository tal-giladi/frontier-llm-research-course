---
id: "14.2"
module: 14
minutes: 40
practice_minutes: 120
prerequisites: ["14.1", "12.4", "01.3", "01.4"]
objectives:
  - Design an objective comparison that holds prompts, rollouts per step, KL setting, data reuse and compute fixed, and name every setting that comes bundled with each objective.
  - Give every objective the same, stated tuning budget and select its hyperparameters on training data only.
  - Define stability metrics and their thresholds before the runs, and report them per seed next to the task metric.
  - Decide with paired seed-level intervals and a random-reward control arm, and score every final policy with Eval Suite v2.
  - Read a published objective comparison and identify which of its differences a controlled comparison would remove.
volatility: concept
sources:
  - title: "Khatri et al., The Art of Scaling Reinforcement Learning Compute for LLMs (sections 3-4: leave-one-out comparisons; Figure 5a: loss types; Figure 8a: run-to-run variance)"
    url: https://arxiv.org/abs/2510.13786
  - title: "Zheng et al., Group Sequence Policy Optimization (section 5.1: GSPO vs GRPO setup and ranges)"
    url: https://arxiv.org/abs/2507.18071
  - title: "MiniMax, MiniMax-M1 (section 3.1, Figure 2: CISPO vs DAPO vs GRPO)"
    url: https://arxiv.org/abs/2506.13585
  - title: "Shao et al., Spurious Rewards: Rethinking Training Signals in RLVR (section 2)"
    url: https://arxiv.org/abs/2506.10947
  - title: "Google Research, Deep Learning Tuning Playbook (tuning budgets and comparisons)"
    url: https://github.com/google-research/tuning_playbook
  - title: "Zhou et al., Instruction-Following Evaluation for Large Language Models (IFEval)"
    url: https://arxiv.org/abs/2311.07911
last_verified: "2026-10-07"
---

# 14.2 · A controlled objective comparison

Every objective paper in lesson 14.1 reports beating a baseline, and each comparison changes several things at once: the objective, its clip range, its normaliser, its batch rule and often the learning rate, against a baseline tuned by someone who wanted the new method to win. This lesson builds the comparison the papers did not run. The prompts, rollouts per step, KL setting, data reuse and compute are the same for every arm. Every objective gets the same tuning budget, stability is defined before the runs, decisions use paired seeds, and a random-reward control arm checks that the verifier's signal is doing the work. Every final policy goes through Eval Suite v2.

## Why this matters at a frontier lab

Switching the RL objective of a production recipe is a decision about a run that may cost thousands of GPU-hours, and it is usually made from a figure in someone else's paper. ScaleRL's leave-one-out design is the published model of how to do this: start from one strong recipe, change one component at a time, and compare at equal compute with the noise margin stated (sections 3–4). It ranks CISPO and GSPO above DAPO on the fitted asymptote (Figure 5a), with run-to-run variance of ±0.015 in that asymptote (Figure 8a). Even so, it is one lab, one data mix, two model families. When your model, data or length budget differs, you rerun the comparison, and at a smaller scale the danger is the opposite one: differences within noise read as rankings. A research engineer has to produce a result that a sceptical reviewer cannot explain away by tuning effort, by a lucky seed, or by "the model would have learned that from any reward".

## The idea

### What "the same" has to mean

Lesson 14.1 showed that an objective comes as a bundle: a ratio and clipping rule (the objective proper) plus paired choices (normaliser, advantage scale, batch filter, clip range). A controlled comparison has to say which of these it varies. This lesson's choice is to compare the **bundles as the papers define them**, because that is what a team adopts, and to hold everything outside the bundles fixed:

| Held fixed for every arm | Value in the CPU lab | Why it matters |
|---|---|---|
| start checkpoint, prompt stream per seed | Module 12 SFT start; the loop's seeded prompt sampler | identical first batches across arms, so comparisons pair by seed |
| rollouts per step | 16 prompts × 8 responses | equal samples is the comparison axis |
| data reuse | 1 epoch × 4 minibatches | on-policy minibatches cannot separate objectives (14.1) |
| KL setting | off ($\beta = 0$), no entropy bonus | GRPO's paper includes a KL term; keeping it would change two things |
| temperature, token budget, gradient clip | 1.0, 8 tokens, 1.0 | sampling and update size |
| evaluation | 300 held-out problems, fixed sampling seed; Eval v2 pins | paired items |

Only the bundle and its tuned learning rate change. Dynamic sampling is DAPO's batch rule, so DAPO's arm filters zero-variance groups and the others keep them. The rollout count is the same, but DAPO computes gradients on fewer groups. State that, because it is part of what DAPO is.

### Tuning budget

An untuned new method against a tuned baseline (or the reverse) is the most common way objective comparisons go wrong. The rule here: each objective gets **the same budget** (3 learning rates × 1 tuning seed × 60 steps), selected by **training reward over the last 25 steps** on the tuning seed, which is never reused in the comparison. Held-out data are not touched, so the comparison's evaluation stays clean. Ties within 0.005 go to the smaller learning rate, and runs that collapse are not eligible. Clip ranges stay at each paper's values: tuning them would multiply the budget by five, and the papers' values are part of the bundles. Say so, because a reviewer will ask.

### Stability, defined before the runs

"More stable" is the claim GSPO and CISPO make, and it needs a definition before any run. From each run's log (`frontierlab.rlscale.stability`):

- **entropy drop** — mean entropy over the first 10% of steps minus the last 10% (collapse threshold 0.25 nats);
- **gradient spikes** — steps whose pre-clip norm exceeds 5× the median of the previous 20 (threshold 3 per run), and the 99th-percentile norm;
- **ratio blow-up** — the largest importance ratio any minibatch saw (threshold 50);
- **drawdown** — the largest fall of held-out accuracy from its running maximum (collapse at 0.10);
- clip fraction and exact KL to the start, reported but not thresholded: they mean different things per objective (14.1).

The thresholds are judgment calls. Writing them down first is what turns them from a story told after the runs into a test.

### The decision rule and the control arm

The primary metric is held-out sampled accuracy at the last step. For each objective and seed, subtract the baseline arm's (GRPO's) value at the same seed and form a 95% t-interval over the per-seed differences: lesson 01.4's paired design. With 3 seeds the t-multiplier is $t_{0.975, 2} = 4.30$, so only large, consistent differences will clear the bar. That is the honest power of a 3-seed comparison.

The **control arm** is GRPO with a random reward: Bernoulli(0.5) per response, independent of what the response says, scored on held-out data with the strict verifier like every other arm. Shao et al. report that on Qwen2.5-Math-7B a random reward raised MATH-500 by 21.4 points against 29.1 for the true reward (section 2), so for Qwen-family models "reward went up and accuracy went up" proves nothing without it. The rule applies the control first. An arm whose interval against the control does not lie above 0 is reported as **below control**, and nothing about objectives is concluded from it. Otherwise the verdict against GRPO is better, worse or inconclusive.

### Eval v2 and what each arm cost

Each final policy is scored with Eval Suite v2 (lesson 12.4) against the SFT start, with guards of 0.02 on subtraction, instruction following and SFT-data loss. An objective that wins on the task while failing the retention guard in some seeds has not won. Report the task verdict and the guard result per seed side by side.

## Worked example

**A paired interval with 3 seeds.** CISPO's final accuracies are 0.37, 0.35, 0.40 and GRPO's at the same seeds 0.34, 0.33, 0.36. The differences are 0.03, 0.02, 0.04, with mean 0.030 and sample standard deviation 0.010. The half-width is $4.30 \times 0.010/\sqrt{3} = 0.0248$, so the interval is $[0.005, 0.055]$: "better than GRPO", provided the control check passed. Change the third seed to 0.36 vs 0.36. The differences are 0.03, 0.02, 0.00, with mean 0.0167, sd 0.0153 and half-width 0.038, giving $[-0.021, 0.055]$: inconclusive, although every seed is at least as good.

**Seeds needed.** If the seed-to-seed standard deviation of the paired difference is 0.02 and you want to detect a 0.02 difference, the interval half-width $t_{0.975, n-1} \cdot 0.02/\sqrt{n}$ must drop below 0.02. That needs $t/\sqrt{n} < 1$: $n = 3$ gives $4.30/1.73 = 2.48$, $n = 5$ gives $2.78/2.24 = 1.24$, $n = 7$ gives $2.45/2.65 = 0.92$. So you need 7 seeds per arm.

**The control check.** The control's final accuracies are 0.22, 0.25, 0.21. CISPO minus control is 0.15, 0.10, 0.19, with mean 0.147, sd 0.045, half-width 0.112: $[0.035, 0.259]$, above 0, so the check passes.

**Tuning selection.** DAPO's tuning scores are 0.32 at $10^{-4}$, 0.398 at $3\times10^{-4}$ and 0.401 at $10^{-3}$, and the last run had a drawdown of 0.12, a collapse. That run is ineligible, so $3 \times 10^{-4}$ is selected. Without the collapse rule the tie rule would also pick $3 \times 10^{-4}$ (0.398 is within 0.005 of 0.401).

## Shapes and cost

| Item | Count | CPU lab | Main path (PROJECTED, pending the pilot) |
|---|---|---|---|
| tuning runs | 5 objectives × 3 lr × 1 seed | 60 steps each | 60 steps each, 30–50 s/step: 7.5–12.5 GPU-hours |
| comparison runs | (5 + 1 control) × seeds | 3 seeds × 120 steps | 2 seeds × 200 steps: 20–33 GPU-hours |
| Eval v2 per final policy | 18 (CPU) / 12 (main) | seconds each | 0.3–0.5 GPU-hours each: 3.6–6 GPU-hours |
| per-step tensors | as lesson 14.1 | (128, 8) | (256, 512) |

Main-path total, PROJECTED: about 31–52 H100-hours, USD 62–156 at USD 2–3 per hour. It comes from the formula (runs × steps × s/step + evaluations × cost each) with lesson 12.2's 30–50 s/step for P = 32, G = 8, 512 new tokens on Qwen3-1.7B-Base. The tuning budget is a quarter of the total, and it is what makes the comparison defensible.

## Build it

```python
from frontierlab.rlscale import runner
from frontierlab.rlscale.stability import run_stability

arms = {"cispo": {"objective": "cispo", "over": {"lr": 3e-4}},
        "control-random": {"objective": "grpo", "control": "random", "over": {"lr": 3e-4}}}
res = runner.run_arms(base_cfg, arms, seeds=[0, 1, 2], root="runs/m14/l142/compare")
run_stability("runs/m14/l142/compare/cispo-s0")      # entropy drop, spikes, ratio max, drawdown, ...
```

`runner.run_arms` gives every arm the objective's paired settings and skips finished runs. Interrupted runs resume exactly (the Module 12 loop's checkpoints). The control reward draws from the loop's checkpointed RNG, so a control run is reproducible too. Held-out evaluation is forced to the strict verifier for control arms (`runner.patched_loop`). The lab's `compare_lab.py` runs the whole design. Your `select_lr`, `paired_interval`, `verdict` and `stability_flags` make the decisions, and the script prints them next to the raw per-seed values, so a reader can redo the arithmetic. Correctness checks: your TODO tests; `test_rlscale.py` (the control rewards' rates; the patches are undone after each run; the stability metrics on a constructed log).

**What frameworks would hide.** In verl v0.9.1 the five bundles are `loss_mode` (`vanilla`, `gspo`, `cispo`) plus `loss_agg_mode` and `clip_ratio_low/high`, and DAPO's dynamic sampling and overlong shaping are separate recipe settings. A comparison of verl configs that differ in `loss_mode` only compares the policy-loss rule under verl's default token-mean aggregation, which is not the bundles the papers describe. Neither is wrong, but the write-up has to say which one it compared.

## What the evidence says

- **Published objective comparisons are not controlled in this sense: PUBLICLY DOCUMENTED** from their own setups. GSPO compares against GRPO with ranges 0.2/0.27 on its own cold-start model (section 5.1); MiniMax-M1 compares CISPO with DAPO and GRPO on Qwen2.5-32B zero-RL (Figure 2). Each tunes its own method. ScaleRL's leave-one-out runs come closest (one recipe, one component changed, 16,000 GPU-hours each, stated noise margin).
- **Ranking at scale (ScaleRL): CISPO ≳ GSPO > DAPO on the asymptote: PROMISING**, one lab's runs.
- **Random-reward controls are necessary for Qwen-based RLVR claims: PUBLICLY DOCUMENTED** (Shao et al.); REASONABLE INDUSTRY PRACTICE in this course.
- **Equal tuning budgets per arm: REASONABLE INDUSTRY PRACTICE** (Google's tuning playbook makes the same point for optimizer comparisons, and Module 7 applied it there).
- **Course measurement (free CPU, 2026-10-07, 3 seeds):** every objective beat a random-reward control (which collapsed the toy skill). DAPO was indistinguishable from GRPO, CISPO slightly worse (−0.030 [−0.047, −0.013]), GSPO inconclusive, and Dr. GRPO worse at the smaller learning rate its tuning selected. It was also the only arm that passed the Eval v2 retention guard in every seed. No stability threshold was broken. Details in the lab's results box.

## Lab

**Folder:** [`labs/module-14/lesson-02/`](../../labs/module-14/) · **Time:** about 2 hours (most of it unattended) · **Pass check:** `pytest labs/module-14/lesson-02` passes; `compare_lab.py` prints the tuning table, the decision table and the stability/Eval v2 table; your write-up states the decision for each objective under the rule, and the smallest difference your design could have detected.

### Experiment contract

- **Question:** at equal samples, tuning budget and KL setting, does any of DAPO, Dr. GRPO, GSPO or CISPO (each as its paper bundles it) reach a different held-out accuracy from GRPO after 120 steps, is any of them more or less stable by the pre-stated metrics, and does each beat a random-reward control? Decision informed: the objective for the Module 14 project and its default learning rate.
- **Hypothesis:** no objective differs from GRPO beyond the 3-seed interval at this scale; GSPO shows the highest clip fraction and the smallest ratio maxima; every real-reward arm beats the control. Status: rankings reported at large scale (ScaleRL Figure 5a); may not appear here, where responses are 2–4 tokens and the policy has 308k parameters.
- **Baseline:** GRPO at its own tuned learning rate (same budget as every arm).
- **Changed variable:** the objective bundle and its tuned learning rate. **Controlled:** see the table above (start, prompt stream, 16 × 8 rollouts, 1 epoch × 4 minibatches, KL off, temperature, budget, evaluation pins); seeds 0–2; tuning seed 100.
- **Comparison axis:** equal sampled responses (and equal updates). Equal wall-clock would favour no objective here, because the losses cost the same.
- **Budget:** free CPU, measured runtime in the results box. Tuning: 3 runs per objective, 60 steps each.
- **Metrics and decision rule:** primary: final held-out sampled accuracy, paired by seed, 95% t-interval vs GRPO and vs the control; rule as in `verdict` (control first). Stability: the four thresholds in `lab.THRESHOLDS`, flagged per seed. Retention: Eval v2 guard (0.02) per seed. Noise floor: report the per-seed spread of GRPO's arm and the half-width of each interval.
- **Correctness checks:** lesson 14.1's objective tests; `test_rlscale.py`; your TODO tests; the run cards show identical configurations except objective, lr and the objective's paired settings.
- **Fallback evidence:** ScaleRL Figures 5a and 8a, labelled as published.
- **Limits:** 3 seeds, 120 steps, toy task and model, clip ranges untuned, one tuning seed, item intervals in Eval v2 within a seed.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 14 pilot | `python labs/module-14/lesson-02/compare_lab.py --variant main --print` prints the 15 tuning runs and the 12 comparison runs (`frontierlab.rlscale.hf_rl`, Qwen3-1.7B-Base, GSM8K), then score each final policy with `labs/module-12/lesson-04/eval_main.py`. **PROJECTED:** 31–52 GPU-hours (table above) |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --print` (Qwen3-0.6B-Base, 8 prompts, 256 new tokens, fp32); budget about 20 T4 sessions. Run the tuning for GRPO and CISPO only and say so |
| Free CPU | laptop; measured 35 minutes for the 33 runs and 18 Eval v2 scorings (other jobs running) | the steps below |

### Steps

1. **Write your contract** (copy the one above and change what you disagree with) before running anything.
2. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-14/lesson-02`.
3. **Tune:** `python labs/module-14/lesson-02/compare_lab.py --phase tune`. Check that each selected learning rate is the one your rule should pick.
4. **Compare:** `--phase compare` (resumes the tuning results). Apply the rule. For every "inconclusive", compute how many seeds would be needed to resolve a difference of the size you observed (worked example).
5. **Write up:** the decision table, the stability table with flags, the Eval v2 guards per seed, the control check, and a paragraph on which published claim from lesson 14.1 your result is and is not evidence about.

<details>
<summary>Hint for TODO 2</summary>

`scipy.stats.t.ppf(0.975, n - 1)` gives the multiplier; use the sample standard deviation (divide by $n - 1$). Return NaN bounds for one seed rather than a zero-width interval.

</details>

<details>
<summary>Hint for TODO 3</summary>

Check the control first: `vs_control[1] > 0` must hold (a NaN bound fails this comparison, which is what you want). Then compare `vs_baseline[1] > 0` and `vs_baseline[2] < 0`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, another module's jobs running): 15 tuning runs and 18 comparison runs plus 18 Eval v2 scorings, 35 minutes in all. The SFT start scores 0.203 on Eval v2's `add_pass1`.

**Tuning** (training pass rate, last 25 of 60 steps, tuning seed 100):

| Objective | lr $10^{-4}$ | lr $3 \times 10^{-4}$ | lr $10^{-3}$ | selected |
|---|---|---|---|---|
| GRPO | 0.326 | 0.342 | 0.240 | $3 \times 10^{-4}$ |
| DAPO | 0.330 | 0.352 | 0.284 | $3 \times 10^{-4}$ |
| Dr. GRPO | 0.323 | 0.323 | 0.222 (collapsed) | $10^{-4}$ (tie → smaller) |
| GSPO | 0.289 | 0.333 | 0.192 | $3 \times 10^{-4}$ |
| CISPO | 0.325 | 0.344 | 0.155 | $3 \times 10^{-4}$ |

**Comparison** (final held-out sampled accuracy, seeds 0, 1, 2; 95% t-intervals of the paired differences):

| Arm | per seed | vs GRPO | vs random control | verdict |
|---|---|---|---|---|
| GRPO | 0.410, 0.357, 0.397 | — | +0.320 [+0.252, +0.388] | baseline (beats control) |
| DAPO | 0.420, 0.357, 0.400 | +0.004 [−0.008, +0.017] | +0.324 [+0.245, +0.404] | inconclusive |
| Dr. GRPO | 0.310, 0.273, 0.320 | −0.087 [−0.117, −0.057] | +0.233 [+0.175, +0.291] | worse than baseline |
| GSPO | 0.363, 0.343, 0.350 | −0.036 [−0.083, +0.012] | +0.284 [+0.258, +0.311] | inconclusive |
| CISPO | 0.373, 0.327, 0.373 | −0.030 [−0.047, −0.013] | +0.290 [+0.225, +0.355] | worse than baseline |
| random control | 0.067, 0.067, 0.070 | −0.320 | — | control |

**Stability and retention** (means over seeds; flags and Eval v2 guard per seed):

| Arm | entropy drop | grad p99 | max ratio | clip | KL | flags | Eval v2 guard |
|---|---|---|---|---|---|---|---|
| GRPO | 0.188 | 4.4 | 3.3 | 0.008 | 0.24 | none | FAIL, FAIL, FAIL |
| DAPO | 0.184 | 9.7 | 2.7 | 0.009 | 0.25 | none | FAIL, PASS, FAIL |
| Dr. GRPO | 0.117 | 0.7 | 1.6 | 0.001 | 0.15 | none | PASS, PASS, PASS |
| GSPO | 0.169 | 3.6 | 1.4 | 0.314 | 0.23 | none | FAIL, FAIL, FAIL |
| CISPO | 0.182 | 4.0 | 3.3 | 0.006 | 0.26 | none | FAIL, FAIL, FAIL |
| random control | −0.005 | 3.2 | 2.9 | 0.006 | 1.38 | collapse (seed 2) | FAIL ×3 (every component) |

Reading it under the rule: every real-reward arm beat the control by a wide margin. The random reward did not "elicit" anything in this toy policy; it destroyed the skill (0.07, far below the 0.20 start) and every Eval v2 component. DAPO is indistinguishable from GRPO. CISPO is slightly but consistently worse, $-0.030$ $[-0.047, -0.013]$. GSPO is inconclusive. Dr. GRPO is clearly worse, and the tuning table says why. Its constant normaliser makes the loss about 2.4 times smaller here (mean response length 3.4 tokens against a constant of 8; lesson 12.3), its $10^{-3}$ run collapsed, and the tie rule picked $10^{-4}$. It ran at a smaller effective step than everyone else, so this is a statement about the tuning grid as much as about the objective. A grid scaled per objective (or a wider one) is the fix to propose. The same smaller step is why Dr. GRPO alone passed Eval v2 in all three seeds: it moved least (KL 0.15). GRPO, GSPO and CISPO failed the retention guard in every seed, mostly on the SFT-data loss and subtraction, as lesson 12.4 found for GRPO without KL. No arm broke a stability threshold. GSPO's clip fraction (0.31) and smallest ratio maxima (1.4) are the 14.1 signature again, not instability.

What this is evidence about: with 2–4-token responses, one epoch and three seeds, the objectives that keep PPO-style token clipping (GRPO, DAPO) did at least as well as the ones designed for long responses and MoE (GSPO) or for rare tokens (CISPO). It is not evidence against ScaleRL's ranking, which concerns asymptotes of 16,000-GPU-hour runs on long reasoning traces.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-14/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-14/lesson-02`.

</details>

## Common mistakes

- **Tuning the new objective and not the baseline** (or tuning both on held-out data). Give each the same budget and select on training reward.
- **Comparing with one minibatch per batch.** Then the objectives' gradients coincide (14.1), and the experiment compares normalisers.
- **Keeping GRPO's KL term in one arm.** Then two things differ. Hold the KL setting fixed.
- **Defining stability after looking at the curves.** Write down the metrics and thresholds first.
- **No control arm, or a control scored with its own reward.** Score the control on held-out data with the strict verifier like every other arm.
- **Reading three seeds that all point one way as a result.** With $t_{0.975,2} = 4.30$ they often do not clear the interval. Report the interval, and the seeds needed.

## References

- D. Khatri et al., *The Art of Scaling Reinforcement Learning Compute for LLMs*, 2025, sections 3–4, Figures 5a and 8a. https://arxiv.org/abs/2510.13786
- C. Zheng et al., *Group Sequence Policy Optimization*, 2025, section 5.1. https://arxiv.org/abs/2507.18071
- MiniMax, *MiniMax-M1*, 2025, section 3.1, Figure 2. https://arxiv.org/abs/2506.13585
- R. Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, 2025, section 2. https://arxiv.org/abs/2506.10947
- Google Research, *Deep Learning Tuning Playbook*. https://github.com/google-research/tuning_playbook
- J. Zhou et al., *Instruction-Following Evaluation for Large Language Models*, 2023. https://arxiv.org/abs/2311.07911
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[14.3 · Rollout systems and staleness](lesson-03.md)
