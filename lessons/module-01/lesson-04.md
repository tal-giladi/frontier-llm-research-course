---
id: "01.4"
module: 1
minutes: 35
practice_minutes: 90
prerequisites: ["01.1", "01.3"]
objectives:
  - Measure the seed-to-seed noise floor of a training recipe and separate it from evaluation sampling noise.
  - Compare two models with a paired bootstrap over fixed evaluation windows, and explain what that interval does and does not cover.
  - Build a seed-level interval for a claim about a method, paired by seed and unpaired, and read it against a pre-stated decision rule.
  - Correct for multiple comparisons with Holm's procedure.
  - Compute the minimum detectable effect for a planned ablation and choose the number of seeds before running it.
volatility: concept
sources:
  - title: "Evan Miller — Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (sections 2–5: CLT standard errors, clustering, variance reduction, paired differences, power)"
    url: https://arxiv.org/abs/2411.00640
  - title: "Philipp Koehn — Statistical Significance Tests for Machine Translation Evaluation (paired bootstrap resampling), EMNLP 2004"
    url: https://aclanthology.org/W04-3250/
  - title: "Sture Holm — A Simple Sequentially Rejective Multiple Test Procedure, Scandinavian Journal of Statistics 6(2):65–70, 1979"
    url: http://www.jstor.org/stable/4615733
  - title: "Kimi K2: Open Agentic Intelligence (section 2.3: 64 vs 128 heads ablation)"
    url: https://arxiv.org/abs/2507.20534
last_verified: "2026-10-03"
---

# 01.4 · Uncertainty

Two training runs that differ only in their random seed do not end at the same loss. Any difference you measure between two methods is the true effect plus this seed noise plus the noise of which evaluation items you happened to score. This lesson measures both kinds of noise, shows how pairing removes the evaluation-item part and why it cannot remove the seed part, adds the corrections you need when you test several things at once, and turns the noise floor into the question every contract asks before the runs: how many seeds do I need to see the effect I care about?

## Why this matters at a frontier lab

Architecture and recipe ablations at a fixed budget often produce differences of a few hundredths of a nat. The Kimi K2 report (section 2.3), for example, justifies 64 instead of 128 attention heads with a validation-loss ablation in which 128 heads were only about 0.5–1.2% better, small next to the inference cost they would add. Whether a gap like that is real depends entirely on the noise floor at the scale the ablation ran. A lab that decides on single-seed differences will adopt noise about as often as it adopts improvements, and every adopted false positive becomes part of the baseline for the next round. The noise floor of Baseline-0, measured in this module's project, sets the seed count of every Stage B experiment in this course.

## The idea

### Three sources of noise

Write the score of a run as

$$\text{score} = \mu_{\text{method}} + \varepsilon_{\text{seed}} + \varepsilon_{\text{eval}}$$

- $\mu_{\text{method}}$ is what you want: the expected score of the method over all seeds and all evaluation items.
- $\varepsilon_{\text{seed}}$ is training noise: initialisation and data order (both set by `--seed` in the course loop), and nondeterministic kernels (lesson 01.5). Its spread across seeds is the **seed standard deviation** $\sigma_{\text{seed}}$.
- $\varepsilon_{\text{eval}}$ is evaluation sampling noise: you score $n$ windows or questions, not the whole distribution. Its size is the **standard error** of the item mean, $\text{SE} = s_{\text{items}} / \sqrt{n}$, where $s_{\text{items}}$ is the standard deviation of per-item scores (Miller 2024, section 2.1).

A difference between two single runs carries both noises from both runs.

### Pairing removes shared item difficulty

Some validation windows are hard (code, tables, unusual names) and some are easy, for every model. If two models are scored on the **same** windows, the per-window difference $d_i = a_i - b_i$ cancels the shared difficulty. The standard error of the mean difference is

$$\text{SE}_{\text{paired}} = \frac{s_d}{\sqrt n}, \qquad s_d^2 = s_a^2 + s_b^2 - 2\rho\, s_a s_b,$$

where $\rho$ is the correlation of the two models' per-window losses. For two models trained on the same data, $\rho$ is typically above 0.9, and $\text{SE}_{\text{paired}}$ is many times smaller than the unpaired $\sqrt{(s_a^2 + s_b^2)/n}$ (Miller, section 4.2). The **paired bootstrap** (Koehn 2004) estimates the interval without assuming normality: resample the $n$ items with replacement many times, compute the mean of $d$ on each resample, and take the 2.5% and 97.5% quantiles. `frontierlab.stats.paired_bootstrap` does exactly this, with the windows from `frontierlab.evals.window_losses` in a fixed order.

### What a paired window interval covers

A paired bootstrap over windows treats the two **trained models** as fixed. Its interval answers: "on this evaluation distribution, is model A better than model B?" It says nothing about whether the **method** behind A is better than the method behind B, because a different seed would have produced different models. With 256 windows the paired interval is often narrower than the seed noise, so two seeds of the *same* method can look "significantly different" by it. The lab's A/A check shows this directly.

For a claim about a method, the replicate is the seed:

- **Paired by seed**: if arm A and arm B with seed $i$ share initialisation and data order (the course loop's `--seed` sets both), the per-seed difference $D_i = \bar b_i - \bar a_i$ cancels part of the seed noise. Interval: $\bar D \pm t_{n-1,0.975}\, s_D/\sqrt{n}$.
- **Unpaired** (Welch): $\bar b - \bar a \pm t_{\nu,0.975} \sqrt{s_a^2/n_a + s_b^2/n_b}$, with the Welch–Satterthwaite degrees of freedom $\nu$.

With 3 seeds, $t_{2,0.975} = 4.30$: small seed counts pay twice, through a large $s/\sqrt n$ and a large $t$.

### Multiple comparisons

Test 10 variants against the baseline at $\alpha = 0.05$, and if none of them does anything, the chance that at least one looks significant is $1 - 0.95^{10} = 40\%$. Holm's procedure (1979) controls the probability of any false rejection at $\alpha$: sort the $m$ p-values, multiply the $k$-th smallest by $m - k + 1$, take a running maximum so the adjusted values never decrease, and reject those still at or below $\alpha$. It is never less powerful than Bonferroni (multiply all by $m$). `frontierlab.stats.holm` implements it. Apply it whenever a contract makes more than one claim from the same experiment: several variants, several metrics, or several evaluation suites.

### Power and the minimum detectable effect

Before the runs, ask: if the true effect were $\delta$, how likely is my design to detect it? With $n$ seeds per arm, seed standard deviation $\sigma$, a two-sided test at level $\alpha$ and power $1 - \beta$, the normal approximation gives the **minimum detectable effect**

$$\text{MDE} = (z_{1-\alpha/2} + z_{1-\beta})\,\sigma\sqrt{2/n}.$$

At $\alpha = 0.05$ and 80% power, $z_{0.975} + z_{0.8} = 1.960 + 0.842 = 2.80$. Effects smaller than the MDE are likely to come out "inconclusive" under the decision rule of lesson 01.3. If the effect you care about is below the MDE, you need more seeds, a larger effect (more tokens, a larger model, a harder evaluation), or a design that pairs by seed; running the underpowered experiment anyway wastes the budget. Miller (section 5) gives the same reasoning for evaluation sets.

## Worked example

### Pairing, with four windows

Two models scored on the same four windows:

| Window | $a_i$ | $b_i$ | $d_i = a_i - b_i$ |
|---|---|---|---|
| 1 | 3.00 | 3.10 | −0.10 |
| 2 | 4.00 | 4.10 | −0.10 |
| 3 | 2.50 | 2.55 | −0.05 |
| 4 | 3.50 | 3.60 | −0.10 |

Means: $\bar a = 3.25$, $\bar b = 3.3375$, $\bar d = -0.0875$. Standard deviations: $s_a = 0.645$, $s_b = 0.665$, $s_d = 0.025$.

- Unpaired SE: $\sqrt{(0.645^2 + 0.665^2)/4} = \sqrt{0.2147} = 0.463$. A difference of $-0.0875$ is 0.19 SE: invisible.
- Paired SE: $0.025 / \sqrt 4 = 0.0125$. The same difference is 7 SE.

Pairing shrank the standard error by a factor of 37, because the windows' difficulty (spread 0.65) is shared and the models' disagreement (spread 0.025) is not.

### Seeds needed

The lab measured $\sigma_{\text{seed}} = 0.029$ nats for the toy model at 300 steps (see "What the evidence says"). For round numbers take $\sigma = 0.01$ nats:

- 2 seeds per arm: $2.80 \cdot 0.01 \cdot \sqrt{2/2} = 0.028$
- 3 seeds: $2.80 \cdot 0.01 \cdot \sqrt{2/3} = 0.0229$
- 5 seeds: $2.80 \cdot 0.01 \cdot \sqrt{2/5} = 0.0177$

To detect 0.01 nats you need $n \ge 2 \cdot (2.80 \cdot 0.01 / 0.01)^2 = 15.7$, so 16 seeds per arm. MDE falls only as $1/\sqrt n$: halving it costs four times the seeds.

### Holm, by hand

Four variants against one baseline, raw p-values $(0.01, 0.04, 0.03, 0.20)$, $m = 4$. Sorted: $0.01, 0.03, 0.04, 0.20$; multiplied by $4, 3, 2, 1$: $0.04, 0.09, 0.08, 0.20$; running maximum: $0.04, 0.09, 0.09, 0.20$. Back in the original order: $(0.04, 0.09, 0.09, 0.20)$. At $\alpha = 0.05$ only the first variant is a claim; without correction three would have been.

## Shapes and cost

- Per run, Eval v0 held-out loss is a vector of 256 per-window means, float64 on CPU after `window_losses` (each window: a (1, $T$) int64 slice, scored in batches of 16 on the model's device, per-token loss (B, $T-1$) float32 averaged per window). Keep it: without per-item scores you cannot pair.
- A bootstrap of $B = 10{,}000$ resamples over $n = 256$ items is a (10,000, 256) index array, 2.56M gathers: milliseconds in NumPy.
- The expensive part is seeds. Each extra seed is a full training run: at Baseline-0 size about 1.84 GPU-hours on one H100 at an assumed 30% MFU (PROJECTED, lesson 01.1). A 2-arm, 5-seed ablation is 10 runs, about 18 GPU-hours; at 3 seeds, 11 GPU-hours. The MDE table is how you decide whether the extra 7 hours buy a decision.
- Evaluation sampling noise is cheap to shrink: more windows cost only evaluation. Seed noise is not: only more runs shrink it. So spend evaluation until $\text{SE}_{\text{eval}}$ is well below $\sigma_{\text{seed}}$, then spend seeds.

## Build it

The shared code is `frontierlab.stats` (`summary`, `bootstrap_ci`, `paired_bootstrap`, `min_detectable_effect`, `holm`) and `frontierlab.evals.window_losses`. The lab adds the pieces you write yourself: the MDE and the seed count from it, an unpaired bootstrap to compare against the paired one, and seed-level t-intervals, paired and Welch. The tests check your intervals against SciPy's `ttest_rel` and `ttest_ind(equal_var=False)` confidence intervals and your MDE against the hand value above.

```python
from frontierlab.stats import paired_bootstrap, holm, summary
r = paired_bootstrap(new_window_losses, base_window_losses)    # same 256 windows, same order
print(r["mean_diff"], r["ci"])                                  # covers eval noise for THESE two models
s = summary(per_seed_means_of_base)                             # n, mean, std (ddof=1), sem
```

`labs/module-01/lesson-04/run_seeds.py` trains the toy preset with 5 seeds at learning rate 3e-3 and 3 seeds at 4e-3, each scored on the same 256 validation windows; `analyze.py` prints the noise floor, the MDE table, paired versus unpaired intervals for one seed pair, the paired window interval for every seed pair, seed-level intervals and Holm-adjusted p-values. It imports your `lab.py`, so it only runs once your TODOs are done (or with `LAB_TARGET=solution`).

## What the evidence says

- **ESTABLISHED (statistics):** standard errors of item means, paired differences and power analysis for evaluations (Miller 2024, sections 2, 4 and 5); the paired bootstrap for comparing systems on the same test items (Koehn 2004); Holm's step-down correction (Holm 1979).
- **REASONABLE INDUSTRY PRACTICE:** reporting seed variance for training ablations and choosing seed counts from a measured noise floor. Most technical reports do not report seed variance for their pretraining ablations, which is one reason to treat single ablation numbers in reports as weak evidence.
- **Course measurement, toy scale** (filled in below from the lab run): measured 2026-10-03 with `run_seeds.py` defaults (toy preset, 300 steps of 16 × 128 tokens, Data-v0 CPU size, 256 validation windows of 128 tokens; Windows 11 laptop, 16 threads, torch 2.14.1 CPU; 14 minutes for all 8 runs):

  | Quantity | Value |
  |---|---|
  | base, mean validation loss per seed 0–4 | 6.3016, 6.2864, 6.2337, 6.2524, 6.2904 |
  | seed std (5 seeds) | **0.0286** nats |
  | MDE for 2 / 3 / 5 seeds per arm | 0.080 / 0.066 / 0.051 nats; 33 seeds per arm to detect 0.02 |
  | seed 0 pair, lr4e-3 − base: unpaired window bootstrap | +0.0061, CI [−0.041, +0.054] |
  | seed 0 pair: paired window bootstrap | +0.0061, CI [+0.0007, +0.0114] (per-window loss std 0.276, std of differences 0.044) |
  | paired window CIs, seeds 1 and 2 | [+0.026, +0.038], [+0.003, +0.013] |
  | **A/A**: base seed 1 − base seed 0, paired window bootstrap | −0.0153, CI [−0.024, −0.007] |
  | seed-level, paired by seed (3 pairs) | +0.0154, CI [−0.020, +0.051] |
  | seed-level, unpaired Welch (5 vs 3 runs) | +0.0164, CI [−0.067, +0.100] |
  | correlation of per-seed losses across arms | +0.94 |

  Four lessons in one table. Pairing over windows shrank the interval about ninefold. The A/A row shows two seeds of the *same* recipe "significantly" different by a window bootstrap — the reason a window interval cannot support a method claim. Pairing by seed narrowed the seed-level interval by 2.3× compared with Welch, because seed effects are shared across arms (correlation 0.94). And under the contract's rule the decision is **inconclusive**: the paired interval's lower bound, −0.0204, is just below −0.02, although all three seed pairs point the same way (lr4e-3 slightly worse). At this tiny scale and short budget the seed std is large; Baseline-0's, measured in the module project, will be different, and nothing here says anything about learning rates at scale.

## Lab

**Folder:** [`labs/module-01/lesson-04/`](../../labs/module-01/) · **Time:** about 90 minutes (15 of them unattended) · **Pass check:** `pytest labs/module-01/lesson-04` passes; `analyze.py` runs on your seeds; your write-up states the seed std, the MDE for 2, 3 and 5 seeds, and a decision for the lr4e-3 arm under the rule in the contract.

### Experiment contract

- **Question:** what is the toy recipe's seed noise floor on Eval v0 held-out loss, and does learning rate 4e-3 beat 3e-3 by at least 0.02 nats at 300 steps? Decision informed: the seed count to use for toy-scale comparisons in later CPU labs.
- **Hypothesis:** a 33% higher learning rate changes loss by a few hundredths at this short budget; direction unknown. Status: may not appear at this scale.
- **Baseline:** `base`, learning rate 3e-3 (the course default), seeds 0–4. Not tuned beyond the default; the variant is one other point, so the tuning budget is equal (one value each).
- **Changed variable:** the learning rate. **Controlled:** Data-v0 at the CPU size, `toy` preset, 300 steps of 16 × 128 tokens, warmup 50, cosine schedule, seeds 0–2 shared by both arms (same initialisation and data order), the same 256 validation windows (seed 1234).
- **Comparison axis:** equal tokens (and equal FLOPs, since the model is the same).
- **Budget:** free CPU, 8 runs, 14 minutes measured on a 16-thread laptop.
- **Metrics and decision rule:** primary: seed-level paired 95% interval of (lr4e-3 − base) mean validation loss over seeds 0–2. Adopt lr4e-3 if the upper bound is below −0.02; reject if the lower bound is above −0.02; otherwise inconclusive. Reported alongside: paired and unpaired window intervals, Holm-adjusted per-seed tests.
- **Correctness checks:** every run's card shows the same `data_files` hashes; `analyze.py` pairs only seeds present in both arms; the window losses of all runs have the same length (256).
- **Fallback evidence:** none needed; a null or inconclusive result is a valid outcome.
- **Limits:** a 1.8M-parameter model at 0.6M tokens per run; the noise floor at this scale says nothing about Baseline-0's, which the module project measures.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× GPU (L4/A100/H100), about 15 min. Not run in this build; part of the Module 1 pilot | `python labs/module-01/lesson-04/run_seeds.py --preset pilot-10m --device cuda --batch 64 --seq 512 --steps 2000 --out runs/l14-gpu`, then `analyze.py --out runs/l14-gpu` |
| Free GPU (Colab/Kaggle T4) | T4 | the main-path command with `--batch 32 --steps 1000` |
| Free CPU | laptop; measured 14 minutes for `run_seeds.py` (8 runs) and a few seconds for `analyze.py`, 2026-10-03, 16 threads | `python labs/module-01/lesson-04/run_seeds.py`, then `python labs/module-01/lesson-04/analyze.py` |

### Steps

1. **Implement** `mde`, `seeds_needed`, `unpaired_bootstrap` and `seed_level_ci` in `lab.py`; run `pytest labs/module-01/lesson-04`.
2. **Train the seeds** with `run_seeds.py` (unattended). While it runs, write down your prediction of the seed std and of the decision.
3. **Analyse** with `analyze.py`. Copy its output into your notes.
4. **Answer in writing:** (a) the seed std and the MDE for 2, 3 and 5 seeds; how many seeds would you need to detect 0.01 nats? (b) For seed pair 0, why is the paired window interval so much narrower than the unpaired one, and why is neither of them the right interval for the decision? (c) Do the per-seed paired window intervals agree in sign? What does that tell you about using one seed and a window bootstrap? (d) Apply the contract's decision rule to the seed-level paired interval.
5. **Holm.** You make three claims from this experiment (held-out loss, LAMBADA log-probability, LAMBADA accuracy) with raw p-values 0.012, 0.030 and 0.300. Which survive Holm at $\alpha = 0.05$?

<details>
<summary>Hint for step 4(b)</summary>

Look at the two numbers `analyze.py` prints first in section 3: the standard deviation of base's per-window losses and the standard deviation of the per-window differences. Their ratio is roughly the ratio of the two interval widths.

</details>

<details>
<summary>Answer for step 5</summary>

Sorted: 0.012 × 3 = 0.036, 0.030 × 2 = 0.060, 0.300 × 1 = 0.300 (running maximum unchanged). Only the held-out-loss claim survives (0.036 ≤ 0.05); the LAMBADA log-probability claim does not (0.060), although it would have without correction.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-01/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-01/lesson-04`, and run `LAB_TARGET=solution python labs/module-01/lesson-04/analyze.py` to see the reference analysis.

</details>

## Common mistakes

- **Reporting a window-bootstrap interval as evidence about a method.** It covers evaluation noise for two fixed models; the method claim needs seeds.
- **Scoring runs on different windows.** Pairing then pairs unrelated items and gives no variance reduction; Eval v0 fixes the window seed for this reason.
- **Using the normal 1.96 with three seeds.** With $n - 1 = 2$ degrees of freedom the 97.5% t-quantile is 4.30.
- **Testing many variants and reporting the best.** Correct with Holm, or confirm the winner in a fresh run with new seeds.
- **Planning without the MDE.** If the effect you expect is below the MDE, the experiment can only end "inconclusive".
- **Shrinking the wrong noise.** More evaluation items cannot fix seed variance; more seeds cannot fix a 20-item evaluation.

## References

- E. Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*, 2024. https://arxiv.org/abs/2411.00640
- P. Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004. https://aclanthology.org/W04-3250/
- S. Holm, *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of Statistics 6(2), 1979. http://www.jstor.org/stable/4615733
- Moonshot AI, *Kimi K2*, section 2.3. https://arxiv.org/abs/2507.20534

## Next

[01.5 · Reproducibility and the experiment record](lesson-05.md)
