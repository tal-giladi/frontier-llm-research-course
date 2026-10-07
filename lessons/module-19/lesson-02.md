---
id: "19.2"
module: 19
minutes: 35
practice_minutes: 90
prerequisites: ["19.1", "01.4", "01.5", "03.3"]
objectives:
  - Scope a reproduction to one claim at one location in a paper and to the smallest experiment that can test it, and say whether it is a reproduction or a replication in the National Academies' sense.
  - State before the runs what counts as reproduced, with a tolerance derived from seed noise, and decide direction and magnitude separately.
  - Compute Wortsman et al.'s learning-rate sensitivity by hand and with code, and apply the reproduction decision to per-seed effects with a t-interval.
  - Keep a deviation log that covers every aspect of the setup and say what each deviation is expected to do to the claim; write a specific question to the authors when the paper leaves a gap.
volatility: concept
sources:
  - title: "Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities (section 2.1 setup; section 2.2 LR sensitivity; Figure 1; section 3.1.1 attention-logit growth; section 3.2.1 warm-up)"
    url: https://arxiv.org/abs/2309.14322
  - title: "National Academies of Sciences, Engineering, and Medicine, Reproducibility and Replicability in Science (2019), Summary: definitions"
    url: https://www.nationalacademies.org/read/25303/chapter/2
  - title: "Pineau et al., Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)"
    url: https://arxiv.org/abs/2003.12206
  - title: "ML Reproducibility Challenge (MLRC 2026, a NeurIPS 2026 track)"
    url: https://reproml.org/
  - title: "Henderson et al., Deep Reinforcement Learning that Matters (Figure 5: seed variance)"
    url: https://arxiv.org/abs/1709.06560
  - title: "John Schulman, An Opinionated Guide to ML Research (Personal Development: reimplement papers and compare with published results)"
    url: http://joschu.net/blog/opinionated-guide-ml-research.html
last_verified: "2026-10-07"
---

# 19.2 · Reproducing a paper

Every capstone starts by reproducing a published claim before extending it, and every research engineer reproduces papers to decide whether to build on them. This lesson makes a reproduction a decision rather than an impression: one claim at one location in the paper, the smallest experiment that can test it, a tolerance stated before the runs and derived from seed noise, direction and magnitude decided separately, a deviation log, and a short, specific question to the authors when the paper leaves a gap. The lab reproduces Wortsman et al.'s claim that QK-layernorm reduces learning-rate sensitivity on CPU with the course's own training code, and writes a validated reproduction record.

## Why this matters at a frontier lab

A lab adopts outside results all the time: an optimizer trick, a normalisation, an RL objective. Each adoption is a bet that the published effect will appear in your setting. A careful reproduction at small scale is the cheap way to check the bet, and its value depends entirely on whether its outcome was decided honestly. "We tried it and it sort of worked" cannot be acted on; "the claimed direction appears at 1.8M parameters, with an effect of 0.3 nats against a tolerance of 0.057 stated in advance, magnitude not comparable for these three reasons" can. Schulman recommends reimplementing papers and comparing with the published results early in a career because the target is known, so the feedback is fast; that only works if you decide in advance what agreement means.

## The idea

### Reproduce, replicate, or neither

The National Academies' report separates two words the field uses loosely. **Reproducibility** is "obtaining consistent computational results using the same input data", with the same code, steps and conditions of analysis. **Replicability** is "obtaining consistent results across studies aimed at answering the same scientific question", each with its own data. Running the authors' released code on their data is a reproduction. Re-implementing a method in your own code base, on your own data, at a smaller scale, is closer to a replication of the claim, and it can fail for reasons that have nothing to do with the paper being wrong. This course calls it "reproducing a claim at small scale" (as the capstone brief does) and requires the record to say which parts were the same.

### Scoping: one claim, one location, the smallest test

A paper makes many claims. Choose one and quote it with its location (abstract, figure caption, section, table). Then ask what the smallest experiment is that could show it:

- **What is varied** in the claim (here: the learning rate, with and without QK-layernorm).
- **What is measured** (final loss; a summary of the sweep).
- **Which part of the claim is testable at your scale.** Wortsman et al.'s Figure 1 caption makes two claims: "Qk-layernorm reduces LR sensitivity, but LR sensitivity still increases with model scale". The first needs one size; the second needs a ladder. A single-size CPU reproduction tests the first and says plainly that it does not test the second.

### What counts as reproduced, stated before the runs

Write the decision rule before the first run, with its date. The course's rule has two separate parts.

**Direction.** Sign every per-seed effect so that positive means "the paper's direction". With $n$ seeds, mean $\bar e$, sample standard deviation $s$ and the 95% Student-$t$ interval $[\ell, u] = \bar e \pm t_{0.975,\,n-1}\, s/\sqrt n$, and a tolerance $\tau$, the smallest effect that would count:

| Outcome | Rule |
|---|---|
| reproduced | $\ell > 0$ and $\bar e \ge \tau$ |
| contradicted | $u < 0$ |
| not reproduced | $u < \tau$ (an effect as large as the tolerance is excluded) |
| inconclusive | otherwise: more seeds, not a verdict |

The tolerance comes from the noise, not from the result. The lab uses $\tau = 2\sigma_{\text{seed}}$ with $\sigma_{\text{seed}} = 0.0286$ nats, the toy recipe's seed std measured in lesson 01.4 before this lab existed. A per-seed $t$-interval is wide with three seeds ($t_{0.975,2} = 4.303$), which is the honest price of three seeds; Henderson et al. show why fewer is worse: two groups of five runs of the *same* algorithm can look like different algorithms (Figure 5).

**Magnitude.** Compare the size with the paper's only when the published number is on the same footing: same metric, same units, a setup you matched. Then "matches" if $\lvert \bar e - e_{\text{paper}} \rvert \le$ a magnitude tolerance stated in advance, else "differs". At a different scale, data set or sweep, the honest outcome is **not comparable**, with the reasons. "Direction reproduced at this scale; magnitude not comparable" is the normal result of a small-scale reproduction, and it is a useful one.

### Recording deviations

Every reproduction differs from the paper. The deviation log has one line per aspect (scale, data, tokens, optimizer, learning rates, evaluation, seeds, code), even when the line is "same as the paper": what the paper did, what you did, why, and what you expect the difference to do to *this claim* (weakens it, strengthens it, unknown, none expected). A deviation found after the results is still recorded, marked "after", with what prompted it. Two habits:

- **Match what is cheap to match.** Wortsman et al. state their optimizer settings in section 2.1 (AdamW with $\beta_2 = 0.95$, $\varepsilon = 10^{-8}$, gradient clipping 1.0, z-loss $10^{-4}$, independent weight decay $10^{-4}$, warm-up then cosine). All of those cost nothing to copy, so the lab copies them. Independent decay means the decay is not multiplied by the learning rate; PyTorch's AdamW multiplies it by the scheduled rate, so passing $\lambda/\eta$ as `weight_decay` gives the paper's $\lambda$ times the schedule.
- **Predict the direction of each unavoidable deviation.** The paper reports that "a longer warm-up period reduces LR sensitivity" (section 3.2.1); a deviation in warm-up therefore has a known direction, and the lab keeps the paper's 5% to avoid it. Fewer learning rates in the sweep (three, not seven) removes the extreme rates where the difference is largest, so it weakens the measured effect.

### Contacting the authors

Write when the paper leaves a gap that changes the result and that you cannot close yourself (an unstated hyperparameter, an ambiguous definition, seed counts behind a figure). Before writing: read the appendix, the released code and the issues of its repository. The message is short: the claim and its location, what you ran, the specific question, and what you will do if there is no answer. Record the question, the date and the answer (or the absence of one) in the record. Do not wait for a reply to run what you can run, and do not present an answer from the authors as evidence; it is a clarification. The NeurIPS reproducibility program (Pineau et al.) introduced a code-submission policy, the ML Reproducibility Challenge (in 2026 an official NeurIPS track) and a checklist precisely to reduce how often such questions are needed.

## Worked example

**Learning-rate sensitivity by hand.** Wortsman et al. section 2.2: for a sweep of rates $\eta \in [a, b]$,

$$\text{LR sensitivity} = \mathbb{E}_{\eta \in [a, b]}\big[\min(\ell(\mathcal A(\eta)), \ell_0) - \ell^*\big], \qquad \ell^* = \min_{\eta \in [a,b]} \ell(\mathcal A(\eta)),$$

with $\ell_0$ the loss at initialisation, so a diverged run counts as $\ell_0$. With vocabulary 8,192, $\ell_0 = \ln 8192 = 9.011$ (a uniform prediction; the course uses this for $\ell_0$ and records it as a deviation). An arm with final losses 6.40, 6.44, 6.56 at three rates: $\ell^* = 6.40$, mean $= 6.467$, sensitivity $= 0.067$. An arm with 6.31, 6.56 and a diverged run: values 6.31, 6.56, 9.011; $\ell^* = 6.31$; mean $= 7.294$; sensitivity $= 0.984$.

**The decision by hand.** Per-seed effects (sensitivity without minus with) $0.30, 0.25, 0.28$, tolerance $\tau = 0.057$. Mean $0.2767$, $s = 0.0252$, $s/\sqrt 3 = 0.01453$, half-width $4.303 \times 0.01453 = 0.0625$, interval $[0.214, 0.339]$. $\ell > 0$ and $\bar e \ge \tau$: **reproduced**. With effects $0.040, 0.041, 0.042$: interval $[0.0385, 0.0435]$, above zero but entirely below $\tau$: **not reproduced** — real at this scale, perhaps, but smaller than anything the rule said would count. With $0.20, -0.05, 0.10$: mean $0.083$, $s = 0.126$, interval $[-0.23, 0.40]$: **inconclusive**.

## Shapes and cost

| Item | Size | Cost |
|---|---|---|
| one CPU run | toy preset (1,836,416 parameters), 300 steps × 16 × 128 = 614,400 tokens, fp32, max-logit probe every 5 steps | measured 101–115 s on the build laptop |
| the CPU sweep | 2 arms × 3 rates × 3 seeds = 18 runs | measured 37.7 minutes |
| T4 sweep | `pilot-10m`, 1,000 steps × 16 × 256, 5 rates, 3 seeds, fp32: 30 runs | PROJECTED 2.3 T4-hours: $30 \times 5.56 \times 10^{14}$ FLOPs ÷ ($8.1 \times 10^{12}$ fp32 peak × 0.25 assumed MFU) |
| main path | `pilot-30m`, 4,000 steps × 64 × 1,024, the paper's 7 rates, 3 seeds, bf16: 42 runs; second rung `pilot-70m` | PROJECTED 4.0 H100-hours ($42 \times 8.41 \times 10^{16}$ FLOPs ÷ ($989 \times 10^{12}$ × 0.25)), and 7.6 for `pilot-70m` |
| the record | per-seed sensitivities, effects, decision, deviation log | milliseconds |

## Build it

```python
import math
from frontierlab.research.reproduce import lr_sensitivity, decide, Deviation, Record, validate, render

l0 = math.log(8192)
print(lr_sensitivity({3e-3: 6.40, 1e-2: 6.44, 3e-2: 6.56}, l0))           # 0.0667
print(lr_sensitivity({3e-3: 6.31, 1e-2: 6.56, 3e-2: float("nan")}, l0))   # 0.984
print(decide([0.30, 0.25, 0.28], tolerance=0.057))   # direction 'reproduced', magnitude 'not comparable', ci (0.214, 0.339)
```

`validate(record)` refuses a record whose tolerance is dated after its first result, whose stated outcome does not follow from its own numbers, whose deviation log misses an aspect or has an entry without a reason or expected effect, that says nothing about contacting the authors, or that claims "reproduced" at a different scale without naming the scale in its scope. `render` writes it as markdown. Tests: `pytest labs/common/tests/test_research.py -k "sensitivity or decide or record or t_interval"`.

The sweep itself is the course loop with the Module 7 wrapper: `python -m frontierlab.optim.train --optimizer adamw --qk-norm on|off --z-loss 1e-4 --weight-decay <1e-4/lr> --warmup <5% of steps> --stability-log ...`. Two arms of the same rate and seed differ in the QK-norm switch only, `config.qk_norm` and its record `optim.qk_norm` (`python -m frontierlab.record <a> <b> --changed config.qk_norm optim.qk_norm` confirms it).

## What the evidence says

- **Attention-logit growth at high learning rates, and QK-layernorm as its fix: ESTABLISHED.** Reported at large scale (Dehghani et al., lesson 07.5), reproduced at small scale by Wortsman et al. (section 3.1.1, Figure 2), adopted independently (Gemma 3, OLMo 2, Qwen3: lesson 03.3). This course measured it twice before this lesson, on one seed each (lessons 03.3 and 07.2).
- **"Qk-layernorm reduces LR sensitivity": PUBLICLY DOCUMENTED** in Wortsman et al.'s Figure 1 (sizes and seed counts per point are not stated in the caption; the course did not find per-point seed counts in the paper). **"LR sensitivity still increases with model scale": PUBLICLY DOCUMENTED** there and not tested by the CPU lab.
- **Reproducibility practices** (code release, checklists, reproduction challenges): PUBLICLY DOCUMENTED as programmes (Pineau et al.); their effect on how often results reproduce is not established by these sources.
- **The decision rule and tolerance:** the course's formalisation, REASONABLE INDUSTRY PRACTICE in spirit (pre-registered thresholds relative to measured noise).

## Lab

**Folder:** [`labs/module-19/lesson-02/`](../../labs/module-19/) · **Time:** about 90 minutes (about 40 of them unattended on CPU) · **Pass check:** `pytest labs/module-19/lesson-02` passes; `repro_lab.py` writes `runs/m19/l192/cpu/record.md` with no record problems.

### Experiment contract

- **Question:** at toy scale on Data-v0, does QK-norm reduce learning-rate sensitivity over the rates 3e-3, 1e-2 and 3e-2, as Wortsman et al. claim for QK-layernorm (Figure 1 caption, section 3.1.1)? Decision informed: whether the course can treat "QK-norm protects against learning-rate-driven logit growth" as reproduced at small scale when the capstone uses it (QK-Clip vs QK-norm).
- **Hypothesis:** sensitivity is lower with QK-norm on every seed. **Status:** reported effect (Wortsman et al.), seen once at toy scale with one seed in lesson 03.3.
- **Baseline:** QK-norm on (Baseline-0's default), the same sweep.
- **Changed variable:** QK-norm off. **Controlled:** toy preset; AdamW ($\beta_2$ 0.95, $\varepsilon$ $10^{-8}$, clipping 1.0) with z-loss $10^{-4}$ and independent decay $10^{-4}$ as in the paper's section 2.1; 15 warm-up steps (5%) then cosine; 300 steps × 16 × 128 tokens; seeds 0, 1, 2 (initialisation and data order, shared by seed across arms); the same 128 validation windows.
- **Comparison axis:** equal tokens per run, across a learning-rate sweep. It does not answer anything about wall-clock or about larger models.
- **Budget:** free CPU, 18 runs, measured 37.7 minutes; main path PROJECTED 4.0 H100-hours (formula above).
- **Metrics and decision rule:** final held-out loss per run; sensitivity per arm and seed with $\ell_0 = \ln 8192$; effect per seed = sensitivity without QK-norm minus with; the direction rule above with $\tau = 2 \times 0.0286 = 0.057$ nats, stated 2026-10-07 before the first run; magnitude "not comparable" (different scale, data, sweep). Secondary, observation only: the ratio of maximum attention logits at 3e-2.
- **Correctness checks:** `pytest labs/module-19/lesson-02`; `pytest labs/common/tests/test_research.py`; the run-card diff above; the record validates.
- **Fallback evidence:** Wortsman et al.'s Figure 1 and lesson 03.3's single-seed measurement, labelled as published result and earlier course observation.
- **Limits:** one size of 1.8M parameters; three rates out of the paper's seven; 0.6M tokens per run; RMSNorm rather than LayerNorm on q and k; Data-v0 instead of C4.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100. Not run in this build; part of the Module 19 pilot | `python labs/module-19/lesson-02/repro_lab.py --variant main --print` prints the 42 runs at `pilot-30m` with the paper's seven rates; `--variant main70` the second rung, which makes the "sensitivity grows with scale" part testable. Then `--variant main --part analyse`. PROJECTED 4.0 and 7.6 H100-hours |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `--variant t4 --print`: `pilot-10m`, 5 rates, 3 seeds; PROJECTED 2.3 T4-hours across sessions (runs resume) |
| Free CPU | laptop; measured 37.7 minutes for the 18 runs | the steps below |

### Steps

1. **Read the claim in place.** Open the paper at Figure 1 and section 2.1. Write down the claim, its location, and which half of the caption one size can test.
2. **State the rule.** Check `NOISE_FLOOR`, `TOLERANCE` and `STATED_ON` in `lab.py`; if you change the tolerance rule, change the date to today and do it now, before running.
3. **Implement** `lr_sensitivity`, `seed_effects`, `decide` and `deviations` (TODOs 1–4) and run `pytest labs/module-19/lesson-02`. For `deviations`, read the paper's section 2.1 and the arguments `repro_lab.py` passes in `argv_for`.
4. **Run:** `python labs/module-19/lesson-02/repro_lab.py` (about 40 minutes; rerun to resume). Read `runs/m19/l192/cpu/record.md`.
5. **Write** to the authors, as an exercise only (do not send it): the one question the paper leaves open that could change your outcome. Put it in the record's author-contact line.
6. **Write up** (one page): the outcome by the rule; the per-seed effects and interval; the deviation you think matters most and its expected direction; and one sentence on what the main-path ladder would add.

<details>
<summary>Hint for TODO 3</summary>

`t_interval(effects)` returns the mean and the interval. Check the outcomes in the order of the table: "reproduced" first, then "contradicted", then "not reproduced", otherwise "inconclusive". Magnitude is independent of direction.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (Windows 11, Python 3.12.13, torch 2.14.1+cpu, 16 threads, another module's jobs sharing the CPU for part of the time): 18 runs in 37.7 minutes, 126 s per run on average (101–115 s for the runs before the other jobs started). Final held-out loss on 128 validation windows, with the run's maximum attention logit in brackets:

| Arm | Seed | lr 3e-3 | lr 1e-2 | lr 3e-2 | Sensitivity |
|---|---|---|---|---|---|
| QK-norm | 0 | 6.409 [7.1] | 6.441 [5.9] | 6.560 [16.4] | 0.061 |
| QK-norm | 1 | 6.385 [6.6] | 6.593 [6.0] | 6.501 [12.7] | 0.108 |
| QK-norm | 2 | 6.391 [6.1] | 6.433 [7.4] | 6.523 [10.4] | 0.058 |
| no QK-norm | 0 | 6.309 [29.7] | 6.561 [211] | 7.008 [2,502] | 0.317 |
| no QK-norm | 1 | 6.317 [21.1] | 6.659 [210] | 7.110 [3,096] | 0.378 |
| no QK-norm | 2 | 6.260 [32.2] | 6.933 [296] | 6.964 [3,833] | 0.459 |

Per-seed effects +0.256, +0.270, +0.401; mean +0.309, 95% t-interval [+0.110, +0.508] against the tolerance 0.057 stated before the runs: **direction reproduced; magnitude not comparable.** The record validated with no problems. Three observations the rule did not ask about. No run diverged (no nan): without QK-norm the loss degraded to 7.0–7.1 at 3e-2 while the maximum logit reached 2,500–3,800, 150–370 times the QK-norm arm's, the mechanism of section 3.1.1 at 1.8M parameters (Wortsman et al. report that runs with logits above $10^4$ diverged, section 3.3). At the lowest rate, 3e-3, the model *without* QK-norm was better on every seed (6.26–6.32 against 6.39–6.41), as lesson 03.3 saw on one seed: the claim is about sensitivity, not about which arm wins at a given rate. And seed 1 with QK-norm is worse at 1e-2 than at 3e-2, a reminder that one seed per point (common in published sweeps) can put a bump in a curve. The scope sentence says what this does not show: anything about larger models, or about the paper's "sensitivity increases with scale".

</details>

<details>
<summary>Reference solution</summary>

`labs/module-19/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-19/lesson-02`.

</details>

## Common mistakes

- **Reproducing "the paper".** Pick one claim, quote it, and give its location. A paper is not a unit of reproduction.
- **Choosing the tolerance after the results.** Date it; the validator refuses a record whose tolerance is dated after the first run.
- **Reporting magnitude when it is not comparable.** A smaller effect at 1.8M parameters is not "partly reproduced"; it is a different setting. Say "not comparable" and why.
- **Silent deviations.** Every aspect gets a line, including "same as the paper". A deviation discovered later is recorded as "after".
- **Calling an underpowered null "not reproduced".** If the interval still contains the tolerance, the outcome is "inconclusive".
- **Waiting on the authors.** Ask a specific question, record it, and run what you can; an answer clarifies, it does not count as evidence.

## References

- M. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, 2023, sections 2.1, 2.2, 3.1.1, 3.2.1 and Figure 1. https://arxiv.org/abs/2309.14322
- National Academies of Sciences, Engineering, and Medicine, *Reproducibility and Replicability in Science*, 2019, Summary. https://www.nationalacademies.org/read/25303/chapter/2
- J. Pineau et al., *Improving Reproducibility in Machine Learning Research*, 2020. https://arxiv.org/abs/2003.12206
- ML Reproducibility Challenge. https://reproml.org/
- P. Henderson et al., *Deep Reinforcement Learning that Matters*, 2017, Figure 5. https://arxiv.org/abs/1709.06560
- J. Schulman, *An Opinionated Guide to ML Research*, section Personal Development. http://joschu.net/blog/opinionated-guide-ml-research.html
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[19.3 · Writing and defending results](lesson-03.md)
