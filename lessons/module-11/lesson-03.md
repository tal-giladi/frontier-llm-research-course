---
id: "11.3"
module: 11
minutes: 35
practice_minutes: 120
prerequisites: ["11.1", "11.2", "07.3", "07.5", "09.4"]
objectives:
  - Design a ladder for a target run (sizes, token ratios, schedule, share of the target's compute) and state what each choice protects against.
  - Check hyperparameter transfer along the ladder with a learning-rate sweep and Wortsman et al.'s sensitivity metric before trusting a fit.
  - Pre-register a prediction for a target run 3x beyond the ladder (point, interval, a predicted loss curve, stopping rules) and check it afterwards, including the timing of the record.
  - Apply a go/no-go checklist before launch and an off-band rule during the run, and measure how much compute the rule saves on a run with a planted data-loader bug.
volatility: concept
sources:
  - title: "Bhagia et al., Establishing Task Scaling Laws via Compute-Efficient Model Ladders (sections 2-4, Table 1)"
    url: https://arxiv.org/abs/2412.04403
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.2.1)"
    url: https://arxiv.org/abs/2407.21783
  - title: "OpenAI, GPT-4 Technical Report (section 3, Predictable Scaling)"
    url: https://arxiv.org/abs/2303.08774
  - title: "DeepSeek-AI, DeepSeek LLM (section 3.1: hyperparameter scaling laws, Eq. 1; Figure 5)"
    url: https://arxiv.org/abs/2401.02954
  - title: "Hu et al., MiniCPM (section 3: model wind tunnel experiments)"
    url: https://arxiv.org/abs/2404.06395
  - title: "Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities (sections 2.2, 3.1)"
    url: https://arxiv.org/abs/2309.14322
  - title: "Choshen, Zhang and Andreas, A Hitchhiker's Guide to Scaling Law Estimation (sections 4, 6, 8)"
    url: https://arxiv.org/abs/2410.11840
  - title: "Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (section 5)"
    url: https://arxiv.org/abs/2405.18392
last_verified: "2026-10-06"
---

# 11.3 · De-risking a run

A ladder is cheap insurance for an expensive run. Before the final pretraining run starts, a lab trains a set of small models with exactly the run's recipe, checks that its hyperparameters transfer along them, fits a law through them, tests that law on rungs it did not see, and writes down what the big run should do. Then it watches the big run against that record and stops it if it leaves the predicted band. This lesson builds that procedure end to end: a learning-rate transfer check, a pre-registered prediction for a target three times beyond the ladder (point, interval, a predicted loss curve and stopping rules), the target run itself, the check against the record, and a second target with a planted data-loader bug that the stopping rule has to catch.

## Why this matters at a frontier lab

The large run is where mistakes are expensive and slow to see: a learning rate that was right at 100M and is wrong at 3B, a data loader that silently reads one shard, a numerical instability that only grows with width (lesson 07.5), a node failure (lesson 09.4). Each costs days of a cluster if it is found late, and some, like a quietly worse loss, are never found unless someone had a number to compare against. Published reports show the pattern: GPT-4's final loss was predicted from runs with at most 1/10,000 of the compute, shortly after the run started (section 3.1); DeepSeek LLM fitted learning rate and batch size as functions of compute on small runs and validated them at $10^{20}$ FLOPs before scaling up (section 3.1); the OLMo ladder predicts its 7B and 13B targets from about 1% of their compute. The engineering skill is not the fit; it is designing the ladder so its failures are informative, and writing the decision rules down before the result can bend them.

## The idea

### What can go wrong, and which part of the ladder catches it

| Risk | Ladder element that catches it | If it fails |
|---|---|---|
| hyperparameters that do not transfer (learning rate, batch, warmup) | a sweep at two or more rungs; the best value and the sensitivity should agree, or follow a fitted rule | refit with µP (lesson 07.3) or a hyperparameter scaling law |
| instabilities that grow with scale | the same rungs at a raised learning rate, with max logits and gradient norms logged (lesson 07.5) | add the stabiliser before the big run |
| the law extrapolating wrongly | a held-out rung or budget; the error there is the honest uncertainty | widen the interval, add rungs closer to the target |
| a recipe or pipeline bug in the big run | the predicted loss curve (band) and a stopping rule | stop, diagnose, restart from the last good checkpoint (lesson 09.4) |
| changing the rules after seeing the result | a pre-registration, hashed and timestamped | nothing: this one has to be prevented |

### Designing the ladder

- **Same recipe, only size and tokens change.** Architecture, optimizer, data mixture, tokenizer, schedule shape, precision. A ladder trained with AdamW predicts nothing about a Muon run.
- **Sizes spanning at least one order of magnitude, ideally close to the target.** Choshen, Zhang and Andreas (485 models, over 1,000 fitted laws) find that 5 models "is a safe bet", that models close in size to the target extrapolate best, and that about 4% absolute relative error is "the best ARE typically obtained" (sections 4, 8).
- **Token ratios that bracket the target's.** The OLMo ladder trains 190M, 370M, 760M and 1.3B models at 1×, 2×, 5× and 10× Chinchilla's ratio (20, 40, 100 and 200 tokens per parameter) because its targets are heavily over-trained: 16 models, about 1% of the targets' compute (section 2, Table 1).
- **Every run with its own schedule length**, or one long WSD run per size with short cooldowns from several checkpoints (Hägele et al., section 5: less than half the FLOPs of a Chinchilla-style suite). Intermediate checkpoints of a cosine run are not finished runs (lesson 11.1); Choshen et al. recommend intermediate checkpoints only with care, dropping the earliest ones (section 6).

### Transfer: does the best learning rate move?

A fit through rungs that were each trained at the wrong learning rate predicts a wrong target. Two ways to make the rate right at every size: parametrise so it does not move (µP, lesson 07.3; MiniCPM's "model wind tunnel experiments", section 3, tune at small scale under µP and transfer), or fit how it moves. DeepSeek LLM fits, on runs from $10^{17}$ to $2 \times 10^{19}$ FLOPs (section 3.1, Eq. 1),

$$\eta_{\text{opt}} = 0.3118 \, C^{-0.1250}, \qquad B_{\text{opt}} = 0.2920 \, C^{0.3271} \text{ tokens},$$

and validates it at $10^{20}$. Either way the ladder should *measure* it. Wortsman et al. define a learning-rate sensitivity for a sweep over rates $\eta \in [a, b]$ (section 2.2):

$$\text{LR sensitivity} = \mathbb{E}_{\eta \in [a,b]}\big[\min(\ell(\mathcal{A}(\eta)), \ell_0) - \ell^*\big], \qquad \ell^* = \min_{\eta \in [a,b]} \ell(\mathcal{A}(\eta)),$$

with $\mathcal{A}(\eta)$ the weights trained at rate $\eta$ and $\ell_0$ the loss at initialisation (a diverged run counts as $\ell_0$, not as infinity). A recipe whose sensitivity grows along the ladder is one whose best rate is getting harder to hit. They also show that attention-logit growth and output-logit divergence, instabilities of large runs, "also appear in small models when training at high learning rates", and that QK-layernorm and z-loss mitigate both (section 3.1). The ladder is where those are found cheaply.

### Held-out extrapolation and honest intervals

The fit's in-sample error says almost nothing: five parameters through sixteen points fit well by construction. Fit on all but the largest rung or budget and predict it; that error, at a distance comparable to the target's, is the uncertainty to carry. A bootstrap over ladder runs gives an interval for resampling noise only; it does not cover a wrong functional form. When the held-out error is larger than the bootstrap half-width, the interval is too narrow, and the record should say so.

### A predicted curve, not only a point

A final-loss prediction is checked only at the end. To stop a bad run early, predict the whole curve: read every ladder run at the same fractions of *its own* schedule (10%, 20%, …), and at each fraction refit $E_f, A_f, B_f$ with the final fit's exponents held fixed (a linear least-squares problem), bootstrapped for an interval. Because all runs share the schedule's shape, fraction $f$ means the same point of the schedule in each. The result is a band: the target's validation loss at 30% of its run should be inside it. Holding the exponents fixed makes the band narrower than the final-loss interval, which bootstraps all five parameters; the Recipe-R project shows a stopping rule on such a band firing on a run that then met its success criterion. Widen the band to at least the endpoint interval at $f = 1$, or state its false-alarm risk in the record.

### Pre-registration and the rules

The record, written before the run, holds: the target (size, tokens, compute), the point prediction with its interval and level, the band, the fit's parameters and a digest of the ladder results it came from, the pre-launch checks and their outcomes, and the rules: what counts as success (here, the final loss inside the 90% interval) and when to stop (here, validation loss above the band's upper edge by more than 0.02 nats at 10% or more of the run, or a gradient norm above 4× the running median after 10%). The course's version is a JSON file with a SHA-256 of its content and a UTC timestamp, refused if it already exists; the check afterwards verifies the hash and that the record is older than the run's first log line. Ruan et al. (lesson 11.2) did the same in public for predictions about future models.

Before launch, a **go/no-go checklist** turns the ladder into a decision: held-out error under a stated tolerance; every iso-FLOP minimum interior; the learning rate transfers; the interval narrow enough to be useful; the target within a stated distance of the ladder (here at most 10× its largest budget); the ladder a meaningful share of the target's compute. A failed check does not forbid the run; it forbids launching it without a written reason.

## Worked example

### Learning-rate sensitivity

A sweep at rates $10^{-3}, 3 \times 10^{-3}, 10^{-2}$ gives losses $4.5, 4.0, 9.0$ (the last diverged above the initial loss $\ell_0 = 7.0$). $\ell^* = 4.0$; capped losses $4.5, 4.0, 7.0$, mean $5.17$; sensitivity $5.17 - 4.0 = 1.17$ nats. The same sweep with losses $4.2, 4.0, 4.3$ has sensitivity $0.17$.

### DeepSeek LLM's rule at $10^{20}$ FLOPs

$\eta_{\text{opt}} = 0.3118 \cdot (10^{20})^{-0.125} = 0.3118 \cdot 10^{-2.5} = 0.3118 \cdot 0.00316 = 9.9 \times 10^{-4}$. $B_{\text{opt}} = 0.2920 \cdot 10^{20 \cdot 0.3271} = 0.2920 \cdot 10^{6.542} = 0.2920 \cdot 3.48 \times 10^{6} = 1.0 \times 10^6$ tokens per batch. Ten times more compute multiplies the rate by $10^{-0.125} = 0.75$ and the batch by $10^{0.327} = 2.1$. Their constants hold for their data and model; the lesson is the shape, not the numbers.

### The stopping rule

The band's upper edge is 4.10 at 30% and 4.00 at 40% of the run. An evaluation at 35% reads 4.09. The interpolated edge is 4.05; with the 0.02 tolerance the rule fires above 4.07, so the run stops at 35% and saves 65% of its compute. Had the evaluation read 4.06, the run would continue: the tolerance is the price of not stopping on noise, and it is set before the run.

### Ladder share of compute

The CPU ladder of 11.1 spends $7.3 \times 10^{13}$ FLOPs and this lesson's target $3 \times 10^{13}$: the ladder is larger than the target, which only happens at toy scale. The main-path Recipe-R project's ladder is $1.45 \times 10^{19}$ FLOPs against a $5.0 \times 10^{19}$ target (29%, PROJECTED); the OLMo ladder is about 1% of its targets. The fraction a lab can afford falls as the target grows, which is why the ladder's design, not its size, carries the weight.

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| bootstrap fits | 300 × 5 parameters, float64, CPU | grid fit without refinement per resample, tens of seconds in all |
| band | 10 fractions × (lo, mid, hi), float64 | fixed exponents: one 3-column least-squares solve per fraction and resample |
| pre-registration | one JSON file of a few kB, with SHA-256 and UTC time | never overwritten |
| target run | one training run, evaluated at every 10% | its `metrics.jsonl` is what the monitor reads |

The cost is the runs: the transfer sweep (4 runs at $3 \times 10^{12}$ FLOPs on the CPU), the target ($3 \times 10^{13}$) and the bad target (half of one). Main path (`--variant main`): sweep 4 runs at $3 \times 10^{17}$, target $3 \times 10^{18}$, bad run half of that: PROJECTED $(1.2 + 3 + 1.5) \times 10^{18} / (989 \times 10^{12} \cdot 0.3) / 3600 \approx 5.3$ H100-hours.

## Build it

```python
from frontierlab.scaling import derisk, fit
boots = fit.bootstrap(N, D, L, n_boot=300)
mid, lo, hi = fit.prediction_interval(boots, N_t, D_t, level=0.9)
band = derisk.curve_band(ladder_runs, N_t, D_t, fractions=[0.1, 0.2, 0.3], alpha=f["alpha"], beta=f["beta"])
derisk.preregister("runs/m11/l113/cpu/prereg.json", {"prediction": {...}, "band": band, "rules": {...}})
derisk.check_after("runs/m11/l113/cpu/prereg.json", target_result, target_dir)   # inside/above/below, timing
derisk.monitor(val_curve, steps, band, tol=0.02, grad_norms=gn)                     # first step a rule fires
derisk.go_no_go({"held-out error": (ok, detail)})
```

Correctness checks in `labs/common/tests/test_scaling.py`: a pre-registration cannot be overwritten and an edited one fails its digest; the band reproduces a planted law exactly at the final fraction; the monitor stays quiet on an on-band curve, fires at the first off-band evaluation with the right saved fraction, and catches a planted gradient spike.

## What the evidence says

- **Ladders of small runs with the target's recipe: ESTABLISHED practice** (GPT-4 section 3, Llama 3 section 3.2.1, DeepSeek LLM section 3, OLMo ladder, MiniCPM section 3); how much of the target's compute they need varies (about 1% in the OLMo ladder).
- **Hyperparameter scaling laws and µP transfer: ESTABLISHED in principle, MODEL-SPECIFIC in their constants** (DeepSeek LLM's fitted exponents; MiniCPM's µP; lesson 07.3).
- **Small-scale proxies for instabilities: PROMISING** (Wortsman et al.; whether every large-scale instability has a small-scale proxy is open).
- **Predicted-curve monitoring and pre-registration as described here: REASONABLE INDUSTRY PRACTICE.** Labs report predicting final loss early in a run (GPT-4) but do not publish their stopping rules.
- **Course measurement (free CPU, 2026-10-07):** every pre-launch check passed. The ladder's pre-registered 90% interval for a run 3× beyond it was [3.486, 3.554], and the run reached 3.434: a miss by 0.076 nats, five times the ladder's own held-out error. The predicted floor was too high and the target ran at more tokens per parameter than its size had seen. The stopping rule stopped a run with a planted one-shard data loader at its first evaluation (10% of the run).

## Lab

**Folder:** [`labs/module-11/lesson-03/`](../../labs/module-11/) · **Time:** about 120 minutes (about 40 of them unattended) · **Pass check:** `pytest labs/module-11/lesson-03` passes; `predict` writes the pre-registration before `run`; `check` reports where the measurement fell and that the timing check passed; `bad` reports the step at which the rule fired.

### Experiment contract

- **Question:** does the 11.1 ladder predict the final validation loss of a run at $3 \times 10^{13}$ FLOPs (3× its largest budget, at the size it recommends) inside a pre-registered 90% interval, and does the pre-stated stopping rule stop a run with a planted data-loader bug early? Decision informed: whether the procedure is good enough to size and monitor the Recipe-R run.
- **Hypothesis:** the measured loss falls inside the interval; the bad run leaves the band within the first half of its schedule. Status: reported effect for the first (the ladders above); the second depends on the bug and on the band's width.
- **Baseline:** the pre-registered prediction itself.
- **Changed variable:** none for the target (it tests a prediction); the bad run changes only the data loader (training windows from the first 65,536 tokens).
- **Controlled:** the 11.1 recipe (data, tokenizer, learning rate, warmup fraction, cosine, sequence 128, batch 16, seed 0, evaluation windows).
- **Comparison axis:** equal FLOPs between the prediction and the run (the target's steps are set from its exact FLOPs per token).
- **Budget:** free CPU: 4 sweep runs, 1 target, 1 half target (measured below).
- **Metrics and decision rule:** success if the final validation loss lies inside the 90% interval; report the error and whether the 11.1 held-out error had already predicted its size. Bad run: the first evaluation at ≥ 10% of the run above the band by more than 0.02 nats; report the compute the rule saves.
- **Correctness checks:** `pytest labs/common/tests/test_scaling.py -k "prereg or band"`; the pre-registration's digest verifies and the record is older than the target's first log line; the target's `budget.train_flops` equals the record's.
- **Limits:** one target, one seed (seed noise is part of the error and is not separated); tiny models; a crude planted bug; a 90% interval misses one time in ten even when everything is right.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100. Not run in this build; part of the Module 11 pilot | the same five commands with `--variant main` (sweep at $3 \times 10^{17}$; target at $3 \times 10^{18}$ chosen among pilot-70m, Baseline-0, m11-200m, m11-350m). PROJECTED 5.3 GPU-hours (formula above) |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `--variant t4` (target $3 \times 10^{14}$ among m11-r6, m11-r7) |
| Free CPU | laptop; measured below | the steps below |

### Steps

1. **Implement** the five TODOs in `lab.py`; run `pytest labs/module-11/lesson-03`.
2. **Transfer:** `python labs/module-11/lesson-03/derisk_lab.py transfer`.
3. **Predict:** `python labs/module-11/lesson-03/derisk_lab.py predict`. Read the launch checklist. If a check fails, write one paragraph: launch anyway or not, and why. Do this before step 4.
4. **Run** (unattended): `python labs/module-11/lesson-03/derisk_lab.py run`, then `python labs/module-11/lesson-03/derisk_lab.py check`.
5. **The bad run:** `python labs/module-11/lesson-03/derisk_lab.py bad`.
6. **Write up:** the sweep table and whether the rate transfers; the checklist and your launch decision; the prediction against the measurement; whether the 11.1 held-out error had warned you; the step at which the rule fired on the bad run, and what you would check first in its logs.

<details>
<summary>Hint for TODO 3</summary>

`np.interp(f, fractions, his)` gives the band's upper edge at any fraction between the band's points (and clamps outside them). Skip evaluations with `step / steps + 0.005 < min_fraction` before comparing (the slack makes the evaluation at step `steps // 10` count as the 10% point).

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1+cpu, 16 threads, with another module's jobs sharing the CPU). Timings: `transfer` 7 min (4 runs), `predict` 2 min 39 s (300 bootstrap refits), `run` 13 min 46 s (5,665 steps), `check` 13 s, `bad` 6 min 38 s (half a run).

**Transfer** at $3 \times 10^{12}$ FLOPs (validation loss):

| rung | lr 1e-3 | 3e-3 | 1e-2 | best | sensitivity |
|---|---|---|---|---|---|
| m11-r2 (105K) | 4.260 | **3.953** | 3.984 | 3e-3 | 0.113 |
| m11-r4 (394K) | 4.636 | **4.362** | 4.779 | 3e-3 | 0.231 |

The best rate is the same at both sizes, so the check passes. The sensitivity doubles from the smaller rung to the larger. On the larger model a factor of 3 either side costs more, which is the trend Wortsman et al. warn grows with scale. A wider sweep or µP would be the next step before a larger ladder.

**Prediction** (9 of the 16 ladder runs, those with at least 4 tokens per parameter): $E = 3.288$, $\alpha = 0.80$, $\beta = 1.11$. Candidates at $3 \times 10^{13}$ FLOPs:

| rung | tokens per parameter | predicted loss, 90% interval |
|---|---|---|
| m11-r4 | 29 | 3.510 [3.486, 3.554] |
| m11-r5 | 5.5 | 3.534 [3.468, 3.600] |
| m11-r6 | 1.0 | 3.753 [3.670, 3.830] |
| m11-r7 | 0.2 | 4.252 [4.123, 4.342] |

The script chose m11-r4. The m11-r6 and m11-r7 rows lie in the regime the fit had excluded (under 4 tokens per parameter), so their numbers are extrapolations the fit had no right to make. Every checklist item passed:

- held-out error 0.014 nats;
- interior minima;
- the learning rate transfers;
- interval width 0.068 nats;
- the target 3× the largest budget;
- the ladder 243% of the target's compute, possible only at toy scale.

Decision: **go**.

**Check.** Measured final loss **3.434**: 0.076 nats *below* the interval [3.486, 3.554]. The timing check passed: the record was written before the run's first log line. Under the contract this is a **miss**. The model did better than predicted, and the curve shows how. It sat inside the band at 10% (4.145 against [4.134, 4.801]). It fell below the band's lower edge from 20% onwards, and the gap grew to 0.072 at the end. The stopping rule watches only the upper edge, so it correctly did not fire.

The likely cause is the floor. The fit put $E$ at 3.29 from ladder runs whose best loss was 3.67, and the target came 0.15 above that floor where the fit expected 0.22. The target was also further out in tokens than any run of its size: m11-r4 had been trained at most at 9.8 tokens per parameter, and the target ran at 29.

So the 0.014-nat held-out error of 11.1 was the error of one budget step along the ladder's own diagonal. It was not the error of a new size and token ratio combination. Its rule had also been chosen after seeing that budget. The honest uncertainty for this kind of extrapolation was about 0.08 nats, five times the interval's half-width. That is the number to carry into the Recipe-R prediction.

**The bad run** (training windows from the first 65,536 tokens; stopped at 50%):

| fraction | measured | band upper edge |
|---|---|---|
| 0.1 | 5.241 | 4.801 |
| 0.2 | 6.817 | 4.152 |
| 0.5 | 9.300 | 3.731 |

The rule fired at the first evaluation, step 566 (10% of the run), saving 90% of its compute. By 40% the training loss of the bad run was 0.48 against 3.75 in the healthy run, while its validation loss was above the uniform-guess loss ($\ln 1024 = 6.93$). That is the signature from the quiz: falling training loss and rising held-out loss point to the data, so check the data accounting first.

The first version of the rule had compared $f < 0.1$ exactly. The evaluation at step $5665 // 10 = 566$ is at $f = 0.0999$, so that version skipped it and fired one evaluation later. The rule now allows 0.005 of slack. Rounding like this is why stopping rules are tested on a planted failure before they guard a real run.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-11/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-11/lesson-03`.

</details>

## Common mistakes

- **A ladder with a different recipe.** Any change between ladder and target (optimizer, data, precision, schedule) makes the prediction about a different run.
- **One learning rate for every rung, unchecked.** Measure the best rate at two sizes at least; a moving optimum biases the fit.
- **Quoting the bootstrap interval as the uncertainty.** It covers resampling of the ladder runs, not a wrong law; compare it with the held-out error.
- **Testing extrapolation only along the ladder's diagonal.** A held-out budget whose runs look like the training runs (same sizes, similar tokens per parameter) says little about a target at a new combination; hold out what the target will be (the build's 0.014-nat hold-out became a 0.076-nat miss).
- **Writing the prediction after the run started.** Even unintentionally, early curve points leak into it. Timestamp and hash the record, and check the timing.
- **A stopping rule without a tolerance, or one chosen after the run.** Without a tolerance noise stops good runs; chosen afterwards, the rule protects nothing.
- **Monitoring training loss only.** A loader that reads one shard lowers training loss while validation loss rises; monitor held-out loss against the band.

## References

- A. Bhagia et al., *Establishing Task Scaling Laws via Compute-Efficient Model Ladders*, 2024, sections 2–4, Table 1. https://arxiv.org/abs/2412.04403
- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, section 3.2.1. https://arxiv.org/abs/2407.21783
- OpenAI, *GPT-4 Technical Report*, 2023, section 3. https://arxiv.org/abs/2303.08774
- DeepSeek-AI, *DeepSeek LLM*, 2024, section 3.1, Eq. 1, Figure 5. https://arxiv.org/abs/2401.02954
- S. Hu et al., *MiniCPM*, 2024, section 3. https://arxiv.org/abs/2404.06395
- M. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, 2023, sections 2.2, 3.1. https://arxiv.org/abs/2309.14322
- L. Choshen, Y. Zhang and J. Andreas, *A Hitchhiker's Guide to Scaling Law Estimation*, 2024, sections 4, 6, 8. https://arxiv.org/abs/2410.11840
- A. Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, 2024, section 5. https://arxiv.org/abs/2405.18392
- Y. Ruan, C. J. Maddison and T. Hashimoto, *Observational Scaling Laws*, 2024, section 4 (pre-registered predictions). https://arxiv.org/abs/2405.10938
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The module project: [the Recipe-R run](../../projects/module-11-recipe-r.md). Module 12 then starts the post-training stage.
