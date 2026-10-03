---
id: "01.3"
module: 1
minutes: 35
practice_minutes: 90
prerequisites: ["01.1", "01.2"]
objectives:
  - Write a hypothesis, a baseline, exactly one changed variable and the list of controlled variables for an architecture or recipe change.
  - Choose between equal tokens, equal parameters, equal training FLOPs and equal wall-clock, and state what the chosen axis does not answer.
  - Give the baseline the same tuning budget as the new method, and separate scientific, nuisance and fixed hyperparameters.
  - Keep data and hyperparameter selection on train and validation, and touch the test split once, for a pre-stated decision rule.
  - Find the flaws in a flawed experiment write-up and fill in an experiment contract for a corrected version.
volatility: concept
sources:
  - title: "Melis, Dyer, Blunsom — On the State of the Art of Evaluation in Neural Language Models (abstract: properly regularised LSTMs outperform more recent models)"
    url: https://arxiv.org/abs/1707.05589
  - title: "Narang et al. — Do Transformer Modifications Transfer Across Implementations and Applications? (abstract)"
    url: https://arxiv.org/abs/2102.11972
  - title: "Google Research — Deep Learning Tuning Playbook (scientific, nuisance and fixed hyperparameters)"
    url: https://github.com/google-research/tuning_playbook
  - title: "Kimi K2: Open Agentic Intelligence (section 2.3: 64 vs 128 heads, inference-FLOPs argument)"
    url: https://arxiv.org/abs/2507.20534
  - title: "The Llama 3 Herd of Models (section 3.2.1: scaling-law experiments over compute budgets)"
    url: https://arxiv.org/abs/2407.21783
last_verified: "2026-10-03"
---

# 01.3 · Designing an experiment

Most wrong conclusions in ML research do not come from bad statistics; they come from experiments that compared the wrong things. This lesson is about the design that comes before any number: a hypothesis, a baseline, one changed variable and everything else held fixed; the choice of comparison axis, because "better" at equal tokens and "better" at equal wall-clock are different claims; a tuning budget that treats the baseline as fairly as the new idea; and a decision rule and data splits fixed before the results exist. The lab ends with a CPU run in which the same two models swap places when the axis changes.

## Why this matters at a frontier lab

A single pretraining ablation at the scale where decisions are made costs thousands of GPU-hours, and a wrong "yes" is worse than no answer: it ships into the next model and every later experiment branches from it. Labs therefore review experiment designs before the runs, the way code is reviewed before it is merged. The reviewer's questions are always the same. What exactly changed? What was held fixed, and can you show it from the run cards? Equal what — tokens, parameters, FLOPs or time? Was the baseline tuned as hard as your method? Which split chose the hyperparameters, and which one reported the result? What did you decide, before running, would count as a win? You will answer these questions in every lab of this course.

## The idea

### Hypothesis, baseline, one change, controls

A **hypothesis** is a falsifiable statement with a direction and a size: "At 125M parameters and 2.5B tokens, replacing GQA with MLA lowers 32K decode memory by at least 30% without raising held-out loss by more than 0.02 nats." Its **status** matters: an established effect (we expect to see it), a reported effect (someone saw it, at some scale), or one that may not appear at our scale.

The **baseline** is a specific run card, not "the usual setup". The **changed variable** is one thing. Everything else is a **controlled variable** and must be shown to be identical: data version and hashes, tokenizer, token budget, evaluation windows and version, seed set, software and hardware type. When a change forces others (a new attention kind changes parameter count), the contract says so and picks an axis that makes the comparison fair anyway.

The Google tuning playbook gives the vocabulary for hyperparameters. **Scientific** hyperparameters are the ones whose effect you are measuring (the attention kind). **Nuisance** hyperparameters must be re-optimised for each value of the scientific one so the comparison is fair (usually the learning rate, sometimes warmup and weight decay). **Fixed** hyperparameters are held constant, and every conclusion inherits them as a caveat ("at batch 256 and this schedule").

### Four comparison axes and the questions they answer

| Axis | Holds equal | Answers | Does not answer |
|---|---|---|---|
| Equal tokens | training tokens $D$ | which design learns more per token (sample efficiency) | which is cheaper; a bigger model "wins" by spending more compute |
| Equal parameters | parameter count $N$ (total, or active for MoE) | which design uses capacity better | compute: same $N$ can mean very different FLOPs (attention at long context, MoE routing) |
| Equal training FLOPs | $C \approx F_{\text{train}} \cdot D$ | which design is better for a fixed compute budget; matches how scaling laws are fit | speed on real hardware: a design can save FLOPs and still be slower (memory-bound kernels, immature code) |
| Equal wall-clock | seconds on stated hardware and software | what you can get on this machine today | anything about another machine or next year's kernels; heavily dependent on implementation |

There is no "correct" axis; there is the one that matches the decision. A team choosing an architecture for a fixed compute grant wants equal FLOPs. A team with a deadline on a fixed cluster wants equal wall-clock. Equal parameters is a constraint on the *models* (it says nothing about how long to train), so it is always combined with one of the other three. Equal tokens is the cheapest to run and the easiest to misread. Decode cost is a fifth axis that training comparisons ignore: Kimi K2 section 2.3 chose 64 attention heads over 128 because of an 83% inference-FLOPs difference at 128K context, a cost that equal-training-FLOPs comparisons would not show.

### Tuning budgets

A new method tuned with six learning rates against a baseline run once at its old setting will usually win, and the win measures the tuning, not the method. This is documented, not hypothetical. Melis et al. (2017) re-tuned standard LSTMs with large-scale automatic search and found that, properly regularised, they outperformed several more recent architectures. Narang et al. (2021) re-implemented dozens of Transformer modifications in one codebase and found that most did not meaningfully improve performance, suggesting that reported gains may depend strongly on implementation details. The rule this course uses: **the baseline gets the same tuning budget as the method** (the same sweep over the same nuisance hyperparameters, on the same split), and the contract states that budget.

### Pre-stated decision rules

Write the rule before the runs: what result means *adopt*, what means *reject*, and what means *we cannot tell*. A rule written after the results can always be bent to fit them ("0.01 is meaningful" after seeing 0.021). A good rule names the metric, the interval, and a minimum effect worth acting on. With a 95% interval $[\ell, u]$ for $\Delta = \text{loss}_{\text{new}} - \text{loss}_{\text{base}}$ and a minimum gain $\delta$:

$$\text{adopt if } u < -\delta, \qquad \text{reject if } \ell > -\delta, \qquad \text{otherwise inconclusive.}$$

"Inconclusive" is a legitimate outcome; it means more seeds, a bigger budget or a different question. Lesson 01.4 shows how to compute the interval and how many seeds you need to avoid "inconclusive" for a given $\delta$.

### Splits for data and for decisions

Data-v0 splits documents by a hash of their normalised text (`frontierlab/data/prepare.py`), so exact duplicates cannot leak across splits. Decisions need the same separation:

- **train** — gradients only;
- **validation** — every choice: hyperparameters, checkpoints, early stopping, which variant to keep;
- **test** — read once, for the final pre-registered comparison.

Choosing the best of six learning rates by test loss turns the test split into a validation split, and the reported number is now optimistically biased by the selection. The bias grows with the number of configurations compared and with the noise of each estimate. The same rule applies to evaluation windows: every run is scored on the same fixed windows (`TokenData.eval_windows`), so that differences come from the models, not from which tokens were sampled.

## Worked example

### Matching budgets for two small models

Two models from the lab, both Baseline-0's architecture at vocabulary $V = 8192$, context $T = 128$:

- **small** (`toy`): $C = 128$, 4 layers, non-embedding $N = 787{,}840$;
- **wide**: $C = 256$, 6 layers, $N = 4{,}722{,}688$.

Forward FLOPs per token, $2N + 2LT(Hd) + 2VC$ (lesson 01.1):

- small: $2 \cdot 787{,}840 + 2 \cdot 4 \cdot 128 \cdot 128 + 2 \cdot 8192 \cdot 128 = 1{,}575{,}680 + 131{,}072 + 2{,}097{,}152 = 3{,}803{,}904$
- wide: $2 \cdot 4{,}722{,}688 + 2 \cdot 6 \cdot 128 \cdot 256 + 2 \cdot 8192 \cdot 256 = 9{,}445{,}376 + 393{,}216 + 4{,}194{,}304 = 14{,}032{,}896$

Training FLOPs are 3× these: 11.4M and 42.1M per token, a ratio of 3.69. (In models this small the output head is more than half the cost; at Baseline-0 size it is 19%.)

If wide trains 400 steps of $16 \times 128$ tokens, the budgets for small are:

- equal tokens: 400 steps;
- equal training FLOPs: $400 \cdot 3.69 = 1{,}476$ steps;
- equal wall-clock: $400 \times (\text{tok/s}_{\text{small}} / \text{tok/s}_{\text{wide}})$. In the build run the calibration measured 2,519 and 868 tokens/s (a ratio of 2.90; the laptop was shared with other jobs, so both are lower than on an idle machine), so 1,161 steps.

Equal FLOPs (1,476 steps) and equal wall-clock (1,161) differ by 27% here: the small model runs at lower utilisation (its matrices are too small to keep the cores busy), so a FLOP of small costs more seconds than a FLOP of wide. On a GPU the gap is usually larger, which is why a claim at equal FLOPs and a claim at equal time are different claims.

### Applying a decision rule

Pre-stated rule: adopt if the 95% CI of $\Delta$ lies below $-0.02$. Three outcomes:

- CI $[-0.05, -0.03]$: $u = -0.03 < -0.02$ → adopt;
- CI $[-0.015, 0.01]$: $\ell = -0.015 > -0.02$ → reject (any gain is smaller than the minimum we said was worth acting on);
- CI $[-0.04, 0.02]$: contains $-0.02$ → inconclusive.

## Shapes and cost

Design costs no compute, but it decides how much compute the experiment needs:

- **Arms × seeds × tuning points.** Two arms, 3 seeds and a 4-point learning-rate sweep for each arm is $2 \cdot 3 \cdot 4 = 24$ runs, not 2. At Baseline-0's main-path size (about 1.8 h per run on one H100 at an assumed 30% MFU, lesson 01.1) that is about 43 GPU-hours, PROJECTED. A common saving is to sweep with one seed and then run the chosen point of each arm with 3 seeds: $2 \cdot 4 + 2 \cdot 2 = 12$ runs.
- **Equal-FLOPs runs differ in length.** The cheaper arm trains more steps, so its wall-clock and its checkpoint count differ. The schedule (warmup, decay) must be defined relative to each run's own length, as `compare_axes.py` does (warmup = 10% of steps).
- **Evaluation is shared.** All arms are scored on the same 256 fixed validation windows of shape (256, $T$), int64 on CPU, moved to the model's device in batches; a run's score is a list of 256 per-window losses, kept for the paired comparison of lesson 01.4.

## Build it

The lab code is three functions every later contract relies on: `matched_steps` (budget parity on an axis), `select_then_report` (selection on validation, one read of test) and `decide` (the pre-stated rule). The test for `select_then_report` wraps the test scores in a dictionary that records every key read, and fails if the test split is read for any configuration other than the one chosen on validation — the leak is checked mechanically, not by trust.

The comparison script `labs/module-01/lesson-03/compare_axes.py` wraps the course loop without editing it: it registers a second preset (`toy-wide`) in `frontierlab.model.config.PRESETS` for its own process, measures tokens/s of both models on your machine (20 warm-up steps, 30 timed), then runs `frontierlab.train.loop` for the three arms and scores each on the same 256 validation windows with `frontierlab.evals.window_losses`.

## What the evidence says

- **ESTABLISHED (research practice):** a tuned baseline is necessary for a fair comparison (Melis et al. 2017; Narang et al. 2021, both abstracts); scientific / nuisance / fixed separation (Google tuning playbook); selection on validation and a single read of test (standard practice; REASONABLE INDUSTRY PRACTICE in how labs implement it).
- **PUBLICLY DOCUMENTED:** labs choose axes to match decisions. Llama 3 section 3.2.1 fits scaling laws over compute budgets (equal FLOPs) to choose model size; Kimi K2 section 2.3 justifies its head count by inference FLOPs, a cost axis outside training.
- **What the course experiment shows (measured 2026-10-03, Windows 11 laptop, 16 threads, torch 2.14.1 CPU, Data-v0 CPU size, seed 0, 256 validation windows of 128 tokens):**

  | Arm | Steps | Tokens | Measured wall-clock | Validation loss |
  |---|---|---|---|---|
  | small, equal tokens | 400 | 819,200 | 347 s | 6.165 |
  | wide | 400 | 819,200 | 439 s | 6.073 |
  | small, equal wall-clock | 1,161 | 2,377,728 | 420 s | 5.421 |

  At equal tokens the wide model is 0.092 nats better; at equal wall-clock the small model is 0.652 nats better. The winner flips. The wall-clock arm used 420 s against the wide run's 439 s (within 5%, so the contract's check passes). The small model's equal-tokens run was slowed by other jobs on the machine, which is why its time is far more than 400/1,161 of 420 s: wall-clock measurements need a quiet machine and repeats (Module 2). This is a demonstration of why the axis must be stated, from one seed at about 1M parameters, not evidence about any architecture at scale.

## Lab

**Folder:** [`labs/module-01/lesson-03/`](../../labs/module-01/) · **Time:** about 90 minutes (30 of them unattended) · **Pass check:** `pytest labs/module-01/lesson-03` passes; your contract for the corrected GatedMix experiment covers every field of the template; your axes run reports both comparisons with measured wall-clock.

### Experiment contract

This is the contract for step 4, the axes run. Copy the [template](../../templates/experiment-contract.md) for your own.

- **Question:** for two Baseline-0-shaped models of different width and depth, does the comparison axis (equal tokens vs equal wall-clock on this machine) change which one has lower validation loss? Decision informed: which axis this course's later contracts must name explicitly.
- **Hypothesis:** at equal tokens the wide model wins; at equal wall-clock the small model wins, because it processes about 2.8× more tokens in the same time. Status: established effect in direction (larger models are more sample-efficient; smaller ones are cheaper per token), size unknown at this scale and may be zero on some machines.
- **Baseline:** the small model (`toy` preset) at 400 steps; not tuned (both arms share learning rate 3e-3 and a warmup of 10% of steps — a stated limit).
- **Changed variable:** the model (width 128 → 256, depth 4 → 6). **Controlled:** Data-v0 at the CPU size (hashes in the run cards), batch 16, sequence 128, seed 0, schedule shape, the same 256 validation windows (seed 1234).
- **Comparison axes:** both equal tokens and equal wall-clock (that is the point). Equal FLOPs is printed for reference. Neither axis says anything about other hardware.
- **Budget:** free CPU, three runs, measured 23 minutes in total on a 16-thread laptop; main path about 10 GPU-minutes on any CUDA GPU (projected).
- **Metrics and decision rule:** mean validation loss over 256 windows. One seed per arm, so no interval: the rule is only "report which arm is lower on each axis and the gap"; lesson 01.4 shows why a single-seed gap below about 0.02 at this scale would not support a claim.
- **Correctness checks:** both runs use the same `data_files` hashes and eval windows (compare their `run_card.yaml`); the wall-clock arm's measured seconds are within 15% of the wide run's.
- **Fallback evidence:** if the winner does not flip on your machine, report that; the measured throughput ratio explains why.
- **Limits:** about 1M-parameter models, 1–2M tokens, one seed, one machine; a demonstration of axis dependence, not an architecture result.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | any CUDA GPU (L4, A100, H100); about 10 min. Not run in this build; part of the Module 1 pilot | `python labs/module-01/lesson-03/compare_axes.py --device cuda --batch 64 --seq 512 --steps 2000`. On a GPU the throughput ratio is usually further from the FLOPs ratio than on a CPU; record both |
| Free GPU (Colab/Kaggle T4) | T4 | the main-path command with `--batch 32 --steps 1000` |
| Free CPU | laptop; measured at 23 minutes for the default command (2026-10-03, 16 threads, shared with other jobs) | the default command |

### Steps

1. **Find the flaws.** Read `labs/module-01/lesson-03/flawed_writeup.md`. List every flaw you can find, each with the rule it breaks (one changed variable, controls, axis, tuning budget, splits, uncertainty, decision rule, limits). There are at least ten.
2. **Write the corrected contract.** Fill in the [experiment contract template](../../templates/experiment-contract.md) for a corrected GatedMix experiment on Baseline-0. Pick and justify an axis. State the tuning budget for both arms and the decision rule.
3. **Implement the three functions** in `lab.py` and run `pytest labs/module-01/lesson-03`.
4. **Run the axes comparison** (`python labs/module-01/lesson-03/compare_axes.py`). While it runs, predict the two winners and write down your prediction. Afterwards, compare `runs/l13-axes/*/run_card.yaml` for the two arms and confirm which fields differ.
5. **Write three sentences:** which arm won on each axis, which axis you would use for a team with a fixed cluster and a deadline, and what this run cannot tell you.

<details>
<summary>Hint for step 1</summary>

Go line by line through Setup and Evaluation and ask, for each line: is this the one changed variable, or a second one? Was the baseline given the same treatment? Which split did this decision use? Would this line be the same if the result had gone the other way?

</details>

<details>
<summary>Reference list of flaws (step 1)</summary>

1. Two changed variables beyond the FFN: inner width (122M → 140M parameters) and warmup (200 → 500).
2. No comparison axis stated; with 15% more parameters and 10% more steps, the new arm had more compute.
3. Unequal tuning budget: six learning rates for GatedMix, none for the baseline.
4. Unequal training length (10,500 vs 9,500 steps), chosen after seeing the curve.
5. Different data: Data-v0 was re-prepared from a newer snapshot, so hashes differ from the baseline's run card.
6. The baseline is an old run (last month's code and software), so software versions are uncontrolled.
7. Hyperparameters and checkpoints were selected on the test split, then the test score was reported.
8. Best-of-checkpoints reporting for both arms on test inflates both, by different amounts.
9. Different evaluation windows (128 vs 256), so the scores are not on the same tokens and cannot be paired.
10. One seed per arm: no noise floor, no interval; "significant" has no basis.
11. The decision threshold (0.01) was stated after the results.
12. The conclusion extrapolates to frontier scale with no evidence ("the gain should grow").

</details>

<details>
<summary>Reference contract (step 2), abbreviated</summary>

Question: does GatedMix lower held-out loss versus SwiGLU in Baseline-0 at equal training FLOPs? Hypothesis: reported effect (internal), may not appear. Baseline: Baseline-0 retrained now, same code and data. Changed: the FFN only; GatedMix's width set so non-embedding parameters (and FLOPs per token) match SwiGLU within 1%. Controlled: Data-v0 hashes, 9,500 steps, warmup 200, seed set {0, 1, 2}, Eval v0 windows. Axis: equal FLOPs (and equal parameters by construction); does not answer wall-clock if GatedMix's kernel is slower. Tuning: the same 4-point learning-rate sweep for both arms, seed 0, chosen on validation. Metric: paired bootstrap 95% CI of the per-window loss difference, 3 seeds per arm; noise floor from the baseline seeds. Rule: adopt if CI upper bound < −0.02; reject if lower bound > −0.02; else inconclusive. Test split read once, after the decision. Limits: 122M parameters, 2.5B tokens, one dataset.

</details>

<details>
<summary>Reference solution (code)</summary>

`labs/module-01/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-01/lesson-03`.

</details>

## Common mistakes

- **"Equal parameters" as the whole budget.** It fixes the models, not the run length; you still need tokens, FLOPs or time.
- **Equal FLOPs treated as equal time.** A FLOP of a small or memory-bound model costs more seconds; say which one you matched.
- **Re-tuning only the new method.** Give the baseline the same sweep, or report that it did not get one as a limit.
- **Changing "just one more thing".** Every extra change is a second variable; if it is needed, it is part of a factorial design with its own cells.
- **Peeking at test.** Any choice made after looking at test numbers makes test a validation split.
- **Letting the run length depend on the curve.** Stopping "when it plateaus" differs between arms; fix steps or tokens in the contract.

## References

- G. Melis, C. Dyer, P. Blunsom, *On the State of the Art of Evaluation in Neural Language Models*, 2017. https://arxiv.org/abs/1707.05589
- S. Narang et al., *Do Transformer Modifications Transfer Across Implementations and Applications?*, 2021. https://arxiv.org/abs/2102.11972
- Google Research, *Deep Learning Tuning Playbook*, "Identifying scientific, nuisance, and fixed hyperparameters". https://github.com/google-research/tuning_playbook
- Moonshot AI, *Kimi K2*, section 2.3. https://arxiv.org/abs/2507.20534
- Meta, *The Llama 3 Herd of Models*, section 3.2.1. https://arxiv.org/abs/2407.21783
- Templates: [experiment contract](../../templates/experiment-contract.md), [experiment rubric](../../templates/experiment-rubric.md).

## Next

[01.4 · Uncertainty](lesson-04.md)
