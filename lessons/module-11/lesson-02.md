---
id: "11.2"
module: 11
minutes: 35
practice_minutes: 90
prerequisites: ["11.1", "01.4"]
objectives:
  - Explain why downstream metrics are harder to predict than loss, naming the transformations between them (Schaeffer et al. 2024) and what each does to the signal.
  - Fit the two-step prediction (task loss from scale, accuracy from task loss) on a ladder and test it on a held-out budget against a direct fit of accuracy on compute.
  - Compute principal components of a published benchmark table and test an observational scaling fit on models held out by compute.
  - Tell an "emergent" curve produced by an all-or-nothing metric from one produced by a loss threshold, and choose the metric to report for a decision.
volatility: concept
sources:
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.2.1, Figure 4)"
    url: https://arxiv.org/abs/2407.21783
  - title: "Bhagia et al., Establishing Task Scaling Laws via Compute-Efficient Model Ladders (Eq. 1-2, sections 2-4)"
    url: https://arxiv.org/abs/2412.04403
  - title: "Gadre et al., Language models scale reliably with over-training and on downstream tasks (Eq. 5, section 4)"
    url: https://arxiv.org/abs/2403.08540
  - title: "Ruan, Maddison and Hashimoto, Observational Scaling Laws and the Predictability of Language Model Performance (sections 3-4, Eq. 3-6)"
    url: https://arxiv.org/abs/2405.10938
  - title: "ryoungj/ObsScaling, eval_results/base_llm_benchmark_eval.csv at commit 4d6e1e4 (Apache-2.0)"
    url: https://github.com/ryoungj/ObsScaling/blob/4d6e1e43fd2635d04654aa77d1df9d5266ea0382/eval_results/base_llm_benchmark_eval.csv
  - title: "Wei et al., Emergent Abilities of Large Language Models (section 2)"
    url: https://arxiv.org/abs/2206.07682
  - title: "Schaeffer, Miranda and Koyejo, Are Emergent Abilities of Large Language Models a Mirage? (section 2)"
    url: https://arxiv.org/abs/2304.15004
  - title: "Schaeffer et al., Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?"
    url: https://arxiv.org/abs/2406.04391
  - title: "Du et al., Understanding Emergent Abilities of Language Models from the Loss Perspective (sections 2.3, 4)"
    url: https://arxiv.org/abs/2403.15796
  - title: "OpenAI, GPT-4 Technical Report (sections 3.1-3.2)"
    url: https://arxiv.org/abs/2303.08774
last_verified: "2026-10-06"
---

# 11.2 · Predicting downstream capability

Nobody ships a validation loss. The decision to launch, continue or stop a run is made on benchmark accuracy, and accuracy scales far less tidily than loss: it saturates at chance and at 100%, it moves in steps, and on some tasks it seems to appear from nothing. This lesson builds the two methods labs use to predict it anyway, the two-step method (scale to task loss, task loss to accuracy) and observational scaling laws across many existing models, and takes the emergence debate apart into what the metric does and what the model does. You will score the 11.1 ladder on a small cloze task, predict its largest budget from the smaller ones, and fit an observational law on a published table of 148 models.

## Why this matters at a frontier lab

A run is approved on a forecast like "about 70 on MMLU, a few points better than the last model", not "loss 1.95". Two published forecasts show it can work and how narrowly: the GPT-4 report predicted the final loss from runs with at most 1/10,000 of the compute, and the HumanEval pass rate (as a mean log pass rate on a subset of problems) from runs with at most 1/1,000 (sections 3.1–3.2); Llama 3 predicted the flagship's ARC Challenge accuracy over about four orders of magnitude of compute and "only slightly underestimates" it (section 3.2.1). Both predict a *continuous* quantity first and only then a benchmark. Teams that skip that step, or report an all-or-nothing metric, either cannot forecast at all or see "emergence" that is the metric's doing. Getting this right decides which numbers go in a launch review and how much they can be trusted.

## The idea

### From loss to accuracy is a chain of transformations

Schaeffer et al. (2024) take 5 model families and 12 multiple-choice benchmarks and follow the chain from the model's negative log-likelihoods to accuracy: the log-probability of the correct choice, its probability, the probability renormalised over the offered choices, and finally whether it beats every wrong choice. Each step "progressively degrades the statistical relationship" with scale, because the later metrics depend not only on the correct answer's probability but on "how probability mass fluctuates on the alternative incorrect choices" (abstract). Accuracy is a threshold on a difference of noisy quantities, item by item.

The practical consequence: predict the most continuous quantity that still means something (the NLL of the correct answer), then map it to the metric in a separate, explicitly fitted step.

### The two-step method

Llama 3 (section 3.2.1) first relates the compute-optimal model's normalised NLL of the correct answer on a benchmark to training FLOPs, then relates that NLL to accuracy with a sigmoid, fitted on the scaling-law models together with Llama 2 models. The OLMo ladder (Bhagia et al.) makes both steps explicit:

$$\text{step 1:}\quad L_{\text{task}}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} , \qquad \text{step 2:}\quad \text{Acc}(L_{\text{task}}) = \frac{a}{1 + e^{-k (L_{\text{task}} - L_0)}} + b .$$

$L_{\text{task}}$ is the per-token NLL (they use bits per byte) of the correct answer; $a$, $b$, $k$, $L_0$ are fitted ($k < 0$ so accuracy rises as task loss falls). With a ladder of 190M–1.3B models at 1×, 2×, 5× and 10× Chinchilla's tokens (16 models, about 1% of the targets' compute), they predict the accuracy of OLMo 2 7B (4T tokens) and 13B (5T) "within 2 points of absolute error" on 4 tasks (MMLU, HellaSwag, PIQA, Social IQa), with an average error of 4 points across all 8 tasks and ARC accuracy underestimated (PUBLICLY DOCUMENTED). Gadre et al. use a different step 2, $\text{Err}(L) = \epsilon - k\, e^{-\gamma L}$ (Eq. 5), on the *average* top-1 error over 17 tasks and predict a 6.9B model to within 0.05% from 20× less compute; individual tasks are much noisier (their Table 2), and removing one model from the error fit raises the 6.9B error to 10.6%.

In this course's lab, step 2 uses chance (0.25) and 1.0 as fixed asymptotes, so only $L_0$ and the slope are fitted: with a handful of points, fewer parameters is the honest choice.

### Observational scaling laws

Training a ladder is not the only source of scale. Ruan et al. use public models: 77 base models from 21 families on 8 benchmarks (section 3.2). The benchmark scores are compressed by PCA into a capability vector $S_m$; "top 3 PCs explaining ~97% of the variance", PC1 alone nearly 80%. Then (Eq. 3–6):

$$\sigma^{-1}(E_m) \approx \beta^\top S_m + \alpha , \qquad S_m \approx \theta_f \log C_m + \nu_f ,$$

a capability $E_m$ (an emergent task, an agent benchmark) is a sigmoid of a linear function of the PCs, and within each family $f$ the PCs are linear in log-compute with a family-specific efficiency. Families with very different data (code-heavy, math-heavy) sit at different places for the same compute, which is exactly what a compute-only fit cannot see. They test it on held-out sets of stronger models for emergent capabilities, agentic tasks and chain-of-thought gains, and pre-registered predictions for future models (section 4). The data are public (Apache-2.0), which the lab uses.

### Emergence: the metric or the model?

Wei et al. (section 2): "An ability is emergent if it is not present in smaller models but is present in larger models." Schaeffer, Miranda and Koyejo (2023) argue that many such curves are produced by "nonlinear or discontinuous metrics". If the per-token probability of the right token improves smoothly, $p(N)$, then exact match on an $L$-token answer is about $p^L$, which stays near zero and then rises steeply; token edit distance, about $L(1 - p)$, stays smooth (section 2). Changing multiple-choice grade to the Brier score has the same effect.

Du et al. take the other side with a different variable: plotted against *pretraining loss* instead of parameters or compute, abilities on MMLU, C-Eval, GSM8K and GSM8K-Chinese stay at chance until the loss falls to about 2.2 and then rise, for continuous metrics too (section 2.3); they define emergence as an ability "not present in models with higher pre-training loss but is present in models with lower pre-training loss" (section 4). The two views are not contradictory: a metric can manufacture a jump, and a task can still require the model to cross a loss threshold first. What to do in practice: report a continuous metric next to the headline one, and predict through loss.

## Worked example

### Exact match from per-token accuracy

Per-token accuracy rises from $p = 0.5$ to $0.6$ to $0.8$ across three models (a linear-looking improvement). Exact match on 4 tokens: $0.5^4 = 0.063$, $0.6^4 = 0.130$, $0.8^4 = 0.410$. On 8 tokens: $0.004$, $0.017$, $0.168$. The longer the answer, the later and steeper the "emergence", with no change in the model.

### Multiple-choice metrics on one item

Four options with total log-probabilities $(-10, -11, -12, -13)$, correct option first. Softmax over options: $e^{0} : e^{-1} : e^{-2} : e^{-3} = 1 : 0.368 : 0.135 : 0.050$, sum $1.553$, so $p_{\text{correct}} = 0.644$. Accuracy 1 (it is the largest). Brier score $(0.644 - 1)^2 + 0.237^2 + 0.087^2 + 0.032^2 = 0.127 + 0.056 + 0.008 + 0.001 = 0.191$. NLL of the correct option per token, with 8-token options: $10/8 = 1.25$. If the wrong options each gain 0.6 nats while the correct one stays, $p_{\text{correct}}$ falls to $1/(1 + 0.670 + 0.247 + 0.091) = 0.50$ with the correct-answer NLL unchanged; one more such shift on the closest wrong option and accuracy flips. That is Schaeffer et al.'s mechanism: the metric depends on the wrong options too.

### Two-step by hand

Step 2 fitted as $\text{acc} = 0.25 + 0.75 / (1 + e^{s (x - x_0)})$ with $x_0 = 2.6$, $s = 4$. Step 1 predicts task NLL $x = 2.45$ for the target: $s(x - x_0) = -0.6$, $e^{-0.6} = 0.549$, accuracy $0.25 + 0.75/1.549 = 0.734$. An error of 0.05 nats in step 1 ($x = 2.50$) gives $0.25 + 0.75/1.670 = 0.699$: 3.5 points. The slope of the sigmoid at the target decides how much a small loss error costs in accuracy, which is why predictions are worst in the steep middle of the curve.

### PCA of two benchmarks

Three models score $(0.30, 0.40)$, $(0.50, 0.70)$, $(0.70, 0.70)$ on two benchmarks. Standardised with the population std: benchmark 1 has mean 0.50 and std 0.163, giving $z_1 = (-1.22, 0, 1.22)$; benchmark 2 has mean 0.60 and std 0.141, giving $z_2 = (-1.41, 0.71, 0.71)$. Their correlation is $r = \tfrac{1}{3}\sum z_1 z_2 = \tfrac{1}{3}(1.73 + 0 + 0.87) = 0.866$. For two standardised variables the eigenvalues of the correlation matrix are $1 + r$ and $1 - r$, so PC1 explains $(1 + 0.866)/2 = 93\%$ of the variance.

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| cloze items | contexts (500, 48), options (500, 4, 8), answers (500,), int64, CPU | built once from the test split, seed 0 |
| scored sequences | (2,000, 56) int64 per model, in batches of 128 | logits per batch (128, 56, 1,024) float32 = 29 MB |
| option log-probabilities | (500, 4) float64 | everything else is computed from these |
| greedy decoding for exact match | 4 forward passes over (≤128, 52) per batch | no KV cache: 4 short passes are cheaper than writing one |
| benchmark table | 148 models × 8 benchmarks, float64 | 145 with all six benchmarks used here |

Scoring is forward-only: about $2 N_{\text{total}} \cdot 112{,}000$ tokens of FLOPs per model plus the 4 greedy passes, a few seconds per ladder model on a CPU; the whole `ladder` command is dominated by loading 16 checkpoints.

## Build it

```python
from frontierlab.scaling import downstream as ds, fit
items = ds.build_cloze(TokenData("test", root), n_items=500, ctx=48, cont=8, seed=0)
s = ds.score_cloze(model, items)            # acc, p_correct, brier, nll_correct (+ per-item values)
em = ds.exact_match(model, items, k=4)      # greedy exact match, teacher-forced token accuracy, token_acc**k
nll_fit = fit.fit_parametric(N, D, nll)     # step 1
sig = ds.fit_sigmoid(nll, acc, lo=0.25, hi=1.0)   # step 2
p = ds.pca_capabilities(scores, 3)          # observational: S, loadings, explained variance
f = ds.fit_logistic(S_train, y_train)       # benchmark = sigmoid(w·S + b)
```

The distractors of every cloze item are real text of the same token length from other test documents, so a model can only pick the right one by linking it to the context; a unigram model scores at chance. Correctness checks in `labs/common/tests/test_scaling.py`: the true option of each item continues its context in the source; option log-probabilities equal a direct per-token computation in float64; the metrics match a hand computation; the sigmoid fit recovers planted parameters; PCA recovers a planted one-dimensional structure; the logistic fit recovers planted weights.

## What the evidence says

- **Two-step prediction through task loss: PROMISING.** Llama 3 (one flagship, one task shown), the OLMo ladder (two targets, 8 tasks, errors of 2–4 points, systematic misses on ARC) and Gadre et al. (excellent on an average over 17 tasks, noisy per task) all report it working within limits; none claims it for every task.
- **Observational scaling laws: PROMISING** (one group's method and data, with pre-registered predictions; depends on which benchmarks and families are in the table).
- **Emergence as a metric artefact: debated.** Schaeffer et al. show many jumps disappear under continuous metrics; Du et al. show loss thresholds for some tasks under continuous metrics too. Teach and report both.
- **Course measurement (free CPU, 2026-10-06):** on the 11.1 ladder the two-step prediction of the held-out budget's cloze accuracy was off by 0.012 on average against 0.070 for a direct fit on compute, and 4-token exact match stayed at 0–0.002 while accuracy rose from 0.28 to 0.62. On 148 public models, two components of other benchmarks predicted held-out MMLU to 0.046 against 0.239 from compute, but GSM8K only to 0.178. A 4-way cloze task on 46K–2.2M-parameter models says nothing about any real benchmark, only about the method.

## Lab

**Folder:** [`labs/module-11/lesson-02/`](../../labs/module-11/) · **Time:** about 90 minutes (about 10 of them unattended) · **Pass check:** `pytest labs/module-11/lesson-02` passes; both commands print their tables; your write-up applies the decision rule.

### Experiment contract

- **Question:** from the 11.1 ladder's two smaller budgets, can the cloze accuracy of the largest budget's models be predicted better through task NLL (two-step) than by a direct fit of accuracy on compute? And on published models, do two principal components of other benchmarks predict a held-out benchmark for models above a compute cutoff better than log-compute alone? Decision informed: which prediction lesson 11.3 and the Recipe-R project pre-register for downstream metrics.
- **Hypothesis:** two-step beats direct on held-out runs; PCs beat compute across families. Status: reported effects (OLMo ladder, Llama 3; Ruan et al.); at CPU scale the task is weak and the accuracy range narrow, so the first may not appear.
- **Baseline:** the direct fit (accuracy linear in $\log_{10} C$; and, for published models, a sigmoid of $\log_{10}$ FLOPs).
- **Changed variable:** the predictor. **Controlled:** the same runs, items (500, seed 0, test split), held-out set (the largest budget; models with $\geq 10^{23}$ training FLOPs).
- **Comparison axis:** the same held-out points for every predictor.
- **Budget:** free CPU, evaluation only (measured below); one 28 kB download.
- **Metrics and decision rule:** mean absolute error on the held-out points. Prefer two-step if its error is lower than the direct fit's by more than the items' standard error of accuracy (about 0.02 at 500 items); otherwise report "no difference shown". For the observational part, prefer PCs per benchmark on the same rule with the benchmark's own noise ignored (state that limit).
- **Correctness checks:** `pytest labs/common/tests/test_scaling.py -k "cloze or option or sigmoid or pca or logistic"`; the CSV's SHA-256 matches the pinned value.
- **Limits:** one task, chance-level floor, 16 models, one seed; the observational part is a simplified version of Ruan et al. (no per-family compute fit, PCs from five benchmarks).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× GPU. Not run in this build; part of the Module 11 pilot | `downstream_lab.py ladder --variant main --device cuda --items 1000` on the 11.1 main-path ladder; minutes of GPU time |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --device cuda` on the T4 ladder |
| Free CPU | laptop; measured below | the steps below |

### Steps

1. **Implement** the five TODOs in `lab.py` and run `pytest labs/module-11/lesson-02`.
2. **Ladder:** `python labs/module-11/lesson-02/downstream_lab.py ladder` (needs the 11.1 runs). Before reading the held-out table, write down which predictor you expect to win and why.
3. **Observational:** `python labs/module-11/lesson-02/downstream_lab.py observational`.
4. **Write up:** which metric moved smoothly with compute and which did not; the exact-match column against token accuracy to the fourth power; the held-out errors and the decision; for the published table, the variance explained, the per-benchmark errors and what the largest misses have in common.

<details>
<summary>Hint for TODO 4</summary>

`np.linalg.svd(Z, compute_uv=False)` returns the singular values of the standardised matrix; the variance along each component is proportional to the square of its singular value.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-06/07 on the build laptop (torch 2.14.1+cpu, 16 threads). `ladder`: 69 s for 16 checkpoints, 500 items each. `observational`: 3 s plus a 28 kB download.

**The ladder on the cloze task** (selected rows; chance is 0.25; the 95% interval of each accuracy is about ±0.043 at 500 items):

| run | validation loss | task NLL | $p_{\text{correct}}$ | accuracy | token accuracy | exact match @4 | (token acc)$^4$ |
|---|---|---|---|---|---|---|---|
| m11-r5, $10^{12}$ | 5.967 | 5.960 | 0.279 | 0.284 | 0.036 | 0.000 | 0.000 |
| m11-r2, $10^{12}$ | 4.418 | 4.383 | 0.470 | 0.470 | 0.163 | 0.000 | 0.001 |
| m11-r3, $3 \times 10^{12}$ | 3.926 | 3.900 | 0.547 | 0.546 | 0.226 | 0.000 | 0.003 |
| m11-r4, $10^{13}$ | 3.669 | 3.628 | 0.616 | 0.620 | 0.264 | 0.002 | 0.005 |

Every continuous metric moves with validation loss across all 16 runs; task NLL follows validation loss within 0.08 nats. Accuracy rises from 0.28 to 0.62. Token accuracy rises from 0.04 to 0.27. Exact match on 4 tokens stays at 0.000–0.002 for every model, as $p^4 \le 0.005$ says it must. Judged by exact match, this ladder shows no ability at all. A ladder ten times larger would show it "emerging" with no change in mechanism.

**Two-step against direct, on the held-out $10^{13}$ budget** (fitted on the 11 runs of the two smaller budgets). Step 1 is task NLL $= E + A N^{-\alpha} + B D^{-\beta}$. Step 2 is a sigmoid with $x_0 = 3.53$ and $s = 0.99$, RMS 0.014 in-sample. Mean absolute accuracy error on the 5 held-out runs:

- two-step: **0.012**
- step 2 alone, on the measured NLL: 0.018
- accuracy linear in $\log_{10} C$: 0.070

The direct fit gives every run of a budget the same prediction, 0.534. It cannot see that the size split of a budget matters, and that cost it most of its 0.070. Step 1 itself was off by up to 0.15 nats (m11-r4), but the sigmoid is shallow here ($s \approx 1$), so the accuracy error stayed small. On a steeper part of a real benchmark's curve the same NLL error would cost several points. Two-step beat direct by 0.058, more than the 0.02 item standard error, so the decision is **two-step**. The difference between two-step and step 2 alone (0.006) is below the item noise and means nothing.

**Observational fit on 148 public models** (ObsScaling table at commit 4d6e1e4; 145 models have all six benchmarks used here). Variance explained by PC1–PC4: 0.790, 0.170, 0.024, 0.009, so 98.4% for the top three (Ruan et al. report about 97% with eight benchmarks). PC1 loads almost evenly on every benchmark (+0.29 for TruthfulQA, +0.40 to +0.44 for the rest): it is a general-capability axis. Hold-out: fit on the 79 models below $10^{23}$ training FLOPs, predict the 41 at or above it. Mean absolute error:

| target | 2 PCs of the other 5 benchmarks | log-FLOPs alone |
|---|---|---|
| MMLU | 0.046 | 0.239 |
| ARC-C | 0.023 | 0.094 |
| GSM8K | 0.178 | 0.205 |

Decision: **PCs** for MMLU and ARC-C; **no difference shown** for GSM8K under the rule's spirit (benchmark noise not estimated). The largest GSM8K misses are informative. Yi-34B was predicted 0.95 and measured 0.51. Yi-6B was predicted 0.54 and measured 0.12. StarCoder2-15B was predicted 0.18 and measured 0.52. A math benchmark depends on how much math and code was in the data, and the other benchmarks do not show that well. Ruan et al.'s full method adds family-specific compute efficiencies and a code-and-math component for this reason. Compute alone is a weak predictor across families on every benchmark.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-11/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-11/lesson-02`.

</details>

## Common mistakes

- **Fitting accuracy directly on compute across a jump.** A line or a sigmoid in compute fitted below the jump says nothing about where it is; go through task loss.
- **Reporting exact match alone.** For multi-token answers it is close to $p^L$ and manufactures jumps; report a continuous metric with it.
- **Ignoring the floor.** At chance level accuracy carries no information about scale; points near chance should not drive the step-2 fit.
- **Letting a benchmark predict itself.** In an observational fit, compute the components from the *other* benchmarks, standardised on the training models only.
- **Treating compute as capability across families.** Different data make different capabilities per FLOP (Ruan et al.'s family-specific efficiency); a compute-only fit across families is a weak baseline, not a law.
- **Quoting a two-step error from a 17-task average for one task.** Averages are far more predictable than single tasks (Gadre et al.).

## References

- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, section 3.2.1. https://arxiv.org/abs/2407.21783
- A. Bhagia et al., *Establishing Task Scaling Laws via Compute-Efficient Model Ladders*, 2024, Eq. 1–2, sections 2–4. https://arxiv.org/abs/2412.04403
- S. Y. Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, 2024, Eq. 5, section 4, Table 2. https://arxiv.org/abs/2403.08540
- Y. Ruan, C. J. Maddison and T. Hashimoto, *Observational Scaling Laws and the Predictability of Language Model Performance*, 2024, sections 3–4. https://arxiv.org/abs/2405.10938 ; data: https://github.com/ryoungj/ObsScaling (commit 4d6e1e4, Apache-2.0)
- J. Wei et al., *Emergent Abilities of Large Language Models*, 2022, section 2. https://arxiv.org/abs/2206.07682
- R. Schaeffer, B. Miranda and S. Koyejo, *Are Emergent Abilities of Large Language Models a Mirage?*, 2023, section 2. https://arxiv.org/abs/2304.15004
- R. Schaeffer et al., *Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?*, 2024. https://arxiv.org/abs/2406.04391
- Z. Du et al., *Understanding Emergent Abilities of Language Models from the Loss Perspective*, 2024, sections 2.3, 4. https://arxiv.org/abs/2403.15796
- OpenAI, *GPT-4 Technical Report*, 2023, sections 3.1–3.2. https://arxiv.org/abs/2303.08774
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[11.3 · De-risking a run](lesson-03.md)
