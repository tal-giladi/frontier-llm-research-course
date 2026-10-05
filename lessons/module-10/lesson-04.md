---
id: "10.4"
module: 10
minutes: 40
practice_minutes: 150
prerequisites: ["10.1", "01.4", "07.4"]
objectives:
  - State the data mixing law and RegMix's regression method with their symbols, and what each report measured (scales, numbers of runs, targets, gains).
  - Run a RegMix-style study on CPU (24 tiny runs over four sources), fit linear, quadratic and mixing-law regressions, and judge them by leave-one-out rank correlation rather than in-sample fit.
  - Confirm a predicted-best mixture against uniform and natural mixtures at equal tokens with seeds, and report when the prediction does not survive.
  - Score candidate datasets with micro-anneals from a stable-phase checkpoint against a control anneal, and decide with a target gain and a guard on general held-out loss.
volatility: concept
sources:
  - title: "Ye et al., Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance (Eq. 7, Eq. 8, Algorithm 1, section 4.2)"
    url: https://arxiv.org/abs/2403.16952
  - title: "Liu et al., RegMix: Data Mixture as Regression for Language Model Pre-training (sections 3.1-3.4, Table 2, section 5)"
    url: https://arxiv.org/abs/2407.01492
  - title: "OLMo Team, 2 OLMo 2 Furious (section 4.1 annealing; Table 11; section 4.4.2 microanneals, Table 12; section 4.5 Dolmino Mix 1124, Table 13)"
    url: https://arxiv.org/abs/2501.00656
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.1.3: annealing to assess data quality)"
    url: https://arxiv.org/abs/2407.21783
last_verified: "2026-10-04"
---

# 10.4 · Mixtures and micro-anneals

Once each source is clean (10.1), filtered (10.2) and possibly rephrased (10.3), the remaining question is how much of each to train on. This lesson learns the answer the way published recipes do at small scale: many tiny runs on randomly drawn mixtures, a regression from mixture to loss, a predicted best mixture that is then confirmed with seeds; and, for the end of training, short "micro-anneals" that score a candidate dataset by what it does to a nearly finished model.

## Why this matters at a frontier lab

Mixture weights are among the most consequential and least principled numbers in a pretraining recipe. Setting them by intuition is common, and the published methods show why it is risky: RegMix finds that the source most correlated with downstream quality is web text, not the "high-quality" source a person would pick (section 5.2), and that sources interact in ways "difficult for human experts to fully comprehend" (section 5.4). Each full-scale mixture trial costs a full run. Methods that turn a few dozen cheap runs into a prediction, and short anneals that turn a candidate dataset into a number in hours, are how mixture decisions get evidence at all.

## The idea

### The data mixing law

Ye et al. (Eq. 7) model the validation loss on domain $i$ as a function of the training proportions $r_1, \dots, r_M$ (non-negative, summing to 1):

$$L_i(r) = c_i + k_i \exp\Big(\sum_{j=1}^{M} t_{ij} r_j\Big),$$

with fitted constants $c_i$ (an irreducible part), $k_i > 0$ (a scale) and $t_{ij}$ (how much training domain $j$ helps, $t_{ij} < 0$, or hurts, $t_{ij} > 0$, validation domain $i$). For a validation set that mixes domains with weights $s_i$ (Eq. 8):

$$L(r) = \sum_{i=1}^{K} s_i \Big[ c_i + k_i \exp\Big(\sum_j t_{ij} r_j\Big) \Big].$$

To reach a target scale without running it, the paper nests this with power laws in training steps and in model size (Algorithm 1): fit the step law on small runs, extrapolate each small model to the target steps, fit the size law, extrapolate to the target size, then fit the mixing law on the extrapolated losses. In section 4.2 the target is a 1B model on 100B tokens of RedPajama, evaluated on the Pile; the small runs are 70M–410M models trained for 30B tokens; the optimised mixture reaches "performance comparable to a model trained on default mixture with 48% more steps" (PUBLICLY DOCUMENTED).

### RegMix: regression over many tiny runs

RegMix (section 3) drops the parametric form and regresses directly:

1. **Sample mixtures.** For each proxy run, draw a factor $u \sim \text{Uniform}(0.1, 5.0)$ and a mixture $r \sim \text{Dirichlet}(u \cdot p)$, where $p$ is the sources' natural token distribution; small $u$ gives sparse mixtures, large $u$ near-natural ones (section 3.1).
2. **Train proxies.** 512 models of 1M parameters, 1B tokens each.
3. **Fit** a regression $\hat L(r)$ from mixture to the target loss: ridge regression, and LightGBM (gradient-boosted trees) (section 3.2). The paper judges the fits by **rank correlation** between predicted and measured loss on held-out mixtures (Table 2): about 90% (linear) and 98% (LightGBM) at 1M parameters, 88% and 97% when the same procedure is checked on 1B-parameter models.
4. **Optimise.** Sample many mixtures, predict each, and average the top 100 (section 3.4), which is more robust than the single best point.
5. **Train the large model** (1B parameters, 25B tokens) on the chosen mixture.

The target in the main study is Pile-CC validation loss; the whole search costs about 10% of the FLOPs of the DoReMi baseline the paper compares with (3.5e18 vs 3.7e19, Table 4; PUBLICLY DOCUMENTED).

What makes this work is that ranking mixtures transfers across scale better than absolute losses do. That is also its weak point: if the ranking at 1M parameters differs from the ranking at the target size, the method optimises the wrong thing, which is why the paper checks rank correlation at 1B as well.

### Micro-anneals: scoring a dataset by a short anneal

Most recent recipes end pretraining with an **anneal** (also called mid-training): the learning rate is driven to zero on a different, higher-quality mix. OLMo 2 linearly decays to zero over 50B tokens for its 7B model and 100B or 300B for the 13B and 32B (section 4.1), on Dolmino Mix 1124 (Table 13: filtered DCLM 752B tokens available, FLAN 17B, peS2o 58.6B, Wikipedia and Wikibooks 3.7B, StackExchange 1.26B, math 10.7B, some of them repeated up to 4×), and averages ("soups") several anneals that differ only in data order (section 4.5).

To choose the math sources, OLMo 2 ran **microanneals** (section 4.4.2): pick a candidate source, collect "roughly the same quantity of data from the general data mix", and "train this 50/50 mixture as if it were an annealing run, making sure to linearly drive the learning rate down at the proper rate for this smaller collection of data". They ran 19 of them, 130B tokens in total, fewer than three full 50B anneals, with effects "visible after training for less than 10B tokens", and judged them on a 200-question development subset of GSM8K (GSM*) and on MMLU as a guard. Table 12's three experiments, from a 7B model that had finished pretraining (GSM* baseline 28.5):

| Experiment | Mix | GSM* | MMLU |
|---|---|---|---|
| 1: proportion | math 35/65 web, 576M tokens | 63.5 | 60.1 |
| 1: proportion | math 10/90 web, 1.72B tokens | 61.0 | 60.9 |
| 2: duplication | 2× math | 66.0 | 60.3 |
| 2: duplication | 4× math | 65.0 | 60.5 |
| 3: rewriting | TinyGSM, code-style answers inline | 25.0 | 60.4 |
| 3: rewriting | TinyGSM rewritten in natural language (MIND) | 65.5 | 61.4 |

(The paper's text gives 61 for the 1× math anneal where Table 12 gives 63.5; the table is quoted here.) Three lessons came out: a little domain data in the anneal is enough, some duplication helps, and rewriting into a form the model's pretraining mix supports (natural language rather than code, which was 2% of the mix) can decide whether a source helps or hurts.

Llama 3 used the same idea earlier and at larger scale (section 3.1.3; PUBLICLY DOCUMENTED): it judges "the value of small domain-specific datasets" by "annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens", with "30% weight to the new dataset and the remaining 70% weight to the default data mix", which it finds "more efficient than performing scaling law experiments for every small dataset".

The design point to copy is the **control**: an anneal on the general mix alone, same tokens, same schedule, same starting checkpoint. Annealing by itself improves a model: in OLMo 2's Table 11, a 50B-token anneal on the *pretraining* mix alone raised the OLMES average by 4.4 points but not GSM* (−1.5). A candidate's score is its difference from that control, on its target domain, with a guard on everything else.

## Worked example

### The mixing law, by hand

Two training domains, one validation domain: $c = 2$, $k = 1$, $t = (0, \ln 2)$. Training only on domain 1 ($r = (1, 0)$): $L = 2 + e^{0} = 3$. Half and half: $L = 2 + e^{0.5 \ln 2} = 2 + \sqrt 2 = 3.414$. Only domain 2: $L = 2 + e^{\ln 2} = 4$. Domain 2 hurts this validation set ($t_2 > 0$), and the law says by how much at every proportion.

### RegMix sampling

Natural token shares $p = (0.33, 0.23, 0.24, 0.20)$ for (edu, web, wiki, math) (the CPU sources' training tokens). With $u = 0.2$, $\alpha = (0.066, 0.046, 0.048, 0.040)$: a Dirichlet with all $\alpha_j \ll 1$ puts almost all mass on one or two sources. With $u = 4.5$, $\alpha = (1.5, 1.0, 1.1, 0.9)$: mixtures near the natural one. Drawing $u$ per run covers both regimes, so the regression sees corners and the middle of the simplex.

### Leave-one-out rank correlation

Four runs with measured losses $(5.10, 5.30, 5.20, 5.40)$ and leave-one-out predictions $(5.15, 5.25, 5.28, 5.35)$. Ranks of the measured: $(1, 3, 2, 4)$; of the predicted: $(1, 2, 3, 4)$. Spearman $\rho = 1 - 6 \sum d^2 / (n(n^2 - 1)) = 1 - 6 \cdot 2 / 60 = 0.8$. In-sample fit would always look better; leave-one-out is the honest number for a regression fitted on a few dozen runs.

### Micro-anneal schedule

`frontierlab.datax.train --anneal` with peak rate $\eta = 3 \times 10^{-3}$, 100 steps, 10 warmup: step 0 trains at $3 \times 10^{-4}$, step 9 at $3 \times 10^{-3}$, then $\eta (100 - s)/90$: step 55 at $1.5 \times 10^{-3}$, step 99 at $3.3 \times 10^{-5}$.

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| mixtures | (runs, M) float64, CPU | rounded to the sampler's 1% blocks before fitting: the regression sees the realised weights |
| targets | (runs,) and (runs, K) float64 | mean held-out loss per domain from `frontierlab.datax.evaluate` |
| ridge design | (runs, 1 + M) or (runs, 1 + M + M(M+1)/2) | closed-form solve |
| mixing-law parameters | c, log k, t (M,) float64 | Adam then L-BFGS on squared error |

Cost is dominated by the proxy runs. The CPU study is 24 runs of 200 steps × 2,048 tokens (0.41M tokens each, about 9.8M tokens in all) plus 9 confirmation runs; RegMix's own study is 512 runs of 1B tokens. Each proxy run in the course costs $3 \cdot (2N + \dots)$ FLOPs per token of the toy model, about $1.2 \times 10^7$, so the whole CPU study is about $1.6 \times 10^{14}$ FLOPs. A micro-anneal costs its tokens only; the stable checkpoint is shared by every candidate.

## Build it

```python
from frontierlab.datax import regmix
X = regmix.sample_mixtures(24, prior=[24.1, 16.6, 17.8, 14.5], seed=0)   # RegMix 3.1
fit = regmix.fit_ridge(X, y, quadratic=True, l2=1e-3)                   # or fit_mixing_law(X, y_domain)
print(regmix.rank_corr(regmix.loo_predictions("ridge2", X, y, l2=1e-3), y))
best = regmix.best_mixture(fit, M=4, prior=[24.1, 16.6, 17.8, 14.5])    # average of the top 100 of 100,000
```

```python
from frontierlab.datax import anneal
runs = anneal.run_microanneals("runs/m10/l104", "runs/m10/l104/stable/checkpoint.pt", base_mix,
                               {"math": SourceRef("math", "labs/common/data/m10/math", 1.0)},
                               preset="toy", steps=100, batch=16, seq=128, lr=3e-3, warmup=10, seeds=[0, 1, 2])
```

`run_microanneals` builds the control (the base mix alone) and one 50/50 arm per candidate, and trains each through `frontierlab.datax.train --init-from <stable checkpoint> --anneal` per seed. Correctness checks in `labs/common/tests/test_datax.py`: the mixing-law fit recovers a planted law exactly; ridge leave-one-out ranks a planted law correctly; `best_mixture` returns a point on the simplex that favours the source with the most negative coefficient; the anneal schedule reaches zero at the last step; a wrapper run started with `--init-from` records the parent run and step.

## What the evidence says

- **Choosing mixtures by regression over small runs: PROMISING.** RegMix and the data mixing law each report gains in their own setting (PUBLICLY DOCUMENTED, sections cited); both rest on the assumption that small-scale rankings transfer, which RegMix checks up to 1B parameters.
- **Annealing on a higher-quality mix at the end of pretraining: ESTABLISHED** (OLMo 2 section 4.1 and many other reports describe it).
- **Short anneals to score datasets: PROMISING**, documented by two labs (OLMo 2 section 4.4.2 with a 200-question development set, whose text and table disagree on one number; Llama 3 section 3.1.3, without published per-dataset results).
- **Course measurement (free CPU, 2026-10-05, other jobs running):** 24 RegMix proxy runs of 0.41M tokens. Leave-one-out Spearman: plain ridge 0.09, quadratic ridge 0.83, the mixing law 0.81. The quadratic ridge's predicted best mixture (Wikipedia 0.59, FineMath 0.40, almost no Data-v0 or web) was predicted 0.12 nats better than uniform; confirmed with 3 seeds it was 0.07 nats *worse* on the target, 95% CI [−0.14, +0.28]: inconclusive, with a seed standard deviation (0.072) as large as most of the differences the regression was ranking. Micro-anneals from a 400-step stable checkpoint, 100 steps, 3 seeds: the control anneal alone lowered Data-v0 loss by 0.105 nats; against it, Wikipedia improved its own domain by 0.117 [0.102, 0.131] with a guard of +0.043 [−0.009, +0.095] (inconclusive); FineMath improved its own by 1.07 [0.91, 1.24] but cost +0.097 [+0.050, +0.144] on Data-v0 (reject at a 0.02 budget). A good in-sample regression and a decent leave-one-out rank did not give a confirmed mixture at this scale; the confirmation step is what showed it.

## Lab

**Folder:** [`labs/module-10/lesson-04/`](../../labs/module-10/) · **Time:** about 150 minutes (about 105 of them unattended) · **Pass check:** `pytest labs/module-10/lesson-04` passes; `mixture_lab.py regmix` and `mixture_lab.py anneal` print their tables; your write-up applies both decision rules.

### Experiment contract

**Part 1, RegMix.**

- **Question:** can 24 tiny runs predict which mixture of Data-v0, FineWeb, Wikipedia and FineMath minimises the average of the four validation losses, well enough that the predicted mixture beats the uniform and natural mixtures at equal tokens? Decision informed: the starting weights of Data-v1.
- **Hypothesis:** the fits rank held-out runs with Spearman above 0.8 (leave-one-out); the predicted mixture beats uniform by more than the noise floor. Status: reported effect (RegMix Table 2); the transfer from these runs to longer ones is a hypothesis this lab tests only at one scale.
- **Baseline:** the uniform mixture (0.25 each); also the natural mixture (token-proportional).
- **Changed variable:** mixture weights. **Controlled:** toy preset, 200 steps of 16 × 128 tokens, warmup 20, cosine, learning rate 3e-3; the regression runs all use seed 0 (the mixture is the variable); the confirmation uses seeds 0–2.
- **Comparison axis:** equal tokens.
- **Budget:** free CPU, 33 runs (measured below).
- **Metrics and decision rule:** fit quality: leave-one-out Spearman and RMSE of each regression. Confirmation: seed-level paired 95% interval of (predicted best − uniform) on the average validation loss; adopt the predicted mixture if the upper bound is below 0, reject if the lower bound is above 0, else inconclusive.
- **Correctness checks:** `pytest labs/common/tests/test_datax.py -k regmix`; every run's `mixture_accounting.json` equals its spec's realised weights; the regression inputs are the realised (rounded) weights.
- **Limits:** 4 sources, 24 runs (RegMix used 512), one tiny model, short runs; the target weights all four domains equally, which is a choice, not a fact about what matters.

**Part 2, micro-anneals.**

- **Question:** does a 50/50 micro-anneal with Wikipedia or FineMath improve its target domain against a control anneal, without costing more than 0.02 nats on Data-v0? Decision informed: which candidate sources go into Data-v1's final anneal mix.
- **Hypothesis:** each candidate lowers loss on its own domain by much more than the noise; FineMath costs some Data-v0 loss, Wikipedia little. Status: reported effect (OLMo 2 section 4.4.2) at 7B; the size of the guard cost is unknown at this scale.
- **Baseline:** the control anneal (edu + web only), same starting checkpoint, tokens and schedule.
- **Changed variable:** the candidate half of the anneal mix. **Controlled:** the stable checkpoint (400 steps on edu + web at a constant learning rate), 100 anneal steps with linear decay to zero after 10 warmup steps, seeds 0–2 (data order), evaluation sets.
- **Comparison axis:** equal tokens.
- **Metrics and decision rule:** target: (candidate − control) loss on the candidate's domain, seed-level 95% interval; guard: the same on Data-v0. Adopt if the target upper bound is below 0 and the guard upper bound is at most 0.02; reject if the target lower bound is at least 0 or the guard lower bound is above 0.02; else inconclusive.
- **Correctness checks:** every anneal's run card names the stable checkpoint and its step; the control and the candidate arms of one seed differ only in the mixture.
- **Limits:** the anneal starts from the weights with a fresh optimizer (`--init-from`), not from the full optimizer state; 0.2M anneal tokens; one stable checkpoint.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100. Not run in this build; part of the Module 10 pilot | `mixture_lab.py regmix --variant main` (64 runs of pilot-10m, 2,000 steps × 64 × 1,024 = 131M tokens each, plus 9 confirmation runs) and `mixture_lab.py anneal --variant main`. PROJECTED: 73 runs × 1.31e8 tokens × `flops_per_token(pilot_10m(32768), 1024)` ÷ (989e12 × 0.3) ≈ 1.3 GPU-hours |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4` (32 proxy runs of pilot-10m at 512 tokens) |
| Free CPU | laptop; measured: 33 runs of 74–242 s for `regmix` (about 1.5 hours on a quiet machine), 14 minutes for `anneal` | the steps below |

### Steps

1. **Implement** the seven TODOs in `lab.py`: RegMix sampling, ridge fit and prediction, the mixing law, leave-one-out, the anneal schedule, the anneal decision. Run `pytest labs/module-10/lesson-04`.
2. **RegMix** (unattended): `python labs/module-10/lesson-04/mixture_lab.py regmix`. Before it reaches the confirmation, write down which mixture you expect to win and why.
3. **Micro-anneals** (unattended): `python labs/module-10/lesson-04/mixture_lab.py anneal`.
4. **Write up:** the leave-one-out correlations of the three regressions and which you would trust; the fitted $t_{ij}$ of the mixing law (which sources help which domains); the confirmation decision and how far the predicted loss was from the measured one; the decision for each anneal candidate, and what the guard column shows.

<details>
<summary>Hint for TODO 1</summary>

Draw in exactly this order per row: `u = rng.uniform(lo, hi)`, then `rng.dirichlet(np.maximum(prior * u, 1e-3))`; any other order gives different (valid) mixtures and fails the comparison with the reference.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-05 on the build laptop (torch 2.14.1 CPU, 16 threads shared with another job's runs). RegMix: 24 runs of 200 steps × 16 × 128 tokens (74–242 s each; two runs include stalls of 0.5 and 1.9 hours while the machine was busy or asleep, which do not change their results because resume is exact) plus 9 confirmation runs; scoring 128 windows of 128 tokens per source. Micro-anneals: stable run 400 steps at a constant rate, 330 s; 9 anneals of 100 steps, 47–58 s each; 850 s for the whole `anneal` command.

**RegMix.** The target (mean of the four validation losses) ranged from 6.630 to 7.147 over the 24 mixtures. The five best runs all had 30–40% FineMath and a lot of web; the five worst were single-source corners (all Data-v0 6.989, 95% Wikipedia 7.091, 71% FineMath 7.147). Leave-one-out:

| Regression | Spearman | RMSE |
|---|---|---|
| ridge, linear (your TODO) | +0.090 | 0.198 |
| ridge with pairwise terms | +0.826 | 0.087 |
| data mixing law (Eq. 8, sum of per-domain Eq. 7 fits) | +0.814 | 0.092 |

A linear model cannot represent "a mixture beats every corner", which is exactly what the runs show; both curved models can. Fitted $t_{ij}$ of the law, rows = validation domain, columns = (edu, web, wiki, math): edu (−4.1, −8.9, −5.9, +5.5), wiki (−0.1, +0.3, −4.7, +2.9), math (+2.7, +2.9, +2.9, −6.2); FineMath hurts the other domains and helps only itself. (The web row has $k \approx 0$ and coefficients of ±10 to ±50: that fit is degenerate, a reminder to read fitted parameters before trusting them.) Predicted best (top-100 average of 100,000 sampled mixtures under the quadratic ridge): (0.004, 0.010, 0.590, 0.396), predicted 6.566; under the linear ridge: all web, a corner.

Confirmation, 3 seeds, at the same budget (mean target per seed 0, 1, 2; difference against uniform):

| Arm | target per seed | − uniform, 95% CI |
|---|---|---|
| uniform (0.25 each) | 6.649, 6.734, 6.793 | (baseline) |
| natural (0.33, 0.23, 0.24, 0.20) | 6.682, 6.795, 6.696 | −0.001 [−0.210, +0.209] |
| predicted best | 6.760, 6.861, 6.767 | +0.071 [−0.139, +0.280] |

Decision: **inconclusive** (the rule needed the upper bound below 0). Predicted vs measured mean target: best 6.566 vs 6.796, uniform 6.688 vs 6.725, natural 6.736 vs 6.725. The regression got the two near-uniform mixtures about right and was 0.23 nats optimistic about the corner-like mixture it chose: the optimiser went where the 24 runs said least, which is the mistake the top-100 average only partly protects against. The seed standard deviation of the uniform arm is 0.072 nats, while the 24 regression runs all used seed 0: part of what the regression "learned" is seed-0 noise. With this budget the honest output is "no mixture shown better than uniform or natural", and the next step is more proxy runs with two seeds each, not a bigger optimiser.

**Micro-anneals.** Stable checkpoint losses: edu 6.489, web 6.557, wiki 6.651, math 6.989. The control anneal (edu + web, 100 steps decayed to zero) reached edu 6.383, web 6.451, wiki 6.529, math 6.975: −0.105 on Data-v0 from the decay alone, the effect OLMo 2's Table 11 shows at 7B. Against the control (seed-level 95% intervals):

| Candidate (50/50 with the base mix) | own domain | guard: Data-v0 | web | other |
|---|---|---|---|---|
| Wikipedia | −0.117 [−0.131, −0.102] | +0.043 [−0.009, +0.095] | +0.060 [+0.037, +0.083] | math −0.053 [−0.167, +0.061] |
| FineMath | −1.075 [−1.236, −0.913] | +0.097 [+0.050, +0.144] | +0.112 [+0.096, +0.128] | wiki +0.083 [+0.040, +0.125] |

Decisions: Wikipedia **inconclusive** (the guard interval straddles the 0.02 budget), FineMath **reject** (its guard lower bound, 0.050, is above the budget). Without the control, the Wikipedia anneal would have looked like a gain on every domain (edu 6.427 vs the stable 6.489). Half the anneal tokens going to one narrow domain is a lot at this scale; OLMo 2's 10/90 math anneal is the obvious next arm, pre-stated.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-10/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-10/lesson-04`.

</details>

## Common mistakes

- **Judging a regression by its in-sample fit.** With 24 runs and 11 quadratic coefficients any fit looks good in-sample; use leave-one-out or held-out runs.
- **Fitting the requested weights instead of the realised ones.** A sampler that rounds to 1% blocks trains on the rounded mixture.
- **Trusting the single best predicted point.** It sits where the regression is least constrained; average the top predictions and confirm with seeds.
- **A micro-anneal without a control anneal.** Decaying the learning rate improves the model on its own; the candidate's effect is its difference from the control.
- **Scoring an anneal only on its target.** OLMo 2 watched MMLU while optimising GSM*; keep a guard on general held-out loss.
- **Optimising a target nobody chose deliberately.** "Average of four validation losses" weights a 3% domain like a 40% one; state the target in the contract.

## References

- J. Ye et al., *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance*, 2024, Eq. 7–8, Algorithm 1, section 4.2. https://arxiv.org/abs/2403.16952
- Q. Liu et al., *RegMix: Data Mixture as Regression for Language Model Pre-training*, 2024, sections 3, 5, Tables 2 and 4. https://arxiv.org/abs/2407.01492
- OLMo Team, *2 OLMo 2 Furious*, 2024, sections 4.1, 4.4.2, 4.5, Tables 11–13. https://arxiv.org/abs/2501.00656
- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, section 3.1.3. https://arxiv.org/abs/2407.21783
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[10.5 · Continued training and forgetting](lesson-05.md)
