---
id: "11.1"
module: 11
minutes: 40
practice_minutes: 120
prerequisites: ["01.4", "02.1", "07.3", "07.4", "10.4"]
objectives:
  - Derive the compute-optimal model size and token count from the Chinchilla loss form by hand, and explain why the published approach-3 constants and its replication give different answers at the same budget.
  - Explain the three measurement choices that made Kaplan et al. and Hoffmann et al. disagree (parameter counting, warmup, tuning across scale), with the numbers Porian et al. report for each.
  - Run an iso-FLOP ladder on the CPU, find the loss-minimising size per budget, fit the allocation exponent and the parametric law, and test the law on a held-out budget.
  - Price over-training (released small models trained on thousands of tokens per parameter) and repeated data in compute and in loss, and state when each is the right choice.
volatility: concept
sources:
  - title: "Kaplan et al., Scaling Laws for Neural Language Models (sections 1.2, 2.1-2.2, 6; Eq. 1.1-1.5, 6.1; Table 5)"
    url: https://arxiv.org/abs/2001.08361
  - title: "Hoffmann et al., Training Compute-Optimal Large Language Models (section 3, Table 2, Table 3, appendix B, appendix D.2 Eq. 10)"
    url: https://arxiv.org/abs/2203.15556
  - title: "Besiroglu et al., Chinchilla Scaling: A replication attempt (Table 1)"
    url: https://arxiv.org/abs/2404.10102
  - title: "Porian et al., Resolving Discrepancies in Compute-Optimal Scaling of Language Models (sections 3.1-3.5, Table 1)"
    url: https://arxiv.org/abs/2406.19146
  - title: "Pearce and Song, Reconciling Kaplan and Chinchilla Scaling Laws"
    url: https://arxiv.org/abs/2406.12907
  - title: "Muennighoff et al., Scaling Data-Constrained Language Models (section 3.1 Eq. 5, section 6, appendix A)"
    url: https://arxiv.org/abs/2305.16264
  - title: "Sardana et al., Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws (section 2, Figure 1)"
    url: https://arxiv.org/abs/2401.00448
  - title: "Gadre et al., Language models scale reliably with over-training and on downstream tasks (sections 2.2-2.3, 4)"
    url: https://arxiv.org/abs/2403.08540
  - title: "Meta, Introducing Meta Llama 3 (blog, 2024-04-18: Chinchilla-optimal ~200B tokens for 8B; trained on up to 15T)"
    url: https://ai.meta.com/blog/meta-llama-3/
  - title: "DeepSeek-AI, DeepSeek LLM: Scaling Open-Source Language Models with Longtermism (sections 3.2-3.3, Table 4)"
    url: https://arxiv.org/abs/2401.02954
  - title: "Hu et al., MiniCPM (section 4.5: 192 tokens per parameter under WSD)"
    url: https://arxiv.org/abs/2404.06395
last_verified: "2026-10-06"
---

# 11.1 · Compute-optimal and over-trained regimes

A scaling law turns a few cheap runs into a statement about an expensive one: for a given training budget, how large a model, on how many tokens, and what loss to expect. This lesson derives the compute-optimal allocation from the Chinchilla loss form, explains why the two classic studies disagreed and how later work reconciled them, and then asks why almost every released small model is trained far past that optimum. You will run an iso-FLOP ladder of 16 tiny models on the CPU, fit both of Chinchilla's main estimators to it, test the fit on a budget it never saw, and price over-training and repeated data with published constants.

## Why this matters at a frontier lab

The final pretraining run is the most expensive thing a lab does, and its size and token count are fixed before it starts. Get the allocation wrong by a factor of two and the run either wastes compute or produces a model that costs too much to serve. Scaling laws are how the decision gets evidence, but they come with traps that have misled careful groups: Kaplan et al. concluded that model size should grow as $C^{0.73}$; Hoffmann et al., two years later, found $C^{0.50}$, which moved the field from GPT-3-style models (175B parameters on 300B tokens) to Chinchilla-style ones. Most of the gap came from measurement choices, not from nature. Knowing which choices move the exponent, and which number a released model's token count is actually optimising, is what lets an engineer read a scaling-law plot as evidence instead of as a rule.

## The idea

### The loss as a function of parameters and tokens

Hoffmann et al. (section 3.3, Eq. 2) model the final loss of a run with $N$ parameters trained on $D$ tokens as

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} .$$

$E$ is the loss an infinitely large model trained forever would reach on this data (the entropy of the text, in their reading); $A/N^\alpha$ is what a finite model adds; $B/D^\beta$ is what a finite dataset adds. $A, B, \alpha, \beta, E$ are fitted. Their approach-3 fit (appendix D.2, Eq. 10) is $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$ (PUBLICLY DOCUMENTED).

Training compute is $C \approx 6ND$ FLOPs: each parameter costs 2 FLOPs per token in the forward pass and 4 in the backward pass (parent course lesson 07.4). Minimising $L$ at fixed $C$ (substitute $D = C/(6N)$, set $\partial L / \partial N = 0$) gives

$$N_{\text{opt}}(C) = G \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \qquad D_{\text{opt}}(C) = G^{-1} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}, \qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}} .$$

The exponents are the whole story: if $\alpha = \beta$, parameters and tokens grow equally ($a = b = 0.5$), which is the abstract's "for every doubling of model size the number of training tokens should also be doubled".

### Three ways to estimate it

Hoffmann et al. estimate the exponents three ways (section 3; their Table 2 gives $a$ in $N_{\text{opt}} \propto C^a$ and $b$ in $D_{\text{opt}} \propto C^b$):

1. **Training curves envelope.** For each model size, several runs of different lengths; at each compute, the lowest loss over all runs. $a = 0.50$, $b = 0.50$.
2. **IsoFLOP profiles.** At each of nine fixed budgets, train a range of sizes, each for exactly the tokens the budget allows; fit a parabola of loss against $\log N$ and take its vertex as $N_{\text{opt}}(C)$; fit a line through $\log N_{\text{opt}}$ against $\log C$. $a = 0.49$, $b = 0.51$.
3. **Parametric fit.** Fit $L(N, D)$ above to all runs at once, minimising a Huber loss ($\delta = 10^{-3}$) on log-loss residuals with L-BFGS from a grid of starting points (Eq. 3), then use the closed form. $a = 0.46$, $b = 0.54$.

The approaches do not agree exactly, and the third has its own history. Besiroglu et al. reconstructed the data from Hoffmann et al.'s figures and refitted (Table 1): $E = 1.8172$, $A = 482.01$, $B = 2085.43$, $\alpha = 0.3478$, $\beta = 0.3658$. They report that the original approach-3 estimates "are inconsistent with their first two estimation methods, fail at fitting the extracted data, and report implausibly narrow confidence intervals", that the published parameters imply about 70 tokens per parameter while the refit gives about 20, consistent with approaches 1 and 2 and with Chinchilla itself (70B parameters on 1.4T tokens; PUBLICLY DOCUMENTED, with the replication's numbers as Epoch AI's estimates). The worked example computes both.

The "20 tokens per parameter" rule is not written in the Chinchilla paper; it is read off Table 3 (1B parameters ↔ 20.2B tokens, 10B ↔ 205.1B).

### Why Kaplan got 0.73

Kaplan et al. (Eq. 6.1, Table 5) found $N_{\text{opt}} \propto C^{0.73}$ and $D \propto C^{0.27}$. Two independent reconciliations agree that the difference is mostly method:

- **Counting.** Kaplan's $N$ excludes embeddings and his compute excludes the output head (section 1.3, 2.1). Pearce and Song show that Chinchilla's own law, viewed at Kaplan's small sizes in non-embedding terms, has a local exponent of about 0.78, and about 0.51 in total-parameter terms: "Kaplan counting non-embedding rather than total parameters, combined with their analysis being performed at small scale".
- **Three training choices.** Porian et al. reproduce Kaplan's exponent and remove the gap one factor at a time (Table 1, OpenWebText2 / RefinedWeb): reproduction 0.864 / 0.835; counting the output head's FLOPs 0.699 / 0.706; warmup scaled with model size (warmup tokens = $N$) instead of a fixed 3,000 steps 0.603 / 0.602; cosine decay matched to each run's length 0.574 / 0.571; learning rate, batch size and AdamW $\beta_2$ tuned per size 0.518 / 0.497. On the last point: $\beta_2 = 0.95$ "is suboptimal at smaller batch sizes (128 and below)".

The cosine-length point is Hoffmann et al.'s own appendix B: "setting the cosine cycle length too much longer than the target number of training steps results in sub-optimally trained models"; overshooting by more than 25% gives clear drops. A ladder whose small runs are read off the middle of a longer schedule underrates them. Lesson 07.4's WSD schedule is the modern way around this: one stable run per size and a short cooldown from several of its checkpoints (Hägele et al., arXiv 2405.18392, report that this halves the FLOPs of a scaling-law suite).

### Why released models are over-trained

The compute-optimal point minimises *training* compute for a loss. A model that will be served pays $2N$ FLOPs per generated token forever after. Sardana et al. (section 2) minimise training plus inference compute,

$$\min_N \; 6 N D(N) + 2 N D_{\text{inf}} \quad \text{subject to } L(N, D(N)) = \ell ,$$

and the answer moves to smaller models trained longer as the expected inference demand $D_{\text{inf}}$ grows; with "~1B requests" the optimum is already well past Chinchilla (abstract). Quality kept improving up to the 10,000 tokens per parameter they tested.

Released models show it. Meta's Llama 3 blog (company claim) says the Chinchilla-optimal amount for an 8B model is about 200B tokens and that the 8B and 70B models "continued to improve log-linearly" up to 15T tokens, about 1,875 tokens per parameter for the 8B. Gemma 3 1B is trained on 2T tokens (report section 2.2, with distillation), Qwen3-0.6B is part of a family trained on about 36T (report section 3.1). Over-training is predictable: Gadre et al. fit losses for runs up to 640 tokens per parameter and predict a 1.4B model at that ratio (900B tokens) to 0.7% relative error from runs with 300× less compute (section 4). The optimum also depends on the data and the schedule: DeepSeek LLM finds the allocation exponent for model size rises with data quality (0.450 → 0.524 → 0.578 for its early data, its current data and OpenWebText2, Table 4), and MiniCPM, with WSD, finds about 192 tokens per parameter (section 4.5).

### When the data runs out

If the training set has only $U$ unique tokens, a run of $D > U$ tokens repeats them. Muennighoff et al. (section 3.1, Eq. 5) model the repeated tokens as worth less:

$$D' = U + U R^{*} \left(1 - e^{-R/R^{*}}\right), \qquad R = \frac{D}{U} - 1 ,$$

with $R^* \approx 15.4$ fitted (appendix A). Their abstract: "up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data"; returns diminish fast after about 16 epochs and "at 40 epochs, repeating is worthless" (Figure 1). This is the regime of every high-quality source a lab has more compute than tokens for: math, code, a curated anneal mix (the 2× and 4× repetitions of lesson 10.4).

## Worked example

### Chinchilla's optimum at Gopher's budget

$C = 5.76 \times 10^{23}$ FLOPs (Gopher's budget, section 3.1), with Eq. 10's constants. $\alpha A = 0.34 \cdot 406.4 = 138.2$; $\beta B = 0.28 \cdot 410.7 = 115.0$; ratio $1.202$; $1/(\alpha + \beta) = 1.613$; $G = 1.202^{1.613} = 1.345$. Exponent $\beta/(\alpha + \beta) = 0.452$; $C/6 = 9.6 \times 10^{22}$; $(9.6 \times 10^{22})^{0.452} = 2.39 \times 10^{10}$. So $N_{\text{opt}} = 1.345 \cdot 2.39 \times 10^{10} = 3.2 \times 10^{10}$ (32B) and $D_{\text{opt}} = 9.6 \times 10^{22} / 3.2 \times 10^{10} = 3.0 \times 10^{12}$ tokens: 93 tokens per parameter. With Besiroglu et al.'s refit the same budget gives 72B parameters on 1.33T tokens (18 per parameter), close to Chinchilla's 70B on 1.4T and Table 3's 67B on 1.5T. Same functional form, same data, two sets of constants, a factor of five in tokens per parameter: the reason to distrust a single fit's extrapolation.

### Chinchilla against Gopher with the same law

$L(70 \times 10^9, 1.4 \times 10^{12}) = 1.69 + 406.4/4869 + 410.7/2517 = 1.69 + 0.083 + 0.163 = 1.937$. $L(280 \times 10^9, 300 \times 10^9) = 1.69 + 406.4/7798 + 410.7/1635 = 1.69 + 0.052 + 0.251 = 1.993$. At the same compute the 4× smaller model on 4.7× the tokens is 0.056 nats better, almost all from the data term.

### An iso-FLOP vertex

Three runs at one budget: $\log_{10} N = 5.0, 5.5, 6.0$ with losses $4.20, 3.95, 4.00$. The parabola through them has curvature $c_2 = (4.20 - 2 \cdot 3.95 + 4.00)/(2 \cdot 0.5^2) = 0.6$ and slope at the middle point $(4.00 - 4.20)/1.0 = -0.2$, so the vertex is at $5.5 + 0.2/(2 \cdot 0.6) = 5.667$: $N_{\text{opt}} = 10^{5.667} = 4.6 \times 10^5$.

### The price of over-training

Llama 3 8B at 15T tokens: $C = 6 \cdot 8 \times 10^9 \cdot 1.5 \times 10^{13} = 7.2 \times 10^{23}$. Under Eq. 10 its loss is 1.949; a compute-optimal run reaches 1.949 with about half that compute, so the 8B spent about 99% more training compute than necessary for its loss (451% under the refit, which penalises over-training harder because its $\beta$ is larger). What the extra compute buys is a model about 3–5× smaller than the compute-optimal one of equal loss, which is cheaper at every token served. The lab's `published` command prints these numbers from your functions; they are an extrapolation of a law fitted on other data and should be read as orders of magnitude.

### Repetition

Four epochs: $R = 3$, $D' = U (1 + 15.39 (1 - e^{-3/15.39})) = U(1 + 15.39 \cdot 0.177) = 3.73\,U$: 93% of the value of 4U fresh tokens. Sixteen epochs: $R = 15$, $D' = U(1 + 15.39 \cdot 0.623) = 10.6\,U$, 66% of 16U. Forty epochs: 38%.

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| a ladder run | one training run; its record is `metrics.jsonl` and `run_card.yaml` | `budget.train_flops` from the exact accounting (`frontierlab.attention.accounting.flops_per_token`), not $6ND$ |
| run table | 16 rows × (budget, $N_{\text{total}}$, $N_{\text{nonemb}}$, $D$, $C$, loss), float64, CPU | |
| iso-FLOP fit | (3,) parabola coefficients per budget, float64 | needs at least 3 sizes per budget, on both sides of the minimum |
| parametric fit | 5 parameters, float64 | grid over $(\alpha, \beta)$ (47 × 47) with non-negative least squares for $(E, A, B)$, then L-BFGS on the Huber objective |

At the CPU sizes the exact count matters more than at large scale. The rung `m11-r3` has 98,752 non-embedding and 164,288 total parameters at the lab's vocabulary of 1,024; its training FLOPs per token at context 128 are $1.08 \times 10^6$, while $6 N_{\text{nonemb}} = 5.9 \times 10^5$ and $6 N_{\text{total}} = 9.9 \times 10^5$: the tied output head ($6 V C = 3.9 \times 10^5$ per token) is a third of the compute. That is Porian et al.'s first factor in miniature. At the course's original vocabulary of 8,192 the head is 80% of the smallest rungs, and a CPU pilot (below) found that every budget the CPU can afford is minimised by the smallest model, so no iso-FLOP minimum can be measured. The lab therefore retokenizes Data-v0 with a 1,024-token vocabulary (`python -m frontierlab.data.prepare --vocab 1024 --out labs/common/data/m11-v1024`, same documents and splits).

The CPU grid is 16 runs totalling $7.3 \times 10^{13}$ FLOPs (rungs with more tokens than $3 \times 10^7$ are dropped so no ladder run repeats data). The main-path grid (`--variant main`: pilot-10m, pilot-30m, pilot-70m, Baseline-0 and a 191M-parameter rung at $10^{17}$, $3 \times 10^{17}$, $10^{18}$ FLOPs, sequence 1,024) is 14 runs and $7.0 \times 10^{18}$ FLOPs: PROJECTED $7.0 \times 10^{18} / (989 \times 10^{12} \cdot 0.30) / 3600 \approx 6.6$ H100-hours at an assumed 30% MFU, pending the Module 11 pilot.

## Build it

```python
from frontierlab.scaling import laws, ladder, fit
laws.compute_optimal(5.76e23)                    # N, D, tokens_per_param, loss, exponents
laws.overtraining(8e9, 15e12)                    # loss, the optimal loss at the same C, overhead
laws.inference_aware(L_target=2.0, D_inference=1e13)
laws.effective_data(U=1.0, D_total=16.0)         # 10.6 (of 16)

plan = ladder.plan_isoflop([1e12, 3e12, 1e13], ["m11-r1", "m11-r2", "m11-r3", "m11-r4", "m11-r5", "m11-r6"],
                           seq=128, batch=16, vocab_size=1024)
mins = fit.isoflop_minima(runs, key_N="N_total")  # approach 2
f = fit.fit_parametric(N, D, L)                   # approach 3
fit.holdout(N, D, L, test_mask=budget == 1e13)     # fit on the rest, error on the held-out budget
```

`frontierlab.scaling.ladder` registers the ladder sizes as presets (`m11-r1` … `m11-r7` on CPU, `m11-200m`, `m11-350m`, `m11-1b` on the main path) at run time, so every course training wrapper accepts them; `python -m frontierlab.scaling.train` adds `--unique-tokens U` (windows only from the first $U$ training tokens) for the repetition experiment and `--via optim|datax` to run the Module 7 or Module 10 wrappers. Correctness checks in `labs/common/tests/test_scaling.py`: the closed form agrees with a brute-force minimum; the compute-for-loss inversion round-trips; the overhead is zero at the optimum; the parametric fit recovers a planted law and predicts a held-out size; the iso-FLOP vertex recovers the planted optimum and an allocation exponent of 0.5; the capped wrapper draws only from the first $U$ tokens and resumes exactly.

## What the evidence says

- **Compute-optimal scaling with roughly equal growth of parameters and tokens: ESTABLISHED** (Hoffmann et al. three ways; reconciled with Kaplan by Porian et al. and by Pearce and Song; replicated by Besiroglu et al.; used by Llama 3, section 3.2.1, which fits the optimal token count as $0.29\,C^{0.53}$ and extrapolates to 402B parameters on 16.55T tokens at $3.8 \times 10^{25}$ FLOPs).
- **The fitted constants: MODEL-SPECIFIC.** They depend on data (DeepSeek LLM Table 4), schedule (MiniCPM's 192 tokens per parameter under WSD), tokenizer and parameter counting. Rounded published constants can mislead (Besiroglu et al.).
- **Training past the optimum when the model will be served: ESTABLISHED practice** (Llama 3, Gemma 3, Qwen3 token counts), with the predictability of over-trained runs PROMISING (Gadre et al., one testbed up to 6.9B) and inference-aware sizing PROMISING (Sardana et al.).
- **Up to about 4 epochs of repetition costing little: PROMISING** (one large study, Muennighoff et al., up to 9B parameters and 900B tokens; widely used as a rule of thumb).
- **Course measurement (free CPU, 2026-10-06):** 16 runs of 46K–2.2M parameters. Every budget had an interior minimum. With the restricted procedure (3 lowest points per budget, runs with at least 4 tokens per parameter) the allocation exponent was 0.59 with total parameters and 0.82 with non-embedding parameters, and the parametric fit predicted a held-out budget to 0.014 nats; with all points, 0.96 and 0.165 nats. Four epochs of repetition cost 0.11 nats at this scale. Nothing here says anything about the exponents at frontier scale.

## Lab

**Folder:** [`labs/module-11/lesson-01/`](../../labs/module-11/) · **Time:** about 120 minutes (about 55 of them unattended) · **Pass check:** `pytest labs/module-11/lesson-01` passes; `ladder_lab.py published`, `isoflop` and `repeat` print their tables; your write-up applies the decision rule.

### Experiment contract

- **Question:** on Data-v0 at vocabulary 1,024, does the loss-minimising model size grow with compute in a way a fit on two budgets can predict for a third? Decision informed: whether the course's ladder method (lesson 11.3) can be trusted to size the Recipe-R run, and with which parameter count.
- **Hypothesis:** each budget has an interior minimum; the allocation exponent with total parameters is near 0.5 and larger with non-embedding parameters; a parametric fit on the two smaller budgets predicts the largest within 0.05 nats. Status: reported effect (Hoffmann et al.; Pearce and Song for the counting); may not appear cleanly at 46K–2.2M parameters and 16 runs.
- **Baseline:** none (a measurement, not a comparison); the reference is the fit's own held-out error.
- **Changed variable:** model size and tokens along each iso-FLOP line. **Controlled:** data and tokenizer, sequence 128, batch 16, peak learning rate $3 \times 10^{-3}$, warmup 5% of each run, cosine to 0.1× over each run's own length, seed 0, the same 64 validation windows.
- **Comparison axis:** equal training FLOPs (exact accounting), which is what an allocation question needs; it says nothing about wall-clock, where small models are less efficient.
- **Budget:** free CPU, 16 runs (measured below); repetition part 8 runs.
- **Metrics and decision rule:** held-out mean absolute error of the parametric fit on the largest budget; the method is "usable for sizing" if that error is below 0.05 nats *and* every budget's minimum is interior; otherwise "not usable at this scale", reported as such. Repetition part: penalty of each repeated arm over the unique arm against the seed standard deviation (2 seeds).
- **Correctness checks:** `pytest labs/common/tests/test_scaling.py`; every run's `budget.train_flops` within 2% of its budget; no ladder run trained on more tokens than the training split has.
- **Limits:** one learning rate for all sizes (Porian et al.'s last factor is not controlled; lesson 11.3 tests it), one seed per point, tiny models and vocabulary, three budgets.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100. Not run in this build; part of the Module 11 pilot | Data-v0 at vocabulary 32,768 with about 4M documents (`python -m frontierlab.data.prepare --docs 4000000 --vocab 32768 --out labs/common/data/v0-main`, about 4B tokens, PROJECTED), then `ladder_lab.py isoflop --variant main` (14 runs). PROJECTED 6.6 GPU-hours (formula above) |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `--variant t4`: rungs m11-r3 … m11-r7 at $10^{13}$–$10^{14}$ FLOPs, sequence 256; PROJECTED about 1 hour |
| Free CPU | laptop; measured below | the steps below |

### Steps

1. **Data:** `python -m frontierlab.data.prepare --docs 20000 --vocab 1024 --out labs/common/data/m11-v1024` (2 min 15 s, about 75 MB, measured 2026-10-06).
2. **Implement** the five TODOs in `lab.py` and run `pytest labs/module-11/lesson-01`.
3. **Published numbers:** `python labs/module-11/lesson-01/ladder_lab.py published`. Write down the tokens per parameter of each released model and what the overhead column means for its owner.
4. **The ladder** (unattended): `python labs/module-11/lesson-01/ladder_lab.py isoflop`. Before it finishes, predict from the table of finished runs where each budget's minimum will be.
5. **Repetition** (unattended): `python labs/module-11/lesson-01/ladder_lab.py repeat`.
6. **Write up:** the iso-FLOP minima and the allocation exponent with both parameter counts and both procedures the script prints (all runs, and the restricted one: the 3 lowest runs per budget for approach 2, runs with at least 4 tokens per parameter for approach 3); the held-out error and the decision, and how much the restriction's choice after the fact weakens it; the repetition penalties against the seed noise and against Muennighoff et al.'s $D'$.

<details>
<summary>Hint for TODO 3</summary>

Compute the loss $L$ of $(N, D)$ first. The compute-optimal loss falls as $C$ grows, so bisect on $x = \log_{10} C$: if the optimal loss at $10^x$ is still above $L$, the answer is larger than $10^x$. Return $6ND / 10^{x^*} - 1$.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-06 on the build laptop (Windows 11, 16 threads, torch 2.14.1+cpu; another module's jobs shared the CPU part of the time). `isoflop`: 16 runs of 21–346 s, 31 min 39 s in all; `repeat`: 8 runs of 100–192 s, 17 min 50 s.

**A pilot first, at the course's usual vocabulary of 8,192.** Five sizes at $3 \times 10^{12}$ FLOPs (batch 8): validation loss 6.067, 6.247, 6.443, 6.856, 7.081 from the smallest to the largest. The smallest model won and the curve had no minimum to fit. The tied head was most of each small model's compute, so the affordable budgets all sat in the "smallest model wins" corner. Retokenizing at 1,024 fixed that. This is why the lab uses `m11-v1024`.

**The ladder** (validation loss; tokens per parameter use total parameters):

| budget | m11-r1 (46K) | m11-r2 (105K) | m11-r3 (164K) | m11-r4 (394K) | m11-r5 (919K) | m11-r6 (2.17M) |
|---|---|---|---|---|---|---|
| $10^{12}$ | 4.425 (72) | **4.418** (13) | 4.711 (5.6) | 5.552 (1.0) | 5.967 (0.2) | — |
| $3 \times 10^{12}$ | 4.227 (215) | 3.953 (39) | **3.926** (17) | 4.362 (2.9) | 5.297 (0.6) | 5.655 (0.1) |
| $10^{13}$ | — (would need 33M tokens) | 3.792 (131) | 3.688 (56) | **3.669** (9.8) | 4.204 (1.8) | 4.960 (0.3) |

Every budget has an interior minimum. The best run moves one rung up per budget, at 13–56 tokens per parameter.

**Approach 2.** The first fit, a parabola through every run of a budget, gave $a = 0.96$ and put the $10^{12}$ vertex outside the sampled sizes. The steep right flank (runs of 83–250 steps) dragged every vertex left. A parabola through the 3 lowest runs per budget, the neighbourhood the method is meant to describe, gives $N_{\text{opt}} = 7.0 \times 10^4$, $1.5 \times 10^5$ and $2.7 \times 10^5$ total parameters, so $a = 0.59$ with total parameters. With non-embedding parameters the same runs give $a = 0.82$, and one vertex falls outside its points. That is Pearce and Song's effect in miniature: the same runs give a much larger size exponent once the embedding (a large share of these models) is left out of $N$.

**Approach 3.** On all 11 runs of the two smaller budgets the fit is $\alpha = 1.23$, $\beta = 0.37$. Its in-sample RMS(log) is 0.036, but on the held-out $10^{13}$ budget it is off by 0.165 nats on average (0.34 at worst). Dropping runs with fewer than 4 tokens per parameter leaves 6 training runs and gives $\alpha = 0.98$, $\beta = 1.15$, held-out error 0.014 nats on average (0.033 at worst). Those runs are a few hundred steps of an optimisation that has barely started. On the 9 such runs of all budgets, $\beta/(\alpha + \beta) = 0.58$, in line with approach 2's 0.59.

Both restrictions (3 lowest points, ≥ 4 tokens per parameter) were chosen **after** seeing the all-points fits. The 0.014 nats is therefore optimistic: a rule picked on the test budget has seen it. Lesson 11.3 tests the rule on a run nobody has seen yet. The exponents themselves ($\alpha \approx 1$, $\beta \approx 1.1$, against Chinchilla's 0.34 and 0.28) belong to this scale and data and say nothing about large models.

Decision under the contract: with the all-points procedure, **not usable** (held-out error 0.165 > 0.05 and an edge minimum). With the restricted procedure, usable, with the caveat above.

**Repetition** (m11-r3, 3.07M tokens per run, 2 seeds):

| unique tokens | epochs | loss | − unique | $D'/D$ (Muennighoff) |
|---|---|---|---|---|
| 37.3M (whole split) | 0.08 | 3.902 ± 0.004 | — | 1.000 |
| 768K | 4 | 4.012 ± 0.010 | +0.110 | 0.931 |
| 192K | 16 | 4.309 ± 0.005 | +0.407 | 0.661 |
| 48K | 64 | 7.076 ± 0.141 | +3.174 | 0.252 |

Four epochs are far from free here. The penalty is +0.11 nats, about 30 times the seed standard deviation (0.0035). At 64 epochs the model memorises 48K tokens and ends worse than a uniform guess ($\ln 1024 = 6.93$). The formula's $D'/D$ says 4 epochs keep 93% of the value. At this scale that 7% loss of effective data costs more than "negligible", because the loss is still steep in $D$ here ($\beta \approx 1.1$). Muennighoff et al.'s statement is about their scale and loss curves. Read it as a fitted law with a regime, not as a constant of nature.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-11/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-11/lesson-01`.

</details>

## Common mistakes

- **Reading small runs off a longer schedule.** A run evaluated in the middle of a cosine cycle meant for more tokens is under-trained for its token count; Hoffmann et al.'s appendix B and Porian et al.'s fourth factor. Give every ladder run its own schedule, or use WSD cooldowns.
- **Mixing parameter counts.** Fit and extrapolate with one count, and say which. Non-embedding counts at small scale inflate the size exponent (Pearce and Song).
- **Using $6ND$ when the head dominates.** At small width the output head and attention are a large share of the FLOPs; use the exact count or the iso-FLOP lines are not iso-FLOP.
- **One learning rate for every size.** The best rate moves with size in standard parametrisation; a fixed rate biases the minimum toward the sizes it suits (Porian et al.'s last factor; lesson 11.3 measures it).
- **Trusting an edge minimum.** If the best run of a budget is its smallest or largest size, the vertex is an extrapolation; add sizes before fitting the exponent.
- **Calling an over-trained model a mistake.** It is the right answer to a different objective (training plus serving cost); state which objective a recommendation optimises.

## References

- J. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020, sections 1.2, 2.1–2.2, 6, Table 5. https://arxiv.org/abs/2001.08361
- J. Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022, section 3, Tables 2–3, appendices B and D.2. https://arxiv.org/abs/2203.15556
- T. Besiroglu et al., *Chinchilla Scaling: A replication attempt*, 2024, Table 1. https://arxiv.org/abs/2404.10102
- T. Porian et al., *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*, 2024, sections 3.1–3.5, Table 1. https://arxiv.org/abs/2406.19146
- T. Pearce and J. Song, *Reconciling Kaplan and Chinchilla Scaling Laws*, 2024. https://arxiv.org/abs/2406.12907
- N. Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023, section 3.1, section 6, appendix A. https://arxiv.org/abs/2305.16264
- N. Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, 2023, section 2. https://arxiv.org/abs/2401.00448
- S. Y. Gadre et al., *Language models scale reliably with over-training and on downstream tasks*, 2024, sections 2.2–2.3, 4. https://arxiv.org/abs/2403.08540
- A. Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, 2024, section 5. https://arxiv.org/abs/2405.18392
- Meta, *Introducing Meta Llama 3*, 2024-04-18. https://ai.meta.com/blog/meta-llama-3/
- Llama Team, *The Llama 3 Herd of Models*, 2024, section 3.2.1. https://arxiv.org/abs/2407.21783
- DeepSeek-AI, *DeepSeek LLM*, 2024, sections 3.2–3.3, Table 4. https://arxiv.org/abs/2401.02954
- S. Hu et al., *MiniCPM*, 2024, section 4.5. https://arxiv.org/abs/2404.06395
- Gemma Team, *Gemma 3 Technical Report*, 2025, section 2.2. https://arxiv.org/abs/2503.19786
- Qwen Team, *Qwen3 Technical Report*, 2025, section 3.1. https://arxiv.org/abs/2505.09388
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[11.2 · Predicting downstream capability](lesson-02.md)
