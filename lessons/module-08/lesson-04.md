---
id: "08.4"
module: 8
minutes: 30
practice_minutes: 75
prerequisites: ["08.3", "01.4"]
objectives:
  - Write down the precision scaling law's effective-parameter form and post-training-quantisation term, and compute N_eff and the PTQ degradation by hand for given bits and tokens.
  - Derive the compute-optimal precision of the law's Eq. 5 in closed form (P* ≈ 1.9 γ when α = β) and explain why it does not depend on compute, and why the published 7–8 bits cannot be recomputed from the published constants alone.
  - Generate a small CPU sweep over width and weight precision, fit the law's functional form to it, and check the fit on a held-out width — stating clearly that this is a curve fit at toy scale, not a validation of the law.
  - Measure post-training-quantisation degradation as a function of training tokens and report whether it grows, with intervals.
volatility: concept
sources:
  - title: "Kumar et al. — Scaling Laws for Precision (Eqs. 1–5, sections 2.3, 4.3, appendix K)"
    url: https://arxiv.org/abs/2411.04330
  - title: "Hoffmann et al. — Training Compute-Optimal Large Language Models (Chinchilla)"
    url: https://arxiv.org/abs/2203.15556
last_verified: "2026-10-04"
---

# 08.4 · Scaling laws for precision

Extension: this lesson asks the question behind every lesson of the module — how many bits should training and deployment use? — through the scaling law of Kumar et al., which models low-precision training as a loss of effective parameters and post-training quantisation as a degradation that grows with training data. You will compute the law's terms by hand, derive the compute-optimal precision in closed form, fit the law's form to a small sweep you generate on the CPU, and measure whether post-training quantisation hurts more the longer a model trained. The fit is a curve fit at toy scale; it is explicitly not a validation of the law.

## Why this matters at a frontier lab

Precision is a budget decision. A lab choosing between BF16, FP8 and FP4 for its next run is trading parameters, tokens and bits against one compute budget, and the deployment team will quantise the result again. The paper's two headline findings bear directly on that: "the degradation introduced by post-training quantization increases as models are trained on more data, eventually making additional pretraining data actively harmful" (abstract), and compute-optimal pretraining precision is "around 7-8 bits when fitting our scaling law on runs with quantization done to integer type" (section 4.3.2). Released models train far beyond Chinchilla-optimal token counts (Module 11), which is exactly the regime where the first finding bites. Knowing how such a law is built, what it was fitted on and how far its constants travel is what lets an engineer use it as evidence instead of as a rule.

## The idea

### Low-precision training as fewer parameters

Training the weights in $P_w$ bits (integer quantisation in the forward pass, high-precision master weights) behaves, in the fits, like training a model with fewer parameters (Eq. 3):

$$L(N, D, P_w) = A\,N_{\text{eff}}^{-\alpha} + B\,D^{-\beta} + E, \qquad N_{\text{eff}} = N\big(1 - e^{-P_w/\gamma_w}\big),$$

with $N$ the parameters, $D$ the training tokens, $A, B, E, \alpha, \beta$ the Chinchilla-style constants and $\gamma_w$ the precision's "sensitivity": large $\gamma_w$ means bits matter longer. Quantising activations and the KV cache too multiplies one factor per tensor (Eq. 4): $N_{\text{eff}} = N \prod_{x \in \{w, a, kv\}} (1 - e^{-P_x/\gamma_x})$.

### Post-training quantisation as a degradation that grows with data

Rounding a high-precision-trained model to $P_{\text{post}}$ bits afterwards adds (Eq. 2)

$$\delta_{\text{PTQ}}(N, D, P_{\text{post}}) = C_T \,\frac{D^{\gamma_D}}{N^{\gamma_N}}\, e^{-P_{\text{post}}/\gamma_{\text{post}}} .$$

The $D^{\gamma_D}$ factor is the surprising one: more tokens, more degradation. The paper's fuller form (Eq. 8–9) also shrinks the degradation when the model was trained in low precision already.

### Compute-optimal precision

With the cost of training proportional to $N \cdot D \cdot P$ (Eq. 5 writes $C = \tfrac{6}{16} N D P$: 16-bit training costs $6ND$), and all three tensor types at the same $P$ with one $\gamma$:

$$\min_{N, D, P}\; A\big[N(1 - e^{-P/\gamma})^3\big]^{-\alpha} + B D^{-\beta} + E \quad \text{s.t.} \quad C = \tfrac{6}{16} N D P .$$

When $\alpha = \beta$ (the paper ties them in its Table 2 fit), the optimum over $N$ and $D$ at fixed $P$ depends only on the product $N_{\text{eff}} \cdot D = \tfrac{16 C}{6} \cdot f(P)/P$ with $f(P) = (1 - e^{-P/\gamma})^3$. So $P^*$ maximises $f(P)/P$, whatever $C$ is. Setting the derivative to zero with $u = P/\gamma$:

$$3\,u\,e^{-u} = 1 - e^{-u} \quad\Rightarrow\quad u^* \approx 1.904, \qquad P^* \approx 1.9\,\gamma .$$

This is why the paper finds compute-optimal precision "in general independent of compute" (section 4.3.2 title): the bits trade against parameters at a fixed exchange rate. Section 4.3.3 adds that when $N$ is held fixed instead, $P^*$ grows with $\log C$.

## Worked example

### N_eff by hand

With the paper's $\gamma_w = 2.6745$ (appendix K, Table 2): at 4 bits $1 - e^{-4/2.6745} = 1 - e^{-1.4956} = 1 - 0.2241 = 0.776$; at 8 bits $1 - e^{-2.991} = 0.950$; at 3 bits $0.674$; at 16 bits $0.997$. A 100M-parameter model trained with 4-bit weights behaves, in this fit, like a 77.6M model trained in high precision. With $A = 4{,}299$ and $\alpha = 0.4965$ the parameter term is $4299 \cdot (7.76 \times 10^7)^{-0.4965} = 4299 \cdot 1.21 \times 10^{-4} = 0.520$ nats instead of $4299 \cdot (10^8)^{-0.4965} = 0.459$: 0.061 nats lost to 4-bit weights.

### P* by hand

Check $u = 1.904$: left side $3 \cdot 1.904 \cdot e^{-1.904} = 5.712 \cdot 0.1490 = 0.851$; right side $1 - 0.1490 = 0.851$. With $\gamma = 2.6745$: $P^* = 5.09$ bits. With $\gamma = 4$: $7.6$ bits. So the paper's 7–8 bits corresponds to an effective $\gamma$ of about 3.7–4.2 in this simplified form. Plugging the Table 2 constants straight in gives about 5 bits (weights' $\gamma_w$) or about 3.5 bits (the three separate $\gamma$'s, since $\gamma_{kv} = 0.96$); the table also has shift terms ($n_w, n_i, n_{kv}, b$) whose exact role in Eq. 5 we could not pin down. We therefore **could not recompute** the 7–8 bits from the published constants and teach it as the paper's reported result (PUBLICLY DOCUMENTED claim), with this derivation as the mechanism.

### PTQ degradation by hand

$\gamma_D = 0.5068$, $\gamma_N = 0.3439$: doubling the training tokens multiplies $\delta_{\text{PTQ}}$ by $2^{0.5068} = 1.42$; doubling parameters divides it by $2^{0.3439} = 1.27$. Training 10× past a token budget multiplies the PTQ penalty by $10^{0.5068} = 3.2$. If the penalty was 0.02 nats at the original budget, it is 0.064 nats after — comparable to the whole gain that the extra tokens buy in a flat part of the loss curve (Module 11). That is the paper's "more data can hurt" in two lines of arithmetic.

## Shapes and cost

| Quantity | Paper's sweep (section 2.3) | This lab's CPU sweep |
|---|---|---|
| Model sizes (non-embedding) | 30M, 60M, 110M, 220M (validation to 1.7B) | toy layout at widths 64, 96, 128: 0.20M, 0.43M, 0.79M |
| Tokens | 1.5B–26B | 0.31M per run (150 × 2,048); PTQ ladder 0.15M–1.23M |
| Runs | over 465 | 15 + 4 |
| Precisions | 3–16 bits, integer and floating point | weight-only INT3, INT4, INT6, INT8, unquantised |
| Tensors | weights, activations, KV cache | weights only |

The fit itself is a grid over $(\alpha, \gamma)$ (59 × 117 points) with a 2-column least-squares solve per point: about 7,000 tiny solves, under a second. The sweep's cost is training: 19 runs of models with $N$ under 0.8M parameters, at 2,048 tokens per step.

## Build it

`frontierlab/precision/scaling_law.py`: `n_eff`, `loss` (Eqs. 3–4), `delta_ptq` (Eq. 2), `optimal_precision(C, gamma)` (Eq. 5 by grid search), `p_star_closed_form(gamma)` (the bisection above), `fit_precision_law` (grid over $\alpha, \gamma$, least squares for $A, E$) and `fit_power` (log-log slope). The sweep uses the training wrapper's `--recipe w-int<b>`: weight-only INT-b fake quantisation with one fp32 scale per output row and straight-through gradients — the setting of Eq. 3, where only weights are trained in low precision.

Correctness checks (`test_precision.py`): the grid optimum of Eq. 5 equals the closed form within the grid spacing for three values of $\gamma$, and is identical at $C = 10^{19}$ and $10^{23}$ (compute independence); the fitter recovers $\gamma$ and $\alpha$ exactly from synthetic data generated by the law; the log-log fit recovers a known exponent.

## What the evidence says

- **Precision-aware scaling laws — PROMISING.** One group's fits (465 runs, 30M–220M parameters, up to 26B tokens; validated to 1.7B), with the authors' own warning that "our numerical constants are unlikely to be useful" outside their setting. Section 4.3.2's floating-point check used 220M–1.6B models.
- **"PTQ degradation grows with data" — PROMISING, and consistent with practice.** PUBLICLY DOCUMENTED in the abstract and Eq. 2; it agrees with the experience that heavily over-trained models are harder to quantise (REASONABLE INDUSTRY PRACTICE), but the exponent is specific to the paper's fits.
- **P* ≈ 7–8 bits — the paper's result for integer-type fits.** It is consistent with the industry move to FP8 training, but it is not a statement about FP4 recipes with block scaling (lesson 08.3), which change the error per bit.
- **What a course-scale sweep can and cannot say.** It can show the functional form fitting a handful of points and the mechanics of extrapolation; it cannot confirm the law, its constants or its conclusions.

## Lab

**Folder:** [`labs/module-08/lesson-04/`](../../labs/module-08/) · **Time:** about 75 minutes, about 25 of them unattended on the free CPU path · **Pass check:** `pytest labs/module-08/lesson-04` passes; your notes contain the fitted $\gamma$ and $\alpha$ with the residual RMS, the leave-widest-out prediction error, the PTQ deltas with intervals, and one paragraph on what the fit does not show.

### Experiment contract

- **Question:** does the functional form $L = A[N(1 - e^{-P/\gamma})]^{-\alpha} + E$ describe a small sweep of weight-quantised toy models, does it predict a held-out width, and does PTQ degradation grow with training tokens at this scale? Decision informed: none for Recipe-R directly; this is a methods exercise whose output is a calibrated sense of how much a fitted scaling law can be trusted.
- **Hypotheses and status:** (H1) loss falls with bits and saturates by 6–8 bits — reported (paper's fits), likely at toy scale. (H2) the form fits with residuals below the seed noise floor — may not hold with so few points. (H3) the INT4 and INT3 PTQ deltas grow with $D$ — reported (abstract, Eq. 2), may not appear in 0.15M–1.2M tokens.
- **Baseline:** the unquantised arm at each width.
- **Changed variables:** a declared factorial design — width (3 levels) × weight precision (5 levels); separately, training steps (4 levels) at one width.
- **Controlled:** seed 0, data order, schedule shape (cosine over each run's own length), 256 evaluation windows.
- **Comparison axis:** equal tokens in Part A; Part B varies tokens by design.
- **Budget:** free CPU, measured below; GPU variant a fraction of a GPU-hour.
- **Metrics and decision rule:** fit residual RMS against the seed std of Module 7's toy runs (0.02 nats); leave-widest-out prediction error; PTQ delta per $D$ with paired 95% intervals. Rule: call the form "adequate at this scale" if the residual RMS and the held-out error are both below 0.02 nats; call H3 "observed" only if the INT3 delta's intervals at the smallest and largest $D$ do not overlap and the deltas increase monotonically.
- **Correctness checks:** `pytest labs/module-08/lesson-04` and the scaling-law tests in `test_precision.py` pass.
- **Fallback evidence:** the paper's Figures for its own sweep (analysis of published results).
- **Limits:** under 1M parameters and about 0.3M tokens per run; weight-only; one seed; three widths. Nothing here tests the law; with this many free parameters a smooth curve fits almost anything.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | any 1× GPU, PROJECTED well under 1 GPU-hour (19 runs of 0.8M–7.1M parameters at widths 128–384); not run in this build | `sweep.py --variant gpu --device cuda`, then `fit.py runs/m08/l84/gpu/results.json` |
| Free GPU (Colab/Kaggle T4) | T4 | the same as the main path |
| Free CPU | laptop; measured below | steps 1–3 as written |

### Steps

1. **Implement** `n_eff`, `p_star`, `fit` and `ptq_exponent` in `lab.py`; run `pytest labs/module-08/lesson-04`.
2. **Sweep:** `python labs/module-08/lesson-04/sweep.py` (19 runs; reruns skip finished ones).
3. **Fit:** `python labs/module-08/lesson-04/fit.py`. Then answer: what is your fitted $\gamma$ and the $P^*$ it would imply? Why is that number not evidence about the 7–8 bits of the paper? What would you need — sizes, tokens, seeds — to make it evidence?

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04): `sweep.py` 24.8 minutes for 19 runs (49–150 s each; reruns skip finished runs); `fit.py` about 10 seconds. Held-out loss on 256 windows of 128 tokens.

**Part A** (150 steps, 0.31M tokens per run):

| Width (N non-embedding) | INT3 | INT4 | INT6 | INT8 | unquantised |
|---|---|---|---|---|---|
| 64 (197,440) | 6.9783 | 7.0351 | 7.0265 | 6.9887 | 7.0111 |
| 96 (431,200) | 6.8327 | 6.8459 | 6.7971 | 6.8179 | 6.8108 |
| 128 (787,840) | 6.7985 | 6.7206 | 6.7792 | 6.7744 | 6.7746 |

Fit: $\alpha = 1.45$ and $\gamma = 0.20$ — both **at the edges of their grids**, residual RMS 0.022 nats; leave-widest-out prediction error +0.088 nats on average (max 0.117). The fit is degenerate: within each width the five precisions differ by up to 0.08 nats with no order (INT4 is the best arm at width 128 and the worst at width 64), so the least-squares answer is "precision has no effect" ($\gamma \to 0$ makes every factor 1) and the width trend is carried by $\alpha$ and $E$ alone. By the contract's rule the form is **not adequate at this scale** (residual 0.022 and held-out error 0.088, both above 0.02). H1 is **not observed**: at 0.31M tokens the run-to-run spread of a changed rounding (about ±0.03 nats, lesson 08.2) is larger than any effect of 3 versus 8 bits. The P* the script prints from this fit (0.38 bits) is meaningless and is printed only to show what an unexamined pipeline would report.

**Part B** (width 128, separate runs of 75–600 steps; PTQ per output row):

| Tokens D | BF16 loss | PTQ INT4 delta [95% CI] | PTQ INT3 delta [95% CI] |
|---|---|---|---|
| 153,600 | 7.1427 | −0.0008 [−0.0010, −0.0006] | +0.0007 [+0.0002, +0.0011] |
| 307,200 | 6.7746 | +0.0014 [+0.0010, +0.0019] | +0.0044 [+0.0034, +0.0053] |
| 614,400 | 6.3016 | +0.0004 [−0.0001, +0.0009] | +0.0088 [+0.0076, +0.0100] |
| 1,228,800 | 5.8939 | +0.0017 [+0.0010, +0.0023] | +0.0096 [+0.0081, +0.0109] |

H3 is **observed for INT3** by the rule (monotone increase; the intervals at the smallest and largest $D$ do not overlap): the PTQ penalty grows 14-fold over an 8-fold increase in tokens, a log-log slope of 1.24 (the paper's $\gamma_D$ is 0.51 — a different regime: these models are far from converged, and their weights are still moving fast). INT4's deltas are under 0.002 nats and not monotone: too small to say. So the one finding of the paper this budget can reach — PTQ hurts more the longer a model has trained — shows up in the coarsest format, and the rest of the sweep is a lesson in what a scaling-law fit looks like when the effect is below the noise.

<details>
<summary>Hint for TODO 2</summary>

Let `h(u) = k*u*exp(-u) - (1 - exp(-u))`. `h` is positive just above 0 and negative for large `u`, so bisect: `mid = (lo + hi) / 2`; if `h(mid) > 0` the root is above `mid`. Return `gamma * u`.

</details>

<details>
<summary>Hint for TODO 3</summary>

For fixed $(\alpha, \gamma)$ the model is linear in $A$ and $E$: build `X = np.stack([n_eff(N, P, g) ** -a, np.ones_like(L)], 1)` and call `np.linalg.lstsq(X, L, rcond=None)`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-08/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-08/lesson-04`.

</details>

## Common mistakes

- **Treating a fitted constant as a property of nature.** $\gamma$, $\alpha$ and $C_T$ belong to one data set, architecture, quantiser and token range. The authors say so; repeat it in every report that uses them.
- **Mixing integer and floating-point precision in one law.** The 7–8 bits is for integer-type fits; FP formats with block scales have different error per bit.
- **Counting bits without scales.** A "4-bit" format with a scale per 16 values is 4.5 bits in the cost model.
- **Reading PTQ degradation off intermediate checkpoints of one run.** A checkpoint in the middle of a cosine schedule is not a model trained for that many tokens; the lab trains separate runs, each with its own full schedule.
- **Extrapolating a fit outside its range.** A curve through three widths under 1M parameters says nothing about 1B.

## References

- T. Kumar et al., *Scaling Laws for Precision*, abstract, Eqs. 1–5 and 8–9, sections 2.3 and 4.3, appendix K Table 2. https://arxiv.org/abs/2411.04330
- J. Hoffmann et al., *Training Compute-Optimal Large Language Models*. https://arxiv.org/abs/2203.15556
- Shared code: `labs/common/frontierlab/precision/scaling_law.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

The [Module 8 project](../../projects/module-08-precision-plan.md): a precision plan for a stated model and hardware.
