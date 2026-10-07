---
id: "14.4"
module: 14
minutes: 40
practice_minutes: 100
prerequisites: ["14.2", "12.4"]
objectives:
  - Fit ScaleRL's sigmoid compute-performance curve and interpret A, B and C_mid as asymptote, efficiency and half-way compute.
  - Compute a profile interval for the asymptote and show from it which fit windows identify A and which cannot.
  - Explain why a course-scale RL run cannot support a claim about a recipe's asymptote, and what claim it can support.
  - Compute pass@k curves with the unbiased estimator, find a crossover between an RL policy and its start, and say what the crossover does and does not show about RL adding capability.
  - Read a random-reward control on the same pass@k axes and decide whether an RL gain needs the verifier's signal.
volatility: concept
sources:
  - title: "Khatri et al., The Art of Scaling Reinforcement Learning Compute for LLMs (Eq. 1; section 2.1; sections 3-4; appendix fit-window study; Figures 1, 5, 8)"
    url: https://arxiv.org/abs/2510.13786
  - title: "Yue et al., Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (section 2.2, Eq. 2; section 3.1, Figure 2; section 4.2)"
    url: https://arxiv.org/abs/2504.13837
  - title: "Liu et al., ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models"
    url: https://arxiv.org/abs/2505.24864
  - title: "Shao et al., Spurious Rewards: Rethinking Training Signals in RLVR (section 2; section 3)"
    url: https://arxiv.org/abs/2506.10947
  - title: "Chen et al., Evaluating Large Language Models Trained on Code (section 2.1, Eq. 1: unbiased pass@k)"
    url: https://arxiv.org/abs/2107.03374
  - title: "Hoffmann et al., Training Compute-Optimal Large Language Models (section 3: fitting scaling laws)"
    url: https://arxiv.org/abs/2203.15556
last_verified: "2026-10-07"
---

# 14.4 · RL scaling and the capability debate

Two questions decide whether a reasoning-RL recipe is worth scaling: how high will held-out performance go if we keep spending compute, and is what it buys new capability or a sharper version of what the base model could already do? ScaleRL answers the first with a sigmoid fitted to compute-performance curves of runs up to 100,000 GPU-hours. Yue et al. raise the second by comparing pass@k curves at large k. This lesson fits the sigmoid to a curve built from ScaleRL's published fit and shows from a profile interval why your own short run cannot identify the asymptote. It then reads the pass@k debate on your own policies, with a random-reward control on the same axes.

## Why this matters at a frontier lab

RL compute has become a large line item: xAI describes Grok 4's RL as run at pretraining scale and OpenAI describes o3's RL as an order of magnitude more than o1's (both company claims, relative to their own earlier runs), and no lab publishes what share of its compute goes to RL. Before committing a large budget, a team wants to predict the end of the run from its beginning, and to compare recipes by where they are heading rather than where they are today. ScaleRL's contribution is a method: fit a bounded curve with an asymptote $A$ and an efficiency $B$, then compare recipes on both. The method has a precondition that is easy to forget. The run must have gone far enough up the curve for $A$ to be pinned down, and ScaleRL itself fits only after about 1,500 GPU-hours. The second question decides what an RL gain is worth. If RL mostly raises pass@1 by concentrating probability on answers the base model could already sample, then a test-time method (Module 15) or more base-model sampling may buy the same thing more cheaply.

## The idea

### A bounded compute-performance curve

ScaleRL (Eq. 1) models the expected pass rate $R_C$ on a held-out set after RL compute $C$ (GPU-hours) as

$$R_C - R_0 = \frac{A - R_0}{1 + (C_{\text{mid}}/C)^B},$$

with $R_0$ the starting pass rate, $A \le 1$ the **asymptotic** pass rate, $B > 0$ the **scaling exponent** (larger means the curve rises more steeply around its midpoint, i.e. more compute-efficient), and $C_{\text{mid}}$ the compute at which half of the gain $A - R_0$ is reached. A power law in $C$ has no ceiling, and a pass rate does: that is the reason for the sigmoid. The held-out set is 1,000 prompts from Polaris-53k, scored every 100 steps with 16 generations per prompt (section 2.1).

Two recipes can then differ in where they end up ($A$) and in how fast they get there ($B$, $C_{\text{mid}}$). ScaleRL's leave-one-out study reports, for example, that CISPO and GSPO reach a higher $A$ than DAPO, with CISPO marginally ahead of GSPO late in training (Figure 5a), and that FP32 logits raise $A$ from 0.52 to 0.61 (Figure 5b). Other choices change efficiency at a similar asymptote: CISPO's $B = 2.01$ against DAPO's 1.77 in the leave-one-out runs (Figure 7), and PipelineRL against plain off-policy PPO (Figure 4a). Their main 8B run spent 100,000 GPU-hours, and a 17B×16 mixture-of-experts run 50,000 (Figure 1).

### Fitting, and when a fit means something

For fixed $(B, C_{\text{mid}})$ the curve is linear in $A$: with $f_j = 1/(1 + (C_{\text{mid}}/C_j)^B)$,

$$\hat A(B, C_{\text{mid}}) = R_0 + \frac{\sum_j f_j (y_j - R_0)}{\sum_j f_j^2}$$

minimises the squared error $\sum_j (y_j - R_0 - (A - R_0) f_j)^2$. `frontierlab.rlscale.curves.fit_sigmoid` uses this on a grid of $(B, \log C_{\text{mid}})$ and refines the best point. The same trick gives a **profile**: fix $A$, minimise the error over $(B, C_{\text{mid}})$, and plot that minimum against $A$. With the noise variance estimated from the best fit, $\hat\sigma^2 = \text{SSE}_{\min}/(n - 3)$, the values of $A$ whose profile lies within $\Delta\chi^2 = 3.84$ of the minimum form an approximate 95% interval. When that interval runs all the way to $A = 1$, the data do not bound the asymptote at all.

Why short runs cannot bound it: as long as every observed $C$ lies well below $C_{\text{mid}}$, the curve is in its early, convex part, $R_C - R_0 \approx (A - R_0)(C/C_{\text{mid}})^B$. A larger $A$ with a larger $C_{\text{mid}}$ produces the same early curve, so the data constrain one combination of $A$ and $C_{\text{mid}}$, not each separately. Only points near and beyond the midpoint separate them. ScaleRL's appendix runs exactly this check on its 8B run. Fitting on (1.5k, 50k) GPU-hours gives $A = 0.645$, $B = 1.70$, and fitting on the full (0, 100k) gives $A = 0.655$, $B = 1.56$: the extrapolated asymptote held up to twice the compute. Run-to-run variation of $A$ was at most ±0.015 over three runs, so the paper treats ±0.02 as the margin within which recipes are not distinguishable (Figure 8a).

### pass@k and the capability debate

From $n$ samples per problem, $c$ of them correct, the unbiased estimate of "at least one of $k$ samples is correct" is

$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k},$$

averaged over problems (Chen et al. Eq. 1; Yue et al. Eq. 2 with $n$ equal to the largest $k$). pass@1 measures how often the model is right. pass@k at large $k$ measures **coverage**: on how many problems the model can be right at all within $k$ tries.

Yue et al. (section 3.1, Figure 2) compare RLVR-trained models with their base models on math, code and visual reasoning, with $k$ up to 1,024 on AIME24 and AMC23, 128 on MATH500 and GSM8K and 256 on code. RL wins at small $k$, and "the base models achieve a higher pass@k score when k is large". Their reading is that RLVR mostly *sharpens* the distribution towards solutions the base model can already sample, and narrows coverage on the way, whereas distillation from a stronger model adds new reasoning patterns (section 4.2). It is a debate, not a settled result. ProRL (Liu et al., NVIDIA) reports that prolonged RL with KL control and reference resets gives a 1.5B model pass@k gains over its base "including scenarios where base models fail entirely". ScaleRL's asymptote is a pass@16-averaged pass rate, not a coverage measure, so it does not answer the question either way. And a crossover depends on $k$, sample budget, temperature and the benchmark's size.

**A control on the same axes.** Shao et al. trained Qwen2.5-Math-7B with spurious rewards and report MATH-500 gains of +21.4 points with a random reward and +24.1 with incorrect labels, against +29.1 with ground truth (section 2), while the same rewards "often fail to produce gains" on Llama 3 and OLMo 2 (section 3). For Qwen models some of what RL "teaches" is behaviour (format, the code-reasoning style) the base model already had and any reward signal elicits. So every Stage D result in this course carries a random- or format-reward control: an RL gain counts only to the extent that it exceeds the control's, on pass@1 and on pass@k.

## Worked example

**The curve at its midpoint.** $A = 0.645$, $B = 1.70$, $C_{\text{mid}} = 8{,}000$ GPU-hours, $R_0 = 0.30$. At $C = 8{,}000$: $R = 0.30 + 0.345/(1 + 1) = 0.4725$, half the gain. At $C = 2{,}000$: $(8{,}000/2{,}000)^{1.7} = 4^{1.7} = 10.56$, so $R = 0.30 + 0.345/11.56 = 0.330$. At $C = 50{,}000$: $(0.16)^{1.7} = 0.0443$, so $R = 0.30 + 0.345/1.0443 = 0.630$.

**Two curves that agree early.** The second curve has $A = 0.90$ and the same $B$, with $C_{\text{mid}}$ chosen so that the early slope matches. Early on $R - R_0 \approx (A - R_0)(C/C_{\text{mid}})^B$, so we need $(0.60)\,C_{\text{mid}}'^{-1.7} = (0.345)\,8000^{-1.7}$, which gives $C_{\text{mid}}' = 8{,}000 \cdot (0.60/0.345)^{1/1.7} = 8{,}000 \cdot 1.385 = 11{,}080$. At $C = 2{,}000$ the second curve gives $0.30 + 0.60/(1 + 5.54^{1.7})$. Here $5.54^{1.7} = e^{1.7 \ln 5.54} = e^{2.910} = 18.36$, so $R = 0.30 + 0.60/19.36 = 0.331$, against 0.330 for the first. The curves differ by 0.001 at 2,000 GPU-hours, well inside a ±0.005 noise band, and at 20,000 GPU-hours they give 0.585 and 0.739. A run that stopped at 4,000 GPU-hours cannot tell them apart.

**pass@k on one problem.** $n = 16$ samples. The RL model is right on 8 of them, the base model on 3. pass@1 is $8/16 = 0.5$ against $3/16 = 0.19$. pass@4 for the base model: $1 - \binom{13}{4}/\binom{16}{4} = 1 - 715/1820 = 0.607$. For the RL model it is $1 - \binom{8}{4}/\binom{16}{4} = 1 - 70/1820 = 0.962$. On a second problem where RL is right 0 times and base 1 time, pass@16 is 0 against 1. Averaged over both problems, RL leads at $k = 1$ (0.25 vs 0.125) and trails at $k = 16$ (0.5 vs 1.0): a crossover from one problem the RL policy stopped solving.

## Shapes and cost

| Object | Shape, dtype | Notes |
|---|---|---|
| curve points $(C_j, y_j)$ | (n,) float64 each | ScaleRL: one point per 100 steps; this lab: 198 reconstructed points, your run: 20 |
| fit grid | 38 values of $B$ × 81 of $\log C_{\text{mid}}$, closed-form $A$ | milliseconds |
| profile | ~250 values of $A$ × (grid + Nelder–Mead) | seconds to a minute on a laptop |
| pass@k counts | (P,) int, correct out of $n$ per problem | toy: P = 200, n = 256, so 51,200 samples per model; main path: 200 × 64 |

The fitting is free; the data are what cost. A ScaleRL-style curve needs one long run per recipe: 16,000 GPU-hours per leave-one-out run in the paper, which is why the course fits published-shaped curves rather than producing its own (plan section 12.1). pass@k at large $k$ costs $n$ generations per problem. Main path (PROJECTED, pending the pilot): 200 GSM8K questions × 64 samples × ~300 tokens = $3.8 \times 10^6$ generated tokens per checkpoint, about 0.5–1 GPU-hours with `transformers.generate` on an H100, much less with vLLM.

## Build it

```python
import numpy as np
from frontierlab.rlscale import curves, passk

C = np.linspace(1.5e3, 1e5, 198)
y = curves.sigmoid_curve(C, 0.645, 1.70, 8e3, 0.30)
curves.fit_sigmoid(C, y, R0=0.30)                                  # A 0.645, B 1.70, C_mid 8000
curves.profile_asymptote(*curves.window(C, y, 1.5e3, 4e3), R0=0.30)["bounded_above"]   # False
passk.curve(counts, n=256, ks=[1, 16, 256]); passk.crossover(curve_rl, curve_base, ks)
```

`frontierlab/rlscale/curves.py` has the curve, the fit, the profile interval and fit windows. `passk.py` has pass@k curves, paired per-k differences with a bootstrap over problems, the crossover, solved-ever sets and toy sampling. Both use `frontierlab.evals.suite_v2.core.pass_at_k`. The main-path loop (`hf_rl.py`) saves the per-question correct counts in every evaluation row, so this analysis runs on GSM8K checkpoints without regenerating. Correctness checks (`test_rlscale.py`): the fit recovers known parameters from a full curve; the profile interval contains the true $A$ and is narrow on a long window, and is unbounded on a window ending at half of $C_{\text{mid}}$; pass@k against brute-force enumeration of subsets (lesson 12.4) and the crossover on a constructed sharpened-vs-spread example.

## What the evidence says

- **Sigmoid compute-performance fits for RL: PROMISING.** One lab's method (Meta, ScaleRL), validated by extrapolating from 50k to 100k GPU-hours on its own runs (appendix; PUBLICLY DOCUMENTED). Independent replication at that scale is not published.
- **"Fits need the curve's middle": ESTABLISHED** as a property of the functional form (the identifiability argument above), and stated by ScaleRL ("all our scaling fits begin after ~1.5k GPU hours", section 2.1).
- **Recipe rankings by $A$ and $B$ (CISPO/GSPO over DAPO on $A$, FP32 logits, PipelineRL on $B$): PROMISING, MODEL-SPECIFIC** to ScaleRL's 8B and Scout runs, with the ±0.02 noise margin the paper itself gives.
- **RLVR raises pass@1 more than large-k pass@k: PUBLICLY DOCUMENTED** for the models and budgets in Yue et al. Whether RL can expand coverage with longer training is **debated** (ProRL reports it can). The general statement "RL adds no capability" is not established.
- **Spurious-reward gains on Qwen2.5-Math: PUBLICLY DOCUMENTED** (Shao et al.), and not reproduced on Llama 3 or OLMo 2. That a control arm is necessary for Qwen-based Stage D results is this course's REASONABLE INDUSTRY PRACTICE.
- **Course measurement (free CPU, 2026-10-07):** the reconstructed curve's asymptote was pinned to ±0.003 from windows past $C_{\text{mid}}$ and unbounded from a 1.5k–4k window. The course's own 200-step GRPO run left $A$ in [0.58, 1.00]. GRPO raised pass@1 by 0.22 and left pass@256 unchanged within its interval, with a crossover at $k = 64$ driven by 3 of 200 problems. A random reward destroyed the toy skill. Nothing at this scale bears on the frontier-scale debate; it shows how to read the measurements.

## Lab

**Folder:** [`labs/module-14/lesson-04/`](../../labs/module-14/) · **Time:** about 100 minutes (about 10 minutes unattended) · **Pass check:** `pytest labs/module-14/lesson-04` passes; `scaling_lab.py` prints the three parts; your write-up states, with intervals, what your run does and does not say about its asymptote and about capability.

### Experiment contract

- **Question:** (a) from which fit window can the asymptote of a ScaleRL-shaped curve be identified, and can it be identified from your own 200-step run? (b) Does GRPO on the toy task raise pass@k at large $k$, or only at small $k$, and is either gain larger than a random-reward control's? Decision informed: whether this course's runs may report an asymptote, and how Module 15 compares RL with test-time sampling.
- **Hypothesis:** (a) windows ending before $C_{\text{mid}}$ leave $A$ unbounded above; your run's interval is wide. (b) GRPO raises pass@1 clearly; pass@256 changes little (possibly a crossover); the control arm moves neither beyond noise. Status: (a) a property of the model form; (b) reported for large models (Yue et al.), may not appear for a 308k-parameter policy whose start already covers most problems.
- **Baseline:** the SFT start (the Module 12 checkpoint) for pass@k; for the fit, the curve's generating parameters.
- **Changed variable:** fit window (a); training reward, verifier vs random (b). **Controlled:** the evaluation set (200 held-out problems), $n = 256$, temperature 1, sampling seed, 200 training steps with the same seeds and settings.
- **Comparison axis:** equal training samples (b); compute as sampled responses (a, own run).
- **Budget:** free CPU, measured runtime in the results box.
- **Metrics and decision rule:** (a) the 95% profile interval of $A$; "identified" only if it is bounded above below 1. (b) Per-$k$ paired difference (model − SFT) over problems with a 95% bootstrap interval at $k = 1$ and $k = 256$. "RL raised coverage" only if the $k = 256$ interval lies above 0 *and* above the control's difference. Also report solved-ever sets.
- **Correctness checks:** your TODO tests (pass@k against enumeration); the fit recovers known parameters on a noise-free curve (`test_rlscale.py`).
- **Fallback evidence:** Yue et al.'s Figure 2 and ScaleRL's fit-window table, labelled as published.
- **Limits:** one training seed for the pass@k part; a toy task with 2-digit addition; intervals over problems, not over training seeds; the reconstructed curve's $C_{\text{mid}}$ and $R_0$ are stand-ins, not ScaleRL's values.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 14 pilot | `--variant main --print`: a 300-step CISPO run and a random-reward control on Qwen3-1.7B-Base with 64 samples per question on 200 GSM8K test questions every 25 steps; pass@k from the saved counts; the profile of $A$ from the eval curve. Also digitise ScaleRL's Figure 1 (for example with WebPlotDigitizer) and refit it instead of the reconstruction. **PROJECTED:** 2 × 300 steps × 30–50 s = 5–8 GPU-hours plus 13 evaluations × 0.5–1 GPU-hours = 12–21 GPU-hours, USD 24–63 |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --print` (Qwen3-0.6B-Base, 8 prompts, 256 new tokens); fewer evaluation samples |
| Free CPU | laptop; measured about 7 minutes in all, including two 200-step runs (other jobs running) | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-14/lesson-04`.
2. **Published-shaped curve:** `python labs/module-14/lesson-04/scaling_lab.py --part published`. For each window, compare the fitted $A$ and its profile interval with the generating 0.645. Where does the interval first become bounded, relative to $C_{\text{mid}}$?
3. **Your run:** `--part own` trains 200 steps of GRPO (about 4 minutes) and profiles $A$ on the first 50, 100 and 200 steps. Write one sentence you could defend about your run's asymptote.
4. **pass@k:** `--part passk` also trains a 200-step random-reward control and samples 256 responses per problem from each model. Apply the decision rule.
5. **Write up** both questions with intervals, and one paragraph: what would you need, in compute and in evaluation design, to make either claim at the main-path scale?

<details>
<summary>Hint for TODO 2</summary>

`f = 1 / (1 + (C_mid / C) ** B)`; the least-squares slope of `y - R0` on `f` through the origin is `f @ (y - R0) / (f @ f)`. Cap $A$ at `A_max` and compute the residuals *at the capped value*.

</details>

<details>
<summary>Hint for TODO 3</summary>

If `n - c < k` the probability of drawing $k$ wrong samples is 0, so pass@k is 1. Otherwise multiply $\frac{n-c-i}{n-i}$ for $i = 0, \dots, k-1$. Never compute $\binom{256}{128}$ directly.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, another module's jobs running): `--part published` 58 s, `--part own` 4.7 minutes (including the 200-step run), `--part passk` 48 s plus the 200-step control run (trained by `own_run` on first use).

**Published-shaped curve** (A = 0.645, B = 1.70 from the paper; C_mid = 8,000 and R_0 = 0.30 stand-ins; noise 0.005):

| Fit window (GPU-hours) | points | fitted A | 95% profile interval for A |
|---|---|---|---|
| 1.5k–100k | 198 | 0.644 | [0.643, 0.645] |
| 1.5k–50k | 98 | 0.646 | [0.643, 0.649] |
| 5k–50k | 91 | 0.645 | [0.642, 0.648] |
| 1.5k–16k (2× C_mid) | 30 | 0.659 | [0.637, 0.688] |
| 1.5k–8k (up to C_mid) | 14 | 0.595 | [0.530, 0.805] |
| 1.5k–4k (half of C_mid) | 6 | 0.907 | [0.425, 1.000]: not identified |

Windows that pass well beyond the midpoint pin $A$ to ±0.003, matching ScaleRL's finding that its (1.5k, 50k) fit predicted the 100k curve. A window ending at the midpoint already gives an interval 0.28 wide, and one ending at half of it does not bound $A$ at all, while its point estimate (0.907) looks as precise as any other.

**Your run** (GRPO, 200 steps, $R_0 = 0.220$ from the same evaluation of the SFT start): held-out accuracy rose steadily from 0.23 to 0.48 with no sign of levelling off. Every fit put $A$ at the cap of 1.0, and the profile intervals were [0.264, 1.000] after 50 steps, [0.370, 1.000] after 100 and [0.582, 1.000] after 200. The defensible sentence: "after 25,600 sampled responses the run had not reached its curve's midpoint; its asymptote is at least about 0.58 and otherwise unidentified."

**pass@k** on 200 held-out problems, $n = 256$:

| k | 1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| SFT start | 0.206 | 0.534 | 0.858 | 0.965 | 0.990 |
| GRPO, 200 steps | 0.427 | 0.768 | 0.921 | 0.965 | 0.980 |
| random reward, 200 steps | 0.035 | 0.106 | 0.209 | 0.287 | 0.345 |

GRPO minus SFT: pass@1 +0.220 [+0.191, +0.247]; pass@256 −0.010 [−0.030, +0.005]. The curves meet at $k = 64$ and the SFT start is ahead beyond it. Solved ever: 195 by both, 3 only by the start, 1 only by GRPO. That is the Yue et al. pattern in miniature: a large, certain gain in reliability, no gain in coverage (the interval at $k = 256$ includes 0, so "coverage fell" is not shown either), and a crossover driven by 3 of 200 problems. The control arm behaves unlike Shao et al.'s Qwen2.5-Math. On this toy policy a random reward destroyed the skill (pass@1 −0.171, pass@256 −0.645, 131 problems lost), which is what a reward independent of the answer should do to a policy with nothing latent for it to elicit. So the GRPO gain here needs the verifier's signal. On a Qwen base model the same control is the open question, which is why the main path runs it.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-14/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-14/lesson-04`.

</details>

## Common mistakes

- **Reporting a fitted asymptote without its interval.** A least-squares fit always returns a number. The profile says whether the data chose it.
- **Fitting from step 0.** The early regime follows different dynamics (warm-up, format learning). ScaleRL starts its fits after about 1,500 GPU-hours.
- **Comparing recipes on $A$ inside the noise margin.** ScaleRL's own runs vary by ±0.015 in $A$.
- **Using the biased $1 - (1 - c/n)^k$** for pass@k, or computing pass@k with $n = k$ samples (much higher variance).
- **Reading a crossover as "RL removed capability" from one benchmark and one sample budget.** Report the solved-ever sets and the interval at each $k$, and say how many problems drive the crossover.
- **No control arm.** On Qwen models a random reward can produce a large pass@1 gain. Without the control on the same axes, the gain cannot be attributed to the verifier.

## References

- D. Khatri et al., *The Art of Scaling Reinforcement Learning Compute for LLMs*, 2025, Eq. 1, section 2.1, sections 3–4, appendix, Figures 1, 4, 5, 7, 8. https://arxiv.org/abs/2510.13786
- Y. Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, 2025, sections 2.2, 3.1, 4.2. https://arxiv.org/abs/2504.13837
- M. Liu et al., *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models*, 2025. https://arxiv.org/abs/2505.24864
- R. Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, 2025, sections 2–3. https://arxiv.org/abs/2506.10947
- M. Chen et al., *Evaluating Large Language Models Trained on Code*, 2021, section 2.1. https://arxiv.org/abs/2107.03374
- J. Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022, section 3. https://arxiv.org/abs/2203.15556
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The [Module 14 project](../../projects/module-14-reasoning-rl.md) puts the module together: a reasoning-RL run on the base model with a stability report, Eval v2 retention and a pass@k analysis. Module 15 then asks how the same compute is better spent at inference.
