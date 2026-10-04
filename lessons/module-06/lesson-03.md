---
id: "06.3"
module: 6
minutes: 35
practice_minutes: 75
prerequisites: ["01.3", "01.4", "06.1", "06.2"]
objectives:
  - Define main effects and the interaction contrast of a two-factor experiment and compute them, with a paired interval, from measured cell losses.
  - Explain, with numbers, why picking the best result of each separately tested branch overstates the combination's gain (selection on noise, interactions, adding costs) and what a defensible integration plan does instead.
  - Choose between a full factorial and a factorial-lite design for k components and compute the run count and the projected GPU-hours of each.
  - Run a 2 × 2 factorial of MTP and mHC that reuses the branch runs, and report the interaction with its uncertainty and the limits of what it shows.
volatility: concept
sources:
  - title: "DeepSeek-AI — DeepSeek-V3 Technical Report (section 2: MLA, DeepSeekMoE, aux-loss-free balancing and MTP combined; section 4.5 ablations, one change at a time)"
    url: https://arxiv.org/abs/2412.19437
  - title: "Moonshot AI — Kimi K2: Open Agentic Intelligence (section 2.1: MLA rules out QK-Norm, so Muon needs QK-Clip)"
    url: https://arxiv.org/abs/2507.20534
  - title: "DeepSeek-AI — Conditional Memory via Scalable Lookup (Engram), section 2.4: the module adapted to an mHC backbone"
    url: https://arxiv.org/abs/2601.07372
  - title: "NIST/SEMATECH e-Handbook of Statistical Methods, section 5.3.3 (selecting a design; full and fractional factorials in 5.3.3.3–5.3.3.4)"
    url: https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm
last_verified: "2026-10-04"
---

# 06.3 · Combining changes

Modules 3 to 6 tested one change at a time against Baseline-0: an attention design, a context method, multi-token prediction, a residual design. A real model carries several of them at once, and the gains measured one by one rarely add up — some changes overlap, some help each other, and every one adds cost. This lesson defines what an interaction is and how to measure it, shows why "take the best result of each branch" is not an integration plan, sizes the experiment that combining needs, and runs a 2 × 2 factorial of MTP and mHC that reuses the runs of lessons 06.1 and 06.2.

## Why this matters at a frontier lab

Frontier reports describe combinations: DeepSeek-V3 is MLA + DeepSeekMoE with auxiliary-loss-free balancing + MTP + FP8 training (sections 2–3); its ablations change one thing at a time on a fixed backbone (section 4.5). Kimi K2 shows a documented interaction between two choices made in different places: with MLA, QK-Norm cannot be applied to the keys (the key matrices are not materialised at inference), so training with Muon at scale needed a new mechanism, QK-Clip (K2 section 2.1, lesson 07.2). Engram had to be redesigned to fit an mHC backbone — one table shared, one gate per residual branch (Engram section 2.4, lesson 06.4). An integration run is expensive (the module project's main path is a ~350M-parameter model), so a lab plans it: which branches go in, what the combination is compared against, and how many runs it takes to know whether the combination is better than its best single branch.

## The idea

### Main effects and interaction

Two changes A and B, each on or off: four cells with mean held-out losses $m_{00}$ (Baseline-0), $m_{10}$ (A only), $m_{01}$ (B only), $m_{11}$ (both). Write

$$m_{ab} = \mu + a\,\alpha + b\,\beta + a b\,\gamma, \qquad a, b \in \{0, 1\}.$$

$\alpha = m_{10} - m_{00}$ is A's effect without B, $\beta = m_{01} - m_{00}$ is B's without A, and the **interaction**

$$\gamma = m_{11} - m_{10} - m_{01} + m_{00} = (m_{11} - m_{01}) - (m_{10} - m_{00})$$

is how much A's effect changes when B is on (symmetrically, B's when A is on). For losses (lower is better): $\gamma = 0$ — the effects add; $\gamma > 0$ — the combination gains less than the sum (the changes overlap, or one disturbs the other); $\gamma < 0$ — they help each other. The **additive prediction** for the combination is $m_{00} + \alpha + \beta$; the measured $m_{11}$ minus that prediction is exactly $\gamma$.

### Measuring it with uncertainty

Every cell is scored on the same evaluation windows, so the contrast can be computed *window by window*, $d_w = \ell_{11,w} - \ell_{10,w} - \ell_{01,w} + \ell_{00,w}$, and bootstrapped over windows (lesson 01.4): window difficulty cancels exactly. Seed variance does not cancel. If one cell's seed-averaged loss has standard error $s$, the contrast — a sum of four independent cells with coefficients $\pm 1$ — has standard error $\sqrt{4}\,s = 2s$. **An interaction is twice as hard to detect as a main effect** with the same seeds: the minimum detectable interaction is about twice the MDE of a single comparison.

### Why "best of each branch" is not a plan

Suppose four branches were each tested against Baseline-0 and the integration takes every branch whose result beat Baseline-0. Four things go wrong:

1. **Selection on noise (the winner's curse).** Each branch's measured gain is its true gain plus noise. Keeping branches *because* their measured gain was large keeps the lucky ones; the combination then disappoints. If four branches all have true effect 0 and each comparison has noise $\sigma$, the expected best measured gain is $1.03\sigma$ (the expected maximum of four standard normals) — a gain produced by selection alone.
2. **Interactions.** Effects measured on Baseline-0 are effects *on Baseline-0*. Two changes that both shorten the path to the same information (two mechanisms for using long-range context, two ways of adding capacity) overlap; their sum overstates the combination.
3. **Costs add, gains do not.** Each branch's FLOPs, wall-clock, memory and engineering cost add, and the combination's budget must be charged with all of them. A branch that won at equal tokens may lose at the combination's equal-FLOPs axis.
4. **The combination is a new model.** Its best learning rate and schedule may differ (lesson 07.3); a combination run with the baseline's hyperparameters can understate it, and a combination tuned harder than its baseline can overstate it.

A defensible plan states in advance: the components and why each is in (it passed its own correctness suite and its own test), the axis (equal training FLOPs against a Baseline-0 *scaled to the same budget*), the decision rule (the combination must beat its *best single branch* at that axis, not just Baseline-0), and a design that can attribute the result.

### Designs

- **One-factor-at-a-time** (what Modules 3–6 did): baseline + $k$ singles. Cheap; says nothing about interactions.
- **Full factorial:** all $2^k$ cells. Every main effect and every interaction is estimable; the run count doubles with each component.
- **Factorial-lite** (the course's name for a common compromise; NIST's fractional factorials are the formal version): baseline, each single, the full combination, and the combination with each component left out — at most $2k + 2$ cells. The singles give each component's effect on Baseline-0, the leave-one-out cells its effect *inside* the combination; the difference between the two is that component's total interaction with the rest. Pairwise interactions are not separately identifiable.

## Worked example

### A 2 × 2 by hand

Cell means (nats): $m_{00} = 6.00$, $m_{10} = 5.80$ (A), $m_{01} = 5.90$ (B), $m_{11} = 5.75$.

- $\alpha = -0.20$, $\beta = -0.10$; additive prediction $6.00 - 0.20 - 0.10 = 5.70$.
- $\gamma = 5.75 - 5.80 - 5.90 + 6.00 = +0.05$: with B on, A helps only $5.90 - 5.75 = 0.15$ instead of 0.20. A third of B's gain is "used up" by A.
- If one cell's seed-averaged standard error is $s = 0.007$ (seed std 0.01, two seeds), the contrast's is $0.014$; with the course's normal-approximation MDE factor $1.96 + 0.84 = 2.8$, the smallest reliably detectable interaction is about $0.04$ — so $\gamma = 0.05$ would be only just visible.

### The winner's curse in numbers

Four branches with true effect 0; each measured gain has std $\sigma = 0.01$ nats. The expected largest of four measured gains is $1.03 \cdot 0.01 = 0.0103$ nats; taking every branch that "beat" the baseline keeps on average two of them, each with an expected measured gain of $0.8 \cdot 0.01 = 0.008$ (the mean of a half-normal), and an additive prediction for the combination of $-0.016$ nats — for a combination whose true effect is 0.

### The integration budget for Lineage-F

Four components (MLA, MTP, mHC, MoE), 2 seeds:

| Design | Cells | Runs | What it identifies |
|---|---|---|---|
| one-at-a-time | 5 | 10 | main effects on Baseline-0 only |
| factorial-lite | 10 | 20 | main effects, each component's effect inside the combination, combination vs best single |
| full factorial | 16 | 32 | every main effect and interaction |

Plus the equal-FLOPs Baseline-0 (2 runs). On the project's main path a run costs about 25 H100-hours (PROJECTED, plan 12.1); 22 runs is 550 GPU-hours, which is why the project runs factorial-lite at full scale only for the cells the decision needs, and the full design on the pilot ladder.

## Shapes and cost

There are no new tensors in this lesson. The costs are runs:

| Quantity | Formula | CPU factorial here |
|---|---|---|
| cells | 4 (2 × 2) | b0, mtp-ds, mhc, mtp-ds+mhc |
| runs | cells × seeds | 8, of which 6 already exist from 06.1 and 06.2 |
| new training FLOPs | $\sum_{\text{new cells}} \text{FLOPs/token} \times \text{tokens}$ | $2 \times 19.99 \times 10^6 \times 409{,}600 = 1.6 \times 10^{13}$ |
| SE of the interaction | $2 \times$ SE of one cell | from b0's seed std |

Reusing runs is only valid when they are the same runs: same command line, code, data, seeds and evaluation windows (check with `python -m frontierlab.record`). The course's helper keeps every Module 6 arm at `runs/m06/<variant>/<arm>/s<seed>` so that a cell trained in one lesson is found, not retrained, by the next.

## Build it

The lab code is small on purpose: `lab.py` has four functions — `interaction`, `interaction_ci` (window-paired bootstrap of the contrast with `frontierlab.stats.bootstrap_ci`), `additive_prediction` and `design_cells` (full and factorial-lite cell sets). `factorial.py` trains the missing cells through `frontierlab.blocks.train` (the combined cell is `--mtp deepseek --mtp-schedule deepseek --residual mhc --streams 4`, both switches in one `BlockLM`) and runs the analysis. The combined model passes the same correctness suite as its parts: causal check and cached-decode agreement for `mhc + moe + deepseek MTP + Engram` and for `mhc + deepseek MTP + MoE` over MLA attention are in `labs/common/tests/test_blocks.py`. A combination is a new architecture; it gets its own correctness run before its own comparison.

## What the evidence says

- **Combining published changes — REASONABLE INDUSTRY PRACTICE, rarely documented with interactions.** Reports present the final combination and single-change ablations on a fixed backbone (DeepSeek-V3 section 4.5; mHC section 5 runs on an MLA + MoE backbone; Engram on an mHC backbone). We found no frontier report that publishes a factorial of its architecture changes. PUBLICLY DOCUMENTED interactions exist as design constraints rather than measurements: MLA and QK-Norm in Kimi K2 (section 2.1), Engram's per-branch gates for mHC (section 2.4).
- **Factorial and fractional-factorial designs — ESTABLISHED** in experimental design (NIST handbook sections 5.3.3.3–5.3.3.4); their use for architecture ablations at scale is limited by cost.
- **What would change the plan:** an interaction larger than the main effects (then the combination must be tuned and tested as a unit), or a component whose leave-one-out effect is zero (then it is not earning its cost inside the combination, whatever it did alone).

## Lab

**Folder:** [`labs/module-06/lesson-03/`](../../labs/module-06/) · **Time:** about 75 minutes (about 25 of them unattended) · **Pass check:** `pytest labs/module-06/lesson-03` passes; your notes contain the filled contract, the 2 × 2 table, the interaction with its paired interval and per-seed values, the comparison of additive prediction and measurement, and the integration budget for Lineage-F.

### Experiment contract

- **Question:** do the effects of MTP and the mHC residual on held-out loss add, at this scale? Decision informed: whether the Lineage-F project may plan with the additive prediction or must measure the combination directly (it must anyway; this tells you how wrong the shortcut would be).
- **Hypothesis and status:** $\gamma = 0$ (additivity) is the null; no published measurement exists for this pair — **unknown, and at this scale both main effects may themselves be within noise** (lessons 06.1 and 06.2).
- **Baseline:** the b0 cell (Baseline-0's runs from lesson 06.1).
- **Changed variables (a declared factorial):** MTP (DeepSeek, $D = 1$, V3 λ schedule) and the residual (plain → mHC, $n = 4$, $t_{\max} = 20$), crossed. **Controlled:** everything else, identical to 06.1/06.2 (preset, steps, seeds {0, 1}, data order, learning rate, Eval v0 windows).
- **Comparison axis:** **equal tokens** for every cell. The interaction is a question about mechanisms — does MTP's per-token effect change when the residual changes — so every cell sees the same data; the cells' costs differ (reported per cell) and the adoption decision is made at equal FLOPs in the project, not here.
- **Budget:** free CPU about 25 minutes for the new cell's two seeds (measured below); main path PROJECTED below.
- **Metrics and decision rule:** cell means per seed; main effects and $\gamma$ with a 95% bootstrap over the 256 windows (seed-averaged) and per seed. Rule: report "interaction detected" only if the interval excludes 0 *and* both per-seed values have the same sign; otherwise "no interaction detected at this power", with the minimum detectable interaction ($\approx 2\times$ the cell MDE).
- **Correctness checks:** the combined model passes `pytest labs/common/tests/test_blocks.py -k causal`; `python -m frontierlab.record runs/m06/cpu/mhc/s0 runs/m06/cpu/mtp-ds+mhc/s0 --changed config.extra.blocks.mtp config.extra.blocks.mtp_depth` shows no unexpected INVALIDATES line (the reused cells must be the same runs).
- **Fallback evidence:** none needed: a null is a result, reported with its detectable size.
- **Limits:** two seeds, one scale, one pair of changes; an interaction (or its absence) at 1.8M parameters does not transfer to 350M.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100. Not run in this build; part of the Module 6 pilot | `factorial.py --variant main --device cuda` after lessons 06.1 and 06.2's main runs (only the combined cell is new: 2 runs of pilot-30m with MTP + mHC, $2 \cdot 4.55 \times 10^8 \cdot 2.62 \times 10^8 = 2.4 \times 10^{17}$ FLOPs, PROJECTED 0.3 GPU-hours at 25% MFU, times mHC's measured wall-clock factor) |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `factorial.py --variant t4 --device cuda` |
| Free CPU | laptop; measured below | as written |

### Steps

1. **Implement** `interaction`, `interaction_ci`, `additive_prediction` and `design_cells`; run `pytest labs/module-06/lesson-03`. The pairing test gives the four cells window losses with a standard deviation of 1 nat between windows: why is the interval still about ±0.002?
2. **Predict** the sign of $\gamma$ for MTP × mHC before you look, with one sentence of reasoning.
3. **Run:** `python labs/module-06/lesson-03/factorial.py` (trains the two missing runs, then analyses).
4. **Report** the table, $\gamma$ with its interval and per-seed values, and the additive-prediction error. Then write the integration-budget paragraph for the project: which design, how many runs, at what cost.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, with another build job sharing the CPU, 2026-10-04):

| Cell | MTP | mHC | Held-out, seeds 0 / 1 | Mean | Training MFLOP/token | Wall-clock, seed 0 |
|---|---|---|---|---|---|---|
| b0 | 0 | 0 | 6.6857 / 6.6762 | 6.6810 | 11.41 | 112 s |
| mtp-ds | 1 | 0 | 6.6746 / 6.6630 | 6.6688 | 19.18 | 209 s |
| mhc | 0 | 1 | 6.6642 / 6.6719 | 6.6680 | 12.22 | 267 s |
| mtp-ds+mhc | 1 | 1 | 6.6350 / 6.6474 | 6.6412 | 19.99 | 298 s |

- Main effects (paired, seed-averaged): MTP without mHC −0.0122 [−0.0155, −0.0089], *with* mHC −0.0268 [−0.0298, −0.0239]; mHC without MTP −0.0130 [−0.0156, −0.0103].
- **Interaction $\gamma = −0.0147$ [−0.0185, −0.0109]**, per seed −0.0181 and −0.0113. Additive prediction 6.6558, measured 6.6412.
- Runtime: the two new runs of the combined cell 10 minutes in all (the other six cells reused from 06.1 and 06.2); analysis seconds.
- Integration budget for Lineage-F's four components with 2 seeds: full factorial 32 runs, factorial-lite 20.

What to write about it. By the pre-stated rule this is "interaction detected": the window-paired interval excludes 0 and both seeds agree — at this scale the two changes help each other (the combination gains more than the sum). Two cautions belong in the same paragraph. First, the window interval does not contain seed noise; the contrast's seed-level standard error is about $2 \times 0.0067/\sqrt 2 \approx 0.0095$, so $-0.0147$ is about 1.5 standard errors — suggestive, not strong. Second, every effect here is at equal tokens, and both main effects are below the 0.019 MDE; the combined cell also costs 75% more FLOPs and 2.7× the wall-clock of b0. "They interact" is a statement about mechanisms at 1.8M parameters; whether the combination is worth its cost is the project's equal-FLOPs question.

<details>
<summary>Hint for TODO 2</summary>

Build one array `d = l11 - l10 - l01 + l00` (element-wise over windows) and pass it to `bootstrap_ci(d, n_boot=..., seed=...)`, which returns `(point, low, high)` for the mean. Bootstrapping each cell separately and combining would throw the pairing away.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-03`.

</details>

## Common mistakes

- **Reading the additive prediction as a forecast.** It is the null hypothesis of the factorial, not an estimate of the combination; the combination must be measured.
- **Keeping branches because they "won" on noisy single comparisons.** Correct for selection: decide with pre-registered rules and confirm the combination against its best single branch on fresh seeds.
- **Forgetting that the contrast has four cells.** Its standard error is twice a single cell's; two seeds that barely resolve a main effect cannot resolve an interaction of the same size.
- **Reusing runs that are not the same runs.** A cell from another lesson with a different learning rate, eval window seed or code version is a different experiment; `frontierlab.record` tells you.
- **Comparing the combination with Baseline-0 only.** The question is whether the combination beats its best single branch at the same budget; beating Baseline-0 is already true of that branch.

## References

- DeepSeek-AI, *DeepSeek-V3 Technical Report*, sections 2 and 4.5. https://arxiv.org/abs/2412.19437
- Moonshot AI, *Kimi K2: Open Agentic Intelligence*, section 2.1. https://arxiv.org/abs/2507.20534
- DeepSeek-AI, *Conditional Memory via Scalable Lookup (Engram)*, section 2.4. https://arxiv.org/abs/2601.07372
- NIST/SEMATECH, *e-Handbook of Statistical Methods*, section 5.3.3 (selecting a design; full and fractional factorials in 5.3.3.3–5.3.3.4). https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm
- Lesson 01.4 (paired bootstrap, MDE): [01.4 · Uncertainty](../module-01/lesson-04.md).

## Next

[06.4 · Conditional memory and lookup sparsity: Engram](lesson-04.md)
