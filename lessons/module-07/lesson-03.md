---
id: "07.3"
module: 7
minutes: 40
practice_minutes: 90
prerequisites: ["07.1", "01.3", "01.4"]
objectives:
  - State the µP rules for initialisation, learning rate and the output multiplier relative to a base width, and derive the hidden-layer learning-rate rule from how one Adam step changes a layer's output.
  - Derive the corresponding learning-rate rule for Muon from its update scaling, and explain why the Adam rule and the spectral rule disagree for an orthogonalised update.
  - Implement µP for the course model and verify it with a coordinate check before trusting any sweep.
  - Design and run a transfer test with a pre-stated decision rule, and report whether the best learning rate moved across widths under SP and under µP.
volatility: concept
sources:
  - title: "Yang et al. — Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (Table 1, Table 3, Definition 4.1)"
    url: https://arxiv.org/abs/2203.03466
  - title: "Yang, Simon, Bernstein — A Spectral Condition for Feature Learning"
    url: https://arxiv.org/abs/2310.17813
  - title: "Essential AI — Practical Efficiency of Muon for Pretraining (section 3: muP with Muon, telescoping)"
    url: https://arxiv.org/abs/2505.02222
  - title: "Microsoft — mup package (MuReadout, MuSharedReadout, base shapes, coordinate check)"
    url: https://github.com/microsoft/mup
  - title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (µP-based 'model wind tunnel' experiments)"
    url: https://arxiv.org/abs/2404.06395
last_verified: "2026-10-04"
---

# 07.3 · Hyperparameter transfer

The learning rate that is best for a 10M-parameter model is usually not the best for a 1B one, and nobody can afford to sweep learning rates at 1B. µP (the maximal update parametrization) rescales initialisation, learning rate and the output layer with width so that the best hyperparameters stop moving as the model widens; you tune a small proxy and transfer the result. This lesson derives the rules, including the one for Muon, implements them for the course model relative to a base width, checks the implementation with a coordinate check, and runs a transfer test — a learning-rate sweep at three widths under the standard parametrization and under µP — with a decision rule written before the runs.

## Why this matters at a frontier lab

Every frontier run is a single shot at its size: its learning rate, initialisation and schedule were chosen on smaller models. Tensor Programs V reports tuning a 40M-parameter proxy and transferring to GPT-3 6.7B, outperforming the published GPT-3 6.7B with a tuning cost of "only 7% of total pretraining cost" (abstract). MiniCPM ran its hyperparameter search on small µP models ("model wind tunnel experiments", section 3.1) before training its released models. Essential AI tuned Muon with µP up to 3.7B parameters (section 3.4). A research engineer who can say "this learning rate came from a transfer test at widths X and Y, and here is the evidence that it transfers" saves the lab a sweep at the target size — and one who cannot is guessing with the most expensive run of the year.

## The idea

### What a parametrization is

A **parametrization** says, for each weight matrix, how its initialisation scale, its learning rate and any fixed multiplier depend on width $n$. The **standard parametrization (SP)** used by the course so far — and by most codebases — draws every weight from $\mathcal N(0, 0.02^2)$ and uses one learning rate for everything. Under SP, the best learning rate shrinks as the model widens, because one optimizer step changes a wide layer's output more than a narrow one's. µP chooses the scalings so that, at every width, every layer's activations change by $\Theta(1)$ per step (they neither vanish nor blow up as $n \to \infty$), which is the condition under which the training dynamics — and therefore the best hyperparameters — have a width-independent limit.

### Why the hidden learning rate must scale as $1/n$ for Adam

Take a hidden layer $y = W x$ with $W \in \mathbb{R}^{n \times n}$ and input entries of size $\Theta(1)$. Its gradient is $\nabla W = \delta\, x^\top$ (outer product of the output gradient $\delta$ and the input). Adam normalises each entry, so the update is roughly $\Delta W \approx -\eta\, \mathrm{sign}(\delta)\,\mathrm{sign}(x)^\top$: every entry has size $\eta$, and the update is aligned with $x$. Then

$$\Delta y = \Delta W\, x \approx -\eta\, \mathrm{sign}(\delta) \sum_{j=1}^{n} |x_j| = \Theta(\eta\, n).$$

For $\Delta y = \Theta(1)$ at every width we need $\eta \propto 1/n$. The initialisation follows the usual forward argument: $y_i = \sum_j W_{ij} x_j$ has variance $n \sigma^2$, so $\sigma^2 \propto 1/n$ (fan-in initialisation). The output layer is special: its output (the logits) must not be dominated by the random initialisation, so µP gives it variance $1/n^2$ (TP V Table 3) — equivalently, a fixed multiplier $1/n$ on the logits.

### The rules, relative to a base width

Tensor Programs V, Table 3, for Adam:

| Weight | Init variance | Adam learning rate |
|---|---|---|
| input (embedding) | $1/\text{fan\_in}$, constant in width | constant |
| hidden | $1/\text{fan\_in}$ | $1/\text{fan\_in}$ |
| output | $1/\text{fan\_in}^2$ | $1/\text{fan\_in}$ |

In practice the rules are stated relative to a **base width** $n_0$ at which µP and SP coincide (the `mup` package's "base shapes"), so that hyperparameters tuned in SP at $n_0$ are the starting point. With width multiplier $m = n / n_0$, for the course model (`frontierlab.optim.mup`):

- hidden matrices (q, k, v, o; gate, up, down): init std $0.02/\sqrt m$, learning rate $\eta / m$;
- embedding: std $0.02$, learning rate $\eta$ (its fan-in is the vocabulary, which does not grow);
- tied output head: Baseline-0's head *is* the embedding, so it cannot have its own init or learning rate. Like `mup.MuSharedReadout`, the logits are multiplied by $1/m$ instead (`MuReadout`; TP V shows that multiplier and init/LR forms are equivalent);
- norm gains and vectors: learning rate $\eta$;
- attention: TP V uses $q^\top k / d$ instead of $q^\top k/\sqrt d$ (Definition 4.1). That matters only if the head dimension grows with width; our width ladder keeps $d$ fixed and adds heads, so $1/\sqrt d$ stays (REASONABLE INDUSTRY PRACTICE).

TP V's Table 1 separates what transfers — "optimization related, init, parameter multipliers" — from what does not — "regularization (dropout, weight decay, etc)" — and transfers *across* width, with depth, batch size, training time and sequence length marked as only empirically validated. A transfer test that changes width only is testing the part the theory covers.

### Muon under µP

The Adam argument used two facts about the update: entries of size $\eta$, and alignment with the input (a low-rank update). A Muon update is neither: it is *full rank* by construction. With Moonlight's matching, $\Delta W = -\eta \cdot 0.2\sqrt n\, O$ where $O$ has all singular values $\approx 1$, so $\|\Delta W x\| = 0.2\,\eta \sqrt n\, \|x\|$ and, with $\|x\| = \Theta(\sqrt n)$, each output entry changes by $0.2\,\eta\sqrt n$. For $\Theta(1)$ we need

$$\eta \propto 1/\sqrt{m} \quad (\text{Muon, update RMS matched to 0.2}), \qquad \eta = \text{const} \quad (\text{Muon, Jordan's } \sqrt{\max(1, A/B)}).$$

The second case is the **spectral condition** of Yang, Simon and Bernstein: feature learning requires the spectral norm of weight updates to scale like $\sqrt{\text{fan\_out}/\text{fan\_in}}$; Jordan's scaling does exactly that at a fixed learning rate. So for an RMS-matched Muon the entry-size intuition ("Muon looks like Adam, so use Adam's $1/m$") and the spectral argument disagree by $\sqrt m$. This derivation is INFERENCE from the two papers' general statements, not a rule either of them states for Moonlight's scaling. Essential AI report that "the same muP scaling that is used for AdamW … works for Muon" in their experiments (section 3.4; their Table 5 gives the scalings they used); `frontierlab.optim.mup.lr_scales` implements both rules (`muon_rule="spectral"` or `"adam"`) so the transfer test can decide for your setup.

### Designing a transfer test

A transfer test is an experiment with a contract, like any other:

1. **Ladder:** at least three widths, the largest the target or as close as you can afford; everything else fixed (depth, tokens, batch, schedule, data order, seed).
2. **Grid:** a log-spaced learning-rate grid wide enough that the optimum is interior at every width (if the best value sits on the edge of the grid, you have not found it).
3. **Both parametrizations,** SP and µP, with the same grid and the same number of runs: the comparison is "does µP move the optimum less than SP?", not "does µP work".
4. **Decision rule, stated now:** the optimum at every width is within one grid step of the optimum at the smallest width (with a factor-3 grid, that is a factor 3 in learning rate). Secondary: the loss at the transferred learning rate is within the seed noise floor of the best loss at that width.
5. **Noise:** a one-seed sweep can move the argmin by one step through seed noise alone when neighbouring grid points are within the noise floor (lesson 01.4). Report the loss differences between the best and second-best grid points next to the noise floor.

## Worked example

### The Adam rule with $n = 4$ and $n = 16$

Input $x = (1, 1, 1, 1)$, Adam update with every entry $-\eta$ in the aligned direction: $\Delta y = -4\eta$ per output. At $n = 16$, $x = (1, \dots, 1)$: $\Delta y = -16\eta$. With $\eta$ scaled by $1/m = 1/4$ at the wider layer: $-16 \cdot \eta/4 = -4\eta$ — the same change as the narrow layer.

### The course model at width 512 from base 128

$m = 4$. Hidden init std $0.02/\sqrt 4 = 0.01$; AdamW hidden learning rate $\eta/4$ (with $\eta = 10^{-2}$: $2.5 \times 10^{-3}$); embedding std $0.02$, learning rate $10^{-2}$; logit multiplier $0.25$; Muon (RMS-matched, spectral rule) hidden learning rate $\eta/\sqrt 4 = 5 \times 10^{-3}$. At width 128 every factor is 1: µP and SP are the same parametrization (with different random draws).

### The Muon rule with $n = 4$

$O = I_4$ (orthogonal), RMS-matched scale $0.2\sqrt 4 = 0.4$, so $\Delta W = -0.4\eta\, I$ and $\Delta y = -0.4\eta\, x = -0.4\eta$ per entry. At $n = 16$: scale $0.2 \cdot 4 = 0.8$, $\Delta y = -0.8\eta$ per entry — twice as large for 4× the width, i.e. $\propto \sqrt m$. Scaling $\eta$ by $1/\sqrt 4 = 0.5$ restores $-0.4\eta$. (An Adam-style $1/m$ would give $-0.2\eta$: too small by $\sqrt m$.)

## Shapes and cost

The width ladder (`mup.width_config`, toy preset, head dimension 32 fixed, vocabulary 8,192):

| width | heads / KV heads | SwiGLU width | non-embedding params | total params |
|---|---|---|---|---|
| 64 | 2 / 1 | 192 | 0.20M | 0.72M |
| 128 (base, = toy) | 4 / 2 | 384 | 0.79M | 1.84M |
| 256 | 8 / 4 | 768 | 3.15M | 5.25M |
| 512 | 16 / 8 | 1,536 | 12.6M | 16.8M |

µP itself costs nothing at run time: one extra multiply on the logits (B, T, V) and per-group learning-rate factors in the optimizer (`lr_scale` in `MuonAdamW`'s parameter groups). The cost is the sweep: a transfer test with $W$ widths, $G$ grid points and two parametrizations is $2WG$ runs, dominated by the widest. On the main path (widths 256/512/1,024 of `pilot-30m`, 2,000 steps of 32 × 1,024 tokens, 5 grid points) that is 30 runs; PROJECTED cost from `accounting.flops_per_token` × tokens: $6.55 \times 10^{7}$ tokens per run; at width 1,024, $1.02 \times 10^{9}$ training FLOPs per token, so $6.7 \times 10^{16}$ FLOPs, about 0.09 H100-hours at an assumed 20% MFU for each widest run; the whole sweep is $9.5 \times 10^{17}$ FLOPs, about 1.3 GPU-hours (formula: $\sum_{\text{runs}} \text{FLOPs per token} \times \text{tokens} / (989 \times 10^{12} \cdot 0.2)$; pending the Module 7 pilot).

Tensors and dtypes are unchanged from Baseline-0: weights fp32 (master), activations bf16 under autocast on the GPU, logits (B, T, V) fp32 — now multiplied by $1/m$ before the loss.

## Build it

`labs/common/frontierlab/optim/mup.py`:

```python
cfg = mup.width_config(toy(vocab_size=8192), 512)       # head_dim fixed, heads and SwiGLU scaled
model = OptLM(cfg)                                       # frontierlab.optim.stabilizers.OptLM = LM + options
mup.apply_mup(model, base_width=128)                     # hidden std 0.02/sqrt(m); head -> MuReadout(1/m)
scales = mup.lr_scales(model, 128, "adamw")              # {"model.layers.0.self_attn.q_proj.weight": 0.25, ...}
opt = MuonAdamW(param_groups(model, optimizer="adamw", lr=1e-2, lr_scales=scales))
```

From the command line: `python -m frontierlab.optim.train --width 512 --mup-base-width 128 --optimizer adamw ...`. The run card's `config.extra.mup` records base and width, and `frontierlab.optim.train.load_model` rebuilds the readout multiplier when a checkpoint is evaluated (a plain `LM` would silently drop it — see Common mistakes).

Correctness checks (`labs/common/tests/test_optim.py`): at the base width every learning-rate factor is 1 and the multiplier is 1; at $m = 4$ the measured init std of a hidden matrix is $0.01 \pm 0.001$ and of the embedding $0.02 \pm 0.001$; the head keeps the name `lm_head.weight` and stays tied to the embedding; exact resume holds with µP on. The **coordinate check** is the end-to-end test: train a few steps at several widths and look at how much the activations changed (`coord_check.py`, step 2 of the lab). A µP bug — a missing multiplier, a group that did not get its factor — shows up there as a change that grows or shrinks with width.

## What the evidence says

- **µP / µTransfer — ESTABLISHED.** PUBLICLY DOCUMENTED in TP V (GPT-3 6.7B from a 40M proxy; BERT-large from 13M) and used by several labs in published work (MiniCPM's wind-tunnel experiments, section 3.1 of its report, which found the optimal base learning rate stayed "around 0.01" over a 10× change in model size; Essential AI with Muon). Not every lab uses it, and what transfers beyond width is "empirically validated" only (TP V Table 1).
- **Muon + µP — PROMISING.** Essential AI report transfer up to 3.7B with their telescoping protocol (section 3.3–3.4) and that the AdamW µP scaling worked for Muon. Which learning-rate rule applies to an RMS-matched Muon is not settled in the papers we checked; the spectral-condition argument in this lesson is INFERENCE.
- **Hyperparameter drift.** Even under µP the optimum drifts at finite width (Essential AI, appendix J, a $1/n$ correction); a transfer test at two or three widths measures that drift instead of assuming it away.
- **Course-scale hypotheses** (lab): (H1) the coordinate check is flat across width under µP and grows under SP; (H2) the best learning rate moves by at most one grid step across widths 64–256 under µP; (H3) it moves more under SP. H1 is a property of the implementation and should hold at any scale. H2 and H3 may not show cleanly at 0.2M–3M non-embedding parameters and 0.3M tokens per run, where the embedding and head dominate the parameter count and one-seed noise can move the argmin.

## Lab

**Folder:** [`labs/module-07/lesson-03/`](../../labs/module-07/) · **Time:** about 90 minutes (about 60 of them unattended) · **Pass check:** `pytest labs/module-07/lesson-03` passes; `coord_check.py` output and the two sweep tables are in your notes with the decision rule applied to each parametrization.

### Experiment contract

- **Question:** at widths 64, 128 and 256 of the toy model, does the best AdamW learning rate stay put under µP (relative to width 128) and move under SP? Decision informed: whether the module project may tune at width 64 and transfer to width 128 (and the main path from 384 to 768).
- **Hypotheses and status:** H1–H3 above; H1 established (TP V), H2–H3 reported at larger scale (TP V, Essential AI), may not appear cleanly at this scale.
- **Baseline:** SP at the same widths and grid; the same number of runs (12) as µP — equal tuning budgets.
- **Changed variable:** the parametrization (SP vs µP). **Controlled:** depth 4, head dimension 32, 150 steps of 16 × 128 tokens (0.31M tokens), cosine schedule with 15 warmup steps, AdamW (β = 0.9, 0.95, weight decay 0.1) in `MuonAdamW`, seed 0, data order, 256 fixed held-out windows of 128 tokens.
- **Comparison axis:** equal tokens at each width (the transfer claim is about a fixed training recipe across width; it says nothing about compute-optimal size).
- **Budget:** free CPU, measured below; main path PROJECTED about 1.3 H100-hours (formula in Shapes and cost).
- **Metrics and decision rule:** mean held-out loss per run; best grid point per width. Rule (stated now): a parametrization *transfers* if the best learning rate at 128 and at 256 is within one grid step (a factor of about 3) of the best at 64. H3 is observed if SP's shift is larger than µP's. Report next to each argmin the gap to the second-best grid point; a gap below the Module 1 seed noise floor means the argmin is not resolved.
- **Correctness checks:** `pytest labs/common/tests/test_optim.py -k mup` and `pytest labs/module-07/lesson-03` pass; the coordinate check (step 2) shows µP's change roughly flat across width before any sweep counts.
- **Fallback evidence:** TP V's published sweeps (its Figure 1, Figure 3) and the Module 7 pilot's sweep at `pilot-30m` widths, labelled as analysis of published or provided results.
- **Limits:** one seed, short runs, tiny widths where the embedding dominates; width only (not depth, batch or duration).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100; PROJECTED about 1.3 GPU-hours. Not run in this build; part of the Module 7 pilot | `python labs/module-07/lesson-03/sweep.py --variant main` (widths 256/512/1,024 of `pilot-30m`, 5 learning rates, 2,000 steps of 32 × 1,024) |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `sweep.py` with `--variant cpu` but `--device cuda` added to the loop arguments in `VARIANTS` and widths (128, 256, 512) |
| Free CPU | laptop; measured below | the steps below as written |

### Steps

1. **Implement** `width_mult`, `mup_init_std`, `mup_lr_scale`, `readout_mult`, `best_lr` and `transfer_verdict` in `lab.py`; run `pytest labs/module-07/lesson-03`.
2. **Coordinate check:** `python labs/module-07/lesson-03/coord_check.py`. Write down how the change in the last block's output after 5 steps scales from width 64 to 512 under SP and under µP.
3. **Sweep** (unattended): `python labs/module-07/lesson-03/sweep.py`. Before it finishes, write your prediction of the best learning rate at each width for SP and µP.
4. **Decide:** `python labs/module-07/lesson-03/sweep.py --analyze` applies your `transfer_verdict`. For each parametrization, state "transfers" or "does not transfer" by the rule, the gap between the best and second-best grid point at each width, and whether that gap is above the noise floor you measured in Module 1.
5. **Extension (optional):** `python labs/module-07/lesson-03/sweep.py --optimizer muon` repeats the test for Muon with the spectral rule; edit `--mup-muon-rule adam` into the script's arguments to test the Adam rule instead.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04):

`coord_check.py` (46 seconds): RMS before training, and RMS of the change after 5 AdamW steps at learning rate $10^{-2}$ (µP relative to width 128).

| width | SP logits | SP Δlogits | SP last block | SP Δlast block | µP logits | µP Δlogits | µP last block | µP Δlast block |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.160 | 1.43 | 0.031 | 3.1 | 0.321 | 2.72 | 0.064 | 22.5 |
| 128 | 0.226 | 2.93 | 0.067 | 17.6 | 0.226 | 2.69 | 0.065 | 16.6 |
| 256 | 0.320 | 3.48 | 0.145 | 56.3 | 0.160 | 3.02 | 0.064 | 12.3 |
| 512 | 0.451 | 3.30 | 0.304 | 470.8 | 0.113 | 3.06 | 0.066 | 14.3 |

Under SP the change in the last block's output grows 150× from width 64 to 512 and its initial size 10×; under µP both stay within a factor of 2 (the initial size is flat by construction, the change wanders between 12 and 22). The logits' *change* grows only 2.3× under SP because the tied embedding (learning rate unchanged in both) dominates them at this size; µP's is flat (1.13×). The initial logits shrink with width under µP, as the $1/m$ multiplier intends. H1 observed.

`sweep.py`: 24 runs in 41.7 minutes (about 50 s each at width 64, 80 s at 128, 150 s at 256). Held-out loss, 256 windows of 128 tokens, after 150 steps:

| width | SP 1e-3 | SP 3e-3 | SP 1e-2 | SP 3e-2 | µP 1e-3 | µP 3e-3 | µP 1e-2 | µP 3e-2 |
|---|---|---|---|---|---|---|---|---|
| 64 | 7.216 | **6.990** | 7.033 | 7.225 | 7.048 | **6.994** | 6.999 | 7.221 |
| 128 | 7.012 | 6.966 | **6.921** | 7.206 | 7.006 | **6.933** | 7.055 | 7.218 |
| 256 | 6.835 | **6.781** | 7.132 | 7.225 | 7.011 | **6.817** | 6.953 | 6.963 |

Verdicts by the rule. **µP transfers:** the best learning rate is $3 \times 10^{-3}$ at all three widths (shift 0). **SP also "transfers" by the letter of the rule** (best $3 \times 10^{-3}$, $10^{-2}$, $3 \times 10^{-3}$; largest shift 1 grid step), so **H3 is not observed** as an argmin shift. Two things the argmin hides, and the reason to read the whole table. First, the ties: at width 64 µP's $3 \times 10^{-3}$ and $10^{-2}$ differ by 0.005 nats, and SP's width-128 optimum beats its neighbour by 0.045 — against a toy seed std of 0.029 nats (lesson 01.4, 300 steps), neither argmin is resolved by one seed. Second, the shape: under SP the width-256 curve has turned sharply against large learning rates ($10^{-2}$ is 0.35 nats worse than $3 \times 10^{-3}$, while at width 64 it is 0.04 worse), which is exactly the drift µP removes; under µP the width-256 curve stays flat out to $3 \times 10^{-2}$ (6.96). At 150 steps all of these models are still near the start of training (losses around 7), so this is a check of the mechanism, not a measurement of transfer for real training lengths; the module project tunes at 200 steps with three seeds at the large width, and the pilot repeats the sweep at `pilot-30m` widths.

<details>
<summary>Hint for TODO 3</summary>

Decide "hidden" from the name and the number of dimensions only: a 2-D parameter whose name contains `self_attn.` or `mlp.`. The embedding, the norm gains (1-D) and anything else get factor 1. For Muon, the factor depends on the update scaling, not on Adam.

</details>

<details>
<summary>Hint for step 4</summary>

If the best grid point at width 256 is the edge of the grid (the smallest learning rate under SP is a common case), the rule cannot be applied honestly: the true optimum may be further out. Say so, and say which extra grid point you would add.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-07/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-07/lesson-03`.

</details>

## Common mistakes

- **Evaluating a µP checkpoint with a plain model.** The $1/m$ logit multiplier is not a weight; `LM(cfg)` with the saved state dict computes logits $m$ times too large. Use `frontierlab.optim.train.load_model`, which rebuilds it from `config.extra.mup`.
- **Using the chunked loss with µP.** `frontierlab.perf.chunked_ce` reads `lm_head.weight` directly and skips the multiplier; the Module 7 wrapper refuses the combination.
- **Transferring weight decay or dropout as if they were learning rates.** TP V lists regularization as not µTransferable.
- **Widening the head dimension.** Then $q^\top k/\sqrt d$ is no longer µP; either keep $d$ fixed (as here) or switch to $1/d$.
- **Calling a one-seed argmin a transfer result.** Neighbouring grid points within the noise floor are a tie, and a tie at the edge of the grid is not a result at all.
- **Applying Adam's $1/m$ to Muon without checking.** Muon's update is full rank; the scaling depends on which shape factor your implementation uses.

## References

- G. Yang, E. J. Hu et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, abstract, Table 1, Table 3, Definition 4.1. https://arxiv.org/abs/2203.03466
- G. Yang, J. B. Simon, J. Bernstein, *A Spectral Condition for Feature Learning*. https://arxiv.org/abs/2310.17813
- Essential AI, *Practical Efficiency of Muon for Pretraining*, section 3 and appendix J. https://arxiv.org/abs/2505.02222
- Microsoft, `mup` (MuReadout, MuSharedReadout, base shapes, coordinate check). https://github.com/microsoft/mup
- S. Hu et al., *MiniCPM*, section 3 (wind-tunnel experiments with µP). https://arxiv.org/abs/2404.06395
- Shared code: `labs/common/frontierlab/optim/mup.py`, `muon.py`, `train.py`.

## Next

[07.4 · Schedules](lesson-04.md)
