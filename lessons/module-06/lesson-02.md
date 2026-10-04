---
id: "06.2"
module: 6
minutes: 40
practice_minutes: 90
prerequisites: ["01.4", "02.1", "02.4", "06.1"]
objectives:
  - Write the hyper-connection update for an n-stream residual and show, with a two-layer numerical example, why an unconstrained mixing matrix breaks the identity path while a doubly-stochastic one keeps it.
  - Implement Sinkhorn-Knopp normalisation and the mHC sub-layer update, and verify the projection, the n = 1 reduction to the plain residual, and gradients in float64.
  - Compute the parameters, FLOPs and memory traffic HC and mHC add to Baseline-0, and explain why the cost axis is wall-clock, not FLOPs.
  - Run an HC vs mHC vs Baseline-0 comparison at the standard and at raised learning rates with a contract that is informative whether or not instability appears, and measure the step-time overhead with the Module 2 harness.
volatility: concept
sources:
  - title: "Xie et al. (DeepSeek-AI) — mHC: Manifold-Constrained Hyper-Connections (sections 3–5, Table 2, Table 5)"
    url: https://arxiv.org/abs/2512.24880
  - title: "Zhu et al. — Hyper-Connections (sections 2.1–2.3, 5)"
    url: https://arxiv.org/abs/2409.19606
  - title: "DeepSeek-AI — DeepSeek-V4 Technical Report (mHC in the V4 architecture; Sinkhorn t_max = 20)"
    url: https://arxiv.org/abs/2606.19348
  - title: "DeepSeek-AI — Conditional Memory via Scalable Lookup (Engram), section 2.4: experiments use mHC with M = 4"
    url: https://arxiv.org/abs/2601.07372
last_verified: "2026-10-04"
---

# 06.2 · Residual-stream design: hyper-connections and mHC

Every block of Baseline-0 adds its output to one residual stream, and the bare residual — the identity path — carries any layer's signal to any deeper layer unchanged. Hyper-connections widen that stream to n parallel streams per token and let each block learn how to read from them, write to them and mix them. The mixing is where the gains come from and also where the trouble starts: an unconstrained mixing matrix, multiplied over dozens of layers, can blow the signal up or shrink it away. DeepSeek's mHC constrains the mixing to doubly-stochastic matrices with a few Sinkhorn-Knopp iterations. This lesson builds both, checks them, counts what they cost, and runs the stability comparison as a hypothesis test that teaches something whichever way it comes out.

## Why this matters at a frontier lab

The residual stream is the one part of the transformer that almost nobody changed for years. mHC changes it and is not a paper-only idea: DeepSeek-V4 uses manifold-constrained hyper-connections with Sinkhorn $t_{\max} = 20$ (plan claim check 14.1 #3), and DeepSeek's Engram experiments run on an mHC backbone with $M = 4$ streams (Engram section 2.4). mHC reports gains over the baseline at 27B (Table 4: BBH 43.8 → 51.0, DROP 47.0 → 53.9) and a training-time overhead of 6.7% at $n = 4$ — *after* fused kernels, selective recomputation and pipeline-overlap work (section 4.3). Without that engineering, the cost is much larger, because a 4-stream residual moves about 11× the bytes of a plain one. And the stability benefit, the paper's headline over HC, appeared in a 27B run around step 12k (Figure 2). Whether it appears in a 30M model in 4,000 steps is an open empirical question. So the questions are: *does the residual change earn its wall-clock cost, and is the instability it prevents real at the scale where we decide?*

## The idea

### One stream, n streams

Plain pre-norm residual for a sub-layer $\mathcal{F}$ (attention or the FFN, each with its own RMSNorm inside):

$$x_{l+1} = x_l + \mathcal{F}(x_l), \qquad x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i).$$

The first term of the unrolled form is the identity path: $x_l$ reaches layer $L$ multiplied by exactly 1.

**Hyper-connections** (Zhu et al., section 2; written in mHC's notation, Eq. 3) keep $n$ vectors per token, $X_l \in \mathbb{R}^{n \times C}$, and give every sub-layer three small mappings:

$$X_{l+1} = H^{\mathrm{res}}_l X_l + H^{\mathrm{post}\top}_l\, \mathcal{F}\big(H^{\mathrm{pre}}_l X_l\big),$$

- $H^{\mathrm{pre}}_l \in \mathbb{R}^{1 \times n}$ reads: the sub-layer's input is a weighted sum of the streams;
- $H^{\mathrm{post}}_l \in \mathbb{R}^{1 \times n}$ writes: the sub-layer's output is added to each stream with its own weight;
- $H^{\mathrm{res}}_l \in \mathbb{R}^{n \times n}$ mixes the streams: stream $i$ of the next layer is $\sum_j H^{\mathrm{res}}_{l,ij}$ · stream $j$.

Each mapping is a learned static bias plus a dynamic term computed from the current stream ("dynamic hyper-connections", HC Eqs. 10–13: RMSNorm, a linear map, tanh, scaled by a small learned factor). The embedding is copied into all $n$ streams at the bottom; at the top the streams are summed before the final norm (HC section 2.1). HC initialises so that it starts as the pre-norm residual: dynamic weights 0, $H^{\mathrm{post}} = \mathbf{1}$, $H^{\mathrm{pre}} = e_{k \bmod n}$ (sub-layer $k$ reads one stream), $H^{\mathrm{res}} = I$ (HC Eq. 14). With $n = 1$ that *is* the plain residual — the course tests it exactly. HC's own ablation: $n = 1$ does not help; $n = 4$ does, with little extra from $n = 8$ (HC section 5, Table 1). mHC's ablation finds $H^{\mathrm{res}}$ the most important of the three (mHC Table 1).

### Why unconstrained mixing breaks the identity path

Unroll HC over depth (mHC Eq. 4):

$$X_L = \Big(\prod_{i=1}^{L-l} H^{\mathrm{res}}_{L-i}\Big) X_l + \sum_{i=l}^{L-1} \Big(\prod_{j=1}^{L-1-i} H^{\mathrm{res}}_{L-j}\Big) H^{\mathrm{post}\top}_i \mathcal{F}(H^{\mathrm{pre}}_i X_i).$$

The identity path is now a *product* of $L - l$ learned matrices. Nothing keeps it near the identity: if each $H^{\mathrm{res}}$ has row sums of 1.1, the product over 60 sub-layers (a 30-layer model) has row sums $1.1^{60} \approx 304$; with 0.9, $0.9^{60} \approx 0.002$. mHC measures this with the **Amax gain**: the maximum absolute row sum of the composite map (forward signal) and the maximum absolute column sum (backward gradient). For HC in its 27B model it reports peaks of about 3,000 (Figure 3b), alongside a loss surge and gradient-norm spike near step 12k (Figure 2).

### The manifold: doubly-stochastic matrices

mHC (section 4.1, Eq. 6) restricts $H^{\mathrm{res}}$ to the **Birkhoff polytope**: non-negative matrices whose rows and columns each sum to 1. Three properties (section 4.1):

1. the spectral norm is at most 1 — the map never expands a signal;
2. a product of doubly-stochastic matrices is doubly stochastic — so the *composite* identity path keeps every property at any depth, Amax gain exactly 1;
3. it is the convex hull of permutation matrices — a mix of reorderings of the streams; and because columns sum to 1, the mean over streams is conserved.

With $n = 1$ the only doubly-stochastic "matrix" is the scalar 1: the plain residual again. mHC also makes $H^{\mathrm{pre}}$ and $H^{\mathrm{post}}$ non-negative to avoid cancellation (section 4.1).

### Getting there: Sinkhorn-Knopp

mHC computes the coefficients from the whole flattened stream (Eq. 7) and projects (Eq. 8):

$$\vec{x}' = \mathrm{RMSNorm}(\mathrm{vec}(X_l)) \in \mathbb{R}^{nC}, \qquad \tilde H^{\mathrm{res}}_l = \alpha^{\mathrm{res}}_l \cdot \mathrm{mat}(\vec{x}' \varphi^{\mathrm{res}}_l) + b^{\mathrm{res}}_l, \quad \text{(same for pre, post)}$$

$$H^{\mathrm{pre}} = \sigma(\tilde H^{\mathrm{pre}}), \qquad H^{\mathrm{post}} = 2\sigma(\tilde H^{\mathrm{post}}), \qquad H^{\mathrm{res}} = \mathrm{SinkhornKnopp}(\tilde H^{\mathrm{res}}).$$

Sinkhorn-Knopp (Eq. 9) starts from the positive matrix $M^{(0)} = \exp(\tilde H^{\mathrm{res}})$ and alternately normalises columns ($\mathcal{T}_c$) and rows ($\mathcal{T}_r$): $M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)}))$. It converges to a doubly-stochastic matrix as $t \to \infty$; mHC uses $t_{\max} = 20$ (section 4.2, Table 5) and notes the result is approximate — the backward gain "deviates slightly from 1", and the composite reaches "a maximum value of approximately 1.6" in the 27B model, against nearly 3,000 for HC (section 5.4). Every step is differentiable; mHC writes a fused kernel with a recomputing backward pass (section 4.3.1); the course lets autograd trace the 20 iterations.

Course choices where the paper gives no value (stated, and part of the code's docstring): $b^{\mathrm{res}} = 4I$ at initialisation (diagonal ≈ 0.95 after Sinkhorn for $n = 4$, so the stream starts close to the identity), $b^{\mathrm{pre}}$ = +4 on stream $k \bmod n$ and −4 elsewhere (like HC's $e_{k \bmod n}$), $b^{\mathrm{post}} = 0$, $\alpha = 0.01$ (mHC Table 5's "gating factor init"), $\varphi \sim \mathcal{N}(0, 0.02^2)$. Static biases and gating factors get no weight decay — decay would pull $H^{\mathrm{res}}$ towards 0, which is exactly the broken identity path. HC also excludes its static part from weight decay (HC section 5); DeepSeek-V4 keeps "the static biases and gating factors of mHC modules" on AdamW (V4 section 2.4, quoted in lesson 07.1).

## Worked example

### Sinkhorn on a 2 × 2 matrix

$\tilde H = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, so $M^{(0)} = \begin{pmatrix} 2.7183 & 1 \\ 1 & 1 \end{pmatrix}$.

- Columns: sums 3.7183 and 2 → $\begin{pmatrix} 0.7311 & 0.5 \\ 0.2689 & 0.5 \end{pmatrix}$. Rows: sums 1.2311 and 0.7689 → $\begin{pmatrix} 0.5938 & 0.4062 \\ 0.3498 & 0.6502 \end{pmatrix}$, column sums now 0.9436 and 1.0564.
- Iteration 2 ends at $\begin{pmatrix} 0.6208 & 0.3792 \\ 0.3759 & 0.6241 \end{pmatrix}$ (column sums 0.9966, 1.0034); iteration 3 at column sums 0.9998, 1.0002.

The limit is checkable by hand: scaling rows and columns leaves the cross-ratio $M_{11}M_{22}/(M_{12}M_{21}) = e$ unchanged, and a symmetric doubly-stochastic $2 \times 2$ matrix is $\begin{pmatrix} a & 1-a \\ 1-a & a \end{pmatrix}$, so $a^2/(1-a)^2 = e$, $a = \sqrt e/(1 + \sqrt e) = 0.62246$ — what 20 iterations give to six digits. Rows sum to 1 exactly after every iteration (the last operation is the row normalisation); the column error shrinks geometrically.

### Two layers, unconstrained vs constrained, $n = 2$

Unconstrained $H^{\mathrm{res}} = \begin{pmatrix} 1.0 & 0.1 \\ 0.1 & 1.0 \end{pmatrix}$: row sums 1.1. Two sub-layers: $H^2 = \begin{pmatrix} 1.01 & 0.20 \\ 0.20 & 1.01 \end{pmatrix}$, row sums 1.21; 24 sub-layers (Baseline-0's 12 blocks): $1.1^{24} = 9.85$; 60 sub-layers: 304. A stream vector $(1, 1)$ at layer $l$ arrives at layer $L$ multiplied by 9.85 — and the gradient flowing back is multiplied by the column sums, the same 9.85.

Doubly stochastic $H^{\mathrm{res}} = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{pmatrix}$: $H^2 = \begin{pmatrix} 0.82 & 0.18 \\ 0.18 & 0.82 \end{pmatrix}$, row and column sums 1 at every depth. $(1, 1)$ arrives as $(1, 1)$; $(1, 0)$ arrives as a convex mix whose mean is still 0.5 — the streams exchange information without growing or shrinking.

### What it costs Baseline-0 ($C = 768$, $n = 4$, 24 sub-layers)

- **Parameters.** mHC's fused projection $\varphi$ is $nC \times (n^2 + 2n) = 3{,}072 \times 24 = 73{,}728$ per sub-layer, plus an RMSNorm over $nC$ (3,072), 3 gating factors and $4 + 4 + 16$ biases: 76,827; × 24 = 1,843,848 (+1.9% of Baseline-0's 96.75M non-embedding). HC: 5,402 per sub-layer, 129,648 in all.
- **FLOPs.** Per token and sub-layer, beyond $\mathcal{F}$: the projection $2 \cdot nC(n^2 + 2n) = 147{,}456$, the read $2nC$, the mix $2n^2C = 24{,}576$, the write $2nC$, norm and Sinkhorn (about $20 \cdot 4n^2 = 1{,}280$). Training per token rises from 788.1M to 802.2M FLOPs: **+1.8%** (HC: +0.8%).
- **Memory traffic.** mHC Table 2, per token and sub-layer, forward, excluding $\mathcal{F}$: the plain residual reads $2C$ and writes $C$ = 2,304 elements; the 4-stream residual reads $(5n + 1)C + n^2 + 2n = 16{,}152$ and writes $(3n + 1)C + n^2 + 2n = 10{,}008$: 26,160 elements, **11.4×**. In BF16 that is 52 KB per token per sub-layer instead of 4.6 KB. Activation memory for backward grows by about $n$ for the residual too, which is why mHC recomputes the cheap mHC kernels in blocks of $L_r^* \approx \sqrt{nL/(n+2)}$ layers (section 4.3.2, Eq. 20).

FLOPs say +1.8%; bytes say ×11 on the residual. Where the residual's traffic is a visible share of step time — small models, short sequences, unfused code — the wall-clock overhead is far above the FLOP overhead. That is why this lesson's cost axis is **wall-clock**.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| residual stream $X$ | (B, T, n, C) | bf16 under autocast, fp32 on CPU | GPU (main path) / CPU |
| flattened, normalised stream $\vec{x}'$ | (B, T, nC) | same (RMSNorm computes in fp32) | same |
| fused coefficients $\vec{x}'\varphi$ | (B, T, n² + 2n) | same | same |
| $H^{\mathrm{pre}}$, $H^{\mathrm{post}}$ | (B, T, n) | same | same |
| $H^{\mathrm{res}}$ after Sinkhorn | (B, T, n, n) | same; mHC uses mixed-precision kernels (section 4.3.1) | same |
| sub-layer input / output | (B, T, C) | same | same |

| Model | Params added (mHC / HC) | Training FLOPs added (mHC / HC) | Residual traffic per sub-layer |
|---|---|---|---|
| CPU toy ($C = 128$, 8 sub-layers) | 102,616 / 7,376 | +7.0% / +2.9% | 4,400 vs 384 elements (11.5×) |
| Baseline-0 ($C = 768$, 24 sub-layers) | 1.84M / 0.13M | +1.8% / +0.8% | 26,160 vs 2,304 elements (11.4×) |

## Build it

`labs/common/frontierlab/blocks/hyperconn.py`: `sinkhorn_knopp`, `HyperConnection` (Zhu et al.'s dynamic HC with the paper's initialisation), `ManifoldHC` (mHC, Eqs. 7–9), `amax_gain` (mHC section 3.1), `residual_io_elements` (Table 2) and `hc_flops_per_token_sublayer`. `BlockLM` wraps each attention and FFN sub-layer of Baseline-0 with one of them:

```python
from frontierlab.blocks import BlockLM, with_blocks
model = BlockLM(with_blocks(toy(vocab_size=8192), residual="mhc", streams=4, sinkhorn_iters=20))
model.track_hyper = True
out = model(idx)
out.extras["hyper"]          # {"composite_fwd": 1.0000, "composite_bwd": 1.0001, ...}
```

```bash
python -m frontierlab.blocks.train --residual mhc --streams 4 --blocks-log --hyper-every 20 \
    --run runs/m06/try-mhc --preset toy --steps 200 --batch 16 --seq 128 --lr 1.5e-3
```

Correctness checks in `labs/common/tests/test_blocks.py`, all passing: Sinkhorn output doubly stochastic to $10^{-3}$ after 20 iterations for $\mathcal{N}(0, 1)$ logits and to $10^{-12}$ after 200, and exactly 1 for $n = 1$; float64 gradcheck of Sinkhorn and of both wrappers with respect to the stream and every parameter; **HC with $n = 1$ equals the plain residual exactly** (bit-identical sub-layer output and a whole BlockLM equal to Baseline-0 to $10^{-12}$ in float64); mHC with $n = 1$ equals Baseline-0 once its gating factors are zero (the read scale $\sigma(b^{\mathrm{pre}})$ is removed by the sub-layer's own pre-norm — exactly, with RMSNorm $\epsilon = 0$); causal check and cached-decode agreement (1 and 5 tokens per step) for HC and mHC models; the composite gain of a freshly built mHC model is 1 to $10^{-4}$.

## What the evidence says

- **HC — PROMISING.** Zhu et al. (ByteDance) report gains on OLMo-1B/7B dense and OLMoE (section 5; for OLMoE "a reduction in training loss of approximately 0.027"). PUBLICLY DOCUMENTED; replicated as a baseline in mHC.
- **mHC — PROMISING, adopted by its authors.** mHC (DeepSeek): 3B/9B/27B MoE models with MLA, trained on data proportional to size, $n = 4$ (section 5.1); at 27B a final loss reduction of 0.021 against the baseline and stable gradient norms where HC surged (section 5.2, Figure 5); benchmark gains in Table 4; the advantage "robustly maintained" from 3B to 27B "showing only marginal attenuation" (section 5.3). 6.7% time overhead with fused TileLang kernels, recomputation and DualPipe overlap (section 4.3, company claim for an in-house system). DeepSeek-V4 adopts it. No independent replication that we found; all evidence is from the authors.
- **The stability claim is about scale.** The HC failure mHC shows is a 27B run at step ~12k. Whether a 10–70M model in a few thousand steps shows any HC instability at all is not known; the course treats it as a hypothesis and runs it on a scale ladder (plan section 12.1). **Provided pilot traces** (the 10M/30M/70M ladder at standard and raised learning rates) **will be added after the Colab pilot**; until then the free variants run their own small arms.
- **Open questions.** How much of mHC's gain is the extra stream width versus the constraint; whether fewer Sinkhorn iterations suffice; what $n$ to use with other changes (Engram uses mHC as its backbone, lesson 06.4).

## Lab

**Folder:** [`labs/module-06/lesson-02/`](../../labs/module-06/) · **Time:** about 90 minutes (about 45 of them unattended) · **Pass check:** `pytest labs/module-06/lesson-02` passes; your notes contain the filled contract, the stability table at all three learning rates with the H1/H2 verdicts, the seed-noise comparison at the standard rate, and the measured step-time ratio with its interval next to the FLOP ratio.

### Experiment contract

- **Question:** does constraining the residual mixing (mHC) prevent instabilities that unconstrained HC shows, at this scale and at raised learning rates — and what does the n-stream residual cost per step? Decision informed: whether Lineage-F carries an n-stream residual, and which kind.
- **Hypotheses and status:** H1: HC shows instability (non-finite loss, more loss spikes than Baseline-0, or composite Amax gain above 2) at a raised learning rate — **reported at 27B, may not appear at this scale**. H2: mHC's composite gains stay within [1, 1.1] forward and [1, 1.6] backward at every learning rate — expected by construction (doubly stochastic), reported in mHC Figure 7. H3: the step-time overhead of mHC is far above its +1.8% / +7% FLOP overhead — expected from Table 2.
- **The design is informative either way.** If H1 appears, the lab shows the failure and the fix. If it does not, the composite-gain trace shows *how far* HC's identity path drifted (is it growing with the learning rate? with training?), which is what a scale ladder extrapolates; and the quality and cost numbers still decide H3 and the adoption question. Either outcome is reported as observed, with its limits.
- **Baseline:** Baseline-0 at the same preset, data, seeds and schedule; no tuning for any arm.
- **Changed variable:** the residual (plain → HC or mHC, $n = 4$, $t_{\max} = 20$), and in the stress arms the learning rate (1.5e-3, 1e-2, 3e-2), applied identically to all three residuals. **Controlled:** Data-v0, tokens, data order, initial weights of every shared module, Eval v0 windows, software.
- **Comparison axes:** stability and quality at **equal tokens** (stability is a per-step property: the same steps at the same learning rate); cost on **equal wall-clock** per step, measured with interleaved rounds on one machine. The adoption decision uses quality at equal wall-clock: an mHC gain must exceed what Baseline-0 would gain from the extra time. FLOPs are reported but do not decide — they understate the cost.
- **Budget:** free CPU about 45 minutes of training (9 runs) and 2 minutes of timing (measured below); main path PROJECTED below.
- **Metrics and decision rule:** held-out loss (256 windows, paired); max gradient norm; loss spikes (`frontierlab.optim.stability.detect_spikes`, window 5, $k = 6$); composite Amax gains every 20 steps; step-time ratio with 95% interval (`frontierlab.perf`). Rules, stated now: H1 OBSERVED / NOT OBSERVED per raised rate as defined above; H2 HOLDS if mHC's maximum gains stay inside the bounds; adopt mHC for Lineage-F only if its held-out gain over Baseline-0 at the standard rate is outside the 2-seed noise *and* larger than Baseline-0's gain from training for the measured extra time (check with the equal-wall-clock arm of the project) — otherwise do not adopt at this scale, and carry the ladder question to the pilot.
- **Correctness checks:** `pytest labs/common/tests/test_blocks.py -k "sinkhorn or hc or hyper or mhc or causal"` and `pytest labs/module-06/lesson-02` pass first.
- **Fallback evidence:** mHC Figures 2, 3, 5 and 7 (27B), labelled as the paper's results; the course's pilot traces when published, labelled as analysis of provided traces.
- **Limits:** one tiny model, 200 steps, one seed at raised rates, unfused CPU code (its timings say nothing about fused GPU kernels), the course's initialisation choices.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100. Not run in this build; part of the Module 6 pilot (plan 12.1: HC vs mHC at 10M/30M/70M, standard and raised LR) | the ladder: `train_arms.py --variant t4 --device cuda` (pilot-10m), `--variant main` (pilot-30m), `--variant main70` (pilot-70m); `compare.py --variant main --device cuda`; `step_time.py --device cuda --preset pilot-30m --batch 16 --seq 1024 --vocab 32768 --bf16`. PROJECTED: 12 runs per rung; pilot-30m $\approx 12 \cdot 3.25 \times 10^8 \cdot 2.62 \times 10^8 = 1.0 \times 10^{18}$ FLOPs, 1.2 GPU-hours at an assumed 25% MFU, times the measured HC/mHC wall-clock factor (unfused: expect 1.5–3×) ≈ 2–3.5 GPU-hours; pilot-70m about twice that |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `train_arms.py --variant t4` (pilot-10m, 2,000 steps × 32 × 512), `--max-minutes 80`, rerun to resume |
| Free CPU | laptop; measured below | every step as written |

### Steps

1. **Implement** `sinkhorn`, `mhc_mix`, `composite_gain` and `residual_io` in `lab.py`; run `pytest labs/module-06/lesson-02`. One test feeds logits of 800: what happens to `exp` without the shift, and why is the shift harmless?
2. **Predict** before training: write down the composite forward gain you expect for HC at 200 steps and lr 3e-2, and for mHC.
3. **Train** (unattended, resumable): `python labs/module-06/lesson-02/train_arms.py`.
4. **Compare:** `python labs/module-06/lesson-02/compare.py`. Record the table and the rules' verdicts.
5. **Time:** `python labs/module-06/lesson-02/step_time.py`. Put the time ratios next to the FLOP ratios and the 11.4× traffic ratio and explain the gap in two sentences.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, with another build job sharing the CPU, 2026-10-04):

| lr | Arm | Held-out (seed 0) | Max grad norm | Loss spikes | Composite fwd gain, first → last logged | Max fwd / bwd gain |
|---|---|---|---|---|---|---|
| 1.5e-3 | b0 | 6.6857 | 1.11 | 0 | – | – |
| 1.5e-3 | HC | 6.6715 | 1.11 | 0 | 1.12 → 2.20 | 2.32 / 1.96 |
| 1.5e-3 | mHC | 6.6642 | 1.11 | 0 | 1.000 → 1.000 | 1.000 / 1.004 |
| 1e-2 | b0 | 6.6775 | 1.01 | 0 | – | – |
| 1e-2 | HC | 6.5566 | 2.02 | 0 | 1.25 → 4.30 | 4.30 / 3.22 |
| 1e-2 | mHC | 6.5075 | 1.12 | 0 | 1.000 → 1.000 | 1.000 / 1.031 |
| 3e-2 | b0 | 6.7834 | 0.64 | 0 | – | – |
| 3e-2 | HC | 6.4891 | 3.93 | 0 | 1.49 → 6.59 | 6.59 / 5.20 |
| 3e-2 | mHC | 6.4628 | 0.84 | 0 | 1.000 → 1.000 | 1.000 / 1.050 |

- Standard rate, seeds 0 / 1: b0 6.6857 / 6.6762, HC 6.6715 / 6.6755, mHC 6.6642 / 6.6719; b0 seed std 0.0067, MDE 0.019. Paired, seed-averaged: HC − b0 = −0.0075 [−0.0106, −0.0044]; mHC − b0 = −0.0130 [−0.0156, −0.0103]; mHC − HC = −0.0055 [−0.0082, −0.0027]. All below the MDE.
- Rules: **H1 OBSERVED at 1e-2 and 3e-2, by the gain criterion only** — HC's composite forward gain grew to 4.3 and 6.6 and its gradient norm peaked at 2.0 and 3.9 (b0: 1.0 and 0.6), but there were no loss spikes and no non-finite losses. **H2 HOLDS** (mHC's maximum gains 1.000 forward, ≤ 1.05 backward). At the raised rates mHC beats HC by 0.049 [0.043, 0.055] and 0.026 [0.020, 0.033] nats (one seed).
- Wall-clock (`step_time.py`, toy, 16 × 128, 15 interleaved rounds): b0 479 ms per step; HC **1.54× [1.26, 1.81]**; mHC **2.65× [1.99, 2.99]** — against FLOP ratios of 1.029 and 1.071 and an 11.5× residual-traffic ratio.
- Runtimes: 9 new training runs 30 minutes (mHC 4.6 minutes each, b0 1.5–2.3); `compare.py` 83 s; `step_time.py` 59 s.

What to write about it. The mechanism mHC targets is visible at 1.8M parameters and 200 steps: HC's identity path drifts — the composite gain doubles at the standard rate and reaches 6.6 at the highest — while mHC's stays at exactly 1. What is *not* visible is the consequence the paper reports at 27B, a loss surge: here the drift has not yet hurt training, and every residual design gained from the raised rates in so short a run (b0 too, until 3e-2). So the honest result is "drift observed and growing with the learning rate; instability in the loss not observed at this scale" — the ladder (and the provided pilot traces, once added) says whether the drift turns into a failure as models grow. The quality differences at the standard rate are below the noise floor; the cost difference is not: unfused, mHC makes each step 2.65× slower for a 7% FLOP increase, so at equal wall-clock it would have to beat a Baseline-0 trained 2.6× longer.

<details>
<summary>Hint for TODO 2</summary>

Use `torch.einsum`: the read is `"btj,btjc->btc"` (weights over streams), the mix `"btij,btjc->btic"` (output stream $i$ from all $j$). The write adds the same sub-layer output to every stream with its own weight: broadcast `post.unsqueeze(-1) * y.unsqueeze(-2)`.

</details>

<details>
<summary>Hint for TODO 3</summary>

Multiply newest-on-the-left: `comp = m if comp is None else m @ comp`. Row sums are over the last axis (`sum(-1)`), column sums over the one before it (`sum(-2)`); take `abs()` before summing, then the maximum over the $n$ rows (or columns), then the mean over batch and positions.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-02`.

</details>

## Common mistakes

- **Weight decay on the static mappings.** Decay pulls $b^{\mathrm{res}}$ and HC's $A_r$ towards 0 — towards *no* identity path. Exclude static biases and gating factors from decay (the wrapper does).
- **Normalising rows last and declaring the matrix doubly stochastic.** After a finite number of iterations rows sum to 1 exactly and columns only approximately; test the column error, and know that 20 iterations is an approximation (mHC section 5.4).
- **Calling mHC "free" because FLOPs grow by 2%.** The residual's memory traffic grows about 11× at $n = 4$; measure time.
- **Comparing stability at one learning rate.** Instability is a property of the learning-rate range; a single standard-rate run that is stable for both says nothing about H1.
- **Reading a null H1 at 10M parameters as "HC is stable".** It is a statement about this scale and training length; the trend along a scale ladder is the informative part.
- **Breaking symmetry by accident — or not at all.** If every stream starts identical, reads the same and receives the same write, the streams stay identical and HC is a scaled plain residual. HC's $e_{k \bmod n}$ read (and the course's matching mHC bias) is what makes the streams differ.

## References

- Z. Xie et al. (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections*, sections 3–5, Tables 1, 2, 4, 5, Figures 2, 3, 5, 7. https://arxiv.org/abs/2512.24880
- D. Zhu et al., *Hyper-Connections*, sections 2.1–2.3 and 5. https://arxiv.org/abs/2409.19606
- DeepSeek-AI, *DeepSeek-V4*, sections 2.1–2.4. https://arxiv.org/abs/2606.19348
- DeepSeek-AI, *Conditional Memory via Scalable Lookup (Engram)*, section 2.4. https://arxiv.org/abs/2601.07372
- Shared code: `labs/common/frontierlab/blocks/hyperconn.py`, `model.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[06.3 · Combining changes](lesson-03.md)
