---
id: "08.3"
module: 8
minutes: 45
practice_minutes: 100
prerequisites: ["08.2"]
objectives:
  - State the four ingredients of NVIDIA's NVFP4 pretraining recipe (2-D 16 × 16 weight scaling, 1 × 16 activation and gradient scaling, random Hadamard transforms on weight-gradient inputs, stochastic rounding of gradients) plus its high-precision layers, and say which failure each one addresses.
  - Show by hand that stochastic rounding is unbiased where round-to-nearest is not, and that a random Hadamard transform leaves the weight gradient unchanged in exact arithmetic while changing what the quantiser sees.
  - Run an ablation of the NVFP4 recipe under an experiment contract on emulated FP4, and report each ingredient's effect against the noise floor without overstating a course-scale result.
  - Distinguish FP4 pretraining from 4-bit weights obtained by post-training quantisation or quantisation-aware training, using DeepSeek-V4, gpt-oss and Kimi K2 Thinking as documented, and measure PTQ against QAT at equal tokens.
volatility: implementation
sources:
  - title: "NVIDIA — Pretraining Large Language Models with NVFP4 (sections 3–5, appendices A–B, E)"
    url: https://arxiv.org/abs/2509.25149
  - title: "NVIDIA Transformer Engine — NVFP4 training (release 2.18)"
    url: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/features/low_precision_training/nvfp4/nvfp4.html
  - title: "DeepSeek-AI — DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (section 5.2.1: FP4 quantization-aware training)"
    url: https://arxiv.org/abs/2606.19348
  - title: "OpenAI — gpt-oss-120b & gpt-oss-20b Model Card (section 2.1)"
    url: https://arxiv.org/abs/2508.10925
  - title: "Moonshot AI — Kimi-K2-Thinking model card and config.json"
    url: https://huggingface.co/moonshotai/Kimi-K2-Thinking
  - title: "NVIDIA DGX B200 product page (FP4 and FP8 tensor-core peaks, HBM bandwidth)"
    url: https://www.nvidia.com/en-us/data-center/dgx-b200/
last_verified: "2026-10-04"
---

# 08.3 · FP4 training and quantisation-aware training

Four bits leave a format with seven magnitudes and less than four binades of range, so every 16 or 32 values need their own scale and every rounding choice shows up in the loss. This lesson studies the two ways frontier labs have used 4-bit numbers: training in FP4 from the start (NVIDIA's NVFP4 recipe for a 12B model on 10T tokens) and teaching an already trained model to live with 4-bit weights (quantisation-aware training in DeepSeek-V4 and Kimi K2 Thinking; MXFP4 weights during gpt-oss post-training). You will build each ingredient of the NVFP4 recipe, ablate them on emulated FP4, and compare post-training quantisation with quantisation-aware training at equal tokens. The main path needs a Blackwell GPU, which Colab does not offer, so its figures here are projections and published numbers, labelled as such.

## Why this matters at a frontier lab

On Blackwell, FP4 tensor cores have twice the FP8 peak and four times BF16 (DGX B200, 8 GPUs: 72 dense FP4 PFLOP/s, 36 dense FP8; company claim), and 4-bit weights are a quarter of BF16 memory. That changes what fits on one node and what serving costs. DeepSeek-V4 notes that FP4 × FP8 already has the same peak as FP8 × FP8 on current hardware and expects it to be "1/3 more efficient on future hardware" (section 5.2.1). The decisions are sharper than for FP8: train in FP4 or only deploy in FP4? Which layers stay in BF16? Is a recipe's gap of 1% in loss acceptable, and does it survive the learning-rate decay? A research engineer has to know what each ingredient buys, so that a failed FP4 run can be diagnosed rather than abandoned.

## The idea

### Why plain FP4 fails

E2M1 has the values $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$: a relative rounding error up to 25% and a range of $6/0.5 = 12$. With one scale per tensor, lesson 08.1 measured 74% of a toy model's gradient values flushing to zero. Three problems follow: small values vanish (range), the forward and backward passes can see different weights (consistency), and round-to-nearest errors in gradients do not average out (bias).

### NVIDIA's NVFP4 recipe (arXiv 2509.25149 section 4, company claim)

The paper's recommendation list, each with its reason:

1. **Scale granularity.** "Use two-dimensional (2D) scaling over 16 × 16 blocks for weights, and one-dimensional scaling over 1 × 16 blocks for activations and gradients." 1 × 16 blocks (16 values per E4M3 scale, plus one FP32 scale per tensor) limit how far one outlier reaches. The 16 × 16 weight blocks make $W$ and $W^\top$ quantise to the same values, so Fprop and Dgrad use one weight: with 1 × 16 blocks along $K$ for Fprop and along $N$ for Dgrad, the backward pass would differentiate a different function from the one the forward pass computed.
2. **Random Hadamard transforms on Wgrad inputs.** "Apply Random Hadamard transforms of size 16 × 16 to inputs of weight gradient GEMMs." For an orthogonal $Q$, $(\partial L/\partial y)^\top x = \big((\partial L/\partial y)^\top Q\big)\big(Q^\top x\big)$, so in exact arithmetic nothing changes; after quantisation, each block of 16 tokens is a mixture of all 16, and an outlier token no longer sets one block's scale on its own. Section 4.2 restricts it to Wgrad because "Hadamard transforms show no measurable benefit for Fprop and Dgrad at smaller scales", and uses "a single random sign vector that is shared across all linear layers throughout training".
3. **Stochastic rounding on gradients.** "Use stochastic rounding for gradients and round-to-nearest-even for weights and activations." Section 4.4: other backward tensors do not benefit, and appendix E.3 reports that stochastic rounding of activations or weights caused divergence.
4. **Keep some layers in BF16.** For the 12B model, "keeping the first two blocks in addition to the final eight blocks ... in BF16, representing 16% of the linear layers" (section 4.1; the summary recommendation says about 15%, with the high-precision layers mostly at the end).

Results (section 3, Table 2; company claim): a 12B hybrid Mamba-Transformer trained on 10T tokens against an FP8 baseline (not BF16); the relative loss error of NVFP4 "remains consistently below 1%, and widens to slightly above 1.5% as the learning rate is decayed". Downstream, FP8 vs NVFP4: MMLU-Pro 62.62 vs 62.58, MMLU 77.36 vs 76.57, GSM8k 89.08 vs 92.27, MATH 83.32 vs 81.48, HumanEval+ 59.93 vs 57.43, MBPP+ 59.11 vs 55.91. On an 8B model trained on 1T tokens, "MXFP4 matches NVFP4 loss when trained on 36% more tokens" (section 5). Transformer Engine exposes the recipe as `NVFP4BlockScaling`, with switches `disable_rht`, `disable_stochastic_rounding` and `disable_2d_quantization`, for training on SM 10.0 and 10.3 (B200/B300-class) only.

### 4-bit weights without 4-bit training

Three documented models used 4-bit weights without pretraining in FP4:

- **gpt-oss** (model card section 2.1, PUBLICLY DOCUMENTED): "We post-trained the models with quantization of the MoE weights to MXFP4 format, where weights are quantized to 4.25 bits per parameter." The MoE weights are over 90% of the parameters, so the 120b model fits one 80 GB GPU.
- **Kimi K2 Thinking** (model card, PUBLICLY DOCUMENTED): "Quantization-Aware Training (QAT) during the post-training phase", applying "INT4 weight-only quantization to the MoE components", for "a roughly 2x generation speed improvement" (company claim). Its `config.json` gives the scheme: symmetric INT4, groups of 32, activations unquantised; attention, shared experts, dense MLP layers and the output head are left out. How the scale is computed is not stated beyond an "observer" named `minmax`; the course uses $s = \operatorname{amax}/7$ on the grid $[-7, 7]$ and says so.
- **DeepSeek-V4** (section 5.2.1, PUBLICLY DOCUMENTED): "Quantization-Aware Training (QAT) ... during the post-training stage", applying "FP4 (MXFP4) quantization" to the MoE expert weights and to the query-key path of the sparse-attention indexer. The FP32 master weights are quantised to FP4 and dequantised back to FP8 for computation, which the report calls lossless; FP4 sub-blocks are 1 × 32 tiles inside FP8 128 × 128 tiles; gradients reach the master weights "equivalent to" a straight-through estimator, so the existing FP8 training framework is reused unchanged; rollouts and inference use native FP4 weights.

### QAT with a straight-through estimator

In QAT the forward pass uses $\hat W = q(W)$ and the master weights $W$ stay in high precision:

$$y = x \hat W^\top, \qquad \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\hat W, \qquad \frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial \hat W} = \Big(\frac{\partial L}{\partial y}\Big)^\top x .$$

The rounding function has zero derivative almost everywhere; the straight-through estimator (STE) treats it as the identity in the backward pass. The weights then drift toward places where rounding costs less loss, which PTQ — rounding once after training — cannot do. At export, $\hat W$ is stored and is exactly what training used.

## Worked example

### Stochastic rounding is unbiased

A gradient value, already scaled, is $1.3$. Its E2M1 neighbours are 1.0 and 1.5. Round to nearest gives 1.5 every time: a bias of $+0.2$, the same sign on every step that sees this value. Stochastic rounding gives 1.5 with probability $(1.3 - 1.0)/(1.5 - 1.0) = 0.6$ and 1.0 with probability 0.4; the mean is $0.6 \cdot 1.5 + 0.4 \cdot 1.0 = 1.3$, the variance $0.6 \cdot 0.2^2 + 0.4 \cdot 0.3^2 = 0.06$. Summed over 100 steps, RNE is off by 20 with certainty; SR is off by about $\sqrt{100 \cdot 0.06} = 2.4$ in standard deviation, centred on zero. SGD averages noise; it does not average bias. That is why the recipe uses SR for gradients — and only for gradients, because in the forward pass the noise itself perturbs the function being trained.

### A Hadamard transform, by hand

Take blocks of 4 (size 16 in the recipe) with $H_4 = \tfrac{1}{2}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix}$, an orthogonal matrix. One channel of activations over 4 tokens is $x = (8, 0.3, 0.3, 0.3)$ — one outlier token — and the matching output-gradient channel is $g = (0.1, 0.2, -0.3, 0.4)$. The contribution to $\partial L/\partial W$ is $g \cdot x = 0.8 + 0.06 - 0.09 + 0.12 = 0.89$. After the transform, $x H_4 = (4.45, 3.85, 3.85, 3.85)$ and $g H_4 = (0.2, -0.4, 0.1, 0.3)$; their dot product is $0.89 + (-1.54) + 0.385 + 1.155 = 0.89$. Same answer. What changed is what a 1 × 4 E2M1 block sees: before, $s = 8/6$ and $0.3/s = 0.225$ flushes three of four values to zero; after, all four values are within a factor 1.2 of each other and none flushes. Whether the *product* becomes more accurate depends on the data — the lab measures it on real gradients, and finds that at toy scale it reduces underflow but not the Frobenius error of $\partial L/\partial W$.

### FP4 to FP8 without loss

DeepSeek-V4 dequantises MXFP4 expert weights to FP8 for its existing kernels and calls this lossless. Check when it is: an FP4 value is $q \cdot 2^{j}$ relative to the FP8 tile's scale, with $q \in \{0.5, 1, 1.5, 2, 3, 4, 6\}$ (at most 2 significant bits) and $2^j$ the ratio of the 1 × 32 sub-block's power-of-two scale to the 128 × 128 tile's scale. E4M3 has 3 mantissa bits, so precision is never the problem; range is. The largest value $6 \cdot 2^j = 1.5 \cdot 2^{j+2} \le 448 = 1.75 \cdot 2^8$ needs $j \le 6$; the smallest, $0.5 \cdot 2^j$, and $1.5 \cdot 2^j = 3 \cdot 2^{j-1}$ must be multiples of E4M3's smallest subnormal $2^{-9}$, which needs $j \ge -8$. So the conversion is exact whenever every sub-block scale is within $2^{-8}$ to $2^{6}$ of its tile's scale — 15 binades (our arithmetic, checked by enumeration in the lab code; how V4 chooses the tile scale is not stated in the sections we read).

## Shapes and cost

| Tensor (Baseline-0 linear, B200 main path) | Shape | Stored as | Scales | Device |
|---|---|---|---|---|
| activation $x$ (Fprop) | (16,384, 768) | E2M1, 2 per byte: 6.3 MB | (16,384, 48) E4M3 + 1 fp32 | GPU |
| $x^\top$ after RHT (Wgrad) | (768, 16,384) | E2M1 | (768, 1,024) E4M3 | GPU |
| weight $W$ | (2,816, 768) | E2M1, 1.08 MB | 16 × 16: (176, 48) E4M3 | GPU |
| $\partial L/\partial y$ (stochastically rounded) | (16,384, 2,816) | E2M1 | 1 × 16 | GPU |
| RHT matrix | (16, 16) | ±1/4, one sign vector for all layers | — | GPU |
| QAT master weights / fake-quantised copy | (2,816, 768) | fp32 / bf16 holding INT4 values | per 32: (2,816, 24) | GPU |
| emulation in the course | same | float32 holding FP4 values | float32 | CPU or GPU |

**Throughput on B200, PROJECTED** with `frontierlab.precision.cost.projected_speedup` (per-GPU dense peaks from NVIDIA's DGX B200 page divided by 8: FP4 9, FP8 4.5, BF16 2.25 PFLOP/s; HBM 64 TB/s / 8 = 8 TB/s; same model as lesson 08.2): for Baseline-0 at 16 × 1,024 tokens, the block linears alone become 2.71× faster, the whole step 1.08× with unfused casts and 1.41× with fused casts, against an Amdahl limit of 1.53× (the linears are 46% of the BF16 step). The RHT adds a 16 × 16 matmul per block of each Wgrad input: $2 \cdot 16$ FLOPs per element, negligible next to the GEMM but one more memory pass unless fused. Stochastic rounding needs one random number per gradient element.

**Memory.** At 4.5 bits per value, NVFP4 weights are 28% of BF16 (4.5/16); MXFP4 at 4.25 bits 27%; INT4 with an fp32 scale per 32 (the course's emulation) 5 bits, 31% — with BF16 scales, as deployments store them, 4.5 bits. gpt-oss-120b's checkpoint is 60.8 GiB for 116.83B parameters (Table 1), about 4.47 bits per parameter overall because the non-MoE weights stay in BF16.

## Build it

`frontierlab.precision` already has the pieces; this lesson adds no new mechanism, only recipes:

- `RECIPES["nvfp4"]`: `act = QuantSpec("e2m1", (1, 16), "nvfp4")`, `weight = QuantSpec("e2m1", (16, 16), "nvfp4")`, `grad` the same as `act` with `rounding="sr"`, `rht=16`; ablations `nvfp4-no-sr`, `nvfp4-no-rht`, `nvfp4-1d-weights`; `mxfp4` is the naive baseline (1 × 32, E8M0, round to nearest, no RHT).
- `hadamard.apply_rht(x, 16, seed)` multiplies blocks of 16 along the last dimension by $H_{16}\operatorname{diag}(s)$; the sign vector comes from its own generator (seed 0), shared by every layer and step, so it never touches the training RNG.
- Stochastic rounding draws from the global torch RNG, which the loop checkpoints; stop-and-resume with `--recipe nvfp4` is bit-identical (test in `test_precision.py`).
- `--keep-high first2,last8` leaves those blocks' linears in high precision; `--only mlp` quantises only the SwiGLU matrices (the dense stand-in for expert weights).
- QAT: `--recipe int4-qat` or `mxfp4-qat` (weight-only fake quantisation; Dgrad through $\hat W$; Wgrad by the STE); `qat.export_(model)` replaces each layer by an `nn.Linear` holding $\hat W$; `linear.ptq_(model, spec)` rounds weights once.

Correctness checks (all pass): the Hadamard matrix is orthogonal and the RHT leaves $\partial L/\partial W$ unchanged to $10^{-10}$ in float64; 16 × 16 blocks give identical $q(W)$ and $q(W^\top)^\top$, 1 × 16 blocks do not; SR lands only on the two neighbours and is unbiased to 0.005; averaged over 300 draws the SR weight gradient's error is less than half of RNE's; the nvfp4 layer's backward uses the global RNG (same seed, same gradient); an exported QAT model equals the fake-quantised forward, and PTQ of the starting weights equals it too.

## What the evidence says

- **NVFP4 pretraining — PROMISING.** One lab's evidence (NVIDIA: one 12B run on 10T tokens against FP8, plus 8B ablations; company claim), with a widening gap during learning-rate decay that the paper reports itself. The individual ingredients have reasons and ablations in that paper; independent replications at scale were not found.
- **Random Hadamard transforms — PROMISING.** Benefit claimed for Wgrad at scale and explicitly not found for Fprop/Dgrad at smaller scales (section 4.2); our toy measurement agrees that its effect at small scale is not a lower GEMM error.
- **Stochastic rounding of gradients — ESTABLISHED as a principle** (unbiasedness) **and PROMISING in this recipe.**
- **QAT for 4-bit weights — ESTABLISHED as a technique;** its use in Kimi K2 Thinking (INT4, MoE weights, post-training) and DeepSeek-V4 (MXFP4, expert weights and indexer QK path, post-training) is MODEL-SPECIFIC, PUBLICLY DOCUMENTED, without published ablations against PTQ in those reports. Kimi's 2× generation speed is a company claim.
- **MXFP4 weights in gpt-oss — MODEL-SPECIFIC, PUBLICLY DOCUMENTED.** The card says the models were post-trained with MXFP4 MoE weights; it does not give a PTQ-vs-QAT comparison.
- **Not verified:** the exact INT4 scale rule of Kimi K2 Thinking; how DeepSeek-V4 chooses its FP8 tile scale; any NVFP4 speed figure on a real training step (the course has not run one).

## Lab

**Folder:** [`labs/module-08/lesson-03/`](../../labs/module-08/) · **Time:** about 100 minutes attended, about 45 minutes unattended on the free CPU path · **Pass check:** `pytest labs/module-08/lesson-03` passes; your notes contain the Wgrad table (step 2), the ablation table with the contract's verdict per ingredient (step 3), and the PTQ-vs-QAT comparison with intervals (step 4).

### Experiment contract (ablation, step 3)

- **Question:** at course scale and equal tokens, which ingredients of the NVFP4 recipe measurably change held-out loss of emulated FP4 training relative to BF16? Decision informed: which ingredients a Recipe-R FP4 trial must keep, and whether course-scale FP4 evidence can inform that at all.
- **Hypotheses and status:** (H1) naive MXFP4 has a larger gap than NVFP4 — reported (section 5: 36% more tokens to match at 8B). (H2) removing SR or RHT increases the gap — reported at 12B, may not appear at this scale. (H3) keeping the last block in BF16 reduces the gap — reported (section 4.1).
- **Baseline:** BF16 (fp32 on the CPU) through the same wrapper, AdamW at the loop's defaults; the noise floor from BF16 seeds 0 and 1, and the NVFP4 arm also at seeds 0 and 1.
- **Changed variable:** one recipe ingredient per arm. **Controlled:** seed 0 (initial weights and data order identical across arms), tokens (150 × 2,048), schedule, data, 256 evaluation windows, software and machine.
- **Comparison axis:** equal tokens. It says nothing about speed (emulation) or about FP4 at 10T tokens.
- **Budget:** free CPU about 33 minutes of training (measured below); main path B200 NOT PILOTED.
- **Metrics and decision rule:** held-out loss on 256 fixed windows of 128 tokens, evaluated in the arm's training precision; gap to BF16 (seed 0) paired by window with a 95% bootstrap interval. Rule: an ingredient "matters at this scale" if removing it widens the gap to BF16 by more than both the NVFP4 arm's seed-to-seed difference and the width of the paired interval; "no detectable effect" otherwise. A null result is a result.
- **Correctness checks:** `pytest labs/common/tests/test_precision.py` (RHT exactness, 2-D block consistency, SR unbiasedness, exact resume with SR) and the lab tests pass.
- **Fallback evidence:** the NVFP4 paper's ablation figures (sections 4.1–4.4, appendix E), labelled as analysis of published results.
- **Limits:** a 4-block, 1.8M-parameter model for 0.31M tokens; the recipe's ingredients are reported to matter over trillions of tokens and at 12B; one seed for the ablation arms.

The QAT step (step 4) has its own short contract in the script header: question "at equal extra tokens from the same trained weights, does QAT give a lower held-out loss after export than BF16 training followed by PTQ, for INT4 (all linears) and MXFP4 (MLP only)?"; hypothesis QAT < PTQ (established for low bit widths, size unknown here); control = BF16 continuation + PTQ with the same scheme; metric and interval as above; rule: QAT better if the paired interval of QAT − PTQ lies below zero.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× B200 (sm100) with Transformer Engine, `NVFP4BlockScaling`; **not piloted** (Colab has no Blackwell) and not run in this build | `bench_nvfp4.py --device cuda` (GEMM throughput, BF16 vs MXFP8 vs NVFP4 with real kernels; the TE API must be re-checked against the installed version), and `train_fp4.py --variant main --print` for the emulated numerics arms on pilot-70m |
| Free GPU (Colab/Kaggle T4) | T4, emulation (no FP4 hardware on any free GPU) | `train_fp4.py --variant t4`, then `qat_ptq.py` as written |
| Free CPU | laptop; measured below | steps 1–4 as written |

### Steps

1. **Implement** `sr_round_e2m1`, `hadamard16`, `rht_blocks`, `int4_group_qdq` and `qat_backward` in `lab.py`; run `pytest labs/module-08/lesson-03`.
2. **Wgrad on real tensors:** `python labs/module-08/lesson-03/wgrad_error.py` (uses lesson 08.1's checkpoint; trains it if missing).
3. **Ablation:** `python labs/module-08/lesson-03/train_fp4.py`, then `python labs/module-08/lesson-03/compare_fp4.py`.
4. **PTQ vs QAT:** `python labs/module-08/lesson-03/qat_ptq.py`.
5. **Main path (B200 only):** `python labs/module-08/lesson-03/bench_nvfp4.py` prints the PROJECTED bounds anywhere; with a B200, add `--device cuda`.

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04):

**Step 2, `wgrad_error.py`** (5.4 minutes; 28 linears of the 60-step toy checkpoint, one batch of 2,048 tokens, float64; "residual" is the error left after averaging 64 draws, which for SR still contains the fixed round-to-nearest error of $x^\top$):

| Variant | real tensors: rel. error / residual / dy underflow | one token in 64 scaled ×30: rel. error / residual / dy underflow |
|---|---|---|
| RNE | 0.117 / 0.117 / 0.165 | 0.035 / 0.035 / 0.264 |
| SR | 0.138 / 0.086 / 0.165 | 0.060 / 0.031 / 0.264 |
| RHT + RNE | 0.110 / 0.110 / 0.069 | 0.067 / 0.067 / 0.056 |
| RHT + SR (the NVFP4 recipe) | 0.138 / 0.075 / 0.069 | 0.066 / 0.055 / 0.056 |

SR trades a larger single-draw error for a smaller error on average — the unbiasedness at work. The RHT cuts gradient underflow by a factor of 2.4 (4.7 with outlier tokens), but with outlier tokens it *doubles* the single-draw error of the GEMM: mixing a ×30 token into its block raises every block's scale. Lower underflow is not lower error.

**Step 3, `train_fp4.py` + `compare_fp4.py`** (training 32.5 minutes for 8 runs: BF16 1.3–2.0 min, emulated FP4 3.9–5.8 min; comparison 86 s). Toy model, 150 steps × 2,048 tokens, held-out loss on 256 windows:

| Arm | Held-out loss | Gap vs BF16 seed 0, paired [95% CI] |
|---|---|---|
| bf16 seed 0 / seed 1 | 6.7746 / 6.7652 | — / −0.0094 [−0.0190, +0.0001] |
| mxfp4 (naive) | 6.7948 | +0.0202 [+0.0140, +0.0265] |
| nvfp4 seed 0 / seed 1 | 6.7615 / 6.7672 | −0.0131 [−0.0183, −0.0078] / −0.0074 [−0.0166, +0.0018] |
| nvfp4, last block BF16 | 6.7158 | −0.0587 [−0.0641, −0.0534] |
| nvfp4 without SR | 6.7623 | −0.0123 [−0.0179, −0.0068] |
| nvfp4 without RHT | 6.7286 | −0.0460 [−0.0514, −0.0408] |

By the contract's rule (threshold = the larger of the NVFP4 seed difference, 0.0057, and the half-width of the paired interval): SR "no detectable effect" (+0.0008 [−0.0046, +0.0061]); RHT, the BF16 last block and NVFP4-vs-MXFP4 scaling "matter at this scale" (−0.033, −0.046 and +0.033 against `nvfp4-s0`).

How to read it — and a change after results. Three things say the rule was too lax. First, the directions are implausible as precision effects: NVFP4 beats BF16, and *removing* the RHT helps by 0.033 nats. Second, lesson 08.2 measured that changing only the rounding of the GEMMs, at the same seed, moves the final loss by ±0.02–0.03 nats — the trajectory diverges like a new seed even though "same-recipe, different-seed" pairs differ by only 0.006–0.009. Third, the paired intervals resample windows, not trajectories. A sounder threshold is that recipe-change spread, about 0.03 nats. Under it, RHT (0.033) and NVFP4-vs-MXFP4 (0.033) are at the edge and the BF16 last block (0.046) barely beyond; none is a reliable effect, and SR remains undetected. We record this as a change after results, as the [contract template](../../templates/experiment-contract.md) requires, rather than quietly adopting the stricter rule. What the lab does show robustly is mechanical: every FP4 recipe trains without diverging for 150 steps, the precision log reports 14% gradient relative error and 6.5% gradient underflow for NVFP4, and the course scale cannot rank the ingredients. That is the expected outcome for an effect NVIDIA reports at 12B parameters and 10T tokens.

**Step 4, `qat_ptq.py`** (3.9 minutes; base = BF16 toy at 200 steps, held-out 6.5656; control = 60 more BF16 steps, 6.4964):

| Comparison | Gap [95% CI] |
|---|---|
| base + PTQ INT4 (groups of 32, 5.0 bits incl. fp32 scales), all linears | +0.0007 [+0.0003, +0.0010] |
| base + PTQ MXFP4 (4.25 bits), MLP only / all linears | +0.0002 [−0.0002, +0.0005] / −0.0008 [−0.0013, −0.0003] |
| control + PTQ INT4 vs control | +0.0009 [+0.0005, +0.0012] |
| INT4 QAT exported vs control; QAT − PTQ | +0.0016 [+0.0008, +0.0025]; +0.0008 [+0.0000, +0.0015] |
| control + PTQ MXFP4 (MLP) vs control | +0.0003 [−0.0001, +0.0007] |
| MXFP4 QAT (MLP) exported vs control; QAT − PTQ | +0.0000 [−0.0004, +0.0005]; −0.0003 [−0.0008, +0.0003] |

By the rule, QAT is not better than PTQ here (INT4: slightly worse, at the edge of the interval; MXFP4: no difference). The reason is visible in the first rows: PTQ of a 200-step toy model costs under 0.001 nats, so there is nothing for QAT to recover. The PTQ-vs-QAT question is real for heavily trained models (lesson 08.4: the PTQ penalty grows with training tokens), which this budget cannot reach; the result answers "is QAT needed for a barely trained 1.8M-parameter model?" and nothing more.

<details>
<summary>Hint for TODO 1</summary>

`grid = E2M1_GRID`; `i = torch.searchsorted(grid, a, right=True) - 1` is the index of the largest grid value not above `a = |x|` (clamp `a` to 6 first and `i` to the last index). The upper neighbour is `grid[i + 1]` (clamped). Round up when `u < (a - lo) / (hi - lo)`; when `a` is on the grid, `a - lo` is 0 and it never rounds up.

</details>

<details>
<summary>Hint for TODO 4</summary>

The derivative of $x \hat W^\top$ with respect to $x$ uses $\hat W$, the weight the forward pass multiplied by. With respect to $W$ the STE pretends $\hat W = W$: `gy.t() @ x`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-08/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-08/lesson-03`.

</details>

## Common mistakes

- **Using 1 × 16 blocks for weights.** $q(W)$ and $q(W^\top)^\top$ then differ, and the backward pass differentiates a function the forward pass did not compute. Use 2-D blocks for any tensor used in both orientations.
- **Stochastic rounding everywhere.** SR in the forward pass adds noise to the function itself; the NVFP4 paper reports divergence with SR on activations or weights (appendix E.3). Use it where unbiasedness matters: gradients.
- **Drawing SR random numbers from an unsaved generator.** The run then cannot resume exactly. Use a checkpointed RNG and test stop-and-resume.
- **Comparing an FP4 recipe against BF16 and quoting NVIDIA's gap.** The NVFP4 paper's baseline is FP8, not BF16.
- **Calling QAT "FP4 training".** DeepSeek-V4 and Kimi K2 Thinking pretrained in higher precision and used 4-bit weights from post-training on; their results say nothing about FP4 pretraining.
- **Counting the model as 4-bit.** Scales, kept-high-precision layers and non-quantised components count: gpt-oss-120b averages about 4.5 bits per parameter overall.

## References

- NVIDIA, *Pretraining Large Language Models with NVFP4*, sections 2–5, appendices A, B and E. https://arxiv.org/abs/2509.25149
- NVIDIA, Transformer Engine 2.18, NVFP4 training. https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- DeepSeek-AI, *DeepSeek-V4*, section 5.2.1 and introduction. https://arxiv.org/abs/2606.19348
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, section 2.1 and Table 1. https://arxiv.org/abs/2508.10925
- Moonshot AI, *Kimi-K2-Thinking* model card and `config.json` (`quantization_config`). https://huggingface.co/moonshotai/Kimi-K2-Thinking
- NVIDIA, DGX B200 product page. https://www.nvidia.com/en-us/data-center/dgx-b200/
- Shared code: `labs/common/frontierlab/precision/linear.py`, `hadamard.py`, `qat.py`, `cost.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[08.4 · Scaling laws for precision](lesson-04.md) (extension), or the [Module 8 project](../../projects/module-08-precision-plan.md).
