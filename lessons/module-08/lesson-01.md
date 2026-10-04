---
id: "08.1"
module: 8
minutes: 40
practice_minutes: 70
prerequisites: ["02.1", "01.4"]
objectives:
  - Decode and encode FP8 (E4M3, E5M2), FP4 (E2M1) and E8M0 values from their bit layouts, and state each format's largest value, smallest normal and subnormal values and relative rounding error.
  - Quantise a tensor by hand with per-tensor, per-row, per-tile and per-block scales, and with OCP MX (E8M0, 32 elements) and NVFP4 (E4M3 per 16 elements plus an FP32 tensor scale) scaling.
  - Implement exact round-to-nearest-even emulation of these formats in PyTorch and test it bit for bit against torch.float8_e4m3fn and torch.float8_e5m2 casts.
  - Measure relative error, underflow and saturation of real activations, weights and gradients under each format and granularity, and explain which failure each scaling choice prevents.
volatility: concept
sources:
  - title: "Micikevicius et al. — FP8 Formats for Deep Learning (Table 1: E4M3 and E5M2)"
    url: https://arxiv.org/abs/2209.05433
  - title: "Open Compute Project — OCP Microscaling Formats (MX) v1.0 Specification (Table 1, sections 5.3–5.4, 6.3)"
    url: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - title: "Rouhani et al. — Microscaling Data Formats for Deep Learning (Algorithm 1)"
    url: https://arxiv.org/abs/2310.10537
  - title: "NVIDIA — Pretraining Large Language Models with NVFP4 (section 2, appendix B)"
    url: https://arxiv.org/abs/2509.25149
  - title: "NVIDIA Technical Blog — Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"
    url: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
  - title: "DeepSeek-V3.1 model card (UE8M0 FP8 scale data format)"
    url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
  - title: "OpenAI — gpt-oss-120b & gpt-oss-20b Model Card (section 2.1: MXFP4, 4.25 bits per parameter)"
    url: https://arxiv.org/abs/2508.10925
  - title: "PyTorch 2.14 — tensor attributes (float8_e4m3fn, float8_e5m2, float8_e8m0fnu, float4_e2m1fn_x2)"
    url: https://docs.pytorch.org/docs/2.14/tensor_attributes.html
last_verified: "2026-10-04"
---

# 08.1 · Number formats and scaling

An 8-bit or 4-bit number cannot hold the values a transformer produces on its own: it has too few values and too little range. It works only together with a scale, a higher-precision number that says where the small format's values sit. This lesson takes the formats apart bit by bit — FP8 E4M3 and E5M2, the 4-bit E2M1, the exponent-only E8M0 — then the scaling schemes that make them usable (per tensor, per row, per tile, per block, OCP Microscaling and NVIDIA's NVFP4), works the rounding errors out by hand, and builds an exact emulation in PyTorch that matches the hardware casts bit for bit, so that the next lessons can study training recipes on any machine.

## Why this matters at a frontier lab

Every recent open frontier model has a precision story, and they differ. DeepSeek-V3 trained in FP8 with fine-grained scaling; DeepSeek-V3.1 changed the scale format to UE8M0 "to ensure compatibility with microscaling data formats" (model card); gpt-oss shipped its MoE weights in MXFP4 at "4.25 bits per parameter" (model card section 2.1); NVIDIA pretrained a 12B model on 10T tokens in NVFP4; Kimi K2 Thinking and DeepSeek-V4 quantised expert weights to 4 bits during post-training. Halving the bits halves weight memory and, on hardware with matching tensor cores, doubles peak GEMM throughput (H100: 989 dense BF16 TFLOP/s, 1,979 FP8). Whether a precision choice is safe depends on details that look minor: how many elements share a scale, what format the scale is stored in, whether it rounds up or down. A research engineer has to be able to read a recipe at the bit level, predict what it loses, and test that prediction before an expensive run depends on it.

## The idea

### A floating-point format is a grid

A binary float with $e$ exponent bits, $m$ mantissa bits and bias $b$ stores a sign $s$, an exponent field $E$ and a mantissa field $f$:

$$\text{normal } (E \ge 1):\; (-1)^s \, 2^{E-b} \left(1 + \frac{f}{2^m}\right), \qquad \text{subnormal } (E = 0):\; (-1)^s \, 2^{1-b} \, \frac{f}{2^m}.$$

Inside one binade $[2^k, 2^{k+1})$ the representable values are evenly spaced, $2^{k-m}$ apart; below the smallest normal value $2^{1-b}$ the spacing stays at $2^{1-b-m}$ (the subnormals) down to zero. Rounding to nearest therefore has a **relative** error of at most half a spacing over the value, $2^{-(m+1)}$, everywhere in the normal range — the format's precision. The exponent bits set the **range**: how many binades fit between the smallest and the largest value.

| Format | Bits (s.e.m) | Bias | Largest | Smallest normal | Smallest subnormal | Max relative error (normal) | Special values |
|---|---|---|---|---|---|---|---|
| E4M3 (FP8) | 1.4.3 | 7 | 448 | $2^{-6}$ | $2^{-9}$ | 6.25% | NaN = S.1111.111, no infinity |
| E5M2 (FP8) | 1.5.2 | 15 | 57,344 | $2^{-14}$ | $2^{-16}$ | 12.5% | IEEE infinities and NaNs |
| E2M1 (FP4) | 1.2.1 | 1 | 6 | 1 | 0.5 | 25% | none |
| E8M0 (scale) | 0.8.0 | 127 | $2^{127}$ | $2^{-127}$ | — | powers of two only | 0xFF = NaN, no zero |
| BF16 | 1.8.7 | 127 | $3.4 \times 10^{38}$ | $2^{-126}$ | $2^{-133}$ | 0.39% | IEEE |

The FP8 rows are PUBLICLY DOCUMENTED in Table 1 of Micikevicius et al. (arXiv 2209.05433), the FP4 and E8M0 rows in the OCP MX v1.0 specification (Tables 5 and 7). E4M3 makes one unusual choice: it has no infinities and only one NaN mantissa pattern, so its top exponent is an ordinary binade. Its largest value is $1.75 \cdot 2^8 = 448$ instead of 240, "one extra power of 2, from 17 to 18 binades". The E2M1 grid is tiny enough to list: $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$.

The FP8 paper's recommendation: E4M3 (more precision) for weights and activations, E5M2 (more range) for gradients. DeepSeek-V3 instead used E4M3 "on all tensors for higher precision" and handled range with fine-grained scales (lesson 08.2).

### Scaling: spend the bits where the values are

A tensor of activations might have values from $10^{-4}$ to $50$; gradients often span 20+ binades (the lab measures $2^{73}$ between a toy model's largest and smallest nonzero gradient). Even E5M2's 32 binades cannot cover everything, and E2M1 covers under 4. So a low-precision tensor is stored as small numbers $q$ plus a scale $s$:

$$x \approx s \cdot q, \qquad s = \frac{\operatorname{amax}(x_{\text{group}})}{q_{\max}}, \qquad q = \operatorname{round}\!\left(\frac{x}{s}\right),$$

where $\operatorname{amax}$ is the largest absolute value in the group of elements that share the scale and $q_{\max}$ is the format's largest value (448, 57,344 or 6). The group's largest value lands exactly on $q_{\max}$, and everything smaller than $s \cdot q_{\min}$ (the smallest nonzero value times the scale) becomes **zero** — underflow. One large value in a group pushes the scale up and flushes the small values of that group. So the central design choice is the group size:

| Granularity | Group | Used by |
|---|---|---|
| per tensor | the whole tensor | torchao `tensorwise`, Transformer Engine delayed/current scaling |
| per row | one row (one token, or one output channel) | torchao `rowwise` |
| 1 × 128 tile | one token, 128 channels | DeepSeek-V3 activations (section 3.3.2) |
| 128 × 128 block | a square block | DeepSeek-V3 weights |
| 1 × 32 block | 32 consecutive elements | OCP MX formats (MXFP8, MXFP6, MXFP4) |
| 1 × 16 / 16 × 16 | 16 elements / a 2-D block | NVFP4 activations and gradients / weights |

Smaller groups contain outliers better but cost more scale storage and more work. A block-scaled GEMM wants one scale per block along the dimension it contracts over (the $K$ of $XW^\top$), which is why every scheme above groups along rows of the left operand; lesson 08.2 shows what this means for the backward pass.

### What format is the scale?

- **FP32 scales** (torchao, DeepSeek-V3): $s = \operatorname{amax}/q_{\max}$ exactly.
- **Power-of-two scales.** Multiplying by $2^k$ only changes the exponent, so it is exact and cheap in hardware. The OCP MX spec (section 6.3) stores scales as **E8M0** and gives the conversion as "the largest power-of-two less than or equal to $\max(|V_i|)$, divided by the largest power-of-two representable in the element data type"; in symbols $X = 2^{\lfloor \log_2 \operatorname{amax} \rfloor - e_{\max}}$ with $e_{\max} = 2$ for E2M1 and 8 for E4M3 (our restatement; Algorithm 1 of arXiv 2310.10537 writes it this way). Elements that then exceed the maximum "should be clamped". Because $\lfloor\cdot\rfloor$ rounds the scale **down**, the top values of a block can land above $q_{\max}$ and saturate. The alternative, $s = 2^{\lceil \log_2(\operatorname{amax}/q_{\max}) \rceil}$, rounds the scale **up**: nothing saturates, at the cost of up to one binade of unused range. torchao's `rowwise` recipe and DeepSeek's UE8M0 scales are power-of-two scales (DeepSeek-V3.1: "trained using the UE8M0 FP8 scale data format on both model weights and activations"; the card does not say which rounding).
- **NVFP4's two levels** (arXiv 2509.25149 section 2 and appendix B; NVIDIA blog): 16 E2M1 elements share one **E4M3** block scale, and the whole tensor has one **FP32** scale. The tensor decode scale is $t = \operatorname{amax}(x)/(6 \cdot 448)$; each block's scale is $\operatorname{amax}_b/6$, divided by $t$ and rounded to E4M3. An E4M3 scale has three mantissa bits, so it can sit between powers of two and match the block's amax more closely than E8M0 can; the FP32 level keeps the E4M3 scales inside their own range. Storage: $4 + 8/16 = 4.5$ bits per value ("4.5 bits per value", NVIDIA blog), against MXFP4's $4 + 8/32 = 4.25$.

### Rounding: to nearest, or stochastically

Hardware casts round to nearest, ties to even (RNE): the result is deterministic and biased for any fixed input. **Stochastic rounding** (SR) rounds $x$ between neighbours $a < x < c$ up with probability $(x - a)/(c - a)$, so $\mathbb{E}[\operatorname{SR}(x)] = x$: unbiased, at the price of noise. With 4-bit gradients the bias of RNE adds up over many steps; NVFP4 uses SR for gradients only (lesson 08.3).

## Worked example

### Encoding 0.3 in E4M3 by hand

$0.3$ is in the binade $[0.25, 0.5)$, so $k = -2$. The spacing there is $2^{k-m} = 2^{-5} = 0.03125$. $0.3 / 0.03125 = 9.6$, which rounds to 10, so the stored value is $10 \cdot 0.03125 = 0.3125$ — a relative error of $4.2\%$, under the $6.25\%$ bound. Bits: exponent field $E = k + b = -2 + 7 = 5 = 0101_2$; mantissa $0.3125 / 0.25 = 1.25 = 1 + 2/8$, so $f = 010_2$. The byte is $0\,0101\,010_2 = \text{0x2A}$, which is what `torch.tensor([0.3]).to(torch.float8_e4m3fn).view(torch.uint8)` returns. In E5M2 the spacing is $2^{-4}$, $0.3/0.0625 = 4.8 \to 5$, also $0.3125$; in E2M1 (no scale) 0.3 lies between 0 and 0.5 and rounds to 0.5, a 67% error.

### One block, four scaling rules

Take a block whose meaningful entries are $x = (0.9, -0.1, 0.02, 3.5)$ (the rest of the block is zero) and quantise to E2M1.

- **FP32 scale**: $s = 3.5/6 = 0.5833$; $x/s = (1.543, -0.171, 0.034, 6)$; rounded: $(1.5, 0, 0, 6)$; back: $(0.875, 0, 0, 3.5)$. The two small values underflow.
- **OCP E8M0** ($e_{\max} = 2$): $\lfloor \log_2 3.5 \rfloor = 1$, $s = 2^{1-2} = 0.5$; $x/s = (1.8, -0.2, 0.04, 7)$. $7 > 6$ saturates to 6, so 3.5 comes back as **3.0**; $1.8 \to 2$ gives 1.0. The largest value lost 14%.
- **Power of two, rounded up**: $s = 2^{\lceil \log_2(3.5/6) \rceil} = 2^0 = 1$; $3.5$ is a tie between 3 and 4 and rounds to the even mantissa, **4.0**; $0.9 \to 1$. Nothing saturates, but the grid is coarser.
- **NVFP4**, if this block holds the tensor's amax: $t = 3.5/2688 = 0.001302$; block scale $3.5/6/t = 448$, exactly representable in E4M3; $s = 448\,t = 0.5833$, the same as the FP32 scale, so the result is $(0.875, 0, 0, 3.5)$.

Now add a second block $(0.05, 0.01, -0.03, 0.002)$ to the same tensor. With **one scale for the whole tensor** ($s = 0.5833$) every value of the second block divides to less than $0.25$ and all four become zero. With NVFP4 the second block gets its own scale: $0.05/6/t = 6.4$, rounded in E4M3 to 6.5, $s = 6.5\,t = 0.008464$, and the block comes back as $(0.0508, 0.0085, -0.0339, 0)$ — three of four values survive. This is the whole case for fine-grained scaling, in eight numbers (the lab's `nvfp4_qdq` reproduces them).

### Error budget per format

For a value in the normal range, RNE's relative error is at most $2^{-(m+1)}$: 6.25% (E4M3), 12.5% (E5M2), 25% (E2M1). Averaged over values spread across a binade, the root-mean-square relative error is roughly a third to a half of that bound; on real tensors the lab measures 2.6% (E4M3), 5.3% (E5M2) and 9–12% (block-scaled FP4). Values below the scaled grid's smallest step are lost entirely; that, not the mantissa, is what decides the FP4 numbers.

## Shapes and cost

| Tensor (Baseline-0, main path) | Shape | Stored dtype | Scales | Device |
|---|---|---|---|---|
| activation entering `gate_proj` | (B·T, 768) = (32,768, 768) per micro-batch of 32 × 1,024 | E4M3 (FP8) or E2M1 packed 2 per byte | 1 × 128 tiles: (32,768, 6) fp32; MX: (32,768, 24) E8M0; NVFP4: (32,768, 48) E4M3 + 1 fp32 | GPU |
| `gate_proj.weight` | (2,816, 768) | E4M3 / E2M1 | 128 × 128: (22, 6) fp32; NVFP4 16 × 16: (176, 48) E4M3 | GPU |
| emulation in this course | same shapes | float32 holding exactly representable values | float32 | CPU or GPU |

PyTorch 2.14 has `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch.float8_e8m0fnu` (since 2.7) and the packed `torch.float4_e2m1fn_x2` (since 2.8); the last two are "shell" dtypes with limited operator support (PyTorch docs), which is one reason the course emulates.

**Memory** for Baseline-0's 96.75M non-embedding weights: BF16 193.5 MB; FP8 with 128 × 128 FP32 block scales $96.75\text{M} \cdot (8 + 32/16{,}384)/8 \approx 96.8$ MB; MXFP4 at 4.25 bits 51.4 MB; NVFP4 at 4.5 bits 54.4 MB.

**Time.** Quantising is elementwise and memory-bound (lesson 02.1). Casting the activation above from BF16 to FP8 reads $32{,}768 \cdot 768 \cdot 2 = 50.3$ MB and writes 25.2 MB: on an H100 at 3.35 TB/s that is at least 22.5 µs. The FP8 GEMM it feeds, $2 \cdot 32{,}768 \cdot 768 \cdot 2{,}816 = 1.42 \times 10^{11}$ FLOPs at 1,979 dense FP8 TFLOP/s, takes at least 71.6 µs. An unfused cast costs about 31% of the GEMM it accelerates (PROJECTED from the roofline; real kernels are slower than both bounds). That is why production kernels fuse quantisation into the preceding operation, and why lesson 08.2 measures speed only with real kernels.

## Build it

`labs/common/frontierlab/precision/` holds the emulation; everything is plain PyTorch you can read.

```python
import torch
from frontierlab.precision import E4M3, QuantSpec, decode, encode, qdq, quant_error, round_to_format

round_to_format(torch.tensor([0.3]), E4M3)                  # tensor([0.3125])
hex(int(encode(torch.tensor([0.3125]), E4M3)))              # '0x2a'
spec = QuantSpec("e2m1", (1, 16), "nvfp4")                  # format, block (rows, cols; 0 = whole), scale kind
qdq(torch.tensor([[0.9, -0.1, 0.02, 3.5] + [0.0] * 12]), spec)   # [[0.875, 0, 0, 3.5, ...]]
quant_error(torch.randn(256, 512), "mxfp4")                 # {'rel_err': ..., 'underflow': ..., 'saturated': ..., 'bits': 4.25}
```

- `formats.round_to_format` finds each value's binade with `torch.frexp` (exact), divides by the binade's spacing with `torch.ldexp` (exact), rounds (`torch.round` is half-to-even; `"sr"` adds a uniform random number before `floor`), multiplies back and saturates. All formats here have at most 7 mantissa bits, so every step is exact in float32: this is the hardware's rounding, not an approximation of it.
- `quant.quantize(x, spec)` views the tensor as `(R, C)` blocks, computes one scale per block from its amax with the chosen scale rule (`"fp32"`, `"pow2"`, `"e8m0"`, `"nvfp4"`), rounds `x / s`, and returns the grid values and the scales.

Correctness checks in `labs/common/tests/test_precision.py` (all pass): every one of the 256 E4M3 and E5M2 bit patterns decodes to the value PyTorch gives; rounding 200,000 values spread over 36 binades (subnormals and ties included) equals `x.to(torch.float8_e4m3fn)` and `.to(torch.float8_e5m2)` bit for bit, and `encode` gives the same bytes; E8M0 rounding matches `torch.float8_e8m0fnu`; stochastic rounding is unbiased to within 0.005 over 200,000 draws. Overflow is checked separately, because casts differ: on torch 2.14.1 (CPU, measured in this build) an E4M3FN cast saturates to 448 and an E5M2 cast overflows to infinity from 61,440 (57,344 plus half a spacing) — the OCP FP8 spec allows both a saturating and an overflowing mode, so never assume which one a kernel uses.

## What the evidence says

- **FP8 E4M3/E5M2 — ESTABLISHED.** Standardised (OCP 8-bit spec, arXiv 2209.05433), supported by H100, L4 and later tensor cores, used in production training (DeepSeek-V3 section 3.3) and serving.
- **Block scaling with power-of-two scales (MX) — ESTABLISHED as a format, PROMISING for training.** The OCP MX v1.0 spec (September 2023) fixes $k = 32$ and E8M0 scales for MXFP8, MXFP6, MXFP4 and MXINT8; the spec leaves "detailed algorithms for computing the block scale" out of scope, so two "MXFP4" implementations can round scales differently. Rouhani et al. report MXFP6 training of GPT models from 20M to 1.5B parameters with "minimal accuracy loss" (Table 7). gpt-oss's MoE weights are MXFP4, 4.25 bits per parameter (PUBLICLY DOCUMENTED, model card section 2.1).
- **NVFP4 — PROMISING.** Format PUBLICLY DOCUMENTED by NVIDIA (blog, Transformer Engine docs, arXiv 2509.25149); its training results come from NVIDIA (company claim), with limited independent replication. Lesson 08.3.
- **UE8M0 scales in DeepSeek-V3.1 — MODEL-SPECIFIC.** One sentence on the model card; no ablation published.
- **What we could not verify:** how the H800's or B200's FP8 casts round a block scale inside fused kernels; we emulate the documented rules and label the rest.

## Lab

**Folder:** [`labs/module-08/lesson-01/`](../../labs/module-08/) · **Time:** about 70 minutes · **Pass check:** `pytest labs/module-08/lesson-01` passes (your `round_fp` matches the PyTorch FP8 casts bit for bit); your notes give the error table from step 3 and answer the three questions in step 4.

### Experiment contract

- **Question:** for the tensors a training step actually quantises (linear inputs, weights, output gradients), how much does each format and scaling granularity lose? Decision informed: which granularities are worth testing in training (lessons 08.2–08.3).
- **Hypothesis and status:** (H1) FP8's relative error is about the same for every granularity on well-behaved tensors, because floating-point error is relative; granularity matters for FP8 only when the range is exceeded. (H2) For FP4, smaller blocks reduce underflow sharply. (H3) An outlier channel hurts per-tensor scaling far more than 1 × 128 tiles. Established effects (format arithmetic; DeepSeek-V3 section 3.3.2 for H3).
- **Baseline:** E4M3 with one FP32 scale per tensor.
- **Changed variable:** format and scaling rule. **Controlled:** the same tensors (one batch of 16 × 128 tokens from the same 60-step toy checkpoint, fixed data seed 123), float64 emulation.
- **Comparison axis:** equal tensors (same values quantised every way). It says nothing about training quality, which depends on how errors compound over steps.
- **Budget:** free CPU about 1.5 minutes measured; GPU variants a few minutes.
- **Metrics and decision rule:** mean over the 28 linears of relative error, underflow fraction and saturation fraction, plus bits per value. Rule: carry a scheme into the training labs if its gradient underflow is below 1% (FP8) or it is the best FP4 scheme on underflow at no more than 4.5 bits.
- **Correctness checks:** `pytest labs/module-08/lesson-01` and `pytest labs/common/tests/test_precision.py` pass first.
- **Fallback evidence:** none needed; this is deterministic arithmetic on fixed tensors.
- **Limits:** one tiny model early in training; real activations at 7B+ have systematic outlier channels much larger than the ×100 injected here.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | any 1× CUDA GPU, a few minutes (emulation; not run in this build, part of the Module 8 pilot) | steps 1–4; step 3 as `error_tour.py --device cuda --preset pilot-30m --steps 300` |
| Free GPU (Colab/Kaggle T4) | T4 | `error_tour.py --device cuda --preset pilot-10m --steps 300` |
| Free CPU | laptop; measured at 81 seconds (16-thread laptop, torch 2.14.1+cpu, 2026-10-04) | steps 1–4 exactly as written |

### Steps

1. **Implement** `round_fp`, `block_qdq`, `mx_shared_scale` and `nvfp4_qdq` in `lab.py` with plain torch operations; run `pytest labs/module-08/lesson-01`. The first test compares your `round_fp` with PyTorch's FP8 casts on 50,000 values across 32 binades.
2. **By hand, then by code:** redo the two blocks of the worked example with your `nvfp4_qdq` and with `block_qdq(..., block=16)` on E2M1 (FP32 scales). Check every number.
3. **Measure:** `python labs/module-08/lesson-01/error_tour.py` (it trains the toy model for 60 steps first, about 40 seconds).
4. **Write down:** (a) which granularity has the lowest *gradient* underflow among the FP4 schemes, and why the 16 × 16 weight layout is a poor choice for gradients; (b) why `mxfp4 1x32` shows saturation and `mxfp4 1x32 ceil` does not, and what the ceil variant pays instead; (c) the ratio between the largest and smallest nonzero gradient the script prints, and which formats could hold it without a scale.

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04), mean over the toy model's 28 linears; columns are relative error / underflow fraction / saturation fraction:

| Spec | Bits | Activations x | Weights W | Gradients dy |
|---|---|---|---|---|
| E4M3 per tensor | 8.00 | 0.0266 / 0.000 / 0.000 | 0.0265 / 0.000 / 0.000 | 0.0265 / 0.002 / 0.000 |
| E5M2 per tensor | 8.00 | 0.0528 / 0.000 / 0.000 | 0.0525 / 0.000 / 0.000 | 0.0527 / 0.001 / 0.000 |
| E4M3 1 × 128, FP32 | 8.32 | 0.0247 / 0.000 / 0.000 | 0.0257 / 0.000 / 0.000 | 0.0240 / 0.000 / 0.000 |
| MXFP8 1 × 32 (E8M0) | 8.25 | 0.0312 / 0.000 / 0.011 | 0.0301 / 0.000 / 0.007 | 0.0316 / 0.000 / 0.008 |
| MXFP4 1 × 32 (E8M0) | 4.25 | 0.118 / 0.138 / 0.030 | 0.116 / 0.080 / 0.024 | 0.125 / 0.174 / 0.022 |
| MXFP4 1 × 32, scale rounded up | 4.25 | 0.115 / 0.158 / 0.000 | 0.118 / 0.114 / 0.000 | 0.127 / 0.212 / 0.000 |
| NVFP4 1 × 16 | 4.50 | 0.094 / 0.109 / 0.035 | 0.095 / 0.068 / 0.034 | 0.093 / 0.139 / 0.034 |
| NVFP4 16 × 16 | 4.03 | 0.111 / 0.149 / 0.005 | 0.112 / 0.102 / 0.002 | 0.141 / 0.381 / 0.002 |
| E2M1 per tensor | 4.00 | 0.136 / 0.218 / 0.000 | 0.125 / 0.141 / 0.000 | 0.370 / 0.740 / 0.000 |
| INT4, groups of 32 | 5.00 | 0.098 / 0.181 / 0.001 | 0.096 / 0.131 / 0.001 | 0.109 / 0.240 / 0.001 |

With one input channel multiplied by 100: E4M3 per tensor 0.0261 / 0.001 underflow, E4M3 1 × 128 0.0080 / 0.000, NVFP4 1 × 16 0.075 / 0.184, E2M1 per tensor 0.256 / **0.963**. The gradient range was $2^{73.5}$ (largest $1.05 \times 10^{-3}$, smallest nonzero $7.8 \times 10^{-26}$).

How to read it. H1 holds: every FP8 scheme loses 2.4–3.2% (MXFP8 a little more, because its rounded-down scales saturate about 1% of values). H2 holds: per-tensor FP4 flushes 74% of the gradient values to zero; 1 × 16 NVFP4 blocks flush 14%. NVFP4's saturation (3.4%) comes from rounding each block scale to E4M3 to nearest, which sometimes lands below $\operatorname{amax}_b/6$; that is the documented rule, and those values lose at most one grid step. H3 holds for FP4 and holds only weakly for FP8 at ×100 — E4M3's 18 binades absorb a ×100 outlier, so the relative-error column hardly moves. Relative error is a norm and is dominated by the largest values; underflow is the column that shows what small values lose.

<details>
<summary>Hint for TODO 1</summary>

`m, e = torch.frexp(x)` gives $x = m \cdot 2^e$ with $0.5 \le |m| < 1$, so the binade exponent is `e - 1`. Clamp it with `torch.clamp(e - 1, min=1 - bias)` so subnormals use the smallest normal spacing. Then `torch.ldexp(torch.round(torch.ldexp(x, -shift)), shift)` with `shift = k - man_bits` (as `int32`). Zero needs no special case: `frexp(0)` gives `e = 0`, and the clamp handles it.

</details>

<details>
<summary>Hint for TODO 4</summary>

Reshape to `(R, C // 16, 16)` and take `amax` over the last dimension with `keepdim=True`. The block scale is rounded with *your* `round_fp(..., 4, 3, 7, 448.0)`; the elements with `round_fp(..., 2, 1, 1, 6.0)`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-08/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-08/lesson-01`.

</details>

## Common mistakes

- **Reading a FP8 relative error as the cost of FP8.** A 2.6% error per element is not a 2.6% loss increase; errors in a GEMM partly average out over the contraction, and training compounds them in ways only a training run shows (lesson 08.2).
- **Using the OCP scale rule and expecting no clipping.** $\lfloor \log_2 \operatorname{amax} \rfloor - e_{\max}$ lets the largest values exceed $q_{\max}$ by up to a factor of two; they are clamped. Rounding the scale up avoids it but wastes range. Say which rule you use: "MXFP4" alone does not say.
- **Assuming casts saturate (or overflow).** Both modes exist; on torch 2.14.1 CPU the two FP8 casts behave differently. Clamp explicitly before casting if your recipe needs saturation.
- **Counting 4 bits per value for FP4.** Scales are storage too: MXFP4 is 4.25 bits, NVFP4 4.5, INT4 with an FP32 scale per 32 elements 5.0.
- **Scaling along the wrong axis.** A block must lie along the GEMM's contraction dimension. Per-row scales of an activation are per token; per-row scales of a weight are per output channel; the backward pass contracts over other dimensions (lesson 08.2).

## References

- P. Micikevicius et al., *FP8 Formats for Deep Learning*, Table 1 and section 3. https://arxiv.org/abs/2209.05433
- Open Compute Project, *OCP Microscaling Formats (MX) Specification v1.0* (September 2023), Table 1, sections 5.3–5.4 and 6.3. https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- B. D. Rouhani et al., *Microscaling Data Formats for Deep Learning*, Algorithm 1 and Table 7. https://arxiv.org/abs/2310.10537
- NVIDIA, *Pretraining Large Language Models with NVFP4*, section 2 and appendix B. https://arxiv.org/abs/2509.25149
- NVIDIA Technical Blog, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- DeepSeek-AI, *DeepSeek-V3.1* model card, Introduction. https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, section 2.1. https://arxiv.org/abs/2508.10925
- PyTorch 2.14, tensor attributes (FP8, E8M0 and packed FP4 dtypes). https://docs.pytorch.org/docs/2.14/tensor_attributes.html
- Shared code: `labs/common/frontierlab/precision/formats.py`, `quant.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[08.2 · FP8 training](lesson-02.md)
