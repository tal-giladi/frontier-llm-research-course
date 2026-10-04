---
id: "08.2"
module: 8
minutes: 40
practice_minutes: 90
prerequisites: ["08.1", "02.4", "07.5"]
objectives:
  - Lay out which tensors of a training step DeepSeek-V3 computes in FP8 and which it keeps in BF16 or FP32, and justify each choice from the section 3.3 evidence.
  - Quantise each operand of the three GEMMs of a linear layer along its contraction dimension (1 × 128 tiles forward, 128 × 1 for the weight gradient, 128 × 128 weight blocks), and test the emulated layer against the reference.
  - Explain with a worked example and a measurement why limited accumulation precision needs FP32 promotion every N_C = 128 elements.
  - Run BF16 and FP8 arms under an experiment contract, report the loss gap with seed-paired and window-paired intervals, and state the decision the pre-stated rule gives.
  - Project the step-time gain of FP8 linears from the roofline, and say which number only real kernels (torchao Float8 on an L4 or H100) can provide.
volatility: implementation
sources:
  - title: "DeepSeek-AI — DeepSeek-V3 Technical Report, section 3.3 and appendix B"
    url: https://arxiv.org/abs/2412.19437
  - title: "torchao 0.18.0 — float8 training (convert_to_float8_training, Float8LinearConfig), source at tag v0.18.0"
    url: https://github.com/pytorch/ao/tree/v0.18.0/torchao/float8
  - title: "torchao 0.18.0 — training workflows documentation"
    url: https://github.com/pytorch/ao/blob/v0.18.0/docs/source/workflows/training.md
  - title: "Micikevicius et al. — FP8 Formats for Deep Learning"
    url: https://arxiv.org/abs/2209.05433
  - title: "NVIDIA Transformer Engine — common API (DelayedScaling, Float8CurrentScaling, Float8BlockScaling, MXFP8BlockScaling)"
    url: https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html
  - title: "NVIDIA H100 and L4 product pages (FP8 and BF16 tensor-core peaks)"
    url: https://www.nvidia.com/en-us/data-center/h100/
last_verified: "2026-10-04"
---

# 08.2 · FP8 training

FP8 tensor cores do twice the BF16 work per second, but a training run is not one GEMM: it is three GEMMs per linear layer, each with operands that have different ranges, plus normalisations, attention, an optimizer and the output head. This lesson takes DeepSeek-V3's published FP8 recipe apart — which tensors go to FP8, how they are scaled (1 × 128 tiles, 128 × 128 blocks), why partial sums are promoted to FP32 every 128 elements, and what stays in BF16 — builds an emulation of it and of torchao's tensorwise and rowwise recipes, trains BF16 and FP8 arms under an experiment contract, and projects what the real kernels can save, which only a run on an FP8-capable GPU can confirm.

## Why this matters at a frontier lab

DeepSeek-V3 (671B total parameters) was trained with an FP8 mixed-precision framework, and the report says its validation runs kept the "relative loss error of our FP8-training model ... consistently below 0.25%" against BF16 (section 3.3). That is the evidence that made FP8 pretraining a production option rather than a demo. But the same section lists five places where the recipe *had* to deviate from "cast everything to FP8": fine-grained scales, higher-precision accumulation, E4M3 everywhere, online scaling, and components left in BF16. Each is a decision a lab must re-make for its own model and hardware, and each can be wrong in a way that shows up only after many tokens. The engineering questions are concrete: does the FP8 path keep the loss within a stated tolerance of BF16 at our scale, and does it buy enough throughput on our GPU to be worth the risk?

## The idea

### Three GEMMs per linear layer

A linear layer $y = x W^\top$ with $x \in \mathbb{R}^{M \times K}$ (M tokens, K input features) and $W \in \mathbb{R}^{N \times K}$ costs three GEMMs per training step:

| GEMM | Computes | Contracts over | FP8 operands |
|---|---|---|---|
| Fprop | $y = x W^\top$ | $K$ (input features) | $x$, $W$ |
| Dgrad | $\partial L/\partial x = \partial L/\partial y \cdot W$ | $N$ (output features) | $\partial L/\partial y$, $W$ |
| Wgrad | $\partial L/\partial W = (\partial L/\partial y)^\top x$ | $M$ (tokens) | $\partial L/\partial y$, $x$ |

A block-scaled GEMM needs one scale per block along the contraction dimension of each operand (lesson 08.1). So $x$ is scaled per token across channels for Fprop but per channel across tokens for Wgrad; DeepSeek-V3 appendix B.2 says exactly this: activations need 1 × 128 grouping in the forward pass and 128 × 1 in the backward pass. The weight is used along $K$ in Fprop and along $N$ in Dgrad; with 128 × 128 square blocks both orientations get the same scales, so the backward pass differentiates the same quantised weight the forward pass used.

### DeepSeek-V3's recipe (section 3.3, PUBLICLY DOCUMENTED)

- **Fine-grained quantisation (3.3.2).** "For activations, we group and scale elements on a 1x128 tile basis (i.e., per token per 128 channels); and ... for weights ... on a 128x128 block basis." One outlier then affects 128 values, not a whole tensor.
- **Online scaling.** The amax is computed "online for each 1x128 activation tile or 128x128 weight block", from the current tensor, instead of the delayed scaling (a history of past amaxes) that earlier frameworks used.
- **E4M3 on all tensors** ("Mantissa over Exponents"): prior work used E4M3 for Fprop and E5M2 for the gradients; DeepSeek uses E4M3 everywhere and relies on the small groups for range.
- **Increasing accumulation precision.** The "accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits"; for two random matrices with $K = 4096$ this gives "a maximum relative error of nearly 2%". The fix: after $N_C = 128$ elements (4 WGMMA instructions) the partial result is promoted to FP32 registers on the CUDA cores and accumulated there. The per-group scales along $K$ are multiplied in on the CUDA cores during that promotion; matching the group size along $K$ to $N_C = 128$ makes each promoted partial sum share one scale per operand (our reading of section 3.3.2, INFERENCE).
- **What stays in higher precision (3.3.1):** "the embedding module, the output head, MoE gating modules, normalization operators, and attention operators" stay in BF16 or FP32; master weights and weight gradients stay FP32. **Low-precision storage (3.3.3):** AdamW's first and second moments are kept in BF16 rather than FP32; activations saved for the backward pass of a linear are cached in FP8, except the inputs of the linear after attention, which use a custom E5M6 format with power-of-two scales.
- **Evidence:** compared with BF16 at two scales, about 16B total parameters on 1.33T tokens and about 230B on about 0.9T tokens, the relative loss error stays below 0.25% (appendix B.1, Figure 10).

### torchao Float8 (the course's main path)

torchao 0.18.0 (released 2026-08-03; API checked at tag v0.18.0) swaps `nn.Linear` modules for `Float8Linear`, whose three GEMMs call `torch._scaled_mm` on FP8 operands:

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
config = Float8LinearConfig.from_recipe_name("rowwise")       # "tensorwise" | "rowwise" | "rowwise_with_gw_hp"
convert_to_float8_training(model, module_filter_fn=lambda mod, fqn: fqn != "lm_head", config=config)
model = torch.compile(model)
```

Its recipes, from `torchao/float8/config.py` at v0.18.0: **tensorwise** (the default, "fastest"): one scale per tensor, E4M3 for inputs and weights, E5M2 for the output gradient. **rowwise**: per-row (axiswise) scales, E4M3 everywhere, scales rounded to powers of two. **rowwise_with_gw_hp** ("most accurate"): rowwise, with the weight-gradient GEMM left in high precision. The GEMM needs both weight dimensions to be multiples of 16, and the docs pair the recipes with `torch.compile` "for competitive performance" — without it the amax reductions and casts run as separate kernels. torchao has no 1 × 128 / 128 × 128 recipe for dense linears, so on the main path the DeepSeek recipe is studied through emulation (numerics) and the roofline (cost), and torchao's rowwise is the real-kernel stand-in for fine-grained scaling.

## Worked example

### Which way to scale, by hand

$x$ has 2 tokens and 4 channels, $x = \begin{pmatrix} 1 & 2 & 3 & 400 \\ 0.1 & 0.2 & 0.3 & 0.4 \end{pmatrix}$ — token 1 has an outlier in channel 4. Quantise to E2M1 (largest value 6) to make the effect visible.

- **Fprop** contracts over channels, so the groups are rows. Row 1: $s = 400/6 = 66.7$; its entries 1, 2, 3 become 0.015, 0.03, 0.045 and flush to zero — the outlier costs its own token. Row 2: $s = 0.4/6 = 0.0667$; $0.1/s = 1.5$, $0.2/s = 3$, $0.3/s = 4.5 \to 4$, $0.4/s = 6$: back to $(0.1, 0.2, 0.267, 0.4)$, all kept.
- **One scale for the tensor** ($s = 66.7$) flushes *all* of row 2 as well: the outlier costs every token.
- **Wgrad** contracts over tokens, so the groups are columns. Column 4 is $(400, 0.4)$: $s = 66.7$ and 0.4 flushes; columns 1–3 have their own scales ($1/6$, $2/6$, $3/6$) and keep both tokens: $0.1/(1/6) = 0.6 \to 0.5$, back to 0.083.

The same tensor loses different values in different GEMMs. A 1 × 128 tile confines an outlier to 128 channels of one token in Fprop; the 128 × 1 tile of the backward pass confines it to 128 tokens of one channel in Wgrad.

### Why promote every $N_C$ elements

Model an accumulator that keeps 2 bits after the leading one and truncates the rest. Add sixteen products of 0.125 (exact sum 2.0). The running sum goes 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 — each representable with 2 fractional bits at its binade — and then $1.0 + 0.125 = 1.125 = 1.001_2$ needs 3 bits and truncates back to 1.0. It stays at 1.0 for the remaining eight adds: the result is 1.0, a 50% error. Promote every 8 elements instead: each group of eight sums exactly to 1.0 in the accumulator, and the two groups are added in full precision: 2.0, exact. The bigger the running sum gets relative to each new product, the more of each product falls below the accumulator's last bit; restarting the accumulator every $N_C$ elements keeps the running sum small. The same mechanism at 14 bits and $K = 4096$ is DeepSeek's 2%.

The lab measures this with a model of the accumulator (`frontierlab.precision.accum`, labelled a model: the H800's real adder is not documented). Maximum relative error over 128 dot products of non-negative E4M3 values:

| Accumulator bits after the leading one | $K = 512$ | $K = 4096$ | $K = 4096$, promoted every 128 |
|---|---|---|---|
| 13 (truncating) | 1.20% | 8.54% | 0.29% |
| 14 (truncating) | 0.61% | 4.40% | 0.14% |
| 15 (truncating) | 0.31% | 2.20% | 0.07% |
| 13 (round to nearest) | 0.08% | 0.36% | — |

A 15-bit truncating model lands near DeepSeek's "nearly 2%" at $K = 4096$; how "around 14 bits" maps onto a bit count and a rounding rule is not public, so treat the bit count as a fit. The structure is the point: without promotion the error grows roughly linearly with $K$; with promotion it is flat at the $K = 128$ level.

### The size of the tolerance

DeepSeek's criterion is a *relative* loss error below 0.25%. At Baseline-0's projected loss of about 3.3 nats that is 0.008 nats; at the toy model's 6.6 nats after 200 steps it is 0.016 nats. Compare it with the seed noise floor you measured in Module 1: a tolerance smaller than the noise floor cannot be checked with two runs; the design must pair arms by seed (same initial weights and data order) so the seed noise cancels in the difference.

## Shapes and cost

| Tensor (one linear, main path) | Shape | Stored as | Scales | Device |
|---|---|---|---|---|
| $x$ for Fprop | (M, K) = (16,384, 768) | E4M3 | 1 × 128: (16,384, 6) fp32 | GPU |
| $x$ for Wgrad | $x^\top$ (768, 16,384) | E4M3 | 1 × 128 along tokens: (768, 128) fp32 | GPU |
| $W$ | (2,816, 768) | E4M3 | 128 × 128: (22, 6) fp32 | GPU |
| $\partial L / \partial y$ | (16,384, 2,816) | E4M3 (DeepSeek) / E5M2 (torchao tensorwise) | 1 × 128 | GPU |
| master $W$, $\partial L/\partial W$ | (2,816, 768) | fp32 | — | GPU |
| AdamW moments | (2,816, 768) × 2 | bf16 (DeepSeek), fp32 (torchao default) | — | GPU |
| emulation in the course | same shapes | float32 holding E4M3 values | float32 | CPU or GPU |

**Optimizer memory per parameter**: standard mixed precision keeps fp32 master weight, fp32 gradient and two fp32 moments, 16 bytes; DeepSeek-V3's choice (fp32 master and gradient, BF16 moments) is 12 bytes. **Activation memory**: caching the inputs of every linear in FP8 halves those saved tensors relative to BF16.

**Throughput, PROJECTED** with `frontierlab.precision.cost.projected_speedup` (Module 2 roofline: each op at its bound, backward = 2 × forward; the seven block linears at the FP8 peak, 2 × BF16 on H100 and L4, with FP8 operands; output head, attention and elementwise ops unchanged; casts as extra memory traffic unless fused):

$$t_{\text{FP8}} = \sum_{\text{non-linear ops}} t_{\text{op}} + \sum_{\text{linears}} 3\max\!\left(\frac{F}{2P_{\text{BF16}}}, \frac{Q_{\text{FP8}}}{W}\right) + \frac{Q_{\text{casts}}}{W}$$

| Model, batch, GPU | Block linears' share of the BF16 step | FP8, casts unfused | FP8, casts fused | Amdahl limit |
|---|---|---|---|---|
| Baseline-0, 16 × 1,024, H100 SXM | 46% | 0.95× | 1.28× | 1.30× |
| pilot-30m, 8 × 512, H100 SXM | 31% | 0.81× | 1.03× | 1.18× |
| pilot-30m, 8 × 512, L4 | 28% | 0.83× | 1.08× | 1.17× |

At course scale the output head (vocabulary 32,768) and the memory-bound elementwise ops are a large part of the step, so FP8 linears can save at most about 30% even with perfectly fused casts — and unfused casts can make FP8 slower than BF16. Large models are different: their linears dominate the step. These are bounds, not measurements; real kernels reach a fraction of both peaks. The lab's `bench_fp8.py` measures the real number on an L4 or H100.

## Build it

`frontierlab/precision/linear.py` implements the recipes as one autograd function:

```python
from frontierlab.precision import QLinear, get_recipe
r = get_recipe("fp8-deepseek")      # act 1x128 E4M3, weight 128x128 E4M3, grad 1x128 E4M3, FP32 scales
q = QLinear.from_linear(lin, r)     # same Parameter object: state-dict keys and values unchanged
```

`_QLinearFn.forward` quantises $x$ along $K$ and $W$ along $K$, multiplies in float32 (the products of E4M3 values are exact, the sum is an FP32 accumulation, i.e. DeepSeek's promotion target), and saves $x$ and $W$. `backward` quantises $\partial L/\partial y$ along $N$ and re-quantises $W^\top$ along $N$ for Dgrad, then transposes $\partial L/\partial y$ and $x$ so the token dimension is last and quantises both along it for Wgrad. Gradients pass through every quantiser unchanged (straight-through), and both results go to the fp32 master weight. Recipes `fp8-tensorwise` and `fp8-rowwise` reproduce torchao's numerics (tensorwise: E5M2 gradients; rowwise: E4M3, power-of-two row scales). Autocast is switched off inside the function so the emulation is never silently done in BF16.

The training wrapper changes one `loop.main()` call: `python -m frontierlab.precision.train --recipe fp8-deepseek <loop args>`. It builds the model exactly as the loop does (same random draws), then swaps every block linear for a `QLinear`; the embedding and output head are never swapped. It writes a `precision` block (recipe, specs, number of quantised linears, `emulated: true`) into the run card, and `--precision-log` writes `precision.jsonl` (relative error, underflow and saturation of every quantised tensor, every N steps) next to the Module 7 `stability.jsonl`, which still works (`--stability-log` with `--optimizer adamw`). On a GPU, `--torchao rowwise --compile` uses real torchao kernels instead.

Correctness checks (`labs/common/tests/test_precision.py`, all pass): with no quantisers `QLinear` equals `nn.Linear` exactly in float64, including gradients; the recipe's three GEMMs equal hand-built ones to $10^{-12}$; the emulated FP8 gradients are within 5% (relative norm) of the exact ones; the weight-swap leaves every state-dict key and value unchanged; the BF16 arm of the wrapper is bit-identical to the plain course loop; and stop-and-resume is bit-identical for `fp8-deepseek`, for NVFP4 with stochastic rounding, and for INT4 QAT with Muon. Building that last test found a latent issue in the course loop: `accounting.param_counts` instantiates attention modules (drawing random numbers) *after* the loop restores the checkpoint's RNG state. No earlier module drew random numbers during training, so it never mattered; stochastic rounding does. The wrapper runs the accounting with the RNG forked, and the loop fix is proposed in the Module 8 inbox.

## What the evidence says

- **FP8 training of large LLMs — ESTABLISHED.** PUBLICLY DOCUMENTED at production scale by DeepSeek-V3 (section 3.3, appendix B.1: below 0.25% relative loss error at about 16B and 230B parameters), with production frameworks (torchao, Transformer Engine's delayed, current-scaling and block-scaling recipes) supporting it.
- **The specific fine-grained recipe (1 × 128 / 128 × 128, E4M3 everywhere, promotion at $N_C = 128$) — MODEL-SPECIFIC → PROMISING.** One lab's evidence, designed around the H800's accumulator; Transformer Engine's `Float8BlockScaling` recipe is a related block-scaled option, but we found no independent ablation of DeepSeek's exact choices.
- **The 14-bit accumulator and the 2% error** are DeepSeek's measurements (company claim) on H800; whether they carry over to H100 or B200 kernels is not stated.
- **Speed.** FP8 tensor-core peaks are vendor numbers (company claim: H100 SXM 1,979 dense FP8 TFLOP/s, L4 242.5). The end-to-end gain depends on how much of the step is in large GEMMs and on cast fusion; at course scale the roofline bound is 1.1–1.3×. No speed figure in this lesson is measured; the pilot measures it on an L4.
- **Open question at course scale:** whether any loss gap is detectable at all. DeepSeek's tolerance is far below a toy model's seed noise; the lab therefore pairs arms by seed and by window, and states the result as a hypothesis test, not a reproduction.

## Lab

**Folder:** [`labs/module-08/lesson-02/`](../../labs/module-08/) · **Time:** about 90 minutes attended, about 40 minutes unattended on the free CPU path · **Pass check:** `pytest labs/module-08/lesson-02` passes; your notes contain the accumulation table, the contract's decision for each FP8 arm with its interval, and the projected (and on a GPU, measured) speed-up with its formula.

### Experiment contract

- **Question:** at equal tokens, do FP8 linears (torchao tensorwise and rowwise numerics; DeepSeek-V3's fine-grained recipe) keep Baseline-0-family held-out loss within 0.25% of BF16, and on the main-path GPU, by how much do real FP8 kernels change step time? Decision informed: the precision of Recipe-R's linears.
- **Hypothesis and status:** (H1) every FP8 arm's paired gap to BF16 is within the margin — reported effect (DeepSeek-V3 appendix B.1 at 16B–230B), may not be measurable at course scale. (H2) the fine-grained recipe's gap is no larger than tensorwise's — reported (DeepSeek-V3 section 3.3.2's motivation), likely too small to resolve here. (H3) real FP8 kernels speed up the step by less than the roofline bound (1.08× fused on L4 pilot-30m; 1.28× on H100 Baseline-0) — established that real kernels fall short of bounds.
- **Baseline:** the BF16 arm (fp32 on the CPU), the plain course loop through the same wrapper (bit-identical to `frontierlab.train.loop`), AdamW at the loop's defaults; not re-tuned for either arm (the precision change is a numerics change, not a new optimizer).
- **Changed variable:** the precision recipe of the 28 block linears. **Controlled:** seeds {0, 1} (main path {0, 1, 2}), which fix initial weights and data order per seed for every arm; tokens; schedule; data and its hashes; evaluation windows; hardware and software versions within a variant.
- **Comparison axis:** equal tokens for quality; equal work per step for speed (`bench_fp8.py`). Neither says which reaches a target loss sooner in wall-clock; that follows only if both hold up (the project combines them).
- **Budget:** free CPU about 38 minutes of training, measured below; Colab L4 pilot PROJECTED about 1.5–2.5 GPU-hours for `--variant l4` (8 runs of pilot-30m × 2,000 steps × 16,384 tokens = $3.3 \times 10^7$ tokens at $2.3 \times 10^8$ FLOPs per token, $7.5 \times 10^{15}$ FLOPs, about 5 minutes per real-kernel run at an assumed 20% MFU of 121 TFLOP/s; the two emulated runs, assumed 5–10× slower, dominate); main path 1× H100, PROJECTED about 6–9 GPU-hours for `--variant main` (12 runs of pilot-70m × 4,000 × 65,536 tokens = $2.6 \times 10^8$ tokens at $6.1 \times 10^8$ FLOPs per token, $1.6 \times 10^{17}$ FLOPs, 0.23 hours per real-kernel run at 20% MFU of 989 TFLOP/s; 9 such runs plus 3 emulated runs at 5–10× that).
- **Metrics and decision rule:** held-out loss on 256 fixed validation windows of 128 tokens, evaluated in the arm's training precision; gap = FP8 − BF16 per seed, then averaged over seeds window by window with a 95% paired bootstrap interval; BF16 seed std and MDE reported. Margin = 0.25% of the BF16 loss. Rule (stated now, `decide()` in the lab): **reject** if the interval's lower end is above the margin; **inconclusive** if any single seed pair's gap exceeds the margin in either direction (the window interval does not contain seed-to-seed variation); **adopt** if the interval's upper end is at most the margin *and* the measured speed-up interval (real kernels) is entirely above 1; **numerics ok, speed not measured** for emulated arms that pass; otherwise **inconclusive**.
- **Correctness checks:** `pytest labs/module-08/lesson-02` and `pytest labs/common/tests/test_precision.py` pass; `python -m frontierlab.record runs/m08/l82/cpu/bf16-s0 runs/m08/l82/cpu/fp8-deepseek-s0 --changed precision` shows no other INVALIDATES line; on the GPU, `bench_fp8.py`'s first-step loss gate passes.
- **Fallback evidence:** DeepSeek-V3's Figure 10 curves (analysis of published results, labelled); the pilot's L4 traces once published.
- **Limits:** a 1.8M-parameter model for 200 steps (0.41M tokens) cannot show effects that appear after many tokens or with outlier features at scale; emulation measures numerics with exact FP32 accumulation (it does not emulate the 14-bit accumulator in training); speed is only what `bench_fp8.py` measures on real kernels.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM 80 GB; PROJECTED about 6–9 GPU-hours; not run in this build (part of the Module 8 pilot plan) | `train_fp8.py --variant main --data <vocab-32768 Data-v0>` (BF16, torchao tensorwise and rowwise, emulated DeepSeek; pilot-70m, 3 seeds), `compare_fp8.py --variant main --device cuda --speedup ...`, `bench_fp8.py --device cuda --hw H100-SXM --preset baseline0 --batch 16 --seq 1024` |
| Free GPU with FP8 (Colab L4, sm89) | L4 24 GB; PROJECTED 1.5–2.5 GPU-hours; this is the Module 8 pilot, not run in this build | `train_fp8.py --variant l4`, `compare_fp8.py --variant l4 --device cuda`, `bench_fp8.py --device cuda --hw L4 --preset pilot-30m --batch 8 --seq 512` |
| Free GPU without FP8 (T4) | T4 has no FP8 tensor cores (sm75) | the CPU variant with `--device cuda` added to the arms is possible but slow; prefer the CPU variant |
| Free CPU | laptop; measured below | steps 1–5 as written |

### Steps

1. **Implement** `tile_qdq`, `block_qdq`, `promoted_sum`, `fp8_backward` and `decide` in `lab.py`; run `pytest labs/module-08/lesson-02`.
2. **Accumulation:** `python labs/module-08/lesson-02/accum_demo.py` (about 30 seconds). Find the smallest accumulator width that keeps $K = 4096$ under 1% without promotion, and check the promoted rows.
3. **Train the arms:** `python labs/module-08/lesson-02/train_fp8.py` (8 runs; reruns skip finished runs and resume interrupted ones).
4. **Compare:** `python labs/module-08/lesson-02/compare_fp8.py`, then `python -m frontierlab.record runs/m08/l82/cpu/bf16-s0 runs/m08/l82/cpu/fp8-deepseek-s0 --changed precision`. Read `runs/m08/l82/cpu/fp8-deepseek-s0/precision.jsonl`: which layer's gradient loses the most?
5. **Cost:** `python labs/module-08/lesson-02/bench_fp8.py --preset baseline0 --batch 16 --seq 1024` prints the PROJECTED bounds; on an L4 or H100 add `--device cuda --hw ...` and record the measured speed-up with its interval.

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, another agent's jobs running, 2026-10-04; toy model, 200 steps × 16 × 128 tokens = 0.41M tokens, seeds 0 and 1; held-out loss on 256 windows of 128 tokens):

| Measurement | Result |
|---|---|
| `accum_demo.py` | 30 s; table in the worked example |
| training, 8 runs | 37.6 minutes in all: BF16 4.0 and 1.4 min (first run cold), emulated FP8 4.1–7.0 min per run |
| BF16 held-out loss | 6.5656 (seed 0), 6.5595 (seed 1); seed std 0.0043; margin 0.0164 nats |
| `fp8-tensorwise` | gap per seed +0.0257, −0.0254; seed-averaged paired gap +0.0001 [−0.0033, +0.0036] |
| `fp8-rowwise` | +0.0202, −0.0344; −0.0071 [−0.0105, −0.0038] |
| `fp8-deepseek` | +0.0338, −0.0192; +0.0073 [+0.0039, +0.0108]; precision log at step 200: relative error 2.5% (activations), 2.6% (weights), 2.3% (gradients), gradient underflow 0.09%, worst gradient layer `layers.3.self_attn.o_proj` |
| decision by the rule | **inconclusive** for all three arms (every arm has a seed whose gap exceeds the 0.0164 margin) |
| emulated speed | BF16 4,548 tokens/s; emulated FP8 1,191–1,526 tokens/s — the cost of emulation, not an FP8 result |
| `compare_fp8.py` | 69 s (evaluation of 8 checkpoints) |
| `frontierlab.record` bf16-s0 vs fp8-deepseek-s0 `--changed precision` | COMPARABLE (only `precision.*` differs) |

How to read it. The averaged gaps are all inside the margin, and two of them have window intervals that exclude zero — but in opposite directions (rowwise lower, DeepSeek higher), and the per-seed gaps are ±0.02–0.03 nats, six times the BF16 seed std. That is the important result: changing the rounding of every GEMM changes the training *trajectory* about as much as changing the seed, so the per-seed differences are dominated by trajectory noise, and the window interval (which resamples windows, not seeds) understates the uncertainty. With two seeds the experiment cannot resolve a 0.016-nat effect. Resolving DeepSeek's 0.25% tolerance at this scale would need many more seeds — the MDE scales with the std of the per-seed gaps (about 0.03) as $2.8 \cdot 0.03 / \sqrt{n}$, so roughly 25 seed pairs — or a larger model and more tokens, where both the noise and the effect may differ. "Inconclusive" here is a sound result, not a failed lab.

<details>
<summary>Hint for TODO 3</summary>

Dgrad contracts over $N$: `tile_qdq(gy)` already groups along the last dim, which is $N$. Wgrad contracts over the tokens $M$: transpose first (`gy.t().contiguous()` is $(N, M)$, `x.t().contiguous()` is $(K, M)$), quantise both with `tile_qdq`, then `gt @ xt.t()`. The weight's 128 × 128 blocks are the same in both orientations, so `block_qdq(w)` serves Dgrad directly.

</details>

<details>
<summary>Hint for TODO 2</summary>

Keep two float64 tensors, `total` and `acc`. For each `k`: `acc = round_mantissa(acc + products[..., k], mant_bits, "trunc")`; when `(k + 1) % n_c == 0`, add `acc` to `total` and reset `acc`. Return `total + acc`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-08/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-08/lesson-02`.

</details>

## Common mistakes

- **Reporting emulated tokens/s as an FP8 result.** Emulation is several times slower than BF16 on any device; it answers "what do the numerics do", never "how fast is FP8".
- **Quantising the Wgrad operands along the wrong axis.** Reusing the forward pass's per-token scales for $x$ in Wgrad gives a GEMM whose scales are not constant along its contraction dimension; a real block-scaled kernel cannot run it, and an emulation that does it measures a recipe nobody can deploy.
- **Leaving the output head or attention in FP8 "because it is a GEMM too".** DeepSeek-V3 keeps them in higher precision; the output head's logits feed a softmax over 32K–128K classes and are sensitive to error.
- **Testing a 0.25% tolerance with unpaired seeds.** The seed-to-seed spread of a small model is larger than the effect. Pair by seed (same initial weights and data order) and by evaluation window.
- **Benchmarking torchao without `torch.compile`.** The casts and amax reductions then run as separate kernels and can erase the gain; compile both arms, warm up, synchronise (lesson 02.4).
- **Assuming a recipe composes with your other tools.** FSDP sharding, activation checkpointing and `torch.compile` each interact with Float8 casts; record what you tested in the run card (Module 9 builds the tested-capability matrix).

## References

- DeepSeek-AI, *DeepSeek-V3 Technical Report*, sections 3.3.1–3.3.3 and appendix B. https://arxiv.org/abs/2412.19437
- PyTorch, torchao 0.18.0 `torchao/float8` (`float8_linear_utils.py`, `config.py`) at tag v0.18.0. https://github.com/pytorch/ao/tree/v0.18.0/torchao/float8
- PyTorch, torchao training workflows documentation at v0.18.0. https://github.com/pytorch/ao/blob/v0.18.0/docs/source/workflows/training.md
- P. Micikevicius et al., *FP8 Formats for Deep Learning*. https://arxiv.org/abs/2209.05433
- NVIDIA, Transformer Engine common API (FP8 recipes). https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html
- NVIDIA, H100 product page (FP8 and BF16 peaks, with sparsity; dense = half). https://www.nvidia.com/en-us/data-center/h100/ ; L4: https://www.nvidia.com/en-us/data-center/l4/
- Shared code: `labs/common/frontierlab/precision/linear.py`, `accum.py`, `cost.py`, `train.py`, `torchao_path.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[08.3 · FP4 training and quantisation-aware training](lesson-03.md)
