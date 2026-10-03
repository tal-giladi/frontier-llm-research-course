---
id: "02.1"
module: 2
minutes: 35
practice_minutes: 60
prerequisites: ["01.1"]
objectives:
  - Compute the arithmetic intensity of a GEMM, RMSNorm, softmax and decode attention by hand and say whether each is compute-bound or memory-bound on a named GPU.
  - Predict Baseline-0's step time from its FLOPs, a datasheet peak and an assumed MFU, and bound it from below with a per-op roofline model, before measuring anything.
  - Measure a machine's empirical roofline and one training step, and compute the implied MFU.
  - Distinguish MFU from HFU and say which one a recomputation-heavy run should report.
volatility: concept
sources:
  - title: "NVIDIA H100 Tensor Core GPU — specifications (H100 SXM: BF16 1,979 TFLOPS with sparsity, 3.35 TB/s)"
    url: https://www.nvidia.com/en-us/data-center/h100/
  - title: "NVIDIA A100 Tensor Core GPU — specifications (BF16 312 TFLOPS dense; 80GB SXM 2,039 GB/s)"
    url: https://www.nvidia.com/en-us/data-center/a100/
  - title: "NVIDIA L4 Tensor Core GPU — specifications (BF16 242 TFLOPS with sparsity; 300 GB/s)"
    url: https://www.nvidia.com/en-us/data-center/l4/
  - title: "NVIDIA T4 Tensor Core GPU — specifications (65 FP16 TFLOPS; 320+ GB/s)"
    url: https://www.nvidia.com/en-us/data-center/tesla-t4/
  - title: "Williams, Waterman, Patterson — Roofline: An Insightful Visual Performance Model (UC Berkeley EECS-2008-134; CACM 52(4), 2009)"
    url: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf
  - title: "Chowdhery et al. — PaLM: Scaling Language Modeling with Pathways (section 4.1, Table 3, Appendix B: MFU vs HFU)"
    url: https://arxiv.org/abs/2204.02311
  - title: "Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    url: https://arxiv.org/abs/2205.14135
last_verified: "2026-10-03"
---

# 02.1 · From FLOPs to time

A FLOP count says how much arithmetic a training step needs; it does not say how long the step takes. This lesson adds the second half — bytes moved and the memory bandwidth that moves them — so you can predict Baseline-0's step time on a named GPU before you measure it, and know which part of the gap between prediction and measurement is physics and which part is your code.

## Why this matters at a frontier lab

Every systems decision in this course is a comparison of times: MLA against GQA at decode, a linear-attention hybrid against FlashAttention at 128K, FP8 against BF16, one parallel layout against another. A measured time means little on its own. "The new kernel takes 1.9 ms" becomes useful only next to "and the roofline says it cannot take less than 1.2 ms", because then you know whether there is 40% left to win or nothing. Labs do this arithmetic before they rent the cluster: a projected step time times the step count is the GPU-hours line in the budget of every experiment contract you will write. If the prediction is off by 3×, the budget is off by 3×.

## The idea

### Two ceilings

A GPU does two things for every operation: arithmetic on its tensor cores, and moving data between its high-bandwidth memory (HBM) and the chip. Each has a ceiling. Symbols:

- $F$ — FLOPs the operation performs (one multiply-add = 2 FLOPs).
- $Q$ — bytes it must move between HBM and the chip, if each input is read once and each output written once.
- $P$ — the GPU's peak FLOP/s at the dtype in use.
- $W$ — the GPU's memory bandwidth in bytes/s.

The operation cannot finish faster than either ceiling allows:

$$t_{\min} = \max\left(\frac{F}{P},\ \frac{Q}{W}\right)$$

The **arithmetic intensity** is $I = F/Q$, FLOPs per byte. The **ridge point** $P/W$ is the intensity at which both terms are equal. An op with $I$ below the ridge is **memory-bound**: the arithmetic units wait for data, and only moving fewer bytes makes it faster. An op above the ridge is **compute-bound**. Plotting attainable FLOP/s, $\min(P,\ I \cdot W)$, against $I$ on log axes gives the "roofline" (Williams, Waterman and Patterson, 2009): a slanted roof for memory-bound ops meeting a flat roof at the ridge.

### The numbers for the GPUs this course uses

PUBLICLY DOCUMENTED (vendor datasheets, checked 2026-10-03). NVIDIA lists tensor-core FLOPs "with sparsity" (2:4 structured sparsity, which ordinary training does not use); the dense number is half.

| SKU | Dense peak $P$ | Bandwidth $W$ | Ridge $P/W$ |
|---|---|---|---|
| H100 SXM5 80 GB | 989 TFLOP/s BF16 (datasheet: 1,979 with sparsity) | 3.35 TB/s HBM3 | 295 FLOP/B |
| A100 SXM4 80 GB | 312 TFLOP/s BF16 | 2,039 GB/s HBM2e | 153 FLOP/B |
| L4 24 GB | 121 TFLOP/s BF16 (datasheet: 242 with sparsity) | 300 GB/s GDDR6 | 403 FLOP/B |
| T4 16 GB | 65 TFLOP/s FP16 (no BF16 tensor cores) | 320 GB/s GDDR6 | 203 FLOP/B |

Check the exact SKU you rent: an "H100" can be the SXM part above, a PCIe part or an NVL part, with different peaks and bandwidths. `frontierlab.perf.roofline.HARDWARE` holds these four entries with their source URLs.

### MFU and HFU

**Model FLOPs utilization (MFU)** is the FLOPs the model *needs* per second, divided by the peak: tokens per second × training FLOPs per token ÷ $P$. **Hardware FLOPs utilization (HFU)** counts the FLOPs the hardware actually *executes*, including recomputation. The PaLM report (section 4.1 and Appendix B) introduced MFU because HFU rewards wasted work: PaLM 540B reports 46.2% MFU and 57.8% HFU, the difference being rematerialised activations. With full activation checkpointing (lesson 02.2) the forward pass runs twice, so the hardware does about 4 forward-equivalents instead of 3 and $\text{HFU} \approx \tfrac{4}{3}\,\text{MFU}$. Report MFU when comparing runs; it is what you pay for.

## Worked example

### Intensity by hand, tiny numbers

A $4 \times 4$ by $4 \times 4$ GEMM in BF16 (2 bytes per element): $F = 2 \cdot 4 \cdot 4 \cdot 4 = 128$; $Q = 2 \cdot (16 + 16 + 16) = 96$ bytes; $I = 128 / 96 = 1.33$ FLOP/B. On an H100 (ridge 295) that is deeply memory-bound — tiny matrix multiplies are.

### The four ops of the lesson, at Baseline-0 shapes on an H100 SXM

Baseline-0: $C = 768$, $I_{\text{mlp}} = 2816$, $H = 12$ query heads, $K = 4$ key/value heads, $d = 64$, $L = 12$. BF16 throughout.

1. **GEMM, training.** The up-projection on a micro-batch of $B \cdot T = 8 \cdot 1024 = 8192$ tokens: $(8192 \times 768) \cdot (768 \times 2816)$. $F = 2 \cdot 8192 \cdot 2816 \cdot 768 = 35.4$ GFLOP. $Q = 2 \cdot (8192 \cdot 768 + 768 \cdot 2816 + 8192 \cdot 2816) = 63.0$ MB. $I = 562$ > 295: compute-bound, $t_{\min} = 35.4 \times 10^9 / 989 \times 10^{12} = 35.8$ µs.
2. **The same GEMM at decode.** One token: $M = 1$. $F = 4.33$ MFLOP, $Q = 4.33$ MB (almost all of it the weight matrix), $I = 1.0$: memory-bound, $t_{\min} = 4.33 \times 10^6 / 3.35 \times 10^{12} = 1.3$ µs. Same weights, same code, the other side of the ridge. This is why decoding is batched.
3. **RMSNorm** on $8192 \times 768$: about 4 FLOPs per element (square, sum, scale by $1/\text{rms}$, multiply by the gain), and each element is read once and written once: $F = 25.2$ MFLOP, $Q = 25.2$ MB, $I = 1.0$. Memory-bound, $t_{\min} = 7.5$ µs — a fifth of the GEMM's time for 0.07% of its FLOPs.
4. **Softmax** over unfused attention scores, $B \cdot H = 96$ rows of $T = 1024$ per query, i.e. a $98{,}304 \times 1024$ matrix: about 5 FLOPs per element, $I = 5/4 = 1.25$. $Q = 403$ MB, $t_{\min} = 120$ µs per layer. FlashAttention (Dao et al., 2022) exists because this matrix never needs to reach HBM at all.
5. **Attention at decode.** One new query per sequence against $S$ cached tokens: $F = 4 \cdot H \cdot S \cdot d$ (query-key scores and the weighted sum of values), $Q = 2 \cdot K \cdot S \cdot d \cdot 2$ bytes (reading the K and V caches). So $I = \frac{4 H S d}{4 K S d} = H/K = 3$ FLOP/B, whatever $S$ and the batch are. At $S = 32{,}768$: $Q = 33.6$ MB per layer per sequence, $t_{\min} = 10.0$ µs per layer, 120 µs per token for 12 layers. Grouped-query attention raises intensity by the group size $H/K$; that, not FLOPs, is why KV layout matters (Module 3).

### Predicting a Baseline-0 step before measuring

The main-path step from lesson 01.1: $B = 32$, $T = 1024$, gradient accumulation 8, so 262,144 tokens per optimizer step, at 788.1 MFLOP of training per token. $F_{\text{step}} = 2.066 \times 10^{14}$ FLOPs.

$$t_{\text{step}} = \frac{F_{\text{step}}}{P \cdot \text{MFU}}: \quad \frac{2.066 \times 10^{14}}{989 \times 10^{12} \cdot 1.0} = 0.209\ \text{s}, \quad \text{at 40\%: } 0.52\ \text{s}, \quad \text{at 30\%: } 0.70\ \text{s}$$

That prediction needs an MFU you have not measured. The roofline gives a floor that needs no guess: add up $\max(F/P, Q/W)$ for every op of the step. `frontierlab.perf.roofline.step_time_model` does this (forward ops listed explicitly, backward taken as 2× forward FLOPs and bytes, AdamW reading and writing 16 bytes per parameter). For the step above on an H100 SXM:

| Part | Lower bound |
|---|---|
| GEMMs, attention, output head (compute-bound) | 0.201 s |
| norms, RoPE, SiLU·mul, residual adds, cross-entropy (memory-bound, unfused) | 0.146 s |
| AdamW update | 0.0006 s |
| **total** | **0.347 s** |

So even perfect kernels, if every elementwise op is a separate kernel reading and writing HBM, cap MFU at $0.209 / 0.347 = 60\%$. The 0.146 s of memory-bound work is the target of kernel fusion and `torch.compile` (lesson 02.2). It is large here because Baseline-0 is narrow: GEMM FLOPs grow with $C^2$ per token while elementwise bytes grow with $C$, so at $C = 768$ the memory-bound share is far higher than at a frontier model's $C \approx 7{,}000$. All of these figures are PROJECTED (formula above, datasheet peaks), pending the Module 2 pilot.

## Shapes and cost

| Tensor (one Baseline-0 layer, micro-batch 8 × 1024) | Shape | dtype | Bytes | Op that reads it |
|---|---|---|---|---|
| hidden states | (8, 1024, 768) | bf16 | 12.6 MB | RMSNorm, q/k/v projections |
| gate / up activations | (8, 1024, 2816) | bf16 | 46.1 MB each | SiLU·mul (memory-bound) |
| q after RoPE | (8, 12, 1024, 64) | bf16 | 12.6 MB | attention |
| K/V cache at decode, S = 32,768 | (1, 4, 32768, 64) × 2 | bf16 | 33.6 MB per layer | decode attention (I = 3) |
| logits | (8, 1024, 32768) | fp32 | 1.07 GB | cross-entropy (memory-bound) |

All on the GPU on the main path; on the free CPU path the same shapes are fp32 on the CPU, with the bandwidth of DDR memory (tens of GB/s) instead of HBM (TB/s).

## Build it

The roofline helpers live in `frontierlab/perf/roofline.py`:

```python
from frontierlab.model import baseline0
from frontierlab.perf.roofline import HARDWARE, gemm, decode_attention, step_time_model, predicted_step_time

h100 = HARDWARE["H100-SXM"]
op = gemm(8192, 2816, 768)                         # up_proj on 8 x 1024 tokens, bf16
print(op.intensity, op.bound(h100), op.time(h100))   # 562.0 compute 3.58e-05
print(decode_attention(1, 12, 4, 32768, 64).intensity)   # 3.0
m = step_time_model(baseline0(), 32, 1024, h100, grad_accum=8)
print(round(m["t_total"], 3))                                       # 0.347
print(round(predicted_step_time(baseline0(), 32, 1024, h100, mfu=0.4, grad_accum=8), 3))   # 0.522
```

`step_time_model` returns the forward op list too (`m["ops"]`), so you can see which op the bound is made of. Its tests (`labs/common/tests/test_perf.py`) check the worked numbers above and that the model's compute part never exceeds the FLOPs-only prediction at MFU 1.

What the model leaves out, on purpose: kernel launch overhead (lesson 02.2), communication (02.3), the data loader, Python, and kernels that reach only part of either ceiling. A measured step slower than the model is normal; a measured step *faster* than the model means the model double-counts something (for example, an op the compiler fused away) or the measurement is wrong (no synchronisation — lesson 02.4).

## What the evidence says

- **ESTABLISHED.** The roofline model (Williams et al., 2009) and its use for GPU kernels; the IO-aware argument for attention (FlashAttention, Dao et al., 2022). The datasheet peaks are PUBLICLY DOCUMENTED (company claim); sustained GEMM throughput is lower and depends on shape, clocks and power limits.
- **ESTABLISHED.** MFU as the comparable efficiency metric (PaLM section 4.1, Appendix B). Published MFU values for large dense runs are in the 40–55% range (PaLM 46.2%); they are not targets for a 122M model, which has narrow GEMMs and a large memory-bound share.
- **INFERENCE.** The 60% cap and the 0.146 s memory-bound share come from this course's per-op model, which counts every elementwise op as unfused. A fused or compiled step moves fewer bytes; the pilot measures how many.

## Lab

### Experiment contract

- **Question:** how far is a training step's measured time from the time predicted from FLOPs and the roofline, on this machine, and which ops explain the gap?
- **Hypothesis and status:** the measured step is slower than the FLOPs-only lower bound by a factor of 2–5 for Baseline-0 on one GPU (reported effect for small models; the exact factor is what the pilot measures).
- **Baseline:** the prediction, made and written down *before* the timing runs: `predict_step_time` at MFU 1.0 and 0.4, and `step_time_model`.
- **Changed variable:** none; this is prediction against measurement. **Controlled:** model preset, batch, sequence length, dtype, device, torch version.
- **Comparison axis:** equal work (the same step, same tokens); the question is time per step, not quality.
- **Budget:** main path 1× H100 SXM (or the A100 of the Colab pilot), under 10 GPU-minutes; free CPU about 1–2 minutes of compute.
- **Metrics and decision rule:** median step time over 10 timed steps after 3 warm-up steps, with a 95% bootstrap interval; implied MFU = predicted time at MFU 1 ÷ measured median. Decision rule: if the implied MFU is below 25% on a GPU, the step is profiled in lesson 02.2 before any later systems comparison uses it.
- **Correctness checks:** the lab tests pass (your cost functions equal the reference); timings synchronise the device before reading the clock.
- **Fallback evidence:** the Module 2 pilot's measurement on A100, labelled as such, if you have no GPU.
- **Limits:** one GPU, one shape; CPU results describe the CPU's roofline only.

**Folder:** [`labs/module-02/lesson-01/`](../../labs/module-02/) · **Time:** about 60 minutes · **Pass check:** `pytest labs/module-02/lesson-01` passes, and you have written your prediction down before running step 3.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM (or A100 80 GB), about 5 GPU-minutes | `measure.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --hw H100-SXM` (not run in this build; part of the Module 2 pilot) |
| Free GPU (Colab/Kaggle T4) | T4 | same with `--dtype fp32 --preset pilot-10m --batch 8 --seq 512 --hw T4`; you will not see BF16 tensor-core rates, and fp32 on a T4 runs on CUDA cores (8.1 TFLOP/s datasheet) |
| Free CPU | laptop; measured at about 40 seconds on a 16-thread laptop (2026-10-03) | `measure.py` as written (toy model, fp32) |

1. **Cost functions.** In `lab.py`, implement `gemm_cost`, `rmsnorm_cost`, `softmax_cost` and `decode_attention_cost` from the formulas above, then `roofline_time` and `predict_step_time`. `pytest labs/module-02/lesson-01` checks them against `frontierlab.perf.roofline`.
2. **Predict on paper.** For the main-path command, write down the predicted step time at MFU 1.0 and 0.4, and `step_time_model(...)["t_total"]` for $B = 8$, $T = 1024$, no accumulation. (At $B=8$ on an H100: 6.5 ms at MFU 1.0, 16 ms at MFU 0.4; the per-op bound is 11.4 ms.)
3. **Measure.**

   ```bash
   python labs/module-02/lesson-01/measure.py
   ```

   Part 1 measures this machine's empirical roofline (best GEMM and a 256 MiB copy). Part 2 prints each op's intensity, roofline time and measured time. Part 3 prints the step prediction, then the measurement and the implied MFU.
4. **Explain the gap.** For each op in part 2 whose measured/roofline ratio is above 2, write one sentence saying why (launch overhead, a kernel that reaches only part of the bandwidth, reduction passes the model did not count). For any ratio *below* 1, explain how the measurement beat the "ceiling" (hint: what does a CPU's cache do to a weight matrix that is read 10 times in a row?).

Measured in this build (free CPU, Windows 11, 16 threads, torch 2.14.1+cpu, fp32; the machine had other processes running, which shows in the intervals): best GEMM 0.151 TFLOP/s, copy 19.4 GB/s, ridge 7.8 FLOP/B. Toy model at $B = 16$, $T = 256$: predicted 320 ms at MFU 1, measured median 1,525 ms (95% CI 985–2,030 ms), implied MFU 21%. The decode-shaped GEMM measured 0.4× its roofline time because its 8.6 MB weight matrix stayed in the CPU cache between repeats. These CPU numbers do not transfer to any GPU: the ridge point differs by a factor of 40.

<details>
<summary>Hint for step 1</summary>

Every cost function returns `(flops, bytes)`. Bytes count each input read once and each output written once: a GEMM reads $MK + KN$ elements and writes $MN$. For decode attention count only the K and V caches.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-02/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-02/lesson-01`.

</details>

## Common mistakes

- **Using the sparsity number.** NVIDIA's headline BF16 figure for the H100 SXM (1,979 TFLOP/s) assumes 2:4 sparsity. Dense training gets at most 989. Using the wrong one halves every MFU you report.
- **Calling a 1-token GEMM compute-bound.** Intensity depends on shape, not on the op's name. The same weight matrix is compute-bound at 8,192 tokens and memory-bound at 1.
- **Including the T × T score matrix in attention bytes when a fused kernel is used.** With FlashAttention it never reaches HBM; counting it makes attention look memory-bound when it is not.
- **Reporting HFU as MFU.** With activation checkpointing the hardware executes about a third more FLOPs than the model needs. HFU looks better and means less.
- **Timing a GPU without synchronising.** `time.perf_counter()` around queued kernels measures the queueing. Lesson 02.4 makes this a rule; `frontierlab.perf.timing.benchmark` already follows it.

## References

- NVIDIA datasheets (checked 2026-10-03): [H100](https://www.nvidia.com/en-us/data-center/h100/), [A100](https://www.nvidia.com/en-us/data-center/a100/), [L4](https://www.nvidia.com/en-us/data-center/l4/), [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/).
- S. Williams, A. Waterman, D. Patterson, *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*, UC Berkeley EECS-2008-134 (published in CACM 52(4), 2009). https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf
- A. Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, section 4.1, Table 3 and Appendix B. https://arxiv.org/abs/2204.02311
- T. Dao et al., *FlashAttention*, abstract and section 3. https://arxiv.org/abs/2205.14135
- Lesson [01.1](../module-01/lesson-01.md) for Baseline-0's FLOPs per token; software versions in [references/versions.md](../../references/versions.md).

## Next

[02.2 · Profiling a training step](lesson-02.md)
