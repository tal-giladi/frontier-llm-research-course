---
id: "05.3"
module: 5
minutes: 40
practice_minutes: 90
prerequisites: ["05.1", "05.2", "04.1", "02.1", "02.4"]
objectives:
  - Profile dense attention, a linear (KDA) layer and DSA-style selection component by component (indexing, top-k, gather, selected attention, recurrent step) for prefill and decode, with warm-up, synchronisation, repeats and paired intervals.
  - Fit growth exponents and locate measured crossover contexts, and explain why they move with hardware, kernel, batch and attention shape.
  - Project main-path component times with the roofline model and identify, for each design, whether compute, memory traffic or kernel launches bound it.
  - Evaluate the Module 5 arms on Eval Suite v1 with paired comparisons, and state MiniMax-M2's argument as a testable counter-hypothesis together with the evaluation power needed to test it.
volatility: implementation
sources:
  - title: "Why Did MiniMax M2 End Up as a Full Attention Model? (MiniMax, Hugging Face blog, 2025-10-30)"
    url: https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
  - title: "DeepSeek-V3.2 (section 2.1: complexity of DSA; section 2.2 and Figure 3: inference cost on H800; masked MHA mode for short prefill)"
    url: https://arxiv.org/abs/2512.02556
  - title: "Kimi Linear (decoding throughput claim at 1M context)"
    url: https://arxiv.org/abs/2510.26692
  - title: "Native Sparse Attention (section 3: group-centric data loading, shared KV fetching)"
    url: https://arxiv.org/abs/2502.11089
  - title: "PyTorch 2.14 documentation: torch.profiler and torch.nn.functional.scaled_dot_product_attention"
    url: https://pytorch.org/docs/stable/profiler.html
last_verified: "2026-10-04"
---

# 05.3 · Measuring cost and quality honestly

A sub-quadratic design is worth it only at contexts where it is actually faster and still good enough. Both halves are easy to get wrong: a FLOP count is not a time, a time on one kernel is not a time on another, and a quality match on short benchmarks is not a match where it matters. This lesson measures the cost of dense attention, a linear layer and DSA-style selection component by component, across contexts, for prefill and decode, finds where each crosses dense attention, projects the components to an H100 with the roofline model, and then evaluates the trained arms of 05.1 and 05.2 on Eval Suite v1 — with MiniMax's argument for full attention as the hypothesis you try to refute.

## Why this matters at a frontier lab

The three lab reports this module uses all make cost claims at long context: Kimi Linear "up to 6× decoding throughput" at 1M tokens, Qwen3-Next "10 times inference throughput for context over 32K tokens", DeepSeek-V3.2 a cost-per-token curve that separates from V3.1's as the position grows. Each is measured on the lab's own kernels and serving stack. MiniMax, with its own stack, concluded that the infrastructure for linear attention was not mature enough and that quality deficits appeared only at scale. Your team's decision depends on *your* crossover: your context distribution, your batch sizes, your hardware, and the kernels you can actually run and debug. Measuring it honestly means separating the components (so you know which one to fix), measuring prefill and decode separately (they have different bottlenecks), and refusing to call a kernel-specific number a property of the method.

## The idea

### What to measure, and why separately

For one attention layer's mixing step (projections cost the same $2N$ FLOPs per token in every design and are measured once, separately):

| Arm | Components | What bounds it |
|---|---|---|
| dense | fused SDPA (FlashAttention on CUDA) | compute in prefill ($\propto L^2$); reading the KV cache in decode ($\propto S$) |
| linear (KDA) | chunked core in prefill; recurrent step in decode | chunk-level matmuls and the sequential chunk loop; the state read in decode (constant) |
| DSA | indexer scores; top-k; gather of the selected entries; attention over them | indexer compute ($\propto L^2$, small constant); top-k memory ($\propto L^2$ in prefill); gather traffic ($\propto L k$) |

Each component is timed alone with the Module 2 harness (`frontierlab.perf.timing.benchmark`: warm-up, synchronise, repeats, bootstrap interval of the median), and the three whole arms are timed in interleaved rounds (`interleaved`) so drift hits all of them, with paired speed-ups against dense (lesson 02.4). Op counts come from `torch.profiler`: aten ops on CPU, kernel launches on CUDA. A design whose step is twenty launches of 3 µs kernels is launch-bound whatever its FLOPs.

### Growth exponents and crossovers

If time grows as $t \approx a L^p$, the slope of $\log t$ against $\log L$ is $p$: about 2 for dense prefill at long context, about 1 for the linear core, 0 for a decode step that reads a fixed state. Fitted slopes below the model's value mean fixed costs still dominate at the measured contexts. A **measured crossover** is the context where the new arm's time falls below dense, interpolated in log-log between the two measured points that bracket it. It is a property of the whole setup, not of the method. It moves:

- with **kernels** — our PyTorch linear core loops over $T/C$ chunks in Python and our DSA gathers with generic indexing; fla's Triton kernel and DeepSeek's sparse kernels fuse these;
- with **attention shape** — a gathered entry serves all query heads that share it. In MQA mode (V3.2: 128 heads share one latent) one gathered entry feeds $2 \cdot 128 \cdot 1{,}088 = 278{,}528$ FLOPs; with Baseline-0's GQA (3 query heads per KV head) a 256-byte entry feeds 768 FLOPs. The first is compute-bound, the second memory-bound;
- with **batch size and phase** — decode at batch 1 reads the whole cache for one token; prefill reuses each key for a whole block of queries;
- with **hardware** — CPU timings never transfer to GPUs (Module 2).

### Projection to the main path

Each component's minimum time on an H100 is the roofline bound $\max(F/(P \cdot u), Q/(W \cdot \eta))$ with $F$ FLOPs, $Q$ minimum bytes, $P = 989$ TFLOP/s, $W = 3.35$ TB/s, and assumed efficiencies $u$ and $\eta$ (lesson 02.1). Launch overheads (a few µs per kernel) are not in the bound and dominate small decode steps. All such numbers are PROJECTED until the pilot measures them.

### Quality: the MiniMax counter-hypothesis

MiniMax's account (company claim): a lightning-attention hybrid "looked just as good as pure full attention" on benchmarks such as MMLU and BBH, yet "at larger scale" showed "clear deficits in complex, multi-hop reasoning tasks"; it adds that "the better the models get, the harder they are to evaluate". Stated as a prediction Eval v1 can test:

> On paired items, a hybrid's evidence effect on **retrieval** (one hop) and its natural-text loss match dense attention's, while its evidence effect on **two-hop** chains is lower.

The test is the paired difference (arm − dense) of the per-item evidence effect, pooled over cells, for one hop and for two hops. A refutation needs both: no difference on retrieval *and* an interval on the two-hop difference that excludes the size of deficit that would matter. A null result with a wide interval refutes nothing; it says the evaluation is not powerful enough at this scale, which is MiniMax's second point.

## Worked example

### One decode step at 128K, projected

Baseline-0's layer shape (12 query heads, 4 KV heads, $d = 64$), BF16, an indexer with 4 heads of width 32, $k = 2{,}048$, H100 SXM with $u = 0.4$, $\eta = 0.7$ (`profile_attn.py` uses the same model):

| Component | FLOPs | Minimum bytes | Bound | Projected |
|---|---|---|---|---|
| dense: read 131,072 cached K and V | $4 \cdot 12 \cdot 64 \cdot 131{,}072 = 4.0 \times 10^8$ | $2 \cdot 4 \cdot 64 \cdot 131{,}072 \cdot 2 = 128$ MiB | memory | 57 µs |
| DSA indexer: score 131,072 indexer keys | $4 \cdot 66 \cdot 131{,}072 = 3.5 \times 10^7$ | 8 MiB | memory | 3.6 µs |
| DSA top-k over 131,072 fp32 scores | — | 0.5 MiB | memory | 0.2 µs |
| DSA gather 2,048 entries (read and write) | — | 4 MiB | memory | 1.8 µs |
| DSA attention over 2,048 | $6.3 \times 10^6$ | 2 MiB | memory | 0.9 µs |
| KDA recurrent step | $3.4 \times 10^5$ | 0.38 MiB of fp32 state | memory | 0.17 µs |

DSA's sum is 6.5 µs against 57 µs: $8.8\times$ less per layer — if the four components cost what their bytes cost. At 8K the same arithmetic gives 2.9 µs against 3.6 µs, and four or more kernel launches of a few µs each erase the difference. That is the honest shape of the claim: a large decode win at long context, nothing at short context unless the components are fused.

### Prefill at 128K, projected

Dense causal prefill does $4 \cdot 12 \cdot 64 \cdot 131{,}072^2/2 = 2.6 \times 10^{13}$ FLOPs per layer: 67 ms at 40% of peak. DSA's indexer does $264 \cdot 131{,}072^2/2 = 2.3 \times 10^{12}$ (5.7 ms), but each of the 131,072 queries reads its own 2,048 selected entries: $131{,}072 \cdot 2{,}048 \cdot 1{,}024$ bytes $\approx 254$ GiB per layer through the attention, about 116 ms at 70% of bandwidth even if no copy is written. With GQA-shaped entries the sparse prefill is slower than dense FlashAttention at 128K. Two published remedies: share the selected entries across many query heads (V3.2's MQA mode, where each entry feeds 128 heads and the step becomes compute-bound), or share them across neighbouring queries (NSA's blockwise selection with group-centric loading); and V3.2 runs short prefills in "a masked MHA mode to simulate DSA" anyway. The linear core at 128K: $80$ GFLOP and 1.5 GiB, about 0.7 ms.

### How many items does the MiniMax test need?

Suppose the per-item paired difference in evidence effect (arm − dense, nats) has standard deviation $\sigma = 0.3$ and the deficit that would matter is $\delta = 0.05$ nats. For 80% power at $\alpha = 0.05$:

$$n = \left(\frac{(1.96 + 0.84)\,\sigma}{\delta}\right)^2 = \left(\frac{2.8 \cdot 0.3}{0.05}\right)^2 = 282 \text{ items}.$$

The CPU default (40 items per cell, 80 two-hop items over two lengths) can detect only $\delta \approx 2.8 \cdot 0.3/\sqrt{80} = 0.094$ nats. Plan the item count from the effect you care about, before the run.

## Shapes and cost

| Tensor (profiling, B = 1) | Shape | dtype / device |
|---|---|---|
| dense q, k, v | (1, H, T, d), (1, KV, L, d) | fp32 CPU / bf16 CUDA |
| linear q, k, v, g; state | (1, H, T, d); (1, H, d, d) | same; state fp32 |
| indexer q, k, w | (1, $H^I$, T, $d^I$), (1, L, $d^I$), (1, T, $H^I$) | same |
| score block | (1, block, L) | fp32 |
| top-k indices per block | (1, block, k) | int64 |
| gathered K and V per block | (1, KV, block, k, d) | same as k |

The profiler processes queries in blocks of 512 so nothing of size $L \times L$ is ever held; peak memory of the DSA prefill is the gathered block, $2 \cdot \text{KV} \cdot 512 \cdot k \cdot d$ elements (128 MiB in fp32 at the CPU defaults, 1 GiB in BF16 at Baseline-0 shape with $k = 2{,}048$ in BF16).

## Build it

`labs/common/frontierlab/attention/subq_bench.py` holds the components (`dense_sdpa`, `linear_core`, `dsa_indexer`, `dsa_topk`, `dsa_gather`, `dsa_attend`), their FLOP and byte models (`component_flops_bytes`) and `measure(shape, L, decode=...)`, which times each component, the three whole arms interleaved, and their op counts. The blocked DSA components equal the full-mask reference to $10^{-15}$ in float64 for prefill and decode (`test_attention_m05.py`), so the thing being timed is the thing that was tested. On CUDA, `linear_mode="fla"` times fla's `chunk_kda`; dense uses PyTorch's SDPA, which dispatches to its FlashAttention kernel for BF16 inputs; DSA stays our PyTorch gather path, because no pinned sparse kernel for this shape exists in the course stack. Main-path DSA timings are therefore component costs of *our* path, and end-to-end sparse speed is labelled not measured.

Two lab scripts drive it: `profile_attn.py` (sweep, exponents, crossovers, projection; writes JSON for the memo) and `quality.py` (Eval v1 on every trained arm, pairs, and the MiniMax table).

## What the evidence says

- **Component profiling and paired, interleaved timing: ESTABLISHED** practice (Module 2).
- **Linear attention's long-context decode advantage: ESTABLISHED** as arithmetic (constant state), PROMISING as measured end to end in serving stacks (Kimi Linear, Qwen3-Next; company claims on their kernels).
- **DSA's long-context advantage:** PUBLICLY DOCUMENTED cost curves on H800 at a stated rental price (company claim); the kernels are DeepSeek's own. That the gather is efficient because entries are shared across 128 heads in MQA mode is INFERENCE from the config and the roofline, consistent with the report's choice of MQA mode.
- **MiniMax-M2's argument:** company claim, no ablation data released; it is the counter-hypothesis, not a result. The infrastructure points (low-precision state, prefix caching, speculative decoding) are about serving stacks at that time and are worth re-checking against your stack's current version.
- **At course scale.** CPU timings of PyTorch reference code measure that code on that CPU. The quality runs are 1.8M-parameter models at 256 tokens: their Eval v1 results are hypotheses checks, and nothing here bears on reasoning at scale.

## Lab

**Folder:** [`labs/module-05/lesson-03/`](../../labs/module-05/) · **Time:** about 90 minutes (about 15 of them unattended) · **Pass check:** `pytest labs/module-05/lesson-03` passes; `profile_attn.py` prints every component with an interval and the exponent and crossover lines; `quality.py` prints the MiniMax table; your notes apply both decision rules.

### Experiment contract

- **Question:** at which context, on this hardware with these kernels, does each sub-quadratic arm become cheaper than dense attention for prefill and for decode, which component bounds it there, and do the trained arms of 05.1 and 05.2 lose quality on Eval v1 against their dense controls, in particular on multi-hop items? Decision informed: the module memo's recommendation for a stated context, hardware and quality bar.
- **Hypotheses and status:** (1) the measured exponents approach 2 (dense prefill and DSA indexer), 1 (linear core, DSA attention) and 0 (linear decode step) as the context grows — established arithmetic, may be masked by fixed costs at CPU sizes; (2) the linear decode step is faster than dense decode at every measured context, and the DSA decode arm crosses dense at some context — expected; (3) on CPU, DSA prefill with our gather path never crosses dense — expected from the byte count; (4) MiniMax: the hybrid matches dense on retrieval and natural-text loss but has a lower two-hop evidence effect — company claim, very likely undetectable at toy scale.
- **Baseline:** dense SDPA with the same layer shape and inputs (cost); `b0-s0` and `control` (quality).
- **Changed variable:** the attention mechanism. **Controlled:** layer shape (pilot-10m: 6 heads, 2 KV heads, $d = 64$; indexer 4 × 32; $k = 256$; chunk 64), inputs and seed, dtype, thread count, block size, repeats and rounds; for quality, the trained runs' own controls and Eval v1's pins.
- **Comparison axis:** equal layer shape at equal context (cost); equal parameters / equal tokens as in 05.1 and 05.2 (quality).
- **Budget:** free CPU about 4 minutes of profiling and 10 minutes of evaluation (measured below); main path about 1 GPU-hour, PROJECTED.
- **Metrics and decision rules:** median time with 95% CI per component and arm; paired speed-up CI per arm; exponents; crossovers. Rule (cost): an arm "is cheaper at L" when its paired speed-up CI against dense lies above 1 at L. Rule (quality, MiniMax test): "deficit shown" if the CI of the two-hop evidence-effect difference lies below 0; "no deficit larger than δ" only if the CI lies above −δ with δ = 0.05 nats; otherwise "not resolved at this power".
- **Correctness checks:** blocked DSA components equal the reference (`test_attention_m05.py`); every arm passed its suite in 05.1 / 05.2.
- **Fallback evidence:** the Module 5 pilot's GPU component traces at 4K–32K, labelled as provided.
- **Limits:** one CPU, PyTorch reference kernels, batch 1, one layer shape; tiny models, 256-token training, 40 items per cell.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB, about 1 GPU-hour (PROJECTED: the sweep is a few minutes per context up to 128K, Eval v1 at 1K–16K about 10 minutes per model). Not run in this build; part of the Module 5 pilot | `python labs/module-05/lesson-03/profile_attn.py --device cuda --dtype bf16 --shape baseline0 --contexts 8192 16384 32768 65536 131072 --topk 2048 --chunk 64 --linear-mode fla --out runs/m05/l53/main-prefill.json`, the same with `--mode decode --out runs/m05/l53/main-decode.json`, then `python labs/module-05/lesson-03/quality.py --variant main --device cuda --bf16 --lengths 1024 4096 16384` |
| Free GPU (Colab/Kaggle T4) | T4, fp32, about 30 minutes (PROJECTED) | `profile_attn.py --device cuda --contexts 2048 4096 8192 16384 32768` (our chunked linear core; no fla on T4 in the pinned stack), both modes |
| Free CPU | laptop; measured below | the steps below |

### Steps

1. **Implement** `fit_exponent`, `crossover`, `project_time` and `decide` in `lab.py`; run `pytest labs/module-05/lesson-03`.
2. **Profile** prefill and decode (close other programs; the timings use every thread):

   ```bash
   python labs/module-05/lesson-03/profile_attn.py --out runs/m05/l53/cpu-prefill.json
   python labs/module-05/lesson-03/profile_attn.py --mode decode --contexts 1024 4096 16384 65536 --out runs/m05/l53/cpu-decode.json
   ```

3. **Evaluate quality** (needs the runs of 05.1 and 05.2):

   ```bash
   python labs/module-05/lesson-03/quality.py
   ```

4. **Report** (one page): for each arm and phase, the fitted exponent, the measured crossover (or "none in range") and the component that dominates at the largest context; the projected H100 table at 8K and 128K and which component bounds each arm; the quality table with the decision for the MiniMax test, and the number of items a decisive test would need (worked example); and one paragraph on what in these measurements would change on an H100 with fused kernels.

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, fp32, 16 threads, pilot-10m layer shape, $k = 256$, chunk 64, 7 repeats and 7 interleaved rounds; another build job was using the CPU). Runtimes: prefill sweep 3.2 minutes, decode sweep 19 seconds, `quality.py` 10.3 minutes for six models.

*Prefill, whole arms (median ms, paired speed-up against dense with 95% CI):*

| L | dense | linear | DSA | linear vs dense | DSA vs dense |
|---|---|---|---|---|---|
| 512 | 5.7 | 124 | 68 | 0.045 [0.021, 0.051] | 0.084 [0.077, 0.108] |
| 2,048 | 34.7 | 527 | 307 | 0.066 [0.062, 0.079] | 0.113 [0.086, 0.126] |
| 8,192 | 501 | 2,384 | 2,189 | 0.226 [0.157, 0.239] | 0.240 [0.216, 0.258] |

Fitted exponents: dense 1.68 (its SDPA component 1.72), linear 1.08, DSA 1.25; the DSA indexer 2.06, top-k 1.60, gather 1.09, selected attention 1.09 — each close to its cost model once the context is large enough. No crossover with dense within 8K: both arms are slower at every measured length, but their ratio to dense improves fourfold from 512 to 8K, as the exponents say it must. Op counts per call: dense 19, DSA 186–3,060 (it grows with the number of query blocks), linear 1,867–29,707 (the Python chunk loop). At 8K the DSA arm's time splits into indexer 552 ms, top-k 331, gather 488 and attention 192: the $O(L^2)$ indexer is already its largest component, before any FP8.

*Decode, one token at context L:*

| L | dense | linear | DSA | linear vs dense | DSA vs dense |
|---|---|---|---|---|---|
| 1,024 | 0.12 | 0.32 | 0.60 | 0.36 [0.32, 0.52] | 0.16 [0.10, 0.27] |
| 4,096 | 0.46 | 0.33 | 1.40 | 0.86 [0.68, 1.37] | 0.39 [0.12, 0.54] |
| 16,384 | 1.61 | 1.20 | 1.74 | 1.34 [0.67, 2.36] | 0.93 [0.66, 1.02] |
| 65,536 | 7.54 | 0.44 | 3.25 | 16.3 [7.7, 19.5] | 2.32 [1.56, 3.09] |

Exponents: dense 0.98, linear step 0.02 (constant, as it must be), DSA 0.38 (its indexer and top-k grow, its gather and attention stay flat at 256 entries). Interpolated crossovers of the medians: linear about 2.9K tokens, DSA about 18K. By the contract's rule (paired CI above 1) the linear arm is cheaper only at 64K here (the 4K and 16K intervals include 1: interference from the other job widened them) and DSA only at 64K. The H100 projection puts dense decode at 14.3 µs at 64K against 2.1 µs for the DSA components and 0.1 µs for the linear step, before launch overheads of a few µs per kernel.

*Quality (Eval v1 at 256 and 512 tokens, pins `eval-v1.0`, 100 documents and 40 items per cell; differences arm − dense, paired):* natural-text loss over position buckets: `hybrid-kda` $+0.185$ $[+0.174, +0.198]$ at 256 (the sigmoid-gate arm of 05.1), `hybrid-gdn` $-0.024$ $[-0.036, -0.011]$, `hybrid-kda-silu` $-0.026$ $[-0.038, -0.015]$, DSA `sparse` against `control` $+0.0005$ $[-0.0008, +0.0018]$; at 512 (twice the trained length) the same signs. The MiniMax test: the dense models' own evidence effects are tiny (retrieval $+0.002$, two-hop $+0.001$ nats for `b0`; $+0.021$ and $+0.009$ for `control`), and every two-hop difference lies within $\pm 0.012$ (for example `hybrid-kda-silu` $+0.002$ $[-0.002, +0.006]$, DSA $-0.003$ $[-0.012, +0.005]$). By the rule's letter that is "no deficit larger than δ = 0.05", but the honest reading is different: these 1.8M-parameter models barely use evidence at all, so there is no headroom in which a hybrid *could* show a deficit. A test whose baseline is at the floor cannot refute MiniMax's claim; it needs models that use the evidence (the main path's 30M–100M models at 1K–16K, and more items).

<details>
<summary>Hint for TODO 2</summary>

Work with $r_i = \log(t_{\text{new},i}/t_{\text{base},i})$. The first index with $r_i < 0$ brackets the crossover with index $i - 1$; interpolate linearly in $\log L$: $x = x_{i-1} + (x_i - x_{i-1}) \cdot r_{i-1}/(r_{i-1} - r_i)$.

</details>

<details>
<summary>Hint for step 4</summary>

Look at the op counts: the linear arm's thousands of aten ops per call on CPU are the Python chunk loop and small einsums, which a fused kernel removes. Then look at the bytes column of `dsa.gather` and `dsa.attend` in prefill: per query, $k$ entries are read with no reuse across queries.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-05/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-05/lesson-03`.

</details>

## Common mistakes

- **Comparing FLOPs and calling it speed.** DSA's prefill does fewer FLOPs than dense at 32K and can still be slower: its gather is memory-bound.
- **One number per arm.** Prefill and decode have different bottlenecks and different crossovers; report both.
- **Timing without synchronisation or warm-up on CUDA,** or timing arms in separate runs instead of interleaved rounds.
- **Generalising from reference kernels.** Our Python chunk loop is launch-bound; fla's kernel is not. Say which kernel each number comes from.
- **Reading a null quality result as parity.** Compute what the evaluation could detect; MiniMax's point is exactly that deficits appear where evaluations are weakest.
- **Forgetting projections and the indexer in the total.** Mixing cost is only part of a layer; at short context the projections dominate every design.

## References

- MiniMax, *Why Did MiniMax M2 End Up as a Full Attention Model?*, 2025-10-30. https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
- DeepSeek-AI, *DeepSeek-V3.2*, sections 2.1–2.2 and Figure 3. https://arxiv.org/abs/2512.02556
- Moonshot AI, *Kimi Linear*. https://arxiv.org/abs/2510.26692
- J. Yuan et al., *Native Sparse Attention*, section 3 (kernel design). https://arxiv.org/abs/2502.11089
- PyTorch documentation: `torch.profiler`, `scaled_dot_product_attention`. https://pytorch.org/docs/stable/profiler.html
- Shared code: `labs/common/frontierlab/attention/subq_bench.py`, `frontierlab/perf/timing.py`, `frontierlab/evals/suite_v1.py`.

## Next

[05.4 · Compressed sparse attention (DeepSeek-V4 CSA/HCA)](lesson-04.md) (extension), then the [Module 5 project](../../projects/module-05-subquadratic-memo.md).
