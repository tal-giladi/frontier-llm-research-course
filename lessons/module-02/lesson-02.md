---
id: "02.2"
module: 2
minutes: 40
practice_minutes: 75
prerequisites: ["02.1"]
objectives:
  - Record a training step with torch.profiler and read kernel time, CPU time, launch overhead and idle gaps from the summary and the timeline.
  - Name the fusion opportunities in Baseline-0's step and say what torch.compile changes and what it cannot change.
  - Measure activation memory by what autograd saves, account for it tensor by tensor, and take a CUDA memory snapshot.
  - Implement a chunked cross-entropy with hand-derived gradients and show it removes the (B, T, V) logits from saved memory with identical gradients.
  - Choose between activation checkpointing and a smaller batch from their measured memory and time costs.
volatility: implementation
sources:
  - title: "PyTorch 2.14 — torch.profiler"
    url: https://docs.pytorch.org/docs/stable/profiler.html
  - title: "PyTorch 2.14 — Understanding CUDA Memory Usage (memory._record_memory_history, _dump_snapshot, pytorch.org/memory_viz)"
    url: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
  - title: "PyTorch 2.14 — torch.utils.checkpoint"
    url: https://docs.pytorch.org/docs/stable/checkpoint.html
  - title: "PyTorch 2.14 — torch.compile"
    url: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
  - title: "Korthikanti et al. — Reducing Activation Recomputation in Large Transformer Models (section 4.1: activation memory per layer)"
    url: https://arxiv.org/abs/2205.05198
  - title: "Wijmans et al. — Cut Your Losses in Large-Vocabulary Language Models (Cut Cross-Entropy)"
    url: https://arxiv.org/abs/2411.09009
  - title: "Hsu et al. — Liger Kernel: Efficient Triton Kernels for LLM Training"
    url: https://arxiv.org/abs/2410.10989
last_verified: "2026-10-03"
---

# 02.2 · Profiling a training step

Lesson 02.1 predicted how long Baseline-0's step should take. This lesson finds out where the time and the memory actually go: a profiler timeline shows which kernels ran, how long the CPU took to launch them and where the GPU sat idle, and an account of the tensors autograd saves shows where the memory went — including the single largest tensor in the step, the logits, and a way to never build it.

## Why this matters at a frontier lab

Two numbers decide most of a run's speed: how busy the GPU is and how large a batch fits. A profile answers the first; it is the only honest way to say "this step is launch-bound" or "attention is 40% of the step" instead of guessing. Memory answers the second: the micro-batch size you can fit sets the GEMM shapes (and so their efficiency, lesson 02.1) and how much gradient accumulation you need. Every Stage B comparison of attention designs and every Stage C precision change is a claim about time or memory, so it starts with a profile of the baseline taken the same way.

## The idea

### What a profiler records

`torch.profiler.profile` records two streams. On the **CPU** side: every operator the dispatcher runs (`aten::mm`, `aten::mul`, ...), with its start and end time and its parent (a `Linear` calls `aten::linear`, which calls `aten::addmm`). On the **GPU** side (with `ProfilerActivity.CUDA`): every kernel, with the time it actually ran. Three readings matter:

- **Self vs total time.** An operator's *self* time excludes its children, so self times add up to the step; *total* time includes them. Sort by self time to find what to fix.
- **Kernel time vs CPU time.** On a GPU the CPU only *queues* kernels; each launch costs a few microseconds of CPU work. If the kernels are shorter than their launch cost, the GPU waits for the CPU: the step is **launch-bound**, and the timeline shows gaps between kernels. `frontierlab.perf.profiling.summarize` reports the GPU's busy fraction inside the step window; well below 1 means idle time.
- **Gaps the CPU causes.** A `.item()`, a `print` of a GPU tensor, or a data loader that is not ready makes the CPU wait for the GPU (a synchronisation) and then the GPU wait for the CPU. They show up as holes in the GPU row of the timeline. (The course loop calls `out.loss.item()` once per micro-batch; Module 2's proposed loop changes move that to logging steps only.)

### Fusion and torch.compile

Lesson 02.1 found 0.146 s of memory-bound work per Baseline-0 step: norms, RoPE, SiLU·mul, residual adds, each a separate kernel that reads its inputs from HBM and writes its output back. RMSNorm alone is several kernels in eager PyTorch (`pow`, `mean`, `add`, `rsqrt`, `mul`, `mul`, plus dtype casts), each a full pass over the tensor. **Fusion** does the whole chain in one kernel that reads once and writes once, cutting bytes by the number of passes it replaces.

`torch.compile(model)` does this automatically. TorchDynamo traces the Python code into a graph; TorchInductor generates fused kernels for it (Triton on NVIDIA GPUs, C++ on CPU). Modes: `"default"`; `"reduce-overhead"`, which also replays the step as a CUDA graph to remove per-kernel launch cost; `"max-autotune"`, which also searches GEMM configurations (and uses CUDA graphs). What it cannot change: the GEMM FLOPs (already near their floor) and the bytes a fused kernel still has to move. The first call pays for tracing and code generation (seconds to minutes); a change of input shape can trigger a recompile. Measure compiled code only after warm-up (lesson 02.4).

### Activation memory: what autograd saves

To compute gradients, autograd keeps some forward tensors alive until the backward pass uses them: a `Linear` saves its input, SiLU saves its input, the elementwise product saves both factors, attention saves q, k, v, its output and a log-sum-exp per row. These saved tensors are the step's **activation memory**. `frontierlab.perf.memory.SavedTensors` installs `torch.autograd.graph.saved_tensors_hooks` and records every saved storage once, on CPU or GPU, so you can account for it exactly. Counted per token per layer for Baseline-0 (measured, fp32):

$$\text{saved values per token per layer} = \underbrace{4 I}_{\text{gate, SiLU(gate), up, product}} + \underbrace{6 C}_{\text{norm in/out, residual stream}} + \underbrace{4 H d}_{\text{q, attention out}} + \underbrace{4 K d}_{\text{k, v}}$$

The larger published formula for GPT-3-style layers, $sbh\,(34 + 5as/h)$ bytes in 16-bit (Korthikanti et al., section 4.1), includes dropout masks, a GeLU MLP and the $s \times s$ attention matrix that a fused attention kernel no longer stores.

### The logits hotspot and a chunked cross-entropy

The output head turns $N = B \cdot T$ hidden vectors into $N \times V$ logits. `frontierlab.model.LM` casts them to fp32 and calls `F.cross_entropy`, whose `log_softmax` saves its $N \times V$ fp32 output for backward; the returned logits tensor is a second $N \times V$ fp32 tensor that stays alive while the loop holds the model output. The gradient of the mean cross-entropy with respect to the logits $z_i$ of row $i$ has a closed form:

$$\frac{\partial L}{\partial z_i} = \frac{\text{softmax}(z_i) - \text{onehot}(y_i)}{N}$$

So the loss and all gradients can be produced $n$ rows at a time inside the forward pass: compute the chunk's logits $z = h_c W^\top$, its loss, $g = \partial L / \partial z$, then $\partial L/\partial h_c = g W$ and $\partial L / \partial W \mathrel{+}= g^\top h_c$, and drop the chunk. Only an $n \times V$ block ever exists. The cost is a fixed $V \times C$ fp32 buffer for $\partial L/\partial W$, so it pays when $N \gg C$. This is the idea behind fused linear cross-entropy kernels (Liger Kernel) and Cut Cross-Entropy (Wijmans et al.), here in plain PyTorch.

### Activation checkpointing

`torch.utils.checkpoint.checkpoint(block, x, ...)` saves only the block's input and recomputes the block's forward during backward. Memory per layer drops from the formula above to $C$ values per token (plus one block's internals during its recompute). The price is one extra forward pass per checkpointed block: training goes from about 3 to about 4 forward-equivalents, roughly +33% compute, which is exactly the HFU-vs-MFU gap of lesson 02.1. Checkpointing everything is rarely optimal; checkpoint only enough blocks to fit the batch you want, or (Korthikanti et al.) only the cheap-to-recompute, memory-heavy parts.

## Worked example

### Cross-entropy gradient by hand

One row, $V = 3$, logits $z = (2, 0, 1)$, target $y = 0$. $e^{z} = (7.389, 1.000, 2.718)$, sum 11.107, softmax $p = (0.665, 0.090, 0.245)$. Loss $= \log 11.107 - z_0 = 2.408 - 2 = 0.408$. Gradient $= p - \text{onehot}(0) = (-0.335, 0.090, 0.245)$; it sums to 0, as it must. With $N$ rows each row's gradient is divided by $N$.

### Baseline-0's activation memory, by hand

Per token per layer: $4 \cdot 2816 + 6 \cdot 768 + 4 \cdot 768 + 4 \cdot 256 = 11{,}264 + 4{,}608 + 3{,}072 + 1{,}024 = 19{,}968$ values, 79.9 KB in fp32. For one 1,024-token sequence and 12 layers: $19{,}968 \cdot 4 \cdot 1024 \cdot 12 = 936$ MiB. Logits saved by `log_softmax`: $1023 \cdot 32768 \cdot 4 = 128$ MiB.

Measured in this build with `SavedTensors` (CPU, fp32, B = 1, T = 1024, full Baseline-0): 1,080 MiB saved in total, of which 128 MiB is the logits and the rest matches the formula plus the embedding output, final norm and head input. Under CPU bf16 autocast the same pass saved 1,041 MiB: GEMM inputs in bf16, RMSNorm and RoPE internals in fp32, the logits in fp32, plus about 230 MiB of bf16 weight copies that autocast makes once per forward.

Scaled to the main-path micro-batch $B = 32$ (activation memory is linear in $B$; PROJECTED from the B = 1 measurement): about 25 GiB of saved activations under bf16 autocast, of which the saved logits are 4.0 GiB, plus another 4.0 GiB for the returned logits. The logits are the largest single tensor and, with the returned copy, about a quarter of activation memory — not the majority, which is the 12 layers' MLP activations.

### What chunking buys, and when

The chunked loss removes the $N \times V$ fp32 saved logits and adds the $V \times C$ fp32 gradient buffer: $32768 \cdot 768 \cdot 4 = 96$ MiB. At $N = 1023$ the saving is $128 - 96 = 32$ MiB (measured: 1,080 → 1,048 MiB in fp32); at $N = 32 \cdot 1023$ it is $4{,}092 - 96 \approx 4{,}000$ MiB, before counting the returned logits the chunked path also never creates.

## Shapes and cost

| Tensor | Shape | dtype | Device | Lives until |
|---|---|---|---|---|
| hidden states before the head | (B, T, 768) | fp32 residual stream; the head saves a bf16 copy under autocast | GPU | backward of the head |
| logits | (B, T, 32768) | fp32 | GPU | the loop drops `out` |
| saved log-softmax | (B·(T−1), 32768) | fp32 | GPU | backward of the loss |
| chunk logits (chunked loss) | (n, 32768), n = 4096 | fp32 | GPU | the next chunk |
| ∂L/∂W buffer (chunked loss) | (32768, 768) | fp32 | GPU | backward |
| checkpointed block input | (B, T, 768) | fp32 (the residual stream stays fp32 under autocast) | GPU | that block's recompute |

Cost: the chunked loss does the same GEMMs as the ordinary forward plus backward (2·N·V·C each for $z$, $\partial L/\partial h$, $\partial L/\partial W$), just in the forward pass; it adds no FLOPs. Checkpointing adds one block forward per block (about $2N_{\text{layer}}$ FLOPs per token per layer).

## Build it

```python
import torch
from frontierlab.model import LM, toy
from frontierlab.perf.memory import saved_activation_bytes
from frontierlab.perf.chunked_ce import lm_loss_chunked
from frontierlab.perf.profiling import profile_steps, summarize, format_summary

model = LM(toy(vocab_size=8192)); x = torch.randint(0, 8192, (16, 256))
st = saved_activation_bytes(model, lambda: model(x, labels=x).loss)
print(st.total_bytes / 2**20, st.largest(1))      # 326.6 MiB, the (4080, 8192) fp32 log-softmax first
print(saved_activation_bytes(model, lambda: lm_loss_chunked(model, x, 1024)).total_bytes / 2**20)   # 203.0

prof = profile_steps(lambda: model(x, labels=x).loss.backward(), steps=2)
print(format_summary(summarize(prof)))
```

On a GPU, add a memory snapshot of two steps and a timeline:

```python
from frontierlab.perf.memory import memory_snapshot
with memory_snapshot("runs/l22/mem.pickle"):          # open at https://pytorch.org/memory_viz
    for _ in range(2):
        step()
prof = profile_steps(step, steps=3, device="cuda", trace_path="runs/l22/trace.json")   # https://ui.perfetto.dev
```

`memory_snapshot` wraps `torch.cuda.memory._record_memory_history(max_entries=...)` and `torch.cuda.memory._dump_snapshot(path)`, then stops recording with `_record_memory_history(enabled=None)`. These are underscore functions: documented, but their arguments can change between releases (checked against torch 2.14.1).

Correctness of the chunked loss is a float64 test, not a hope: `labs/common/tests/test_perf.py` checks the loss and the gradients of $h$ and $W$ against `F.cross_entropy` to $10^{-12}$ for chunk sizes 1, 8, 37 and 100 (with an ignored position), and `lm_loss_chunked` against the model's own loss and every parameter gradient.

## What the evidence says

- **ESTABLISHED.** Activation recomputation as a memory–compute trade (Korthikanti et al. report a 5× activation-memory reduction from sequence parallelism plus selective recomputation and over 90% less recomputation overhead than full checkpointing for their setup; abstract and section 4). Fusion of memory-bound elementwise chains (the reason torch.compile, Liger Kernel and FlashAttention exist).
- **ESTABLISHED as a technique; speed figures are company claims.** Not materialising the logits: Cut Cross-Entropy reports the loss computation's memory for Gemma 2 (2B) going from 24 GB to 1 MB (abstract); Liger Kernel reports on average 20% higher throughput and 60% lower memory against Hugging Face implementations (abstract), across all its kernels, not the loss alone.
- **Measured here, CPU only.** The activation accounting above. The time effects of fusion and compile on a GPU are PROJECTED until the Module 2 pilot; on this build's Windows CPU, `torch.compile` failed with "InvalidCxxCompiler: cl is not found" (Inductor needs a C++ compiler; on Linux it uses gcc).

## Lab

### Experiment contract

- **Question:** for Baseline-0's step, how much activation memory do (a) checkpointing every block, (b) the chunked loss, (c) both save, and what does each cost in step time?
- **Hypothesis and status:** (a) saves most of the layer activations at about +25–35% step time; (b) saves the logits at no FLOP cost and little time cost; established mechanisms, magnitudes to be measured here.
- **Baseline:** the plain loss on the same model, batch and inputs.
- **Changed variable:** the loss/recompute variant. **Controlled:** model weights (same object), inputs, batch, sequence, dtype, device.
- **Comparison axis:** equal work (same tokens, same model, same gradients).
- **Budget:** main path 1× H100 SXM or A100, about 10 GPU-minutes; free CPU about 2 minutes.
- **Metrics and decision rule:** saved activation bytes (exact, no noise); step time as the median of interleaved rounds with a 95% interval. Decision rule: adopt the chunked loss as the default for later modules if its gradients match (test passes) and its step-time interval does not lie entirely above the baseline's by more than 5%; use checkpointing only where a run would otherwise not fit.
- **Correctness checks:** `pytest labs/module-02/lesson-02` (chunked gradients and checkpointed gradients equal the reference); `labs/common/tests/test_perf.py`.
- **Fallback evidence:** the pilot's GPU memory snapshot, labelled as provided.
- **Limits:** one model size; CPU times are not GPU times; memory measured as saved tensors, not allocator peak (the GPU snapshot shows the peak).

**Folder:** [`labs/module-02/lesson-02/`](../../labs/module-02/) · **Time:** about 75 minutes · **Pass check:** `pytest labs/module-02/lesson-02` passes; you can account for the largest saved tensors by name.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB, about 10 GPU-minutes | `profile_step.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --trace runs/l22/trace.json --snapshot runs/l22/mem.pickle --compile` (not run in this build; part of the Module 2 pilot) |
| Free GPU (Colab/Kaggle T4) | T4 | same with `--dtype fp32 --preset pilot-10m --batch 8 --seq 512`; snapshot and trace work, BF16 rates and FlashAttention-2 kernels do not apply on a T4 |
| Free CPU | laptop; measured at about 2 minutes on a 16-thread laptop (2026-10-03) | `profile_step.py` as written (toy model, B = 16, T = 256, V = 8192); no GPU timeline, no snapshot, `--compile` needs a C++ compiler |

1. **Chunked loss by hand.** Implement `ce_and_grads(h, w, targets, chunk_size)` in `lab.py`: loss and both gradients, chunk by chunk, without autograd. The test compares it with autograd in float64.
2. **Checkpointing.** Implement `checkpointed_loss(model, idx)` with `torch.utils.checkpoint.checkpoint(block, x, positions, None, use_reentrant=False)` around each block. The tests check that the loss and gradients are unchanged and the saved bytes drop by more than half.
3. **Categorise.** Implement `categorize(top)`, which sums the profiler's time shares into matmul / attention / loss / optimizer / other.
4. **Profile and account.**

   ```bash
   python labs/module-02/lesson-02/profile_step.py
   ```

   Read part 1 (top operators, categories), part 2 (per-op overhead × ops per step) and part 3 (saved bytes and step time per variant). Then answer in writing: which saved tensor is largest and why; what fraction of saved memory the logits are at this $V/C$ ratio, and what it would be at Baseline-0's; whether the step is launch-bound on your machine.
5. **On a GPU (main path).** Open the snapshot at https://pytorch.org/memory_viz and find the logits allocation and the moment it is freed; open the trace at https://ui.perfetto.dev and find the largest gap in the GPU row and the CPU call that caused it.

Measured in this build (free CPU, Windows 11, 16 threads, torch 2.14.1+cpu, fp32, toy model B = 16, T = 256, V = 8192; the machine had background processes, so times are noisy): 4,312 aten operators per training step; matmuls 42% of self CPU time in the top 15 rows. Saved activations: plain 326.6 MiB (of which the fp32 logits 127.5 MiB), checkpointed 141.6 MiB, chunked 203.0 MiB, both 18.0 MiB. Step time medians over 5 interleaved rounds: plain 2.22 s, checkpointed 2.50 s, chunked 2.32 s, both 2.52 s, with 95% intervals 0.5–1.2 s wide, so this run cannot rank the times; the memory numbers are exact. A tiny op cost 16.7 µs from Python, about 72 ms (3%) of the step for 4,312 ops — this CPU step is not launch-bound; a GPU step with the same op count and much faster kernels can be.

<details>
<summary>Hint for step 1</summary>

For a chunk with logits `z` of shape `(n, V)`: `lse = torch.logsumexp(z, -1)`; the summed loss is `(lse - z.gather(1, t[:, None]).squeeze(1)).sum()`. For the gradient start from `g = torch.softmax(z, -1)`, subtract 1 at each row's target column, and divide by the total N (not by the chunk size).

</details>

<details>
<summary>Hint for step 2</summary>

Copy the body of `LM.forward` (positions, `embed_tokens`, the loop over `model.model.layers`, `model.model.norm`, `model.lm_head`, `.float()`, cross-entropy of `logits[:, :-1]` against `idx[:, 1:]`) and replace `layer(x, positions, None)` with the checkpoint call. `use_reentrant=False` is the recommended mode; it works with keyword arguments and with inputs that do not require grad.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-02/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-02/lesson-02`.

</details>

## Common mistakes

- **Profiling the first steps.** They include allocator growth, cuBLAS initialisation and (with compile) code generation. `profile_steps` warms up outside the profiler.
- **Reading total time as self time.** `aten::linear`'s total includes `aten::addmm`; adding totals double-counts.
- **Calling GPU time "CPU time".** On a GPU the CPU column is launch work. A step can show 50 ms of CPU time and 30 ms of kernel time and still be GPU-idle 40% of the step.
- **Forgetting the returned logits.** Even with a chunked loss, code that also returns full logits (for logging or evaluation) brings the tensor back. Return the loss only in training.
- **Using `torch.cuda.memory_allocated` after the step.** Activations are freed by then; the peak is what limits the batch. Use the snapshot or `max_memory_allocated` after `reset_peak_memory_stats`.
- **Checkpointing with `use_reentrant=True` by habit.** The reentrant variant needs at least one input with `requires_grad` and does not support some patterns; PyTorch recommends `use_reentrant=False`.

## References

- PyTorch 2.14 documentation: [torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html), [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html), [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html), [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html). API names and signatures checked against the installed torch 2.14.1.
- V. Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, abstract and section 4.1. https://arxiv.org/abs/2205.05198
- E. Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, abstract. https://arxiv.org/abs/2411.09009
- P.-L. Hsu et al., *Liger Kernel: Efficient Triton Kernels for LLM Training*, abstract. https://arxiv.org/abs/2410.10989
- Shared code: `labs/common/frontierlab/perf/` (`profiling.py`, `memory.py`, `chunked_ce.py`); versions in [references/versions.md](../../references/versions.md).

## Next

[02.3 · Communication and overlap](lesson-03.md)
