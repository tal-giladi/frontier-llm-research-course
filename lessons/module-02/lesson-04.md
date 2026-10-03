---
id: "02.4"
module: 2
minutes: 35
practice_minutes: 75
prerequisites: ["02.3", "01.3", "01.4"]
objectives:
  - Write a benchmark that warms up, synchronises, repeats and reports a median with an interval, and explain what goes wrong without each part.
  - Compare two implementations with interleaved rounds and a paired speed-up interval, at representative shapes.
  - State the caveats of benchmarking torch.compile and CUDA graphs and handle them.
  - Validate one improvement to Baseline-0 end to end with an experiment contract, an equal-work comparison and a gradient-agreement correctness check, and apply a pre-stated decision rule.
volatility: concept
sources:
  - title: "PyTorch 2.14 — CUDA semantics: asynchronous execution"
    url: https://docs.pytorch.org/docs/stable/notes/cuda.html
  - title: "PyTorch 2.14 — torch.utils.benchmark"
    url: https://docs.pytorch.org/docs/stable/benchmark_utils.html
  - title: "PyTorch 2.14 — torch.compile (modes, CUDA graphs in reduce-overhead)"
    url: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
  - title: "Wijmans et al. — Cut Your Losses in Large-Vocabulary Language Models"
    url: https://arxiv.org/abs/2411.09009
  - title: "Hsu et al. — Liger Kernel: Efficient Triton Kernels for LLM Training"
    url: https://arxiv.org/abs/2410.10989
last_verified: "2026-10-03"
---

# 02.4 · Validating a performance claim

"It's 1.8× faster" is the most common unsupported claim in ML engineering. This lesson turns a performance claim into a measurement: warm-up, synchronisation, repeats with an interval, interleaved arms, representative shapes — and then uses that method on one real improvement to Baseline-0, the chunked cross-entropy from lesson 02.2, with an experiment contract, an equal-work comparison and a correctness check that the faster step computes the same loss and gradients.

## Why this matters at a frontier lab

A kernel or recipe change is merged because someone shows a number. If the number came from a cold first call, an unsynchronised GPU timer, a smaller batch, or the best of five runs against a single run, the team ships a regression or wastes a week chasing a phantom. Worse, an "optimisation" that quietly changes the computation — a mask dropped, a reduction done in lower precision, tokens skipped — can be faster and wrong, and nothing in the throughput number shows it. Labs therefore treat performance results like any other experimental result: a question, a pre-stated rule, equal work, a correctness check and uncertainty. Every systems comparison later in this course (MLA vs GQA decode in Module 3, sparse vs dense attention in Module 5, FP8 in Module 8) uses this lesson's method.

## The idea

### Five rules for one timing

1. **Warm up.** First calls pay one-off costs: CUDA context creation, cuBLAS/cuDNN handle setup and algorithm selection, the caching allocator growing its pool, `torch.compile` tracing and code generation, CPU caches filling. Run a few calls untimed.
2. **Synchronise.** CUDA work is asynchronous: a PyTorch call on a GPU tensor queues kernels and returns (PyTorch "CUDA semantics", asynchronous execution). `time.perf_counter()` around it measures the queueing. Call `torch.cuda.synchronize()` before starting and before reading the clock — or record `torch.cuda.Event(enable_timing=True)` pairs and synchronise once at the end.
3. **Repeat, report a median and an interval.** One sample is an anecdote. Take $n \ge 10$–30 samples, report the median with a bootstrap 95% interval (lesson 01.4's method) and the coefficient of variation. Do not report the minimum unless both arms report the minimum and you say why.
4. **Representative shapes.** Intensity depends on shape (lesson 02.1): a kernel that wins at $B \cdot T = 256$ can lose at 8,192. Benchmark at the shapes, dtype and device of the real step, and report the end-to-end step time, not only the op.
5. **Interleave when comparing.** Run A, B, A, B, ... on the same inputs so drift — thermal throttling, clock changes, another tenant on a shared GPU, a background process on your laptop — hits both arms equally. Then compute the speed-up from paired rounds.

### The paired speed-up

With $r$ interleaved rounds and times $a_k$ (baseline) and $b_k$ (new) in round $k$, the per-round ratio is $s_k = a_k / b_k$. Report the median of $s_k$ and a bootstrap interval obtained by resampling rounds. Pairing removes the shared drift from the comparison, as pairing on items did for losses in lesson 01.4.

### Amdahl's law: what an op speed-up is worth

If an op takes a fraction $f$ of the step and becomes $k$ times faster, the step becomes

$$\frac{t_{\text{new}}}{t_{\text{old}}} = (1 - f) + \frac{f}{k}$$

so the end-to-end speed-up is at most $1/(1 - f)$. Always convert an op-level result into a step-level one.

### torch.compile and CUDA graphs: caveats

- The first compiled call can take seconds to minutes (graph capture and code generation); it is warm-up, not step time — but it is real cost for a short run, so report it separately.
- A new input shape or dtype can trigger a recompile; a benchmark with one shape hides the cost of variable shapes the real loop has.
- `mode="reduce-overhead"` (and `"max-autotune"`) replay the step as a CUDA graph. Graphs need static shapes and static memory addresses; they remove launch overhead, so a launch-bound microbenchmark can look much faster than the real loop if the real loop breaks the graph (a `.item()` or a Python branch on a tensor value causes a graph break).
- Compare compiled against eager at the same shapes after both have warmed up, and check the compiled output against eager (same loss within a stated tolerance).
- `torch.utils.benchmark.Timer` handles warm-up and synchronisation for you, but defaults to `num_threads=1` on CPU (checked in torch 2.14.1): set it explicitly or you are measuring a different machine.

### Equal work and correctness

An improvement passes two gates before its speed counts. **Equal work**: both arms process the same tokens, with the same model, batch, sequence length and dtype; if the change saves memory, a second, separate comparison at *equal memory* (each arm at its largest micro-batch) answers a different question and is labelled as such. **Correctness**: on the same inputs, the loss and every parameter gradient agree with the baseline's within a stated tolerance (float64 for the check, where possible), and over a short training run both arms' losses stay within rounding of each other.

## Worked example

### A naive claim, measured properly

The project's debugging script times the plain loss once, cold, and the chunked loss on half the batch, best of five. It prints (measured in this build, CPU) **1.78× faster**. The corrected measurement — same batches, 3 warm-up steps, 30 interleaved rounds, medians with intervals — is `compare_ce.py`, which gives **1.11×, 95% CI [1.05, 1.19]** for the same toy model. Most of the "speed-up" was half the work and a cold baseline.

### Paired speed-up by hand

Four rounds, baseline times $(10, 11, 10, 12)$ ms, new $(8, 9, 8, 9)$ ms. Ratios $s = (1.25, 1.22, 1.25, 1.33)$; median $1.25$. The unpaired ratio of medians, $10.5 / 8.5 = 1.24$, is close here, but if round 4 had been slowed by a background job for *both* arms ($a_4 = 18$, $b_4 = 13.5$), the paired ratio for that round stays $1.33$ while the unpaired medians shift.

### Amdahl for the chunked loss on Baseline-0

From lesson 02.1's per-op model at $B = 8$, $T = 1024$ on an H100, the output head and cross-entropy are about 20% of the step's lower bound. Even if the chunked loss made that part 2× faster, the step would be at most $0.8 + 0.2/2 = 0.9$ of the old time, a 1.11× speed-up. Its real value is memory: at $B = 32$ it frees about 8 GiB of logits (lesson 02.2), which can buy a larger micro-batch — the equal-memory comparison. PROJECTED, pending the pilot.

## Shapes and cost

| Arm | Model | Batch × seq | Logits ever materialised | Extra memory | FLOPs |
|---|---|---|---|---|---|
| plain | Baseline-0 (same initial weights) | 8 × 1024 (main path) | (8, 1024, 32768) fp32 = 1.07 GB, plus the saved log-softmax of the same size | — | baseline |
| chunked | copy of the same weights | same batches, same order | (4096, 32768) fp32 = 0.54 GB per chunk, one at a time | ∂L/∂W buffer (32768, 768) fp32 = 0.10 GB | same GEMMs, done in the forward pass |

Both arms on the same device and dtype (main path: GPU, BF16 autocast with fp32 logits; free CPU: fp32). Correctness check in float64 on the CPU, on a copy of the model.

## Build it

`frontierlab/perf/timing.py` implements the rules:

```python
from frontierlab.perf.timing import benchmark, interleaved, speedup

t = benchmark(step, warmup=3, repeats=20, device="cuda")      # synchronises around every sample
print(t)                                                       # median, 95% CI, n, coefficient of variation
res = interleaved({"plain": step_a, "chunked": step_b}, warmup=3, rounds=30, device="cuda")
print(speedup(res["plain"], res["chunked"]))                   # paired: {'speedup': ..., 'ci': (lo, hi)}
```

`benchmark` calls `sync(device)` (which is `torch.cuda.synchronize` on CUDA and nothing on CPU) before starting and before reading the clock for every sample; warm-up samples are kept separately so you can see what the first calls cost. `interleaved` alternates the arms within each round. `speedup` resamples paired rounds when the arms have the same number of samples.

The end-to-end comparison (`labs/module-02/lesson-04/compare_ce.py`) builds two copies of one initial model, draws each batch once from one generator and feeds it to both arms, times each arm on that batch in turn, and records both arms' losses. Its correctness gate runs your `grad_agreement` in float64 before any timing.

## What the evidence says

- **REASONABLE INDUSTRY PRACTICE.** Warm-up, synchronisation, repeated measurement with intervals and interleaving are standard benchmarking hygiene; the asynchronous-execution behaviour that makes synchronisation necessary is PUBLICLY DOCUMENTED (PyTorch CUDA semantics).
- **ESTABLISHED as a memory technique.** Not materialising logits (Cut Cross-Entropy, Liger Kernel's fused linear cross-entropy). Their speed figures are company claims for their kernels and settings; this course's plain-PyTorch version makes no speed claim beyond what you measure.
- **Measured here, CPU only:** the 1.11× [1.05, 1.19] speed-up and the 0.62 memory ratio below are for the toy model ($C = 128$, $V = 8192$) on a laptop CPU. On the CPU the chunked path also avoids allocating and copying two large fp32 tensors, which is plausibly where its time gain comes from (INFERENCE). On a GPU the time effect may be zero or negative; the main path measures it.

## Lab

### Experiment contract

- **Question:** does replacing Baseline-0's loss with the chunked cross-entropy reduce activation memory without slowing the training step or changing the loss and gradients?
- **Hypothesis and status:** saved activation memory drops by roughly the size of the fp32 logits minus the $V \times C$ buffer; step time changes by less than 5%; established memory mechanism, time effect to be measured.
- **Baseline:** the plain loss (`frontierlab.model.LM` forward), same commit, same initial weights; no tuning involved (the change is exact).
- **Changed variable:** the loss implementation. **Controlled:** initial weights, every batch and its order, batch size, sequence length, dtype, optimizer and learning rate, device, torch version, number of warm-up and timed steps.
- **Comparison axis:** equal work (same tokens per step). Main path adds equal memory (each arm at its largest micro-batch), which answers "what throughput does the saved memory buy", a different question.
- **Budget:** main path 1× H100 SXM or A100, about 15 GPU-minutes; free CPU about 1–3 minutes.
- **Metrics and decision rule:** saved activation bytes (and CUDA peak on GPU); median step time per arm over 30 interleaved rounds after 3 warm-up rounds; paired speed-up with a 95% bootstrap interval. Decision rule, stated before running: **adopt** if the correctness check passes, the speed-up interval's lower end is at least 0.95, and memory is at most 80% of baseline (or the lower end is above 1.0); **reject** if the check fails or the whole interval is below 0.95; otherwise **inconclusive**.
- **Correctness checks:** float64 loss difference and max gradient difference below $10^{-6}$ on one batch (the bound is set by `LM.forward`'s fp32 cast of the logits); largest training-loss difference between arms over the timed steps reported.
- **Fallback evidence:** the pilot's GPU result, labelled as provided, if you have no GPU.
- **Limits:** one model size and shape; the CPU result says nothing about GPU time; memory measured as saved tensors (GPU adds the allocator peak).

**Folder:** [`labs/module-02/lesson-04/`](../../labs/module-02/) · **Time:** about 75 minutes · **Pass check:** `pytest labs/module-02/lesson-04` passes; your report states the decision the rule gives, with its numbers.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB, about 15 GPU-minutes | `compare_ce.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --steps 40 --max-batch --out runs/l24/result.json` (not run in this build; part of the Module 2 pilot) |
| Free GPU (Colab/Kaggle T4) | T4 | `--device cuda --dtype fp32 --preset pilot-10m --vocab 32768 --batch 8 --seq 512 --steps 30 --max-batch`; random tokens unless Data-v0 is prepared at vocab 32768; no BF16 |
| Free CPU | laptop; measured at about 70 seconds on a 16-thread laptop (2026-10-03) | `compare_ce.py` as written (toy model on Data-v0, B = 16, T = 256) |

1. **The harness.** Implement `bench(fn, warmup, repeats, sync)` in `lab.py`. Compare it with `naive_time` on the toy step: the first call is usually the slowest by a wide margin.
2. **The correctness gate.** Implement `grad_agreement(model, idx, loss_a, loss_b)` with `torch.autograd.grad`, so nothing accumulates in `.grad`.
3. **The rule.** Implement `decide(...)` exactly as stated in the contract above. Write the rule in your notes *before* step 4.
4. **Run the comparison.**

   ```bash
   python labs/module-02/lesson-04/compare_ce.py
   ```

5. **Report.** The decision, the numbers behind it, and one sentence each on: why the arms did equal work; what the float64 gate would have caught; whether the speed-up would survive at Baseline-0's width (use Amdahl and lesson 02.1's op shares).
6. **Main path only.** Read part 5 of the output (largest micro-batch per arm and tokens/s there) and state the equal-memory result separately from the equal-work one.

Measured in this build (free CPU, Windows 11, 16 threads, torch 2.14.1+cpu, fp32, toy model on Data-v0, B = 16, T = 256, V = 8192, 3 warm-up and 30 timed interleaved rounds; other processes running): float64 gate |loss diff| $3.9 \times 10^{-7}$, max |grad diff| $1.7 \times 10^{-8}$ (PASS; the size of these is the fp32 cast inside `LM.forward`); saved activations 326.6 → 203.0 MiB (ratio 0.62); plain median 799 ms [757, 831], chunked 711 ms [680, 766]; paired speed-up 1.11, 95% CI [1.05, 1.19]; largest training-loss difference between arms over 30 steps $9.5 \times 10^{-7}$. Decision by the rule: **adopt** (for this CPU setting). The main-path decision waits for the pilot.

<details>
<summary>Hint for step 1</summary>

Run `fn()` `warmup` times, then for each repeat: `sync()`, read `time.perf_counter()`, `fn()`, `sync()`, read the clock again. The test checks that a sync happens between every two timed calls.

</details>

<details>
<summary>Hint for step 2</summary>

`params = [p for p in model.parameters() if p.requires_grad]`; `ga = torch.autograd.grad(loss_a(model, idx), params)`; same for `b`; then the largest `(a - b).abs().max()` over the pairs.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-02/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-02/lesson-04`.

</details>

## Common mistakes

- **Timing the first call.** It measures initialisation. The project's debugging script does this and inflates its claim.
- **Comparing best-of-n against a single run,** or minimum against median. Use the same statistic for both arms.
- **Different work per arm.** A smaller batch, a shorter sequence, a skipped mask or a skipped synchronisation in one arm makes any speed-up meaningless.
- **Benchmarking a microkernel at toy shapes and claiming a step speed-up.** Apply Amdahl at the real shapes.
- **Benchmarking compiled code without warming it up, or with a shape the real loop never uses.**
- **Skipping the correctness gate because "the loss looks the same".** A 1% gradient error does not show in a loss curve for hundreds of steps; a float64 comparison shows it at once.
- **Reading an overlapping interval as a win.** If the rule says inconclusive, the result is inconclusive; run more rounds or report it as not shown.

## References

- PyTorch 2.14 documentation: [CUDA semantics — asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html), [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html), [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html); `Timer`'s `num_threads=1` default and the compile modes checked against the installed torch 2.14.1.
- E. Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*. https://arxiv.org/abs/2411.09009
- P.-L. Hsu et al., *Liger Kernel*. https://arxiv.org/abs/2410.10989
- The experiment contract template: [templates/experiment-contract.md](../../templates/experiment-contract.md); shared code `labs/common/frontierlab/perf/timing.py`; versions in [references/versions.md](../../references/versions.md).

## Next

The module project: [Performance report on Baseline-0](../../projects/module-02-performance-report.md). Module 3 then asks how attention should spend KV memory, and measures its decode costs with this lesson's method.
