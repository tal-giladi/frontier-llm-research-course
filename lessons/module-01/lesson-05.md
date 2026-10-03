---
id: "01.5"
module: 1
minutes: 30
practice_minutes: 75
prerequisites: ["01.1", "01.3", "01.4"]
objectives:
  - List what must be identical for two runs to be comparable (code, config, data hashes, evaluation windows, seeds, software, hardware where time is compared) and check it from run cards.
  - Use a run-card diff to separate the declared change from bookkeeping, replicates and differences that invalidate a comparison.
  - Explain the main sources of nondeterminism in training and inference, including batch-size-dependent reductions, and say which ones a seed does not control.
  - Demonstrate a bitwise-reproducible CPU rerun and find the one setting that breaks it.
volatility: implementation
sources:
  - title: "Horace He and Thinking Machines Lab — Defeating Nondeterminism in LLM Inference (2025-09-10): lack of batch invariance; 80 unique completions of 1000 at temperature 0 vs 1000 identical with batch-invariant kernels"
    url: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  - title: "PyTorch 2.14 documentation — Reproducibility (notes/randomness)"
    url: https://docs.pytorch.org/docs/stable/notes/randomness.html
  - title: "PyTorch — torch.use_deterministic_algorithms (CUBLAS_WORKSPACE_CONFIG values)"
    url: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
  - title: "FineWeb-Edu dataset card (Data-v0 source, pinned revision)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
last_verified: "2026-10-03"
---

# 01.5 · Reproducibility and the experiment record

An experiment is only as good as its record. If you cannot say exactly which code, data, configuration, seeds and software produced a number, you cannot compare it with anything, and nobody can check it. This lesson defines the record the course uses for every run (the run card), builds a tool that compares two run cards and says which differences invalidate a comparison, and looks at the part a seed does not control: floating-point arithmetic whose result depends on thread counts, kernel choices and the size of the batch a row happens to be in.

## Why this matters at a frontier lab

Large-lab experiments are long, shared and resumed. A run is started by one person, preempted, resumed on different nodes, evaluated by another person's script and compared months later with a run on a newer software stack. Each of those steps can silently change something: a re-prepared validation shard, a library upgrade that changes a kernel, a learning rate that someone "fixed" in one arm. A comparison that survives review is one where every difference between the runs is either the declared change or has been shown not to matter. The run card is how you show it, and a diff tool is how you check it in seconds instead of by reading two YAML files side by side.

## The idea

### What must match

For two runs to be comparable, everything that can change the result must be equal except the variable under test:

| Must match | Where it is recorded | Why |
|---|---|---|
| Data: dataset, revision, tokenizer, file hashes | `data`, `data_files` (copied from Data-v0 `meta.json`) | a different shard is a different experiment |
| Evaluation: split, window count, window seed, sequence length | `args.eval_windows`, `args.seq`; windows are fixed by `TokenData.eval_windows` (seed 1234) | scores must be on the same tokens to be paired |
| Configuration and every training argument | `config`, `args` | each difference is a changed variable |
| Seeds (unless seeds are the replicate axis) | `args.seed`, `args.data_seed` | init and data order |
| Budget on the comparison axis | `budget.tokens` / `budget.train_flops`, measured hours | lesson 01.3 |
| Code version | `git.commit`, `git.dirty` | a dirty tree means the code is not recoverable |
| Software (and hardware, for wall-clock) | `hardware.torch`, `hardware.cuda`, `hardware.gpu` | kernels and their numerics change across versions |

Some differences never matter: the run name, logging cadence, where the checkpoint was stopped and resumed (exact resume makes it invisible, lesson 01.1). The record has to make both kinds visible so a reviewer can tell them apart.

### Data hashes

Data-v0's `meta.json` (written by `frontierlab/data/prepare.py`) records the dataset name, the pinned FineWeb-Edu revision, the split rule, and a SHA-256 of every file: the tokenizer, each split's `.bin` token file and its document-offset file. A SHA-256 is a 256-bit fingerprint of the bytes; any change to any byte changes it, and two different files with the same hash are, for practical purposes, impossible. So "same data" becomes a string comparison: `data_files.val.bin_sha256` equal in both run cards. The training loop copies these fields into every run card automatically.

### Nondeterminism: what a seed does not control

A seed fixes the random numbers: initial weights, data order, dropout masks, sampling. It does not fix the order in which floating-point numbers are added, and floating-point addition is not associative:

$$(0.1 + 0.2) + 0.3 = 0.6000000000000001 \neq 0.6 = 0.1 + (0.2 + 0.3) \quad \text{(float64)}$$

Any reduction — a sum in a matrix multiply, a norm, a softmax denominator, a gradient all-reduce — can be computed in different orders, and each order gives slightly different bits. Sources of different orders:

1. **Thread and core counts.** A CPU or GPU kernel splits a reduction across threads; a different split is a different order.
2. **Kernel selection.** Libraries pick an algorithm by shape and hardware (cuBLAS heuristics, cuDNN benchmarking when `torch.backends.cudnn.benchmark = True`); a different algorithm is a different order.
3. **Atomic accumulation.** Some GPU kernels add partial results with atomic operations whose order depends on scheduling; these are nondeterministic run to run. `torch.use_deterministic_algorithms(True)` makes PyTorch use deterministic alternatives where they exist and raise an error where they do not; on CUDA it also needs the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`).
4. **Batch size.** A kernel's tiling and reduction strategy often depends on the number of rows, so the result for one row depends on how many other rows are in the batch.

The fourth source is the subject of the Thinking Machines post "Defeating Nondeterminism in LLM Inference" (Horace He, 2025-09-10). Its claim, checked against the post: the usual explanation, "concurrency plus floating point", is incomplete; the forward pass of an inference server is run-to-run deterministic for a fixed batch, but its kernels lack **batch invariance**, and the batch a request lands in depends on server load, which is nondeterministic. They show `torch.mm(a[:1], b)` differing from `torch.mm(a, b)[:1]` on a GPU, build batch-invariant RMSNorm, matmul and attention kernels, and report that 1,000 temperature-0 completions of one prompt on Qwen3-235B-A22B gave 80 distinct outputs with the default kernels (first divergence at token 103) and 1,000 identical outputs with batch-invariant kernels. The cost they measured on Qwen3-8B with vLLM: 26 s by default, 55 s unoptimised deterministic, 42 s with an improved attention kernel (company claim; their benchmark).

For training, the same mechanism means that changing the micro-batch size, the number of data-parallel workers or the gradient-accumulation split changes the bits of every gradient, even with the same global batch and seed. That is expected and usually harmless — but it means "bitwise identical" is a property of a fixed configuration on fixed software, not of a seed.

### What reproducibility to ask for

The PyTorch reproducibility notes open by saying that completely reproducible results are not guaranteed across PyTorch releases, individual commits or platforms, and may differ between CPU and GPU even with identical seeds. So the course asks for three levels, and the run card says which one a run claims:

- **Bitwise:** same code, config, data, seeds, software, hardware type and parallel layout give identical checkpoints. Achievable on CPU and, with deterministic settings, on one GPU. Use it to test resume and to catch hidden state.
- **Statistical:** different software or hardware gives results within the seed noise floor of lesson 01.4. This is what a comparison actually needs.
- **Directional:** a re-implementation reproduces the sign and rough size of a published effect. This is what "reproducing a paper" means at course scale (Module 20).

## Worked example

### A run-card diff, by hand

Two run cards from the lab (`labs/module-01/lesson-05/cards/`): Baseline-0 seed 0 and an MLA branch, declared change `config.attention` (plus its settings in `config.extra` and the `--attention` argument). Leaving out bookkeeping (run name, question, logging cadence), the diff lists 15 differing keys. Classified:

| Key | Baseline-0 → MLA | Verdict |
|---|---|---|
| `config.attention`, `config.extra.*`, `args.attention` | gqa → mla, latent settings | changed (declared) |
| `params.total` | 121,917,696 → 120,590,592 | changed (follows from the config) |
| `args.lr` | 3e-3 → 4e-3 | **invalidates**: a second variable (and a tuning-budget question, lesson 01.3) |
| `args.seq` | 1024 → 2048 | **invalidates**: a different evaluation and training context |
| `args.batch` | 32 → 16 | **invalidates**: a second variable (it kept tokens equal: 9,500 × 16 × 8 × 2,048 = 9,500 × 32 × 8 × 1,024 = 2,490,368,000) |
| `data_files.val.bin_sha256` | 3ca73e… → 77d0e1… | **invalidates**: the validation split was re-prepared |
| `budget.train_flops` | 1.963e18 → 2.104e18 | changed: allowed on the equal-tokens axis (longer context costs more attention FLOPs) |
| `git.commit`, `hardware.torch`, `hardware.cuda` | different | warn: record it; invalidates only a wall-clock comparison |

Four invalidating differences, so the comparison is off until they are fixed or declared (a factorial design with its own cells). Notice that the token budget matches: a reviewer who checks only `budget.tokens` would pass this pair.

### Bitwise checks, with tiny numbers

`torch.equal` compares values, not bits. In IEEE 754 float32, $0.0$ is `0x00000000` and $-0.0$ is `0x80000000`: equal values, different bits. NaN is never equal to itself, yet a NaN copied from one tensor to another has identical bits. A bitwise check therefore compares the raw bytes (`tensor.view(torch.uint8)`): $[1.0, 0.0]$ vs $[1.0, -0.0]$ is equal by `torch.equal` and different bitwise. For "is my rerun identical?" you want bitwise; for "do two kernels agree?" you want a tolerance (the correctness suite of lesson 01.1).

## Shapes and cost

- A run card is a few kilobytes of YAML; the diff flattens it to a few hundred dotted keys and runs in milliseconds.
- `state_diff` compares every tensor of two checkpoints: for Baseline-0 that is 122M parameters plus two AdamW moments, about $3 \times 122\text{M} \times 4$ bytes ≈ 1.5 GB read per checkpoint in fp32; on CPU it takes seconds. Tensors are moved to CPU and compared as raw bytes (uint8 views), so dtype and shape must match exactly.
- Determinism has a cost. Deterministic GPU algorithms can be slower, and batch-invariant kernels give up shape-dependent optimisations; the Thinking Machines measurement above (26 s → 42 s) is one data point. The course does not require deterministic kernels for main-path training; it requires the run card to say which level of reproducibility a run claims.

## Build it

The reusable code is a new subpackage, `frontierlab.record`:

```python
from frontierlab.record import diff_cards, comparable, state_diff

findings = diff_cards("runs/b0-s0", "runs/mla-s0",
                      changed=["config.attention", "config.extra", "args.attention"], axis="tokens")
for f in findings:
    print(f)              # INVALIDATES args.lr: 0.003 -> 0.004  (a second changed variable) ...
assert comparable(findings), "fix or declare every INVALIDATES line"
```

```bash
python -m frontierlab.record runs/b0-s0 runs/mla-s0 --changed config.attention config.extra args.attention
python -m frontierlab.record runs/b0-s0 runs/b0-s1 --replicates      # seed replicates
```

The rules live in one function, `frontierlab.record.diff.classify`, documented in the module docstring: bookkeeping is ignored; the declared change and its consequences (parameter counts, budgets the axis does not hold equal) are expected; seeds are replicates only when you say so; data and evaluation differences always invalidate; software and hardware are warnings except on the wall-clock axis; any other config or argument difference is a second changed variable. The command exits with status 1 when the runs are not comparable, so it can guard an analysis script. `labs/common/tests/test_record.py` checks each rule, the exit code, and that `state_diff` is bitwise (it catches $-0.0$ and accepts identical NaNs).

## What the evidence says

- **ESTABLISHED:** floating-point non-associativity and order-dependent reductions (IEEE 754 arithmetic); the PyTorch documentation's statement that bitwise reproducibility is not guaranteed across releases, platforms or CPU vs GPU, and its deterministic-algorithm switches.
- **PUBLICLY DOCUMENTED (company claim, with a reproducible demonstration):** lack of batch invariance as the main cause of nondeterminism in LLM inference endpoints (Thinking Machines, 2025-09-10). The course's own measurement (last item) reproduces the *mechanism* — the same row computed alone or in a batch gives different bits — on a CPU; it does not reproduce their GPU or serving results.
- **Measured in this build** (2026-10-03, Windows 11 laptop, 16 threads, torch 2.14.1 CPU):
  - `rerun.py` (toy model, 100 steps of 8 × 128 tokens, about 1 minute for three runs): A vs B with 16 threads each — 0 of 47 weight tensors and 0 optimizer entries differ, logged losses identical. A vs C with 8 threads — all 47 weight tensors and 92 optimizer entries differ; the logged losses agree at the first logged step and differ from the second on, by up to 0.011 nats after 100 steps. Same seed, same code, same data, same config: the thread count alone broke bitwise identity.
  - `batch_invariance.py`: one row of a 2048 × 2048 fp32 matmul computed alone differs from the same row computed inside the full batch by up to 5.1 × 10<sup>-5</sup> (6.1 × 10<sup>-5</sup> with 2 rows; 0 only when the batch is the full 2,048 rows). The course model's logits for one sequence alone vs inside batches of 2–32 differ by 3–6 × 10<sup>-7</sup>; the same batch run twice is identical. The mechanism of the Thinking Machines post, on a CPU.
- **REASONABLE INDUSTRY PRACTICE:** recording git commit, config, data hashes, seeds and software versions per run; treating a dirty working tree as a warning; separating bitwise, statistical and directional reproducibility.

## Lab

**Folder:** [`labs/module-01/lesson-05/`](../../labs/module-01/) · **Time:** about 75 minutes · **Pass check:** `pytest labs/module-01/lesson-05` passes; `rerun.py` shows A and B bitwise identical; you can explain every line the run-card diff prints for the provided cards.

> [!NOTE]
> This lab checks records and reruns; it compares no methods, so it has no experiment contract.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× GPU; about 15 min. Not run in this build; part of the Module 1 pilot | steps 1–4, then step 5 on the GPU |
| Free GPU (Colab/Kaggle T4) | T4 | as the main path, `--dtype fp32` |
| Free CPU | laptop; `rerun.py` measured at about 1 minute, `batch_invariance.py` 10 s (2026-10-03, 16 threads) | steps 1–4 |

1. **Implement** `flatten`, `classify` and `same_bits` in `lab.py`; run `pytest labs/module-01/lesson-05`.
2. **Diff the provided cards.** Run `python -m frontierlab.record labs/module-01/lesson-05/cards/b0-s0.yaml labs/module-01/lesson-05/cards/mla-s0.yaml --changed config.attention config.extra args.attention`. For each INVALIDATES and WARN line, write what you would do: rerun, declare, or accept and record.
3. **Rerun bitwise.** Run `python labs/module-01/lesson-05/rerun.py`. Runs A and B use the same thread count; run C uses half. Record how many weight tensors differ for A vs B and for A vs C. Then look at the run-card diff it prints for A vs C and decide whether `hardware.cpu_threads` should be "bookkeeping" for a bitwise claim.
4. **Measure batch invariance.** Run `python labs/module-01/lesson-05/batch_invariance.py`. Which of its three checks would change if you reran it? Which would change if you ran a single request on a busy server?
5. **Main path only:** run `batch_invariance.py --device cuda`, then rerun step 3 on the GPU with and without `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set (add them at the top of a copy of `rerun.py`). Record the speed cost.

<details>
<summary>Hint for step 3</summary>

A seed fixes the random numbers, not the order of additions. With half the threads, every matrix multiply and every sum is split differently. Look at the first logged loss row that differs: is it the very first step?

</details>

<details>
<summary>Reference solution</summary>

`labs/module-01/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-01/lesson-05`.

</details>

## Common mistakes

- **Comparing budgets but not data hashes.** Two runs can match on tokens and FLOPs and still train or evaluate on different files.
- **Treating "same seed" as "same run".** Thread count, micro-batch size, library version and GPU type all change the bits.
- **Using `torch.equal` as a bitwise check.** It is value equality; use raw bytes when you claim "identical".
- **Leaving the tree dirty.** The commit hash then does not describe the code that ran; commit or record a patch.
- **Re-preparing a split "because it was deleted".** Hash it before you use it; a changed hash ends every comparison with older runs.
- **Asking for bitwise reproducibility across hardware.** Ask for statistical reproducibility (within the noise floor) instead, and say so in the contract.

## References

- Horace He and Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*, 2025-09-10. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- PyTorch, *Reproducibility* (2.14 notes). https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch, `torch.use_deterministic_algorithms`. https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- Hugging Face, *FineWeb-Edu* dataset card. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Templates: [run card](../../templates/run-card.md), [experiment contract](../../templates/experiment-contract.md).

## Next

The module project, [Baseline-0, Eval Suite v0 and the noise floor](../../projects/module-01-baseline0.md), puts all five lessons together.
