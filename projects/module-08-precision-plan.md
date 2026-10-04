# Module 8 project · A precision plan with measured throughput and an error budget

Which precision should Recipe-R — the training recipe carried into the Module 11 run — use for each part of the model, and in what precision will the result be served? This project answers that for a stated model and hardware with the evidence a reviewer would ask for: a per-component error budget measured on a trained model, the training-time loss gap from controlled runs (lesson 08.2, and lesson 08.3 if you consider FP4), throughput measured with real kernels on the training GPU next to its roofline projection, memory per parameter, and a decision rule written before the measurements. It uses all three required lessons: formats and scaling (08.1), FP8 training (08.2), FP4 and quantisation-aware training (08.3); the extension 08.4 informs the deployment part.

**Time:** 5–7 attended hours plus unattended GPU time. **Folder:** [`labs/module-08/project/`](../labs/module-08/) (`error_budget.py`, `buggy_budget.py`; it also uses `lesson-02/train_fp8.py`, `compare_fp8.py` and `bench_fp8.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## The stated model and hardware

| | Main path | Free GPU (Colab L4) | Free CPU |
|---|---|---|---|
| Model | Baseline-0 (or your Module 6 architecture), the Module 1 checkpoint | pilot-30m trained in lesson 08.2 (`--variant l4`) | the toy `bf16-s0` model of lesson 08.2 |
| Training hardware | 1× H100 SXM 80 GB | 1× L4 24 GB (FP8 tensor cores, sm89) | laptop: emulation only |
| Serving target | 1× L4 24 GB, 4-bit weights allowed | same | — |
| Throughput evidence | measured: `bench_fp8.py --device cuda --hw H100-SXM --preset baseline0 --batch 16 --seq 1024` | measured: `bench_fp8.py --device cuda --hw L4 --preset pilot-30m --batch 8 --seq 512` | PROJECTED only (roofline formula), labelled |
| Cost | **PROJECTED, pending the Module 8 pilot:** lesson 08.2's main-path runs (6–9 GPU-hours) plus about 0.5 GPU-hours for the error budget and the benchmark | 1.5–2.5 GPU-hours (lesson 08.2 L4 variant) plus about 0.3 | about 10 minutes beyond lesson 08.2's runs |

> [!IMPORTANT]
> Main-path and L4 commands were not run in this build. Every figure for them in this page is PROJECTED with its
> formula; replace it with your measurement and record the measurement in the run card's `measured:` block.

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. Fixed by the project:

- **Question:** for the stated model and hardware, which precision does each tensor class get — the linears' three GEMMs, attention, output head and embedding, norms, master weights, gradients, optimizer moments, and the served weights — such that (a) the training-time held-out loss gap to BF16 stays within 0.25% of the BF16 loss, (b) the served model's gap stays within your deployment budget (default 1% of the BF16 loss), and (c) the training step is measurably faster than BF16 on the training GPU? Decision informed: Recipe-R's precision and the serving format.
- **Hypotheses and status:** (H1) FP8 linears (torchao rowwise or tensorwise) meet (a) — reported effect (DeepSeek-V3 appendix B.1), may not be resolvable at course scale; (H2) FP8 linears meet (c) with a smaller gain than the roofline bound — established that kernels fall short of bounds; (H3) 4-bit MLP weights by PTQ exceed the deployment budget where QAT does not — reported in general, unknown here; (H4) the error budget is roughly additive across components (attention + MLP ≈ all) — an assumption to test.
- **Baseline:** the BF16 arm of lesson 08.2 (same seeds and data), not re-tuned; for serving, the BF16 checkpoint.
- **Changed variables:** a declared set of components, each evaluated alone and then together (a factorial-style budget, not one changed variable).
- **Controlled:** checkpoint, evaluation windows (256 fixed windows), seeds, data version and hashes, software versions (torch 2.14.1, torchao 0.18.0), GPU type for the speed measurement.
- **Comparison axes:** equal tokens for the training gap; equal work per step for throughput; the same checkpoint for the serving gaps.
- **Budget:** as in the table above; record measured GPU-hours.
- **Metrics and decision rule (state now):** for each component, the paired gap with its 95% interval; the training-time gaps from lesson 08.2 (seed- and window-paired); bits per weight; measured speed-up interval. **Rule:** adopt a training precision only if its training-gap interval's upper end is within 0.25% of the BF16 loss *and* the measured speed-up interval is entirely above 1 (emulation alone gives "numerics ok, speed not measured", which keeps BF16); adopt a serving format only if its serving-gap interval's upper end is within the deployment budget; when two options pass, take the one with fewer bits, then the simpler one.
- **Correctness checks:** `pytest labs/common/tests/test_precision.py` and the Module 8 lab tests pass on the machine you use; `bench_fp8.py`'s first-step loss gate passes; `python -m frontierlab.record <bf16 run> <fp8 run> --changed precision` shows no other INVALIDATES line.
- **Fallback evidence:** DeepSeek-V3's FP8 curves (appendix B.1), NVIDIA's NVFP4 results, labelled as analysis of published results; the Module 8 pilot traces once published.
- **Limits:** one model size and token budget; emulated numerics with exact FP32 accumulation; throughput on one GPU without sharding (Module 9 adds FSDP and communication); the deployment gap is measured on held-out loss only, not on downstream tasks.

## Steps and deliverables

1. **Contract** (deliverable 1), before step 2.
2. **Training gap:** lesson 08.2's runs and `compare_fp8.py` for your variant (main: `--variant main`; L4: `--variant l4`; CPU: as written). Reuse them; do not rerun.
3. **Error budget:** `python labs/module-08/project/error_budget.py` (CPU), or with `--ckpt <your Baseline-0 checkpoint> --device cuda --hw H100-SXM --T 1024` on the main path.
4. **Throughput:** `bench_fp8.py` on the training GPU (main or L4). On the CPU path, copy the PROJECTED numbers and their formula from `bench_fp8.py` without a GPU, and label them.
5. **Memory:** bytes per parameter of your training plan (master weights, gradients, moments, cached activations) and of the served weights (bits per weight including scales).
6. **The plan** (deliverable 2): one table, one row per tensor class: training precision, serving precision, evidence (which measurement, with its interval), and the rule's verdict. Then the overall decision for Recipe-R.
7. **Record** (deliverable 3): run cards of all runs used, the `error_budget/` JSON files, the `bench_fp8.py` output, the record diffs.
8. **Debugging task** (deliverable 4), below.
9. **Written defence** (deliverable 5), below.

### What the free CPU variant gave in this build

Measured 2026-10-04 on the build laptop (16 threads, torch 2.14.1+cpu, other jobs running), on lesson 08.2's `bf16-s0` toy checkpoint (200 steps, held-out loss 6.5656 on 256 windows of 128 tokens): `error_budget.py` took 69 seconds. This is a reference for what your output should look like, not a statement about Baseline-0.

| Component (forward-pass quantisation of the trained weights) | Gap vs BF16 [95% CI] | Bits per weight | PROJECTED step speed-up (Baseline-0, 16 × 1,024, casts fused) |
|---|---|---|---|
| FP8 DeepSeek recipe, attention linears | +0.0000 [−0.0001, +0.0002] | 8.00 | 1.28× (H100) |
| FP8 DeepSeek recipe, MLP linears | −0.0001 [−0.0002, +0.0000] | 8.00 | |
| FP8 DeepSeek recipe, all block linears | −0.0001 [−0.0003, +0.0001] | 8.00 | 1.28× (H100) |
| FP8 tensorwise, all | +0.0002 [+0.0000, +0.0004] | 8.00 | 1.28× (H100) |
| NVFP4, attention / MLP / all | +0.0014 / +0.0012 / +0.0027 | 4.03 | 1.41× (B200) |
| PTQ INT4 groups of 32, MLP weights | +0.0003 [+0.0001, +0.0005] | 5.00 | — |
| PTQ MXFP4, MLP weights | +0.0002 [−0.0002, +0.0005] | 4.25 | — |

Additivity: NVFP4 attention + MLP = +0.0026 against +0.0027 for all; FP8 is below resolution. Training-time gaps (lesson 08.2): every FP8 arm **inconclusive** by its rule, with per-seed gaps of ±0.02–0.03 nats.

What a write-up should say. The serving budget is easy here — every format costs under 0.003 nats on this barely trained model, an order of magnitude inside a 1% (0.066-nat) deployment budget, so the rule picks MXFP4 MLP weights (4.25 bits) for serving at this scale. The training decision is the hard part and the honest answer is "keep BF16": the training-gap evidence is inconclusive at two seeds, and no speed has been measured — the 1.28× is a roofline bound on hardware the build did not use. The plan's value on the CPU path is the procedure; the main path, with real kernels and Baseline-0, is where it becomes a decision. Two cautions belong in the defence: forward-pass error budgets of a 200-step model understate what heavily trained models lose (lesson 08.4), and the FP8 training gap is dominated by trajectory divergence that only more seeds can average out.

## Debugging task

`labs/module-08/project/buggy_budget.py` is a colleague's version of the report. It prints a gap for NVFP4 MLP linears, the memory of 4-bit MLP weights and an FP8 speed verdict, and concludes that 4-bit MLP weights cost almost nothing and that FP8 should not be used. All three numbers are produced by mistakes. For each, name the check that exposes it, fix it and show the corrected number.

```bash
python labs/module-08/project/buggy_budget.py
```

<details>
<summary>Hint</summary>

Look at how each of the two loss lists is produced: are they over the same windows? Then look at what `bits` counts, and at where the speed number comes from.

</details>

<details>
<summary>Reference diagnosis</summary>

1. **The two arms are scored on different windows.** The quantised model is evaluated with `seed=4321`, the baseline with the default 1234, so "paired" differences are differences between unrelated windows; the gap mixes the precision effect with window difficulty, and its interval is meaningless. Check: print the first window start of each list (`TokenData.eval_windows`), or rerun the quantised arm with the default seed — the gap changes. Fix: one window list for every arm (`m08.eval_model` / `error_budget.py`).
2. **Memory without scales.** `get_spec("nvfp4-2d").format.bits` is 4, the element width; NVFP4 also stores one E4M3 scale per 16 × 16 block (and one FP32 per tensor), so the weights cost `QuantSpec.bits_per_value(shape)` ≈ 4.03 bits for 2-D 16 × 16 blocks and 4.5 bits for 1 × 16 blocks. Check: compare with `quant_error(w, spec)["bits"]`. Fix: report bits per value including scales.
3. **Speed from emulation.** The FP8 throughput is the ratio of *emulated* training tokens/s from the run logs; emulation reproduces FP8 rounding in float32 and is slower than BF16 by construction. Check: the run card's `precision.emulated: true`. Fix: no speed claim without `bench_fp8.py` on real kernels; on the CPU path report the PROJECTED bound with its formula and label it.

In this build the buggy script printed a gap of +0.0080 [−0.0362, +0.0511] (an interval 50 times too wide, because unrelated windows were "paired"), 0.295 MB for 589,824 MLP weights (4 bits; with scales 4.03 bits, 0.297 MB for 16 × 16 blocks, 0.33 MB for 1 × 16) and an FP8 throughput of 0.72× BF16 (emulation). With these fixed, the report is what `error_budget.py` prints (NVFP4 MLP: +0.0012 [+0.0007, +0.0016]), and the FP8 decision is "numerics ok, speed not measured" on the CPU path.

</details>

## Written defence

One to two pages, answering:

1. Which components of your plan are backed by a measurement on your model, which by a published result, and which by an assumption? Label each (measured / PUBLICLY DOCUMENTED / INFERENCE).
2. Your training-gap margin is 0.25% of the loss. What is your noise floor, and could your design have detected a gap of that size? If not, what would you change: seeds, tokens, model size?
3. Was the error budget additive (attention + MLP ≈ all)? If not, what does that mean for a plan built component by component?
4. Your measured FP8 speed-up is below the roofline bound. Where does the difference go: cast kernels, small GEMMs, the output head, launch overhead? Which measurement from Module 2 would tell you?
5. NVIDIA's NVFP4 recipe keeps the first two and last eight blocks of a 12B model in BF16. How would you decide which layers of your model to keep in high precision, and with what experiment?
6. With 10× the budget, what would you measure first to make this plan trustworthy at the Module 11 run's scale?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the margins, the deployment budget and the adoption rule are written before any measurement |
| 2 | Controls | one checkpoint and one window list for every component; run cards show identical data, seeds and software for the training arms |
| 3 | Axis and budget parity | training gaps at equal tokens; speed at equal work on one GPU type, both arms compiled |
| 4 | Correctness | `test_precision.py`, the lab tests and the benchmark's first-step gate pass and are included |
| 5 | Uncertainty | every gap has a paired interval; the speed-up has a bootstrap interval; the noise floor is stated |
| 6 | Conclusion matches evidence | emulated results claim numerics only; projections are labelled; no speed claim without real kernels |
| 7 | Limits | scale, single GPU, held-out loss only, and what result would change the plan |
