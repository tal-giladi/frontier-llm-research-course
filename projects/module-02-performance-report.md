# Module 2 project — Performance report on Baseline-0

Write the performance report that every later systems comparison in the course will point back to:
what Baseline-0's training step *should* cost on your hardware, what it *does* cost, where the
difference goes, and one improvement validated end to end. Lessons [02.1](../lessons/module-02/lesson-01.md),
[02.2](../lessons/module-02/lesson-02.md), [02.3](../lessons/module-02/lesson-03.md) and
[02.4](../lessons/module-02/lesson-04.md) each produce one part of it; the project assembles them, adds
the missing measurements and defends the conclusions.

**Time:** 5–8 hours attended. **Compute:** main path about 1 GPU-hour on one H100 SXM or A100 (plus
about 15 minutes on a 2–8 GPU node for part 3, if you have one); free GPU (T4) or free CPU with the
scaled-down commands in [`labs/module-02/README.md`](../labs/module-02/README.md), stating what the
smaller setting cannot show.

## Deliverables

1. **Prediction (before any timing).** For Baseline-0 at your main-path shape (default: B = 8,
   T = 1024, BF16, V = 32768, one GPU): FLOPs per step, predicted step time at MFU 1.0 and at an
   assumed MFU you state, and the per-op roofline bound from `frontierlab.perf.roofline.step_time_model`,
   with the datasheet peak and bandwidth of the exact SKU you rent and their source.
2. **Measurement.** Median step time with a 95% interval (warm-up, synchronisation, at least 20
   repeats, `frontierlab.perf.timing`), tokens per second, implied MFU; the empirical roofline of your
   GPU (`labs/module-02/lesson-01/measure.py`).
3. **The gap, explained.** A profile of the step (`profile_step.py`, trace attached): time by category,
   GPU busy fraction, the three largest contributors to the gap between the per-op bound and the
   measurement, each with the evidence from the trace. One paragraph on whether the step is compute-,
   memory- or launch-bound, and why.
4. **Memory.** Saved activation bytes by tensor class, the logits share, and a CUDA memory snapshot of
   two steps with the peak annotated. The largest micro-batch that fits, with and without checkpointing.
5. **Communication** (main path with ≥ 2 GPUs, Kaggle 2× T4, or CPU gloo labelled as such): exposed
   communication for DDP at two bucket sizes and for FSDP2, with intervals, and the bus bandwidth you
   measured.
6. **One validated improvement.** Your own choice (the chunked loss of 02.4, `torch.compile`, a fused
   optimizer, a different micro-batch with accumulation, removing a per-micro-batch `.item()`, ...),
   with the filled-in experiment contract below, the correctness check, the equal-work comparison and
   the decision your pre-stated rule gives. A null or negative result is a valid outcome.
7. **Run cards** for every run (`templates/run-card.md`), including hardware, software versions and cost.

## Experiment contract (fill in before part 6)

Copy [`templates/experiment-contract.md`](../templates/experiment-contract.md) and fill every line.
Minimum content:

- **Question:** "Does <change> reduce Baseline-0's <time per step / peak memory / time to N tokens> on
  <GPU> at <shape> without changing the loss or gradients?"
- **Hypothesis and status:** with the expected direction and size, and where that expectation comes
  from (02.1 roofline, 02.2 profile).
- **Baseline:** the unmodified step, same commit, same shapes.
- **Changed variable:** one. **Controlled:** model, initial weights, batches, dtype, torch version, GPU
  type and clocks (note any other tenants), number of steps.
- **Comparison axis:** equal work (same tokens per step) — and, if your change saves memory, also
  equal memory budget (each arm at the largest micro-batch that fits), saying what each answers.
- **Budget:** GPU-minutes per arm; cost at your rental price.
- **Metrics and decision rule:** median step time per arm from interleaved rounds, paired speed-up with
  a 95% bootstrap interval; peak memory; the rule (for example, 02.4's: adopt if correct, the
  speed-up interval's lower end ≥ 0.95, and memory ≤ 80% of baseline or the lower end > 1.0).
- **Correctness checks:** float64 loss and gradient agreement on a batch (tolerance stated); training
  losses of both arms over the timed steps on the same batches (largest difference stated).
- **Limits:** one model size and shape; what would change the answer at a frontier model's width,
  vocabulary or sequence length.

## Debugging task

[`labs/module-02/project/claim.py`](../labs/module-02/project/claim.py) prints "1.78x faster" for the
chunked loss on the toy model (measured in this build on CPU). The claim is not a valid measurement.
List every flaw, fix them, and report what the corrected script shows.

<details>
<summary>Reference list of flaws</summary>

1. **Not equal work.** The optimized arm trains on `x[:8]`, half the batch. Both arms must process the
   same tokens.
2. **Cold baseline.** The baseline is a single first call: allocator growth and first-call costs are
   in it. Both arms need the same warm-up.
3. **Different statistics.** One sample for the baseline, the *minimum* of five for the new arm. Use
   the same number of samples and the median for both, with an interval.
4. **No interleaving.** The arms run one after the other, so any drift (and the baseline warming the
   caches for the second arm) favours one side. Alternate A, B, A, B.
5. **No synchronisation.** Harmless on CPU, wrong on a GPU: `time.perf_counter()` would time kernel
   queueing. Call `torch.cuda.synchronize()` before reading the clock.
6. **No correctness check and no decision rule.** Nothing shows the two arms compute the same loss and
   gradients, and the conclusion was not tied to a rule stated in advance. It also ignores the actual
   point of the chunked loss, which is memory.

With the fixes, `labs/module-02/lesson-04/compare_ce.py` is the corrected measurement.

</details>

## Written defence (1–2 pages)

Answer these as a reviewer would ask them:

1. Your implied MFU is X%. Which number in your report shows where the other (1 − X) went, and how
   sure are you of it?
2. Which datasheet figure did you use for the peak, and is it the dense or the sparsity number? What
   happens to every MFU in the course if it is the wrong one?
3. Why should your improvement's speed-up be believed? Name the warm-up, synchronisation, repeat
   count, interleaving and interval.
4. Is your comparison at equal work, equal memory, or both? What would the other axis have answered?
5. What is the smallest speed-up your measurement could distinguish from 1.0, given your intervals?
6. Would the improvement still help at a frontier model's width (C ≈ 7,000), vocabulary (V ≈ 150,000)
   and sequence length? Use the roofline arithmetic, not intuition.
7. Which part of your report would change on a different GPU SKU, and which would not?

## Self-check against the rubric

Score each criterion of [`templates/experiment-rubric.md`](../templates/experiment-rubric.md) 0–2 for
part 6 and the report as a whole. For this project, "done" looks like:

| # | Criterion | Done looks like here |
|---|---|---|
| 1 | Question and decision | The improvement's question and decision rule are dated before the first timing run. |
| 2 | Controls | Same weights, batches, shapes, dtype and software for both arms, shown in the run cards. |
| 3 | Axis and budget parity | Equal work stated (and equal memory if relevant); both arms got the same warm-up and repeats. |
| 4 | Correctness | Float64 gradient agreement and training-loss agreement reported with numbers. |
| 5 | Uncertainty | Every time has a 95% interval; the speed-up is paired and interval-based. |
| 6 | Conclusion matches evidence | The decision follows the rule; no claim about other GPUs or model sizes without arithmetic. |
| 7 | Limits | One size, one shape, one SKU; what would change the answer. |

A project passes at 10 of 14 with no zero. Whether the improvement wins is not a criterion.
