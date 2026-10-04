# Module 7 project · Optimizer decision report: Muon or tuned AdamW?

Should Recipe-R — the training recipe carried into the Module 11 run — use Muon or AdamW? This project answers that for Baseline-0 with an experiment a reviewer would accept: both optimizers tuned with the same budget, a µP transfer test from a small width to Baseline-0's width (so the tuning was done where it is cheap), a comparison at equal tokens **and** at equal wall-clock (Muon's extra optimizer work counts), seed noise and paired intervals, and a decision rule written before the runs. It uses all five lessons: Muon and its cost (07.1), Muon's scaling and logit control (07.2), µP transfer (07.3), the schedule (07.4) and the stability log (07.5). The answer goes into Recipe-R; the same method decides any later optimizer question.

**Time:** 6–8 attended hours plus unattended GPU time. **Folder:** [`labs/module-07/project/`](../labs/module-07/) (`run_project.py`, `buggy_report.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Widths, tokens and grid | Hardware | Cost |
|---|---|---|---|
| Main path | Baseline-0 layout; tune at width 384 (µP base), transfer to 768 (Baseline-0); 9,500 steps × 32 × 8 × 1,024 tokens (2.49B, the Module 1 budget); grid $\{1, 2, 4, 8\} \times 10^{-3}$ for both optimizers; 3 seeds at 768; Data-v0 at vocabulary 32,768 | 1× H100 SXM 80 GB | **PROJECTED, pending the Module 7 pilot.** Training FLOPs per token (`accounting.flops_per_token`, $T = 1{,}024$): $2.49 \times 10^8$ at width 384, $7.88 \times 10^8$ at 768. Runs: 8 at 384, 6 + 4 at 768 at equal tokens, 3 AdamW runs at equal wall-clock (assumed $1.05\times$ the steps; the measured ratio decides). FLOPs $= 2.49 \times 10^9 \text{ tokens} \times (8 \cdot 2.49 + 10 \cdot 7.88 + 3 \cdot 1.05 \cdot 7.88) \times 10^8 \approx 3.1 \times 10^{19}$; at an assumed 30% MFU on 989 TFLOP/s, about 29 GPU-hours, roughly USD 60–90 at USD 2–3 per H100-hour. Muon's Newton–Schulz adds 0.41% FLOPs at this batch (lesson 07.1); its wall-clock share is measured, not assumed |
| Free GPU (Colab/Kaggle T4) | `pilot-10m` layout, widths 192 → 384, 2,000 steps × 32 × 512 | T4, fp32 | PROJECTED about 4–6 hours over several sessions (`--max-minutes`, rerun to resume) |
| Free CPU | toy layout, widths 64 → 128 (base 64); 200 steps × 16 × 128 tokens (0.41M); grid $\{1, 3, 10, 30\} \times 10^{-3}$; 3 seeds at 128 | laptop | measured: below |

> [!IMPORTANT]
> The main path needs Data-v0 prepared at the main-path size (vocabulary 32,768, Module 1 project). The free CPU
> variant uses the course's CPU-size Data-v0 (20,000 documents, vocabulary 8,192). Commands for the main path are
> printed by `python labs/module-07/project/run_project.py --variant main --print`; they were not run in this build.

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. Fixed by the project:

- **Question:** at Baseline-0's width and token budget, does Muon (Moonlight's update-RMS matching, quintic × 5, Nesterov momentum 0.95, AdamW for embedding and vectors) reach a lower held-out loss than AdamW, (a) at equal tokens and (b) at equal wall-clock on the same GPU, when both are tuned with the same budget at a smaller width and transferred with µP? Decision informed: the optimizer of Recipe-R.
- **Hypotheses and status:** (H1) Muon's held-out loss is lower at equal tokens — reported effect (Moonlight: ~52% of AdamW's FLOPs at compute-optimal size, section 3.2; Essential AI: 10–15% fewer tokens, section 2.4), may be smaller or absent at 125M and 2.5B tokens; (H2) the advantage shrinks at equal wall-clock by about Muon's measured step-time overhead; (H3) the best learning rate at the large width is within one grid step of the one transferred from the small width, for both optimizers (lesson 07.3).
- **Baseline:** AdamW (β = 0.9, 0.95, weight decay 0.1, the loop's settings) through `MuonAdamW`, so both arms run the same code; µP on for both.
- **Tuning budget:** the same for both: 4 learning rates at the small width, 1 seed each, chosen on validation loss; at the large width the transferred value and one grid step either side. Nothing else is tuned (momentum, β, weight decay, warmup, schedule stay at their defaults for both). State this budget in your report.
- **Changed variable:** the optimizer of the hidden matrices. **Controlled:** data and its hashes, tokenizer, the loop's cosine schedule with warmup, gradient clipping 1.0, QK-norm on, µP base width, seeds {0, 1, 2}, the 256 fixed held-out windows, GPU type and software versions (wall-clock comparisons are invalid across either).
- **Comparison axes:** equal tokens (2.49B on the main path) — answers "which optimizer learns more per token"; equal wall-clock — AdamW gets $N_{\text{wc}} = N \cdot t_{\text{Muon}}/t_{\text{AdamW}}$ steps, with the step times measured in the seed runs on the same machine — answers "which is better for this GPU-hour budget". Neither answers which is better at compute-optimal size or at 1T parameters, or with a sharded optimizer (Module 9).
- **Budget:** fill in GPU type, hours and cost, projected now (table) and measured afterwards in each run card's `measured:` block. Count Muon's optimizer FLOPs separately (`budget.optimizer_flops` in its run card).
- **Metrics and decision rule:** held-out loss on 256 fixed windows (main path: 1,024 tokens each), paired by window (lesson 01.4), seed-averaged; per-seed differences; the seed noise floor and MDE with 3 seeds per arm. **Rule (stated now):** adopt Muon for Recipe-R if, *at equal wall-clock*, the paired interval of Muon − AdamW has its upper bound below −0.02 nats and every per-seed difference is negative; keep AdamW if the lower bound is above −0.02; otherwise inconclusive — then the decision falls back to the simpler, established choice (AdamW) and the report says so.
- **Correctness checks:** `pytest labs/common/tests/test_optim.py` passes on the training machine (Newton–Schulz, Muon vs `torch.optim.Muon`, AdamW vs `torch.optim.AdamW`, µP, exact resume with Muon); the 07.3 coordinate check passes at both widths; `python -m frontierlab.record <adamw run> <muon run> --changed optim.optimizer optim.muon_matrices optim.optimizer_flops_per_step args.lr` shows no other INVALIDATES line (for the equal-wall-clock pair add `args.steps budget`, on `--axis wallclock`); every run's stability log shows max attention logit below 20 and no detected spike (or the report explains the spike with lesson 07.5's evidence).
- **Fallback evidence:** if the effect is within the noise floor, that is the result; cite Moonlight's and Essential AI's curves as published evidence at other scales, labelled as such.
- **Limits:** one model size and token budget; one data set; one seed set; no sharding; the free variants are far smaller than the main path and their numbers do not transfer.

## Steps and deliverables

1. **Contract** (deliverable 1), before step 2.
2. **Measure the step-time ratio** with `labs/module-07/lesson-01/cost_table.py` on the training GPU (or let the project measure it from the seed runs; record both).
3. **Run** (unattended; every stage resumes):

   ```bash
   python labs/module-07/project/run_project.py --variant cpu
   python labs/module-07/project/run_project.py --variant main --print        # main path, run each printed command
   ```

   Stages: `tune` (8 runs), `transfer` (up to 6), `seeds` (4), `wallclock` (3), `report`.
4. **Report** (deliverable 2): the tuning table for both optimizers; the transfer verdict per optimizer; the noise floor and MDE; the equal-tokens and equal-wall-clock comparisons with intervals and per-seed differences; the decision by your rule; Muon's measured step-time overhead and its FLOP share; the stability-log summary.
5. **Record** (deliverable 3): run cards of every run, `report.json`, the record diffs of step 3's correctness check.
6. **Debugging task** (deliverable 4), below.
7. **Written defence** (deliverable 5), below.

### What the free CPU variant gave in this build

Measured 2026-10-04 on the build laptop (16 threads, torch 2.14.1+cpu, other jobs running): 23 runs, 30.5 minutes of training in all (about 70 s per run at width 64, 80 s at width 128), plus about 3 minutes of evaluation. This is a reference for what your output should look like, not a statement about Baseline-0. Held-out loss, 256 windows of 128 tokens, after 200 steps (0.41M tokens):

| Stage | AdamW | Muon |
|---|---|---|
| tuning at width 64, lr 1e-3 / 3e-3 / 1e-2 / 3e-2 / 1e-1 | 7.108 / **6.852** / 6.853 / 6.887 / 7.013 | 7.130 / 6.543 / 6.206 / **6.205** / 6.251 |
| transfer to width 128 (µP), transferred lr and neighbours | 1e-3: 7.072, **3e-3: 6.756**, 1e-2: 6.788 → shift 0 | **1e-2: 6.107**, 3e-2: 6.126, 1e-1: 6.227 → shift 1 |
| seeds 0/1/2 at width 128, transferred lr | 6.756 / 6.745 / 6.783 (std 0.020) | 6.126 / 6.147 / 6.107 (std 0.020) |
| equal wall-clock: AdamW 242 steps (= 200 × measured 1.211) | 6.604 / 6.610 / 6.637 | (Muon's 200-step runs) |

MDE with 3 seeds per arm: 0.046 nats. Equal tokens: Muon − AdamW = −0.635 [−0.654, −0.616] (paired by window, seed-averaged), per-seed −0.630, −0.598, −0.676. Equal wall-clock: −0.491 [−0.507, −0.474], per-seed −0.479, −0.463, −0.530. Verdict by the rule: **adopt Muon** — at this scale.

What a write-up should say. Muon's advantage is enormous here — 0.6 nats at equal tokens, still 0.5 after paying its 21% wall-clock overhead — far larger than anything the published ~2× compute-efficiency claims imply for long runs, where both optimizers' loss curves are flat and a 2× compute saving is a modest loss difference at equal compute (INFERENCE from the shape of scaling curves, not a published number). The reason is the regime: 200 steps of a 1.8M-parameter model is the very start of training, where an optimizer that moves every direction of a matrix at once makes fast early progress; the Moonlight and Essential AI comparisons are at hundreds of millions of parameters and billions of tokens, where the curves converge. The result answers "which optimizer is better for a 200-step toy run on this laptop" and nothing more — which is exactly why the main path trains Baseline-0 for 2.49B tokens. Two details deserve a sentence each: AdamW's tuning grid had a near-tie at width 64 (3e-3 and 1e-2 within 0.001 nats), and Muon's optimum moved one grid step down from width 64 to 128 under the spectral rule (by 0.018 nats, below one seed std), a hint — not evidence — that the $1/\sqrt m$ rule may under-correct at these widths (lesson 07.3's extension tests the alternative).

## Debugging task

`labs/module-07/project/buggy_report.py` is a colleague's version of the report stage. It prints the same number for both axes and concludes "Muon wins by a wide margin at equal tokens *and* at equal wall-clock, so the overhead does not matter". Both numbers are produced by bugs in how the script picks and pairs runs (in this build's CPU runs it reported −0.96 nats on both axes; the correct values are −0.63 and −0.49). Whether Muon still wins after the fix is for your data to say; the claim that the overhead does not matter is not supported by a comparison that never paid it. Start from the symptoms; for each bug, name the check that isolates it (a run-card diff, a number you can recompute by hand), fix it and show the corrected numbers.

```bash
python labs/module-07/project/buggy_report.py runs/m07/project/cpu
```

<details>
<summary>Hint</summary>

Print which run directories the script compares in each line, then run `python -m frontierlab.record` on each pair. What does the equal-tokens line hold equal besides tokens, and what does the equal-wall-clock line hold equal at all?

</details>

<details>
<summary>Reference diagnosis</summary>

1. **The AdamW arm is the untuned one.** The script takes the first AdamW run it finds at the large width (`sorted()` puts `lr0.001` first), not the run at AdamW's transferred learning rate, while Muon gets its tuned value. Check: `python -m frontierlab.record <adamw run> <muon run> --changed optim.optimizer` prints `INVALIDATES args.lr` — a second changed variable; the tuning budgets are no longer equal. Fix: select both arms by their own best learning rate from the tuning stage (as `run_project.py` does).
2. **"Equal wall-clock" is the equal-tokens AdamW run.** The script labels the seed-0 AdamW run (same steps as Muon) as the wall-clock arm, so the "wall-clock" result is the equal-tokens result again, and Muon's overhead never enters. Check: the two runs' `args.steps` are equal and their measured `wallclock.jsonl` seconds differ by the step-time ratio; `frontierlab.record ... --axis wallclock` flags the wall-clock difference. Fix: use the `-n<N_wc>` runs from the `wallclock` stage.

With both fixes, the report is the one `run_project.py --stage report` prints. One more, subtler problem: the script picks Muon's run as the best of the large-width runs *on the same held-out windows it reports*, which turns the report set into a selection set (lesson 01.3); the project uses the learning rate transferred from the small width instead.

</details>

## Written defence

One to two pages, answering:

1. Which comparison axis decides Recipe-R's optimizer for this course, and why not the other one? What would change your answer on a cluster where the optimizer step must gather sharded matrices (Module 9)?
2. How do you know AdamW was tuned as hard as Muon? List every hyperparameter that was *not* tuned for either and argue why that is fair, or not.
3. What is the smallest difference your design can detect (MDE with your seed std)? Is the effect you measured above it?
4. Did the learning rate transfer from the small width for both optimizers? If one did not, what does that do to the comparison at the large width?
5. Moonlight reports about 2× compute efficiency. Is your result consistent with that, given your scale, token budget and axis? Name two reasons a course-scale result could differ from it in either direction.
6. With 10× the budget, what would you run first: more seeds, a third width, a compute-optimal ladder (Module 11), or Muon with QK-Clip at a raised learning rate? What result would make you abandon Muon for Recipe-R?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the adoption rule and the axis that decides are written before the runs |
| 2 | Controls | run cards show the same data hashes, schedule, seeds and µP base; only the optimizer (and `args.lr` within the tuning stage) differs |
| 3 | Axis and budget parity | equal tokens verified from `budget.tokens`; equal wall-clock from measured seconds on one GPU type; the same tuning grid and run count for both optimizers |
| 4 | Correctness | `test_optim.py`, the coordinate check, exact resume with Muon and the stability log are included |
| 5 | Uncertainty | paired intervals, per-seed differences, the noise floor and the MDE are reported |
| 6 | Conclusion matches evidence | "Muon is better" is claimed only on the axis and at the scale where the rule says so |
| 7 | Limits | scale, data, single GPU (no sharding), and what would change the answer |
