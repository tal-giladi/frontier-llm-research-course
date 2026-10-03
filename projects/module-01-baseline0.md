# Module 1 project · Baseline-0, Eval Suite v0 and the noise floor

Every architecture and recipe experiment in this course is "Baseline-0 with one change", judged by Eval Suite v0 against Baseline-0's noise floor. This project builds those three artefacts: a tuned Baseline-0 trained with three seeds on Data-v0, Eval Suite v0 (held-out loss on fixed windows plus LAMBADA last-word prediction, scored per item), and a measured noise floor with minimum detectable effects. It uses all five lessons of Module 1: the cost arithmetic of 01.1–01.2, the contract of 01.3, the statistics of 01.4 and the record of 01.5.

**Time:** 6–8 attended hours plus unattended GPU time. **Folder:** [`labs/module-01/project/`](../labs/module-01/) (scripts `run.py`, `evaluate.py`, `noise_floor.py`, `buggy_evaluate.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model and data | Hardware | Runs | Cost |
|---|---|---|---|---|
| Main path | `baseline0` (97M non-embedding, 122M total), Data-v0 at the main-path size (about 2.5B tokens, vocab 32,768), sequence 1,024, 9,500 steps × 32 × 8 sequences = 2.49B tokens | 1× H100 SXM 80 GB | sweep: 3 runs at 25% length; seeds: 3 full runs | **PROJECTED, pending the Module 1 pilot:** 1.96 × 10<sup>18</sup> training FLOPs per run (`frontierlab.flops.run_flops`); at an assumed 30% MFU on 989 TFLOP/s dense BF16 that is $1.96 \times 10^{18} / (0.30 \cdot 989 \times 10^{12}) / 3600 = 1.84$ GPU-hours per run, about 7 GPU-hours in total including the sweep, roughly USD 14–21 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | `pilot-10m` (9.4M non-embedding at vocab 32,768; fewer at the 8,192 vocabulary of a small Data-v0), Data-v0 prepared with `--docs 200000` (about 240M tokens), sequence 512, 4,000 steps × 32 × 2 = 131M tokens | T4, fp32 | as above | PROJECTED: $1.08 	imes 10^{16}$ training FLOPs per run; at an assumed 30% of the T4's 8.1 TFLOP/s fp32 peak, about 1.2 hours per run, about 5 hours in total. Use `--max-minutes` and rerun after disconnects. You will not see Baseline-0's actual noise floor, only the method |
| Free CPU | `toy` (0.8M non-embedding), Data-v0 at the CPU size (24M tokens, vocab 8,192), sequence 128, 600 steps × 16 = 1.2M tokens | laptop | as above | measured on the build laptop (16 threads, 2026-10-03): about 30 minutes in total (sweep about 4 minutes, three 600-step seed runs 16 minutes, Eval v0 about 65 s per run); the noise floor of a model this small says nothing about Baseline-0's |

> [!IMPORTANT]
> The main path needs Data-v0 at its main-path size: `python -m frontierlab.data.prepare --docs 2600000 --vocab 32768` (the script keeps about 12 GB of text in memory and streams 2.6M documents, so use a machine with ample RAM; run it once and keep `meta.json` with the run cards). If you start from the course-provided Baseline-0 checkpoint instead of training it, label every number from it as analysis of a provided artefact, not as your measurement.

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. The fields below are fixed by the project; the rest are yours.

- **Question:** what is the seed-to-seed noise of Baseline-0 on Eval Suite v0, and what is the smallest effect a 3-seed Stage B experiment can detect? Decision informed: the seed count and token budget of every Stage B contract.
- **Hypothesis:** at main-path size the seed std of held-out loss is small (order 0.005–0.02 nats) and the 3-seed MDE is a few hundredths of a nat. Status: may not hold at this scale; the plan's feasibility check (pilot section 12.1) exists because seed variance at ~125M might swamp typical architecture effects.
- **Baseline:** none to beat; this run *is* the baseline. Its tuning: a 3-point learning-rate sweep (0.5×, 1×, 2× of 3e-3) at 25% length, seed 0, chosen on validation. Say in the contract what you will do if the best point is at an edge of the sweep (extend it, with the same budget rule).
- **Changed variable:** the seed only (0, 1, 2) for the seeds stage; the learning rate only for the sweep.
- **Controlled:** Data-v0 hashes, tokenizer, preset, steps, batch, sequence, schedule, Eval v0 version and window seed (1234), LAMBADA revision `900124b…`, software versions.
- **Comparison axis:** not applicable to replicates; for the sweep, equal tokens (all points 25% length).
- **Budget:** fill in GPU type, hours and cost, projected now and measured afterwards.
- **Metrics:** held-out loss over 256 fixed validation windows (primary), LAMBADA target log-probability per passage (secondary, continuous), LAMBADA accuracy (reported for comparison with published numbers). Uncertainty: seed std, standard error of the seed mean, eval-sampling SE within a run, MDE for 2/3/5 seeds.
- **Decision rule, stated now:** for example, "if the 3-seed MDE on held-out loss exceeds 0.02 nats, Stage B contracts use 5 seeds or double the token budget".
- **Correctness checks:** the seed runs are replicates by `python -m frontierlab.record RUN_A RUN_B --replicates`; every run's `eval_v0_val.json` carries the same suite pins (`noise_floor.py` refuses otherwise); a resumed run is identical to an uninterrupted one (lesson 01.1).
- **Fallback evidence:** if the main path is out of reach, the course-provided pilot traces (10M/30M/70M × 3 seeds) — labelled as analysis of provided traces.
- **Limits:** one dataset, one model size, one token budget; the noise floor at 125M does not transfer to other sizes without re-measurement.

## Eval Suite v0

`frontierlab/evals/suite_v0.py`, version `eval-v0.1`:

1. **Held-out loss** on 256 non-overlapping windows of the run's sequence length from the Data-v0 validation split, window seed 1234 — the same tokens for every run, so runs can be paired window by window.
2. **LAMBADA (OpenAI variant)**: 5,153 passages whose last word humans can guess from the whole passage but not from the last sentence alone (Paperno et al. 2016). The file is pinned: Hugging Face `EleutherAI/lambada_openai` at commit `900124bf3b8235c6daf21033af9948b3f07346c4`, `data/lambada_test_en.jsonl`, SHA-256 `4aa8d02c…db226`, MIT licence, 1.8 MB. Scoring follows lm-evaluation-harness's `lambada_openai`: context = every word but the last, target = a space plus the last word; accuracy means greedy decoding reproduces every target token; the target log-likelihood is also kept. With Data-v0's tokenizer, a target averages 2.2 tokens and 24% of targets are one token (measured).

Why LAMBADA and why log-likelihood: it exists, it is small and pinned, it needs no prompt engineering, and it tests something held-out loss averages away (using long-range context to pin down one word). For scale: GPT-2's smallest model (117M parameters, about 40 GB of WebText) scored 45.99% accuracy (GPT-2 paper, Table 3). A ~100M model trained on 2.5B FineWeb-Edu tokens is likely to score much lower, so accuracy will be noisy; the per-passage target log-probability is a continuous score with a much smaller noise floor and is the LAMBADA metric used in decisions. Numbers are not comparable with lm-evaluation-harness results for other models: the tokenizer differs, so "per-token" means something different.

Not in v0, by design: long context (Eval v1, Module 4), instruction following and retention (Eval v2, Module 12), contamination-checked frontier evaluations (Eval v3, Module 18).

## Steps and deliverables

1. **Contract** (deliverable 1). Filled in before step 2, committed next to the runs.
2. **Sweep.** `python labs/module-01/project/run.py --variant main --stage sweep` (or `t4` / `cpu`). Score each sweep run on validation with `evaluate.py RUN` and choose the learning rate by held-out loss. Record what you would have chosen by test, without looking at test: you cannot, which is the point.
3. **Seeds.** `python labs/module-01/project/run.py --variant main --stage seeds --lr <chosen>`. Fill in the `measured:` block of each run card by hand (wall-clock, GPU-hours, cost; [run card template](../templates/run-card.md)).
4. **Evaluate** every seed run on validation: `python labs/module-01/project/evaluate.py RUN` (main path: add `--device cuda --bf16`).
5. **Noise floor** (deliverable 2): `python labs/module-01/project/noise_floor.py runs/m01/main/seeds-* --out reports/m01-noise-floor.json`. Report per metric: seed std, SE of the mean, eval-sampling SE, MDE for 2/3/5 seeds and the A/A interval.
6. **Record** (deliverable 3): run cards for every run, the run-card diff showing the three seeds are replicates, the Data-v0 `meta.json`, `eval_v0_val.json` per run.
7. **Debugging task** (deliverable 4): below.
8. **Written defence** (deliverable 5): below.
9. **Test once.** Only after deliverables 2–5 are written: `evaluate.py RUN --split test` for the three seeds, reported as the Baseline-0 reference numbers of Eval v0.

### What the free CPU variant gave in this build

Measured 2026-10-03 (toy preset, Windows 11 laptop, 16 threads, torch 2.14.1 CPU), as a reference for what your output should look like, not as Baseline-0 numbers. The 150-step sweep chose learning rate 1.5e-3 (validation loss 6.917, against 6.975 at 6e-3 and 6.996 at 3e-3; note that lesson 01.4's 300-step runs used 3e-3: a sweep at 25% length can favour a different value than the full run, which belongs in your limits). With 1.5e-3 and seeds 0–2 at 600 steps:

| Metric | Per seed | Seed std | Eval-sampling SE in a run | MDE, 3 seeds | A/A (seed 1 − seed 0), paired over items |
|---|---|---|---|---|---|
| held-out loss (256 windows) | 6.0321, 6.0211, 6.0008 | 0.016 | 0.018 | 0.036 | −0.011, CI [−0.019, −0.003] |
| LAMBADA target log-prob (5,153 passages) | −16.42, −16.25, −16.30 | 0.089 | 0.079 | 0.204 | +0.172, CI [+0.148, +0.195] |
| LAMBADA accuracy | 0, 0, 0 | 0 | 0 | — | — |

At this scale LAMBADA accuracy is exactly zero for every seed, so it carries no information; the log-probability does. Both A/A intervals exclude zero — two seeds of the same recipe "differ significantly" on the same items — which is the result written-defence question 2 asks you to interpret with your own numbers.

Projected versus measured: put the projected GPU-hours from the table above in your contract and the measured hours in the run cards. If they differ by more than 30%, explain why in one paragraph (your measured MFU, `--compile` or not, evaluation overhead, a different GPU SKU).

## Debugging task

`labs/module-01/project/buggy_evaluate.py` is a colleague's Eval v0 script. Their report: with it, Baseline-0's seed std on held-out loss is larger than in your noise floor, paired comparisons between seeds look no better than unpaired ones, and on LAMBADA the passages with the longest contexts score worst. There are two bugs.

Start from the symptoms, not from a line-by-line comparison with `evaluate.py`. For each bug: name the check that isolates it, run the check, show its output, fix the bug, and show the symptom gone.

<details>
<summary>Hint</summary>

For the held-out symptom: what has to be identical between two runs for a paired comparison to mean anything? Print the window start positions the script uses for two different runs. For the LAMBADA symptom: bucket passages by context length in tokens and compare the score per bucket between the two scripts (use the target log-probability: at small scale accuracy is zero everywhere).

</details>

<details>
<summary>Reference diagnosis</summary>

1. `window_losses(model, val, 256, T, seed=seed)` passes the **run's training seed** as the window seed, so each run is scored on different windows. Seed std then includes eval-sampling noise twice over, and a "paired" bootstrap pairs unrelated windows (no shared difficulty, so no variance reduction). Check: `TokenData("val").eval_windows(256, T, seed)` differs between runs. In the build's CPU run the buggy script gave a seed std of 0.038 on held-out loss against 0.016 with the fixed windows, and its "paired" A/A interval was six times wider (width 0.10 against 0.016). Fix: always use the suite's window seed (1234).
2. `c = c[:max_len - len(t)]` truncates the context **from the right**, keeping the beginning of the passage and cutting the words just before the target, then jumps from the middle of the passage to the target. Only passages longer than the window are affected, which is why long contexts score worst and why lowering `--max-len` hurts far more than it should. Check: score by context-length bucket. In the build's CPU run, passages with 64–125 context tokens scored identically under both scripts (mean target log-probability −16.78 and −16.98), while the 26 passages with at least 126 tokens scored −17.76 under the buggy script against −16.83 under `evaluate.py`. Fix: keep the last tokens, `c[-(max_len - len(t)):]`.

</details>

## Written defence

One to two pages, answering:

1. Your 3-seed MDE on held-out loss: what is it, and which Stage B effects reported in the course's reading (for example the loss differences in Kimi K2's head-count ablation, section 2.3) could a 3-seed experiment at Baseline-0 size detect?
2. Your A/A interval on held-out loss between seeds 0 and 1: does it exclude zero? What does that tell a colleague who wants to compare two architectures with one seed each and a paired bootstrap over windows?
3. Why is the LAMBADA log-probability, not accuracy, the decision metric of Eval v0 at this scale? Use your measured seed std and eval-sampling SE of both.
4. How do you know your learning rate was not chosen with information from the test split? Point to the record.
5. What would make you redo the noise floor (a new data version, a new tokenizer, a different sequence length, a new software version)? Which of these does `frontierlab.record` flag automatically?
6. With 10× the budget, what would you measure next to make the noise floor more useful for Stage B?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the decision rule for Stage B seed counts is written before the runs |
| 2 | Controls | the run-card diff shows the three seeds differ only in `args.seed` (and bookkeeping) |
| 3 | Axis and budget parity | the sweep points have equal tokens; the sweep budget is stated |
| 4 | Correctness | resume check and Eval v0 pins pass; the debugging task's checks are included |
| 5 | Uncertainty | seed std, SE, eval SE and MDE reported for every metric |
| 6 | Conclusion matches evidence | the noise floor is stated for this size and budget only; any provided artefact is labelled as analysis |
| 7 | Limits | what would change the noise floor, and when to re-measure it |
