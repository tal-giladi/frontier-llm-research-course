# Module 10 project · Data-v1: a recipe in which every choice has an ablation behind it

Data-v0 is one pinned slice of FineWeb-Edu. Data-v1 is the data recipe Module 11 trains Recipe-R on: which sources, at which weights, filtered how, with or without rephrased data and document masking, and with which final anneal mix. This project assembles it from the decisions of lessons 10.1–10.5, writes the provenance manifest and the leakage report that make it shippable, and runs one final comparison against Data-v0 at equal tokens. Every choice in the recipe file must point at the ablation (with its interval) that supports it, or say plainly that it is a default without evidence.

**Time:** 6–8 attended hours plus unattended runs. **Folder:** [`labs/module-10/project/`](../labs/module-10/) (`run_project.py`, `data_v1_example.json`, `buggy_report.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Data and model | Hardware | Cost |
|---|---|---|---|
| Main path | Data-v0 and every Module 10 source prepared with the vocabulary-32,768 tokenizer (`sources prepare ... --tokenizer <main-path tokenizer>`; web pool of 2M documents, PROJECTED a few GB streamed); Baseline-0, 9,500 steps × 32 × 1,024 tokens (2.49B), 2 recipes × 3 seeds | 1× H100 80 GB | **PROJECTED, pending the Module 10 pilot:** 6 runs × 2.49e9 tokens × 0.788 GFLOP/token ÷ (989e12 FLOP/s × an assumed 0.30 MFU) = 11 GPU-hours, plus the lesson ablations whose results you reuse (about 6 GPU-hours, lessons 10.1–10.5 main-path rows), about USD 35–50 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | `--variant t4`: pilot-10m, 4,000 steps × 32 × 512 | T4, fp32 | PROJECTED about 1 hour per run; use `--max-minutes` and rerun the same command after a disconnect |
| Free CPU | `--variant cpu`: toy, 600 steps × 16 × 128 per run | laptop | measured: about 35 minutes (below) |

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. Fixed by the project:

- **Question:** at equal training tokens, does Data-v1 lower mean held-out loss over its sources' validation sets without costing more than 0.02 nats on Eval v0's held-out set? Decision informed: whether Module 11's Recipe-R run uses Data-v1 or Data-v0.
- **Hypothesis:** yes for the mean (Data-v1 contains the domains being evaluated; Data-v0 does not), with a small cost on Eval v0 (fewer FineWeb-Edu tokens). Status: expected by construction for the mean; the guard is the real test. Say so in your contract: a recipe evaluated on its own sources' validation sets is favoured, which is why the guard and LAMBADA are reported too.
- **Baseline:** Data-v0 (100% FineWeb-Edu), the Module 1 recipe and learning rate.
- **Changed variable:** the data recipe as a whole (a declared bundle of choices, each backed by its own lesson ablation). **Controlled:** model, tokenizer, steps, batch, sequence length, schedule, seeds 0–2, evaluation windows and LAMBADA file.
- **Comparison axis:** equal training tokens. Not answered: the cost of preparing Data-v1 (classifier scoring, rephrasing compute); report it from lessons 10.2 and 10.3.
- **Metrics and decision rule:** primary: seed-level paired 95% interval of (v1 − v0) on the mean held-out loss over the sets; guard: the same on Data-v0 validation (Eval v0). Adopt Data-v1 if the primary upper bound is below 0 and the guard upper bound is at most +0.02; reject if the primary lower bound is above 0 or the guard lower bound exceeds 0.02; else inconclusive. Report the seed standard deviation and the 3-seed MDE.
- **Correctness checks:** `pytest labs/common/tests/test_datax.py`; the manifest has no BLOCK finding; the integrity report has zero exact cross-split duplicates and a planted-control recall of 1.0, and every near-duplicate pair is resolved; every run's `mixture_accounting.json` equals its planned accounting; `python -m frontierlab.record` on a seed pair shows only the declared differences.
- **Fallback evidence:** the Module 10 pilot traces (Colab) when published, labelled as analysis of provided traces.
- **Limits:** state them for each component (they are the limits of the lesson ablations) and for the bundle: one size, 3 seeds, evaluation sets drawn from the same sources as the training data.

## Steps and deliverables

1. **Contract** (deliverable 1), before any run.
2. **Recipe file** (deliverable 2). Copy `labs/module-10/project/data_v1_example.json` and change it to your decisions. Every source and weight, the filter, the rephrased source (if any), the document-mask flag and the anneal candidates get an `evidence` string: the lesson, the run folder and the interval. A choice without an ablation says "default, no evidence".
3. **Provenance and leakage report** (deliverable 3): run the project script; it writes `runs/m10/project/manifest.json` and `integrity.json`. Write one page: the licences and what they oblige (attribution list; share-alike sources and what that means for releasing Data-v1); every near-duplicate pair and what you did with it; the leakage numbers of Eval v0, Eval v1 and LAMBADA with $n$, threshold, too-short counts and the planted-control recall.
4. **Final comparison** (unattended):

   ```bash
   python labs/module-10/project/run_project.py --recipe my_data_v1.json --variant cpu
   python labs/module-10/project/run_project.py --recipe my_data_v1.json --variant main --lr <your Module 1 rate> --print
   ```

   The main-path line prints the six training commands (not run in this build; part of the Module 10 pilot).
5. **Report** (deliverable 4): the comparison table, the decision, the seed std and MDE, the per-source accounting and epochs; then a table "choice → evidence → interval → decision" for every component, with nulls reported as nulls and their MDEs.
6. **Debugging task** (deliverable 5), below.
7. **Written defence** (deliverable 6), below.

### What the free CPU variant gave in this build

Measured 2026-10-05 on the build laptop (torch 2.14.1 CPU, 16 threads) with `data_v1_example.json` (natural token shares: Data-v0 0.33, FineWeb 0.23, Wikipedia 0.24, FineMath 0.20; no filter, no rephrased source, no document mask): 33 minutes in all, of which the integrity report 15 minutes and six runs of 600 steps × 16 × 128 tokens 159–169 s each.

- **Manifest:** 4 sources, 0 BLOCK, 1 WARN: Wikipedia is CC BY-SA 3.0 and GFDL, share-alike.
- **Integrity:** 0 exact cross-split duplicates for every source and split; 0 verified near-duplicates (2 LSH candidates below the threshold); leakage 0 of 256 Eval v0 windows, 0 of 5,153 LAMBADA passages, 1 of 648 Eval v1 haystack documents flagged (13-grams, overlap ≥ 0.5); planted-control recall 20/20.
- **Accounting (seed 0):** Data-v0 405,504, FineWeb 282,624, Wikipedia 294,912, FineMath 245,760 tokens, each 0.017 epochs (natural shares give every source the same epoch count).

| Set (v1 − v0, 3 seeds) | difference | 95% CI |
|---|---|---|
| mean of the four held-out losses (primary) | −0.381 | [−0.521, −0.241] |
| Data-v0 validation (Eval v0, the guard) | −0.034 | [−0.236, +0.168] |
| FineWeb validation | −0.099 | [−0.278, +0.081] |
| Wikipedia validation | −0.178 | [−0.294, −0.062] |
| FineMath validation | −1.214 | [−1.318, −1.111] |
| LAMBADA log-probability (higher is better) | +0.29 | [−0.31, +0.89] |

Decision under the rule: **inconclusive**. The primary metric clearly favours Data-v1, but most of that is FineMath, a domain Data-v0 never saw (the self-evaluation bias the contract warns about); the guard interval runs from −0.24 to +0.17 and so cannot show that Data-v1 costs at most 0.02 nats on Eval v0. Seed standard deviation of Data-v0 on the mean held-out loss 0.032; unpaired 3-seed MDE 0.073; standard deviation of the paired differences 0.056. The guard is the noisy number here, and the defensible next step is more seeds on the guard, fixed before running, not a different rule.

## Debugging task

`labs/module-10/project/buggy_report.py` is a colleague's Data-v1 report. Their summary: Data-v1 (Data-v0 plus FineWeb plus a small "edu-extra" source) lowers Eval v0 held-out loss clearly, their leakage check found nothing, and the Data-v1 run trained on exactly the planned tokens per source. There are two bugs. Start from the symptoms; for each, name the check that isolates it, run it, and show what the fixed pipeline reports.

```bash
python labs/module-10/project/buggy_report.py
python labs/module-10/project/buggy_report.py --fixed     # only after your diagnosis: the corrected pipeline
```

<details>
<summary>Hint</summary>

For the leakage half: which split of each source does the mixture train on, and which split did their n-gram index read? `python -m frontierlab.datax.provenance manifest runs/m10/project-buggy/data-v1.json` is one command away. For the accounting half: look at the learning rate in `runs/m10/project-buggy/v1-part2/metrics.jsonl` at its first steps, and at `datax.init_from` in its run card. What window does the mixture stream of a *new* run start from?

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-05 on the build laptop (CPU; `buggy_report.py` 526 s, `--fixed` 383 s). The colleague's report prints: 0 of 128 Eval v0 windows flagged, Data-v1 better on Eval v0 by −0.202 nats [−0.228, −0.175], and tokens per source edu 163,840, web 40,960, edu-extra 204,800, "exactly as planned".

**Bug 1: the training mix contains the evaluation set, and the leakage check looked elsewhere.** `data-v1.json` declares `edu-extra` as Data-v0 with `"split": "val"`: the mixture trains on Eval v0's own validation documents (0.95 epochs of them in the second part of the run alone). The leakage check built its n-gram index from `TokenData("train", s.root)` for every source, so it indexed Data-v0's *training* split twice and never saw what the mixture actually reads. Isolating check: the manifest (`python -m frontierlab.datax.provenance manifest runs/m10/project-buggy/data-v1.json`) lists each source's split; an index built from each source's own split flags **128 of 128** Eval v0 windows. The −0.20 "gain" is memorisation of the test set. Lesson 10.1's rule applies: the leakage check must read exactly the data the sampler reads, and its zeros mean nothing without a planted control.

**Bug 2: the "continued" run is not a continuation.** The second part was launched in a new folder with `--init-from`, which loads only the weights: its `metrics.jsonl` starts at the peak learning rate again (3e-3 at step 20, then a fresh cosine), the optimizer moments are new, and its mixture stream starts at window 0, so the first 3,200 windows were trained twice. Its run card shows `init_from: v1-part1/checkpoint.pt`, `init_step: 200`, and its accounting covers 200 steps (409,600 tokens), not the 400 the report claims. Isolating check: compare the first `lr` values of `v1-part2/metrics.jsonl` with the schedule of a 400-step run, and the card's `init_from` with the resume procedure of lesson 10.1. Fix: rerun the same command in the same run folder; the loop then resumes optimizer, schedule, RNG and the sampler's window counter exactly.

**The fixed pipeline** (`--fixed`: edu-extra dropped, Data-v0 0.9 + web 0.1, stopped at 200 and resumed in the same folder): Eval v0 loss 6.403 against Data-v0's 6.461, paired difference −0.058 [−0.068, −0.049], a third of the reported gain, and still one seed with a window-level interval, which says nothing about seed noise (lesson 01.4's point); accounting over the whole run edu 737,280 and web 81,920 tokens, 819,200 = 400 × 16 × 128. A defensible report would rerun both arms with 3 seeds before claiming anything.

</details>

## Written defence

One to two pages, answering:

1. Which of your recipe's choices rest on an inconclusive or null ablation? For each, what is its MDE, and why did you keep, drop or default it?
2. Your final comparison scores Data-v1 on its own sources' validation sets. Why does that favour Data-v1, and which reported number is free of that bias?
3. Which source in Data-v1 has the highest epoch count over a Module 11-length run, and is it past the point where repetition loses value? What would you change if it is?
4. What does your leakage report *not* rule out (for example paraphrased benchmark content, or benchmarks you did not check)?
5. One source is share-alike. What does that oblige if Data-v1 or a model trained on it is released, and what would you do if the obligation were unacceptable?
6. With 10× the budget, which single ablation would you rerun at larger scale first, and what result would make you change Data-v1?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the adoption rule and guard are written before the runs; the recipe file exists before the final comparison |
| 2 | Controls | run cards show the same model, steps, seeds and evaluation pins; the recipe is the declared changed bundle |
| 3 | Axis and budget parity | equal tokens verified from `budget.tokens` and the accounting files; data-preparation compute reported separately |
| 4 | Correctness | `test_datax.py`, manifest without BLOCK, integrity report with planted-control recall 1.0, resume evidence |
| 5 | Uncertainty | every number with a seed-level interval; seed std and MDE stated; nulls reported with their power |
| 6 | Conclusion matches evidence | "Data-v1 is better" is claimed only on the measured axis and sets, with the self-evaluation bias stated |
| 7 | Limits | scale, seeds, sources, evaluation sets, and the licence obligations |
