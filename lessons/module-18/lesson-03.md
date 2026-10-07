---
id: "18.3"
module: 18
minutes: 45
practice_minutes: 100
prerequisites: ["12.4", "16.2", "01.4"]
objectives:
  - Place a benchmark in its lifecycle (new, in use, contaminated, saturated, audited, retired) from primary sources, using SWE-bench Verified, SWE-bench Pro, Humanity's Last Exam, FrontierMath, ARC-AGI-2 and GPQA Diamond as cases, and state the flags a report using it must carry.
  - Fit METR's 50% time horizon by hand from a logistic curve, reproduce METR's published horizons from its released runs, and quantify how much the number moves with the hierarchical bootstrap, task weighting and task sources.
  - Run three contamination checks (n-gram overlap with declared data, Min-K% Prob membership, a fresh matched item set) on a planted leak, and explain from the measurement which check can detect what.
  - Extend Eval Suite v2 to v3 and apply its decision rule, which refuses to compare results that fail their contamination check.
volatility: implementation
sources:
  - title: "Jimenez et al., SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
    url: https://arxiv.org/abs/2310.06770
  - title: "OpenAI, Introducing SWE-bench Verified (2024-08-13)"
    url: https://openai.com/index/introducing-swe-bench-verified/
  - title: "OpenAI, Why we no longer evaluate SWE-bench Verified (2026-02-23)"
    url: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
  - title: "Deng et al. (Scale AI), SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? (dataset overview, results)"
    url: https://arxiv.org/abs/2509.16941
  - title: "Phan et al., Humanity's Last Exam (section 3.1, Table 1)"
    url: https://arxiv.org/abs/2501.14249
  - title: "FutureHouse, About 30% of Humanity's Last Exam chemistry/biology answers are likely wrong (2025-07-23)"
    url: https://www.futurehouse.org/research/hle-exam
  - title: "Glazer et al. (Epoch AI), FrontierMath; and Epoch AI, FrontierMath tiers 1-4 and the OpenAI clarification (2025-01-23)"
    url: https://arxiv.org/abs/2411.04872
  - title: "Chollet et al., ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems (sections 4.3, 5, 6)"
    url: https://arxiv.org/abs/2505.11831
  - title: "Rein et al., GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
    url: https://arxiv.org/abs/2311.12022
  - title: "Kwa et al. (METR), Measuring AI Ability to Complete Long Tasks"
    url: https://arxiv.org/abs/2503.14499
  - title: "METR, Clarifying limitations of time horizon (2026-01-22)"
    url: https://metr.org/notes/2026-01-22-time-horizon-limitations/
  - title: "METR, eval-analysis-public (released runs, commit 52cb829)"
    url: https://github.com/METR/eval-analysis-public
  - title: "Shi et al., Detecting Pretraining Data from Large Language Models (Min-K% Prob, section 3)"
    url: https://arxiv.org/abs/2310.16789
  - title: "Zhang et al., A Careful Examination of Large Language Model Performance on Grade School Arithmetic (GSM1k)"
    url: https://arxiv.org/abs/2405.00332
  - title: "Touvron et al., Llama 2 (appendix A.6: contamination analysis, citing PaLM's 8-gram rule)"
    url: https://arxiv.org/abs/2307.09288
  - title: "Epoch AI, Benchmarking hub"
    url: https://epoch.ai/benchmarks
last_verified: "2026-10-07"
---

# 18.3 · Frontier evaluation

A benchmark is a measuring instrument with a life cycle: it is built hard, used, leaked into training data, saturated, audited and finally retired. This lesson follows that cycle through the evaluations frontier labs report in 2026, then studies two measurement problems in depth: how METR turns agent runs into a time horizon and how uncertain that number is, and how to tell whether a score has been inflated by contamination. You reproduce METR's published horizons from its released runs, plant a leak in your own toy model and see which contamination checks catch it, and extend Eval Suite v2 into v3, which refuses to compare a result that fails its contamination check.

## Why this matters at a frontier lab

Evaluation results decide which checkpoint is promoted, which recipe is kept and what a model card claims. They are also the most common place for a research result to be wrong without any bug in the training code: a benchmark whose tests reject correct fixes, a test set that leaked into the training mix, a metric that saturated a year ago. OpenAI stopped reporting SWE-bench Verified in February 2026, citing flawed tests and contamination, after the benchmark had been the headline coding number for a year and a half. The research engineer's job is to know where each benchmark is in its life, to check for contamination before trusting a gain, and to report long-horizon measurements with the uncertainty their authors attach to them.

## The idea

### The benchmark lifecycle, by example

Facts below are PUBLICLY DOCUMENTED in the cited sources unless marked (S): seen only in search results or secondary coverage because the page refused automated access; check the page yourself.

| Benchmark | Built | Where it is now (2026-10-07) |
|---|---|---|
| **SWE-bench Verified** | 500 SWE-bench tasks screened by 93 professional developers for underspecified issues and tests that reject valid fixes (S) | Retired from OpenAI's reporting on 2026-02-23: an audit of 138 tasks its models kept failing found at least 59.4% with tests that reject functionally correct fixes, and frontier models reproduced gold patches, evidence of contamination (S, company claim). OpenAI recommends SWE-bench Pro |
| **SWE-bench Pro** | 1,865 tasks from 41 repositories: public (731, 11 repos), held-out (858, 12 repos, private), commercial (276, 18 proprietary startup repos). The public set uses only copyleft (GPL) repositories, a legal barrier to inclusion in commercial training corpora | In use. GPT-5 scored 23.3% in the first version of the paper; the public leaderboard's top entry was 61.5 ± 3.1 when checked (company page, no date shown; scaffold, turn limit and cost cap differ between entries) |
| **Humanity's Last Exam** | 2,500 expert-written questions over 100+ subjects, about 14% multimodal, 24% multiple choice; at release the best model (o3-mini high) scored 13.4% with 80% calibration error (Table 1) | In use; published in Nature (2026). Audited: FutureHouse estimated 29 ± 3.7% of the text-only chemistry and biology answers conflict with peer-reviewed literature; the HLE team's own review found about 18% of a subset problematic |
| **FrontierMath** | unpublished expert problems with automatic checking: 300 in Tiers 1–3 and 50 in Tier 4; under 2% solved at release | In use. Epoch AI's 2025-01-23 clarification: OpenAI commissioned the problems and has access to most statements and solutions, with a holdout set; an access asymmetry between developers that any comparison must state |
| **ARC-AGI-2** | grid puzzles kept only if at least two people solved them within two attempts (407 participants); o3 (medium) scored 3.0%, and the authors treat scores below 5% as not meaningful | In use; ARC Prize 2025's top private-set score was 24.0% (S). ARC-AGI-3 moves to interactive game environments |
| **GPQA Diamond** | 198 of GPQA's 448 questions that both experts answered correctly and most skilled non-experts (34% accuracy with web access) did not; experts 65% (74% after their own acknowledged mistakes) | Near saturation: frontier models report above 90% (S, aggregators); multiple choice, so 25% by chance |

Read together: a benchmark's *design* (private sets, copyleft sources, unpublished problems, human-verified solvability) buys time against contamination; its *audit* history tells you how much of the remaining error is in the labels rather than the model; and its *ceiling* tells you whether a difference is still informative. Eval Suite v3's lifecycle registry (`frontierlab.evals.suite_v3.lifecycle`) records all three per benchmark, dated, and turns them into flags a report must carry.

### Time horizons

METR (Kwa et al. 2025) asks how long a task, measured in the time a skilled human takes, an agent can complete. For each model it fits a logistic curve of run success against the log of human task time $t$ (minutes):

$$p(\text{success} \mid t) = \sigma\big(a + b \log_2 t\big), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}, \quad b < 0.$$

$a$ is the intercept, $b$ the change in log-odds per doubling of task length, $\beta = -b$ the slope. The **50% time horizon** $h_{50}$ solves $a + b \log_2 h_{50} = 0$, so $h_{50} = 2^{-a/b}$. The 80% horizon solves $a + b\log_2 h_{80} = \ln 4$ ($\sigma(\ln 4) = 0.8$), so $h_{80} = h_{50} \cdot 2^{-\ln 4/\beta}$: it comes from the same two parameters and is not an independent measurement. Uncertainty comes from a three-level **hierarchical bootstrap**: resample task families, then tasks within each family, then runs within each task, and refit each time.

The original suite had 170 tasks (HCAST 97, RE-Bench 7, SWAA 66) and found horizons doubling about every 7 months from 2019 to 2025. The Time Horizon 1.1 suite (228 tasks) gives a doubling time of 128.7 days [104, 158] since 2023. METR's own note on limitations (2026-01-22) is required reading before quoting any of it: error bars are about 2× in each direction, horizons differ 40–100× across domains, choices about the human baselines can move results by more than 25%, a 50% success rate says little where high reliability is needed, and the suite is nearing saturation for the longest horizons.

### Contamination

Three checks, each answering a different question (`frontierlab.evals.suite_v3.contamination`):

1. **Overlap with the training data.** Needs the data. An item is flagged if enough of its n-grams occur in the training corpus. Published rules differ: GPT-3 flagged 13-gram overlaps (S); PaLM called an item contaminated when at least 70% of its 8-grams appear (as cited in Llama 2, appendix A.6); Llama 2 used token n-grams longer than 10. A canary string in benchmark files (BIG-bench's GUID) lets anyone search a corpus for them.
2. **A membership signal from the model.** Needs only the model. Min-K% Prob (Shi et al.) is the mean log-probability of the $k\%$ least likely tokens of the item ($k = 20\%$ by default): seen text tends to have fewer very unlikely tokens. It is a signal with false positives, compared against items known to be unseen, never proof.
3. **A fresh matched set.** Needs new items. GSM1k wrote new grade-school problems matched to GSM8K; its first version reported accuracy drops of up to 13% for some model families, and the current version says up to 8% with minimal overfitting in frontier models. The gap between published and fresh accuracy, with its interval, is the measurement.

### Eval Suite v3

v3 keeps every v2 component (task, retention, instruction following; the same items, pins and paired decision rule) and adds, per published item, the three contamination checks and, per benchmark, the lifecycle flags. Its rule: **a comparison passes v3 when it passes v2 and neither side fails its contamination check**; a retired benchmark is not reported as a headline; a saturated or audited one is reported with its flag. Contamination is a property of a model and its data, not of a difference between two models, so it is never "paired away".

## Worked example

**A horizon by hand.** A fit gives $a = 3.5$, $b = -0.6$. Then $\log_2 h_{50} = 3.5/0.6 = 5.83$, $h_{50} = 2^{5.83} = 57$ minutes. For 80%: $\log_2 h_{80} = (\ln 4 - 3.5)/(-0.6) = (1.386 - 3.5)/(-0.6) = 3.52$, $h_{80} = 11.5$ minutes. Check with the formula: $h_{80} = 57 \cdot 2^{-1.386/0.6} = 57 \cdot 2^{-2.31} = 57 \cdot 0.20 = 11.5$. A flatter curve ($\beta = 0.4$, as fitted for Claude Opus 4.6) puts $h_{80}$ much further below $h_{50}$: $2^{-1.386/0.4} = 0.09$.

**Min-K% Prob.** An item's token log-probabilities are $[-0.1, -5.0, -0.2, -3.0, -0.3]$; with $k = 40\%$, $\lceil 0.4 \cdot 5 \rceil = 2$ lowest are $-5.0$ and $-3.0$, so the score is $-4.0$.

**PaLM's rule.** The item `.07+35=42` has four character 6-grams; three of them occur in the training index: $3/4 = 0.75 \ge 0.7$, flagged.

**A fresh gap.** 60 of 100 published items correct, 40 of 100 fresh: gap $0.20$. The standard error of a difference of two independent proportions is $\sqrt{0.6 \cdot 0.4/100 + 0.4 \cdot 0.6/100} = 0.069$, so a 95% interval is about $0.20 \pm 0.136 = [0.06, 0.34]$. With 200 items per set and a true gap of 0.04, the half-width is about 0.09: a small leak is invisible to this check at this size.

## Shapes and cost

| Item | Shape / size | Notes |
|---|---|---|
| METR runs (TH1.1) | 24,008 rows × 23 columns; 21 aliases, 228 tasks, 79 families; 15.0 MB | downloaded once, SHA-256 checked |
| one logistic fit | Newton's method on 2 parameters over ~1,000 runs, float64 | milliseconds |
| hierarchical bootstrap | 200 resamples × one fit | measured about 4 s per model on CPU |
| toy contamination items | 200 published + 200 fresh plain-addition problems; log-probabilities (B, 10) float64 | CPU |
| main-path v3 | Eval v2 suite + 500 GSM8K questions and their number-perturbed copies scored for Min-K% + an 8-gram index of the declared post-training shards | **PROJECTED, pending the Module 18 pilot:** 0.4–0.6 GPU-hours per checkpoint on 1× H100 (Module 13's Eval v2 projection of 0.3–0.5 h plus 1,000 forward passes of about 150 tokens); the index build is CPU work of a few minutes |

## Build it

```python
from frontierlab.evals.suite_v3 import horizon as Hz, toy as T3
from frontierlab.evals.suite_v3.core import compare, report
rows = Hz.load_runs(Hz.download("runs/m18/data/metr_th11_runs.jsonl"), "o3 (Inspect)")
Hz.fit_horizon(rows)                       # {'h50': 119.7, 'h80': 30.0, 'beta': 0.69, ...}
Hz.hierarchical_bootstrap(rows, n_boot=200)
res = T3.run_suite(model, training_texts=declared)   # Eval v2 + contamination section
print(report(compare(res_before, res_after)))
```

`horizon` re-implements METR's fit from the paper's description: weighted logistic regression of `score_binarized` on $\log_2$(`human_minutes`) with the `invsqrt_task_weight` column METR publishes, the closed-form horizons, and the three-level bootstrap. It is not METR's code; the lab compares its numbers with METR's. `contamination` has the three checks; `toy` and `hf` attach them to the v2 suites; `core.compare` applies the rule; `lifecycle` holds the dated registry. Tests: `pytest labs/common/tests/test_alignment.py -k "horizon or ngram or suite_v3 or lifecycle"`.

## What the evidence says

- **Benchmark retirement for flawed tests and contamination: PUBLICLY DOCUMENTED** for SWE-bench Verified (company claim; page seen only in search results). **Contamination-resistant design** (private and copyleft sets, unpublished problems): PROMISING; it delays rather than prevents leakage, and access asymmetries (FrontierMath) are a separate problem.
- **Label error in hard benchmarks: PUBLICLY DOCUMENTED** for HLE's chemistry and biology subset by an external audit; the size is disputed between the auditors and the authors.
- **Time horizons growing exponentially: PUBLICLY DOCUMENTED** (METR), with large stated uncertainty; extrapolating them is INFERENCE/SPECULATION.
- **Min-K% Prob and fresh-set gaps: ESTABLISHED as methods**, with known weaknesses: membership signals are noisy and depend on the reference set; fresh sets must match difficulty.
- **Course measurement, METR runs (free CPU, 2026-10-07, TH1.1 runs at commit 52cb829):** the re-implementation reproduces METR's published 50% horizons to the decimal for six models (Claude 3.7 Sonnet 60.4, o3 119.7, GPT-5 203.0, Claude Opus 4.5 293.0, GPT-5.2 352.2, Claude Opus 4.6 718.9 minutes). Hierarchical bootstrap: Claude 3.7 Sonnet [35, 99], GPT-5.2 [186, 689]. Equal task weights instead of METR's weights move o3 to 131 and GPT-5.2 to 398 minutes (+9% and +13%); dropping RE-Bench moves GPT-5.2 to 370. A least-squares doubling time over these seven models is 149 days, against METR's 128.7 [104, 158]. Part A took 15 s.
- **Course measurement, planted leak (free CPU, 2026-10-07, the Module 13 project's toy model):** a fine-tune that showed 100 of the 200 published items 24 times each, mixed with training-split replay, raised published accuracy from 0.775 to 0.945 and fresh accuracy from 0.820 to 0.910. Against the control (the same fine-tune with replay in place of the leak) the leak's own effect was +0.09 on published and +0.05 on fresh items: most of the gain was the extra training, not memorisation. The overlap check flagged all 100 leaked items when the leak was declared and none when it was not; the fresh gap (+0.035 [−0.015, +0.085]) and Min-K% Prob (AUC 0.52 for leaked vs fresh items) did not detect it. Eval v2 alone reported the leaked model as improved on every task component.

## Lab

**Folder:** [`labs/module-18/lesson-03/`](../../labs/module-18/) · **Time:** about 100 minutes (the script runs in 3–5 minutes) · **Pass check:** `pytest labs/module-18/lesson-03` passes; `eval_lab.py` prints your horizons next to METR's with "ok" on every row and the contamination table; your write-up applies the v3 rule to both comparisons.

### Experiment contract

- **Question:** when a model has seen part of an evaluation set, which of three contamination checks detects it at the toy scale: n-gram overlap with declared data, Min-K% Prob membership, or a fresh matched set? Decision informed: which checks Eval v3 may use as verdicts and which only as signals.
- **Hypothesis:** overlap detects every leaked item when the leak is declared and none when it is not; the fresh gap is positive for the leaked model; Min-K% separates leaked from fresh items (AUC above 0.5). **Status:** the overlap result is true by construction; the other two are reported effects whose size at toy scale may not appear.
- **Baseline:** the same fine-tune with training-split replay in place of the leaked items (the control isolates the leak from the extra training), and the untouched model.
- **Changed variable:** whether 100 of the published items are in the fine-tuning data. **Controlled:** the starting checkpoint, 150 steps, batch 64 (16 leaked or replay plus 48 replay), learning rate $10^{-3}$, seed 0, the v3 item sets and pins.
- **Comparison axis:** equal fine-tuning steps and examples.
- **Budget:** free CPU, measured 2.6 minutes for part B (another module's jobs sharing the CPU).
- **Metrics and decision rule:** flagged count; fresh gap with a bootstrap interval, failing when its lower bound exceeds 0.05; membership AUC reported, never a verdict; the leak effect against the control on published and on fresh items.
- **Correctness checks:** your `palm_contaminated` and `min_k_percent` agree with the suite on every item (the script checks); `pytest labs/common/tests/test_alignment.py`.
- **Fallback evidence:** GSM1k's measured gaps and the published Min-K% Prob results (WikiMIA), labelled as published.
- **Limits:** one template of 200 short items; a 0.3M-parameter model that generalises arithmetic rather than memorising strings; one seed; a large real model memorises differently.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 18 pilot | `python labs/module-18/lesson-03/eval_lab.py --variant main --print` prints Eval v3 (`frontierlab.evals.suite_v3.hf`) for the base model and your Module 13 SFT and RLVR checkpoints, and the v3 comparison. **PROJECTED:** 0.4–0.6 GPU-hours per checkpoint, 1.2–1.8 in total. Part A is CPU work in every variant |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --print` (the Qwen3-0.6B-Base pipeline, 200 GSM8K items) |
| Free CPU | laptop; measured 4.4 minutes for all parts with other jobs running | the steps below |

### Steps

1. **Write your contract** (copy the one above, change what you disagree with, keep the hypothesis status).
2. **Implement** the four TODOs (`horizon_from_fit`, `palm_contaminated`, `min_k_percent`, `v3_verdict`) and run `pytest labs/module-18/lesson-03`.
3. **Horizons:** `python labs/module-18/lesson-03/eval_lab.py --part horizon`. Every row must say "ok" and your $h_{50}$ must match METR's column. Then read the sensitivity lines: write down, for GPT-5.2, the spread of $h_{50}$ across the weighting choices and the bootstrap interval, and compare them.
4. **Contamination:** `--part contamination`. Before reading the output, predict each of the four rows. Then explain the two surprises the build found: why the fresh gap stayed inside its interval, and why the leak's effect on fresh items was not zero.
5. **Lifecycle and verdicts:** `--part lifecycle`, then write the v3 verdict for clean → leaked (undeclared) and clean → leaked (declared), and one sentence on why they differ.
6. **Write up** (one page): the horizon table with intervals, your three sentences on what METR's limitations note changes about quoting a single number, the contamination table, and which v3 checks you would let decide and why.

<details>
<summary>Hint for TODO 1</summary>

The curve crosses $p$ where $a + b \log_2 t = \log(p/(1-p))$. Solve for $\log_2 t$ and raise 2 to it. Use the natural logarithm for the log-odds: METR's slope is per doubling of $t$, but the logistic itself is in natural units.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (Windows 11, Python 3.12, torch 2.14.1+cpu, other jobs sharing the CPU): part A 15 s, part B 2.6 minutes.

| Model | runs | h50 (min) | METR's h50 | h80 | β | bootstrap 95% |
|---|---|---|---|---|---|---|
| GPT-4 0314 | 1,401 | 4.0 | — | 0.9 | 0.64 | |
| Claude 3.7 Sonnet | 1,043 | 60.4 | 60.4 | 12.1 | 0.60 | [35, 99] |
| o3 | 1,044 | 119.7 | 119.7 | 30.0 | 0.69 | |
| GPT-5 | 1,056 | 203.0 | 203 | 38.3 | 0.58 | |
| Claude Opus 4.5 | 1,044 | 293.0 | 293 | 49.4 | 0.54 | |
| GPT-5.2 | 961 | 352.2 | 352 | 66.0 | 0.57 | [186, 689] |
| Claude Opus 4.6 | 1,391 | 718.9 | 719 | 69.9 | 0.41 | |

The weighting choice alone moves GPT-5.2 from 352 to 398 minutes, inside the bootstrap interval but a 13% change in a headline number. Opus 4.6's flatter slope (β = 0.41) means its 80% horizon is a tenth of its 50% horizon: the two numbers describe different reliabilities, and quoting only $h_{50}$ hides that.

| Model | v2 add_greedy | published | fresh | gap [95% CI] | flagged | AUC | v3 |
|---|---|---|---|---|---|---|---|
| clean | 0.620 | 0.775 | 0.820 | −0.045 [−0.125, +0.035] | 0 | 0.52 | pass |
| replay only (control) | 0.840 | 0.855 | 0.860 | −0.005 [−0.070, +0.065] | 0 | 0.51 | pass |
| leaked, leak declared | 0.820 | 0.945 | 0.910 | +0.035 [−0.015, +0.085] | 100 | 0.50 | FAIL |
| leaked, leak undeclared | 0.820 | 0.945 | 0.910 | +0.035 [−0.015, +0.085] | 0 | 0.50 | pass |

The leak's own effect (leaked minus control) was +0.09 on published and +0.05 on fresh items. A 0.3M-parameter model learns arithmetic from 100 extra problems rather than memorising them, so only 0.04 of the gain is specific to the leaked items, below what 200 fresh items can resolve. Min-K% saw nothing for the same reason: the leaked strings are not unusually likely. The only check that caught the leak needed the leak to be declared. On a large model that memorises verbatim text the membership signal is reported to be stronger (Shi et al.); this toy shows why a fresh set and a membership score are signals, and why the training data must be declared for a verdict.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-18/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-18/lesson-03`.

</details>

## Common mistakes

- **Quoting a benchmark without its lifecycle.** A retired or saturated benchmark can still be computed; it can no longer support the claim.
- **Treating $h_{50}$ as how long a model works unsupervised.** It is the human-time length of tasks the model completes half the time; METR says so explicitly.
- **Quoting $h_{80}$ as a second measurement.** It is the same fit read at another height.
- **Calling a model clean because Min-K% found nothing.** Membership signals have false negatives, as the lab measured; only declared-data overlap can give a verdict.
- **Comparing a fresh set that is easier or harder than the published one.** The gap then measures difficulty, not contamination.
- **Letting Eval v2's "improved" stand without the contamination section.** The leaked model passed v2 on every component.

## References

- C. Jimenez et al., *SWE-bench*, 2023. https://arxiv.org/abs/2310.06770
- OpenAI, *Introducing SWE-bench Verified*, 2024-08-13, and *Why we no longer evaluate SWE-bench Verified*, 2026-02-23 (both seen through search results; check the pages). https://openai.com/index/introducing-swe-bench-verified/ ; https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- X. Deng et al., *SWE-Bench Pro*, 2025, dataset overview and results. https://arxiv.org/abs/2509.16941 ; leaderboard https://labs.scale.com/leaderboard/swe_bench_pro_public
- L. Phan et al., *Humanity's Last Exam*, 2025, section 3.1 and Table 1. https://arxiv.org/abs/2501.14249
- FutureHouse, *HLE chemistry and biology audit*, 2025-07-23. https://www.futurehouse.org/research/hle-exam
- E. Glazer et al., *FrontierMath*, 2024; Epoch AI, tiers 1–4 page and *OpenAI and FrontierMath*, 2025-01-23. https://arxiv.org/abs/2411.04872 ; https://epoch.ai/frontiermath/tiers-1-4/about ; https://epoch.ai/blog/openai-and-frontiermath
- F. Chollet et al., *ARC-AGI-2*, 2025, sections 4.3, 5 and 6. https://arxiv.org/abs/2505.11831 ; ARC-AGI-3: https://arcprize.org/competitions/2026/arc-agi-3
- D. Rein et al., *GPQA*, 2023. https://arxiv.org/abs/2311.12022 ; Epoch AI, GPQA Diamond page: https://epoch.ai/benchmarks/gpqa-diamond
- T. Kwa et al. (METR), *Measuring AI Ability to Complete Long Tasks*, 2025. https://arxiv.org/abs/2503.14499
- METR, *Clarifying limitations of time horizon*, 2026-01-22. https://metr.org/notes/2026-01-22-time-horizon-limitations/ ; Time Horizon 1.1 results: https://metr.org/assets/benchmark_results_1_1.yaml
- METR, *eval-analysis-public*, commit `52cb829c7a2efb2d659285c4b1768d191d97f8d2` (no licence stated: download, do not redistribute). https://github.com/METR/eval-analysis-public
- W. Shi et al., *Detecting Pretraining Data from Large Language Models*, 2023, section 3. https://arxiv.org/abs/2310.16789
- H. Zhang et al., *A Careful Examination of Large Language Model Performance on Grade School Arithmetic*, 2024. https://arxiv.org/abs/2405.00332
- H. Touvron et al., *Llama 2*, 2023, appendix A.6. https://arxiv.org/abs/2307.09288
- Epoch AI, *Benchmarking hub*. https://epoch.ai/benchmarks
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[18.4 · Safety frameworks and system cards](lesson-04.md)
