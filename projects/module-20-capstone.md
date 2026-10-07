# Module 20 project · Capstone: reproduce, extend, defend

This is the course's capstone: one claim from a 2025–26 report, reproduced at a scale you can afford, extended with one new ablation, and defended in writing after review. It starts from your [Module 19 capstone proposal](module-19-capstone-proposal.md) and its filled experiment contract, uses the claim list, package format and checker of lesson 20.1, and the review, defence and revision procedure of lesson 20.2. It is graded on soundness, never on whether the method you test wins: a sound null result scores exactly as well as a positive one.

**Time:** 30–50 attended hours (plan section 8) plus your claim's compute: free CPU 15–90 minutes of runs depending on the claim (lesson 20.1's list), or the main path's PROJECTED 2–44 GPU-hours. **Folder:** [`labs/module-20/project/`](../labs/module-20/) (`capstone_project.py`, `pieces.py`, `buggy_capstone.py`, `test_pieces.py`) and your own package. **Assessment:** self-check against the rubric below (the course rubric in [`templates/experiment-rubric.md`](../templates/experiment-rubric.md) plus three capstone criteria); the module quiz covers the same material.

## Variants and cost

| Variant | What you run | Hardware | Cost |
|---|---|---|---|
| Main path | your claim's main path from lesson 20.1 (for QK-Clip: `capstone_lab.py --variant main --print`, then the pilot-70m and Baseline-0 rungs) | 1× H100 80 GB (one claim needs 1× A100 or H100 80 GB) | **PROJECTED, pending the Module 20 pilot:** 1.5–2.5 (micro-anneals) to 27–44 (GSPO) GPU-hours; about 20 for QK-Clip; USD 3–130 at USD 2–3 per H100-hour. Add every rerun the review requires, projected with the same formula |
| Free GPU (Colab/Kaggle T4) | the claim's T4 variant | T4 | PROJECTED 1.5–10 hours across sessions, per the claim record |
| Free CPU | the claim's CPU variant; for QK-Clip exactly lesson 20.1's scaffold run | laptop | 15–90 minutes of runs (QK-Clip measured in lesson 20.1); `pytest labs/module-20/project` under a second |

## Deliverables

1. **The package** (lesson 20.1's format): `claim.yaml`, `contract.md`, `runs/<arm>-s<seed>/` with run cards whose parents trace back into the course's artefact chain, `results.json`, `claims.yaml`, `report.md`, `checks.json` (your correctness checks' output). `python -m frontierlab.capstone <package>` reports no problems.
2. **The code**: the scripts that produced every run and number, committed, with the exact commands in the report. Reuse `frontierlab` and wrap it in your own module or script; do not edit shared files.
3. **The report** (3–5 pages): the claim as its source states it (section or figure), what your scale can and cannot test, the reproduction and the extension with their intervals and decisions, the noise floor and MDE, the secondary measurements, and a Limits section that says what result would change the conclusion.
4. **The review folder**: `review/questions.md` from `labs/module-20/lesson-02/review_lab.py --package <your package>` (plus any questions a classmate added), `defence.md` answering every one, and `revision_log.yaml`; `review_lab.py --package <your package> --step check` passes.
5. **The debugging report** (below).

## The experiment contract

Start from the contract in your Module 19 proposal and complete it with [the template](../templates/experiment-contract.md). Fixed by the capstone:

- **Question:** one claim from lesson 20.1's list, quoted from its source with section or figure, and the decision it informs. Define in advance what counts as *reproduced*, *not tested* and *not reproduced* at your scale.
- **Hypothesis:** one per comparison (reproduction and extension), each with its status: established effect, reported effect (cite), or may not appear at this scale.
- **Baseline:** a run card in the course chain (Baseline-0, a Module 7 or 14 run, the Module 13 model) and its tuning budget, equal to every other arm's.
- **Changed variable:** one per comparison, declared in `claim.yaml` as the run-card keys it changes. **Controlled:** data and hashes, tokens, seeds (at least 2; 3 on the main path), evaluation version and items, software.
- **Comparison axis:** the one the claim needs (equal training FLOPs including teacher compute for distillation; equal tokens for the others), and what it does not answer.
- **Budget:** GPU type, GPU-hours (PROJECTED with the formula, then measured), teacher, judge or generator compute where used.
- **Metrics and decision rule:** the primary metric with the paired hierarchical bootstrap over seeds and items and the seed t-interval; the equivalence margin and the rule, stated now; the noise floor and MDE from the earlier lesson's measurements.
- **Correctness checks:** the earlier module's checks for the mechanism (for example QK-Clip's cap, DSA's indexer KL and causal checks, mHC's doubly stochastic error, the distillation ledger's totals), the package checker, `pytest labs/common/tests/test_capstone.py`.
- **Fallback evidence:** the published figures and the course pilot traces, labelled analysis.
- **Limits:** scale, data, architecture, hardware, and the part of the claim your scale cannot test.

## Debugging task

`labs/module-20/project/buggy_capstone.py` is a colleague's rewrite of the three analysis pieces in `pieces.py`: pairing two arms' per-window losses by seed, the paired interval, and the decision. Their message is at the top of the file: with their pieces, the QK-Clip reproduction "now comes out 'a higher' instead of 'equivalent'", so clipping costs loss after all, and the extension's interval is "much tighter". There are three bugs, and only two of them change the build's package: the third waits for a package whose runs finished in a different order. For each, name the sentence of the report it changes, the test that isolates it, and the fix.

```bash
CAPSTONE=buggy pytest labs/module-20/project                                       # which checks fail
CAPSTONE=buggy python labs/module-20/project/capstone_project.py --package runs/m20/l201/capstone-qkclip
python labs/module-20/project/capstone_project.py --package runs/m20/l201/capstone-qkclip   # the course pieces
```

<details>
<summary>Hint</summary>

Ask of each piece what it assumes about the two dictionaries' order, what population the interval resamples, and in which order the decision rule's cases are tested.

</details>

<details>
<summary>The three bugs (after your own diagnosis)</summary>

1. `decide` tests "excludes zero" before "inside the margin". The build's reproduction interval, +0.0015 [+0.0013, +0.0018] nats, excludes zero but lies far inside the 0.02 margin, so the rule gives "equivalent"; the buggy order gives "a higher", and the report's sentence becomes "clipping costs loss". A difference too small to matter is reported as a cost. `test_decide_checks_equivalence_first` isolates it. Fix: test equivalence first, as the contract states.
2. `interval` flattens seeds × windows and resamples the 768 values as if independent, so seed-to-seed variation disappears. On the build's package the extension's interval shrinks from [+0.0056, +0.0332] to [+0.0166, +0.0246]: the "much tighter interval" is the symptom, and it hides that the seed t-interval, [−0.017, +0.058], includes zero. `test_interval_resamples_seeds` isolates it. Fix: resample seeds, then windows within them.
3. `pair` takes `dict.values()` in insertion order. In the build's package every arm's runs were evaluated in seed order, so nothing changes; in a package whose runs finished or were resumed in different orders, seed 0 of one arm is paired with seed 2 of the other and the "paired" difference contains the seed-to-seed variation it was meant to remove. `test_pairs_by_seed_not_by_insertion_order` isolates it. Fix: index both by the sorted shared seeds. A bug that the current data does not show is still a bug; the test is what catches it.

</details>

## Written defence

Answer every question `review_lab.py` generates for your package (generic, claim-specific and problem questions), plus these five, in 1–2 pages:

1. Which part of the source's claim does your scale test, and which part not at all? Quote the source's setting.
2. Did the mechanism engage (the clip fired, the instability appeared, the teacher's scoring was charged)? Show it from the run logs.
3. What is your minimum detectable effect, and how does it compare with the effect the source reports?
4. Which result changed during revision, and how does your contract record it?
5. What would you run next with ten times the budget, and which result would make you abandon the claim?

## Self-check against the rubric

Score each criterion 0 (missing), 1 (partly) or 2 (done). Pass: 15 of 20 with no zero. A sound null scores the same as a positive result.

| # | Criterion | 2 = done looks like |
|---|---|---|
| 1 | Question and decision | the claim quoted with its section or figure; reproduced / not tested / not reproduced defined before the runs |
| 2 | Controls | one declared changed variable per comparison; the run cards show everything else equal |
| 3 | Axis and budget parity | the axis justified; budgets and tuning budgets equal, teacher or judge compute counted |
| 4 | Correctness | the mechanism's checks and the package checker passed, output included |
| 5 | Uncertainty | at least 2 seeds (3 on the main path); hierarchical interval and seed t-interval; noise floor and MDE |
| 6 | Conclusion matches evidence | every claim labelled; measured claims follow the rule and stay at course scale |
| 7 | Limits | scale, data, architecture and what would change the answer |
| 8 | Reproduction scoped | the testable and untestable parts of the claim separated, and the mechanism shown to engage |
| 9 | One extension, powered | one new ablation with its own comparison, seeds and budget |
| 10 | Defence and revision | every question answered with evidence; a complete revision log; changes after results recorded |

- [ ] `python -m frontierlab.capstone <package>` reports no problems.
- [ ] `review_lab.py --package <package> --step check` passes.
- [ ] The debugging report names each bug by the sentence of the report it changes.
- [ ] Every main-path figure you did not measure is labelled PROJECTED with its formula; every measured one names the machine and versions.

<details>
<summary>Reference solution sketch (the QK-Clip capstone)</summary>

A reference capstone for the QK-Clip claim, at free-CPU scale, is what lesson 20.1's scaffold produces in `runs/m20/l201/capstone-qkclip` plus a review. In outline:

- **Claim and scope.** Kimi K2 Appendix D (Figure 12): QK-Clip at an aggressive tau has negligible impact on loss for 0.5B-activated / 3B-total MoE models trained with Muon. Testable at toy scale: whether clipping at a tau that binds costs held-out loss. Not testable: logit explosion beyond 1,000, MoE, MLA (the toy uses GQA, where only query rows are scaled; a course choice). Extension: QK-norm against QK-Clip at equal tokens, which neither K2 nor DeepSeek-V4 reports.
- **Design.** Arms `muon-noqk` (baseline, parent `m07-l72-muon-noqk`), `muon-clip` (reproduction), `muon-qknorm` (extension); seeds 0–2; 200 steps × 8 × 128 tokens at lr $10^{-2}$; tau by the rule round(0.6 × the seed-0 baseline's final maximum logit), stated before the clip runs; margin 0.02 nats; 256 validation windows of 128 tokens; equal tuning (one rate for every arm).
- **Correctness.** QK-Clip caps recomputed per-head logits at min(S_max, tau) to $3.6\times10^{-15}$ in float64; it refuses a QK-norm model; causal and cached-decode checks pass; the package checker reports no problems.
- **Results.** As measured in lesson 20.1's lab section: the reproduction and extension intervals, their decisions under the rule, the noise floor, the clip's first firing step and the number of clipped head-updates (the evidence that the mechanism engaged), maximum logits per arm.
- **Claims.** MEASURED, course scale: one sentence per comparison, worded by its decision. PUBLICLY DOCUMENTED: K2's Appendix D statement, with its section. Nothing about MoE or about K2's scale.
- **Limits.** A 1.8M-parameter dense model for 200 steps; logits of order 10 against K2's 1,000; one learning rate; GQA. A different decision at the pilot-70m rung would change the conclusion.
- **Review.** Expected questions: why tau is not 100 (because at this scale logits never reach it and the claim would be untested, lesson 07.2); whether the clip fired (first clip step and count from `stability.jsonl`); why QK-norm is one changed variable although it adds parameters (the declared variable is "the logit fix"; parameter counts follow from it and the run-card diff classifies them as such); what the MDE is. Revision: none of the rerun kind if the package passed the checker; the defence concedes that the toy cannot test logit explosion.

</details>
