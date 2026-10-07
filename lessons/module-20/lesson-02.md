---
id: "20.2"
module: 20
minutes: 35
practice_minutes: 120
prerequisites: ["20.1", "19.3"]
objectives:
  - Generate the reviewer questions a capstone must answer, from the rubric, from the specific claim and from every problem the soundness checker finds.
  - Classify each weakness by the revision it needs (rerun, reanalyse, rewrite or declare) and do the revisions in an order that cannot hide a changed result.
  - Write a defence that answers every question with evidence from the package, concedes what the evidence does not support, and adds no new claims.
  - Keep a revision log that a second reviewer can check against the package, including every result that changed after review.
volatility: concept
sources:
  - title: "Cortes and Lawrence, Inconsistency in Conference Peer Review: Revisiting the 2014 NeurIPS Experiment (2021)"
    url: https://arxiv.org/abs/2109.09774
  - title: "Pineau et al., Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)"
    url: https://arxiv.org/abs/2003.12206
  - title: "Google Research, Deep Learning Tuning Playbook (comparing to a tuned baseline)"
    url: https://github.com/google-research/tuning_playbook
  - title: "Zheng et al., Group Sequence Policy Optimization (sections 5.1 and 5.3: the setting the sample capstone reviews)"
    url: https://arxiv.org/abs/2507.18071
last_verified: "2026-10-07"
---

# 20.2 · Review and defence

A capstone is not finished when its runs end; it is finished when it has survived a reviewer. This lesson turns review into a procedure: where reviewer questions come from (the rubric, the claim, and every mechanical problem in the package), what kind of revision each weakness needs, in which order to revise so that a changed result cannot slip through, how to write a defence that concedes what it must, and how to keep a revision log that someone else can check. You practise on a sample capstone with ten planted weaknesses, then apply the same procedure to your own package.

## Why this matters at a frontier lab

At a frontier lab a result that informs a large run is reviewed before anyone commits compute: by the team that owns the baseline, by someone who has been burned by the same mistake, sometimes by a formal review committee. The review is adversarial on purpose. Its job is to find the strongest reason the conclusion could be wrong while it is still cheap to check. Engineers who treat review as an attack defend everything and fix nothing; engineers who treat it as a checklist fix the wording and miss that a rerun changed the answer. The habit this lesson builds is the one that makes a result trusted: every question answered with evidence, every concession made in writing, every change recorded where the next reader will see it.

Review is noisy even at its best. In the 2014 NeurIPS experiment, where two independent committees reviewed the same papers, Cortes and Lawrence estimate that about half of the variation in reviewers' quality scores was subjective, and that the process was good at identifying poor papers but poor at identifying good ones (PUBLICLY DOCUMENTED, 2021 re-analysis). That is an argument for making as much of the review mechanical as possible, so that the subjective part is spent on the questions only a person can answer.

## The idea

### Where the questions come from

A reviewer's questions come from three places, and lesson 20.1's package lets you generate all three:

1. **The rubric** (generic, asked of every capstone): why this comparison axis and what it does not answer; how you know the baseline was tuned as hard as the new method; the smallest effect your design could detect; what you would run with ten times the budget, and what result would make you abandon the claim.
2. **The claim** (specific to what you reproduced). Each claim in `frontierlab.capstone.claims` carries three. For example:

   | Claim | A question only this claim raises |
   |---|---|
   | GSPO vs GRPO | GSPO clips far more tokens by design: which stability metric could that bias, and in which direction? |
   | QK-Clip vs QK-norm | How many head-updates did the clip rescale, and from which step? If it never fired, what did you test? |
   | DSA vs dense | At what context length would the sparse path be cheaper on your hardware, and is that measured or projected? |
   | mHC vs HC | Your stability metric is the composite gain, not a loss spike: why is that evidence about stability at all? |
   | Micro-anneals | Your control anneal also lowered loss: how much of each candidate's gain is the anneal itself? |
   | On-policy distillation | Which FLOPs did you count for the teacher in each arm, and does the comparison survive charging the teacher's training? |

3. **The package's problems.** Every problem the checker finds becomes a question in a reviewer's words: a `TUNING` problem becomes "how do you know the baseline was tuned as hard as the arm you favour?", a `CLAIM_SCOPE` problem becomes "what evidence at that scale do you have?". `frontierlab.capstone.review.questions(pkg)` produces all three groups with stable ids (`G1`–`G4`, `C1`–`C3`, `P1`...), so the defence and the revision log can refer to them.

The mechanical questions are not the hard ones. A package can pass the checker and still have a baseline tuned on the test split, a margin chosen after seeing the interval, or a reproduction of the wrong half of the claim. The claim-specific and generic questions are where those surface, and they are where your written answers carry the weight.

### Four kinds of revision

Each weakness needs one kind of revision, and mixing them up is the commonest way to revise badly:

| Kind | What is wrong | What fixes it | Problem codes |
|---|---|---|---|
| **rerun** | the evidence itself: arms not comparable, too few seeds, unequal tuning, a run missing | new runs, with run cards | `PARITY`, `TUNING`, `SEEDS`, `UNC_SEEDS`, `RUNCARD_MISSING` |
| **reanalyse** | the numbers drawn from sound runs | recomputing intervals, means and decisions | `UNC_INTERVAL`, `UNC_MEAN`, `UNC_DECISION`, `UNC_NOISE`, `RESULTS_MISSING` |
| **rewrite** | the words claim more than the numbers | rewording claims and the report | `CLAIM_*`, `REPORT_*` |
| **declare** | the record hides or omits something | stating it in the contract or card | `CONTRACT_*`, `RUNCARD_PARENT`, `PKG_*` |

You cannot fix a rerun problem by rewriting ("we note the extension ran longer"): the comparison is still not on the declared axis. You can sometimes *decline* a non-blocking problem with a reason (a missing parent for a run that genuinely started from scratch, say), but the problems that make a result not count (`PARITY`, `TUNING`, `SEEDS`, `UNC_SEEDS`, `UNC_INTERVAL`, `UNC_DECISION`, `CLAIM_DIRECTION`, `CLAIM_SCOPE`, `CONTRACT_UNCHECKED`, `RUNCARD_MISSING`) cannot be declined.

### Revise in order: rerun, reanalyse, rewrite, declare

Order matters because each kind can invalidate the next. A rerun can change a result; a changed result changes the decision; a changed decision changes which claims are allowed; and the contract must then record what changed after results. So:

1. **Rerun** everything the review requires, with the same contract.
2. **Reanalyse**: recompute every interval and decision from the new runs, with the rule as stated before.
3. **Rewrite** claims and the report to say exactly what the new decisions allow.
4. **Declare** in the contract's "Changes after results" section every result that changed, and why.
5. **Check again.** Run the checker on the revised package. New problems can appear after a rerun; the sample shows one.

If you reverse the order, rewriting first, you will polish sentences about numbers that are about to change.

### The written defence

The defence is one to two pages, one section per question (`### G1`, `### P7`...). Each answer does three things:

- **Answers the question directly** in its first sentence. "Equal tokens, because the claim is about loss at a given amount of training."
- **Points to evidence** in the package: a file (`results.json`, `runs/gspo-s1/run_card.yaml`) or a number with its interval. An answer with no evidence is an opinion.
- **Concedes** what the evidence does not support, in plain words. "A dense toy policy cannot test the Routing Replay part; we deleted that claim." A concession is not a weakness of the defence; an unconceded overclaim is.

It adds no new claims. If answering a question needs a new measurement, that is a rerun, recorded in the log, not a sentence in the defence. `review.defence_problems` checks the mechanical part: every question has a section, the answer has at least two sentences, and it cites a package file or an interval.

### The revision log

The revision log (`revision_log.yaml`) has one entry per problem found before revision:

```yaml
- problem: PARITY@comparison extension seed 0      # the checker's key, CODE@where
  kind: rerun                                       # rerun | reanalyse | rewrite | declare | decline
  change: reran the extension at 120 steps, the declared budget
  files: [runs/gspo-stale4-s0, runs/gspo-stale4-s1, runs/gspo-stale4-s2]
  result_changed: true                              # then contract.md must record it
```

`review.revision_log_problems` checks that every problem has an entry; that the kind matches the problem (or it is a reasoned decline of a non-blocking problem); that a problem logged as fixed is gone; that a rerun names runs that exist; and that a changed result is recorded under "Changes after results". The log is what lets a second reviewer check your revision without redoing the first review.

## Worked example

**A chain of weaknesses in the sample.** The sample capstone (GSPO vs GRPO, a **constructed fixture** with invented numbers) reports, for held-out accuracy of GSPO minus GRPO, per-seed means $0.381, 0.366, 0.392$ against $0.402, 0.371, 0.388$:

$$\bar d = \frac{0.381 + 0.366 + 0.392}{3} - \frac{0.402 + 0.371 + 0.388}{3} = 0.37967 - 0.38700 = -0.0073,$$

with a reported interval $[-0.031, +0.017]$ and margin $0.02$. The rule: the interval is not inside $[-0.02, +0.02]$ (its lower end is $-0.031$), it includes zero, so the decision is **inconclusive**. The package reports "a higher" (`UNC_DECISION`, reanalyse), its accuracy claim says "higher" (`CLAIM_DIRECTION`, rewrite), and its report says the result "proves" GSPO is better (`REPORT_OVERCLAIM`, rewrite). One wrong decision produced three problems; fixing the decision first makes the other two obvious.

**The MDE answers G3.** GRPO's seed std is $0.0155$; with three seeds per arm the minimum detectable effect is $2.8 \cdot 0.0155 \cdot \sqrt{2/3} = 0.036$. The observed $-0.007$ is a fifth of it. The defence says "no difference detected; the design could detect 0.036", not "GSPO is as accurate as GRPO".

**A rerun changes a result.** The extension (GSPO with staleness bound 4) ran one seed for 160 steps instead of three seeds for 120: `PARITY`, `SEEDS`, `UNC_SEEDS`, all reruns. Before: one seed, $0.352 - 0.381 = -0.029$, interval $[-0.061, +0.004]$, inconclusive, and the claim "staleness 4 did not change GSPO's accuracy". After the rerun with seeds 0–2 at 120 steps: $\frac{0.352 + 0.341 + 0.367}{3} - 0.37967 = 0.35333 - 0.37967 = -0.0263$, interval $[-0.041, -0.012]$, decision **a lower**. The old claim, which was allowed under "inconclusive", is now wrong: the rerun created a new `CLAIM_DIRECTION` problem that was not in the first review. This is why you check again after revising, and why the contract must record the change.

**Kinds by count.** The sample's twelve problems (two `TUNING`, one per comparison it affects) split into 5 rerun (two `TUNING`, `SEEDS`, `PARITY`, `UNC_SEEDS`), 1 reanalyse (`UNC_DECISION`), 3 rewrite (`CLAIM_DIRECTION`, `CLAIM_SCOPE`, `REPORT_OVERCLAIM`) and 3 declare (`CONTRACT_STATUS`, `CONTRACT_UNCHECKED`, `RUNCARD_PARENT`).

## Shapes and cost

| Item | Where | Size |
|---|---|---|
| reviewer questions | `<package>/review/questions.json`, `questions.md` | 19 for the sample (4 generic, 3 claim, 12 problems) |
| problems before revision | `<package>/review/before.json` | one `{code, where, message}` per problem |
| defence | `<package>/defence.md` | 1–2 pages; one `### <id>` section per question |
| revision log | `<package>/revision_log.yaml` | one entry per problem before revision |
| compute | CPU | the whole sample review runs in about 6 s (measured, build laptop); reviewing your own capstone costs whatever its reruns cost, PROJECTED from the same formula as the original runs |

## Build it

```python
from frontierlab.capstone import review as RV, package as PK, sample as SA

pkg = SA.write_sample("runs/m20/l202/sample")        # constructed fixture, ten planted weaknesses
before = PK.check_package(pkg)                        # 12 problems
qs = RV.questions(pkg, before)                        # G1-G4, C1-C3, P1-P12
RV.revision_kind("TUNING")                            # 'rerun'
SA.apply_reruns(pkg)                                  # the reruns the review requires (simulated for the fixture)
# ... reanalyse, rewrite, declare by hand ...
RV.defence_problems(open(pkg / "defence.md").read(), qs)
RV.revision_log_problems(log, before, PK.check_package(pkg), pkg, open(pkg / "contract.md").read())
```

`questions` words each problem as a reviewer would ask it and attaches the revision kind; `defence_problems` and `revision_log_problems` return `Problem` lists like the package checker, empty when the defence and the log are complete. Tests: `pytest labs/common/tests/test_capstone.py -k "revision or defence or questions"`.

## What the evidence says

- **Peer review is noisy** (PUBLICLY DOCUMENTED: Cortes and Lawrence's re-analysis of the 2014 NeurIPS experiment). Mechanical checks reduce the part of review that depends on who the reviewer is; they do not replace judgement about the claim.
- **Checklists and code submission improve reproducibility practice** (PUBLICLY DOCUMENTED: the NeurIPS 2019 reproducibility programme introduced a code submission policy, a reproducibility challenge and the ML Reproducibility Checklist, reported by Pineau et al.). The course's package checker is a checklist of the same kind, made executable.
- **Comparing against a tuned baseline** with the same tuning budget is REASONABLE INDUSTRY PRACTICE (Google's Deep Learning Tuning Playbook), and the most common reason small reproductions disagree with their sources.
- **The four revision kinds, the order and the log format** are the course's procedure (REASONABLE INDUSTRY PRACTICE in spirit; the specific taxonomy is ours). The sample capstone is a constructed fixture; its numbers are invented and say nothing about GSPO or GRPO.

## Lab

**Folder:** [`labs/module-20/lesson-02/`](../../labs/module-20/) · **Time:** about 2 hours · **Pass check:** `pytest labs/module-20/lesson-02` passes; `review_lab.py --step check` prints `REVIEW PASSED` for the sample; your own capstone package has a `review/` folder, a defence and a revision log.

### Experiment contract

This lab reviews; it runs no comparison of its own, so its contract is the sample's (`runs/m20/l202/sample/contract.md`) and, in step 6, yours. What the lab fixes in advance:

- **Question:** does the sample capstone's conclusion survive review, and what revision does each weakness need? Decision informed: whether the sample's claims may be quoted, and in what words.
- **Hypothesis:** every planted weakness is found by the checker or by a generic or claim question. **Status:** established effect for the mechanical ones (the checker's tests); for your own package, the review is a measurement.
- **Baseline:** the package as submitted (`review/before.json`). **Changed:** the package, by revisions of the four kinds. **Controlled:** the decision rule and margin as stated before the original runs.
- **Comparison axis:** problems before against problems after, by key.
- **Budget:** CPU seconds for the sample; for your own package, the reruns' PROJECTED cost by the 20.1 formula.
- **Metrics and decision rule:** the package is sound (no problems), the defence answers every question with evidence, and the log is complete. No partial credit for a package that still has a blocking problem.
- **Correctness checks:** `pytest labs/module-20/lesson-02`; `pytest labs/common/tests/test_capstone.py`.
- **Fallback evidence:** none needed.
- **Limits:** a constructed sample; the checker finds only mechanical weaknesses; the claim and generic questions need your judgement.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | the GPU your capstone's reruns need (lesson 20.1's list). Not run in this build | step 6 on your main-path package: every rerun the review requires, PROJECTED with the same formula as the original runs |
| Free GPU (Colab/Kaggle T4) | T4 | step 6 on your T4 package |
| Free CPU | laptop; the sample review measured at about 6 s | steps 1–6 as written |

### Steps

1. **Review.** `python labs/module-20/lesson-02/review_lab.py`. It writes the sample to `runs/m20/l202/sample`, checks it and lists 19 reviewer questions (`review/questions.md`). Before reading the problem questions, read the package yourself (contract, claim, run cards, results, claims, report) and write down the weaknesses you find. Compare with the checker's list: which did you miss, and which question did the checker not ask that you would?
2. **Implement** the three TODOs in `lab.py` (`revision_kind`, `defence_problems`, `revision_log_problems`) and run `pytest labs/module-20/lesson-02`.
3. **Rerun** first: `python labs/module-20/lesson-02/review_lab.py --step rerun`. It writes the rerun outputs (simulated, since the sample is a fixture) and checks again. Find the new problem the rerun created.
4. **Reanalyse, rewrite, declare** by editing `results.json`, `claims.yaml`, `report.md`, `contract.md` and `runs/control-random-s1/run_card.yaml`. Record in "Changes after results" what changed.
5. **Defend and log.** Write `defence.md` (one `### <id>` section per question) and `revision_log.yaml` (one entry per problem in `review/before.json`) in the sample folder, then `python labs/module-20/lesson-02/review_lab.py --step check` until it prints `REVIEW PASSED`. Compare with `--step reference` afterwards.
6. **Your own capstone.** `python labs/module-20/lesson-02/review_lab.py --package <your package>` generates the questions for your Module 20 project. Swap packages with a classmate if you can: their review of your package is worth more than yours.

<details>
<summary>Hint for step 4</summary>

Start from the decision: recompute it from the reported interval and the margin in `claim.yaml`. Then every claim that names that comparison must use a direction the new decision allows, and so must the report's sentence. The MoE claim cannot be reworded into a course-scale measurement; delete it and keep the paper's own statement as PUBLICLY DOCUMENTED.

</details>

<details>
<summary>What the build's run gave</summary>

Measured 2026-10-07 on the build laptop (Windows 11, Python 3.12, torch 2.14.1+cpu), with the reference solution: the sample has 12 problems before revision (ten planted weaknesses: the one-seed extension triggers both `SEEDS` and `UNC_SEEDS`, and `TUNING` appears for the two comparisons it affects); 8 remain after the reruns, including one new `CLAIM_DIRECTION` on the staleness claim that the rerun created; the reference revision ends with a sound package, a defence with every question answered and a complete log. The four steps together took about 6 s.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-20/lesson-02/solution.py` for the TODOs; `review_lab.py --step reference` writes the reference revision, defence and log to `runs/m20/l202/reference`. Check the TODOs with `LAB_TARGET=solution pytest labs/module-20/lesson-02`.

</details>

## Common mistakes

- **Defending everything.** A defence that concedes nothing is not credible. Concede what the evidence does not support, in one sentence, and move on.
- **Rewriting before rerunning.** The rerun may change the result; check again after every rerun before touching the claims.
- **Fixing the wording of a rerun problem.** "We note the extension ran 33% longer" leaves the comparison off its axis. Rerun it or drop it.
- **New claims in the defence.** If an answer needs a new number, that is a rerun or a reanalysis with a log entry, not a sentence.
- **Silent changes after results.** A result that changed in revision and is not in "Changes after results" looks, to the next reader, like the result you planned. The log check refuses it.
- **Treating a passing checker as a passing review.** The checker cannot see a margin chosen after the results or a baseline tuned on the test split. Answer the generic and claim questions as if it did not exist.

## References

- C. Cortes and N. D. Lawrence, *Inconsistency in Conference Peer Review: Revisiting the 2014 NeurIPS Experiment*, 2021. https://arxiv.org/abs/2109.09774
- J. Pineau et al., *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*, 2020. https://arxiv.org/abs/2003.12206
- Google Research, *Deep Learning Tuning Playbook*. https://github.com/google-research/tuning_playbook
- C. Zheng et al., *Group Sequence Policy Optimization*, 2025, sections 5.1 and 5.3. https://arxiv.org/abs/2507.18071
- Course templates: [experiment rubric](../../templates/experiment-rubric.md), [experiment contract](../../templates/experiment-contract.md).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The [Module 20 project](../../projects/module-20-capstone.md): your capstone report and code, reviewed, defended and revised.
