---
id: "19.3"
module: 19
minutes: 35
practice_minutes: 90
prerequisites: ["19.2", "01.4", "01.5"]
objectives:
  - Write a research note whose every claim is registered with the runs behind it, and check it mechanically against the run cards before anyone reads the prose.
  - Design a figure that answers one claim, shows uncertainty over seeds, and puts the arms on the experiment contract's comparison axis at matched budgets.
  - Review someone else's write-up with a template and find its missing seeds, unmatched budget, cherry-picked checkpoint and claim beyond evidence, each tied to the field that shows it.
  - Write a written defence that answers reviewer questions with numbers from the record rather than with adjectives.
volatility: concept
sources:
  - title: "Lipton and Steinhardt, Troubling Trends in Machine Learning Scholarship (abstract: four trends)"
    url: https://arxiv.org/abs/1807.03341
  - title: "Dodge et al., Show Your Work: Improved Reporting of Experimental Results (expected validation performance against computation budget)"
    url: https://arxiv.org/abs/1909.03004
  - title: "Henderson et al., Deep Reinforcement Learning that Matters (Figure 5: two groups of 5 seeds of the same algorithm)"
    url: https://arxiv.org/abs/1709.06560
  - title: "NeurIPS 2026 Paper Checklist (Claims; Limitations; Experiment Statistical Significance; Experiments Compute Resources)"
    url: https://neurips.cc/public/guides/PaperChecklist
  - title: "John Schulman, An Opinionated Guide to ML Research (Keep a Notebook, and Review It)"
    url: http://joschu.net/blog/opinionated-guide-ml-research.html
  - title: "Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities (Figure 1 as an example of a figure that answers one question)"
    url: https://arxiv.org/abs/2309.14322
last_verified: "2026-10-07"
---

# 19.3 · Writing and defending results

A result that is not written down so that someone else can check it has not been delivered. This lesson covers the three documents a research engineer writes after the runs: the research note (claims, each tied to its runs), the figure (one claim, with its uncertainty, on the right axis), and the written defence (answers to a reviewer, with numbers). It also covers the document you write about other people's results: the review. The lab registers claims so a linter can check them against run cards, draws a figure from your lesson 19.2 reproduction, and reviews a colleague's note with four planted problems.

## Why this matters at a frontier lab

Decisions about expensive runs are made by people who did not run your experiment. They read a note, look at one figure, and ask a few questions. If the note says "QK-Clip beats QK-norm" and the evidence is one seed per arm, a longer run for one arm and the best checkpoint of six, the decision is wrong and nobody can tell from the prose. Every lab has internal review for this reason, and every engineer is a reviewer as often as an author. Reviewing well is a skill: finding the field in a run card that contradicts a sentence is faster and more useful than a general feeling that "the evidence seems thin".

## The idea

### The research note: claims with their evidence

Schulman's advice is to keep a daily notebook and condense it every week or two into a review of findings, insights and next steps. The note you share is that condensed review for one question. Structure it so the reader gets the answer first:

1. **Summary** — the answer to the contract's question in two or three sentences, with the number, its interval and the scope (size, data, budget).
2. **Setup** — what changed, what was held fixed, the comparison axis, seeds; a link to the contract and run cards.
3. **Result** — the figure that answers the question, then the table.
4. **Limits** — what the result does not cover and what would change it.
5. **Claims register** — every claim in the summary, as data.

The register is the part this lesson adds. It is a fenced YAML block (info string `yaml claims`) with one entry per claim: the sentence, the runs per arm, the variables the comparison is allowed to change, the axis, the value and its interval, which checkpoint each number came from, the split anything was selected on, and the scope. Prose can hide a problem; a register cannot, because `frontierlab.research.writeup.lint_claims` reads the run cards it names and checks each claim:

| Code | Raised when | Why it matters |
|---|---|---|
| `seeds` | an arm rests on fewer than 2 distinct seeds | one run per arm cannot separate the method from seed noise: Henderson et al. split 10 runs of the *same* algorithm into two groups of 5 and got curves that look like different algorithms (Figure 5) |
| `uncertainty` | no interval, or an interval containing 0 under a comparative sentence | the reader cannot tell a difference from noise |
| `budget` | the arms differ on the budget the axis holds equal (via `frontierlab.record.diff_cards`) | the extra tokens, not the method, may be the effect |
| `control` | anything else that invalidates the comparison differs (data, evaluation, a second changed variable) | two changes, one claim |
| `checkpoint` | a number is not from the final checkpoint, or anything was selected on the test split | picking the best of several evaluations of one arm biases it upward; the test split stops being a test |
| `scope` | the text generalises beyond the evidence ("frontier", "all models", "should transfer") | a 1.8M-parameter result is evidence about 1.8M parameters |

Lipton and Steinhardt's survey of troubling trends names two of these directly: "failure to identify the sources of empirical gains" (a gain from extra tuning or extra tokens credited to an architecture change) and the failure to distinguish explanation from speculation (a scope sentence is often speculation written as a finding). The NeurIPS 2026 paper checklist asks the same questions of every submission: whether claims match results (Claims), whether limitations are stated, whether error bars are reported and "the factors of variability that the error bars are capturing" stated, and how much compute each run used.

### Figures that answer the question

A figure is a claim made visually. Four rules, each checked by `lint_figure` on a small figure specification you write next to the plot:

- **One claim per figure.** Write the claim as the figure's title or first caption sentence. If you need two sentences joined by "and", you need two figures.
- **Uncertainty shown.** Points per seed, a band over seeds, or an interval over items, with at least 2 seeds per arm; the caption says what the band is and over how many seeds.
- **The comparison axis on x.** If the contract compares at equal tokens, plot against tokens, not optimizer steps (a run with twice the batch is "ahead" at every step). If the contract compares at equal FLOPs, plot against FLOPs.
- **Matched budgets.** Every arm ends at the same point on that axis. A curve that runs further right invites the reader to compare its end with the other curve's end.

Dodge et al. add a fifth that matters when tuning differs: report performance as a function of the tuning budget (expected best validation score after $k$ hyperparameter trials), because which model looks best can change with how much search each got. Wortsman et al.'s Figure 1 is a good model of the rules: final loss against learning rate per model size (one question: how sensitive is each to the rate?), with the summary statistic plotted underneath.

### The written defence

Every project in this course ends with a defence: answers to reviewer questions, in 1–2 pages. A good answer has three parts: the direct answer, the number from the record that supports it, and what would change it. Compare:

- "The baseline was well tuned." (adjective)
- "The baseline got the same 3-point learning-rate sweep as the method (run cards `b0-lr*`), on the validation split; its best rate was interior, 1e-2 of {5e-3, 1e-2, 2e-2}. A wider sweep could still move it; that is limit 3." (record)

The rubric in `templates/experiment-rubric.md` scores the defence through criteria 5–7: uncertainty respected, conclusion no stronger than the evidence, limits stated with what would change the answer.

### Reviewing someone else's result

A review is a list of problems, each with where it is, the evidence that shows it, and what the author should run or change, followed by a recommendation (accept, accept with narrowed scope, revise and rerun, reject the claim). The lab's review template walks through it: restate the claim in your own words with its scale; table each claim against its runs, seeds, interval, checkpoint and scope; diff the run cards; check uncertainty and selection; check each figure; list problems; recommend. Two habits make reviews useful. Review the evidence, not the author. And say what the evidence *does* support: "QK-Clip and QK-norm cannot be distinguished at this scale" is a finding the author can use, where "unconvincing" is not.

## Worked example

A colleague's note says QK-Clip reaches held-out loss 6.268 against QK-norm's 6.302, "a 0.034-nat improvement", larger than the "0.02 we usually consider meaningful". The run cards show one seed per arm, 400 steps for QK-Clip against 300, and the QK-Clip number is the best of 8 evaluations on the test split.

How big could the effect of each problem alone be? Seed noise: the toy recipe's seed std is 0.0286 nats (lesson 01.4); the difference of two single runs has std $0.0286\sqrt{2} = 0.040$, so a 0.034 difference is under one standard deviation of pure noise. Budget: QK-Clip got 33% more tokens, and the note itself says it "was still improving at 300 steps"; any loss it gained after step 300 is credited to the method, by an amount the record cannot separate. Selection: the best of 8 noisy evaluations is biased downward by roughly the expected maximum of 8 draws, about $1.4\sigma_{\text{eval}}$ for normal noise; with 256 windows and a per-window std of 0.276 (lesson 01.4), $\sigma_{\text{eval}} = 0.276/\sqrt{256} = 0.017$, a bias of about 0.024 nats. Any one of the three could produce the whole claimed difference. The note's claim does not survive review; the revised note (lab folder `fixed/`) with 3 seeds per arm, equal tokens and final checkpoints reports −0.004 [−0.016, +0.008]: no difference at this scale (both notes' numbers are invented for the exercise).

The minimum detectable effect makes the review's request concrete. With $\sigma = 0.0286$ and $n$ seeds per arm, $\text{MDE} = 2.80 \cdot 0.0286 \cdot \sqrt{2/n}$: 0.080 for 2 seeds, 0.065 for 3, 0.036 for 10. A reviewer who asks for "more seeds" should say how many and why: to resolve 0.034 nats at 80% power the author needs about 11 seeds per arm, which tells the author that the toy scale is the wrong place to ask this question.

## Shapes and cost

| Item | Size | Cost |
|---|---|---|
| claims register | one YAML entry per claim (8–10 fields) | minutes; the linter reads it in milliseconds |
| run cards per claim | one YAML file per run (`run_card.yaml`, about 3 KB) | already written by every course run |
| figure spec | one YAML file next to each figure | minutes |
| the lab's figure | 18 runs of lesson 19.2, one PNG at 150 dpi (matplotlib 3.11.2, Agg backend) | about 2 s on CPU |

## Build it

```python
from frontierlab.research import writeup as W

issues = W.lint_writeup("labs/module-19/lesson-03/flawed/writeup.md", "labs/module-19/lesson-03/flawed/cards")
for i in issues:
    print(i)            # [seeds] c1: arm 'qk-clip' rests on 1 seed(s); ...  [budget] ...  [checkpoint] ...  [scope] ...

spec = {"claim": "QK-norm reduces LR sensitivity at toy scale", "x": "lr", "contract_axis": "tokens",
        "uncertainty": "min-max over 3 seeds", "arms": {"qknorm": {"n_seeds": 3}, "noqk": {"n_seeds": 3}},
        "caption": "Final held-out loss, mean of n = 3 seeds; band: min-max range."}
print(W.lint_figure(spec))   # []
```

`lint_claims` uses `frontierlab.record.diff_cards` (lesson 01.5) with the claim's declared changed variables and axis, and treats seeds as the replicate axis; so the budget and control checks are the same rules that decide whether two runs are comparable anywhere else in the course. The linter checks what the cards can show. It cannot see a baseline that was not re-tuned when the tuning runs have no cards, a wrong unit in the prose, or a claim that is missing from the register; those are the reviewer's job. Tests: `pytest labs/common/tests/test_research.py -k "lint or parse"`.

## What the evidence says

- **Seed variance can produce apparent method differences: ESTABLISHED** (Henderson et al. for deep RL; lesson 01.4 measured it for this course's toy recipe). Reporting intervals over seeds is asked for by the NeurIPS checklist and is REASONABLE INDUSTRY PRACTICE for training ablations, though many technical reports still give single-run ablations.
- **Tuning budget changes which method looks best: PUBLICLY DOCUMENTED** (Dodge et al.). Matched tuning budgets are part of this course's contract for that reason.
- **The troubling trends: PUBLICLY DOCUMENTED** as a survey and argument (Lipton and Steinhardt), not a measurement of how common each trend is.
- **The linter's rules: the course's formalisation** of the checklist and rubric. A clean lint is necessary, not sufficient: it says the claims register is consistent with the run cards, not that the experiment answers the question.

## Lab

**Folder:** [`labs/module-19/lesson-03/`](../../labs/module-19/) · **Time:** about 90 minutes · **Pass check:** `pytest labs/module-19/lesson-03` passes (it checks your four functions against the shared linter and that your review names the four planted problems with where, evidence and request); `review_lab.py --part figure` writes a figure whose spec lints clean.

> [!NOTE]
> This lab compares nothing new: it checks the evidence of two fictional notes and draws a figure from your lesson 19.2 runs, whose experiment contract is in lesson 19.2. So it has no experiment contract of its own.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | any machine (the review and linter are CPU code); the figure from your main-path 19.2 sweep if you ran it: `review_lab.py --part figure --variant main` | the steps below |
| Free GPU (Colab/Kaggle T4) | not needed | the steps below; `--variant t4` for a T4 sweep |
| Free CPU | laptop; seconds | the steps below |

### Steps

1. **Read the colleague's note** `labs/module-19/lesson-03/flawed/writeup.md` and its run cards in `flawed/cards/`. Before running anything, fill a copy of `review-template.md` from reading alone.
2. **Implement** `seed_issues`, `checkpoint_issues`, `scope_issue` and `figure_issues` in `lab.py`.
3. **Lint:** `python labs/module-19/lesson-03/review_lab.py`. Compare what your functions and the full linter find with your reading. Which problem did the cards show that the prose hid?
4. **Write your review** into `REVIEW` in `lab.py` (the four planted problems use the labels in the docstring; add anything else you found) and run `pytest labs/module-19/lesson-03`. Then read `fixed/writeup.md`: does it answer every request in your review?
5. **Draw the figure** from your lesson 19.2 sweep: `python labs/module-19/lesson-03/review_lab.py --part figure`. Check the spec lints clean, then look at the PNG and ask whether a reader would get the claim without reading the caption.
6. **Write a defence** (half a page) for your lesson 19.2 record against three reviewer questions: why three learning rates and not seven; why the tolerance is 2 × 0.0286; what result would make you call the claim not reproduced.

<details>
<summary>Hint for TODO 4</summary>

Read `lint_figure`'s docstring and body in `labs/common/frontierlab/research/writeup.py` and return the *set* of codes. Two rules raise `budget` and two raise `uncertainty`; a set counts each code once.

</details>

<details>
<summary>What the build's run gave (compare after your review)</summary>

Measured 2026-10-07 on the build laptop (seconds). On the flawed note the full linter raised 9 issues: `seeds` twice (one seed per arm), `uncertainty` (no interval), `budget` three times (`args.steps`, `budget.steps`, `budget.tokens`: 400 against 300 steps), `checkpoint` twice (best of 8, test split) and `scope` ("should transfer"); its figure spec raised `scope` (two claims), `uncertainty` twice (none shown; caption silent), `seeds` and `budget` (arms end at 300 and 400). The revised note and its figure spec: no issues. The reference review adds one problem the linter cannot raise by itself, the figure, which it reaches only through the spec.

The figure from the build's lesson 19.2 sweep (`runs/m19/l193/figure.png`): final held-out loss against peak learning rate on a log axis, mean of 3 seeds per arm with the min-max band. The QK-norm curve rises from 6.40 to 6.53 across the three rates; the curve without QK-norm starts lower (6.30) and ends at 7.03, crossing the QK-norm curve between 3e-3 and 1e-2. The spec linted clean. One thing a reader might still misread: the bands overlap at 1e-2, so the figure alone does not show the per-seed paired effect; the record's interval does, so the note should quote that interval next to the figure.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-19/lesson-03/solution.py` (including a reference `REVIEW`). Check it with `LAB_TARGET=solution pytest labs/module-19/lesson-03`.

</details>

## Common mistakes

- **Claims that are not in the register.** The summary says "and it is more stable" but the register has only the loss claim. Every sentence in the summary that states a result gets an entry.
- **"Larger than what we consider meaningful" as significance.** A threshold for practical importance is not an interval. Report both: the interval, and whether it excludes the threshold.
- **Plotting against steps when the arms differ in batch or tokens per step.** Use the contract's axis.
- **Best checkpoint for one arm, final for the other.** Either every arm uses the final checkpoint, or a selection rule stated in advance is applied to every arm on the validation split.
- **Reviews without requests.** "The evidence is weak" helps nobody. Name the field, the number, and the run that would fix it.
- **Defending with adjectives.** "Well tuned", "robust", "clearly better": replace each with the number from the record, or delete it.

## References

- Z. C. Lipton and J. Steinhardt, *Troubling Trends in Machine Learning Scholarship*, 2018. https://arxiv.org/abs/1807.03341
- J. Dodge et al., *Show Your Work: Improved Reporting of Experimental Results*, 2019. https://arxiv.org/abs/1909.03004
- P. Henderson et al., *Deep Reinforcement Learning that Matters*, 2017, Figure 5. https://arxiv.org/abs/1709.06560
- NeurIPS, *Paper Checklist Guidelines* (2026). https://neurips.cc/public/guides/PaperChecklist
- J. Schulman, *An Opinionated Guide to ML Research*, section "Keep a Notebook, and Review It". http://joschu.net/blog/opinionated-guide-ml-research.html
- M. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, 2023, Figure 1. https://arxiv.org/abs/2309.14322
- The experiment rubric: [templates/experiment-rubric.md](../../templates/experiment-rubric.md).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The [Module 19 project](../../projects/module-19-capstone-proposal.md): the capstone proposal with a filled experiment contract, reviewed against the rubric.
