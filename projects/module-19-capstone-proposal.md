# Module 19 project · The capstone proposal

This project produces the document Module 20 starts from: a capstone proposal for one claim from the capstone list, with a filled experiment contract, what counts as reproduced, one extension, a cheap proxy and a kill criterion, checked by a validator and self-assessed against the experiment rubric. It uses the three lessons of the module: problem choice and power from lesson 19.1, the reproduction decision and deviation log from lesson 19.2, and the claims register and review from lesson 19.3. Before writing your own, you debug a colleague's proposal with four planted problems, each of which would waste the capstone's GPU budget.

**Time:** 5–7 attended hours, plus the proxy run (about 8 minutes on CPU for the lesson 19.1 ladder; your chosen claim's proxy may take longer). **Folder:** [`labs/module-19/project/`](../labs/module-19/) (`proposal_template.yaml`, `check_proposal.py`, `buggy_proposal.yaml`, `example_proposal.yaml`, `test_proposal.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## The capstone claim list

Pick one. Each is a claim from a 2025–26 report that an earlier module built the code for; `frontierlab.research.proposals.CAPSTONE_CLAIMS` holds the source, the course lessons, the PROJECTED cost of the course lab (one arm set, one seed) and the known risk of the obvious proxy.

| Claim | Source | Course code | Course lab cost (PROJECTED) |
|---|---|---|---|
| GSPO vs GRPO stability | Zheng et al., GSPO, sections 4–5 | 14.1, 14.2 | 31–52 GPU-hours (lesson 14.2, five objectives) |
| QK-Clip vs QK-norm | Kimi K2 section 2.1; DeepSeek-V4 sections 2.3.3, 2.4 | 07.2, 07.5 | 0.8 GPU-hours (lesson 07.2, 7 arms, one seed) |
| DSA quality and cost vs dense | DeepSeek-V3.2 sections 2.1–2.2 | 05.2, 05.3 | about 1 GPU-hour (lesson 05.3) |
| mHC vs HC stability | Xie et al., mHC, sections 3–5 | 06.2 | 2–3.5 GPU-hours per ladder rung (lesson 06.2) |
| Micro-anneal data scoring | OLMo 2 section 4.4.2 | 10.4 | 1.3 GPU-hours (lesson 10.4) |
| On-policy vs off-policy distillation at equal compute | Thinking Machines Lab, On-Policy Distillation; Agarwal et al. (GKD) | 13.2 | 5–7 GPU-hours (lesson 13.2) |

## Variants and cost

| Variant | What you do | Hardware | Cost |
|---|---|---|---|
| Main path | the proposal for a main-path capstone: your claim at the size its lesson's main path uses, with the budget computed from that lesson's formula and your seed count; the proxy run on GPU if your claim's proxy needs it | writing on any machine; proxy on 1× H100 or A100 | the proxy only: for the lesson 19.1 ladder PROJECTED 0.13 H100-hours; for other claims, the free-GPU variant of the claim's lesson is the natural proxy |
| Free GPU (Colab/Kaggle T4) | the same proposal scaled to the claim lesson's T4 variant; say what that scale cannot show | T4 | the claim lesson's T4 figures |
| Free CPU | the same proposal at the CPU variant of the claim's lesson; the checks below run in seconds | laptop | `pytest labs/module-19/project` measured under a second |

Whichever scale you choose, the budget in your proposal is labelled PROJECTED with the formula it comes from, or measured with the run cards that measured it.

## Deliverables

1. **Three ranked proposals** (lesson 19.1, `runs/m19/l191/proposals.md`), at least one from the capstone list, with the robustness of the ranking at 2× and 3×. The top one becomes the capstone unless a proxy says otherwise.
2. **The capstone proposal**, `runs/m19/project/proposal.yaml`, copied from `proposal_template.yaml` and filled, passing `check_proposal.py` with no problems. It contains the experiment contract (plan section 9 as data), what counts as reproduced (direction, tolerance from your noise floor, whether magnitude is comparable), one extension ablation, the proxy and the kill criterion, all dated before any result.
3. **The deviation log** you expect (lesson 19.2): one line per aspect (scale, data, tokens, optimizer, learning rates, evaluation, seeds, code), with the reason and the expected effect on the claim. Deviations you discover later are added with "recorded after".
4. **A claims register stub** (lesson 19.3): the claim or claims your capstone note will make, with the run names you plan, so that the note can be linted the day the runs finish.
5. **The debugging report** (below).
6. **A peer review** of one other learner's proposal with the lesson 19.3 review template, or, if you work alone, of the reference proposal `example_proposal.yaml` after yours is written.
7. **The written defence** (below) and your rubric self-score.

## The experiment contract

Your contract is `proposal.yaml`. Fixed by the project:

- **Question:** one answerable question ending in "?", about one claim on the list, at a stated scale. Decision informed: what Module 20's capstone will reproduce and extend, and what your team (or the course's artefact chain) would change if the answer goes either way.
- **Hypothesis and status:** one of "established effect", "reported effect", "may not appear at this scale". Most capstone claims are the third at course scale; then `fallback` must name the provided traces or published curves you will analyse, labelled as analysis.
- **Baseline:** the run card the claim is measured against, tuned with the same budget as the method (`budget.tuning_per_arm`).
- **Changed variable and controlled variables:** exactly one change, or the declared cells of a factorial design.
- **Comparison axis:** one of equal tokens, parameters, training FLOPs or wall-clock, and what it does not answer.
- **Budget:** hardware, GPU-hours, label (PROJECTED with formula, or measured), teacher/judge/verifier/generator compute counted.
- **Metrics and decision rule:** primary metric, uncertainty method, the noise floor you measured (say where), seeds per arm, the smallest effect that would change the decision; the validator recomputes the minimum detectable effect and refuses an underpowered design. A decision rule with a number.
- **Correctness checks:** the course checks of the claim's lessons, and a run-card diff between arms.
- **Limits:** at least three, one about scale.

## Steps

1. **Rank.** Finish the lesson 19.1 lab with your own three proposals. Run the proxy for the top one, or for the top two if their scores are within a factor of 2.
2. **Debug first.** Run the checker on the colleague's proposal (below) and write the debugging report before you write your own: it is the fastest way to learn what the validator expects.
3. **Write** `proposal.yaml`. For the noise floor, use a measurement: the Module 1 project's noise floor at the size you run, or the seed spread of the claim lesson's own runs. If you have none at your scale, say so and make measuring it the first runs of the capstone.
4. **Check:** `python labs/module-19/project/check_proposal.py runs/m19/project/proposal.yaml` and `PROPOSAL=runs/m19/project/proposal.yaml pytest labs/module-19/project`. Fix every problem in the proposal, not in the checker.
5. **Review** a peer's proposal (or the reference) with `labs/module-19/lesson-03/review-template.md`, and answer the review you receive.
6. **Score** yourself: write `runs/m19/project/rubric.yaml` with a 0, 1 or 2 for each of `question`, `controls`, `axis_budget`, `correctness`, `uncertainty`, `conclusion`, `limits`, and run `check_proposal.py ... --rubric runs/m19/project/rubric.yaml`. At the proposal stage, score `correctness` and `conclusion` on what the proposal commits to (which checks, which decision rule), not on results.

## Debugging task

`labs/module-19/project/buggy_proposal.yaml` is a colleague's proposal for "QK-Clip vs QK-norm". Their message is at the top of the file: "Ready to launch. I trimmed it to fit the budget." It has four planted problems. Each would let the capstone spend its GPU-hours and end with nothing it can defend. For each, name the field that shows it, what the run would have produced, and the fix.

```bash
python labs/module-19/project/check_proposal.py labs/module-19/project/buggy_proposal.yaml
pytest labs/module-19/project
```

<details>
<summary>Hint</summary>

Compare what the message says ("two seeds are plenty", "spent my tuning runs on QK-Clip", "the rule is simple now") with the fields it changed: `metrics`, `budget.tuning_per_arm`, `decision_rule`, `reproduction.tolerance`.

</details>

<details>
<summary>The four problems (after your own diagnosis)</summary>

1. **Underpowered.** `metrics`: noise floor 0.0286 with 2 seeds per arm gives a minimum detectable effect of 0.080 nats against an expected effect of 0.02 (power 0.11). The run would almost certainly end "inconclusive". Fix: more seeds, a smaller noise floor at a larger scale (measure it), or a larger effect that still matters for the decision.
2. **Unequal tuning.** `budget.tuning_per_arm`: 1 run for the baseline, 6 for the method. Any win is confounded with the extra tuning (Dodge et al., lesson 19.3). Fix: the same tuning budget for both arms.
3. **No checkable decision rule.** `decision_rule`: "if it is better" has no threshold, so any difference, including noise, decides. Fix: a number, for example "the 95% interval of the loss difference lies below +0.02".
4. **Tolerance below the noise.** `reproduction.tolerance`: 0.01 nats is a third of the noise floor; an effect that small cannot be told from noise, so "reproduced" would be declared on noise. Fix: a tolerance derived from the noise (lesson 19.2 uses 2 × the seed std).

Also visible: the limits still say "four seeds" after the cut to two. A reviewer reading the prose would catch it; the validator does not.

</details>

<details>
<summary>A reference proposal (read after writing your own)</summary>

`labs/module-19/project/example_proposal.yaml` proposes "QK-Clip vs QK-norm" at `pilot-30m`. Note what it does that a first draft usually does not: the decision names the real alternative (MLA with QK-Clip, since MLA cannot use QK-norm cheaply), the extension tests that alternative, the tolerance and the logit rule come from measured quantities (lesson 07.2's one-step logit growth), the noise floor is labelled an assumption to be measured first, and the kill criterion says when the question cannot be asked at this scale (the unclipped arm never passes $\tau$). `check_proposal.py` reports no problems and power 0.94 at the expected effect, if the assumed noise floor holds.

</details>

## Written defence

Answer in 1–2 pages, as you would to a reviewer:

1. Why this claim and not your second proposal? Give the scores, the robustness, and what the proxy showed.
2. Where does your noise floor come from, and what happens to your design if it is twice as large?
3. What will you call "reproduced", and why is that tolerance not chosen after the results?
4. Which deviation from the paper is most likely to change the answer, and in which direction?
5. What result would make you stop before spending the full budget?

## Self-check against the rubric

Score your proposal with [`templates/experiment-rubric.md`](../templates/experiment-rubric.md):

- [ ] One answerable question, and the decision it informs, written before any run.
- [ ] One changed variable; everything else listed as held fixed.
- [ ] The axis named with what it does not answer; budgets match on it; the baseline gets the same tuning budget.
- [ ] Correctness checks named, from the claim's lessons, plus a run-card diff.
- [ ] Noise floor, seeds and minimum detectable effect stated; the design is not underpowered.
- [ ] "Reproduced" defined in advance (direction and magnitude separately), with a tolerance from the noise.
- [ ] At least three limits, one about scale; a kill criterion.
- [ ] `check_proposal.py` reports no problems, and the proposal is dated before any result.

In Module 20 this proposal becomes the capstone's contract: you will reproduce the claim at small scale, extend it with the ablation you named here, write the note with its claims register, and defend it.
