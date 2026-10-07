# Review template — someone else's result

Copy this file and fill it in. A review is useful when every problem it raises names the sentence or field that
shows it and what the author should run or change. Review the evidence, not the author; a sound null result is a
good result.

## 1. The claim in one sentence

Restate the main claim in your own words, with the scale, data and comparison axis it was measured at. If you cannot,
that is the first problem.

## 2. Does each claim have its evidence?

For every claim in the summary (or every entry of the claims register):

| Claim | Runs behind it | Seeds per arm | Interval reported? | Checkpoint and split | Within its scope? |
|---|---|---|---|---|---|
| | | | | | |

## 3. Controls and budget

- What changed between the arms? Anything besides the variable under test? (Diff the run cards:
  `python -m frontierlab.record <run A> <run B> --changed <key> --axis <axis>`.)
- Which comparison axis is claimed, and do the budgets match on it (tokens, FLOPs, wall-clock, parameters)?
- Was the baseline tuned as hard as the method (same number of runs, same search space)?

## 4. Uncertainty

- Seed noise floor and seeds per arm. Is the reported difference larger than the noise, with an interval?
- Was anything (checkpoint, learning rate, evaluation subset) chosen after looking at the numbers it reports, or on
  the test split?

## 5. Figures

For each figure: the one claim it answers; whether its x-axis is the comparison axis; whether all arms end at the
same budget; whether uncertainty is shown and the caption says over how many seeds and what the bands are.

## 6. Problems found

One row per problem. Use these labels for the common ones: missing seeds, unmatched budget, cherry-picked checkpoint,
claim beyond evidence, uncontrolled change, untuned baseline, no uncertainty, figure.

| Problem | Where | Evidence | Request |
|---|---|---|---|
| | | | |

## 7. Recommendation

Accept as written / accept with the scope narrowed / revise and rerun (say which runs) / reject the claim (say why the
evidence cannot support it at any scope). One paragraph: what the evidence does support.
