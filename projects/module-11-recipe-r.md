# Module 11 project · The Recipe-R run: a pre-registered prediction, checked

Recipe-R is everything Stage C decided: the architecture from Stage B (or Baseline-0), the optimizer of Module 7, the precision of Module 8 and Data-v1 from Module 10. This project spends the largest training budget of the course on it, and does so the way a lab does: a ladder with exactly that recipe, a written and hashed prediction for the final run (its loss, its whole curve, and its score on a downstream task), the run itself under a pre-stated stopping rule, and a check of every prediction afterwards. The grade is the soundness of that chain. A prediction that misses, reported and explained, scores the same as one that hits.

**Time:** 6–8 attended hours plus unattended runs. **Folder:** [`labs/module-11/project/`](../labs/module-11/) (`run_project.py`, `recipe_r_example.json`, `buggy_fit.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Ladder and target | Hardware | Cost |
|---|---|---|---|
| Main path | ladder: pilot-10m, pilot-30m, pilot-70m, Baseline-0 at 5, 20 and 60 tokens per parameter (12 runs); target `m11-350m` (361M parameters) at 60 tokens per parameter, 21.6B tokens of Data-v1 at vocabulary 32,768, sequence 1,024, bf16 | 1× H100 80 GB | **PROJECTED, pending the Module 11 pilot:** target $5.0 \times 10^{19}$ FLOPs ÷ ($989 \times 10^{12}$ × an assumed 0.30 MFU) = 46.9 GPU-hours; ladder $1.45 \times 10^{19}$ FLOPs = 13.6 GPU-hours; about USD 120–180 at USD 2–3 per H100-hour. Needs the shared loop changes of `curriculum/inbox/module-11-shared-changes.md` section 3 to combine the Module 7 optimizer and the Module 10 mixture in one run |
| Free GPU (Colab/Kaggle T4) | `--variant t4`: ladder m11-r3 … m11-r6, target m11-r7 at 10 tokens per parameter | T4, fp32 | PROJECTED about 2 hours; use `--max-minutes` style interruptions freely, every run resumes exactly |
| Free CPU | `--variant cpu`: ladder m11-r1 … m11-r4 at 5 and 20 tokens per parameter (8 runs), target m11-r5 (919K parameters) at 10 tokens per parameter (9.2M tokens) | laptop | measured: about 42 minutes (below) |

The free variants carry only part of Recipe-R: the Module 7 optimizer (`"via": "optim"` in the recipe file) on Data-v0 retokenized at vocabulary 1,024, because the Module 7 and Module 10 training wrappers cannot be stacked until the loop has both natively. Say so in your contract.

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) and your recipe file **before** the ladder runs. Fixed by the project:

- **Question:** does a ladder with the exact Recipe-R predict the final run's validation loss inside a pre-registered 90% interval, and its cloze accuracy (lesson 11.2) inside a pre-registered interval? Decision informed: whether the Recipe-R checkpoint is the model the course's later stages (Module 13's short pipeline on Recipe-R) can rely on, and how far the prediction method can be trusted for the capstone.
- **Hypothesis:** both measurements inside their intervals; the run stays inside the predicted band at every 10% of its schedule. Status: reported effect for loss (ladders in Llama 3, OLMo); downstream prediction is noisier (lesson 11.2), so a miss there is plausible.
- **Baseline:** the pre-registered prediction.
- **Changed variable:** size and tokens only, along the ladder and to the target. **Controlled:** every Recipe-R choice (the recipe file), the schedule shape (warmup 5%, cosine to 0.1× over each run's own length), the learning rate, seed 0, the evaluation windows and the 500 cloze items (test split, seed 0).
- **Comparison axis:** equal recipe; tokens per parameter stated for every run.
- **Metrics and decision rules:** final validation loss inside the 90% interval (success) or not; cloze accuracy inside its interval; the off-band rule (validation loss above the band's upper edge by more than 0.02 nats at 10% or more of the run) and the gradient-spike rule (above 4× the running median after 10%) applied during the run. If a rule fires, stop, diagnose, and decide in writing whether to resume.
- **Correctness checks:** `pytest labs/common/tests/test_scaling.py`; every ladder run and the target have the same recipe block in their run cards (`python -m frontierlab.record` on a ladder rung and the target shows only size and step differences); the pre-registration's digest verifies and it is older than the target's first log line.
- **Fallback evidence:** the Module 11 pilot traces (Colab) when published, labelled as analysis of provided traces.
- **Limits:** one seed; one target size; the cloze task is a proxy, not a benchmark; the free variants' recipe is partial.

## Steps and deliverables

1. **Contract and recipe file** (deliverables 1–2). Copy `labs/module-11/project/recipe_r_example.json`, replace every choice and evidence string with your own Modules 6–10 decisions (with the run card and interval behind each), and set `"via"` and `"args"` to the wrapper and flags that implement them.
2. **Ladder** (unattended): `python labs/module-11/project/run_project.py ladder --recipe my_recipe_r.json`.
3. **Pre-registration** (deliverable 3): `python labs/module-11/project/run_project.py predict --recipe my_recipe_r.json`. It writes `runs/m11/project/<variant>/prereg.json` once. Add one page: the fit, the held-out error of the largest rung, the interval and whether you widened it in your own write-up (lesson 11.3), the band, the downstream prediction, and the rules. Commit this page (or send it to a peer) before step 4.
4. **The run** (unattended): `python labs/module-11/project/run_project.py run --recipe my_recipe_r.json`. On the main path, the printed command is what you launch (`--variant main --print`); watch the band while it runs.
5. **The check** (deliverable 4): `python labs/module-11/project/run_project.py check --recipe my_recipe_r.json`, then one page: hit or miss for each prediction, the error against the held-out error you expected, the band at every 10%, and a post-mortem of any miss (which assumption broke: the law's form, the recipe, the tokens per parameter, the seed).
6. **Debugging task** (deliverable 5), below.
7. **Written defence** (deliverable 6), below.

### What the free CPU variant gave in this build

Measured 2026-10-07 on the build laptop (torch 2.14.1+cpu, 16 threads), with `recipe_r_example.json` (Muon with match-RMS scaling, peak learning rate 3e-3, Data-v0 at vocabulary 1,024). Timings: `ladder` 21 min 25 s (8 runs of 14–701 s), `predict` 3 min 27 s, `run` 17 min (4,487 steps), `check` 11 s; about 42 minutes in all.

**Ladder** (validation loss; m11-r1 … m11-r4 at 5 and 20 tokens per parameter): 5.790 / 4.882, 4.921 / 4.108, 4.412 / 3.830, 3.725 / 3.451. The fit through all 8 runs gives $E = 1.92$, $\alpha = 0.32$, $\beta = 0.37$. Holding out the largest rung (both of its runs) misses it by 0.100 nats on average.

**Pre-registered** for m11-r5 (919K parameters) at 10 tokens per parameter, 9.19M tokens, $5.4 \times 10^{13}$ FLOPs:

- loss 3.113, 90% interval [2.883, 3.360];
- cloze accuracy 0.723 [0.679, 0.760], through a predicted task NLL of 3.074 (the step-2 sigmoid's own uncertainty not included).

**Measured:**

- loss **3.282**: inside the interval, 0.169 above the point prediction;
- cloze accuracy **0.726**: inside its interval;
- task NLL 3.217 against 3.074 predicted.

The record was written before the run. Under the contract both predictions **hit**, and both hits need reading:

- *The loss interval was wide* (0.48 nats). It was honest about a ladder whose held-out error was 0.10, but too wide to decide much. A hit with a wide interval is weak evidence.
- *The accuracy hit was partly luck.* Step 1 missed the task NLL by 0.14 nats. Step 2's sigmoid, fitted only on ladder models, sat at almost the same accuracy for both NLL values, so the miss did not show.
- *The stopping rule would have stopped this run.* At 60% the loss was 3.439, against the band's upper edge of 3.407 plus the 0.02 tolerance, and it stayed above the band to the end (3.282 against an upper edge of 3.276). The band holds the exponents fixed at the final fit's values, so its interval is much narrower than the final-loss interval: [3.059, 3.276] at 100%, against [2.883, 3.360]. The rule would have killed a run that went on to meet its own success criterion.

That inconsistency in the method is the most useful finding of the build's project. A defensible record makes the band at least as wide as the endpoint interval at $f = 1$, for example by bootstrapping the exponents too, or by widening by the held-out error. The record should state the rule's false-alarm risk. Note it in your contract if you keep the default band.

## Debugging task

`labs/module-11/project/buggy_fit.py` is a colleague's prediction for the lesson 11.3 target. Their summary is in the script's header: they turned the 9 usable ladder runs into 36 data points by adding the evaluations logged at 30%, 50% and 70% of each run, and after the 11.3 target finished, their prediction was closer to the measured loss than the pre-registered one, so they want their pipeline used for Recipe-R. There are two bugs, and the closer number is not evidence against them. Start from the symptoms; for each, name the check that isolates it, run it, and show what the fixed pipeline predicts.

```bash
python labs/module-11/project/buggy_fit.py
python labs/module-11/project/buggy_fit.py --fixed     # only after your diagnosis
```

<details>
<summary>Hint</summary>

What is the learning rate of a run at 30% of its cosine schedule, and what would a run that *ends* after that many tokens have done to its learning rate by then? Separately: which parameter count does the fit use, and which one does the prediction plug in?

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07 on the build laptop (no training; `buggy_fit.py` 2 min, `--fixed` 2 min, almost all of it the 200 bootstrap refits). The colleague's pipeline prints: 36 points, in-sample RMS(log) 0.018, prediction for the 11.3 target 3.463 [3.432, 3.568], measured 3.434. That is an error of +0.029, against +0.076 for the pre-registered prediction. Their number is closer. Both bugs are still bugs.

**Bug 1: mid-schedule evaluations used as finished runs.** An evaluation at 30% of a cosine run is taken at a high learning rate. A run that *ended* after that many tokens would have decayed by then and reached a lower loss. So the 27 added points are all too pessimistic for their token counts, and they are most pessimistic where the schedule is earliest. Isolating check: plot (or print) the added points against the finished runs of a similar token count. In the ladder, m11-r3 at 30% of its $10^{13}$ run has seen 2.8M tokens and reads 4.00. The finished $3 \times 10^{12}$ run of the same rung saw 2.8M tokens and reached 3.93. The cause is the cosine-length effect of lesson 11.1 (Hoffmann et al. appendix B), now inside a fit. It also fakes a tighter fit: 36 points from 9 runs are not 36 independent measurements, so the bootstrap, which resamples points, is overconfident.

**Bug 2: the fit and the prediction count parameters differently.** `points()` fits on `N_nonemb`, while the prediction plugs in the target's `N_total` from the pre-registration. At these sizes the embedding is 25–71% of the model, so the target is treated as a much larger model than any fitted point of the same "size". Isolating check: print the fitted $N$ range and the $N$ used for the target: 13K–296K non-embedding against 394K total. Use one count throughout.

**The fixed pipeline** (`--fixed`: final losses only, total parameters for both) reproduces the pre-registered fit exactly ($E = 3.288$, $\alpha = 0.80$, $\beta = 1.11$) and prediction (3.510; its interval [3.490, 3.558] differs slightly from the record's because the bootstrap seed and count differ).

The two bugs partly cancelled the real error. That error is the floor the clean fit put too high, discussed in lesson 11.3. The pessimistic mid-schedule points pulled the fitted floor $E$ down to 2.83, which happened to move the prediction toward the truth. A pipeline that is right for the wrong reason is no protection on the next run: the size of the cancellation depends on the schedule and on the target. The defensible response to the 0.076 miss is the one in lesson 11.3: widen the interval to the error seen at a new size and token ratio combination, and add ladder runs that hold out such a combination. Do not adopt the pipeline that happened to land closer.

</details>

## Written defence

One to two pages, answering:

1. Which Recipe-R choice is the least supported by evidence, and how would a ladder have caught a wrong choice there before the final run?
2. Your target is trained at more tokens per parameter than most of your ladder. Which ladder runs carry the extrapolation in tokens, and how far is the target from them?
3. What was your held-out error on the ladder's largest rung, and was your pre-registered interval at least that wide? If not, why not?
4. If the final loss missed the interval, which of these explains it best, with evidence: the law's form, a recipe difference between ladder and target, seed noise, or a pipeline problem? If it hit, what is the most likely way it could have hit for the wrong reason?
5. What did the downstream prediction add to the loss prediction, and how would you report the cloze accuracy to someone deciding whether to post-train this checkpoint?
6. With 10× the budget, would you spend it on a larger target, more ladder rungs, or more seeds of the target? Which result would change your answer?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | contract, recipe file and rules written before the ladder; the pre-registration written before the target's first step |
| 2 | Controls | run cards show the same recipe for every rung and the target; only size, tokens and steps differ |
| 3 | Axis and budget parity | tokens per parameter and exact FLOPs stated for every run; the ladder's share of the target's compute reported |
| 4 | Correctness | `test_scaling.py`, the record-diff check, the digest and timing checks, all included |
| 5 | Uncertainty | bootstrap interval and held-out error both reported; the interval's level stated; one-seed limit stated |
| 6 | Conclusion matches evidence | hit or miss stated plainly; no claim beyond "this ladder predicted this run" |
| 7 | Limits | partial recipe in the free variants, one seed, proxy downstream task, scale |
