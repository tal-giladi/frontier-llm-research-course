# Module 16 project · An environment pack you can train on

This project turns the module into the artefact a team would hand to an RL run: an environment pack of three environments with repository-level splits, an audit of each environment's reward against the cheaper rewards someone might ship instead, and one agent RL result with a control arm. Before trusting the pack, you debug a colleague's version of it with four planted bugs, each of which trains without an error and produces a number that looks good.

**Time:** 5–7 attended hours plus about 5 minutes of CPU runs. **Folder:** [`labs/module-16/project/`](../labs/module-16/) (`pack.py`, `test_pack.py`, `run_project.py`, `buggy_pack.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

> [!NOTE]
> Scope, as in the whole module: no exploit code and nothing that manipulates a test harness, a report or an exit status. The audit's loophole candidates are lookup-table overfits, repairs that pass only the visible tests, and wrong answers consistent with what an agent observed. If you add candidates, keep to those kinds.

## Variants and cost

| Variant | What runs | Hardware | Cost |
|---|---|---|---|
| Main path | `python labs/module-16/project/run_project.py --variant main --print` prints the RL arms on Qwen3-1.7B-Base (revision `ea980cb`): the probe task in text form with the function-level split, gold reward and a random-reward control, 2 seeds × 150 steps (`frontierlab.agents.hf_agent`). The pack and the audit run on CPU in every variant | 1× H100 80 GB | **PROJECTED, pending the Module 16 pilot:** 4 runs × 150 steps × 12–22 s per step (3 generate calls of assumed 2–4 s plus an update of assumed 6–10 s) = 2–4 GPU-hours, USD 4–12 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | the same commands with `--model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8` | T4 | PROJECTED 1.5–3 hours per run (assumed 3–4× slower per step than the main path at a quarter of the batch) |
| Free CPU | `python labs/module-16/project/run_project.py` | laptop | measured 3.3 minutes (pack and audit 18 s, RL 3.0 minutes including the warm start); `PACK=buggy pytest labs/module-16/project` 15 s |

## Deliverables

1. **The pack:** your `pack.py` (start from the course one, or build your own with the same layout), the split manifest `runs/m16/project/pack.json`, and `pytest labs/module-16/project` passing on it.
2. **The reward-misspecification audit:** for each environment, the training reward and at least one cheaper alternative on candidates of known verdict, with false accepts and false rejects as counts out of totals. Then one paragraph per environment: which loophole the cheaper reward lets through, and which defence of lesson 16.1 the pack's reward relies on.
3. **The RL result:** the experiment record (contract, run cards, `metrics.jsonl` of every run, the printed table), held-out accuracy paired by seed against the control, the detectors of lesson 16.4 with the thresholds stated in advance, and the behaviour metrics of lesson 16.3.
4. **The debugging report** (below) and **the written defence** (below).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running. Fixed by the project:

- **Question:** does RL with the probe task's gold reward raise accuracy on the **held-out functions** of the function-level split (functions the warm start and the RL run never saw) beyond a random-reward control? Decision informed: whether the pack's probe environment measures generalisation of the skill, or only learning of the training functions.
- **Hypothesis:** accuracy on the training functions rises far above the control; on held-out functions, any gain is small and may be absent. Status: the generalisation part may not appear at this scale (a 0.3M-parameter policy, 8 training functions).
- **Baseline:** the random-reward control (same warm start, settings and seeds).
- **Changed variable:** the reward (gold vs Bernoulli 0.5). **Controlled:** the warm start (`runs/m16/sft-probe-fsplit`, trained on the 8 training functions only), 16 × 8 episodes per step, 60 steps, lr $3 \times 10^{-4}$, outcome credit, loss on actions, 2 queries, seeds 0–2, evaluation on 66 held-out-function episodes with a fixed sampling seed.
- **Comparison axis:** equal episodes and updates.
- **Metrics and decision rule:** held-out-function accuracy at step 60, paired by seed against the control, 95% t-interval above 0 for "generalises". Training-function accuracy is reported, not decided on. Detectors: lesson 16.4's thresholds.
- **Correctness checks:** `pytest labs/module-16/project` on your pack; `test_agents.py`; `obs_tokens_in_loss` is 0 in every run.
- **Fallback evidence:** lesson 16.2's split comparison and SWE-smith section 4.1, labelled as published.
- **Limits:** toy task, model and steps; 3 seeds; 3 held-out functions; "generalisation" here means new constants and families in a 25-character grammar.

<details>
<summary>What the build's free-CPU run gave (compare after your own report)</summary>

Measured 2026-10-07 on the build laptop (8 threads, other jobs running).

**Pack** (`pack.json`): code 18 tasks, 12 train / 6 held (held-out kit `kit-arith`); swe 76 tasks, 65 / 11 (held-out repository `mathkit`); probe 11 functions, 8 / 3 (held out `add2`, `aff1`, `aff2`). No group on both sides.

**Audit:**

| Environment | Reward | false accepts | false rejects |
|---|---|---|---|
| code | pack (robust verifier) | 0 of 12 | 0 of 6 |
| code | visible pairs only | 8 of 12 | 0 of 6 |
| swe | pack (all 12 tests) | 0 of 91 | 0 of 76 |
| swe | first 4 tests only | 24 of 91 | 0 of 76 |
| probe | pack (gold, inputs 0–19) | 0 of 42 | 0 of 11 |
| probe | consistent with the agent's queries | 20 of 42 | 0 of 11 |

The swe audit used *natural* overfits: in 15 of 76 tasks, the search solver of lesson 16.2 found a repair that passes the first 4 tests and differs from the reference on the rest. Nobody wrote them; they were found by optimising against the cheaper reward, which is how a policy finds them too. The 24 false accepts of the 4-test reward are those 15 plus 9 synthesised bugs whose failing tests are not among the first 4. The probe's "consistent with the queries" reward accepts every wrong program that agrees with the one point the agent queried, and every one-entry table.

**RL** (held-out functions; start 0.023):

| Arm | held-out-function accuracy (seeds 0, 1, 2) | training-function accuracy (last 10 steps) | queries | turn limit | KL | flags |
|---|---|---|---|---|---|---|
| gold reward | 0.000, 0.000, 0.000 | 0.933 | 1.01 | 0.010 | 0.38 | none |
| random control | 0.015, 0.019, 0.155 | 0.153 | 0.63 | 0.114 | 0.15 | none |

Gold minus control on held-out functions: −0.063 [−0.261, +0.135]. Under the rule this is "does not generalise", a clean negative result. The policy learned the 8 training functions almost perfectly (0.93) and none of the 3 it never saw. A random split of the same episodes would have reported something near 0.93. This is lesson 16.2's point measured on an RL result, and it is why the pack's probe split is by function. No detector fired, correctly: the reward was the gold check itself. The control's seed-2 value (0.155) is noise around a policy that drifted; with three seeds it widens the interval.

</details>

## Debugging task

`labs/module-16/project/buggy_pack.py` is a colleague's pack. Their message is at the top of the file: workers reuse environments "so it runs faster", the code reward is "the quick verifier", the split is by task id "so both sides get every kit", and the probe loss covers the whole response "so the policy also learns to read tool output". A first run "looked great: training reward 0.95 on code within 40 steps". There are four bugs. For each one, name the check that finds it, the training-time symptom it would cause, and the one-line fix.

```bash
PACK=buggy pytest labs/module-16/project                          # which pack checks fail?
PACK=buggy python labs/module-16/project/run_project.py --part pack,audit
pytest labs/module-16/project                                     # the course pack: all pass
```

<details>
<summary>Hint</summary>

Every one of the four is something a lesson of this module measured: a reset check (16.1), a split unit (16.2), a loss mask (16.3), a reward that a lookup table satisfies (16.1 and 16.4). The colleague's 0.95 is a symptom, not a success.

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07: `PACK=buggy pytest labs/module-16/project` fails exactly four tests (15 s).

1. **`LeakyCodeEnv` (reset reuses the workspace).** Found by `test_code_env_resets` (`reset_restores` is False). Symptom: rewards depend on rollout order and worker, and a solution written in one episode is present at the start of the next, so later episodes on the same worker earn reward for files they did not write. Fix: return a fresh copy of the task's files in `_initial_state` (the course `CodeEnv`).
2. **The code reward is the visible-only verifier.** Found by `test_code_reward_rejects_registered_loopholes` (8 of 12 wrong or loophole submissions accepted: every lookup table and two plausible bugs) and by the audit. Symptom: what lesson 16.4 measured on the program world. Training reward climbs fast while a gold check stays flat or falls, and the output monitor (`looks_hardcoded`) flags the submissions. The colleague's "0.95 within 40 steps" is that curve. Fix: the robust verifier.
3. **The code split is keyed by task id.** Found by `test_no_group_crosses_the_split[code]` and the pack table (all three kits on both sides). Symptom: held-out scores include siblings of training tasks; lesson 16.2 measured 0.98 held-out success for a pure memoriser under instance splits against 0.00 under repository splits. Fix: key by `t.repo`.
4. **The probe loss covers observation tokens (`loss_on: "all"`).** Found by `test_probe_loss_excludes_observations`, and in training by `obs_tokens_in_loss > 0` in every row. Symptom: lesson 16.3's all-tokens arm, accuracy −0.22 against the baseline with more turn-limit hits. Fix: `loss_on: "actions"`.

Which ones a training curve would have shown: none clearly. Bugs 2 and 3 make the curves look *better*, bug 1 makes them noisy, and bug 4 makes them a little worse. Only the pack's checks, run before the RL job, find all four.

</details>

## Written defence

One to two pages, answering:

1. Your split units: for each environment, the claim its held-out score supports, and one way the same code could still cross the split.
2. Your audit: which cheaper reward would have done the most damage in RL, and how you know without running it.
3. Your RL result with its interval. What does the gap between training-function and held-out-function accuracy say, and what would you change (data, task design, scale) to test generalisation properly?
4. Which detector would have caught each of the four planted bugs during training, if any, and how quickly?
5. What does a clean audit (no false accepts) not tell you?
6. Name one environment you would add to the pack next, its split unit, its reward, and the first candidate you would register for it.

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the generalisation question, the split units and the detector thresholds written before any run |
| 2 | Controls | run cards identical except the reward; the control scored with the gold check like the gold arm |
| 3 | Axis and budget parity | equal episodes and updates checked in the logs; verifier and evaluation compute stated |
| 4 | Correctness | `pytest labs/module-16/project` output for your pack; all four planted bugs found with their checks |
| 5 | Uncertainty | seed-level interval against the control; counts, not only rates, in the audit |
| 6 | Conclusion matches evidence | no generalisation claim from training-function accuracy; no "safe" claim from a clean audit |
| 7 | Limits | toy scale, three held-out functions, the registered candidate kinds only |
