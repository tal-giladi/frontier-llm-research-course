---
id: "16.1"
module: 16
minutes: 35
practice_minutes: 60
prerequisites: ["12.1", "12.4", "14.2"]
objectives:
  - Name the five parts of an agent environment (task, tools, state, verifier, reset) and say for each one what goes wrong in RL when it is built carelessly.
  - Test a verifier like code by running it on registered correct, wrong and loophole submissions, and report its false-accept and false-reject rates.
  - Compute how many random hidden tests a verifier needs to miss a bug with a stated probability, and check the prediction against measured verdicts.
  - Write a reset check and a determinism check for an environment, and recognise a workspace that leaks between episodes.
  - Choose reward channels that the agent cannot write (hidden tests, fresh inputs, property checks, results compared outside the agent's process) and say what each one does not catch.
volatility: concept
sources:
  - title: "Jimenez et al., SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (section 2: execution-based evaluation with FAIL_TO_PASS and PASS_TO_PASS tests)"
    url: https://arxiv.org/abs/2310.06770
  - title: "OpenAI, Introducing SWE-bench Verified (2024-08-13: 1,699 samples screened by 93 developers; tests that reject valid solutions; 500 kept)"
    url: https://openai.com/index/introducing-swe-bench-verified/
  - title: "Xie et al., OSWorld (section 2.2.3: execution-based evaluation of the final state; 134 evaluation functions; 302 initial states)"
    url: https://arxiv.org/abs/2404.07972
  - title: "Zhong, Raghunathan, Carlini, ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases (sections 4, 5.2)"
    url: https://arxiv.org/abs/2510.20270
  - title: "METR, Recent Frontier Models Are Reward Hacking (2025-06-05)"
    url: https://metr.org/blog/2025-06-05-recent-reward-hacking
last_verified: "2026-10-07"
---

# 16.1 · Environments and verifiers

An agent environment is a program the policy talks to for many turns, and in RL its verifier is the only definition of success the policy will ever see. This lesson builds the course's environment interface (task, tools, state, verifier, reset), then tests the verifiers themselves the way you would test any code: on submissions whose right verdict is known, counting false accepts and false rejects. It ends with the reward channels that a policy cannot write, and with what each of them still misses.

## Why this matters at a frontier lab

Agentic RL runs on thousands of environments written quickly by many people, and every one of them is a reward function. A verifier that accepts a wrong answer is not a small measurement error: RL finds the accepted wrong answers and makes them common, because they are often cheaper to produce than right ones (lesson 16.4 does this on purpose). A verifier that rejects right answers wastes compute and teaches the policy to avoid correct solutions it happened to write differently. Both failures are documented on public benchmarks. OpenAI had 93 developers screen 1,699 SWE-bench samples and dropped those whose tests "filter out valid solutions" or whose issues were underspecified, keeping 500 as SWE-bench Verified (PUBLICLY DOCUMENTED). METR reports that recent models exploited bugs in scoring code on tasks that had been pre-tested against cheating. A research engineer who ships an environment ships its verifier tests with it.

## The idea

### Five parts

The interface is `frontierlab/agents/env.py`. Each part has its own typical bug:

| Part | What it is | Typical bug and its effect in RL |
|---|---|---|
| **Task** | the instance: prompt, files, the hidden data only the verifier sees, and the **grouping keys** (`repo`, `family`) a split must respect | hidden data leaks into the prompt or the workspace; tasks generated from one source split across train and test (lesson 16.2) |
| **Tools** | functions the agent calls: `(name, kwargs) -> (observation, done)` | a tool that crashes ends the episode with a stack trace instead of an observation; a tool that changes state the verifier reads |
| **State** | what the tools read and change (here a dictionary of files) | state shared between rollouts, so one episode's edits are another's starting point |
| **Verifier** | `verify() -> Verdict(reward, passed, details)` from the **final state** | rewards a channel the agent can write (its own report, a log line, an exit status); checks too few cases |
| **Reset** | `reset(task, seed)` returns the exact initial state | reuses the old state object (`LeakyCodeEnv`), so results depend on rollout order |

The verifier looks at what the episode left behind, never at what the agent says it did. OSWorld's design follows the same rule: it scores the final state of the machine with task-specific evaluation scripts (134 evaluation functions over 302 initial states, section 2.2.3), not the agent's transcript.

### A verifier is a classifier

For a set of submissions with known right verdicts, a verifier has two error rates:

$$\mathrm{FA} = \frac{\#\{\text{wrong or loophole submissions accepted}\}}{\#\{\text{wrong or loophole submissions}\}}, \qquad \mathrm{FR} = \frac{\#\{\text{correct submissions rejected}\}}{\#\{\text{correct submissions}\}}.$$

FA is the rate at which reward flows to behaviour you did not want. FR is the rate at which correct behaviour goes unrewarded. Lesson 12.1 measured both for answer-string verifiers on the arithmetic world. Here the submissions are programs, and the probe set is a **candidate registry** (`codeenv.CANDIDATES`): each name maps to `(kind, files)`, with `kind` one of `correct`, `wrong`, `loophole`. `register(name, kind, files)` adds a candidate without editing the module, and `verifier_report(tasks)` runs every verifier on every candidate. The course ships three candidates: the reference solution, a plausible bug (an off-by-one, a missing `.lower()`), and a **visible-pair lookup table** that returns the expected outputs of the examples the agent can see and nothing useful otherwise. The lookup table is the only loophole this module uses. It is what a verifier with a few visible pairs invites, and it is harmless.

### How many hidden tests

Suppose a bug fails on a fraction $p$ of random inputs, and the verifier checks $n$ independent random inputs. It accepts the bug only if every input misses:

$$P(\text{accept bug}) = (1 - p)^n, \qquad n_{\min}(\delta) = \left\lceil \frac{\ln \delta}{\ln(1 - p)} \right\rceil .$$

Here $p$ is the bug's failure rate, $n$ the number of test inputs, and $\delta$ the miss probability you accept. Two consequences follow. A bug that shows on 1 input in 100 needs about 300 random tests for $\delta = 0.05$, so rare-input bugs need targeted tests (edge cases, properties), not more random ones. And a lookup table has $p \approx 1$ on any input it has not seen, so *any* hidden input catches it. The table's weakness is that it saw only the visible ones.

### Reward channels the agent cannot write

The robust verifier (`codeenv.robust_verify`) combines four defences. Each is REASONABLE INDUSTRY PRACTICE, and each has a gap:

1. **Hidden cases** that never enter the workspace. Gap: a fixed hidden set can be overfit across many training epochs if it is reused, and it can be leaked by anything that copies it into the prompt.
2. **Fresh inputs** drawn at verification time from the task's input generator, with a random nonce as the seed. Gap: only as good as the generator; it cannot reach inputs the generator never produces.
3. **Property checks** from the specification that need no expected value (for `clamp`, the output lies in $[lo, hi]$ and equals $x$ when $x$ is inside). Gap: a property is necessary, not sufficient.
4. **Isolation of the comparison.** Only the declared source files are copied out of the workspace into a fresh directory next to a harness the course owns. The harness calls the function and writes plain JSON values to a file whose name contains a random nonce, and the comparison with expected values happens in the verifier's own process. So nothing the agent writes besides its source code takes part in scoring. Gap: the submitted code still runs. The sandbox (`frontierlab/agents/sandbox.py`) blocks accidents, not a determined program, and its docstring says so: code written by a capable model belongs in a container or VM without network.

ImpossibleBench measured the access side of this directly on frontier models. With tests hidden, cheating on its impossible SWE-bench tasks fell "to near zero", and so did legitimate performance; read-only test access was "a middle ground" (section 5.2, PUBLICLY DOCUMENTED). Hiding information from the agent has a cost too.

## Worked example

**Error rates.** Six tasks, three candidates each. The visible-only verifier accepts all 6 lookup tables and 2 of the 6 plausible bugs: $\mathrm{FA} = 8/12 = 0.667$, loophole accept rate $6/6 = 1$, $\mathrm{FR} = 0/6$. The robust verifier accepts none of the 12 and all 6 references: FA 0, FR 0. These are the measured numbers of the lab's first seed.

**Tests needed.** The `second_largest` bug (it forgets to de-duplicate) fails on $p = 0.176$ of random inputs. With the 3 visible pairs plus 2 hidden, $P(\text{accept}) = 0.824^5 = 0.38$. For $\delta = 0.05$: $\ln 0.05 / \ln 0.824 = -2.996 / -0.1936 = 15.5$, so 16 random tests. `rle`'s bug fails on $p = 0.878$: $\ln 0.05/\ln 0.122 = 1.43$, so 2 tests.

**A property catches what a test set misses.** `clamp(x, -5, 7)` with the bug `max(lo, min(x, hi - 1))` returns 6 for $x = 9$. A test set without inputs above 6 never sees it. The property "the output equals $x$ when $lo \le x \le hi$" fails at $x = 7$, and any fresh input in $[7, 20]$ breaks the expected value.

**Reset.** `LeakyCodeEnv` returns the same dictionary on every `reset`. One episode writes `solution.py`, the next episode starts with that file already written, and its reward depends on which episode ran first. The reset check (reset, digest, disturb, reset, digest) shows `reset_restores: False`.

## Shapes and cost

| Object | Size | Notes |
|---|---|---|
| `Task.data["visible"]`, `["hidden"]` | 3 and 8 (input list, expected value) pairs, plain JSON | hidden pairs never enter `files` |
| one verification | one sandboxed `python -I` process | measured 0.30 s per call on the build laptop (Windows process start dominates) |
| `verifier_report` | tasks × candidates × verifiers calls | 18 × 3 × 3 = 162 calls in 49 s |
| program-world rewards | in-process, microseconds | the RL labs score 128 responses per step |

At production scale verification is often the bottleneck of agentic RL: each SWE-style episode needs its repository's container image, and its tests can take minutes. That cost belongs in every comparison's budget, like the judge compute of Module 13 (plan section 9).

## Build it

```python
from frontierlab.agents import codeenv as C
from frontierlab.agents.env import reset_check, determinism_check, run_episode

tasks = C.basic_tasks(seed=0)                      # six toy specs, 3 visible + 8 hidden pairs each
rep = C.verifier_report(tasks)                     # every verifier x every registered candidate
rep["summary"]["visible"]                          # {'false_accepts': 8, 'loophole_accepts': 6, ...}

C.register("my_case", "wrong", lambda task: {"solution.py": "..."})   # add a probe without editing codeenv
reset_check(C.LeakyCodeEnv, tasks[0], disturb=lambda env: env.step(("write", {"path": "solution.py", "text": "x"})))
```

`run_episode(env, task, agent)` drives any environment with any agent, a function from the history to the next action, and returns a `Trajectory` with the verdict. Unknown tools and bad arguments come back as observations, not exceptions, because a crash in the middle of a batch of rollouts loses every episode in it. Correctness checks: `pytest labs/common/tests/test_agents.py -k "sandbox or reset or verifier or registry or properties"` (timeouts, the network and process guard, the reset and determinism checks, the registry's rules, and that every property holds on 50 reference outputs per spec).

## What the evidence says

- **Execution-based verification against hidden tests: ESTABLISHED.** SWE-bench scores a patch by running FAIL_TO_PASS and PASS_TO_PASS tests in the repository (section 2); OSWorld scores the final machine state with per-task scripts (section 2.2.3). Both are PUBLICLY DOCUMENTED.
- **Verifiers have false rejects that matter: PUBLICLY DOCUMENTED** (SWE-bench Verified's screening of tests that reject valid solutions; OSWorld-Verified's July 2025 revision lists overly strict evaluators among about 300 fixed issues, lesson 16.5).
- **Hiding tests reduces exploitation and costs performance: PUBLICLY DOCUMENTED** for ImpossibleBench's setting (section 5.2). How large the trade is in your environment is a measurement you have to make.
- **Property checks and fresh inputs in RL verifiers: REASONABLE INDUSTRY PRACTICE.** No public ablation compares them inside an RL run at scale. Lesson 16.4 runs the toy version.
- **Course measurement (free CPU, 2026-10-07):** on 18 tasks (6 specs × 3 seeds), the visible-only verifier's false-accept rate was 0.667 with every lookup table accepted; the hidden verifier (3 visible + 8 hidden) accepted 1 of 36 wrong or loophole submissions (a `running_max` bug whose failing inputs never came up); the robust verifier accepted none; no verifier rejected a reference solution. Details in the lab's results box.

## Lab

**Folder:** [`labs/module-16/lesson-01/`](../../labs/module-16/) · **Time:** about 1 hour · **Pass check:** `pytest labs/module-16/lesson-01` passes and `verifier_lab.py` prints parts A–C; your write-up gives each verifier's error rates and one candidate you registered yourself.

### Experiment contract

- **Question:** which of the three code verifiers has false-accept and false-reject rates low enough to be an RL reward for the toy coding tasks, and how many hidden tests does each spec need? Decision informed: which verifier the module's environment pack uses.
- **Hypothesis:** the visible-only verifier accepts every lookup table; the hidden verifier accepts some plausible bugs with low failure rates; the robust verifier has FA = FR = 0 on the registered candidates. Status: follows from the construction; the hidden verifier's misses are random and may not appear in 3 seeds.
- **Baseline:** the visible-only verifier.
- **Changed variable:** the verifier. **Controlled:** the tasks (6 specs × seeds 0–2), the candidates, the sandbox settings.
- **Comparison axis:** the same submissions for every verifier.
- **Budget:** free CPU; measured 71 s.
- **Metrics and decision rule:** FA and FR per verifier with their counts. A verifier is usable as a training reward if FA = 0 and FR = 0 on every registered candidate, and every spec's $n_{\min}(0.05)$ is at most its hidden-test count. Counts this small cannot show FA below about 1 in 36, so state the bound.
- **Correctness checks:** `test_agents.py`; your TODO tests; the reset check proves something only if the disturbance changed the state (`reset_verdict` says "untested" otherwise).
- **Fallback evidence:** none needed; the lab is deterministic apart from the fresh inputs.
- **Limits:** three candidate kinds and six tiny specs; the registered loophole is the only one tested, so FA = 0 says nothing about loopholes nobody registered.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | CPU is enough. On a machine with Docker, run the same verifiers with each sandboxed call inside a container with no network and a read-only mount (REASONABLE INDUSTRY PRACTICE for code written by capable models); the course code does not wrap Docker for you | `verifier_lab.py --seeds 10` for tighter counts (PROJECTED about 4 minutes) |
| Free GPU | not needed | — |
| Free CPU | laptop; measured 71 s (part A 49 s for 162 sandboxed calls) on the build laptop (Windows 11, Python 3.12.13, torch 2.14.1+cpu) | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` (`confusion`, `miss_probability`, `tests_needed`, `reset_verdict`) and run `pytest labs/module-16/lesson-01`.
2. **Run** `python labs/module-16/lesson-01/verifier_lab.py`. Read part A1 against your contract. In A2, compare the measured survivals of the plausible bug with the prediction $(1-p)^{n+3}$ (the hidden verifier also runs the 3 visible pairs).
3. **Register a candidate of your own** in a separate file (for example `my_candidates.py` that imports `frontierlab.agents.codeenv` and calls `register`), run `verifier_report` with it, and add it to your table. Good choices: a second plausible bug that fails only on an edge case (an empty list, a negative number), or a correct solution written differently (a false reject would mean the verifier checks form, not behaviour). Keep to candidates of the three kinds above: the module does not test anything that interferes with the harness.
4. **Write up** half a page: the error-rate table, the hidden-test count you would require per spec and why, and which defence of the robust verifier your own candidate tested.

<details>
<summary>Hint for TODO 3</summary>

Solve $(1-p)^n \le \delta$ for $n$: $n \ge \ln\delta / \ln(1-p)$ (both logarithms are negative). Use `math.ceil`, and handle $p \le 0$ and $p \ge 1$ before taking the logarithm.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (16 threads, Windows 11, Python 3.12.13; other jobs running): 71 s in all.

| Verifier | false accept | loophole accept | false reject |
|---|---|---|---|
| visible (3 pairs) | 0.667 (24 of 36) | 1.000 (18 of 18) | 0.000 |
| hidden (3 + 8 pairs) | 0.028 (1 of 36) | 0.000 | 0.000 |
| robust (+ 8 fresh inputs, properties) | 0.000 | 0.000 | 0.000 |

Failure rates of the plausible bugs on 2,000 random inputs: `clamp` 0.330, `running_max` 0.286, `count_vowels` 0.579, `rle` 0.878, `second_largest` 0.176, `digit_sum` 0.504, so $n_{\min}(0.05)$ = 8, 9, 4, 2, 16, 5. The one false accept of the hidden verifier is `running_max` at one seed: its bug shows only when a list starts with a negative number, and none of that seed's 11 inputs did (probability $0.714^{11} \approx 0.025$ per seed). It survived at every hidden count because the counts share their first inputs. Fresh inputs found it. Reset: `CodeEnv` clean, `LeakyCodeEnv` leaks, a read-only disturbance "untested". In the program world (part C), a table earns full `format` and `visible` reward and nothing from `hidden`, `randomised`, `property` or `gold`; an unfinished long string earns the full `length` reward and nothing else. Lesson 16.4 trains against exactly these rows.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-16/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-16/lesson-01`.

</details>

## Common mistakes

- **Testing the environment, not the verifier.** "The reference solution passes" is one cell of the table. Run the verifier on wrong and loophole submissions too.
- **Reporting FA = 0 as "the verifier is safe".** It is zero on the registered candidates. Say how many there were and which kinds.
- **Scoring a channel the agent can write.** Its own summary line, a log file, an exit status. Compare plain values in a process the agent's code does not control.
- **More random tests for a rare-input bug.** At $p = 0.01$ you need about 300 random tests for $\delta = 0.05$. Write the edge case or the property instead.
- **A reset that returns the old object.** Rewards then depend on rollout order. Run the reset check with a disturbance that actually changes the state.
- **Treating the sandbox as a security boundary.** It stops accidents. Code from a capable model runs in a container or VM without network.

## References

- C. E. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, 2023, section 2. https://arxiv.org/abs/2310.06770
- OpenAI, *Introducing SWE-bench Verified*, 2024-08-13. https://openai.com/index/introducing-swe-bench-verified/
- T. Xie et al., *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*, 2024, section 2.2.3. https://arxiv.org/abs/2404.07972
- Z. Zhong, A. Raghunathan, N. Carlini, *ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases*, 2025, sections 4 and 5.2. https://arxiv.org/abs/2510.20270
- S. Von Arx, L. Chan, B. Barnes (METR), *Recent Frontier Models Are Reward Hacking*, 2025-06-05. https://metr.org/blog/2025-06-05-recent-reward-hacking
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[16.2 · Software-engineering tasks](lesson-02.md)
