---
id: "16.2"
module: 16
minutes: 35
practice_minutes: 45
prerequisites: ["16.1", "10.6", "01.3"]
objectives:
  - Describe how SWE-smith and SWE-Gym build software-engineering tasks (synthesised breakage of existing tests versus real issues with their pull requests) and what each choice costs and buys.
  - Synthesise validated tasks from a repository by procedural modification, keeping only candidates that break at least one passing test.
  - Split generated tasks by instance, function, repository and family, and measure the sibling leak of each split.
  - Show with a memorising solver and a non-memorising solver that random splits of related generated tasks overstate generalisation, and choose the split unit for a stated claim.
  - Measure file and failing-test overlap between train and held-out instances of a real SWE-smith shard under three splits.
volatility: concept
sources:
  - title: "Yang et al., SWE-smith: Scaling Data for Software Engineering Agents (abstract; section 2.1 strategies and validation; section 2.2 storage; section 4.1 repository scaling and specialisation)"
    url: https://arxiv.org/abs/2504.21798
  - title: "Pan et al., Training Software Engineering Agents and Verifiers with SWE-Gym (section 3: 2,438 instances, 11 repositories separate from SWE-bench; SWE-Gym Lite)"
    url: https://arxiv.org/abs/2412.21139
  - title: "SWE-bench/SWE-smith dataset (MIT; revision ea6d717)"
    url: https://huggingface.co/datasets/SWE-bench/SWE-smith
  - title: "Jimenez et al., SWE-bench (section 2: task instances from merged pull requests)"
    url: https://arxiv.org/abs/2310.06770
last_verified: "2026-10-07"
---

# 16.2 · Software-engineering tasks

Software-engineering agents are trained on tasks of the form "this repository has a failing test; make it pass without breaking the others". Real issues are scarce, so the largest open task sets are generated: SWE-smith breaks working code on purpose and keeps every break that a test notices. Generation makes many tasks from one source, and those siblings share code and often the same fix. This lesson synthesises tasks the same way on four toy repositories, splits them four ways, and shows with two solvers which held-out scores measure generalisation and which measure memory. It then measures the same overlap on a real SWE-smith shard.

## Why this matters at a frontier lab

A coding-agent result is usually reported as a resolve rate on held-out tasks, and the decision it informs ("this data or this RL recipe makes the agent better at software engineering") is a claim about tasks the model has not seen. If held-out tasks were generated from the same functions, files or repositories as training tasks, the number partly measures how well the model remembers fixes it was trained on. That number goes up with more training on the same sources whether or not anything general was learned. The practice that protects against this is old (split by the unit of duplication, lesson 10.6) but easy to lose when a generator makes hundreds of tasks per repository. The two public task sets made opposite choices about where their tasks come from, and both had to state which repositories they kept away from evaluation.

## The idea

### Two ways to get tasks

| | SWE-Gym (Pan et al., section 3) | SWE-smith (Yang et al., section 2) |
|---|---|---|
| source | real GitHub issues and the pull requests that closed them | working code broken on purpose |
| scale (PUBLICLY DOCUMENTED) | 2,438 Python instances from 11 repositories; SWE-Gym Lite, 230 simpler instances | 50k instances from 128 repositories (abstract); generation cost about USD 1,360 in total (section 2.2 and its cost table) |
| environment | one executable environment per instance | one environment per repository, shared by its instances: 295 GB against "50 to 150 TBs" for per-instance images (section 2.2) |
| evaluation hygiene | its repositories "are separate from those used in SWE-Bench to avoid contamination" | removes "all 12 SWE-bench test repositories from consideration" before generating (section 2.1) |

SWE-smith's four strategies (section 2.1) are: an LM rewrites or modifies a function; procedural AST modification ("remove a conditional/loop, change an operator"); combining candidate bugs from the same file or module; and inverting a merged pull request. A candidate is kept only if it "break[s] one or more existing, passing tests". The test that broke becomes the task's FAIL_TO_PASS test, and the original code is the reference fix. That validation step is a verifier test in lesson 16.1's sense: a candidate that breaks nothing would be a task whose reward the unmodified code already earns.

### Siblings and the split unit

Generation makes **siblings**: tasks from the same function differ by one changed operator or constant, share almost all their code, and have the *same* fix. In the course's toy synthesis, 76 validated tasks come from 14 functions. A random split of instances puts siblings on both sides. A solver that retrieves the fix of the most similar training task then "solves" held-out tasks it has, in effect, already seen.

Choose the split unit from the claim:

| Claim | Split unit |
|---|---|
| "fixes new bugs in code it has trained on" | instance (honest for this claim only) |
| "fixes bugs in functions or files it has not seen" | function or file |
| "works on repositories it has not seen" | repository (the usual claim behind a benchmark score) |
| "handles a kind of bug it has not seen" | family, *and* repository or function, or siblings still leak |

A family split alone does not remove siblings: the other families' modifications of the same function stay in training.

### Two solvers that separate memory from skill

- **Retrieval** proposes the fix of the training task whose buggy code is most similar (token Jaccard). It uses nothing but memory.
- **Search** tries every single modification of the buggy code and keeps the first one that passes 4 visible tests. It uses no training data at all, so its success should not depend on the split.

If a model's held-out score moves like retrieval's across split units, the evaluation is measuring memory. A score that stays flat, like search's, is measuring something that transfers.

## Worked example

Twelve tasks come from four functions, three siblings each. Hold out a quarter of the **instances** at random: 3 tasks. Each held-out task has 2 siblings among the other 11 tasks, and the other 2 held-out tasks are drawn from those 11, so the probability that *both* siblings are also held out is $\binom{2}{2}/\binom{11}{2} = 1/55$. The sibling leak (the share of held-out tasks with a sibling in training) is therefore $1 - 1/55 \approx 0.98$, and a retrieval solver solves about 98% of the held-out tasks. Hold out a quarter of the **functions** instead (1 function, 3 tasks): the leak is 0 and retrieval solves nothing. Search, which repairs by trying modifications, scores the same in both cases up to which tasks happened to be held out.

On the real shard the same arithmetic is starker. 3,696 instances come from 14 repositories, about 264 per repository. Under a random instance split, a held-out instance has hundreds of training siblings from its repository. The lab measures that 98.9% of held-out patches touch a file that some training patch touches.

## Shapes and cost

| Object | Size | Cost |
|---|---|---|
| toy synthesis | 4 repositories, 16 functions, 81 candidates, 12 tests each | one sandboxed process per repository; measured 1.0 s |
| one solver pass | held-out tasks × proposals × 12 tests | one sandboxed batch; part A measured 7 s for 24 passes |
| SWE-smith shard | 3,696 rows (instance id, patch, FAIL_TO_PASS) | 4.1 MB parquet; part B measured 2 s |
| full SWE-smith dataset | 11 shards at revision `ea6d717` | CPU only; PROJECTED under a minute plus the download |

Part B reads metadata only and runs no repository code. Building and running SWE-smith's per-repository images is what the main-path agent RL of this module would need. It is not part of this lab.

## Build it

```python
from frontierlab.agents import swetasks as S
tasks, stats = S.synthesise()                       # validated: each breaks >= 1 of its function's tests
train, held = S.split(tasks, by="repo")             # instance | function | repo | family
S.solve_and_score(held, train, "retrieval")         # {'held': 11, 'solved': 0, 'rate': 0.0}
S.solve_and_score(held, train, "search")            # no training data used
S.swesmith_overlap(ids, patches, fail_to_pass, by="instance")   # real-data overlap from metadata
```

`mutate(source, family)` returns every single-site modification of one family (`flip_compare`, `shift_const`, `swap_binop`, `swap_minmax`, `drop_not`) using Python's `ast` module. Validation runs the reference and every candidate in one sandboxed process per repository, compares outputs, and records why each discarded candidate was dropped. All code here is course code or its mutations; no model output is executed. Correctness checks: `pytest labs/common/tests/test_agents.py -k "mutations or synthesised or group_splits or swesmith"`.

## What the evidence says

- **Repository-level separation between training tasks and evaluation: ESTABLISHED** practice for code-agent data (SWE-Gym section 3; SWE-smith section 2.1). PUBLICLY DOCUMENTED.
- **More repositories help more than more instances per repository: PROMISING.** SWE-smith trains on 700 trajectories drawn from 4, 25, 50 or 100 repositories and reports an "approximately logarithmic" gain with the number of repositories (section 4.1). One lab, one model family.
- **Specialising on one repository raises scores on that repository: PUBLICLY DOCUMENTED, small n.** SWE-smith fine-tuned SWE-agent-LM-32B on 700 SymPy trajectories (tasks generated from a commit before 2022-01-01) and went from 33.3% to 42.4% on 22 SymPy instances of SWE-bench Verified created after that date (section 4.1). The time cutoff is what makes this a legitimate within-repository claim. Without it, the same experiment would measure siblings. With 22 instances, the interval on a 9-point difference is wide.
- **Random splits of generated siblings overstate generalisation: INFERENCE** from the construction, shown here at toy scale. It is not a published measurement on a production model.
- **Course measurement (free CPU, 2026-10-07):** toy synthesis kept 76 of 81 candidates (5 broke no test). Retrieval solved 0.98 of held-out tasks under instance splits, 0.89 under family splits and 0.00 under function and repository splits. Search solved 0.77–0.91 under every split. On the SWE-smith shard (3,696 instances, 14 repositories), an instance-level split left 98.9% of held-out patches touching a training file and 99.2% sharing a failing test with training; a family-level split 99.8% and 100%; a repository-level split 0.0% and 0.0%.

## Lab

**Folder:** [`labs/module-16/lesson-02/`](../../labs/module-16/) · **Time:** about 45 minutes · **Pass check:** `pytest labs/module-16/lesson-02` passes and `swe_lab.py` prints parts A and B; your write-up names the split unit for each claim in the table above and the evidence from your run.

### Experiment contract

- **Question:** for generated software-engineering tasks, how much does the split unit change the held-out success of a solver that memorises, compared with one that does not? Decision informed: the split rule for the module project's environment pack.
- **Hypothesis:** retrieval's held-out success is high under instance and family splits and near zero under function and repository splits; search's success does not depend on the split beyond which tasks are held out. Status: follows from sibling structure; the size is a measurement.
- **Baseline:** the instance-level split.
- **Changed variable:** the split unit. **Controlled:** the synthesised task set, the held-out fraction (0.25 of groups), split seeds 0–2, the solvers and their test counts.
- **Comparison axis:** the same tasks and solvers for every split.
- **Budget:** free CPU, measured about 11 s plus a 4.1 MB download.
- **Metrics and decision rule:** held-out success per split and seed, with the sibling leak. A split unit is acceptable for a claim if retrieval's success under it is at most 0.05 above zero while search stays within its seed range under the instance split.
- **Correctness checks:** your TODO tests; `test_agents.py`; every synthesised task fails at least one test of the reference.
- **Fallback evidence:** SWE-smith section 4.1, labelled as published.
- **Limits:** toy repositories with 14 functions that produced tasks; `shift_const` dominates the families (57 of 76), so a family split with 5 families always holds out the same family; two hand-written solvers, not a trained model.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | CPU | `swe_lab.py --shards all` (every SWE-smith shard at the pinned revision; PROJECTED under a minute plus the download). Training an agent on SWE-smith tasks needs the per-repository images and is not part of this lab |
| Free GPU | not needed | — |
| Free CPU | laptop; measured 11 s on the build laptop | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` (`split_groups`, `sibling_leak`, `patch_files`, `instance_keys`) and run `pytest labs/module-16/lesson-02`.
2. **Run** `python labs/module-16/lesson-02/swe_lab.py`. For each split, compare retrieval's success with the sibling leak: they should nearly coincide. Explain why.
3. **Find the search solver's failures** under any split (print the tasks it does not solve). Some fail because the inverse of the modification is not a modification of any family. Others pass the 4 visible tests and fail the hidden 8: a small instance of lesson 16.1's visible-pair problem, inside a solver.
4. **Write up** half a page: the split unit for each claim in the table, your measured numbers, and one way the same upstream code could still cross a repository split (vendored copies and forks, lesson 10.6).

<details>
<summary>Hint for TODO 1</summary>

Sort first (`sorted(set(keys))`), then shuffle: sets have no fixed order across runs, and the split must be reproducible from the seed.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (Windows 11, Python 3.12.13): 11 s in all.

| Split by | held-out tasks (mean) | sibling leak | retrieval success | search success |
|---|---|---|---|---|
| instance | 19.0 | 0.98 | 0.98 [1.00, 1.00, 0.95] | 0.77 [0.84, 0.68, 0.79] |
| function | 21.7 | 0.00 | 0.00 | 0.84 [0.81, 0.86, 0.85] |
| repository | 11.0 | 0.00 | 0.00 | 0.91 [0.82, 1.00, 0.92] |
| family | 57.0 | 0.89 | 0.89 | 0.81 |

Retrieval's success equals the sibling leak, as it should: it solves exactly the held-out tasks with a training sibling. Search moves between 0.77 and 0.91 with *which* tasks are held out (the date functions' constant shifts are easy to invert), not with leakage. Read naively, the instance split says the memorising solver (0.98) beats the searching one (0.77). The function and repository splits say the opposite.

SWE-smith shard (`train-00002-of-00011`, 3,696 instances, 14 repositories): instance split, 924 held out, 0.989 touch a training file, 0.992 share a failing test, 14 repositories on both sides; family split, 453 held out, 0.998 and 1.000, 9 repositories on both sides; repository split, 521 held out, 0.000 and 0.000.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-16/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-16/lesson-02`.

</details>

## Common mistakes

- **Splitting generated tasks by instance id** and reporting the result as generalisation to new code.
- **Splitting by task family alone.** The same function's other families stay in training.
- **Forgetting the evaluation benchmark's repositories.** Remove them before generating, as SWE-smith does, or the "held-out" benchmark is in the training data.
- **Keeping candidates that break no test.** Their reward is earned by the unmodified code.
- **Comparing within-repository results without a time cutoff.** SWE-smith's specialisation result used tasks from before 2022 and evaluation issues from after.

## References

- J. Yang et al., *SWE-smith: Scaling Data for Software Engineering Agents*, 2025, sections 2.1, 2.2, 4.1. https://arxiv.org/abs/2504.21798
- J. Pan et al., *Training Software Engineering Agents and Verifiers with SWE-Gym*, 2024 (ICML 2025), section 3. https://arxiv.org/abs/2412.21139
- C. E. Jimenez et al., *SWE-bench*, 2023, section 2. https://arxiv.org/abs/2310.06770
- Dataset: https://huggingface.co/datasets/SWE-bench/SWE-smith (revision `ea6d7173829c7ec8fa16c22055699ff2e9188091`).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[16.3 · Multi-turn agentic RL](lesson-03.md)
