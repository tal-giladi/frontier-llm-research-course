# Module 16 prompts: the blocked one and the replacement

Planning file, not imported.

The safety check did not block the prompt itself. It stopped the agent's output partway through, while the agent
was writing `labs/common/frontierlab/agents/codeenv.py`. That file held a dictionary `EXPLOITS` of runnable functions,
each writing files into a toy coding task so that the course's test runner reported success without the task being
solved:

- `exit_early`: `sys.exit(0)` before the asserts.
- `always_equal`: an object whose `__eq__` always returns True.
- `conftest`: a `conftest.py` that rewrites test reports to "passed". To make this one work, the course's own runner
  imported `conftest.py`.
- `edit_tests`: rewrites the test file.
- `read_tests`: reads the expected values out of the test file at runtime.
- `print_pass`: prints a fake "N passed, 0 failed" summary.
- `hardcode_visible`: special-cases the visible test inputs.

The agent ran the catalogue against three verifiers. The exit-code verifier accepted every exploit, the log-parsing
verifier accepted all but `exit_early`, and the hardened verifier accepted none. Those functions were then deleted in
the cleanup pass, so they are no longer in the repo. The rest of `agents/` (the sandbox, the environment interface,
the six tasks, the verifiers) is kept.

## 1. Original prompt (first attempt; the check fired while the agent was writing 16.1 and 16.4 code)

> Write Module 16 ("How do we train agents without fooling ourselves?") of the course at C:\Users\TalGiladi\OneDrive\repos\course-creator\frontier-llm-research-course.
>
> Read curriculum/AGENT-BRIEF.md fully first and follow it exactly (it lists what else to read: the binding guide C:\Users\TalGiladi\OneDrive\repos\tals-academy\docs\new-course-instructions.md, plan.md incl. the Module 16 entry at line ~279, sections 6–7, 12.1 feasibility, 14.1 claim checks, the Module 1 exemplar, shared code, templates) and PUBLISHING_WARNING.md (lesson 16.4 is covered by it; follow what it says). Read the finished Modules 12–14 (lessons and frontierlab/posttrain, pipeline, rlscale, evals/suite_v2) since you build on their RL loop, and Module 10's lesson 10.6 / datax code for the SWE-smith shard and repository-level splits. Use Module 10–14 conventions: experiment contracts, evidence labels, PROJECTED (pending pilot) on unmeasured main-path figures, three variants per lab (CPU / Colab T4 / main path rented GPU). Stage D base model: Qwen/Qwen3-1.7B-Base revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 (main path), Qwen3-0.6B-Base for T4, tiny models/toy environments on CPU (see references/versions.md; vLLM 0.30.0 and verl 0.9.1 pinned there). Every RL result includes a random- or format-reward control arm.
>
> Lessons: 16.1 Environments and verifiers (task, tools, state, verifier, reset; verifier tests; reward that resists cheap exploits), 16.2 Software-engineering tasks (task synthesis: SWE-smith, SWE-Gym; split by repository or task family, showing random splits of related generated tasks overstate generalisation), 16.3 Multi-turn agentic RL (credit over turns, long trajectories, context management), 16.4 Reward hacking (catalogue, detection, mitigation; Anthropic's report that production reward hacking generalised to broader misalignment — verify and cite precisely; whether a student run finds a planted exploit is a hypothesis, and validated traces of a run that did are provided for analysis — generate those traces in-course from a CPU run that does find the exploit, and store them small in the repo or note they go to Hugging Face), 16.5 extension Computer-use agents (OSWorld, UI-TARS; mainly reading and evaluation design). Project: environment pack (3 environments, repository-level split), a hack audit and an RL result. Sandboxed code execution in environments must be safe on the learner's machine (subprocess with timeouts, temp dirs, no network) and state its limits. Code from scratch; frameworks only as mappings. Verify every paper URL and claim from primary sources (web search allowed).
>
> You own: [module-16 files, frontierlab/agents, tests/test_agents.py] ... Do not commit. Report briefly when done.

## 2. Resume message (the check fired again as soon as this arrived)

> Resume Module 16 with the scope you suggested (main session decision): in 16.1 and 16.4, describe the published hacks (AlwaysEqual, early exit, conftest report rewriting, editing/reading tests, fake summaries) in prose at the paper's level with citations, and do not ship runnable code for them. Remove from codeenv.py the exploit functions that subvert a test harness (exit_early, always_equal, conftest, edit_tests, read_tests, print_pass) and the runner's conftest loading. Runnable verifier tests and the planted exploit use only abstract toy-level loopholes that are properties of the course's own toy verifiers (e.g. hardcoding visible test cases against a verifier that only runs visible tests, a format-only or length-based reward, an output-only checker that can be satisfied without solving the task), answered by the hardened verifier with hidden tests, randomised inputs and property checks. ... If the classifier objects again, stop and report.

## 3. Replacement prompt (running now)

> Scope rule (binding): this module ships no exploit code of any kind and no code that manipulates, bypasses or fakes a test harness, test runner, test report or exit status. Published reward hacks (Anthropic's MacDiarmid et al. 2025, arXiv 2511.18397; OpenAI's monitoring paper; METR's June 2025 report; ImpossibleBench) are covered only as reading, summarised at the level of the papers with citations. The hands-on reward-hacking work uses only rewards that are misspecified by design and harmless: a format-only reward, a length-based reward, and a verifier that checks only a few visible input/output pairs (so the policy learns to overfit to them), with a tiny model on toy tasks. Detection is by held-out checks, monitoring of training metrics and outputs, and comparison with a control; mitigation by reward design, held-out tests, randomised inputs, property checks and inoculation as reported. If anything you are asked to write would need more than this, describe it in prose instead. If a safety system objects, stop and report.
>
> Lessons: 16.1 Environments and verifiers (... verifier tests: false accepts and false rejects on correct, wrong and visible-pair-overfitting solutions), 16.2 Software-engineering tasks (unchanged), 16.3 Multi-turn agentic RL (unchanged, plus observation-token masking), 16.4 Reward hacking (reading on the published cases, then a toy RL run against the misspecified rewards above with detection and mitigation; validated traces from a CPU run that found the loophole, kept small in the repo), 16.5 extension Computer-use agents (unchanged). Project: environment pack (3 environments, repository-level split), a reward-misspecification audit and an RL result.
>
> (The rest is the same as the original prompt: reading list, conventions, base models, ownership, checks.)

## What differs for the learner

Same in both: every lesson, every reading, the sandbox, environments, verifier testing, splits, multi-turn RL, the
detection, monitoring and mitigation material, and an RL run in which a policy finds a loophole and the learner
catches and closes it.

Lost with the replacement: the learner never runs the six published harness exploits against a weak and a hardened
verifier. Under the replacement they read about those six exploits; the runnable loopholes are toy ones (visible-pair
overfitting, format and length rewards).

If you restore it with another service, the missing piece is small and self-contained: the `EXPLOITS` functions
above in `labs/common/frontierlab/agents/codeenv.py`, the `candidates` / `verifier_report` helpers that run them
against every verifier, the runner's `conftest.py` loading, and the 16.1 / 16.4 lab steps and text that use them.
Everything else in Module 16 does not depend on it.
