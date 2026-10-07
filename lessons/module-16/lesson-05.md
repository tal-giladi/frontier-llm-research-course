---
id: "16.5"
module: 16
minutes: 30
practice_minutes: 40
prerequisites: ["16.1", "16.3", "01.4"]
objectives:
  - Describe how OSWorld builds and scores computer-use tasks (initial-state setup, execution-based checkers of the final state, infeasible tasks) and why OSWorld-Verified had to repair it.
  - Summarise what UI-TARS and UI-TARS-2 changed in the agent (native screenshot-to-action model, reflective online traces, multi-turn RL) and which reported numbers depend on the step budget.
  - Report a computer-use success rate with a per-task and an application-clustered interval, and say which one a claim needs.
  - Show how the infeasible-task rule, the step budget and the handling of environment errors change a reported score and a ranking.
volatility: implementation
sources:
  - title: "Xie et al., OSWorld (sections 2.1 reward, 2.2 task setup and evaluation, 2.4 actions, 3.2 infeasible tasks, 4.1 15-step limit; 369 tasks; human 72.36%, best model 12.24%)"
    url: https://arxiv.org/abs/2404.07972
  - title: "XLANG Lab, Introducing OSWorld-Verified (2025-07-28)"
    url: https://xlang.ai/blog/osworld-verified
  - title: "Qin et al., UI-TARS: Pioneering Automated GUI Interaction with Native Agents (abstract: OSWorld 24.6 at 50 steps, 22.7 at 15 steps)"
    url: https://arxiv.org/abs/2501.12326
  - title: "UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning (abstract: OSWorld 47.5)"
    url: https://arxiv.org/abs/2509.02544
last_verified: "2026-10-07"
---

# 16.5 · Computer-use agents

Extension: computer-use agents act on a real desktop through screenshots, mouse and keyboard, and their benchmarks are environments in the sense of lesson 16.1, with a task, tools, state, a verifier and a reset. This lesson reads how OSWorld builds and scores such tasks, what UI-TARS and UI-TARS-2 changed on the agent side, and which evaluation choices move the headline number. The lab does not run a GUI agent; it works through those evaluation choices on simulated results.

## Why this matters at a frontier lab

Computer use is where agent evaluation is least stable. Websites change, applications update, anti-bot pages appear, and checkers written for one way of doing a task reject another. OSWorld's maintainers spent "approximately two months" with a team of about ten people fixing "approximately 300 issues" before releasing OSWorld-Verified (company blog, 2025-07-28), and they warn that uncontrolled factors make a benchmark's signal "gradually weaken and become chaotic over time". A research engineer reading a computer-use result has to know the step budget, the version of the benchmark, how infeasible tasks were scored and how environment errors were handled, because each can move the number by more than the gap between two systems.

## The idea

### OSWorld as an environment

OSWorld (PUBLICLY DOCUMENTED, Xie et al. 2024) runs each task in a virtual machine (Ubuntu in the benchmark; Windows and macOS are supported) and maps onto lesson 16.1's five parts:

- **task:** an instruction from real use, 369 tasks in total (abstract);
- **state and reset:** a three-stage setup (start the VM, prepare files, run pre-processing commands) gives 302 distinct initial states (section 2.2.2, Table 3);
- **tools:** screenshots, an accessibility tree or set-of-marks as observations; mouse and keyboard actions plus the terminal actions WAIT, FAIL and DONE (section 2.4);
- **verifier:** execution-based checks of the final state with 134 evaluation functions that read files, application configuration and cloud data (section 2.2.3);
- **budget:** at most 15 steps in the paper's experiments (section 4.1).

The reward is 1 if the final state meets the task's goal, "or if the agent accurately predicts failure for an infeasible task" (section 2.1). 30 tasks (8.1%) are infeasible on purpose: they ask for deprecated or hallucinated features (section 3.2). Humans succeeded on 72.36% and the best model in the paper on 12.24% (abstract).

One consequence of the reward rule, by arithmetic (INFERENCE): an agent that answers FAIL on every task scores $30/369 = 8.1\%$. That is two-thirds of the paper's best model. A score has to be compared with this floor, and a gain made of better FAIL detection has to be reported as such.

### OSWorld-Verified

The July 2025 revision (company blog) repaired the benchmark in four categories: web pages whose structure or URLs had changed, anti-crawling pages and CAPTCHAs, ambiguous instructions with several valid readings, and evaluators so strict that they rejected valid solutions (lesson 16.1's false rejects). It kept the 369 tasks and moved evaluation to AWS, with up to 50 environments in parallel. Results before and after the revision are not comparable without saying which version produced them.

### UI-TARS and UI-TARS-2

UI-TARS (Qin et al., January 2025) is a "native" agent: one model maps screenshots to keyboard and mouse actions, instead of a general model wrapped in hand-written prompts. It combines large-scale GUI screenshot data for perception, a unified action space across platforms, "System-2" reasoning, and iterative training on reflective online traces collected on many virtual machines. It reports 24.6 on OSWorld with 50 steps and 22.7 with 15 steps, against 22.0 and 14.9 for Claude in the same comparison (abstract; PUBLICLY DOCUMENTED, (company claim)). The 15/50 pair is the useful part for this lesson: the step budget is part of the result.

UI-TARS-2 (September 2025) adds a data flywheel, "a stabilized multi-turn RL framework" (lesson 16.3's problems at GUI scale) and a unified sandbox, and reports 47.5 on OSWorld, 50.6 on WindowsAgentArena and 73.3 on AndroidWorld (abstract; (company claim)). The abstract does not say which OSWorld version or step budget produced 47.5. Find it in the report before comparing it with anything.

### What an evaluation of a computer-use agent must state

| Choice | Why it moves the number |
|---|---|
| benchmark version (OSWorld or OSWorld-Verified, date) | repaired checkers and tasks change scores |
| step budget | longer budgets help agents that recover slowly (UI-TARS: 22.7 at 15 steps, 24.6 at 50) |
| observation type | screenshot only, accessibility tree, or set-of-marks |
| infeasible-task rule | FAIL detection alone is worth up to 8.1 points |
| environment errors | failures of the VM, network or website counted against the agent, excluded, or rerun |
| interval | 369 tasks give about ±4 points per task at a 20% success rate, and more once tasks are clustered by application |

## Worked example

**Wilson interval.** 80 of 369 tasks solved: $\hat p = 0.2168$. With $z = 1.96$, the centre is $(\hat p + z^2/2n)/(1 + z^2/n) = (0.2168 + 0.0052)/1.0104 = 0.2197$ and the half-width is $z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}/(1 + z^2/n) = 1.96 \times 0.02162/1.0104 = 0.0419$. The interval is $[0.178, 0.262]$. Two agents 3 points apart cannot be ranked from this alone. A paired comparison over the same tasks is tighter only when their outcomes are correlated task by task (in the lab's simulation they are nearly independent, so pairing barely helps).

**The FAIL floor.** An agent detects 60% of the 30 infeasible tasks and solves 20% of the 339 feasible ones: $0.6 \times 30 + 0.2 \times 339 = 18 + 67.8 = 85.8$ tasks, 23.3%. Another detects 30% and solves the same feasible share: $9 + 67.8 = 76.8$, 20.8%. The 2.5-point lead comes entirely from FAIL detection.

**Budget.** If an agent's successful episodes need a number of steps with mean 14, many need more than 15. Cutting the budget from 50 to 15 then removes a large part of its successes, while a faster agent with mean 9 loses fewer. The ranking can flip between budgets, as the lab shows.

## Shapes and cost

| Item | Size | Notes |
|---|---|---|
| observation | a 1920 × 1080 screenshot per step (section 4.1), optionally an accessibility tree | the context grows with every screenshot, so lesson 16.3's context policies apply |
| actions | mouse and keyboard events, WAIT, FAIL, DONE | |
| one evaluation | 369 tasks × up to 15 or 50 steps | OSWorld-Verified: from "10+ hours" to minutes with 50 parallel environments (company blog) |
| this lab | 369 simulated outcomes per agent, bootstrap | under 2 seconds on a laptop |

## Build it

```python
lab.task_reward(feasible=False, final_action="FAIL", state_ok=False)   # 1.0: OSWorld's infeasible rule
lab.wilson_interval(80, 369)                                            # (0.178, 0.262)
lab.cluster_interval(rewards, apps)                                     # resample applications, not tasks
lab.success_at_budget(steps_needed, 15)
```

The lab's script fills an OSWorld-shaped task list (369 tasks, 30 infeasible, ten application groups with made-up sizes and difficulties) with outcomes of three made-up agents and prints how each evaluation choice changes the score. Every number it prints is labelled simulated.

## What the evidence says

- **Execution-based checking of the final state: ESTABLISHED** for computer-use benchmarks (OSWorld section 2.2.3; PUBLICLY DOCUMENTED).
- **Benchmarks of live software decay and need repair: PUBLICLY DOCUMENTED** (OSWorld-Verified, company blog).
- **Native screenshot-to-action models trained with online traces and multi-turn RL: PROMISING** (UI-TARS, UI-TARS-2; one lab, (company claim) numbers).
- **Step budget and infeasible-task rule change scores and rankings: PUBLICLY DOCUMENTED** for the budget (UI-TARS's two numbers); INFERENCE for the FAIL floor, from OSWorld's reward rule.
- **Course measurement (simulated, 2026-10-07):** on the simulated task list, agent B ranked above agent A at 15 steps (0.244 against 0.214), entirely through infeasible tasks (0.533 against 0.200 there; 0.218 against 0.215 on feasible tasks). A ranked above B at 50 steps on feasible tasks (0.316 against 0.271). The always-FAIL agent scored 0.081. These are properties of the simulation, not of any real agent.

## Lab

**Folder:** [`labs/module-16/lesson-05/`](../../labs/module-16/) · **Time:** about 40 minutes · **Pass check:** `pytest labs/module-16/lesson-05` passes and `cua_eval_lab.py` prints parts 1–5; your write-up is an evaluation protocol of half a page for a computer-use result.

This lab analyses simulated results; it compares no training arms, so it has no experiment contract.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | a machine that can host the OSWorld VMs (or the AWS setup of OSWorld-Verified); not run in this build, and not part of the course pilot | write the protocol below and apply it to a published result, or to your own run if you have one |
| Free GPU | not needed | — |
| Free CPU | laptop; measured 1 s | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` (`wilson_interval`, `task_reward`, `cluster_interval`, `success_at_budget`) and run `pytest labs/module-16/lesson-05`.
2. **Run** `python labs/module-16/lesson-05/cua_eval_lab.py`. For each of parts 1–5, write one sentence on what the choice did to the score or the ranking.
3. **Read** the UI-TARS-2 report for its OSWorld setting (benchmark version, step budget, observation type) and write it next to the abstract's 47.5. If the report does not say, write that.
4. **Write the protocol** (half a page): the benchmark version, budget, observation, infeasible-task rule, environment-error policy, interval method and the FAIL floor you would report with every computer-use number.

<details>
<summary>What the build's run gave (simulated; compare after your write-up)</summary>

Measured 2026-10-07, 1 s. At 15 steps: agent A 0.214 (Wilson [0.175, 0.259], app-clustered [0.164, 0.272]), agent B 0.244 ([0.203, 0.290], [0.191, 0.312]), always-FAIL 0.081 ([0.058, 0.114], [0.045, 0.109]). The app-clustered intervals are wider because applications differ in difficulty and there are only ten of them. On feasible tasks A and B are tied at 15 steps (0.215 against 0.218); B's lead is its infeasible-task detection (0.533 against 0.200). At 50 steps A leads on feasible tasks (0.316 against 0.271), because its successful episodes are longer. Environment errors (5% of tasks): counting them as failures gives 0.214, excluding them 0.224 (n = 353), one rerun 0.222. A − B: −0.030, paired [−0.087, +0.027], unpaired [−0.092, +0.030]. Pairing barely helps here because the simulated agents' outcomes are independent given the application. Real agents fail on many of the same tasks, and then pairing helps more.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-16/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-16/lesson-05`.

</details>

## Common mistakes

- **Comparing numbers across OSWorld versions or step budgets.** State both with every result.
- **Ignoring the FAIL floor.** 8.1% is free under OSWorld's rule; report feasible and infeasible tasks separately.
- **Excluding environment errors after seeing which tasks they hit.** Choose the policy before the run.
- **Per-task intervals for an application-level claim.** Ten applications are ten clusters, not 369 independent draws.
- **Reading a vendor's headline number without its setting.** Find the budget and version in the report.

## References

- T. Xie et al., *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*, 2024, sections 2.1–2.4, 3.2, 4.1. https://arxiv.org/abs/2404.07972
- XLANG Lab, *Introducing OSWorld-Verified*, 2025-07-28. https://xlang.ai/blog/osworld-verified
- Y. Qin et al., *UI-TARS: Pioneering Automated GUI Interaction with Native Agents*, 2025. https://arxiv.org/abs/2501.12326
- H. Wang et al., *UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning*, 2025. https://arxiv.org/abs/2509.02544

## Next

The module project combines lessons 16.1–16.4: [Module 16 project](../../projects/module-16-agent-environments.md). Module 18 runs a chain-of-thought monitor on agent traces like the ones lesson 16.4 provides.
