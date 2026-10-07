# Module 16 labs — How do we train agents without fooling ourselves?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the script the lesson runs. The shared code is
`labs/common/frontierlab/agents/`, with tests in `labs/common/tests/test_agents.py`. It reuses Module 12's policy,
log-probabilities, advantages and loss (`frontierlab.posttrain`) and never edits them.

**Scope.** These labs ship no exploit code and nothing that manipulates, bypasses or fakes a test harness, a test
report or an exit status. The loopholes they study are rewards misspecified by design and harmless: a format-only
reward, a length reward, and a verifier that checks only the visible input/output pairs (which a lookup table
satisfies). Published reward hacks are covered as reading in lesson 16.4.

| Module | What it does |
|---|---|
| `agents/env.py` | the environment interface (task, tools, state, verifier, reset), `run_episode`, the reset and determinism checks |
| `agents/sandbox.py` | runs submitted Python in a child process: temp directory, timeout, output cap, network and process guard. **Not a security boundary** (see its docstring) |
| `agents/codeenv.py` | six toy coding specs, the `visible`, `hidden` and `robust` verifiers, property checks, the candidate registry (`CANDIDATES`, `register`) and `verifier_report`; `LeakyCodeEnv` for the reset check |
| `agents/swetasks.py` | SWE-smith-style task synthesis on four toy repositories (five AST modifications, validated by a breaking test), group splits, a retrieval and a search solver, SWE-smith metadata overlap |
| `agents/dsl.py` | the program world: functions, the program grammar (rules and tables), the rewards (`format`, `length`, `visible`, `hidden`, `randomised`, `property`, `gold`, `random`) |
| `agents/turns.py` | multi-turn rollouts with inserted observations, action and observation masks, turn indexes; the probe task; context cost |
| `agents/agentrl.py` | warm starts and the agent RL loop (outcome or turn credit, loss on actions or the masking bug, any reward, inoculation marker, traces); `python -m frontierlab.agents.agentrl` |
| `agents/monitor.py` | detectors (divergence, output shift, length), the control comparison, `audit_run`, `looks_hardcoded`, `validate_traces` |
| `agents/hf_agent.py` | the main path: the same tasks on Qwen3-1.7B-Base in few-shot text form, multi-turn sampling that keeps the sampled token ids, `--smoke` on CPU |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_agents.py               # the shared Module 16 code (no downloads, about 1-2 minutes)
pytest labs/module-16/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-16             # all reference solutions and the project's pack checks
python -m frontierlab.agents.hf_agent --smoke --task probe --steps 2 --run runs/m16/hf-smoke   # main-path code, CPU
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every script
writes under `runs/m16/` (gitignored), skips finished runs and resumes interrupted ones. The warm starts
(`runs/m16/sft-single`, `runs/m16/sft-probe`, `runs/m16/sft-probe-fsplit`) are trained by the first script that
needs them (20–30 s each).

## What each folder contains

| Folder | Lesson | Script | What it does |
|---|---|---|---|
| `lesson-01/` | 16.1 Environments and verifiers | `verifier_lab.py` | every verifier on every registered candidate (sandboxed); hidden-test sweep against $(1-p)^n$; reset and determinism checks; the reward matrix of the program world |
| `lesson-02/` | 16.2 Software-engineering tasks | `swe_lab.py` | toy SWE-smith-style synthesis; four split units × 3 seeds with a retrieval and a search solver; file and failing-test overlap on one SWE-smith shard (4.1 MB download) |
| `lesson-03/` | 16.3 Multi-turn agentic RL | `multiturn_lab.py` | mask and credit checks on real episodes; context cost of full and windowed histories; 4 arms (baseline, masking bug, bonus with outcome or turn credit) × 2 seeds |
| `lesson-04/` | 16.4 Reward hacking | `hacking_lab.py`, `traces/` | 6 reward arms (3 misspecified, 2 mitigated, 1 control) × 2 seeds, optional marker arm; pre-stated detectors and verdicts; analysis of the provided, validated traces of a run that found the loophole |
| `lesson-05/` | 16.5 Computer-use agents (extension) | `cua_eval_lab.py` | evaluation choices on **simulated** results: intervals, the infeasible-task rule, step budget, environment errors, paired comparison |
| `project/` | Module project | `run_project.py`, `pack.py`, `test_pack.py`, `buggy_pack.py` | the environment pack with repository-level splits, the reward-misspecification audit, the RL result with a control; the planted-bug pack |

`lesson-04/traces/visible-s0.jsonl` (112 records, 28 KB) and its `manifest.json` (SHA-256, run configuration, warm
start, hardware, date) were produced on CPU by `hacking_lab.py --make-traces`. They are analysis material, not a
reproduction of any published result. `validate_traces` re-scores every record.

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 16 pilot, and every main-path figure
in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-07 on a 16-thread Windows 11
laptop (Python 3.12.13, torch 2.14.1+cpu, transformers 5.18.0, 8 threads per script) **with another module's jobs
running at the same time**.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 16.1 verifiers | CPU only (optionally the verifiers inside a container without network) | not needed | `verifier_lab.py` 71 s |
| 16.2 SWE tasks | CPU only; `--shards all` | not needed | `swe_lab.py` 11 s plus the download |
| 16.3 multi-turn | 5–9 GPU-hours (8 runs × 150 steps) | Qwen3-0.6B-Base, two arms | `multiturn_lab.py` 3.7 min |
| 16.4 reward hacking | 4–6 GPU-hours (12 runs × 150 steps) | Qwen3-0.6B-Base, three arms, one seed | `hacking_lab.py` 4.6 min (7 arms) |
| 16.5 computer use | VMs for OSWorld; not part of the pilot | not needed | `cua_eval_lab.py` 1 s |
| Project | 2–4 GPU-hours (4 runs × 150 steps) | Qwen3-0.6B-Base | `run_project.py` see the project page |

Notes:

- Main-path downloads: Qwen3-1.7B-Base (3.4 GB) at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, Qwen3-0.6B-Base
  (1.2 GB) at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; SWE-smith at `ea6d7173829c7ec8fa16c22055699ff2e9188091`.
- The sandbox stops accidents (loops, huge outputs, accidental network use), not a determined program. The labs
  run only course code and its mutations, and toy programs of a 25-character grammar that are parsed, never executed.
  Run code written by a capable model in a container or VM without network.
- Run `python -m frontierlab.agents.hf_agent --smoke ...` (CPU, seconds) before any GPU session to check the install.
