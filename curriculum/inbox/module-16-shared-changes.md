# Module 16 — proposed changes to shared files (for the main session)

Module 16 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/`, `templates/` or
`PUBLISHING_WARNING.md`. New code: `labs/common/frontierlab/agents/` (`__init__`, `env`, `sandbox`, `codeenv`,
`swetasks`, `dsl`, `turns`, `agentrl`, `monitor`, `hf_agent`), tests `labs/common/tests/test_agents.py`. It imports
`frontierlab.posttrain` (policy, log-probabilities, advantages, KL estimators, losses, SFT helpers, `arms.t_interval`),
`frontierlab.datax.groups.group_split`, `frontierlab.metrics`, `frontierlab.model` and `frontierlab.runcard`, and
edits none of them. Module 16 does not use the Module 12 loop (`posttrain.rl.train`) because that loop hard-codes
arithmetic prompts and single-turn responses; `agents.agentrl` is a separate, short loop built from the same pieces.

## 1. `_sidebar.md`: Module 16 lines

Running numbers 67–71 (fix them if Module 15 ends elsewhere):

```markdown
- **Module 16 — How do we train agents without fooling ourselves?**
  - [67 · Environments and verifiers](lessons/module-16/lesson-01.md)
  - [68 · Software-engineering tasks](lessons/module-16/lesson-02.md)
  - [69 · Multi-turn agentic RL](lessons/module-16/lesson-03.md)
  - [70 · Reward hacking](lessons/module-16/lesson-04.md)
  - [71 · Computer-use agents](lessons/module-16/lesson-05.md)
  - [Module 16 quiz](assessments/module-16-quiz.md)
```

The project page `projects/module-16-agent-environments.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-16.md` (32 terms).

## 3. `references/versions.md`

Add under "Notes on reference implementations":

"Module 16: no new packages. Base models Qwen/Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`
(`max_position_embeddings` 32768, checked 2026-10-07) and Qwen/Qwen3-0.6B-Base @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
(T4). Dataset SWE-bench/SWE-smith @ `ea6d7173829c7ec8fa16c22055699ff2e9188091` (MIT; lesson 16.2 reads
`data/train-00002-of-00011.parquet`, 3,696 instances, 14 repositories; columns instance_id, patch, FAIL_TO_PASS).
verl's agent loop documents `AgentLoopOutput.response_mask` (1 = LLM-generated token, 0 = tool response token),
https://verl.readthedocs.io/en/latest/advance/agent_loop.html, checked 2026-10-07. OSWorld-Verified
(xlang.ai/blog/osworld-verified, 2025-07-28) is the current OSWorld version; lesson 16.5 cites both."

## 4. `PUBLISHING_WARNING.md`: replace the lesson 16.4 part of the first bullet

The current text says the labs reproduce "reward hacking in a planted verifier". Module 16 was written under a
narrower scope; please replace the sub-bullet "Labs reproduce published model-organism results (emergent
misalignment from narrow fine-tuning, reward hacking in a planted verifier) only with benign proxy behaviours..."
with:

"- Labs reproduce published model-organism results only with benign proxy behaviours (Module 18: insecure-looking
  toy code or sycophancy). Lesson 16.4 contains no exploit code and nothing that manipulates, bypasses or fakes a
  test harness, test report or exit status: the published reward hacks (Anthropic's MacDiarmid et al. 2025, OpenAI's
  monitoring paper, METR's June 2025 report, ImpossibleBench) are summarised at the level of the papers, and the
  hands-on runs use only rewards misspecified by design and harmless (format-only, length-based, visible
  input/output pairs only, which a lookup table satisfies) on a 0.3M-parameter policy and a 25-character program
  grammar that is parsed, never executed. Check that this still holds, including for any candidate later added to
  `frontierlab.agents.codeenv.CANDIDATES` through `register()`: only correct, wrong or visible-pair-overfit
  submissions belong there."

## 5. Pilot commands (Module 16; none run in this build)

All on 1× H100 80 GB unless noted; PROJECTED times use the stated assumptions (generate call 1.5–4 s, update 6–10 s
per step) until the pilot measures them. Run the CPU smoke first:
`python -m frontierlab.agents.hf_agent --smoke --task probe --split function --steps 2 --run runs/m16/hf-smoke`.

| Lab | Commands | PROJECTED | What to record |
|---|---|---|---|
| 16.3 | `python labs/module-16/lesson-03/multiturn_lab.py --variant main --print` (8 runs of `hf_agent --task probe`, 150 steps) | 0.5–0.9 GPU-h per run, 5–9 GPU-h total | s/step split into generate and update; peak memory; `sampler_gap`; `obs_tokens_in_loss` (0 except the all-tokens arm); queries per episode; held-out gold at each eval |
| 16.4 | `python labs/module-16/lesson-04/hacking_lab.py --variant main --print` (12 runs of `hf_agent --task single`, 150 steps) | 0.3–0.5 GPU-h per run, 4–6 GPU-h total | whether `visible` finds tables (frac_table over steps) — **a hypothesis**: a base model may never sample `T...`; held-out gold vs the random control; the step at which tables pass 0.5 |
| Project | `python labs/module-16/project/run_project.py --variant main --print` (gold and random-control probe arms on the function-level split, 2 seeds, 150 steps) | 0.5–0.9 GPU-h per run, 2–4 GPU-h total | held-out-function gold per seed; flags from `frontierlab.agents.monitor.audit_run` |
| T4 | the same scripts with `--variant t4 --print` (Qwen3-0.6B-Base, 8 prompts) | 1.5–3 T4-hours per run | as above; say which arms were run |

The pilot should also save 16–32 evaluation traces per evaluation of the main-path `visible` arm (add `traces` to
`hf_agent` or write them from its `evaluate`), because Module 18 runs a monitor on Module 16 traces.

## 6. Plan section 14.1 claim checks to record

- V: MacDiarmid et al. 2511.18397 — abstract (mitigations i–iii); section 2 (the three hacks, named in the lesson only
  at paper level); section 3.1.2 (code sabotage 12%); section 4 (preference-model reward and hack classifier prevent
  hacking); section 4.2 (inoculation: 75–90% less misaligned generalisation, hacking still learned).
- V: Baker et al. 2503.11926 — abstract; Table 1 (CoT monitor recall 95% vs 60% action-only on systemic hacks; GPT-4o
  monitor); section 3 (obfuscation under strong optimisation).
- V: METR 2025-06-05 — o3 30.4% of runs on the RE-Bench tasks reported, 0.7% on HCAST; Optimize LLM Foundry, 20 runs per
  prompt: 80 / 95 / 80 / 70%.
- V: ImpossibleBench 2510.20270 — GPT-5 54.0% on Conflicting-SWEbench (section 4); hidden tests near zero, read-only
  middle ground (5.2); abort option 54% → 9% (5.3); 103 LiveCodeBench and 349 SWE-bench Verified items per variant.
- V: SWE-smith 2504.21798 — 50k instances, 128 repositories, 40.2% pass@1 (abstract); strategies and "break one or more
  existing, passing tests" (2.1); 295 GB vs 50–150 TB (2.2); repository scaling and the SymPy specialisation (33.3 → 42.4
  on 22 post-2022 SymPy instances; 4.1).
- V: SWE-Gym 2412.21139 — 2,438 instances, 11 repositories separate from SWE-bench, SWE-Gym Lite 230 (section 3).
- V: OSWorld 2404.07972 — 369 tasks, 72.36% / 12.24% (abstract), reward rule incl. infeasible (2.1), 302 initial
  states and 134 evaluation functions (Table 3, 2.2), WAIT/FAIL/DONE (2.4), 30 infeasible (3.2), 15-step limit (4.1).
- V: UI-TARS 2501.12326 — OSWorld 24.6 (50 steps), 22.7 (15 steps), Claude 22.0 / 14.9; AndroidWorld 46.6.
- V: UI-TARS-2 2509.02544 — OSWorld 47.5, WindowsAgentArena 50.6, AndroidWorld 73.3, Online-Mind2Web 88.2 (abstract;
  the OSWorld setting is not in the abstract).
- V: OSWorld-Verified blog (2025-07-28) — about 300 issues, about two months, about ten people, AWS with up to 50
  parallel environments.
- V: Search-R1 2503.09516 (retrieved token masking, abstract); RAGEN 2504.20073 (StarPO, Echo Trap, StarPO-S);
  Wei et al. 2505.11821 (dense per-turn rewards outperform); MEM1 2506.15841 (3.5× performance, 3.7× memory vs
  Qwen2.5-14B-Instruct on 16-objective multi-hop QA); Wichers et al. 2510.05024 (inoculation prompting, four settings).
- V: SWE-bench Verified (OpenAI, 2024-08-13): 1,699 samples, 93 developers, 500 kept; criteria include FAIL_TO_PASS tests
  that filter out valid solutions (from search results of the page; the page itself was not fetched directly).

## 7. Optional, for a later revision

`frontierlab/posttrain/rl.py` could accept `hooks["make_prompts"]` and `hooks["score"]` so that Module 16's
single-turn program world runs through the Module 12 loop instead of `agents.agentrl`. Not needed now.
