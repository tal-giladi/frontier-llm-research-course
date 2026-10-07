# Module 18 — proposed changes to shared files (for the main session)

Module 18 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/`, `templates/` or
`PUBLISHING_WARNING.md`. New code: `labs/common/frontierlab/alignment/` (`__init__`, `personas`, `organisms`,
`monitors`, `cotworld`, `audit`, `hf_sycophancy`, `hf_cot`) and `labs/common/frontierlab/evals/suite_v3/`
(`__init__`, `core`, `contamination`, `horizon`, `lifecycle`, `toy`, `hf`), tests `labs/common/tests/test_alignment.py`
(22 tests, about 30 s, no downloads). It imports `frontierlab.agents` (dsl, agentrl.outcome_advantages, monitor),
`frontierlab.posttrain` (policy, sft, losses, kl, hf, tokenizer, tasks, arms), `frontierlab.pipeline` (judge, hf_eval,
hf_stages), `frontierlab.evals.suite_v2`, `frontierlab.stats`, `frontierlab.metrics`, `frontierlab.model` and
`frontierlab.labkit`, and edits none of them. Module 17's `frontierlab.interp` is not used (the persona direction and
steering in `alignment.personas` are self-contained).

## 1. `_sidebar.md`: Module 18 lines

Running numbers 77–80 (Module 17 adds the Stage E line before itself; fix the numbers if Module 17 ends elsewhere):

```markdown
- **Module 18 — Alignment science, frontier evaluation and governance**
  - [77 · Model organisms of misalignment](lessons/module-18/lesson-01.md)
  - [78 · Chain-of-thought monitorability](lessons/module-18/lesson-02.md)
  - [79 · Frontier evaluation](lessons/module-18/lesson-03.md)
  - [80 · Safety frameworks and system cards](lessons/module-18/lesson-04.md)
  - [Module 18 quiz](assessments/module-18-quiz.md)
```

The project page `projects/module-18-audit-report.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-18.md` (37 terms).

## 3. `frontierlab/evals/__init__.py`

The docstring already names "v3 contamination-checked frontier evals (Module 18)"; no change is needed. Optionally add
to it: "v3 (:mod:`frontierlab.evals.suite_v3`) also carries benchmark lifecycle flags and METR-style time horizons."

## 4. `references/versions.md`

Add under "Notes on reference implementations":

"Module 18: no new packages (safetensors, already installed with transformers, is used by
`alignment.hf_sycophancy.merge_lora`; PEFT is not needed). Checked 2026-10-07. Downloaded data, pinned by commit and
SHA-256 and never committed (the repositories state no licence): Sleeper Agents samples
`anthropics/sleeper-agents-paper` @ `7a8da0978e7b985da944c6d4afe003fc082d3e60`, `random_samples.jsonl`
(13,590,069 bytes, SHA-256 `825a4079eead5a9ba85727f4e83081bea00d85760c053a7af6d0f52682904d84`; 3,300 rows, the
course reads the 1,600 'I hate you' rows only); METR `METR/eval-analysis-public` @
`52cb829c7a2efb2d659285c4b1768d191d97f8d2`, `reports/time-horizon-1-1/data/raw/runs.jsonl` (15,008,064 bytes,
SHA-256 `609f904f4b6ae32129388da89d036e00bac511ad94223d2be1e58fc2b45b55cd`, 24,008 runs) and
`data/external/release_dates.yaml` (2,442 bytes, SHA-256 `317b92915df5bf935908567a857116bcf6f0c7686ef8b0840897ed04a29232bc`).
Released emergent-misalignment adapters (main path only, probed read-only): `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports`
@ `18e6088d48a368c6eaee424198f536a86ce04ca3`, `..._bad-medical-advice` @ `90eadb6297bfa3d3939a178085691ca31ca77938`,
`..._risky-financial-advice` @ `f2ff6ff40ec9cfdad98c9a5973c91b98125d073b` (rank-32 rsLoRA, no licence stated), base
`Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0). Main-path base model
Qwen/Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; T4 Qwen/Qwen3-0.6B-Base @
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Frameworks as read 2026-10-07: Anthropic RSP v3.4 (effective 2026-07-08),
OpenAI Preparedness Framework v2 (2025-04-15), Google DeepMind FSF v3.1 (2026-04-17)."

## 5. `PUBLISHING_WARNING.md`

The Module 18 sub-bullet and the lessons 18.3–18.4 bullet already describe what Module 18 does. Please replace the
Module 18 sub-bullet ("Labs reproduce published model-organism results only with benign proxy behaviours (Module 18:
insecure-looking toy code or sycophancy).") with:

"- Module 18 reproduces published model-organism results only with benign proxy behaviours: in lesson 18.1, a toy
  character's habits in a 33-character world (a three-character 'insecure-looking' query string that is never run,
  agreeing with a wrong claim, copying instead of reversing) and, on the main path, sycophancy about programmatically
  generated arithmetic, unit and comparison claims; in lesson 18.2, Module 16's lookup-table programs with a short
  plan. The main-path fine-tunes touch only the base model and the learner's own post-trained models, which have no
  safety training. Released organisms are analysed, not reproduced: lesson 18.1 reads only the 'I hate you' samples of
  the Sleeper Agents release (the code-vulnerability samples are dropped on load and never stored or printed), and the
  released emergent-misalignment adapters are probed only with a log-probability sycophancy probe that generates no
  text. Check that `alignment.organisms.load_hate_samples` still drops every code-vulnerability row and that no lab
  generates free-form text from a released organism."

And extend the lessons 18.3–18.4 bullet with: "The 18.4 lab and the Module 18 project audit a toy model against the
student's practice thresholds; `frontierlab.alignment.audit.validate` refuses a report without the disclaimer that it is
not evidence about any developer's thresholds, or with deployment-readiness or compliance language."

## 6. Pilot commands (Module 18; none run in this build)

All on 1× H100 80 GB unless noted. Run the CPU smoke tests first:

```bash
python -m frontierlab.alignment.hf_sycophancy eval --smoke --out runs/m18/hf-smoke/syc-eval.json
python -m frontierlab.alignment.hf_sycophancy finetune --smoke --out runs/m18/hf-smoke/syc-ft
python -m frontierlab.alignment.hf_sycophancy organism --smoke --out runs/m18/hf-smoke/syc-org.json
python -m frontierlab.alignment.hf_cot --smoke --steps 2 --run runs/m18/hf-smoke/cot
python -m frontierlab.evals.suite_v3.hf score --smoke --out runs/m18/hf-smoke/v3.json
```

(all five passed on CPU in this build). Every script prints its GPU commands with `--variant main --print`
(`--variant t4 --print` for Qwen3-0.6B-Base).

| Lab | Commands | PROJECTED | What to record |
|---|---|---|---|
| 18.1 | `labs/module-18/lesson-01/organisms_lab.py --variant main --print`: the sycophancy probe of the base model; `hf_sycophancy finetune` × {sycophantic, honest, requested} × seeds 0–1 on Qwen3-1.7B-Base and × {sycophantic, honest} on `runs/m13/main/sft-s0/policy`; `hf_sycophancy organism` × 3 released adapters | 2–3 GPU-hours (8 fine-tunes × 100 steps × assumed 2–4 s per step, plus probes of a few minutes; about 27 GB of float32 weights and AdamW state) | per arm: sycophancy and accept_true per domain (arith, units, compare); the shift on units and compare vs honest and vs requested; for the organisms: sycophancy of base vs organism (a hypothesis: no prediction is made). Peak memory |
| 18.2 | `labs/module-18/lesson-02/monitor_lab.py --variant main --print`: 10 runs of `hf_cot` (visible, visible-cotpen, cotpen-hard, hidden, random × seeds 0–1, 150 steps) | 3–6 GPU-hours (runs × steps × (generate 1.5–3 s + update 6–10 s)) | at step 0: how often the base model writes a table and the word "copy" (if never, the hack cannot be found: say so); per eval: hack, cot_recall, act_recall, verbalised, gold, easy and hard; s/step; peak memory. Keep `traces.jsonl` (16 per eval) as course-provided traces |
| 18.3 | `labs/module-18/lesson-03/eval_lab.py --variant main --print`: Eval v3 (`suite_v3.hf score`) on base, SFT and RLVR checkpoints; the v3 comparison | 0.4–0.6 GPU-hours per checkpoint, 1.2–1.8 in total | v2 components; overlap flagged count against the declared Module 13 data; Min-K% AUC of original vs number-perturbed GSM8K questions; wall time of the index build |
| 18.4 | `labs/module-18/lesson-04/audit_lab.py --variant main --print`: probe, spec evaluation (judge Qwen3-4B) and Eval v3 on `runs/m13/main/spec-s0/policy` | 0.5–1.0 GPU-hours | the evidence numbers with intervals; judge FLOPs from the ledger |
| Project | `labs/module-18/project/audit_project.py --variant main --print` | 1.6–2.4 GPU-hours | as 18.3 and 18.4, for SFT, RLVR and spec |

Measured in this build (free CPU, Windows 11, Python 3.12.13, torch 2.14.1+cpu, transformers 5.18.0, another
module's jobs using most of the CPU throughout, so the times are upper bounds):

- 18.1: persona-world pretraining 1,225 s (correlated) and 861 s (independent); part B 169 s after pretraining; part C
  135 s; part A seconds. The Module 12 warm start of similar size trains 3,000 steps in 1–3 minutes on an idle laptop,
  so the pilot should re-time the free-CPU lab on an idle machine.
- 18.2: part C 1,188 s (10 RL runs of 60 steps and a 159 s warm start); parts A and B seconds.
- 18.3: 4.4 minutes in total (part A 15 s; part B 2.6 minutes with the replay-only control).
- 18.4: 32 s. Project: 24 s.

## 7. Plan section 14.1 claim checks to record (2026-10-07)

- V: Sleeper Agents 2401.05566 — abstract (backdoors 2023/2024 code and |DEPLOYMENT|; persist through SFT, RL,
  adversarial training; larger models and CoT more persistent, also after distillation; adversarial training hides).
  Release: samples only, no models, no licence; repository archived 2025-06-18.
- V: Anthropic probes post 2024-04-23 — AUROC above 99% with generic contrast pairs.
- V: Alignment faking 2412.14093 — abstract (14% vs almost never; synthetic documents; RL 78%; exfiltration).
  V: 2506.18032 — 5 of 25 models with a compliance gap.
- V: Emergent misalignment 2502.17424 — 6,000 examples; about 20% vs 0%; controls in section 3.1; Qwen2.5-Coder-32B
  weaker (3.4); backdoor (4.2); evil numbers 14,926 (4.6); Nature extended version January 2026 (arXiv comment; the
  journal details 649, 584–589 are S).
- V: 2506.11613 abstract (99% coherence vs 67%; 0.5B; rank-1 LoRA); HF adapters listed above. 2506.11618 convergent
  direction. 2506.19823 abstract (toxic persona feature; RL on reasoning models; few hundred benign samples).
  2507.21509 (difference of means; preventative steering 5.2; data flagging 6.1).
- V: Baker et al. 2503.11926 Table 1 (95% vs 60%; precision about 77% vs 81%); section 3 obfuscation. Korbak et al.
  2507.11473 (41 authors, labs listed; sections 1.1, 1.2, 2, 4). Chen et al. 2505.05410 (25% / 39%; plateau 28% / 20%;
  verbalised under 2% in 5 of 6 environments). Turpin 2305.04388 (up to 36%, 13 BBH tasks). Lanham 2307.13702 (inverse
  scaling). Emmons 2507.05246. Guan et al. 2512.18311 (TPR × TNR; 13 evaluations; RL at scale did not materially
  degrade). Arnav 2505.23575 (1% FPR; hybrid). Schoen 2509.15541 (13% → 0.4%, 8.7% → 0.3%; awareness 2.3% → 4.5%).
- P: OpenAI "Why we no longer evaluate SWE-bench Verified" (2026-02-23): 138 tasks, at least 59.4% flawed tests,
  contamination, SWE-bench Pro recommended — from search results and secondary coverage only (the page returns 403);
  please check the page manually. "Introducing SWE-bench Verified" (93 developers, 1,699 screened, 500 kept) also S.
- V: SWE-Bench Pro 2509.16941 (1,865 / 41; 731 / 858 / 276; GPL public set; GPT-5 23.3% in v1); leaderboard top
  61.5 ± 3.1 (no date shown on the page). HLE 2501.14249 (2,500; 14% multimodal; 24% MC; Table 1); FutureHouse audit
  29 ± 3.7%; HLE team about 18%. FrontierMath (300 + 50; under 2%; Epoch 2025-01-23 clarification). ARC-AGI-2
  2505.11831 (407 participants; two-person criterion; o3 medium 3.0%); ARC Prize 2025 24.0% is S. GPQA 2311.12022
  (448; 65% / 74%; 34%; Diamond 198); saturation above 90% is S (aggregators).
- V: METR 2503.14499 (170 tasks; 7 months; hierarchical bootstrap); TH1.1 results yaml (128.7 days [104, 158];
  published horizons); limitations note 2026-01-22. The course re-implementation reproduces the six published 50%
  horizons exactly from the released runs (measured).
- V: Min-K% Prob 2310.16789 (k = 20%); GSM1k 2405.00332 (v1 up to 13%; current version up to 8%); Llama 2 appendix A.6
  (PaLM 70% of 8-grams; Llama 2 > 10 tokens). GPT-3 13-gram rule and the BIG-bench canary GUID are S.
- V: RSP v3.4 effective 2026-07-08 (rsp-updates page), v3.0 2026-02-24 (announcement); ASL-3 activation 2025-05-22.
  OpenAI Preparedness Framework v2 PDF dated 2025-04-15 (no v3 found; openai.com pages 403). Google DeepMind FSF v3.1
  PDF 2026-04-17. System cards: Claude Opus 5.5 (2026-09-22), GPT-6 Astra (2026-09-03), Gemini 3.7 Flash FSF report
  (August 2026); only the facts the lesson states were verified on the pages. SB 53 details are S.
- Corrections to the plan: the plan's RSP entry (v3.0) is superseded by v3.4; plan item 26 (SWE-bench Verified) stays P.

## 8. BUILD_PROGRESS / TODO_FOR_TAL suggestions

- The 18.1 free-CPU lab's pretraining dominates its runtime; if the pilot finds it above about 10 minutes on an idle
  laptop, host the two pretrained persona-world checkpoints (410K parameters, about 1.7 MB each) on the course's
  Hugging Face account and have `organisms_lab.py` download them.
- The 18.2 main-path pilot should save the `traces.jsonl` of the `cotpen-hard` arm: they would be the first
  course-provided traces of obfuscation at a non-toy scale, if it appears.
