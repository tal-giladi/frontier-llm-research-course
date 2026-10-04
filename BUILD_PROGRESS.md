# BUILD_PROGRESS — Frontier LLM Research Engineering

Working file. Not imported into the Academy.

## Goal

Build the course in `plan.md` (revision 2.1): 20 core modules, 72 required lessons + 13 extensions
(Module 21 elective comes after the first release), import-ready per
`C:\Users\TalGiladi\OneDrive\repos\tals-academy\docs\new-course-instructions.md` (binding).

## Resume procedure

1. Read this file, then `curriculum/status/*.log`, then `plan.md` and `curriculum/AGENT-BRIEF.md`.
2. Skip any lesson whose `lesson-MM done` line exists in its module log. Never rewrite a finished lesson.
3. Python env: `.venv` at the repo root:
   `uv venv .venv --python 3.12`, then
   `uv pip install --python .venv/Scripts/python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`
   and `uv pip install --python .venv/Scripts/python.exe -r labs/common/requirements-cpu.txt -e labs/common`.
4. Tests: `cd labs/common && ../../.venv/Scripts/python.exe -m pytest`.
5. Quiz check: `.venv/Scripts/python.exe curriculum/tools/check_quiz.py lessons/module-NN/*.quiz.yaml assessments/module-NN-quiz.quiz.yaml`.
6. Import check after every module:
   `cd C:/Users/TalGiladi/OneDrive/repos/tals-academy && npm run check-course -- ../course-creator/frontier-llm-research-course`
7. Commit after every finished module (at the latest).

## Gate

Scaled pilots (plan section 12.1) run on Tal's paid Colab account. Until a module's pilots are in
`curriculum/pilots/RESULTS.md`, its lessons may be written, but every main-path figure is marked
PROJECTED (pending pilot) and the free variants are written after the pilot, per plan section 12.

## Units

- [x] Plan revision 2.1 and all decisions (plan section 15)
- [x] Phase 0: repo spine (git, .gitignore, .gitattributes, .venv) — commit 6c54917
- [x] Phase 0: `labs/common/frontierlab` foundation (Baseline-0 model, data prep, training loop with exact resume, run cards, correctness suite, stats helpers) with tests — 13 tests pass; Data-v0 CPU slice prepared (20k docs, vocab 8192, 24.1M train tokens)
- [x] Phase 0: shared docs (README, _sidebar, glossary, templates, references, course-details, AGENT-BRIEF)
- [x] Phase 0: Colab pilot notebook for Tal (`curriculum/pilots/pilot_phase0.ipynb`, built by `build_notebook.py`; P1 noise floor, P2 Module 2). Add later modules' pilots there.
- [ ] Phase 0: pilot results received (`curriculum/pilots/RESULTS.md`) and projections updated
- [x] Module 1 — How do we know a change helped? — commit f83216e; dry-run import 0 problems
- [x] Module 2 — Where does the time go? — commit f83216e; dry-run import 0 problems
- [x] Module 3 — How should attention spend KV memory? (sub-agent C, integrated)
- [x] Module 4 — Does the model use its context? (sub-agent D, integrated)
- [x] Module 5 — When is sub-quadratic attention worth it? (sub-agent E, integrated)
- [x] Module 6 — Which other block changes earn their complexity? (integrated)
- [x] Module 7 — Which optimizer and parametrization? (sub-agent F, integrated)
- [x] Module 8 — How low can precision go? (integrated)
- [x] Module 9 — What does the cluster cost, and how does it fail? (integrated)
- [ ] Module 10 — Which data, in which mix?
- [ ] Module 11 — What will the big run do?
- [ ] Module 12 — Post-training foundations
- [ ] Module 13 — Which post-training pipeline for which target?
- [ ] Module 14 — Which RL objective, at what scale?
- [ ] Module 15 — How should compute be spent at inference?
- [ ] Module 16 — How do we train agents without fooling ourselves?
- [ ] Module 17 — What can we claim about a model's internals?
- [ ] Module 18 — Alignment science, frontier evaluation and governance
- [ ] Module 19 — How is frontier research chosen, reproduced and written?
- [ ] Module 20 — Capstone
- [ ] Course-provided artefacts on Hugging Face (needs Tal's account)
- [ ] Final QA: dry-run import 0 problems, links, git clean
- [ ] (after first release) Module 21 — multimodal elective

## Who is working on what

- 2026-10-04: machine restart interrupted two sub-agents: Module 9 (owns lessons/module-09, labs/module-09, frontierlab/dist, tests/test_dist.py) and Module 10 (owns lessons/module-10, labs/module-10, frontierlab/datax, tests/test_datax.py). Their partial files are uncommitted. On resume: read curriculum/status/module-09.log and module-10.log, keep finished lessons, relaunch each agent to finish only what is missing (same prompts as before, plus "resume: skip lessons marked done").
- Modules 1–8 committed and pushed (db23392).

## Decisions and open questions

| Date | Decision |
|---|---|
| 2026-10-03 | See plan section 15 (name, free, rented-GPU main path, one course, publishing warning, budget accepted, Colab scaled pilots, Hugging Face hosting, multimodal later) |
| 2026-10-03 | Root `pytest.ini` uses `--import-mode=importlib` so every lab folder can keep `test_lab.py` |
| 2026-10-03 | Reuse the MoE course's quiz checker and labkit pattern; lab package `frontierlab` copies from `llmre`/`moelab`, imports neither |
| 2026-10-04 | Continued training from a checkpoint goes through `frontierlab.longctx.extend` (wraps the loop); the native `--init-from` loop change proposed in `curriculum/inbox/module-04-loop-changes.md` is deferred until a later module needs it |
| 2026-10-04 | RMSNorm keeps float64 as float64 (Module 3 finding); float32/bf16 behaviour unchanged |
| 2026-10-04 | Optimizer runs go through `frontierlab.optim.train` (wraps the loop); native loop flags in `curriculum/inbox/module-07-loop-changes.md` deferred. Sidebar running numbers reserve 21–27 for Module 6 (7 lessons). |
| 2026-10-04 | Loop forks the RNG around the accounting calls (Module 8 found that accounting broke exact resume once training uses randomness). `gqa-bidir` is not registered in `attention/__init__` (circular import); 06.7 imports `frontierlab.blocks.diffusion` itself. Block runs use `frontierlab.blocks.train`. |
