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
- [ ] Phase 0: Colab pilot notebook(s) for Tal (`curriculum/pilots/`)
- [ ] Phase 0: pilot results received (`curriculum/pilots/RESULTS.md`) and projections updated
- [ ] Module 1 — How do we know a change helped? (01.1 done by main session as the exemplar; 01.2–01.5 + project + quiz: sub-agent A)
- [ ] Module 2 — Where does the time go? (sub-agent B)
- [ ] Module 3 — How should attention spend KV memory?
- [ ] Module 4 — Does the model use its context?
- [ ] Module 5 — When is sub-quadratic attention worth it?
- [ ] Module 6 — Which other block changes earn their complexity?
- [ ] Module 7 — Which optimizer and parametrization?
- [ ] Module 8 — How low can precision go?
- [ ] Module 9 — What does the cluster cost, and how does it fail?
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

- Sub-agent A: Module 1 lessons 01.2–01.5, project, module quiz (owns lessons/module-01 except 01.1, labs/module-01, frontierlab/calc, frontierlab/evals/suite_v0.py, frontierlab/record).
- Sub-agent B: Module 2 (owns lessons/module-02, labs/module-02, frontierlab/perf).
- Main session: waiting; then integrates, writes the Colab pilot notebook from both agents' GPU command lists, commits.

## Decisions and open questions

| Date | Decision |
|---|---|
| 2026-10-03 | See plan section 15 (name, free, rented-GPU main path, one course, publishing warning, budget accepted, Colab scaled pilots, Hugging Face hosting, multimodal later) |
| 2026-10-03 | Reuse the MoE course's quiz checker and labkit pattern; lab package `frontierlab` copies from `llmre`/`moelab`, imports neither |
