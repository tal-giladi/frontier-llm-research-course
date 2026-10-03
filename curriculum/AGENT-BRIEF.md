# Agent brief — writing one module of Frontier LLM Research Engineering

Planning file, not imported. Read it fully before writing anything. Then read, in this order:

1. `C:\Users\TalGiladi\OneDrive\repos\tals-academy\docs\new-course-instructions.md` — binding rules; they win over everything here.
2. `plan.md` — section 3 (principles), section 4 (hardware), your module's entry in section 5, sections 6–7 (dependencies, artefact chain), 9 (experiment contract), 10 (assessment), 11 (lesson template), 12.1 (scaled pilots and the feasibility row for your module), 14.1 (claim-level checks — use the corrected forms).
3. The finished exemplar: `lessons/module-01/lesson-01.md`, its quiz `lesson-01.quiz.yaml`, and its lab `labs/module-01/lesson-01/` (lab.py, solution.py, test_lab.py, helper scripts). Match its depth, tone and structure.
4. The shared code: every file in `labs/common/frontierlab/` (it is small) and `labs/common/tests/`.
5. The templates in `templates/` (experiment contract, run card, rubric) — your labs and project use them.

## Your scope

You own exactly one module: `lessons/module-NN/`, `assessments/module-NN-quiz.*`, `labs/module-NN/`,
`projects/module-NN-*.md`, `curriculum/status/module-NN.log`, `curriculum/glossary-inbox/module-NN.md`,
`curriculum/inbox/module-NN-*.md` (anything you want added to shared pages), and the new
subpackage(s) of `frontierlab` your task names. You must NOT edit `_sidebar.md`, `glossary.md`,
`README.md`, `BUILD_PROGRESS.md`, `plan.md`, `references/`, `templates/`, or any existing
`frontierlab` file outside your subpackage. If you need a change there, put the exact text in your
final report (or an inbox file) and the main session will make it. Another agent may be working on
another module at the same time; never touch its files.

## Each lesson = four things, finished before the next lesson starts

1. `lessons/module-NN/lesson-MM.md`
2. `lessons/module-NN/lesson-MM.quiz.yaml` (3–5 questions)
3. `labs/module-NN/lesson-MM/` with `lab.py` (TODOs raising `NotImplementedError("TODO n: ...")`),
   `solution.py`, `test_lab.py` (uses `frontierlab.labkit.load_target(__file__)`), and any small
   script the lab needs. `LAB_TARGET=solution pytest labs/module-NN/lesson-MM` must pass and plain
   `pytest` on the untouched `lab.py` must fail on the TODOs. Run both.
4. Append `lesson-MM done` to `curriculum/status/module-NN.log`.

After the last lesson: `assessments/module-NN-quiz.md` (short intro: what it covers, pass mark 70%,
links to the lessons; no questions, no answers, never a heading containing "Answer"),
`assessments/module-NN-quiz.quiz.yaml` (8–10 scenario questions, required lessons only),
`projects/module-NN-<slug>.md` (the module project: experiment contract to fill, deliverables,
debugging task, written-defence questions, self-check against `templates/experiment-rubric.md`),
`labs/module-NN/README.md` (what each lab folder contains, how to run, hardware and time per variant),
then append `module done` to the log.

## Lesson file rules

- Front-matter exactly like the exemplar: `id`, `module`, `minutes`, `practice_minutes`,
  `prerequisites` (ids of earlier lessons in THIS course only; mention parent-course lessons in
  prose, e.g. "parent course lesson 12.3"), `objectives` (measurable), `volatility`
  (`concept` | `implementation`), `sources` (title + url), `last_verified: "2026-10-03"` (or the day
  you verified).
- One H1 `# NN.M · Title`, then one plain intro paragraph (no heading, list, bold, italics or HTML).
  Extension lessons start that paragraph with "Extension:".
- The fixed sections, in this order, in every lesson: `## Why this matters at a frontier lab`,
  `## The idea`, `## Worked example`, `## Shapes and cost`, `## Build it`, `## What the evidence says`,
  `## Lab`, `## Common mistakes`, `## References`, `## Next`. Use `###` inside them.
- No quiz / "Check yourself" / answers section in markdown. No `<div>`, `<script>`, `<style>`, inline
  SVG, iframes. Allowed: GFM, `$...$` / `$$...$$` KaTeX, mermaid, GitHub alerts (`> [!NOTE]` etc.),
  `<details>/<summary>`, `<kbd>`, `<sub>`, `<sup>`, `<br>`.
- Links are relative repo paths (`lesson-02.md`, `../module-02/lesson-01.md`,
  `../../templates/experiment-contract.md`, `../../labs/module-02/`). No absolute links to the course's
  own files, no docsify links. Links to lessons that do not exist yet are not allowed — refer to them
  in prose ("Module 5 measures this").
- Every equation gets a hand-worked numeric example with tiny numbers; every tensor its shape, dtype
  and device. Every claim about a model or framework names its primary source (paper section, config
  field, code file at a tag). Use the plan's evidence labels where it matters: PUBLICLY DOCUMENTED,
  REASONABLE INDUSTRY PRACTICE, INFERENCE/SPECULATION, (company claim). Tag techniques ESTABLISHED /
  PROMISING / MODEL-SPECIFIC in `## What the evidence says`.
- **Claims:** before stating a number from a paper, check it against the paper section (arXiv HTML
  pages work well with WebFetch). Use the corrected forms in plan section 14.1. If you cannot verify a
  claim, do not state it as fact.
- **Measured vs projected:** say "measured" only for numbers you produced by running code in this
  build, with machine and versions. Main-path GPU figures you could not run are PROJECTED (pending
  pilot) and must show the formula they come from. Never present a small course experiment as proof of
  a frontier-scale claim.
- Plain, direct English. Short sentences. Explain what the code does; never "PyTorch handles it".

## Labs

- Every lab has a variants table like the exemplar: **Main path** (rented GPU; the course standard),
  **Free GPU** (Colab/Kaggle T4), **Free CPU** (laptop). The main path is what the lesson is designed
  around; free variants are scaled down and say what the learner will not see.
- Every lab that compares anything starts with a filled-in **experiment contract** (plan section 9)
  as a `### Experiment contract` subsection of `## Lab`: question, hypothesis and its status, baseline,
  changed and controlled variables, comparison axis, budget, metrics with uncertainty and decision
  rule, correctness checks, fallback evidence, limits. Pass checks are about soundness (correctness
  checks pass, contract followed, uncertainty reported), never about the new method winning.
- You can only run CPU code here. Write main-path GPU code carefully (CUDA paths, bf16 autocast,
  `torch.cuda.synchronize()` around timings) but label it "not run in this build; part of the Module
  NN pilot" and list every such command in your final report so it goes into the Colab pilot notebook.
- Training runs use `python -m frontierlab.train.loop` or a module script that extends it (do not edit
  the loop; wrap or subclass it inside your subpackage, or put what you need in your inbox), the course
  data (`labs/common/data/v0`, already prepared at the CPU size: 20,000 documents, vocab 8192) and
  local JSONL metrics. Keep each free-CPU lab under about 60–90 minutes on a 16-thread laptop; state the
  measured runtime.
- Tests compare against a reference with stated tolerances (float64 where it helps).
- New reusable code goes into your `frontierlab` subpackage with tests in
  `labs/common/tests/test_<subpackage>.py`.
- Python env: `C:/Users/TalGiladi/OneDrive/repos/course-creator/frontier-llm-research-course/.venv/Scripts/python.exe`.
  Run tests with `-m pytest -p no:cacheprovider`. Do not install packages without saying so in your report.

## Quizzes

Follow binding guide section 5 exactly. Then run
`.venv/Scripts/python.exe curriculum/tools/check_quiz.py <files>` and fix every ERROR and warning.
Questions test decisions in realistic situations, not wording from the page. Use `>-` block scalars.
Spread `correct` evenly across 0–3 within each file.

## Glossary

New terms go in `curriculum/glossary-inbox/module-NN.md` as `- **Term** — definition. [NN.M]`.

## Before you report

- All lab tests pass with `LAB_TARGET=solution`; `labs/common` tests pass.
- `check_quiz.py` clean on all your quiz files.
- Every link in your markdown points to an existing file.
- Final report: lessons done; measured numbers (runtimes, results) and on what; claims you could not
  verify; GPU commands for the pilot notebook (exact commands, expected runtime, what to record); any
  change you need in shared files (exact text); sidebar lines for your module in the `_sidebar.md`
  format (running numbers will be fixed by the main session).
- Do not commit; the main session commits.
