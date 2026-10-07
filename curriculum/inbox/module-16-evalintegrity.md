# Module 16 — evaluation-integrity lab: shared-file changes for the main session

The missing hands-on evaluation-integrity lab was added (2026-10-07) as
`labs/module-16/evalintegrity/` and integrated into lessons 16.1 and 16.4 (per
`curriculum/module-16-prompts.md` section "What differs for the learner": the six published harness
exploits are runnable again, now confined to a deliberately vulnerable toy runner, labelled as
scripted fixtures, and measured against a defence ladder). Two files outside the module's own folders
had to change; both edits are small, precedented and required by the addition:

1. **`.gitattributes`** — add one line so the new manifest-hashed trace file survives checkout
   byte-for-byte (same mechanism as `labs/module-16/lesson-04/traces/*.jsonl`, commit 2f4c023):

   ```
   labs/module-16/evalintegrity/traces/*.jsonl -text
   ```

2. **`PUBLISHING_WARNING.md`** — the Module 16 bullet previously stated lesson 16.4 "as built
   contains no exploit code and nothing that manipulates, bypasses or fakes a test harness, test
   report or exit status", which the new lab makes inaccurate. The bullet now distinguishes the RL
   lab (unchanged, no exploit code) from the evaluation-integrity lab, and lists exactly what Tal
   should review there: the scripted-and-labelled fixtures, their confinement to the toy runner and
   disposable sandboxes, the hardened verifier as the authoritative path, the honestly stated
   isolation limits, and the tree-unchanged checks. The updated text is already in the file.

No `_sidebar.md` change is needed: no lesson was added or renumbered — the lab folder ships inside
the Module 16 zip, and both lessons' ids, titles and paths are unchanged.

The module's own files changed by this addition (all inside `labs/module-16/`,
`lessons/module-16/`, `curriculum/glossary-inbox/module-16.md` and this inbox file):

- new: `labs/module-16/evalintegrity/{toyrunner.py, demos.py, lab.py, solution.py, test_lab.py,
  tamper_lab.py, traces/acceptance-matrix.jsonl, traces/manifest.json}`
- updated: `lessons/module-16/lesson-01.md`, `lessons/module-16/lesson-04.md`,
  `lessons/module-16/lesson-01.quiz.yaml`, `lessons/module-16/lesson-04.quiz.yaml`,
  `labs/module-16/README.md`, `curriculum/glossary-inbox/module-16.md`,
  `curriculum/status/module-16.log`.

Everything checks: `LAB_TARGET=solution pytest labs/module-16` passes (41 tests),
`pytest labs/module-16/evalintegrity` fails on the untouched `lab.py` TODOs and passes with the
reference, `pytest labs/common` passes (515 tests), `check_quiz.py` is clean on all three Module 16
quiz files, and the Academy dry-run import reports 0 problems.
