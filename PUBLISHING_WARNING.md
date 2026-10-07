# Publishing warning — review before making the course public

Light review needed. The course is legitimate research-engineering material, but two modules
touch dual-use alignment and safety research.

What to check before publishing:

- **Module 18 (alignment science, frontier evaluation and governance)** and **lesson 16.4 (reward hacking)**.
  - Labs reproduce published model-organism results only with benign proxy behaviours (Module 18: insecure-looking
    toy code or sycophancy).
  - Lesson 16.4's RL lab contains no exploit code and nothing that manipulates, bypasses or fakes a test harness,
    test report or exit status; its hands-on runs use rewards misspecified by design (format-only, length-based,
    visible input/output pairs only) on a 0.3M-parameter policy and a 25-character program grammar that is
    parsed, never executed. Review any candidate later added to
    `frontierlab.agents.codeenv.CANDIDATES` through `register()` (see `curriculum/module-16-prompts.md`).
  - **The evaluation-integrity lab (added 2026-10-07, `labs/module-16/evalintegrity/`, lessons 16.1 and 16.4) is
    the one place the module ships runnable tampering code.** Review it before publishing. What to check:
    - Every demonstration is a **scripted, hand-written fixture** in `evalintegrity/demos.py`, labelled with the
      published case it reproduces (MacDiarmid et al. 2025 sec. 2; Baker et al. 2025 sec. 2.1–2.2; METR June
      2025; ImpossibleBench) and whether it is a near-faithful reproduction or a simplified teaching adaptation.
      Nothing is presented as discovered by an RL policy, and the acceptance matrix is measured, never assumed.
    - The demonstrations run **only** against the deliberately vulnerable toy runner in
      `evalintegrity/toyrunner.py`, inside disposable course-sandbox directories, on a toy `fizzbuzz` task with
      course-owned boot scripts. The only plugin hook (the `conftest.py` report rewriting) exists only in that
      file. They never see the course's real tests, CI, `frontierlab` agent tooling or evaluation results;
      `tamper_lab.py` hashes the `labs/` tree before and after each run and the regression tests assert it is
      unchanged.
    - The hardened verifier is the authoritative path and the pedagogical point: the lab measures which design
      choice closes each demonstration (protected tests, trusted result reporting with the complete-execution
      rule, hidden and fresh inputs, property checks), states that isolation is not a security boundary, and
      shows hidden tests alone do not prevent runner tampering (4 of 6 demos still pass that level).
    - Isolation: fresh temporary workspace, scrubbed environment, network and process guard, wall-clock timeout,
      output caps, POSIX rlimits — with honestly stated limits (the child runs as the OS user and can write
      outside its workspace; measured in the lab's part A). Container commands are printed, not run.
  - Check that no lab produces, or explains how to produce, a model with harmful capability, and
    that no lab shows how to remove a released model's safety training.
  - Check that the misalignment discussion stays at the level of the cited papers.
- **lesson 17.4 (steering)**: steering examples use harmless traits only (style, sycophancy, topic).
  No refusal-direction removal (that is the separate `abliteration` course and its own warning).
- **lessons 18.3–18.4 (evaluation and safety frameworks)**: dangerous-capability evals are described from the safety frameworks
  and system cards. No hazardous task content (bio, chem, cyber exploitation) is reproduced.

If all of the above holds, the course needs no risk notice to learners beyond its GPU-cost note.
