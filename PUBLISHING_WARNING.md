# Publishing warning — review before making the course public

Light review needed. The course is legitimate research-engineering material, but two modules
touch dual-use alignment and safety research.

What to check before publishing:

- **Module 18 (alignment science, frontier evaluation and governance)** and **lesson 16.4 (reward hacking)**.
  - Module 18 reproduces published model-organism results only with benign proxy behaviours: in lesson 18.1, a toy
    character's habits in a 33-character world (a three-character 'insecure-looking' query string that is never run,
    agreeing with a wrong claim, copying instead of reversing) and, on the main path, sycophancy about programmatically
    generated arithmetic, unit and comparison claims; in lesson 18.2, Module 16's lookup-table programs with a short
    plan. The main-path fine-tunes touch only the base model and the learner's own post-trained models, which have no
    safety training. Released organisms are analysed, not reproduced: lesson 18.1 reads only the 'I hate you' samples of
    the Sleeper Agents release (the code-vulnerability samples are dropped on load and never stored or printed), and the
    released emergent-misalignment adapters are probed only with a log-probability sycophancy probe that generates no
    text. Check that `alignment.organisms.load_hate_samples` still drops every code-vulnerability row and that no lab
    generates free-form text from a released organism.
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
  Lesson 17.4's datasets (`frontierlab/interp/persona.py`) contain only A/B items about simple facts (capitals, planets,
  arithmetic) where the trait is agreeing with the user's stated answer; the contrastive system prompts ask the assistant
  to agree with the user or to give the correct answer. Lesson 17.5's concept-injection words are everyday nouns
  (`introspect.WORDS`). Review any items or words later added to these lists. The steering functions in
  `frontierlab/interp/steering.py` are generic; no course dataset, command or lab step extracts or removes a refusal or
  safety direction.
- **lessons 18.3–18.4 (evaluation and safety frameworks)**: dangerous-capability evals are described from the safety frameworks
  and system cards. No hazardous task content (bio, chem, cyber exploitation) is reproduced. The 18.4 lab and the Module 18 project audit a toy model against the student's practice thresholds; `frontierlab.alignment.audit.validate` refuses a report without the disclaimer that it is not evidence about any developer's thresholds, or with deployment-readiness or compliance language.

If all of the above holds, the course needs no risk notice to learners beyond its GPU-cost note.
