# Module 20 — proposed changes to shared files (for the main session)

Module 20 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `README.md`, `references/`, `templates/`
or `PUBLISHING_WARNING.md`. New code: `labs/common/frontierlab/capstone/` (`__init__`, `__main__`, `claims`,
`uncertainty`, `package`, `scaffold`, `review`, `sample`), tests `labs/common/tests/test_capstone.py` (26 tests, about
80 s on the build laptop, no downloads; one test runs the scaffold end to end for 6 steps on the `tiny_data` fixture).
It imports `frontierlab.stats`, `frontierlab.record.diff`, `frontierlab.optim` (train wrapper, qkclip, stability),
`frontierlab.evals.heldout`, `frontierlab.data.loader`, `frontierlab.metrics.jsonl`, `frontierlab.model`,
`frontierlab.testing`, `frontierlab.labkit`, and `frontierlab.posttrain.arms.t_interval` (tests only), and edits none
of them. Nothing in Module 19 is imported.

## 1. `_sidebar.md`: Module 20 lines

Running numbers 84–85 (fix if Module 19 ends elsewhere):

```markdown
- **Module 20 — Capstone: reproduce, extend, defend**
  - [84 · Capstone brief and rubric](lessons/module-20/lesson-01.md)
  - [85 · Review and defence](lessons/module-20/lesson-02.md)
  - [Module 20 quiz](assessments/module-20-quiz.md)
```

The project page `projects/module-20-capstone.md` is picked up from `projects/` automatically. It links to
`projects/module-19-capstone-proposal.md`; lesson 20.1 links to `lessons/module-19/lesson-02.md` and
`lessons/module-19/lesson-03.md` and the project page; prerequisites use `"19.3"`.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-20.md` (13 terms).

## 3. `README.md`: proposed final "course complete" section

Add at the end of README.md (after "Course map"), once every module is in:

```markdown
## When you have finished

You will have one traceable chain of run cards, from Data-v0 and Baseline-0 through isolated architecture branches,
an integrated architecture, an optimizer, precision and data recipe, a pre-registered run, post-training, evaluation,
an interpretability claim and an audit, ending in a capstone: one claim from a 2025–26 report reproduced at a scale
you could afford, extended with one ablation nobody had published, reviewed, defended in writing and revised. Every
comparison in it has an experiment contract, paired intervals and stated limits, and a null result in it counts as
much as a positive one. That record, not the quizzes, is what shows you can do the job: keep it, and add the
measured costs to your run cards so the course's PROJECTED figures can be corrected.

The Module 21 elective on native multimodality follows when it is published.
```

(Drop the last sentence, or turn it into a link, depending on when Module 21 ships.)

## 4. `frontierlab/__init__.py`

No change needed (`capstone` is an ordinary subpackage). Optionally extend the package docstring's module list with
"capstone (Module 20): claim list, package checker, scaffold, review".

## 5. `PUBLISHING_WARNING.md`

No change: Module 20 has no dual-use content. Its sample capstone is a constructed fixture about RL objectives with
invented numbers, labelled as such in code, lesson and project.

## 6. `references/versions.md`

Add under "Notes on reference implementations": "Module 20: no new packages. `frontierlab.capstone` uses NumPy, SciPy
(`scipy.stats.t` for the seed t-interval, already a dependency through `posttrain.arms`), PyYAML and the course loop.
Checked 2026-10-07."

## 7. Plan section 14.1 claim checks to record (2026-10-07)

- **#1 (DSA) gap closed:** DeepSeek-V3.2 §2.1.1 says the sparse-stage KL is computed only over the selected token set
  $S_t$; the indexer input is detached; dense warm-up lr $10^{-3}$, sparse lr $7.3\times10^{-6}$. §2.2 has no numeric
  parity table ("no substantial performance degradation"; AA-LCR four points higher in reasoning mode; ChatbotArena
  Elo "closely matched", values not given).
- **#7 (Kimi K2) addition:** Appendix D / Figure 12: two MoE models of 0.5B activated / 3B total, vanilla Muon vs
  MuonClip at $\tau = 30$, "negligible impact on loss", no statistically significant downstream degradation; in the
  full run 12.7% of heads triggered QK-Clip in the first 70k steps and it was inactive after. The "about 30% of
  steps" decay is from Figure 2's caption only.
- **#8 (GSPO):** Figure 1 is a cold-start model fine-tuned from Qwen3-30B-A3B-Base; GRPO baseline ranges 0.2 / 0.27;
  "two orders of magnitude" more clipped tokens (§5.2, Figure 2, no exact fractions in the text); Routing Replay
  §5.1, §5.3, Figure 3.
- **#23 (mHC):** HC loss surge near step 12k at 27B (§3.1, Figure 2); Amax gain peaks of 3,000 (Figure 3b); mHC at
  most about 1.6 (§5.4, Figure 7b); 0.021 final-loss reduction vs baseline (§5.2); Table 4 values as in lesson 06.2.
- **#28 (on-policy distillation):** blog by Kevin Lu (Thinking Machines), 2025-10-27, now V (the page loaded):
  9× with the SFT dataset given, about 18× in GPU-hours, about 30× including teacher sampling; the post notes that
  one experiment actually used Qwen3-8B as the teacher, and a June 2026 update moved its recipes to Qwen3.5-9B.
- **New, micro-anneals:** OLMo 2 2501.00656 §4.4.2 and Table 12 (19 microanneals, 130B tokens, fewer than three full
  50B anneals; MMLU and GSM* of 200 items); Olmo 3 2512.13961 §3.5.1 (5B target + 5B web against a 10B web-only
  microanneal). Lesson 10.4 cites only OLMo 2; consider adding Olmo 3 there.
- **New, review practice:** Cortes and Lawrence 2109.09774 (about 50% of score variation subjective); Pineau et al.
  2003.12206 (NeurIPS 2019 reproducibility programme: code policy, challenge, checklist).

## 8. Pilot commands (Module 20; none run on GPU in this build)

All on 1× H100 80 GB. First the CPU smoke test: `pytest labs/common/tests/test_capstone.py -k scaffold_end_to_end`.

| Lab | Commands | PROJECTED | What to record |
|---|---|---|---|
| 20.1 rung 1 | `python labs/module-20/lesson-01/capstone_lab.py --variant main --print` prints them: `python -m frontierlab.capstone.scaffold --variant main --out runs/m20/l201/capstone-qkclip-main --device cuda` (or the 9 `frontierlab.optim.train` commands: pilot-30m, 4,000 steps × 64 × 1,024, bf16, lr 1e-2, seeds 0–2, arms muon-noqk / muon-clip at the rule's tau / muon-qknorm) | 1.1 GPU-hours (9 × 2.62e8 tokens × 3.21e8 FLOPs per token ÷ (989e12 × 0.2) × 1.03) | the seed-0 baseline's final max logit and the resulting tau; the first clip step and clipped head-updates; reproduction and extension intervals and decisions; noise floor; s/step and MFU per arm |
| 20.1 rung 2 | same with `--preset pilot-70m` | 2.1 GPU-hours | as rung 1: does the reproduction's decision hold, and do logits grow faster? |
| 20.1 rung 3 | Baseline-0, 9 runs × 2.49e9 tokens | 17.0 GPU-hours (30% MFU) | as rung 1; only if rungs 1–2 leave the decision open |
| 20.1 T4 | `--variant t4` (pilot-10m, 2,000 steps × 32 × 512, fp32) | 1.5–2.5 hours | as rung 1 |
| 20.2 | none (CPU only) | — | — |

The other five claims' main paths are the earlier lessons' pilot commands (14.2, 05.2/05.3, 06.2, 10.4, 13.2) with
three seeds and the extension arm; their PROJECTED totals and formulas are in `frontierlab/capstone/claims.py`.

## 9. Measured in this build

Free CPU, Windows 11, Python 3.12.13, torch 2.14.1+cpu, 8 threads, another module's jobs sharing the CPU:

- `python -m frontierlab.capstone.scaffold --out runs/m20/capstone-qkclip` (the lesson 20.1 lab's runs with the
  reference functions): 1,164 s in total; 9 training runs of 110-138 s each. Correctness checks passed (QK-Clip cap
  error 3.6e-15, refusal of a QK-norm model, causal 0, cached decode 7.8e-16). tau = round(0.6 x 13.37) = 8; the clip
  fired from steps 111-120 with 91-114 head-updates per run; unclipped run max logits 14.2-15.8. Reproduction
  clip - no-clip +0.0015 nats [+0.0013, +0.0018] (seed t identical): equivalent within 0.02. Extension QK-norm -
  QK-Clip +0.0205 [+0.0056, +0.0332], seed t [-0.0170, +0.0581]: "a higher" by the rule, t-interval disagrees.
  Noise floor 0.0080, MDE 0.018. Package checker: no problems.
- `review_lab.py` all steps with the reference: about 6 s (12 problems before, 8 after the simulated reruns including
  one new CLAIM_DIRECTION created by a rerun, 0 after the reference revision).
- `capstone_project.py`: 2 s; with `CAPSTONE=buggy` the reproduction becomes "a higher" and the extension interval
  narrows to [+0.0166, +0.0246] (the pairing bug does not show on this package; its test catches it).
- `pytest labs/common/tests/test_capstone.py`: 26 passed, about 80 s.

## 10. BUILD_PROGRESS / TODO_FOR_TAL suggestions

- The plan's capstone budget (50–150 H100-hours, plan section 8) is above the list's projections (1.5–44 GPU-hours)
  because the list reuses earlier labs' sizes; the plan's figure can stay as the upper envelope for learners who add a
  Baseline-0 or larger rung, or be revised down after the Module 20 pilot.
- The QK-Clip main-path pilot is cheap (rungs 1–2 about 3.2 GPU-hours) and would give the course its first
  non-toy evidence on whether QK-Clip and QK-norm differ at equal tokens; worth prioritising among cost-only pilots.
