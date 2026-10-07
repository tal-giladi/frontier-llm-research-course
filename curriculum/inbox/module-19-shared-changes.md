# Module 19 — proposed changes to shared files (for the main session)

Module 19 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/`, `templates/`,
`README.md`, `BUILD_PROGRESS.md` or `PUBLISHING_WARNING.md`. New code: `labs/common/frontierlab/research/`
(`__init__`, `proposals`, `reproduce`, `writeup`, `contract`), tests `labs/common/tests/test_research.py` (18 test functions, 22 cases,
a few seconds, no training, no downloads). It imports `frontierlab.stats` and `frontierlab.record` and edits neither.
The labs train through `frontierlab.optim.train` (Module 7's wrapper of the loop), unchanged.

## 1. `_sidebar.md`: Module 19 lines

Running numbers 81–83:

```markdown
- **Module 19 — How is frontier research chosen, reproduced and written?**
  - [81 · Research taste and problem choice](lessons/module-19/lesson-01.md)
  - [82 · Reproducing a paper](lessons/module-19/lesson-02.md)
  - [83 · Writing and defending results](lessons/module-19/lesson-03.md)
  - [Module 19 quiz](assessments/module-19-quiz.md)
```

The project page `projects/module-19-capstone-proposal.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-19.md` (26 terms).

## 3. `templates/` (optional, recommended)

Plan section 10 mentions a peer "review template" that does not exist in `templates/` yet. Module 19 wrote it as
`labs/module-19/lesson-03/review-template.md`. If you want it in the Templates section, copy it to
`templates/review.md` and add to `templates/README.md`:

"- [Review](review.md) — reviewing someone else's result: the claim restated, a claims table, controls and budget, uncertainty, figures, problems with where/evidence/request, and a recommendation."

(Lessons 19.3 and the project refer to the lab copy, so nothing breaks if you do not.)

## 4. `references/versions.md`

Add under "Notes on reference implementations": "Module 19: no new packages (matplotlib 3.11.2, already pinned, draws
the lesson 19.3 figure with the Agg backend). Checked 2026-10-07."

## 5. Plan section 14.1 / 14.2 records (2026-10-07)

- 14.2 "Research practice": Schulman's guide is now **V** (opened 2026-10-07; posted 2020-01-24, originally written
  December 2017 for the OpenAI Fellows program). Olah (2021-01-09), Karpathy (2019-04-25) and the tuning playbook
  re-opened and V.
- New claim checks, all V at the stated location: Wortsman et al. 2309.14322 section 2.1 (AdamW β2 0.95, ε 1e-8,
  clipping 1.0, independent decay 1e-4, z-loss 1e-4, 5e3 warm-up of 1e5 steps, cosine to 1e-5, batch 256 × 512, C4,
  rates 3e-4 to 3e-1), section 2.2 (LR sensitivity with ℓ0 the loss at initialisation), Figure 1 caption ("Qk-layernorm
  reduces LR sensitivity, but LR sensitivity still increases with model scale"; sizes not listed in the caption),
  section 3.1.1 (qk-layernorm = LayerNorm on queries and keys; 1.2B at learning rate 0.3), section 3.2.1 (longer
  warm-up reduces LR sensitivity); NASEM 2019 definitions of reproducibility and replicability; Pineau et al.
  2003.12206 (code policy, reproducibility challenge, checklist); Henderson et al. 1709.06560 Figure 5 (10 TRPO runs on
  HalfCheetah, two groups of 5 differ significantly); Lipton and Steinhardt 1807.03341 (four trends); Dodge et al.
  1909.03004 (expected validation performance against budget); NeurIPS 2026 paper checklist items; MLRC 2026 as a
  NeurIPS 2026 track.

## 6. Pilot commands (Module 19; none run in this build)

| Lab | Commands | PROJECTED | What to record |
|---|---|---|---|
| 19.1 | `python labs/module-19/lesson-01/choose_lab.py --variant main --print` (8 runs: `pilot-30m` at widths 256 and 512, QK-norm on/off, seeds 0–1, 2,000 steps × 32 × 1,024, lr 1e-2, bf16), then `--variant main --part proxy` | 0.13 H100-hours at 25% assumed MFU ($1.14 \times 10^{17}$ FLOPs) | final held-out loss per run; the effect per width; pooled seed noise; the `proxy_trend` reading; measured MFU and s/step |
| 19.2 | `python labs/module-19/lesson-02/repro_lab.py --variant main --print` (42 runs: `pilot-30m`, 2 arms × 7 rates 3e-4…3e-1 × 3 seeds, 4,000 steps × 64 × 1,024, bf16, z-loss 1e-4, decay 1e-4/lr, warm-up 200), then `--variant main --part analyse`; second rung `--variant main70` (`pilot-70m`) | 4.0 H100-hours (pilot-30m) and 7.6 (pilot-70m) at 25% assumed MFU | per run: final held-out loss, run-maximum attention logit, whether it diverged (nan); per arm and seed: sensitivity; the decision; whether sensitivity grows from pilot-30m to pilot-70m (the half of the claim CPU cannot test); measured MFU |
| 19.2 T4 | `--variant t4 --print` (30 runs, `pilot-10m`, 5 rates, 1,000 steps × 16 × 256, fp32) | 2.3 T4-hours | as above |

Measured in this build (free CPU, Windows 11, Python 3.12.13, torch 2.14.1+cpu, 16 threads): see the lessons'
"What the build's run gave" blocks and `curriculum/status/module-19.log`.
