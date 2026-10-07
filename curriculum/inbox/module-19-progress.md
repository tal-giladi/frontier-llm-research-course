# Module 19 — progress notes for a resuming agent

Status lines are in curriculum/status/module-19.log. Do not commit.

## Design decisions
- Shared code: labs/common/frontierlab/research/ (proposals.py 19.1, reproduce.py 19.2, writeup.py 19.3, contract.py project). Tests: labs/common/tests/test_research.py.
- 19.1 lab (labs/module-19/lesson-01): TODOs score, p_decisive (= power x p_transfer), proxy_trend, PROPOSALS (3 ranked). Script choose_lab.py (to write): part proxy = QK-norm on/off at lr 1e-2, widths 64 and 128 (optim.train --width), 2 seeds, 200 steps -> proxy_trend; part rank = score table, rank_robustness, MDE; writes runs/m19/l191/proposals.md.
- 19.2 lab (labs/module-19/lesson-02): reproduce Wortsman et al. 2309.14322 "Qk-layernorm reduces LR sensitivity" (Fig. 1 caption, sec 3.1.1). repro_lab.py: arms qknorm/noqk, LRs 3e-3,1e-2,3e-2, seeds 0-2, toy 300x16x128, adamw (course MuonAdamW in adamw mode), z-loss 1e-4, weight decay 1e-4/lr (= independent decay), warmup 5%. Tolerance = 2 x 0.0286 (lesson 01.4 toy seed std). Effect per seed = sens(noqk) - sens(qknorm). lab.py still to write (TODOs lr_sensitivity, seed_effects, decide, deviations).
- 19.3 lab: flawed write-up (QK-Clip vs QK-norm, fictional numbers) with run cards; planted: missing seeds, unmatched budget (400 vs 300 steps), cherry-picked checkpoint (best on test split), claim beyond evidence (frontier). Learner implements lint subsets + fills REVIEW. Review template lives in the lab folder (propose promoting to templates/ via inbox).
- Project: projects/module-19-capstone-proposal.md + labs/module-19/project/ (proposal.yaml template, check_proposal.py using research.contract.validate, buggy_proposal.yaml debugging task, example in details). No links to Module 20 files.
- Measured: toy run 300 steps 16x128 adamw takes ~2m13s on the build laptop.

## Verified sources (2026-10-07)
- Schulman, An Opinionated Guide to ML Research, posted 2020-01-24 (written Dec 2017), http://joschu.net/blog/opinionated-guide-ml-research.html — sections Choosing Problems (Honing Your Taste; Idea-Driven vs Goal-Driven; recommends goal-driven; Aim High; 10% improvement must be ~2 lines), Making Continual Progress (Keep a Notebook; When to Switch Problems), Personal Development (reimplement papers, compare to published results).
- Olah, Research Taste Exercises (rough note), 2021-01-09, https://colah.github.io/notes/taste/ — testing ideas is expensive (months), exercises are proxy feedback; Ex.1 mentor rates ideas 1-10; Ex.2 when others try your ideas compare with expectations; failure modes sunk cost etc.
- Karpathy, A Recipe for Training Neural Networks, 2019-04-25, https://karpathy.github.io/2019/04/25/recipe/ — training "fails silently"; steps 1 data, 2 skeleton+dumb baselines (fix seed, verify loss at init), 3 overfit, 4 regularize, 5 tune, 6 squeeze.
- Google Deep Learning Tuning Playbook (Godbole, Dahl, Gilmer, Shallue, Nado), https://github.com/google-research/tuning_playbook — incremental tuning; exploration vs exploitation ("primary goal is to gain insight"); scientific / nuisance / fixed hyperparameters.
- Wortsman et al. 2309.14322: setup sec 2.1 (AdamW b1 .9 b2 .95 eps 1e-8, clip 1, independent decay 1e-4 = not multiplied by lr, z-loss 1e-4, warmup 5e3 of 1e5 steps, cosine to 1e-5, batch 256 x 512, C4, LR 3e-4..3e-1 seven values, pre-LN, no biases); LR sensitivity sec 2.2 with l0 = loss at init; Fig. 1 caption "Qk-layernorm reduces LR sensitivity, but LR sensitivity still increases with model scale"; sec 3.1.1 qk-layernorm = LayerNorm on q,k; allows 1.2B at lr 0.3; sec 3.2.1 longer warm-up reduces LR sensitivity; Fig.1 does not list numeric sizes in caption.
- NASEM Reproducibility and Replicability in Science (2019) definitions: https://www.nationalacademies.org/read/25303/chapter/2
- Pineau et al. 2003.12206 (NeurIPS 2019 reproducibility program: code policy, challenge, checklist).
- Henderson et al. 1709.06560: 10 TRPO trials on HalfCheetah split into two groups of 5 give statistically different curves (Figure 5).
- Lipton & Steinhardt 1807.03341: four troubling trends.
- Dodge et al. 1909.03004: expected validation performance vs computation budget.
- NeurIPS 2026 paper checklist https://neurips.cc/public/guides/PaperChecklist (items incl. Claims, Limitations, Experimental Result Reproducibility, Experiment Statistical Significance — state factors of variability, std vs SE; Experiments Compute Resources).
- MLRC https://reproml.org/ — MLRC 2026 is a NeurIPS 2026 track.

## Status (final)
Module 19 complete; see curriculum/status/module-19.log and curriculum/inbox/module-19-shared-changes.md.

## Next (historical)
choose_lab.py + lesson-01.md + quiz; lesson-02 lab.py/test/lesson/quiz after sweep; lesson-03; module quiz, project, README, glossary inbox, shared-changes inbox.
