# Scaled pilot results

Planning file, not imported. Filled in from `pilot_phase0.ipynb` runs on Tal's Colab account
(plan section 12.1). Each entry: date, GPU, pilot, raw summary (from `PILOT_DIR/summary/`), decision
(go / redesign / downgrade to extension), and which lesson figures change from PROJECTED to measured.

| Pilot | Status | GPU | Date | Decision |
|---|---|---|---|---|
| P1 — Module 1 noise floor ladder | not run | | | |
| P2 — Module 2 performance (02.1, 02.2, 02.4) | not run | | | |
| P2 — 02.3 multi-GPU | not piloted (single-GPU Colab); optional Kaggle 2× T4 | | | |
| P1b — Module 1 lesson labs on GPU | not run | | | |
| P3 — Module 3 decode + logit-control arms | not run | | | |
| P3 — Module 3 project (21 runs, ~39 H100-h PROJECTED) | not piloted (budget) | | | |
| P4 — Module 4 long docs, Eval v1, zero-shot RoPE | not run | | | |
| P4 — Module 4 extension arms and project (~4 H100-h PROJECTED) | not piloted yet | | | |
| P5 — Module 5 fla check, profiles, DSA stages | not run | | | |
| P7 — Module 7 cost, logit ladder, induced failures | not run | | | |
| P7 — Module 7 project (~29 H100-h PROJECTED) | not piloted (budget) | | | |
| P6 — Module 6 MTP, HC/mHC ladder, factorial | not run | | | |
| P6 — Lineage-F integration (~60+ H100-h PROJECTED) | not piloted (budget) | | | |
| P8 — Module 8 FP8 on L4, sweep | not run | | | |
| P8 — NVFP4 on B200 | not piloted (no Blackwell on Colab) | | | |
| P9 — Module 9 single-GPU recovery (kill_and_resume --world 1) | not run | | | |
| P9 — Module 9 torchtitan 8× H100, ring CP, pipelining | not piloted (single-GPU Colab) | | | |
| P10 — Module 10 docmask, RegMix, micro-anneals, continued training (T4 variants) | not run | | | |
| P10 — Module 10 10.2, 10.3 and project (~13 H100-h PROJECTED) | not piloted (budget) | | | |
| P11 — Module 11 iso-FLOP ladder and de-risking (T4 variants) | not run | | | |
| P11 — Module 11 main-path ladder and Recipe-R target (~67 H100-h PROJECTED) | not piloted (budget) | | | |
| P12 — Module 12 RL loop on Qwen3-0.6B-Base, 2 arms x 2 seeds | not run | | | |
| P12 — Module 12 main path on Qwen3-1.7B-Base (~36–54 GPU-h PROJECTED) | not piloted (budget) | | | |
