# Scaled pilot results

Planning file, not imported. Filled in from the `notebooks/*.ipynb` runs on Tal's Colab account
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
| P13 — Module 13 smoke tests and 13.4 budgets on Qwen3-0.6B | not run | | | |
| P13 — Module 13 main-path pipeline on Qwen3-1.7B-Base (~24–37 GPU-h PROJECTED) | not piloted (budget) | | | |
| P14 — Module 14 GRPO vs CISPO on Qwen3-0.6B-Base | not run | | | |
| P14 — Module 14 main path (~73–130 GPU-h PROJECTED) | not piloted (budget) | | | |
| P15 — Module 15 test-time compute at 0.6B and 1.7B, speculative loop | not run | | | |
| P15 — Module 15 main path and project (~9–15 H100-h PROJECTED) | not piloted (budget) | | | |
| P16 — Module 16 multi-turn RL and misspecified rewards (T4 variants) | not run | | | |
| P16 — Module 16 main path and project (~11–19 GPU-h PROJECTED) | not piloted (budget) | | | |
| P17 — Module 17 SAE check, IOI claim, steering on Qwen3-1.7B | not run | | | |
| P17 — Module 17 17.3 circuit-tracer (separate env) and 17.5 introspection | not piloted yet | | | |
| P18 — Module 18 main-path smoke tests, 18.3 eval lab | not run | | | |
| P18 — Module 18 main path and project (~9–14 GPU-h PROJECTED) | not piloted (budget) | | | |
