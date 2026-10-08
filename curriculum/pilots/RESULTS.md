# Scaled pilot results

Planning file, not imported. Filled in from the `notebooks/*.ipynb` runs on Tal's Colab account
(plan section 12.1). Each entry: date, GPU, pilot, raw summary (from `PILOT_DIR/summary/`), decision
(go / redesign / downgrade to extension), and which lesson figures change from PROJECTED to measured.

| Pilot | Status | GPU | Date | Decision |
|---|---|---|---|---|
| P1 — Module 1 noise floor ladder | partial: 6 of 9 runs (70m s0 stopped at 16,600/21,606 steps; 70m s1–s2 not started) | A100-SXM4-40GB | 2026-10-07 | pending 70m |
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

## P1 — partial (received 2026-10-08)

Colab A100-SXM4-40GB, torch 2.14.1+cu130, Python 3.13.15, commit `edddf3e`, bf16, no compile, batch 32 × 1,024,
lr 3e-3 cosine, tokens ≈ 10 × non-embedding parameters. Final held-out loss (128 eval windows):

| Size | Steps | Seeds 0 / 1 / 2 | Mean | Seed std | tok/s | MFU | Peak memory | Wall per run |
|---|---|---|---|---|---|---|---|---|
| pilot-10m | 2,868 | 4.2570 / 4.1885 / 4.1989 | 4.2148 | 0.0369 | 247.5k | 11.6% | 16.7 GB | 0.13 h |
| pilot-30m | 9,613 | 3.6306 / 3.6215 / 3.6166 | 3.6229 | 0.0071 | 142.9k | 14.7% | 23.0 GB | 0.80–0.86 h |
| pilot-70m | 21,606 | s0 at step 12,963: 3.6045 (stopped at 16,600) | — | — | 88.7k | 17.4% | 32.4 GB | ~2.8 h (PROJECTED from tok/s) |

Reading so far: seed std falls from 0.037 at 10m to 0.007 at 30m (three seeds each, so each std is itself uncertain
by roughly ±50%). MFU is low (no `torch.compile`, small models on an A100), which matters for every projection that
assumed 20–30%. Remaining: 70m s0 (~0.65 h) and s1–s2 (~2.8 h each), about 6.3 A100-hours; `01_P1.ipynb` resumes
them. No lesson figure changed yet: wait for the 70m seeds before replacing the Module 1 PROJECTED noise floor.
