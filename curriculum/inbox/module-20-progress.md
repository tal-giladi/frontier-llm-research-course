# Module 20 — progress notes for a resuming agent (not imported)

## Decisions
- Scaffold claim: QK-Clip (frontierlab/capstone/scaffold.py). Arms muon-noqk (baseline), muon-clip (reproduction of
  Kimi K2 App. D: QK-Clip has negligible loss impact), muon-qknorm (extension: QK-Clip vs QK-norm, which no report ran).
  CPU: toy, 200 steps x 8 x 128, lr 1e-2, seeds 0-2; tau = round(0.6 x seed-0 baseline final max logit) (probe run: max
  logit 13.4 at step 200 -> tau 8; one run took 57 s). Margin 0.02 nats; decision on paired hierarchical bootstrap
  (seeds x 256 windows), seed t-interval reported too.
- Package format and checker: frontierlab/capstone/package.py (claim.yaml, contract.md, runs/<arm>-s<seed>/run_card.yaml,
  results.json, claims.yaml, report.md). Review/defence/revision log: review.py.
- Sample capstone for 20.2: frontierlab/capstone/sample.py (GSPO vs GRPO, constructed numbers, planted weaknesses) - TODO.
- Prerequisite id for Module 19: "19.3". Sidebar numbers 84-85.

## Verified claims (primary sources, 2026-10-07)
- GSPO 2507.18071: abstract (superior efficiency, stabilises MoE RL); Eq.5 objective, Eq.7 seq ratio (sec 4.1); ranges 3e-4/4e-4
  vs GRPO 0.2/0.27 (sec 5.1); Fig 1 cold-start from Qwen3-30B-A3B-Base, "proceeds stably"; Fig 2 clip fraction two orders of
  magnitude more (sec 5.2); Routing Replay needed for GRPO MoE, not GSPO (sec 5.1, 5.3, Fig 3).
- Kimi K2 2507.20534 sec 2.1: QK-Norm not applicable to MLA; gamma_h=min(1,tau/S_max^h); tau=100; 9B act/53B logits exceed 1000
  (Fig 2 left); Fig 2 right decays after ~30% steps (caption); 15.5T zero loss spike (abstract, Fig 3). App D Fig 12: two
  0.5B-act/3B-total MoE, tau=30, "negligible impact on loss", no stat. significant downstream degradation; 12.7% of heads
  triggered in first 70k steps, inactive after.
- DeepSeek-V4 2606.19348: 2.3.3 RMSNorm on query heads and KV entries; 2.4 "we do not employ the QK-Clip technique".
- V3.2 2512.02556: sec 2.2 no substantial degradation vs V3.1-Terminus (no table); 2.3 Fig 3 cost on H800 at USD 2/GPU-h;
  top-2048; O(L^2)->O(Lk); warm-up 1000 steps 16x128K 2.1B tokens lr 1e-3; sparse 15000 steps 480x128K 943.7B lr 7.3e-6;
  sparse-stage KL only over selected set (closes plan 14.1 gap); AA-LCR +4 in reasoning mode.
- mHC 2512.24880: HC loss surge ~12k step, 27B (sec 3.1 Fig 2); Amax gain peaks 3000 (Fig 3b); mHC max ~1.6 (5.4, Fig 7b);
  t_max 20, n=4; 3B/9B/27B; 6.7% overhead (4.3); 0.021 loss reduction vs baseline (5.2); Table 4 27B (BBH 43.8/48.9/51.0).
- OLMo 2 2501.00656 sec 4.4.2 Table 12: 50/50 candidate + general (DCLM) data, LR to zero; 19 microanneals 130B tokens (<3 full
  50B anneals); MMLU + GSM* (200 items). Olmo 3 2512.13961 sec 3.5.1: 5B target + 5B web vs 10B web-only baseline microanneal.
- Thinking Machines OPD (K. Lu, 2025-10-27): 9x (SFT data given), ~18x GPU-hours, ~30x incl. teacher sampling; Qwen3-8B-Base
  student, 400k prompts -> 60% AIME'24; OPD 70% in ~150 steps; SFT extrapolated ~2M prompts; per-token reverse KL.
  (Page also says Qwen3-8B was used as teacher in one experiment; June 2026 update: recipes moved to Qwen3.5-9B.)
- Earlier lessons cite micro-anneals via OLMo 2 (10.4), not Olmo 3. Course measured results per claim are in the lessons
  14.2, 07.2, 05.2/05.3, 06.2, 10.4, 13.2.
