# Module 13 project · A post-training pipeline, stage by stage

A lab's post-training pipeline is a sequence of stages, each bought for a reason and each able to undo the last one. This project builds the whole open-recipe sequence, SFT → preference optimisation (DPO, from scratch) → RL with verifiable rewards → distillation, on the course's base model, with Eval Suite v2 after every stage and a random-reward control for the RL stage, then repeats a short form on your own Recipe-R checkpoint from Module 11. The deliverable is the stage table a recipe report should have published: what each stage bought, what it cost (FLOPs, teacher included), what it broke, and how sure you are.

**Time:** 6–8 attended hours plus unattended runs. **Folder:** [`labs/module-13/project/`](../labs/module-13/) (`pipeline_lab.py`, `buggy_pipeline.py`, `pieces.py`, `test_pieces.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model and stages | Hardware | Cost |
|---|---|---|---|
| Main path | Qwen3-1.7B-Base (revision `ea980cb`): `hf_stages sft` (1,000 steps × 32 Tülu 3 / OLMo 2 SFT examples), `dpo` (300 steps × 32 OLMo 2 1B preference pairs, length-normalised, NLL 0.2), `rlvr` (200 GRPO steps on GSM8K, k2 KL 0.05) and `rlvr --control random`, `distill --mode onpolicy` (100 steps, teacher Qwen3-8B); `hf_eval` after each stage, 2 seeds | 1× H100 80 GB | **PROJECTED, pending the Module 13 pilot:** SFT $6 \cdot 1.72 \times 10^9 \cdot 1.9 \times 10^7$ tokens $= 2.0 \times 10^{17}$ FLOPs (0.3–0.5 h with overheads), DPO 0.2–0.4 h, RLVR 2 × 200 steps × 30–50 s (lesson 12.2's measured-step projection) = 3.3–5.6 h, distillation 0.7–1.0 h, Eval v2 7 × 0.3–0.5 h: 7–11 GPU-hours per seed, 14–22 for two, USD 28–66 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | Qwen3-0.6B-Base (revision `da87bfb`), the same commands with `--batch 8`, `--max-new 256`, teacher Qwen3-1.7B (post-trained, same tokenizer), one seed | T4 16 GB | PROJECTED 6–10 hours across sessions; every stage writes its checkpoint, so rerun after a disconnect |
| Free CPU | the toy world (`pipeline_lab.py toy`) and the Module 11 CPU Recipe-R checkpoint (`pipeline_lab.py recipe-r`) | laptop | measured: toy 283 s for 2 seeds (teacher from lesson 13.2 reused); Recipe-R 463 s; `buggy_pipeline.py` 77 s, `--fixed` 57 s (another module's jobs running) |

> [!WARNING]
> Qwen base models are known to gain on math from RL with random rewards (Shao et al., lesson 12.1). On the main path the random-reward control from the DPO checkpoint is not optional: without it, a GSM8K gain from the RLVR stage cannot be attributed to the reward. If the control moves GSM8K, repeat the RLVR comparison on OLMo-2-0425-1B (`a1847dff`), the course's documented alternative.

## Deliverables

1. **The pipeline.** Your stage code (or `frontierlab.pipeline` with your lesson functions plugged in): every stage starts from the previous stage's checkpoint, writes a run card, a metrics file and a FLOP ledger, and is followed by Eval v2.
2. **The stage table** (the experiment contract below): Eval v2 after each stage, compared with the stage before it and with the SFT start, per seed, with paired intervals; the RLVR stage next to its random-reward control; each stage's FLOPs.
3. **The Recipe-R short form**: SFT (with your choice of pretraining-text replay, stated), DPO, a short RLVR and its control, with the pretraining validation loss as the retention component; one paragraph on why there is no distillation stage.
4. **The debugging report** (below).
5. **The written defence** (below).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running. Fixed by the project:

- **Question:** what does each stage of SFT → DPO → RLVR → distillation buy and cost on the base model, and is the RLVR stage's gain due to the verified reward? Decision informed: the stage list, order and settings you would carry into Module 14's RL experiments.
- **Hypothesis:** DPO with an NLL term improves the task and holds retention (lesson 13.1); RLVR improves pass@1 more than its random-reward control does (lesson 12.4); the final distillation stage recovers retention lost by RLVR (Thinking Machines' recovery experiment, lesson 13.2). Status: reported effects; their size at the course's scale is unknown; the random-reward effect on Qwen bases is reported (Shao et al.).
- **Baseline:** the SFT checkpoint for every stage's cumulative comparison; the previous stage for each stage's own comparison; the random-reward arm for RLVR.
- **Changed variable:** the stage. **Controlled:** the data per stage and seed, every hyperparameter stated in the variant table, Eval v2 pins (the same items, sampling seed, budget and prompt format for every checkpoint compared), seeds 0–1.
- **Comparison axis:** stage by stage, with the FLOPs of every stage reported (teacher sampling or scoring included; the teacher's own training stated separately).
- **Metrics and decision rule:** Eval v2's rule with guards 0.02 (toy) or the main path's guards (state them); keep a stage only if its task component is "improved" against the previous stage in both seeds and no guarded component's *mean* falls by more than its guard; treat a "regressed" verdict that comes from a wide interval around a non-negative mean as unresolved and say so.
- **Correctness checks:** `pytest labs/common/tests/test_pipeline.py` and `pytest labs/module-13/project` pass; the random-reward control logs the true pass rate (Module 12's loop does; the main-path monkeypatch does not, so judge it by Eval v2); every comparison's pins match (the suite refuses otherwise).
- **Fallback evidence:** Tülu 3 Table 6 and OLMo 3 Table 22 per-stage results (labelled as published).
- **Limits:** model size; one data mixture per stage; 2 seeds; Eval v2's item intervals are within a seed.

## What the free CPU variant gave in this build

Measured 2026-10-07 (torch 2.14.1+cpu, 8 threads, another module's jobs sharing the CPU). Toy pipeline, Eval v2 summaries (higher is better; `sft_nll` is minus the loss on the SFT data):

| Stage | add_pass1 (s0, s1) | add_greedy | sub_greedy | sft_nll | if_correct | vs previous: add_pass1 Δ [95% CI], guard |
|---|---|---|---|---|---|---|
| S0 SFT | 0.203, 0.203 | 0.305, 0.305 | 0.380, 0.380 | −0.504, −0.504 | 0.275, 0.275 | — |
| S1 DPO + NLL | 0.242, 0.246 | 0.380, 0.395 | 0.420, 0.475 | −0.471, −0.462 | 0.405, 0.438 | +0.039 [+0.021, +0.058] FAIL*; +0.043 [+0.021, +0.064] PASS |
| S2 RLVR (k2 KL 1.0) | 0.313, 0.329 | 0.425, 0.490 | 0.505, 0.580 | −0.497, −0.421 | 0.435, 0.443 | +0.071 [+0.056, +0.087] FAIL (sft_nll −0.026); +0.083 [+0.065, +0.102] FAIL* |
| S2c random reward | 0.247, 0.230 | 0.410, 0.340 | 0.385, 0.515 | −0.491, −0.453 | 0.390, 0.460 | +0.006 [−0.008, +0.019]; −0.016 [−0.026, −0.006] |
| S3 distillation | **0.471, 0.469** | 0.620, 0.715 | 0.715, 0.710 | **−0.312, −0.348** | 0.620, 0.583 | +0.158 [+0.128, +0.189] PASS; +0.141 [+0.111, +0.171] PASS |

\* "FAIL" from a wide interval around a non-negative mean (S1 seed 0: `sub_greedy` +0.040; S2 seed 1: `if_correct` +0.005), unresolved rather than a measured loss.

Each seed's pipeline cost $3.20 \times 10^{12}$ FLOPs: student training $1.50 \times 10^{12}$, teacher sampling $0.73 \times 10^{12}$, reference passes $0.57 \times 10^{12}$, student sampling $0.40 \times 10^{12}$ (the teacher's one-off training, $1.37 \times 10^{13}$, not included). Reading: every stage improved the task against the previous one in both seeds; RLVR's gain (+0.07, +0.08) is far outside its random-reward control (+0.006, −0.016), so the reward caused it; RLVR lost a little SFT-data likelihood in seed 0 (−0.026 nats) and the distillation stage more than recovered it (−0.31 against −0.50 at the start). Distillation from a near-perfect teacher on every task was the largest single gain, and it is the stage that depends on having that teacher.

Recipe-R short form (seed 0, the Module 11 CPU target checkpoint `target-m11-r5-x10`, 919K parameters):

| Stage | add_greedy | sub_greedy | val_nll (nats) | vs previous |
|---|---|---|---|---|
| R0 Recipe-R | 0.000 | 0.000 | −3.410 | — |
| R1 SFT, 25% replay, 1,500 steps | 0.665 | 0.435 | −3.543 | task improved; val_nll −0.133 [−0.146, −0.121], regressed (guard 0.05) |
| R2 DPO + NLL (2,988 pairs) | 0.610 | 0.560 | −3.550 | add −0.055 [−0.14, +0.035], sub +0.125 [+0.045, +0.205]; val_nll −0.006, held |
| R3 GRPO, 60 steps, lr 3e-5 | **0.730** | **0.635** | −3.550 | add +0.120 [+0.05, +0.19], sub +0.075 [+0.005, +0.15]; val_nll held |
| R3c random reward (from R2) | 0.030 | 0.260 | −3.551 | add −0.58 [−0.65, −0.51], sub −0.30; val_nll held |

Total $1.54 	imes 10^{13}$ FLOPs, almost all of it the SFT stage's $1.5 	imes 10^{13}$ (1,500 steps at 919K parameters, replay windows of 64 tokens included). The replay fraction is the stage's decision, and the build's pilot runs (one seed, same 1,500 steps, lr $3 	imes 10^{-4}$) bracket it: no replay reached 0.960 / 0.935 accuracy but drove the validation loss from 3.41 to 7.13 nats, the pretrained model's language ability destroyed; 50% replay kept the loss at 3.49 and learned almost nothing (0.075 / 0.125). At 25% the model learned the task for 0.13 nats. GRPO with the exact-match reward then added 0.12 on addition at no validation cost, while the random-reward control from the same DPO checkpoint collapsed the task: on this pretrained model random rewards do not help, unlike the Qwen-math reports. One build note on the method, in the run cards: GRPO at the toy loop's lr $3 	imes 10^{-4}$ collapsed Recipe-R with the *correct* reward as well (add 0.015), so the learning rate was lowered to $3 	imes 10^{-5}$ after a 40-step check at both rates, and that change is part of the record, not hidden.

## Debugging task

`labs/module-13/project/buggy_pipeline.py` is a colleague's refactor of three shared pieces: the pair builder, the distillation advantage and the example encoder. Their message is at the top of the file: everything runs, DPO's reward accuracy "looks fine", the distillation loss goes down, but the final model is worse than the SFT start and "sometimes never stops writing". There are three bugs. For each, start from the symptom, name the test or log value that isolates it, and show the fixed run.

```bash
python labs/module-13/project/buggy_pipeline.py              # their pieces: a short SFT -> DPO -> distil -> SFT run
PIPE_HOOKS=buggy pytest labs/module-13/project               # the piece tests against their versions
python labs/module-13/project/buggy_pipeline.py --fixed      # only after your diagnosis
```

<details>
<summary>Hint</summary>

Reward accuracy is measured against the pairs DPO was given, so it cannot tell you whether the pairs point the right way: compare the DPO stage's Eval v2 with the SFT start. For the distillation stage, look at whether `rkl` goes up or down while the loss goes down. For the last one, count how many greedy responses contain EOS.

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07 on the build laptop (77 s and 57 s). Their run: after DPO `add_pass1` 0.176 (SFT start 0.203) with reward accuracy 0.72; after distillation every task component 0.000, the sampled reverse KL rose from 11.6 to 19.0 nats while the loss fell to −19.0; after the final SFT 35% of greedy responses had no EOS. The reference pieces on the same run: DPO 0.236, distillation 0.329 with the reverse KL near 1.6–1.7, final 0.388 with every response finished. `PIPE_HOOKS=buggy pytest labs/module-13/project` fails all three piece tests.

**Bug 1: chosen and rejected swapped.** `make_pair` sorts the scores ascending and returns the first index as chosen: the *worst* candidate. DPO then learns to prefer worse responses, and its reward accuracy still rises, because it is measured against the pairs it was given. Isolating check: `test_pair_chosen_is_the_best_scored`; in the run, the DPO stage's task components fall while reward accuracy looks normal. Fix: chosen = argmax, rejected = argmin.

**Bug 2: the distillation advantage has the wrong sign.** `distill_advantage` returns $+(\log \pi_s - \log \pi_T)$, the reverse KL itself, so the surrogate's gradient *ascends* the KL: the student moves away from the teacher. The loss is the negative of the KL estimate, so "the loss goes down" exactly as the KL goes up. Isolating check: `test_distillation_advantage_lowers_reverse_kl` (exact enumeration against the reverse-KL gradient); in the run, `rkl_last` > `rkl_first`. Fix: the advantage is $-(\log \pi_s - \log \pi_T)$.

**Bug 3: no EOS on finished examples.** `sft_example` drops the EOS token, so SFT on those examples teaches the model to keep writing after the answer. Isolating check: `test_finished_examples_end_with_eos`; in the run, `no_eos_rate` 0.35 and a falling `if_strict` (0.54), which demands a finished response. Fix: append EOS when the response finished.

Which detector found what: bug 3 shows in a format metric, bug 2 in a logged KL that moves the wrong way, and bug 1 only through its unit test or a downstream evaluation: the training metric that should reveal it (reward accuracy) is computed against the bug.

</details>

## Written defence

One to two pages, answering:

1. For each stage, which Eval v2 component justifies keeping it, with its interval in both seeds? Which stage would you drop first under a smaller budget, and what would you lose?
2. Your RLVR stage against its random-reward control: what does the difference attribute to the reward, and what would you conclude if the control had moved the task as much (as it can on Qwen bases)?
3. Your distillation stage: off-policy or on-policy, and why for *this* task and response length? Count its teacher compute both ways (amortised and not).
4. The Recipe-R short form: what did each stage cost the pretraining validation loss, how did your replay fraction trade task accuracy against it, and why is there no distillation stage?
5. Which published recipe (Tülu 3, OLMo 3, Llama 3, Qwen3) is your pipeline closest to, and which piece of its evidence would you most want to replicate at the main-path scale?
6. Which result in your stage table would change if you had 5 seeds instead of 2, and how do you know?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the stage list, guards and the keep-a-stage rule are written before any run |
| 2 | Controls | each stage starts from the previous checkpoint; the random-reward arm starts from the same DPO checkpoint as RLVR; Eval v2 pins identical for every checkpoint compared |
| 3 | Axis and budget parity | every stage's FLOPs in a ledger, teacher sampling or scoring included, the teacher's own training stated separately |
| 4 | Correctness | `test_pipeline.py` and the piece tests pass; the three planted bugs found, each with its isolating check |
| 5 | Uncertainty | paired item intervals per stage and seed; unresolved "regressed" verdicts reported as such |
| 6 | Conclusion matches evidence | "the reward caused the RLVR gain" is claimed only against the control; distillation's gain is tied to having the teacher |
| 7 | Limits | model sizes, one mixture per stage, 2 seeds, the toy's shared-weights caveat, the Qwen random-reward caveat for the main path |
