# Module 4 project · Extend Baseline-0 from 2K to 32K and report what it can use

Baseline-0 was trained at 1,024 tokens and its config advertises 2,048 (`max_position_embeddings`). This project extends it to 32,768 tokens in two stages of continued training, with an equal-token control, and reports, with Eval Suite v1 and Eval v0, what the extended model can and cannot use, and what the extension cost at short context. It uses all three lessons: the evaluation and its validity checks (04.1), the RoPE rule and its per-frequency behaviour (04.2), and long-document data, staging, budget and controls (04.3). The extended model and Eval v1 are what Module 5 builds on.

**Time:** 6–8 attended hours plus unattended GPU time. **Folder:** [`labs/module-04/project/`](../labs/module-04/) (`run_project.py`, `buggy_report.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model, lengths and data | Hardware | Cost |
|---|---|---|---|
| Main path | your Module 1 Baseline-0 seed-0 run; 1K → 8K (400M tokens) → 32K (200M tokens), YaRN with original length 1,024 (factor 8, then 32), within-document windows from FineWeb-Edu documents of at least 8K and 32K tokens; control: 600M tokens at 1K | 1× H100 SXM 80 GB | **PROJECTED, pending the Module 4 pilot:** training FLOPs $= \sum_{\text{stage}} \text{tokens} \times$ `flops_per_token(cfg, T)`: $4 \times 10^8 \times 1.184 \times 10^9 + 2 \times 10^8 \times 2.543 \times 10^9 = 9.83 \times 10^{17}$ for the extension and $6 \times 10^8 \times 0.788 \times 10^9 = 4.73 \times 10^{17}$ for the control; at an assumed 30% MFU on 989 TFLOP/s that is $0.92 + 0.44 = 1.36$ GPU-hours, about 2 GPU-hours with evaluation and data preparation on the GPU machine, roughly USD 4–6 at USD 2–3 per H100-hour. Activation memory at 32K, batch 1: about 20–35 GiB, PROJECTED by scaling Module 2's per-token measurement linearly in T; use `--loss chunked` (built in) |
| Free GPU (Colab/Kaggle T4) | `base-t4` (pilot-10m at 512); 512 → 2,048 → 4,096; control 14.7M tokens at 512 | T4, fp32 | PROJECTED about 1.5 hours; you will not see whether a ~100M model uses 32K |
| Free CPU | `base-cpu` (toy at 256); 256 → 1,024 (1.64M tokens) → 2,048 (0.82M tokens), YaRN with original 256; control 2.46M tokens at 256 | laptop | measured: about 45 minutes (below) |

> [!IMPORTANT]
> The main path needs long *training* documents. With the vocabulary-32,768 Data-v0 prepared, run
> `python -m frontierlab.longctx.prepare_long --skip 2600000 --docs 9600000 --min-tokens 8192 --splits train val test --out labs/common/data/v0-long8k`
> (it streams the rest of FineWeb-Edu `sample-10BT` at Data-v0's pinned revision; hours of streaming and several GB; PROJECTED, not run in this build). It keeps Data-v0's hash split, so none of these documents can be in Data-v0's validation or test split, and no held-out document can be a training document. Check `python -m frontierlab.longctx.data --split train --data labs/common/data/v0-long8k` before stage 2: if the documents of at least 32,768 tokens hold fewer than about 200M tokens, stage 2 repeats them; say how many passes in your limits (projected from the CPU slice: on the order of 100–170M tokens in such documents). Then use `labs/common/data/v0-long8k` as the held-out set for Eval v1 too (`--data` of the suite, validation split).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. Fixed by the project:

- **Question:** after a two-stage extension of Baseline-0 to 32K, up to what length does it use its context (natural-text context gain and synthetic evidence effect), which tasks does it still fail (multi-hop, lexical decoys, middle positions), and what did it cost at 1K? Decision informed: whether this checkpoint becomes the 32K model of Module 5, and with which stated limits.
- **Hypothesis:** the far-context penalty of the unextended model disappears; context gain at 8K and 32K becomes positive; single-hop retrieval improves more than two-hop; held-out loss at 1K moves by less than 0.02 nats against the equal-token control. Status: reported effects at 7B+ scale (Llama 3 section 3.4.2; ProLong; YaRN section 4); may not appear at 125M parameters, and the CPU variant suggests a small model may use very little context at all.
- **Baseline:** Baseline-0 seed 0 (Module 1 run card), evaluated (a) unchanged and (b) with static YaRN factor 32 and no training (`base+yarn`).
- **Changed variable:** the two extension stages. **Controlled:** the base checkpoint (SHA-256 in each run card), seed 0, optimizer and schedule (AdamW, learning rate $10^{-3}$, 20 warmup steps, cosine), Eval v1 pins (`eval-v1.0`, seed 0, tokenizer hash, held-out folder), Eval v0 windows and LAMBADA file.
- **Comparison axis:** equal training tokens against the control (600M main path). Say what it does not answer: the extension used about twice the control's FLOPs.
- **Budget:** fill in GPU type, hours and cost, projected now (table above) and measured afterwards.
- **Metrics and decision rule:** primary: context gain at 32K with $W = 1{,}024$ (95% CI over documents); guard: held-out loss at 1K against the control, paired by window, upper bound $\leq 0.02$ nats. Effective length (natural text): the longest length whose gain at $W = 256$ has a lower bound $\geq 0$ at that length and every shorter one. Report retrieval and two-hop accuracy with chance and the evidence effect per depth. State your adoption rule in the same form as lesson 04.3's `decide`.
- **Correctness checks:** `pytest labs/common/tests/test_longctx.py` passes on the machine you train on; `check_eval.py` passes on the base model; each stage's run card has `longctx.init_sha256` equal to the SHA-256 of its parent's checkpoint; `frontierlab.record` shows the control and stage 1 differ only in the declared variables; stage 1 and 2 resumed exactly after any interruption (same command, `resumed from step` in the log).
- **Fallback evidence:** the Module 4 pilot traces (Colab, scaled) when published, labelled as analysis of provided traces.
- **Limits:** one seed; one data source (web text) without books or code; held-out documents of 32K tokens are few (count them); 125M parameters.

## Steps and deliverables

1. **Contract** (deliverable 1), before step 2.
2. **Data:** prepare long documents (main path, above). CPU: the `v0-long` held-out set from lesson 04.1 is enough; training uses Data-v0's own long documents.
3. **Train** (unattended; rerun the same command after any interruption):

   ```bash
   python labs/module-04/project/run_project.py --variant cpu
   python labs/module-04/project/run_project.py --variant main --base runs/m01/main/seeds-lr<chosen>-s0 --long-data labs/common/data/v0-long8k
   ```

   The main-path line is not run in this build (Module 4 pilot); add `--print` to see the three training commands. Fill in each run card's `measured:` block (wall-clock, GPU-hours, cost).
4. **Evaluate** (the script's `eval` stage): Eval v1 at the three lengths and Eval v0 at the original length for base, base+yarn, stage 1, stage 2 and control. For the main path also run the full suite at 1K, 2K, 4K, 8K, 16K and 32K on stage 2 with `--n 100` (`python -m frontierlab.evals.suite_v1 run ... --device cuda --bf16`).
5. **Report** (deliverable 2): one table "can use / cannot use": for each length, context gain, single-hop accuracy and evidence effect by depth, two-hop accuracy, all with intervals and chance; the effective length by your rule; the short-context regression against the base and against the control.
6. **Record** (deliverable 3): run cards of all three runs, the eval JSONs, the `prepare_long` `meta.json`, the record diffs.
7. **Debugging task** (deliverable 4), below.
8. **Written defence** (deliverable 5), below.

### What the free CPU variant gave in this build

Measured 2026-10-03 on the build laptop (16 threads, torch 2.14.1 CPU, other jobs running): stage 1 482 s, stage 2 319 s, control 439 s, evaluation of five models 1,460 s; about 45 minutes in all. This is a reference for what your output should look like, not a statement about Baseline-0. Context gain with $W = 256$ (95% CI over 100 documents); Eval v0 held-out loss at 256 (positive = worse), paired by window:

| Model | gain at 1,024 | gain at 2,048 | held-out at 256 vs base | vs control |
|---|---|---|---|---|
| base | −0.0657 [−0.0722, −0.0599] | −0.1326 [−0.1404, −0.1252] | — | — |
| base+yarn (factor 8, no training) | −0.0026 [−0.0041, −0.0012] | −0.0031 [−0.0041, −0.0021] | +0.056 [+0.053, +0.060] | — |
| stage 1 (1,024) | +0.0003 [−0.0005, +0.0011] | −0.0428 [−0.0469, −0.0391] | +0.014 [+0.005, +0.022] | +0.112 [+0.103, +0.121] |
| stage 2 (2,048) | +0.0005 [−0.0006, +0.0015] | +0.0013 [+0.0008, +0.0018] | +0.069 [+0.058, +0.080] | +0.168 [+0.155, +0.180] |
| control (256, same 2.46M tokens) | −0.0737 [−0.0802, −0.0679] | −0.1471 [−0.1551, −0.1394] | −0.098 [−0.103, −0.094] | — |

Retrieval accuracy stayed at chance at every length for every model (0.15–0.27 with chance 0.20); two-hop accuracy too (0.17–0.30, chance 0.25); the mean evidence effect at 2,048 rose from +0.006 (base) to +0.016–0.017 (stages 1 and 2).

What a write-up should say. Stage 2 is the only model whose far context *helps* at 2,048 (+0.0013 nats, interval above zero), a real but tiny effect; under the effective-length rule ($W = 256$, lower bound $\geq 0$ at every shorter length) its natural-text effective length is still below 1,024, because the 1,024 interval touches zero. Stage 1 fixed 1,024 and not 2,048, as expected for a model never trained there. The control is the striking row: 2.46M more tokens at 256 lowered held-out loss by 0.098 nats, so a 1.8M-parameter base after 6.1M tokens was far from converged and the extension's short-context "cost" against the control (+0.11 to +0.17) is mostly the improvement the extension tokens did not buy. Against the base alone, stage 1 would have looked nearly free (+0.014). That is the case the control exists for. On the main path, Baseline-0 is trained much closer to convergence and the numbers will differ; the method of reading them does not.

## Debugging task

`labs/module-04/project/buggy_report.py` is a colleague's evaluation of your stage-1 model. Their conclusion: "the training did little at long range (far-position loss only 0.03 nats better than the base with YaRN switched on), and it cost 0.085 nats of short-context loss, so extension is too expensive for us". Both numbers are wrong, because of two bugs. Start from the symptoms; for each bug, name the check that isolates it, run it, fix the bug and show the symptom gone.

```bash
python labs/module-04/project/buggy_report.py --base runs/m04/base-cpu --extended runs/m04/project/cpu-stage1 --original 256 --new 1024
```

<details>
<summary>Hint</summary>

For the long-range half: what RoPE rule was the stage-1 model *trained* with (its run card's `config.extra.rope`), and what rule does the script evaluate it with? Print both models' `inv_freq` for a few pairs. For the short-context half: what has to be identical for a paired short-context comparison, and what does `short_context_regression` check before it computes anything?

</details>

<details>
<summary>Reference diagnosis</summary>

1. **The extended model is evaluated with a RoPE rule it was not trained with.** The script "harmonises" both models to YaRN with `original_max_position_embeddings` set to the *new* length (1,024) and applies it to the extended model too, overwriting the rule in its config (`config.extra.rope`: factor 4, original 256). Rotations are then counted over 1,024 tokens, so YaRN leaves pairs unchanged that should be interpolated, for the base and the extended model alike. Check: print `layer.self_attn.rope.inv_freq` of the loaded extended model and of the converted one (they differ), or compare with `freq_table.py --dim 32 --train-len 256 --factor 4`. Fix: keep the extended model as loaded, and give `base+yarn` the trained length as `original_max_position_embeddings`. In the build's CPU run the far-position difference went from −0.033 to −0.051 nats (100 documents).
2. **The short-context check compares different windows at different lengths, unpaired.** The base is scored on 256-token windows and the extended model on 1,024-token windows (`window_losses(ext, val, 128, L1)`): different tokens, positions up to 1,023 (past the base's training length, and crossing document boundaries) and no pairing. Check: the suite's `short_context_regression` refuses this pair, because the Eval v0 pins (`T`) differ; or print the two window lists. Fix: both at the original length on the same fixed windows. In the build's run, with both bugs fixed the change was +0.015 nats instead of +0.085 (with only bug 1 fixed, +0.056, because bug 1 also changed the extended model). Each script run took about 45 s.

With both fixes the honest summary is the project's own: far-position loss 0.05 nats better than static YaRN, and a short-context cost to be judged against the equal-token control, not the base.

</details>

## Written defence

One to two pages, answering:

1. What is your model's effective length on natural text, by your pre-stated rule, and how does it compare with the 32K it now accepts? Which component of Eval v1 limits it?
2. Which of the five failures of lesson 04.1 does your report rule out, and which can it not rule out at this scale (for example multi-hop, or literal-match shortcuts)?
3. The extension used about twice the control's FLOPs at equal tokens. If a reviewer insists on an equal-FLOPs comparison, what run would you add, and what would it answer that yours does not?
4. How much of any short-context change is explained by the extra tokens (the control) and how much by the extension? Give the two numbers with intervals.
5. Your long training documents are web text only. Name one specific way this could bias what the model learned to use, and the evaluation result that would reveal it.
6. With 10× the budget, what would you change first: more tokens at 32K, a third stage, more seeds, or better long data? Justify with your own measurements.

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the adoption rule and effective-length threshold are written before the runs |
| 2 | Controls | run cards show the same base SHA-256, seed and schedule; the control differs only in length and rule |
| 3 | Axis and budget parity | equal tokens verified from `budget.tokens`; the FLOPs difference is stated |
| 4 | Correctness | Module 4 tests, `check_eval.py` and exact-resume evidence are included |
| 5 | Uncertainty | every number has an interval; one-seed limits are stated with the Module 1 noise floor |
| 6 | Conclusion matches evidence | "uses up to X" claims are tied to the context gain and evidence effect, not to NIAH-style accuracy |
| 7 | Limits | data source, document counts at 32K, scale and seeds |
