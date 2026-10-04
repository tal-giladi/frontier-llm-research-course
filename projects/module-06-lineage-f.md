# Module 6 project · Lineage-F integration experiment

Modules 3 to 6 each tested one architecture change against Baseline-0. This project puts the ones that passed their own correctness suites and tests into one model — Lineage-F — and asks the question an integration run must answer before it is adopted: *does the combination beat its best single branch at the same training budget, and which components earn their place inside it?* It is an experiment, not an assumed upgrade: a null or negative result is a sound outcome, and it decides the architecture carried into Module 11's pre-registered run (or keeps Baseline-0's). It uses all three required lessons of this module — MTP and its cost (06.1), the mHC residual and its wall-clock cost (06.2), interactions and the integration budget (06.3) — and the methods of Modules 1–3: the contract, the noise floor, selection on validation and a single test, run cards, and the attention branches.

**Time:** 6–8 attended hours plus unattended GPU time. **Folder:** [`labs/module-06/project/`](../labs/module-06/) (`run_project.py`, `buggy_report.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## The components

| Component | From | Setting | Its own evidence in this course |
|---|---|---|---|
| MLA attention | lesson 03.1, Module 3 project | $d_c = K d$, $d_r = d/2$, no QK-norm | passes the attention suite; Module 3 memo |
| DeepSeek MTP | lesson 06.1 | $D = 1$, λ 0.3 → 0.1 at 10/14.8 of training | passes causal, cache, gradcheck, lossless drafting |
| mHC residual | lesson 06.2 | $n = 4$, $t_{\max} = 20$ | Sinkhorn, n = 1 reduction, gradcheck, gains ≈ 1 |
| fine-grained MoE FFN | the MoE course's `moelab`, copied into `frontierlab/blocks/moe.py` (credited there) | toy: 8 routed experts (width $I/4$), top-2, 1 shared expert, layers 1..L−1; main path: 16 routed experts | equals a dense loop reference; gradcheck; causal and cache |

The MoE layer is a component here, not a topic: routing, balancing, fast expert execution and expert parallelism are taught in the sibling course *Mixture-of-Experts Engineering: Design, Train, Post-train and Serve Sparse LLMs* (repository `tal-giladi/mixture-of-experts-engineering-course`); its `moelab` package is the source of the copied layer. Long-context and sub-quadratic branches (Modules 4–5) are left out of the main design because their decision axis is long-context cost and Eval v1, not the short-context loss this project measures; adding one is the first extension below.

All four live in one `BlockLM` (`frontierlab.blocks`): `--attention mla --mtp deepseek --residual mhc --ffn moe`. Before any comparison the combination passes the same suite as its parts (`pytest labs/common/tests/test_blocks.py -k causal` includes `mhc + deepseek MTP + MoE` over MLA attention).

## Sizes and cost

| Variant | Lineage-F model | Budget | Hardware | Cost |
|---|---|---|---|---|
| Main path | Baseline-0's shape ($C = 768$, 12 layers) with all four components and 16 routed experts of width 704: **371.8M total parameters, 87.1M active non-embedding** (`frontierlab.blocks.accounting`, exact); 0.963 GFLOP/token in training vs Baseline-0's 0.788 | Baseline-0's 2.49B tokens (9,500 × 32 × 8 × 1,024), Data-v0 at vocabulary 32,768 | 1× H100 SXM 80 GB | **PROJECTED, pending the Module 6 pilot.** One combination run: $0.963 \times 10^9 \cdot 2.49 \times 10^9 = 2.4 \times 10^{18}$ FLOPs; at an assumed 20% MFU (an MoE executed as an expert loop and an unfused mHC run below Baseline-0's MFU) $2.4 \times 10^{18} / (0.2 \cdot 989 \times 10^{12}) \approx 3.4$ GPU-hours, times the measured mHC wall-clock factor (lesson 06.2). The plan's feasibility table budgets ~25 GPU-hours per 350M run, a conservative ceiling. 18 runs (below): about 60 GPU-hours by FLOPs, up to 150+ with wall-clock factors — USD 120–450 at USD 2–3 per H100-hour. Not run in this build |
| Pilot ladder (plan 12.1) | pilot-30m and pilot-70m with the same components | 4,000 steps × 64 × 1,024 | Colab A100/L4, or 1× H100 | PROJECTED: the 20 runs at pilot-30m (18 planned + 2 best-single seeds), summed per arm from `blocks.accounting` (0.45 GFLOP/token for the combination) × 262M tokens each, ≈ $2.0 \times 10^{18}$ FLOPs ≈ 2.2 GPU-hours at 25% MFU on an H100, more on an A100 and with mHC's wall-clock factor; pilot-70m about twice that |
| Free GPU (Colab/Kaggle T4) | pilot-10m | 2,000 steps × 32 × 512 | T4, fp32 | PROJECTED 8–12 T4-hours over several sessions (`--max-minutes`, rerun to resume) |
| Free CPU | toy (2.82M total, 0.86M active non-embedding) | 200 steps × 16 × 128 | laptop | measured below |

> [!IMPORTANT]
> The main path needs Data-v0 at the main-path size (vocabulary 32,768, Module 1 project) and Baseline-0's tuned
> learning rate: `run_project.py --variant lineage --lr <that value>`. Every main-path and pilot command was **not
> run in this build**; `--print` lists them for the pilot notebook.

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. Fixed by the project:

- **Question:** at Baseline-0's budget, does Lineage-F (MLA + DeepSeek MTP + mHC + fine-grained MoE) reach a lower held-out loss than its best single branch *at equal training FLOPs*, and which components contribute inside the combination? Decision informed: the architecture of Recipe-R (Module 11).
- **Hypotheses and status:** H1: the combination beats the best single branch at equal FLOPs — **unknown; components' published gains were each measured alone and at much larger scale**, and lessons 06.1–06.2 found MTP rejected at equal FLOPs and mHC's benefit unresolved at toy scale, so H1 may fail. H2: interactions are non-zero (the combination misses the additive prediction) — no published measurement.
- **Baseline and references:** Baseline-0 trained for the steps whose training FLOPs equal the combination's (toy: 351 instead of 200; main path: 11,604 instead of 9,500) — "Baseline-0 scaled to the same budget". The *best single branch*, also at the combination's FLOPs, chosen on validation windows in the `select` stage. Tuning budget: none for any arm beyond Baseline-0's learning rate (stated as a limit; lesson 07.3 says the best rate may move).
- **Changed variables (declared, factorial-lite):** the four components; their settings are fixed in advance (table above).
- **Controlled:** Data-v0 and hashes, tokenizer, preset shape, tokens per step, schedule shape, seeds {0, 1}, data order, Eval v0's window seed, software and hardware type.
- **Comparison axis:** **equal training FLOPs** for the decision (the components change training FLOPs per token on Baseline-0: MTP +27%, MLA +6.5%, mHC +1.8%, MoE −13% active). Equal tokens for the attribution stage (component effects inside vs outside the combination, as in lesson 06.3). Wall-clock is measured and reported for every run; if the combination wins at equal FLOPs but not at equal wall-clock, the report says so — the decision then depends on whether kernels for MoE and mHC exist on the target system.
- **Budget:** from the table, projected now, measured in run cards afterwards (`measured:` block).
- **Metrics and decision rule:** held-out next-token loss of the main head on 256 fixed windows. *Selection* of the best single branch on **validation** windows (seed 0); the *decision* once on **test** windows, seeds 0 and 1, paired by window. Rule, stated now: adopt Lineage-F if the paired 95% interval of `combination − best single` (equal FLOPs, test) has its upper bound below 0 and both per-seed differences are negative; do not adopt if its lower bound is above 0; otherwise inconclusive — then carry the simpler choice (the best single branch if it beats Baseline-0 at equal FLOPs by the same rule, else Baseline-0).
- **Correctness checks:** `pytest labs/common/tests/test_blocks.py` and `pytest labs/common/tests/test_attention_m03.py` pass on the training machine; `python -m frontierlab.record <b0 equal-FLOPs run> <combination run> --changed config.attention config.extra --axis flops` shows no INVALIDATES line other than declared ones; budget parity from `budget.train_flops` within 1%.
- **Fallback evidence:** the components' published results (DeepSeek-V3 Table 4, mHC Table 4, DeepSeek-V2 for MLA, DeepSeekMoE) are evidence at other scales, labelled as such; the pilot ladder's traces once published (analysis of provided traces).
- **Limits:** one data set, one budget, one learning rate, two seeds; MoE executed by an expert loop (its wall-clock is not representative); no long-context evaluation; results at 2.8M or 372M parameters do not settle the question for frontier models.

## Steps and deliverables

1. **Contract** (deliverable 1), before step 2.
2. **Plan and budget:** `python labs/module-06/project/run_project.py --variant cpu --print` (and `--variant lineage --lr ... --print` for the main path). Check the equal-FLOPs step counts against `frontierlab.blocks.accounting` by hand for one arm.
3. **Run** (unattended; every stage resumes, finished runs are skipped, lessons' runs are reused):

   ```bash
   python labs/module-06/project/run_project.py --variant cpu
   ```

   Stages: `select` (4 runs), `decide` (6 runs), `attrib` (up to 8, of which the MTP and mHC singles come from lessons 06.1–06.2), `report`.
4. **Report** (deliverable 2): the selection table (validation); the decision table on test with per-seed values, paired intervals, the noise floor and the decision by the rule; the attribution table (each component alone and inside, total interaction, additive prediction vs measured); the cost table (training FLOPs and measured wall-clock per arm); a paragraph on what changes at equal wall-clock.
5. **Record** (deliverable 3): run cards of every run, `runs/m06/<variant>/lineage_f_report.json`, the record diffs of the correctness step.
6. **Debugging task** (deliverable 4), below.
7. **Written defence** (deliverable 5), below.

### What the free CPU variant gave in this build

Measured 2026-10-04 on the build laptop (16 threads, torch 2.14.1+cpu, another build job sharing the CPU): 16 new runs in 55 minutes (the combination about 7.5 minutes per run, 450 s for seed 0; the MTP and mHC singles and Baseline-0 reused from the lessons). This is a reference for what your output should look like, not a statement about Lineage-F at scale.

| Stage | Result |
|---|---|
| select (validation, seed 0, each single at the combination's 8.2 × 10¹² training FLOPs) | MLA 6.2737 (335 steps), MoE 6.2908 (372), mHC 6.3653 (328), MTP 6.6427 (209) → best single: **MLA** |
| decide (test, seeds 0 / 1, equal training FLOPs) | combination 6.6341 / 6.6223; Baseline-0 at 351 steps 6.4013 / 6.3532; MLA at 335 steps 6.3425 / 6.3392 |
| combination − Baseline-0 (equal FLOPs) | **+0.2509 [+0.2434, +0.2584]**, per seed +0.233, +0.269 |
| combination − best single (equal FLOPs) | **+0.2873 [+0.2793, +0.2953]**, per seed +0.292, +0.283 → **decision: do not adopt** |
| noise floor (Baseline-0 at 351 steps, two seeds) | seed std 0.034, MDE 0.095 |
| attribution (validation, seed 0, equal tokens): effect alone / inside the combination | MLA −0.114 / −0.082; MTP −0.011 / −0.007; mHC −0.022 / −0.009; MoE −0.022 / −0.004 |
| additive prediction vs measured combination (equal tokens) | 6.5175 vs 6.5687 (+0.051: the effects overlap) |

What a write-up should say. At equal tokens the combination is clearly better than Baseline-0 (6.569 vs 6.686 on validation), and every component helps on its own — which is exactly the situation in which "best of each branch" looks convincing. At the decision axis it loses by 0.29 nats to MLA alone, because the combination spends 75% more FLOPs per token, and in the steep first 200 steps of training 135 extra steps of a single-change model buy more than four changes at once. Every component's effect is smaller inside the combination than alone (all four total interactions positive: +0.032, +0.004, +0.013, +0.018), so the additive prediction overstates the combination by 0.05 nats. Two cautions: two seeds give a wide noise floor (MDE 0.095), and the decision is decisively outside it; and the test windows score every model higher than validation (Baseline-0 6.74 vs 6.69), which is why selection and decision use different sets. None of this says how Lineage-F behaves at 372M parameters and 2.49B tokens — that is the main path's question.

## Debugging task

`labs/module-06/project/buggy_report.py` is a colleague's version of the report stage. It prints "the combination beats Baseline-0 and its best single branch at the same budget. Adopt it." It contains planted bugs in which runs it compares and how it picks them. Start from the symptoms: for each printed comparison, write down which run directories it reads, check them with `python -m frontierlab.record` (what do their `budget.train_flops` say?), fix the script, and say whether the conclusion survives.

```bash
python labs/module-06/project/buggy_report.py
```

<details>
<summary>Hint</summary>

Print `m06.run_dir(...)` for every arm the script loads. Which axis does "the same budget" hold equal for the Baseline-0 run it uses? On which windows is the best single branch chosen, and on which is it then compared?

</details>

<details>
<summary>Reference diagnosis</summary>

1. **"Baseline-0 at the same budget" is the 200-step Baseline-0** (equal tokens), not the 351-step run with the combination's training FLOPs. Check: `python -m frontierlab.record runs/m06/cpu/b0/s0 runs/m06/cpu/mla+mtp-ds+mhc+moe/s0 --changed config.attention config.extra --axis flops` flags `budget.train_flops` (4.7e12 vs 8.2e12). Fix: use `run_dir("cpu", "b0", s, steps=351)`.
2. **The best single branch is chosen on the test windows it is then reported on**, and from the 200-step (equal-tokens) singles, not the equal-FLOPs ones. Selecting the minimum of four noisy test scores and reporting the winner on the same windows is selection on the test set (lesson 01.3, lesson 06.3's winner's curse). Fix: select on validation among the equal-FLOPs singles (the `select` stage), then evaluate once on test.
3. A subtler one: the combination is averaged over seeds 0 and 1, Baseline-0 and the singles use seed 0 only, so the pairing mixes seed-averaged and single-seed numbers and the interval ignores seed noise. Fix: the same seeds for every arm in the decision.

With the fixes the report is the one `run_project.py --stages report` prints. In this build's CPU runs the buggy script prints combination − Baseline-0 = −0.112 and combination − best single = −0.006 ("adopt"); the correct equal-FLOPs numbers are +0.251 and +0.287 ("do not adopt").

</details>

## Written defence

One to two pages, answering:

1. Why equal training FLOPs for the decision and not equal tokens or equal wall-clock? Which one would you choose if MoE and mHC had to run with the kernels you have today, and how would the answer change?
2. How do you know the best single branch was chosen fairly? What would have happened to your decision if you had chosen it on test?
3. What is the smallest effect your decision could detect (MDE with two seeds)? What is the smallest *interaction* the attribution stage could detect?
4. Which component's effect inside the combination differs most from its effect alone? Give one mechanistic reason it could overlap with or depend on another component (for example MLA and MTP both changing what the last layers represent; MoE adding capacity that mHC's wider stream also adds).
5. Lessons 06.1–06.2 found MTP rejected at equal FLOPs and mHC unresolved at toy scale. Should those components have been left out of Lineage-F? Argue both sides, then state the rule you would use for the next integration.
6. With 10× the budget: more seeds, the pilot ladder, a learning-rate sweep for the combination, or the full factorial? What result would make you abandon Lineage-F for Recipe-R?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the adoption rule, the reference (best single at equal FLOPs) and the axis are written before the runs |
| 2 | Controls | run cards show the same data hashes, seeds, schedule and windows; the factorial-lite cells are declared |
| 3 | Axis and budget parity | equal FLOPs verified from `budget.train_flops`; wall-clock measured on one machine and reported separately |
| 4 | Correctness | `test_blocks.py`, the combination's causal and cache checks and the record diffs are included |
| 5 | Uncertainty | paired intervals, per-seed values, the noise floor, the MDE for effects and for interactions |
| 6 | Conclusion matches evidence | selection on validation, one test evaluation; "adopt" only where the rule says so |
| 7 | Limits | scale, one learning rate, loop-executed MoE, no long-context evaluation, what would change the answer |
