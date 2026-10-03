# Module 3 project · Attention memo: GQA, MLA and local/global against Baseline-0

Your team must serve a model at long context under a fixed memory budget, and three attention designs are on the table: a cheaper GQA (half the key/value heads), MLA, and interleaved local/global layers. This project compares each against Baseline-0 in a controlled experiment — at equal parameters and at equal training FLOPs, on Eval v0, with seeds — measures what each costs at decode, and ends in a one-page decision memo for a stated serving constraint. It uses all of Module 3 (03.1–03.3; 03.4 optionally) and the methods of Modules 1 and 2: the experiment contract, the noise floor, paired comparisons, run cards and the benchmark method.

**Time:** 6–8 attended hours plus unattended GPU time. **Folder:** [`labs/module-03/project/`](../labs/module-03/) (scripts `run.py`, `evaluate.py`, `compare.py`, `buggy_decode.py`; decode measurements with `labs/module-03/decode_compare.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## The arms

All arms are Baseline-0 with one change to attention, defined relative to the preset (`labs/module-03/m03.py`), so the same names mean the same designs at every size:

| Arm | Change | KV cache per token, Baseline-0 shape, BF16 | Parameters |
|---|---|---|---|
| `b0` | none (GQA, 12 query heads, 4 KV heads of width 64, QK-norm) | 12,288 B | 96.75M non-embedding |
| `gqa-kv` | 2 KV heads | 6,144 B | −2.36M |
| `mla` | MLA, $d_c = K d = 256$, $d_r = d/2 = 32$; no QK-norm, latent RMSNorm instead (lesson 03.1) | 6,912 B | +6.19M |
| `local-global` | 5 local : 1 global, window $T/4 = 256$ (lesson 03.2) | at 32K: 2,128 B | same |

Two comparison axes (lesson 01.3), each answering a different question:

- **Part A — equal parameters, equal tokens.** Each arm's SwiGLU width is changed so its non-embedding parameters equal Baseline-0's within 0.01% (`gqa-kv`: 2,901; `mla`: 2,592; `local-global` unchanged). Question: per parameter, which design uses capacity best? It does not answer which is cheapest to train or serve.
- **Part B — equal training FLOPs.** The arms as designed, each trained for the number of steps whose training FLOPs equal Baseline-0's 9,500 (`gqa-kv` 9,674; `mla` 8,919; `local-global` 9,831; from `frontierlab.attention.accounting`). Question: for a fixed training budget, which design is best? It does not answer wall-clock (MLA's extra projections and our boolean-mask local layers are not FLOP-efficient kernels).

Decode memory and latency are a third, separate measurement: it does not depend on the weights, only on the design.

## The serving constraint for the memo

Use this one unless your team has a real one: **256 concurrent sequences of 32,768 tokens on one 80 GB GPU, BF16 weights and cache, 10% of memory kept free for activations and fragmentation.** The cache budget is then $(72 \text{ GB} - \text{weights}) / 256 \approx 280$ MB per sequence. By the formula (PROJECTED until you measure): Baseline-0 needs 402.7 MB (384 MiB) per sequence and does not fit; `gqa-kv` 201 MB, `mla` 226 MB and `local-global` 70 MB fit. So the memo's question is not "which saves memory" — three do — but which of the designs that fit loses least quality, and what each costs per decoded token.

## Variants and cost

| Variant | Setting | Hardware | Runs | Cost |
|---|---|---|---|---|
| Main path | `baseline0` on Data-v0 at the main-path size (vocab 32,768), $T = 1024$, 9,500 steps × 32 × 8 sequences (2.49B tokens) for Baseline-0; decode at 8K–32K | 1× H100 SXM 80 GB | 7 runs per seed (b0, 3 in part A, 3 in part B), 3 seeds = 21 | **PROJECTED, pending the Module 3 pilot:** $1.96 \times 10^{18}$ training FLOPs per run; at an assumed 30% MFU on 989 TFLOP/s, $1.96 \times 10^{18} / (0.30 \cdot 989 \times 10^{12}) / 3600 = 1.84$ GPU-hours per run, about 39 GPU-hours for 21 runs (MLA and boolean-mask local layers will run at lower MFU than Baseline-0; record your measured MFU), plus about 0.5 GPU-hours of evaluation and decode measurement; roughly USD 80–120 at USD 2–3 per H100-hour. Not run in this build |
| Free GPU (Colab/Kaggle T4) | `pilot-10m`, $T = 512$, 2,000 steps × 32, fp32 | T4 | 2 seeds = 14 runs | PROJECTED: about 10–15 T4-hours; use `--max-minutes` and rerun after disconnects. You will not see Baseline-0-size effects |
| Free CPU | `toy` (0.79M non-embedding), Data-v0 at the CPU size (vocab 8,192), $T = 128$, 200 steps × 16, learning rate 1.5e-3 (the Module 1 project's CPU choice) | laptop | 2 seeds = 14 runs | measured on the build laptop (16 threads, 2026-10-03, another build job sharing the CPU): about 60 minutes in total: 14 training runs 38 minutes, Eval v0 on all 14 (5,153 LAMBADA passages each) 21 minutes, `compare.py` seconds; the noise floor of a model this small says nothing about Baseline-0's |

Main-path commands, all **not run in this build; part of the Module 3 pilot**:

```bash
python labs/module-03/project/run.py --variant main --part A --seeds 0 1 2 --lr <Baseline-0's tuned lr>
python labs/module-03/project/run.py --variant main --part B --seeds 0 1 2 --lr <same>
python labs/module-03/project/evaluate.py runs/m03/main/* --device cuda --bf16
python labs/module-03/decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --train-seq 1024 \
    --arms b0 gqa-kv mla-absorbed local-global --contexts 8192 16384 32768 --batch 8 --rounds 30 \
    --out runs/m03/main-decode.json
python labs/module-03/project/compare.py runs/m03/main --margin 0.02 --context 32768 --concurrent 256 --gpu-gb 80 \
    --decode runs/m03/main-decode.json
```

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything. The fields below are fixed by the project; the rest are yours.

- **Question:** under the serving constraint above, which attention design should replace Baseline-0's, given its quality on Eval v0 and its decode cost? Decision informed: the attention block carried into Module 6's integration experiment.
- **Hypotheses and status:** (1) `gqa-kv` and `mla` are within 0.02 nats of Baseline-0's held-out loss on both axes — reported effects at much larger scale (GQA paper; DeepSeek-V2 for MLA against MHA), may not appear at this scale; (2) `local-global` is within 0.02 nats on Eval v0 — reported at 2B (Gemma 3 Figures 3–4), but Eval v0 cannot see what a 256-token window loses beyond it; (3) cache bytes equal the formula exactly — established; (4) decode latency ranks by bytes read per step on the GPU — expected, memory-bound regime.
- **Baseline:** Baseline-0's run cards from the Module 1 project (same data, same tuned learning rate, same steps), retrained here with this module's code so the run cards are comparable (check with `python -m frontierlab.record`). Tuning budget: none for any arm beyond Baseline-0's learning rate — the arms get the same budget as Baseline-0 gets here (zero extra), and you must say so in the limits.
- **Changed variable:** the attention block (and, in part A, the SwiGLU width that keeps parameters equal; in part B, the number of steps). For `mla` the block change includes dropping QK-norm, as DeepSeek's design does — name it.
- **Controlled:** Data-v0 hashes, tokenizer, preset width, depth and query heads, sequence length, batch, learning rate and schedule shape, seed set {0, 1, 2} shared by every arm, Eval v0 version, window seed and LAMBADA revision, software versions.
- **Comparison axes:** part A equal parameters (with equal tokens); part B equal training FLOPs. Decode: equal model shape at equal context and batch.
- **Budget:** from the table above, projected now, measured in the run cards afterwards.
- **Metrics and decision rule:** primary — held-out loss on Eval v0's 256 validation windows; per seed, the paired mean difference against Baseline-0 of the same seed; across seeds, the mean difference with a 95% t interval. Secondary — LAMBADA target log-probability per passage, the same way; decode cache bytes (measured and formula) and decode-step latency at 32K with interleaved rounds. Rule, stated now: an arm is **non-inferior** if the upper end of its seed-level interval is below +0.02 nats (CPU variant: +0.05, the toy noise floor is larger), **inferior** if the lower end is above the margin, otherwise **inconclusive**. The memo recommends, among arms that fit the serving constraint and are non-inferior on both axes, the one with the lowest measured decode latency; if none is non-inferior, it says so and recommends the next experiment instead of a design.
- **Noise floor:** Baseline-0's seed standard deviation on held-out loss from the Module 1 project; your minimum detectable effect with 3 seeds (lesson 01.4). If the MDE is above the margin, state before running that the rule cannot return "non-inferior" with confidence and what you will do (more seeds, or report as underpowered).
- **Correctness checks (before any result counts):** `pytest labs/common/tests/test_attention_m03.py` (every kind: op-level float64 gradient check, causal check, cached decode in 1-token and chunked steps, MLA naive vs absorbed); run-card diffs between each arm and Baseline-0 of the same seed show only the declared changes (`python -m frontierlab.record runs/m03/main/A-b0-s0 runs/m03/main/A-mla-s0 --changed config.attention config.extra config.intermediate_size --axis params`, and `--axis flops` for part B); measured cache bytes equal the formula.
- **Fallback evidence:** the Module 3 pilot traces (30M and 70M, 4 arms × 2 seeds, decode memory at 8K–32K), labelled as analysis of provided traces.
- **Limits:** one model size, one dataset, one context length for training, no long-context quality evaluation (Module 4), untuned arms, our implementations' kernels.

## Steps and deliverables

1. **Contract** (deliverable 1), written before step 2.
2. **Correctness.** Run `pytest labs/common/tests/test_attention_m03.py` and keep the output.
3. **Train** part A and part B with `run.py`. Fill in the `measured:` block of each run card (wall-clock, GPU-hours, cost, MFU).
4. **Run-card diffs** for every arm against Baseline-0 of the same seed (deliverable 2): only the declared fields may differ; part B must pass `--axis flops`.
5. **Evaluate** every run with `evaluate.py` (validation split).
6. **Decode measurement** with `decode_compare.py` (deliverable 3): cache bytes against the formula, latency with intervals at 8K, 16K and 32K, batch 8 on the main path.
7. **Compare** with `compare.py` (deliverable 4): the per-axis verdicts and the serving-constraint table.
8. **Debugging task** (deliverable 5): below.
9. **Decision memo** (deliverable 6): one page — the constraint, the recommendation (or "no recommendation, run X next"), the evidence for and against with intervals, what was projected rather than measured, and the conditions that would change the answer.
10. **Written defence** (deliverable 7): below.
11. **Test once.** After the memo is written: `evaluate.py RUN --split test` for the chosen arm and Baseline-0, reported alongside.

### What the free CPU variant gave in this build

Measured 2026-10-03/04 (toy preset, 200 steps, seeds 0 and 1, Windows 11 laptop, 16 threads, torch 2.14.1+cpu, another build job sharing the CPU), as a reference for what your output should look like, not as Baseline-0 numbers. Baseline-0: held-out loss 6.681, seed std 0.007 (two seeds). Arm minus Baseline-0 on held-out loss, mean over the two seeds with the 95% t interval over seeds (1 degree of freedom, $t = 12.7$), and the per-seed window-bootstrap intervals:

| Arm | Part A (equal parameters) | Verdict | Part B (equal FLOPs) | Verdict |
|---|---|---|---|---|
| `gqa-kv` | −0.026 [−0.162, +0.109]; s0 −0.016 [−0.022, −0.010], s1 −0.037 [−0.042, −0.032] | inconclusive | −0.036 [−0.497, +0.426]; s0 −0.072, s1 +0.001 | inconclusive |
| `mla` | −0.075 [−0.144, −0.007]; s0 −0.081, s1 −0.070 | non-inferior | −0.074 [−0.191, +0.043]; s0 −0.083, s1 −0.065 | non-inferior |
| `local-global` | +0.021 [−0.146, +0.189]; s0 +0.008, s1 +0.035 | inconclusive | +0.009 [−0.135, +0.152]; s0 −0.003, s1 +0.020 | inconclusive |

LAMBADA accuracy is 0 for every run; the target log-probability moves with held-out loss (MLA +0.22 per passage in both parts, local/global −0.06 in part A). Three lessons from this output. First, with two seeds the seed-level interval is wide enough that only a large, consistent effect (MLA's −0.07 to −0.08, both seeds agreeing) can pass the rule; the per-seed window intervals are far narrower and would have declared every arm "significant" — they measure evaluation noise, not seed noise (lesson 01.4). Second, MLA's toy advantage is not a statement about Baseline-0: at toy size MLA also differs in normalisation (no QK-norm, a normed latent) and in head capacity, and the effect could shrink or reverse with scale. Third, the serving-constraint table `compare.py` prints for the toy model fits everything (its cache is tiny); for the memo, use the Baseline-0 numbers in the constraint section above and the main-path decode measurement. On this CPU the decode measurement of lesson 03.1 favours `gqa-kv` (1.37× faster than Baseline-0 at 16K) over `mla` (0.61×), so the CPU memo would read: MLA is the only arm shown non-inferior, but decodes slowest on this hardware; recommendation withheld pending the GPU decode measurement and three seeds.

## Debugging task

`labs/module-03/project/buggy_decode.py` is a colleague's decode benchmark. Their summary: "MLA and local/global save no decode memory at all, and every arm's latency drifts upward during the run, so the timings are too noisy to use." There are two bugs. Start from the symptoms: for each, name the check that isolates it, run it, fix the bug and show the symptom gone (compare with `decode_compare.py`).

<details>
<summary>Hint</summary>

For the memory symptom: where does the script's memory number come from — the cache, or a formula? Which formula? For the drift: print `cache.length` for one arm before and after the timing loop.

</details>

<details>
<summary>Reference diagnosis</summary>

1. **Memory from the wrong formula.** The script reports `frontierlab.flops.kv_bytes_per_token(cfg)`, which is Baseline-0's GQA formula ($2 L K d$) and ignores `cfg.attention`: every arm gets the same 1.00 MiB at 512 tokens in fp32. Check: compare with `Cache.nbytes()` of the real cache, or `frontierlab.attention.accounting.kv_bytes`. Measured in this build (toy preset, fp32, $S = 512$): Baseline-0 1.00 MiB, MLA 0.62 MiB, local/global (one global layer, three local layers of 32) 0.30 MiB, plus position bookkeeping.
2. **The context grows during timing.** Each timed call appends a token to the cache and nothing restores it, so 400 rounds run at contexts from 512 to 911: the "drift" is the arms getting slower as their caches grow (in this build: Baseline-0 9.6 ms over the first 50 rounds against 18.5 ms over the last 50). Every sample is at a different context, and the global layers of the local/global arm grow too. Fix: restore the cache after every timed step, as `frontierlab.attention.bench` does. A third, smaller flaw: no warm-up rounds, so the first samples include one-off costs.

</details>

## Written defence

One to two pages, answering:

1. Part A and part B can rank the arms differently. If they do in your results, which one should the memo follow for this decision, and why? If they do not, would you expect them to at 32K training context?
2. Your seed-level interval for `mla` on held-out loss: what is your MDE with your seed count, and could the rule ever have returned "non-inferior" at the margin you chose? What would you change in the design to make it able to?
3. MLA's arm drops QK-norm along with the attention change. How would you separate the two effects, and is it worth the budget for this decision?
4. `local-global` may pass Eval v0 and still fail at long range. What is the cheapest experiment that would tell you, and why is it not part of this project?
5. Which of your decode numbers are measured and which are projected? For one projected number, show the formula and say what measurement would replace it.
6. With 10× the budget, what would you run next, and what result would make you abandon your recommendation?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the serving constraint and the decision rule are written before the runs |
| 2 | Controls | every arm's run-card diff against Baseline-0 shows only the declared changes |
| 3 | Axis and budget parity | part A's parameters match within 0.1%; part B's FLOPs within 1% (`--axis flops` passes); tuning budgets equal and stated |
| 4 | Correctness | the Module 3 test suite output is included; cache bytes equal the formula |
| 5 | Uncertainty | seed-level intervals for every arm and metric; the noise floor and MDE stated |
| 6 | Conclusion matches evidence | the memo's recommendation follows the rule; projected decode numbers are labelled; provided traces labelled as analysis |
| 7 | Limits | scale, untuned arms, no long-context evaluation, kernel immaturity, and what would change the answer |
