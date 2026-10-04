# Module 9 project · An infrastructure plan for a 1T-parameter MoE model

How would you train a 1-trillion-parameter mixture-of-experts model with about 40B active parameters on a stated cluster, how long would it take, what would it cost, and how much of the time would be lost to failures? This project answers that the way an infrastructure review at a lab would expect: a parallel layout chosen from estimates of memory and communication per GPU, every assumption that can be checked on one node checked against your own measurements from lessons 09.3 and 09.4, every other number taken from a cited source or labelled as an assumption, and a checkpointing and recovery scheme with its goodput. It uses the four required lessons: layouts (09.1), pipeline schedules (09.2), the measured node investigation (09.3) and failure and recovery (09.4); the extension 09.5 is useful for the storage and hardware discussion.

**Time:** 6–8 attended hours (plus the 09.3 and 09.4 runs, which you reuse). **Folder:** [`labs/module-09/project/`](../labs/module-09/) (`plan_1t.py`, `buggy_plan.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## The stated model and cluster

| | Fixed by the project |
|---|---|
| Model | **M9-1T**, a course construct: Kimi K2's published configuration (61 layers, width 7,168, MLA attention, 384 routed experts of width 2,048, first layer dense; `moonshotai/Kimi-K2-Instruct` config snapshot) with 10 routed and 2 shared experts per token. 1.03T total, 40.8B active parameters, about 253 GFLOPs per training token at 4,096 tokens (`frontierlab.calc`). |
| Run | 15T tokens at 4,096 tokens per sequence, about 16M tokens per optimizer step (you may change both; say why). |
| Cluster | 2,048 H100 SXM 80 GB in 256 nodes of 8 (NVLink inside a node), one 400 Gb/s InfiniBand NIC per GPU, a shared parallel file system. |
| Precision | BF16 with FP32 master weights; AdamW moments in BF16 as DeepSeek-V3 does (lesson 08.2); FP8 activation caching allowed. Your Module 8 precision plan may replace this if you justify it. |

| Variant | What grounds the node-scale assumptions |
|---|---|
| Main path | your torchtitan measurements on one 8× H100 node (09.3: MFU, tokens/s, NVLink bus bandwidth, exposed communication, memory per rank) and your recovery measurements on GPUs (09.4: checkpoint blocking time and size, restart and load time) |
| Free variant (CPU) | your CPU-process measurements from 09.3 and 09.4 (labelled CPU: they ground the *mechanisms* — memory per rank matches the formula, recovery is exact, async save blocks less than sync — not GPU magnitudes) plus published GPU numbers, labelled PUBLISHED |

> [!IMPORTANT]
> The main-path measurements were not run in this build (multi-GPU is not piloted course-side). Every GPU number below
> that is not yours is PUBLISHED (with its source) or ASSUMED (with your reason). The plan's totals are always PROJECTED.

## The experiment contract

A plan is a set of predictions; the contract says how each one is grounded. Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running `plan_1t.py` with your inputs. Fixed by the project:

- **Question:** which layout, schedule, checkpoint scheme and interval let M9-1T train on the stated cluster with the highest projected goodput-adjusted throughput, and what are the run's GPU-hours, wall-clock and cost? Decision informed: whether to request this cluster, a larger one, or a different model shape.
- **Hypotheses and status:** (H1) a no-TP layout with PP and EP (DeepSeek-V3 style) fits in 80 GB per GPU only with FP8 activation caching or recomputation — the planner's estimate, to be checked against your 09.3 memory measurement at node scale; (H2) cross-node expert all-to-all is the largest communication term and needs overlap (DualPipe-style) to stay hidden — published for DeepSeek-V3 (section 3.2.1), projected here; (H3) with asynchronous checkpoints the goodput on 2,048 GPUs exceeds 95% — projected from Llama 3's failure rate.
- **Baseline:** layout A of `plan_1t.py` (PP16 × EP64 × ZeRO-1, no TP, DualPipe), the documented DeepSeek-V3 layout applied to this model.
- **Changed variables:** the layout (A, B, C, or yours), the activation precision (BF16 vs FP8 caching), the checkpoint mode (sync vs async) and interval.
- **Controlled:** model, tokens, sequence length, global batch, cluster, precision policy, the failure model.
- **Comparison axis:** equal tokens (15T) on equal hardware (2,048 GPUs).
- **Budget:** no new GPU runs required beyond 09.3 and 09.4; optional: one more torchtitan run with an MoE model (torchtitan 0.3.0 ships DeepSeek-V3 model code; NOT RUN in this build) to measure an MoE MFU and EP all-to-all on one node.
- **Metrics and decision rule (state now):** per layout: memory per GPU (must be ≤ 72 GB, leaving 10% headroom for fragmentation), projected step time with and without overlap, bubble; for the run: GPU-hours, goodput, wall-clock, cost. Choose the layout with the shortest projected step *with* the overlap you can justify from a measurement or a published report; if two are within 10%, choose the one with fewer cross-node collectives per layer.
- **Correctness checks:** `pytest labs/common/tests/test_dist.py` passes on your machine; the planner reproduces lesson 09.1's DeepSeek-V3 figures; every MEASURED input traces to a run card or a script output you include.
- **Fallback evidence:** DeepSeek-V3's layout and training cost (2.788M H800 GPU-hours for 14.8T tokens, sections 3.2 and 1), Llama 3's MFU and failure record, MegaScale's and Gemini 2.5's recovery figures, labelled PUBLISHED.
- **Limits:** node-scale measurements do not show inter-node effects (InfiniBand congestion, stragglers, all-to-all at 2,048 ranks); the failure model assumes independent Poisson failures at Llama 3's per-GPU rate; MFU is extrapolated.

## Steps and deliverables

1. **Contract** (deliverable 1), before step 2.
2. **Inputs.** `python labs/module-09/project/plan_1t.py --write-template my_inputs.json`, then replace every value you measured (09.3: `intra_bw`, `mfu`; 09.4: `ckpt_write_GBps_per_node`, `async_blocking_fraction`, `restart_min`) and set its label to `MEASURED` with a note saying on what. For each value you scale from node to cluster (for example restart time, which on a cluster includes detection and node exclusion), write the scaling rule in the note. Label the rest PUBLISHED (cite) or ASSUMED (say why).
3. **Layouts.** `python labs/module-09/project/plan_1t.py --inputs my_inputs.json` (and with `--act-bytes 2`). Add at least one layout of your own to `candidates()` in a copy of the script. Explain which communication term each layout puts on which link (lesson 09.1's ordering rule), and which pipeline schedule you choose and why (09.2: bubble at your micro-batch count, activation memory, parameter copies).
4. **Node-scale check.** For the chosen layout, name the node-scale quantities the plan depends on (MFU, NVLink bus bandwidth, exposed fraction of intra-node collectives, memory per rank vs the planner's estimate) and put your 09.3 measurement next to each. Where the measurement disagrees with the planner, say which you trust for the plan and why.
5. **Reliability.** Derive the job MTBF for 2,048 GPUs from Llama 3's record (state the assumption about its GPU count), the checkpoint size, the blocking time (sync and async) and the restart cost, the checkpoint interval (Young and numerical) and the goodput. Then the same with Gemini-style elasticity ($R$ of 30 s): what would it buy?
6. **Capability matrix.** For every framework feature the plan relies on (FSDP2 or ZeRO-1, PP schedule, EP, FP8, DCP async save, resharding on restart), the row from your 09.3 matrix, and what you would test first on the real cluster.
7. **The plan** (deliverable 2): one page — the layout and schedule, memory per GPU, communication per step by dimension and link, step time, GPU-hours, wall clock, goodput, cost, each with its label; and the top three risks with the measurement that would retire each.
8. **Record** (deliverable 3): `my_inputs.json`, the script outputs, the 09.3 and 09.4 run cards and reports you relied on.
9. **Debugging task** (deliverable 4), below.
10. **Written defence** (deliverable 5), below.

### What the defaults give

`plan_1t.py` with its default inputs (all ASSUMED or PUBLISHED, nothing measured; run 2026-10-04 on the build laptop, under a second), FP8 activation caching:

| Layout | Memory per GPU (ESTIMATE) | Largest communication term per step | Step without overlap (PROJECTED) |
|---|---|---|---|
| A: PP16 × EP64 × ZeRO-1 | 81.6 GB (over budget) | EP 111 GB over 8 nodes, 2.8 s | 11.5 s |
| B: PP16 × EP32 × ZeRO-1 | 86.3 GB (over budget) | EP 109 GB over 4 nodes, 2.7 s | 11.6 s |
| C: TP8 × PP8 × EP32 × ZeRO-1 | 67.3 GB | EP 127 GB over 32 nodes, 3.2 s; TP 210 GB in-node, 0.7 s | 11.3 s |

Run totals: $3.80 \times 10^{24}$ FLOPs, 3.55M GPU-hours at an assumed 30% MFU; a 10.3 TB checkpoint written in 20 s (sync) or blocking 2 s (async, assumed 10%); job MTBF 24.7 h; checkpoint every 10 minutes; goodput 98.7%; 73 days; about USD 9M at USD 2.5 per GPU-hour. Every one of these is PROJECTED, and the 30% MFU, 40 GB/s and 10-minute restart are guesses until you replace them. The layouts' compute (7.0 s per step) is the same in all three; the decision turns on how much of the 2.7–3.2 s of expert all-to-all can be hidden, which no single-node measurement shows — say so in the plan, and say what you would run first on the cluster to find out.

## Debugging task

`labs/module-09/project/buggy_plan.py` is a colleague's estimate for the same model and cluster. It concludes that the run would take 5.4 years on 2,048 GPUs, that failures cost 11% of the time, that expert-parallel communication is negligible, and recommends asking for 16 times the GPUs. All three headline numbers come from mistakes. For each: name the check that exposes it, fix it, and give the corrected number.

```bash
python labs/module-09/project/buggy_plan.py
```

<details>
<summary>Hint</summary>

Which parameter count belongs in "6 · N · D" for a mixture-of-experts model? Whose cluster produced the 3.09-hour MTBF, and how many GPUs did it have? Which nodes does an EP64 group with EP innermost span, and which link does its all-to-all use?

</details>

<details>
<summary>Reference diagnosis</summary>

1. **Total instead of active parameters.** $6 N D$ with $N = 1.03$T counts every expert for every token; a token uses 40.8B parameters. Check: `frontierlab.calc.flops_per_token` (or $6 \times$ active), or compare with DeepSeek-V3's published 2.788M H800 GPU-hours for 671B total / 37B active on 14.8T tokens. Fixed: $3.80 \times 10^{24}$ FLOPs, 3.55M GPU-hours, 72 days of compute at 30% MFU before failures — not 5.4 years.
2. **A job MTBF from a cluster eight times larger.** 3.09 h is Llama 3's job MTBF on (by assumption) 16,384 GPUs. Per GPU that is about 50,700 h; on 2,048 GPUs the job MTBF is $50{,}700 / 2{,}048 = 24.7$ h. Check: write the MTBF as per-device MTBF ÷ devices and state the device count. Fixed: with the same 20 s checkpoint and 10-minute restart, the optimal interval is about 31 minutes (Young) and the goodput about 97.2% — failures cost under 3%, not 11%.
3. **Cross-node traffic at NVLink bandwidth.** EP64 with EP innermost spans 8 nodes (`group_span`); its all-to-all crosses InfiniBand, about 40 GB/s per GPU in the plan's assumptions, not 300 GB/s. Check: print the group's nodes; compare with DeepSeek-V3's statement that cross-node expert parallelism gives "an inefficient computation-to-communication ratio of approximately 1:1" (section 3.2.1). Fixed: 111 GB per GPU per step at 40 GB/s is 2.8 s against about 7 s of compute — 40% of the step, which is exactly why DualPipe exists.

With all three fixed, the colleague's conclusion reverses: the cluster is adequate (72 days of compute, about 74 days of wall clock with those failure costs) if the all-to-all can be overlapped; the risk to retire first is the EP communication, not the GPU count.

</details>

## Written defence

One to two pages, answering:

1. Which numbers in your plan are MEASURED (on what), which PUBLISHED (where), which ASSUMED? For the three that move the total most, how much would the wall clock change if each were off by a factor of 1.5?
2. Why this layout and not the others? Which collective of your layout crosses InfiniBand, how many bytes per step, and how much of it do you assume is hidden — on what evidence?
3. Your 09.3 measurements are from one node. Which effects of 2,048 GPUs can they not show, and how would you measure them before the run (smallest experiment that would)?
4. Your goodput assumes independent Poisson failures at Llama 3's per-GPU rate. Name two ways real failures violate that (Llama 3 section 3.3.4 gives evidence for both), and how each changes the checkpoint interval.
5. A node fails at hour 300 and no spare is available for two days. What does your plan do: wait, continue on 2,040 GPUs, or restart in a different layout? What has to be true of your checkpoints and data stream for the second and third options (lesson 09.4)?
6. With twice the budget, would you buy twice the GPUs or the same GPUs for twice as long? Answer with the plan's numbers (MTBF falls with cluster size; communication per GPU does not).

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the decision rule for the layout and the memory budget are written before the plan is run |
| 2 | Controls | one model, token count, batch and cluster for every layout; only the stated variables change |
| 3 | Axis and budget parity | all layouts compared at equal tokens on equal GPUs; GPU-hours and cost include goodput |
| 4 | Correctness | `test_dist.py` passes; the planner reproduces lesson 09.1's DeepSeek-V3 figures; the debugging task is fixed with checks named |
| 5 | Uncertainty | the sensitivity of the total to its three most uncertain inputs is shown |
| 6 | Conclusion matches evidence | node-scale measurements are not presented as cluster-scale evidence; projections are labelled |
| 7 | Limits | what one node cannot show, what the failure model leaves out, and what result would change the plan |
