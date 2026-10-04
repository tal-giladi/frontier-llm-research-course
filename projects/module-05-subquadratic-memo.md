# Module 5 project · Sub-quadratic decision memo: when does it beat dense attention?

Your team will serve and train a model at a stated context length on stated hardware, under a stated quality bar, and three designs are on the table: dense attention (Baseline-0), a 3:1 KDA hybrid (05.1) and DSA-style learned selection (05.2). This project turns the module's measurements into a one-page decision memo for two scenarios, one where a sub-quadratic design should be expected to win and one where dense attention should — and you must defend whichever answer your evidence gives, including "dense wins" in the first scenario if that is what it shows. It uses all of Module 5 (05.1–05.3; 05.4 optionally) and the methods of Modules 1–4: the experiment contract, paired intervals, run cards, the benchmark harness and Eval Suite v1.

**Time:** 5–7 attended hours plus unattended runtime. **Folder:** [`labs/module-05/project/`](../labs/module-05/) (`memo_inputs.py`, `buggy_dsa.py`; the runs come from the lesson labs: `lesson-01/train_arms.py`, `lesson-02/dsa_stages.py`, `lesson-03/profile_attn.py`, `lesson-03/quality.py`, `heldout_compare.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## The two scenarios

Use these unless your team has real ones. Both assume Baseline-0's shape for the cost arithmetic.

| | Scenario L (long documents) | Scenario S (chat) |
|---|---|---|
| Context | 32,768 tokens prefill, 1,024 generated | 4,096 tokens prefill, 512 generated |
| Hardware | 1× H100 SXM 80 GB per replica, BF16 | same |
| Load | 8 concurrent sequences per replica | 64 concurrent sequences per replica |
| Quality bar | held-out loss within 0.03 nats of dense, and no *resolved* deficit on Eval v1 two-hop items | held-out loss within 0.03 nats of dense |
| Cost goal | at least 1.3× decode throughput or 1.3× prefill throughput against dense at 32K | at least 1.3× either, at 4K |

Expectations from the cost model (PROJECTED; your measurements decide): at 32K the hybrid caches 97.8 MiB per sequence instead of 384 MiB and its linear layers' decode step costs the same at any context; DSA's decode reads about a quarter of dense attention's bytes per layer at 32K (05.3 worked example) but keeps the whole cache. At 4K, a linear layer still does less mixing arithmetic than a full one (the per-layer crossover is about 400 tokens), but mixing is a small share of a 4K step next to the projections, and Baseline-0's cache is only 48 MiB per sequence (3 GiB for 64 sequences), so there is little to save; DSA with $k = 2{,}048$ attends to half of a 4K context while still paying for the indexer, top-k and gather. Scenario S is where dense attention should win; the memo must show whether it does, with numbers.

## Variants and cost

| Variant | Setting | Hardware | Runs | Cost |
|---|---|---|---|---|
| Main path | pilot-30m (05.1) and Baseline-0 (05.2) at $T = 1024$; profiles at Baseline-0 shape, 4K–128K, BF16, fla kernels for the linear core; Eval v1 at 1K–16K | 1× H100 SXM 80 GB | the 05.1 and 05.2 main-path runs (3 + 4), 2 seeds for the hybrid and b0 arms | **PROJECTED, pending the Module 5 pilot:** 05.1 about 0.4 GPU-hours per seed (formula in 05.1), 05.2 about 2 GPU-hours, profiles and Eval v1 about 1 GPU-hour; about 4–5 GPU-hours in total, roughly USD 10–15 at USD 2–3 per H100-hour. Not run in this build |
| Free GPU (Colab/Kaggle T4) | the `t4` variants of each lab script | T4 | as main path, 1 seed | PROJECTED: about 4–5 T4-hours; use `--max-minutes` and rerun after disconnects |
| Free CPU | the lesson labs' CPU runs, unchanged | laptop | 05.1: 3 runs; 05.2: 4 runs; 05.3 profiles and Eval v1 | measured in this build: see "What the free CPU variant gave" below |

Main-path commands, all **not run in this build; part of the Module 5 pilot**:

```bash
pytest labs/common/tests/test_attention_m05.py -k fla                         # fla 0.5.2 against the reference, on the GPU
python labs/module-05/lesson-01/train_arms.py --variant main --seed 0
python labs/module-05/lesson-01/train_arms.py --variant main --seed 1
python labs/module-05/lesson-02/dsa_stages.py --variant main --device cuda --ablation
python labs/module-05/lesson-03/profile_attn.py --device cuda --dtype bf16 --shape baseline0 --topk 2048 --chunk 64 \
    --linear-mode fla --contexts 4096 8192 16384 32768 65536 131072 --out runs/m05/project/main-prefill.json
python labs/module-05/lesson-03/profile_attn.py --device cuda --dtype bf16 --shape baseline0 --topk 2048 --chunk 64 \
    --linear-mode fla --mode decode --contexts 4096 8192 16384 32768 65536 131072 --out runs/m05/project/main-decode.json
python labs/module-05/lesson-03/quality.py --variant main --device cuda --bf16 --lengths 1024 4096 16384
python labs/module-05/heldout_compare.py runs/m05/l51/main/b0-s0 runs/m05/l51/main/hybrid-kda-s0 --seq 1024 --device cuda --out runs/m05/project/heldout-l51.json
python labs/module-05/heldout_compare.py runs/m05/l52/main/control runs/m05/l52/main/sparse --seq 1024 --device cuda --out runs/m05/project/heldout-l52.json
python labs/module-05/project/memo_inputs.py --context 32768 --batch 8 --decode runs/m05/project/main-decode.json \
    --prefill runs/m05/project/main-prefill.json --heldout runs/m05/project/heldout-l5*.json
python labs/module-05/project/memo_inputs.py --context 4096 --batch 64 --decode runs/m05/project/main-decode.json \
    --prefill runs/m05/project/main-prefill.json --heldout runs/m05/project/heldout-l5*.json
```

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running anything new. Fixed by the project:

- **Question:** for each scenario, which attention design should the team deploy, given measured and projected cost at the scenario's context and load and quality against the scenario's bar? Decision informed: the attention block for Module 6's integration experiment, and the serving plan.
- **Hypotheses and status:** (1) in scenario L at least one sub-quadratic arm meets the cost goal — expected from the cost model, depends on kernels; (2) in scenario S dense attention is at least as cheap as either arm — expected; (3) both sub-quadratic arms meet the held-out loss bar at course scale — reported at frontier scale (company claims), may not appear here; (4) MiniMax's two-hop deficit — company claim, likely unresolvable at course scale.
- **Baselines:** `b0-s0` for the hybrid (05.1, equal parameters and tokens), `control` for DSA (05.2, equal LM tokens from the same parent); dense SDPA at the same shape for cost.
- **Changed variable:** the attention design. **Controlled:** as in each lesson's contract; for cost, layer shape, inputs, dtype, kernels named per arm.
- **Comparison axes:** equal parameters and tokens (05.1), equal LM tokens from one parent (05.2), equal layer shape at equal context (cost). Say in the memo what none of them answers: training cost of a from-scratch hybrid at frontier scale, end-to-end serving throughput with a real scheduler, prefix caching and speculative decoding.
- **Budget:** from the table above, projected now, measured in the run cards afterwards.
- **Metrics and decision rule (state your own, before looking):** for example, "deploy an arm in a scenario if its decode or prefill speed-up CI at the scenario's context lies above 1.3 *on the main-path GPU* and its held-out loss CI lies inside ±0.03 nats; in scenario L also require that the two-hop evidence-effect difference is not resolved below −0.05 nats; otherwise deploy dense". CPU speed-ups may support a hypothesis; they may not decide a GPU deployment.
- **Correctness checks:** `pytest labs/common/tests/test_attention_m05.py` and every lesson lab with `LAB_TARGET=solution` (or your own `lab.py`) pass; on the GPU, the fla test passes before any fla timing is used.
- **Fallback evidence:** the Module 5 pilot's traces (component profiles 4K–32K on an A100, the 30M hybrid and DSA conversion runs), labelled as provided analysis.
- **Limits:** scale, single seed on CPU, untuned arms, reference kernels for DSA, batch-1 profiles, evaluation power.

## Steps and deliverables

1. **Collect the runs** of 05.1 (three arms), 05.2 (warm-up, sparse, control, no-warmup) and the 05.3 profiles; rerun any that are missing (each script skips finished runs).
2. **Held-out comparisons with saved JSON:**

   ```bash
   python labs/module-05/heldout_compare.py runs/m05/l51/cpu/b0-s0 runs/m05/l51/cpu/hybrid-kda-s0 runs/m05/l51/cpu/hybrid-gdn-s0 runs/m05/l51/cpu/hybrid-kda-silu-s0 --seq 256 --out runs/m05/project/heldout-l51.json
   python labs/module-05/heldout_compare.py runs/m05/l52/cpu/control runs/m05/l52/cpu/sparse runs/m05/l52/cpu/no-warmup --seq 256 --out runs/m05/project/heldout-l52.json
   ```

3. **The memo inputs** for both scenarios:

   ```bash
   python labs/module-05/project/memo_inputs.py --context 32768 --batch 8 --decode runs/m05/l53/cpu-decode.json --prefill runs/m05/l53/cpu-prefill.json --heldout runs/m05/project/heldout-l51.json runs/m05/project/heldout-l52.json
   python labs/module-05/project/memo_inputs.py --context 4096 --batch 64 --decode runs/m05/l53/cpu-decode.json --prefill runs/m05/l53/cpu-prefill.json --heldout runs/m05/project/heldout-l51.json runs/m05/project/heldout-l52.json
   ```

   The CPU profile stops at 8K (prefill) and 64K (decode); the script uses the nearest profiled context and says so. Extend `--contexts` if your machine allows, or use the pilot's GPU traces.

4. **Write the memo** (one page): the recommendation per scenario; the table (memory per sequence and sequences per GPU; measured speed-ups with machine and kernel; projected H100 speed-ups with the formula; quality differences with intervals; the MiniMax test's status); what each number cannot show; and the measurement that would change your recommendation.

Deliverables: the contract, the run cards (each with `m05` metadata), the JSON outputs, the memo, the debugging write-up, the written defence.

### What the free CPU variant gave in this build

Measured 2026-10-04 on the build laptop (16 threads, torch 2.14.1+cpu, fp32, another build job sharing the CPU), from the lesson runs (about 2 hours of runtime in total: 05.1 three arms 55 minutes plus the step-5 arm 24 minutes, 05.2 32 minutes, 05.3 profiles 3.5 minutes and Eval v1 10 minutes; the project scripts themselves take seconds, `buggy_dsa.py` 3 minutes). `memo_inputs.py` with the default `hybrid-kda-silu` arm:

| | dense | 3:1 hybrid (`hybrid-kda-silu`) | DSA (`sparse`) |
|---|---|---|---|
| cache per sequence at 32K, Baseline-0 shape (PROJECTED, tested formula) | 384 MiB, 178 sequences on 80 GB | 97.8 MiB, 699 sequences | 408 MiB, 167 sequences |
| cache at 4K | 48 MiB | 13.8 MiB | 51 MiB |
| decode speed-up at 16K (nearest profiled to 32K), CPU, pilot-10m layer | 1 | 1.34 [0.67, 2.36] | 0.92 [0.66, 1.02] |
| decode speed-up at 4K, CPU | 1 | 0.86 [0.68, 1.37] | 0.39 [0.12, 0.54] |
| prefill speed-up at 8K, CPU (our reference kernels) | 1 | 0.23 [0.16, 0.24] | 0.24 [0.22, 0.26] |
| decode speed-up at 32K / 4K, H100 roofline (PROJECTED, no launch overheads) | 1 | 42× / 10.5× | 5.6× / 3.1× |
| prefill speed-up at 8K / 4K, H100 roofline (PROJECTED) | 1 | 6.1× / 3.0× | 0.1× / 0.05× |
| held-out loss vs its dense control, 256 tokens | — | $-0.043$ $[-0.049, -0.036]$ | $+0.0024$ $[+0.0016, +0.0032]$ |

The rule with CPU decode speed-ups gives "inconclusive" for the hybrid and "keep dense" for DSA in both scenarios; on these numbers dense attention wins scenario S outright (no arm is faster at 4K on this machine, and the 4K cache is small), and scenario L cannot be decided without the GPU profile — exactly the gap the pilot fills. The projected DSA prefill at 0.1× is the gather traffic of GQA-shaped entries (05.3 worked example), a property of our path, not of DeepSeek's MQA-mode kernels. These are the build's numbers; your memo uses yours.

## Debugging task

`python labs/module-05/project/buggy_dsa.py` reproduces a teammate's report: a DSA variant, `dsa-fast`, whose held-out loss after 60 steps is far *lower* than dense attention's. Follow the three steps in the script's docstring: run the correctness suite on the variant before reading its code, find the bug, and name the lab check that would have caught it before any training. Then answer: why does a selection bug of this kind make the model look *better* on held-out loss rather than worse, and what does that say about using held-out loss alone as a correctness signal?

<details>
<summary>What to look for (open after your write-up)</summary>

In this build `buggy_dsa.py` printed held-out losses of 4.9762 (dense) and 4.8018 (`dsa-fast`) after 60 steps; on a sharpened toy model the causal check fails with a maximum logit difference of 0.91 and the one-token cached-decode check with 1.15.

`FastDSA` takes the top-$k$ over the whole score row and never ANDs the selection with the causal mask, so queries attend to future tokens. Held-out loss uses the same forward pass, so it is also computed with access to the answer: the leak lowers it. The causal check of the correctness suite (changing future tokens must not change past logits) fails with a large difference; it is the first test every attention kind in this course passes before any comparison counts. The cached-decode check fails too, because decoding cannot see the future the full forward used.

</details>

## Written defence

One to two pages, answering:

1. Your scenario L recommendation rests on which measurement: a CPU timing, a GPU timing or a projection? If a projection, show its formula and say which pilot measurement would replace it.
2. The hybrid comparison in 05.1 was at 256 tokens. Why is it still evidence for or against the hybrid in scenario L, and what would a fair long-context comparison cost?
3. DSA keeps the full cache. In scenario L with 8 sequences, is memory or bandwidth the binding constraint for dense attention, and does DSA help with the one that binds?
4. What is the smallest two-hop deficit your Eval v1 run could have detected? Would a null result have been enough to dismiss MiniMax's concern?
5. In which scenario does dense attention win in your data, and is that a property of the method or of your kernels? How would you tell?
6. With 10× the budget, what would you run next, and what result would make you abandon your recommendation?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | both scenarios' constraints and your decision rule are written before you look at new numbers |
| 2 | Controls | run-card diffs show only the declared changes (attention kind, SwiGLU width for equal parameters, the DSA stages) |
| 3 | Axis and budget parity | the axis of every comparison is named; parameters equal within 0.1% in 05.1; LM tokens equal in 05.2 with the warm-up's extra forward passes stated |
| 4 | Correctness | the Module 5 suite output is included; on the GPU, the fla test passed before fla timings were used |
| 5 | Uncertainty | every speed-up and quality difference has a paired interval; the evaluation's detectable effect is stated |
| 6 | Conclusion matches evidence | CPU timings support hypotheses only; projected numbers are labelled with their formula; "dense wins" is reported where the data say so |
| 7 | Limits | scale, seeds, kernels, batch-1 profiles, untuned arms, evaluation power, and what would change the answer |
