# Module 5 labs — When is sub-quadratic attention worth it?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. Shared Module 5 code sits at
the top of this folder and in `labs/common/frontierlab/attention/`:

| File | What it is |
|---|---|
| `m05.py` | the arms (`b0`, `hybrid-kda`, `hybrid-gdn`, `linear-kda`, `dsa`), defined relative to a preset; importing it registers every Module 3–5 attention kind |
| `train_arm.py` | trains one arm with the unmodified `frontierlab.train.loop`: `--match-params` (equal parameters), `--init-from` (start from a checkpoint), `--dsa-stage warmup/sparse` (the two DSA stages of DeepSeek-V3.2), `--linear-mode fla` (GPU kernels); exact run cards (`m05` metadata, exact FLOPs) |
| `heldout_compare.py` | held-out loss of several runs on the same windows, paired against the first, with the contract's decision column |
| `frontierlab/attention/deltanet.py`, `hybrid.py`, `dsa.py` | the kinds `gdn`, `kda` (recurrent and chunked-parallel gated delta rule; fla path), `hybrid` (N:1 layouts), `dsa` (lightning indexer, top-k, indexer KL objective, mask and gather paths) |
| `frontierlab/attention/subq_bench.py`, `compressed.py` | component profiling of dense / linear / DSA attention (05.3); CSA/HCA-style compression and its cost model (05.4) |
| `frontierlab/attention/accounting.py` (Module 5 section) | exact parameters, FLOPs (indexer included), cache and state bytes, crossover formulas for every kind |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
python -m frontierlab.data.prepare --docs 20000 --vocab 8192          # once, about 3 minutes (Module 1)
pytest labs/common/tests/test_attention_m05.py                        # the Module 5 correctness suite, about 50 s
pytest labs/module-05/lesson-01                                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-05                             # all four reference solutions
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference.
Lesson 05.2 starts from the Module 4 base model `runs/m04/base-cpu` (lesson 04.1,
`python labs/module-04/lesson-01/train_base.py`, about 30 minutes, if you do not have it).

| Folder | Lesson | Scripts | What it does |
|---|---|---|---|
| `lesson-01/` | 05.1 Linear and hybrid attention at scale | `equivalence.py`, `train_arms.py`, then `../heldout_compare.py` | recurrent vs chunked equivalence of your delta rule; state vs KV memory for Baseline-0, Qwen3-Next, Kimi Linear; b0 vs two 3:1 hybrids at equal parameters and tokens |
| `lesson-02/` | 05.2 Learned sparse attention | `dsa_stages.py [--ablation]` | dense warm-up of the indexer, sparse training, a dense control and a no-warm-up ablation from one parent; attention-mass recall vs oracle and window; paired held-out loss; a k sweep |
| `lesson-03/` | 05.3 Measuring cost and quality honestly | `profile_attn.py [--mode decode]`, `quality.py` | component profiles (indexer, top-k, gather, attention, linear core and step, dense SDPA), exponents, crossovers, H100 roofline projection; Eval v1 on every arm with the MiniMax test |
| `lesson-04/` | 05.4 Compressed sparse attention (extension) | `v4_costs.py` | CSA/HCA-style compression, causality, and KV / attended-entry costs as functions of the unstated hyperparameters |
| `project/` | Module project (sub-quadratic decision memo) | `memo_inputs.py`, `buggy_dsa.py` | the memo table for a stated context, load and quality bar; the debugging task |

## Hardware and time per variant

All main-path and free-GPU commands were **not run in this build**; they are part of the Module 5 pilot.
flash-linear-attention 0.5.2 was not installed here (it needs CUDA and Triton); its path is written
against the v0.5.2 API and must first pass `pytest labs/common/tests/test_attention_m05.py -k fla` on the
GPU. Free CPU times were measured on 2026-10-04 on a 16-thread Windows 11 laptop (torch 2.14.1+cpu)
while another build job was using the same CPU, so expect shorter times on an idle machine.

| Lab | Main path (rented GPU) | Free GPU (Colab/Kaggle T4) | Free CPU (measured) |
|---|---|---|---|
| tests | `pytest labs/common/tests/test_attention_m05.py -k fla` first (fla 0.5.2 vs reference) | as CPU | `test_attention_m05.py` about 50 s (60 passed, 2 GPU-only skipped); each lesson's `test_lab.py` 2–15 s |
| 05.1 | 1× H100/A100, about 1 GPU-hour (PROJECTED): `lesson-01/train_arms.py --variant main` (pilot-30m, 1,024 tokens, fla) | `train_arms.py --variant t4`, about 2 h (PROJECTED) | `equivalence.py` 7 s; three arms 55 min (b0 7.3 min, each hybrid about 22 min); optional `--only hybrid-kda-silu` 24 min; `heldout_compare.py` 42 s |
| 05.2 | 1× H100/A100, about 2 GPU-hours (PROJECTED): `lesson-02/dsa_stages.py --variant main --device cuda --ablation` | `--variant t4 --device cuda`, about 1 h (PROJECTED) | `dsa_stages.py --ablation` 32 min (warm-up 2.4, sparse 11.7, control 7.0, no-warmup 9.1, report 2) |
| 05.3 | 1× H100/A100, about 1 GPU-hour (PROJECTED): `profile_attn.py --device cuda --dtype bf16 --shape baseline0 --contexts 8192 16384 32768 65536 131072 --topk 2048 --chunk 64 --linear-mode fla` (both modes), `quality.py --variant main --device cuda --bf16 --lengths 1024 4096 16384` | `profile_attn.py --device cuda --contexts 2048 4096 8192 16384 32768`, about 30 min (PROJECTED) | prefill sweep 3.2 min, decode sweep 19 s, `quality.py` 10.3 min |
| 05.4 | not needed | not needed | tests 2 s, `v4_costs.py` under 1 s |
| project | about 4–5 GPU-hours in total with the lesson runs (PROJECTED; see the project page) | about 4–5 T4-hours (PROJECTED) | `memo_inputs.py` seconds; `buggy_dsa.py` 3 min |

Notes:

- The main path needs Data-v0 at its main-path size (`python -m frontierlab.data.prepare --docs 2600000 --vocab 32768`, Module 1 project) and, for 05.2, your Module 1 Baseline-0 seed-0 run as the parent.
- Every correctness check runs in float64 with RMSNorm in the input dtype (`checks.exact_rmsnorm`) on sharpened weights (`checks.sharpen`), as in Module 3.
- Our linear core and DSA paths are PyTorch reference code: correct and memory-faithful, not fast kernels. CPU timings never transfer to GPUs, and the DSA gather path says nothing about DeepSeek's fused sparse kernels.
- Outputs go to `runs/` (gitignored); keep the printed tables with your run cards.
