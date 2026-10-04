# Module 8 labs — How low can precision go?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. Shared code is in
`labs/common/frontierlab/precision/` (exact format emulation, scaled quantisation, the accumulation model, random
Hadamard transforms, the emulated low-precision linear layer and its recipes, QAT/PTQ, the precision log, the torchao
Float8 path, the roofline cost model, the precision scaling law, and the training wrapper `train.py`); its tests are
`labs/common/tests/test_precision.py`. `labs/module-08/m08.py` holds the helpers the scripts share (run an arm,
evaluate it on 256 fixed held-out windows in its training precision, compare arms with a paired bootstrap).

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/module-08/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-08             # all four reference solutions
pytest labs/common/tests/test_precision.py            # the shared Module 8 code, incl. exact resume with stochastic rounding
```

Main path (GPU) extra: `pip install torchao==0.18.0` (FP8 kernels need an L4, H100 or newer; use `--compile`). The
B200 script of lesson 08.3 also needs NVIDIA Transformer Engine (not pinned; see the lesson).

Every training run goes through `python -m frontierlab.precision.train` — the unmodified course loop, wrapped by the
Module 7 wrapper (so `--optimizer muon`, `--stability-log`, `--width`, `--init-from` all work), with the block linears
swapped for emulated low-precision ones (`--recipe`) or for torchao Float8 ones (`--torchao`, CUDA only). Exact resume
holds: rerun the same command (or the same lab script) after an interruption. Scripts load your `lab.py` by default;
prefix `LAB_TARGET=solution` to run them with the reference.

> [!IMPORTANT]
> Every `--recipe` other than `bf16` is **emulation**: it reproduces the format's rounding exactly in float32 and is
> several times *slower* than BF16. It measures numerics, never speed. Speed comes only from `--torchao` runs and
> `lesson-02/bench_fp8.py` on an FP8-capable GPU, or from the labelled roofline projections.

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 08.1 Number formats and scaling | `error_tour.py` | trains the toy model 60 steps, captures each linear's x, W and dy, and tabulates relative error, underflow, saturation and bits per value for 12 format/scaling choices, plus an outlier-channel variant |
| `lesson-02/` | 08.2 FP8 training | `accum_demo.py`, `train_fp8.py`, `compare_fp8.py`, `bench_fp8.py` | accumulator model with and without FP32 promotion; BF16 vs emulated FP8 (torchao tensorwise/rowwise numerics, DeepSeek-V3 fine-grained) or real torchao arms; paired comparison and the contract's rule; real-kernel step-time benchmark (GPU) and the roofline projection |
| `lesson-03/` | 08.3 FP4 training and QAT | `wgrad_error.py`, `train_fp4.py`, `compare_fp4.py`, `qat_ptq.py`, `bench_nvfp4.py` | FP4 weight gradients with/without RHT and SR on real tensors; the NVFP4 ablation (naive MXFP4, NVFP4, BF16 last block, no SR, no RHT); PTQ vs QAT at equal tokens (INT4 all linears, MXFP4 MLP only); B200 Transformer Engine benchmark (not piloted) |
| `lesson-04/` | 08.4 Scaling laws for precision (extension) | `sweep.py`, `fit.py` | 3 widths × 5 weight precisions plus a PTQ-vs-tokens ladder; fits the law's form, leave-widest-out check, PTQ exponent |
| `project/` | Module project | `error_budget.py`, `buggy_budget.py` | per-component serving error budget with bits per weight and projected speed-up, additivity check; the debugging task |

Command-line tools from the shared code:

```bash
python -m frontierlab.precision.train --recipe fp8-deepseek --precision-log --run RUN --preset toy --steps 200 --batch 16 --seq 128
python -m frontierlab.precision.train --recipe nvfp4 --keep-high first2,last8 --run RUN ...
python -m frontierlab.precision.train --recipe int4-qat --init-from CKPT --run RUN ...          # QAT from trained weights
python -m frontierlab.precision.train --torchao rowwise --compile --dtype bf16 --device cuda --run RUN ...   # real FP8 (GPU)
```

Recipes: `bf16`, `fp8-tensorwise`, `fp8-rowwise`, `fp8-deepseek`, `mxfp8`, `mxfp4`, `nvfp4`, `nvfp4-no-rht`,
`nvfp4-no-sr`, `nvfp4-1d-weights`, `int4-qat`, `mxfp4-qat`, and `w-int2` … `w-int8` (weight-only INT-b QAT for 08.4).

## Hardware and time per variant

Main-path and Colab commands were **not run in this build**; the L4 commands are the Module 8 pilot, and the B200
commands of 08.3 are **not piloted** (Colab has no Blackwell GPU). Every main-path figure in the lessons is PROJECTED
with its formula. Free CPU times were measured on 2026-10-04 on a 16-thread Windows 11 laptop (torch 2.14.1+cpu)
with another agent's jobs running at the same time, so expect variation.

| Lab | Main path (rented GPU) | Free GPU | Free CPU (measured) |
|---|---|---|---|
| 08.1 | any 1× GPU, minutes: `error_tour.py --device cuda --preset pilot-30m --steps 300` (emulation) | T4: `--preset pilot-10m --steps 300` | `error_tour.py` 81 s (includes a 60-step toy run) |
| 08.2 | 1× H100 SXM, PROJECTED 6–9 GPU-hours: `train_fp8.py --variant main --data <vocab-32768 Data-v0>`, `compare_fp8.py --variant main --device cuda --speedup P LO HI`, `bench_fp8.py --device cuda --hw H100-SXM --preset baseline0 --batch 16 --seq 1024` | **L4 (the pilot)**, PROJECTED 1.5–2.5 GPU-hours: `train_fp8.py --variant l4`, `compare_fp8.py --variant l4 --device cuda`, `bench_fp8.py --device cuda --hw L4 --preset pilot-30m --batch 8 --seq 512`. T4: no FP8 hardware, use the CPU variant | `accum_demo.py` 30 s; `train_fp8.py` 37.6 min (8 runs); `compare_fp8.py` 69 s |
| 08.3 | 1× B200, **not piloted**: `bench_nvfp4.py --device cuda` (Transformer Engine), `train_fp4.py --variant main --print` | T4, emulation: `train_fp4.py --variant t4` (PROJECTED 1–2 hours), then `compare_fp4.py --variant t4 --device cuda` | `wgrad_error.py` 5.4 min; `train_fp4.py` 32.5 min (8 runs); `compare_fp4.py` 86 s; `qat_ptq.py` 3.9 min |
| 08.4 | any 1× GPU, PROJECTED under 1 GPU-hour: `sweep.py --variant gpu --device cuda`, `fit.py runs/m08/l84/gpu/results.json` | same as main path | `sweep.py` 24.8 min (19 runs); `fit.py` 10 s |
| project | Baseline-0 checkpoint on 1× H100: `error_budget.py --ckpt <ckpt> --device cuda --hw H100-SXM --T 1024`, plus 08.2's main-path runs and benchmark (PROJECTED about 0.5 GPU-hours beyond 08.2) | L4: 08.2's L4 runs, `error_budget.py --ckpt runs/m08/l82/l4/bf16-s0/checkpoint.pt --device cuda --hw L4 --preset pilot-30m --batch 8 --seq 512` | `error_budget.py` 69 s; `buggy_budget.py` about 1 min |

Notes:

- The T4 (sm75) has no FP8 or FP4 tensor cores: on a T4 everything is emulation, and the CPU variants are usually as
  useful. The L4 (sm89) does have FP8 and is the free-tier way to run real FP8 kernels.
- `--precision-log` uses hooks and is refused with `--compile`; `--torchao` is refused on a CPU and together with an
  emulated `--recipe`.
- Evaluate a Module 8 checkpoint with `frontierlab.precision.train.load_model` (rebuilds its emulated layers from the
  run card; pass `recipe="bf16"` to evaluate the master weights in high precision). `m08.eval_losses` does this.
- Outputs go to `runs/` (gitignored).
