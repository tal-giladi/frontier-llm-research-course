# Module 3 labs — How should attention spend KV memory?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lab needs. Shared Module 3 code sits at
the top of this folder and in `labs/common/frontierlab/attention/`:

| File | What it is |
|---|---|
| `m03.py` | the arms (`b0`, `gqa-kv`, `mla`, `local-global`, `sliding`, `sink`, `gated`, `no-qknorm`, `wide-heads`, `partial-rope`), defined relative to a preset; importing it registers every Module 3 attention kind |
| `train_variant.py` | trains one arm with the unmodified `frontierlab.train.loop`; `--match-params` (equal parameters) and `--equal-flops-to N` (equal training FLOPs); exact run cards for every kind |
| `decode_compare.py` | decode memory (measured `Cache.nbytes` against the formula) and per-token latency at fixed contexts, interleaved rounds with paired intervals (lesson 02.4 method) |
| `frontierlab/attention/mla.py`, `sliding.py`, `gated.py`, `headshape.py` | the attention kinds `mla`, `sliding`, `local_global`, `sink`, `gated`, `gqa_partial` |
| `frontierlab/attention/ops.py`, `accounting.py`, `checks.py`, `probes.py`, `bench.py` | banded/sink attention, parameter-FLOP-KV arithmetic for any kind, the correctness suite, sink and logit probes, the decode benchmark |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
python -m frontierlab.data.prepare --docs 20000 --vocab 8192          # once, about 3 minutes (Module 1)
pytest labs/common/tests/test_attention_m03.py                        # the correctness suite, about 1-2 minutes
pytest labs/module-03/lesson-01                                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-03                             # all four reference solutions
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference.

| Folder | Lesson | Scripts | What it does |
|---|---|---|---|
| `lesson-01/` | 03.1 Multi-head Latent Attention | `check_mla.py`, `why_decoupled.py`, then `../decode_compare.py` | correctness suite on MLA and on your absorbed attention; why RoPE needs its own key; decode memory and latency of b0, gqa-kv, MLA naive and absorbed |
| `lesson-02/` | 03.2 Local/global attention and KV arithmetic | `kv_table.py`, then `../decode_compare.py --arms b0 local-global sliding` | KV bytes per token for every design at 32K–1M, checked against real caches; decode memory and latency |
| `lesson-03/` | 03.3 Logit control, sinks and gating | `train_arms.py`, `probe.py` | six short training runs (b0, no QK-norm, sink, gated, and two at a raised learning rate); first-token mass, sink mass, max logit, massive activations, paired loss differences |
| `lesson-04/` | 03.4 Head count and head dimension (extension) | `heads.py [--train]` | Kimi K2 64 vs 128 heads from its config; Qwen3-Next partial RoPE; optional three-run head-shape comparison |
| `project/` | Module project (attention memo) | `run.py`, `evaluate.py`, `compare.py`, `buggy_decode.py` | parts A (equal parameters) and B (equal training FLOPs), Eval v0, seed-level verdicts and the serving-constraint table; the debugging task |

## Hardware and time per variant

All main-path and free-GPU commands were **not run in this build**; they are part of the Module 3
pilot. Free CPU times were measured on 2026-10-03 on a 16-thread Windows 11 laptop (torch 2.14.1+cpu)
while another build job was using the same CPU, so expect shorter times on an idle machine and wide
timing intervals in the decode benchmarks.

| Lab | Main path (rented GPU) | Free GPU (Colab/Kaggle T4) | Free CPU (measured) |
|---|---|---|---|
| 03.1 | 1× H100 SXM or A100, ~10 GPU-min: `decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --arms b0 gqa-kv mla-naive mla-absorbed --contexts 8192 16384 32768 --rounds 30` | `--dtype fp32 --preset pilot-10m --contexts 4096 8192 16384` | `check_mla.py` ~20 s; `decode_compare.py` ~1 min |
| 03.2 | ~10 GPU-min: `decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --train-seq 1024 --arms b0 local-global sliding --contexts 8192 16384 32768 --rounds 30` | `--dtype fp32 --preset pilot-10m --train-seq 512 --contexts 4096 8192 16384` | `kv_table.py` ~10 s; `decode_compare.py` ~1 min |
| 03.3 | under 1 GPU-hour (PROJECTED): `train_arms.py --variant main`, `probe.py runs/m03/l33/main --device cuda` | `train_arms.py --variant t4 --max-minutes 80` | 6 training runs 23 min; `probe.py` ~40 s |
| 03.4 | not needed | `heads.py --train` as on CPU | arithmetic seconds; `--train` ~9.5 min (3 runs) |
| project | ~39 GPU-hours for 3 seeds (PROJECTED; see the project page): `run.py --variant main --part A/B --seeds 0 1 2`, `evaluate.py ... --device cuda --bf16`, `decode_compare.py ... --batch 8`, `compare.py` | `run.py --variant t4 ...`, about 10–15 T4-hours (PROJECTED) | ~60 min (14 runs 38 min, Eval v0 21 min) |

Notes:

- The main path needs Data-v0 at its main-path size (`python -m frontierlab.data.prepare --docs 2600000 --vocab 32768`, Module 1 project).
- Every correctness check runs in float64 with RMSNorm computed in the input dtype (`frontierlab.attention.checks.exact_rmsnorm`) on a copy with sharpened weights (`checks.sharpen`); see `curriculum/inbox/module-03-shared-changes.md` for why.
- Our sliding-window and sink kinds use a boolean mask or an explicit softmax: correct and memory-faithful, not fast kernels. Their training speed says nothing about FlashAttention or FlexAttention implementations.
- CPU decode timings never transfer to GPUs: toy-size CPU decode is bound by arithmetic and fixed costs, GPU decode by bytes read.
- Outputs go to `runs/` (gitignored); keep the printed tables with your run cards.
