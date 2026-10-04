# Module 6 labs — Which other block changes earn their complexity?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`), `solution.py`
(the reference), `test_lab.py` and the scripts the lesson runs. Shared Module 6 code is the subpackage
`labs/common/frontierlab/blocks/` (tests: `labs/common/tests/test_blocks.py`):

| File | What it is |
|---|---|
| `blocks/model.py` | `BlockLM`: Baseline-0 with Module 6 switches in `cfg.extra["blocks"]` (bit-identical to Baseline-0 with all switches off) |
| `blocks/mtp.py` | Meta's parallel heads, DeepSeek-V3's sequential MTP modules, the λ schedule, lossless self-speculative greedy decoding, acceptance |
| `blocks/hyperconn.py` | Sinkhorn-Knopp, Zhu et al.'s hyper-connections, mHC, Amax gains, the residual's memory traffic (mHC Table 2) |
| `blocks/moe.py` | the fine-grained MoE FFN **copied from the MoE course's `moelab`** (router, fused experts, Switch balance loss), plus a shared expert and a dense reference |
| `blocks/engram.py` | Engram-style hashed N-gram lookup, gate, causal convolution, collision statistics |
| `blocks/matformer.py` | MatFormer nested FFN, Mix'n'Match, Gemma 3n-style per-layer embeddings |
| `blocks/patching.py` | BLT entropy patching with a count-based byte model |
| `blocks/diffusion.py` | the `"gqa-bidir"` attention kind, LLaDA's objective, likelihood bounds, low-confidence remasking |
| `blocks/latent.py` | Coconut continuous thoughts, a recurrent-depth model with truncated backpropagation |
| `blocks/accounting.py` | exact parameters and FLOPs of every BlockLM (equal-parameter and equal-FLOPs axes) |
| `blocks/checks.py` | op-level float64 gradcheck of a module, causal and cached-decode checks of a model |
| `blocks/train.py` | `python -m frontierlab.blocks.train`: the unmodified course loop with a BlockLM (exact resume holds) |
| `labs/module-06/m06.py` | the arms (relative to a preset), variants, training an arm once, evaluation, paired comparisons |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
python -m frontierlab.data.prepare --docs 20000 --vocab 8192          # once (Module 1), about 3 minutes
pytest labs/common/tests/test_blocks.py                               # the Module 6 correctness suite, about 2 minutes
pytest labs/module-06/lesson-01                                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-06                             # all seven reference solutions
```

Scripts load your `lab.py` where they need it; prefix `LAB_TARGET=solution` to run them with the reference. Runs go to
`runs/m06/<variant>/<arm>/s<seed>` (gitignored) and are **shared between lessons**: lesson 06.3's factorial and the
project reuse the Baseline-0, MTP and mHC runs of 06.1 and 06.2 when the variant, arm and seed match. Run the lessons
in order and nothing is trained twice.

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 06.1 Multi-token prediction | `train_arms.py`, `compare.py`, `acceptance.py` | b0, DeepSeek MTP, Meta heads, b0 at MTP's FLOPs × 2 seeds; equal-tokens and equal-FLOPs comparisons and the rule; acceptance rate (teacher-forced and real speculative decoding, lossless check) and the projected speed-up |
| `lesson-02/` | 06.2 Hyper-connections and mHC | `train_arms.py`, `compare.py`, `step_time.py` | b0 / HC / mHC at lr 1.5e-3 (2 seeds), 1e-2 and 3e-2; spikes, gradient norms, Amax gains, H1/H2 verdicts; interleaved step-time ratios against FLOP and traffic ratios |
| `lesson-03/` | 06.3 Combining changes | `factorial.py` | 2 × 2 factorial MTP × mHC (reusing three cells); interaction with a paired interval, additive prediction, integration budget |
| `lesson-04/` | 06.4 Engram (extension) | `collisions.py`, `iso_param.py` | N-gram counts and hash collisions on Data-v0 vs the uniform prediction; MoE vs MoE + Engram at equal parameters |
| `lesson-05/` | 06.5 Elastic architectures (extension) | `elastic.py` | one MatFormer run; every granularity and Mix'n'Match vs independently trained models; consistency |
| `lesson-06/` | 06.6 BLT entropy patching (extension) | `patching_study.py` | byte entropies of Data-v0 text, thresholds vs patch sizes, FLOPs per byte vs BPE |
| `lesson-07/` | 06.7 Non-autoregressive and latent reasoning (extension) | `nonar.py` | a tiny masked-diffusion LM, its likelihood bound next to Baseline-0's loss, samples, cost arithmetic |
| `project/` | Module project (Lineage-F) | `run_project.py`, `buggy_report.py` | select / decide / attribute / report; the debugging task |

Command-line examples of the wrapper:

```bash
python -m frontierlab.blocks.train --mtp deepseek --mtp-depth 1 --mtp-schedule deepseek --run RUN --preset toy --steps 200 --batch 16 --seq 128
python -m frontierlab.blocks.train --residual mhc --streams 4 --blocks-log --hyper-every 20 --run RUN ...
python -m frontierlab.blocks.train --attention mla --extra '{"kv_lora_rank": 64, "qk_rope_head_dim": 16}' --mtp deepseek --residual mhc --ffn moe --run RUN ...
python -m frontierlab.blocks.train --ffn moe --moe-experts 6 --engram-layers 1 --engram-table 1465 --run RUN ...
python -m frontierlab.blocks.train --ffn matformer --run RUN ...
python -m frontierlab.blocks.train --objective diffusion --run RUN ...
python -m frontierlab.blocks.train --cfg intermediate_size=48 --run RUN ...         # preset fields
```

## Hardware and time per variant

Main-path and free-GPU commands were **not run in this build**; they are part of the Module 6 pilot, and every
main-path figure in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-04 on a 16-thread
Windows 11 laptop (torch 2.14.1+cpu) while another build job shared the CPU, so expect variation (single toy runs
varied between 87 and 138 s for the same configuration).

| Lab | Main path (rented GPU) | Free GPU (Colab/Kaggle T4) | Free CPU (measured) |
|---|---|---|---|
| 06.1 | 1× H100/A100, PROJECTED ~1.0 GPU-hour: `train_arms.py --variant main` (pilot-30m, 8 runs × 262M tokens); ladder rung `--variant main70` ~1.7 GPU-hours | `train_arms.py --variant t4` | 8 runs 22 min; `compare.py` 41 s; `acceptance.py` 31 s |
| 06.2 | the plan 12.1 ladder: `train_arms.py --variant t4 --device cuda` (pilot-10m), `--variant main` (pilot-30m), `--variant main70` (pilot-70m); PROJECTED 2–3.5 GPU-hours for pilot-30m incl. mHC's wall-clock factor | `train_arms.py --variant t4` | 9 new runs 30 min (12 runs incl. 06.1's b0); `compare.py` 83 s; `step_time.py` 59 s |
| 06.3 | `factorial.py --variant main --device cuda` after 06.1/06.2's main runs: 2 new runs, PROJECTED ~0.3 GPU-hours × mHC factor | `factorial.py --variant t4 --device cuda` | 2 new runs 10 min (6 cells reused); analysis seconds |
| 06.4 | optional (`iso_param.py --variant main --device cuda`) | `--variant t4` | `collisions.py` 37 s; `iso_param.py` 4.4 min |
| 06.5 | optional (`elastic.py --variant main --device cuda`) | `--variant t4` | `elastic.py` 3.5 min (2 new runs) |
| 06.6 | not needed | not needed | `patching_study.py` 29 s |
| 06.7 | optional (`nonar.py --variant main --device cuda`) | `--variant t4` | `nonar.py` 1.9 min |
| project | `run_project.py --variant lineage --lr <Baseline-0's lr>`: ~372M-parameter Lineage-F, 20 runs, PROJECTED ~60 GPU-hours by FLOPs at 20% MFU, up to 150+ with wall-clock factors (see the project page); pilot rungs `--variant main` / `main70` ~2.2 / ~4.5 GPU-hours | `--variant t4`, 8–12 T4-hours (PROJECTED) | 16 new runs 55 min (lessons' runs reused); report about a minute |

### Main path and pilot commands (for the Colab pilot notebook)

All on one GPU; every command resumes after a disconnect (rerun it). Record for each: GPU type, measured MFU
(`mfu` in `metrics.jsonl` with `--peak`), wall-clock per run (`wallclock.jsonl`), peak memory (`max_mem_gb`), and
the printed tables of the compare scripts.

```bash
# 06.1 — MTP at pilot scale (PROJECTED ~1.0 + ~1.7 GPU-hours on an H100)
python labs/module-06/lesson-01/train_arms.py --variant main --device cuda
python labs/module-06/lesson-01/train_arms.py --variant main70 --device cuda
python labs/module-06/lesson-01/compare.py --variant main --device cuda
python labs/module-06/lesson-01/acceptance.py --variant main --device cuda

# 06.2 — the HC vs mHC stability ladder (plan 12.1): 10M / 30M / 70M, standard and raised LR
python labs/module-06/lesson-02/train_arms.py --variant t4 --device cuda        # pilot-10m rung
python labs/module-06/lesson-02/train_arms.py --variant main --device cuda      # pilot-30m rung
python labs/module-06/lesson-02/train_arms.py --variant main70 --device cuda    # pilot-70m rung
python labs/module-06/lesson-02/compare.py --variant main --device cuda
python labs/module-06/lesson-02/step_time.py --device cuda --preset pilot-30m --batch 16 --seq 1024 --vocab 32768 --bf16

# 06.3 — the factorial's new cell
python labs/module-06/lesson-03/factorial.py --variant main --device cuda

# project — Lineage-F integration at pilot scale, then the ~372M main path
python labs/module-06/project/run_project.py --variant main --device cuda
python labs/module-06/project/run_project.py --variant main70 --device cuda
python labs/module-06/project/run_project.py --variant lineage --lr <Baseline-0's tuned lr> --device cuda
```

Notes:

- The main path needs Data-v0 at the main-path size (`python -m frontierlab.data.prepare --docs 2600000 --vocab 32768`,
  Module 1 project); the `main`/`main70` variants also assume vocabulary 32,768 for their FLOP projections.
- On a T4 use fp32 (the variants do). HC/mHC and the MoE expert loop are unfused PyTorch: their GPU MFU will be low;
  that is part of what the pilot measures.
- `--compile` is refused together with `--blocks-log`, and `--loss chunked` together with MTP or the diffusion objective.
- Evaluate Module 6 checkpoints with `frontierlab.blocks.train.load_model` (rebuilds BlockLM or DiffusionLM from the
  checkpoint's config); `m06.eval_losses` does.
- Seed replicates of BlockLM runs: `python -m frontierlab.record A B --replicates --changed config.extra.blocks.seed`
  until the record rule in `curriculum/inbox/module-06-shared-changes.md` lands.
