# Module 17 labs — What can we claim about a model's internals?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`), `solution.py`
(the reference), `test_lab.py` and the script the lesson runs. The shared code is `labs/common/frontierlab/interp/`,
with tests in `labs/common/tests/test_interp.py`. It is written from scratch on PyTorch forward hooks and works on
the course's own models and on Hugging Face Qwen3 models; the production libraries (SAELens, TransformerLens,
nnsight, circuit-tracer) appear only on the main path, with a mapping in `frontierlab/interp/hf.py`:

| Module | What it does |
|---|---|
| `interp/hooks.py` | sites by name (`resid_pre.L`, `resid_post.L`, `attn_out.L`, `z.L`, `mlp_in.L`, `mlp_out.L`, `final`), capture, run with edits, left padding; hooks always removed |
| `interp/superposition.py` | the toy model of superposition and its geometry measures |
| `interp/sae.py` | ReLU, TopK (with AuxK) and JumpReLU (straight-through estimators) SAEs; FVU, L0, dead latents; spliced losses and the splice check; top contexts |
| `interp/tasks.py` | the induction model (a known mechanism), clean/corrupt pairs, IOI prompts, the Module 17 language model |
| `interp/patching.py` | denoising and noising, head sweeps, attribution patching, path patching, ablations, random-direction and random-component controls, off-target loss |
| `interp/graphs.py` | transcoders, the explicit forward pass, the local replacement model, attribution graphs, influence and pruning, feature interventions |
| `interp/persona.py`, `interp/steering.py` | harmless A/B sycophancy items with extraction / evaluation / held-out splits; difference-of-means vectors, steering, dose response, side effects, projection |
| `interp/sparse.py`, `interp/introspect.py` | weight-sparse training and circuit pruning on the closing-quote task; the concept-injection harness |
| `interp/claims.py` | the project's claim card and its mechanical check |
| `interp/hf.py` | pinned models and dictionaries, the IOI study, the Qwen-Scope SAE loader and evaluation, circuit-tracer commands |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_interp.py              # the shared Module 17 code (no downloads, a few minutes)
pytest labs/module-17/lesson-01                      # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-17            # all reference solutions and the project's harness tests
python -m frontierlab.interp.hf smoke                # main-path code on a tiny random Qwen3, CPU, seconds
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every script
writes under `runs/m17/` (gitignored) and reuses what it has already trained. Lessons 17.1 and 17.3 need Data-v0
(`python -m frontierlab.data.prepare --docs 20000 --vocab 8192`, Module 1); 17.1 trains the Module 17 language
model (600 steps of the course loop) the first time and 17.3 reuses it. Lessons 17.2 (`--hf`), 17.4, 17.5 (`--hf`)
and the project's IOI run download Qwen/Qwen3-0.6B at revision `c1899de` (1.5 GB).

> [!NOTE]
> Lesson 17.4 steers harmless traits only: agreeing with a user's stated answer to a simple factual question.
> No lab extracts, removes or adds a refusal or safety direction (see `PUBLISHING_WARNING.md`).

## What each folder contains

| Folder | Lesson | Script | What it does |
|---|---|---|---|
| `lesson-01/` | 17.1 Features and sparse autoencoders | `sae_lab.py` | the toy model of superposition at four sparsities; six SAEs (ReLU, TopK, JumpReLU × two settings) on the Module 17 model's residual stream: FVU, L0, dead latents, splice check, delta loss with paired intervals, loss recovered; top contexts of three latents |
| `lesson-02/` | 17.2 Causal interventions | `patch_lab.py` | the induction model; head patching in both directions under two corruptions; attribution vs real patching; path patching into q/k/v/logits; random-head and random-direction controls; held-out gaps and query positions; `--hf`: the IOI study on Qwen3-0.6B |
| `lesson-03/` | 17.3 Transcoders and attribution graphs | `graph_lab.py` | four transcoders and the replacement model; three attribution graphs with pruning and error share; predicted vs real interventions for top and random features |
| `lesson-04/` | 17.4 Steering and persona vectors | `steer_lab.py` | baseline sycophancy of Qwen3-0.6B; CAA vectors at three layers; layer and strength chosen on the selection split; held-out dose response; random-direction, two-sided and shuffled-label controls; side effects; projection monitoring; two generations |
| `lesson-05/` | 17.5 Interpretable by design and introspection (extension) | `sparse_lab.py` | dense vs 5%-dense models on the closing-quote task pruned to their circuits; `--hf`: concept injection on Qwen3-0.6B with no-injection and random-vector controls |
| `project/` | Module project | `run_project.py`, `harness.py`, `buggy_harness.py`, `test_harness.py` | one causal claim end to end (induction rehearsal, IOI on Qwen3-0.6B or Qwen3-1.7B-Base) with a claim card; the planted-bug debugging task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 17 pilot, and every main-path figure in
the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-07 on a 16-thread Windows 11 laptop
(torch 2.14.1+cpu, transformers 5.18.0, 8 threads per script), part of the time with other jobs running.

| Lab | Main path (PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 17.1 SAEs | under 1 GPU-hour (Qwen-Scope evaluation at 3 layers, a course SAE on 4M tokens) | `sae-eval` at one layer, float32 | `sae_lab.py` 3.1 min, plus about 8 min the first time to train the Module 17 model |
| 17.2 patching | under 0.25 GPU-hours (IOI on Qwen3-1.7B-Base) | the same at n = 64 | `patch_lab.py` 7.4 min including the induction model (other jobs running); `--hf` adds 14.5 min |
| 17.3 graphs | about 0.5 GPU-hours, separate circuit-tracer environment | Qwen3-0.6B transcoder set | `graph_lab.py` 76 s after 17.1 (105 s with other jobs running) |
| 17.4 steering | under 0.25 GPU-hours (Qwen3-1.7B) | Qwen3-1.7B in float16 | `steer_lab.py` 26 min (19 random directions for each control) |
| 17.5 sparse, introspection | under 2 GPU-hours | (B) one layer | `sparse_lab.py --hf` 312 s |
| Project | under 0.5 GPU-hours | Qwen3-1.7B-Base, n = 64 | induction rehearsal 2 s after 17.2's model; IOI on Qwen3-0.6B 13.7 min |

Notes:

- Main path downloads: Qwen3-1.7B-Base (3.4 GB), Qwen3-1.7B (4.1 GB), the Qwen-Scope SAEs (one layer: 2 × 2048 ×
  32768 values, about 0.5 GB in fp32; Qwen licence, read it), the Qwen3-1.7B transcoder set (MIT).
- circuit-tracer 0.5.0 pins `transformers<=4.57.3`: install it in its own environment (lesson 17.3).
- Run `python -m frontierlab.interp.hf smoke` (CPU) before any GPU session to check the install.
