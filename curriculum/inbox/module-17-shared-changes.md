# Module 17 — proposed changes to shared files (for the main session)

Module 17 edits no existing `frontierlab` file, `_sidebar.md`, `glossary.md`, `references/`, `templates/`, plan or
PUBLISHING_WARNING file. New code: `labs/common/frontierlab/interp/` (hooks, superposition, sae, tasks, patching,
graphs, persona, steering, sparse, introspect, claims, hf), tests `labs/common/tests/test_interp.py`. It imports
`frontierlab.model`, `attention.base`, `train.loop`, `data`, `stats` and never patches them. It registers no attention
kind (no `attention/__init__.py` change needed).

## 1. `_sidebar.md`: Stage E line and Module 17 lines

Running numbers 72–76 (after Module 16's 67–71; fix them if earlier modules end elsewhere). 17.5 is an extension.
The Stage E line goes once, before Module 17 (Module 18's agent may propose the same line; keep one):

```markdown
- Stage E — Understanding, evaluating and communicating
- **Module 17 — What can we claim about a model's internals?**
  - [72 · Features and sparse autoencoders](lessons/module-17/lesson-01.md)
  - [73 · Causal interventions](lessons/module-17/lesson-02.md)
  - [74 · Transcoders and attribution graphs](lessons/module-17/lesson-03.md)
  - [75 · Steering and persona vectors](lessons/module-17/lesson-04.md)
  - [76 · Interpretable by design and introspection](lessons/module-17/lesson-05.md)
  - [Module 17 quiz](assessments/module-17-quiz.md)
```

The project page `projects/module-17-causal-claim.md` is picked up from `projects/` automatically.

## 2. `glossary.md`

Merge `curriculum/glossary-inbox/module-17.md` (51 terms).

## 3. `references/versions.md`

Add a section:

"## Module 17 models, dictionaries and tools (checked 2026-10-07)

| Item | Revision | Licence | Used in |
|---|---|---|---|
| Qwen/Qwen3-0.6B (free-CPU open model; post-trained, hybrid thinking) | `c1899de289a04d12100db370d81485cdf75e47ca` (as Module 10) | Apache-2.0 | 17.2, 17.4, 17.5, project (free CPU) |
| Qwen/Qwen3-1.7B-Base | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (as Module 12) | Apache-2.0 | 17.1, 17.2, project main path |
| Qwen/Qwen3-1.7B (post-trained) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (as Module 13) | Apache-2.0 | 17.3, 17.4, 17.5 main path |
| Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 (Qwen-Scope TopK SAEs, 32,768 latents, k = 50, residual stream after each of the 28 layers; SAELens release `qwen-scope-3-1.7b-base-w32k-l50`, ids `layer0`..`layer27`) | `ce1a79d9c5163932d65c417380e53230e1086370` | Qwen licence (`license: other`, with a use-restriction clause; read it) | 17.1 main path |
| mwhanna/qwen3-1.7b-transcoders-lowl0 (per-layer transcoders for Qwen/Qwen3-1.7B, not -Base; `mlp.hook_in` -> `mlp.hook_out`) | `9c1b17dfb156d82162ccd2cb7f047ac7f3d3585d` | MIT | 17.3 main path |
| mwhanna/qwen3-0.6b-transcoders-lowl0 | pin at the pilot | MIT (check) | 17.3 T4 |

Tool compatibility (from the packages' metadata, 2026-10-07): sae-lens 6.53.0 requires `transformer-lens>=2.16.1,<4.0.0`;
transformer-lens 4.0.0 requires `transformers>=5.9.0` and removed `HookedTransformer.from_pretrained` (use
`TransformerBridge.boot_transformers(...)`, optionally `enable_compatibility_mode()` for `blocks.L.hook_resid_post` /
`blocks.L.attn.hook_z` names; Qwen3-1.7B and -Base are in its supported_models with full verification);
circuit-tracer 0.5.0 (2026-03-29) requires `transformers>=4.56.0,<=4.57.3`, `transformer-lens>=2.16.0`, `nnsight>=0.6.0`,
`huggingface-hub<1.0.0`. So Module 17 uses two main-path environments: env A (course default: torch 2.14.1,
transformers 5.18.0, nnsight 0.7.0, sae-lens 6.53.0 used only to load SAEs, transformer-lens 4.0.0 optional for
cross-checks — install sae-lens with `--no-deps` or accept its TL<4 pin in a separate venv) and env B (17.3 only:
`pip install circuit-tracer==0.5.0` in its own venv)."

and under "Notes on reference implementations":

"- Module 17: SAELens v6.53.0 `SAE.from_pretrained(release, sae_id, device=..., dtype=...)` returns only the SAE
(`from_pretrained_with_cfg_and_sparsity` returns the tuple); `sae.encode`, `sae.decode`; hook name in
`sae.cfg.metadata.hook_name`. circuit-tracer v0.5.0: `ReplacementModel.from_pretrained(model_name, transcoder_set,
backend="transformerlens"|"nnsight", dtype=...)`, `attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95,
batch_size=512, max_feature_nodes=None)`, `graph.prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)`,
`compute_graph_scores(graph)`, `ReplacementModel.feature_intervention(inputs, [(layer, pos, feature, value)])`.
nnsight 0.7.0: with transformers >= 4.57 `Qwen3DecoderLayer.forward` returns a tensor, so use
`model.model.layers[i].output`, not `.output[0]` (which would select batch item 0)."

No new CPU requirements (`tokenizers` is already in the CPU requirements). The main path needs `sae-lens==6.53.0`,
`nnsight==0.7.0`, optionally `transformer-lens==4.0.0` (env A) and `circuit-tracer==0.5.0` (env B).

## 4. Pilot commands (Colab pilot notebook) — all PROJECTED, none run in this build

Smoke test first (CPU, seconds): `python -m frontierlab.interp.hf smoke`.

| Lesson | Command | PROJECTED cost | Pilot question / what to record |
|---|---|---|---|
| 17.1 | `for L in 7 14 21; do python -m frontierlab.interp.hf sae-eval --layer $L --out runs/m17/qwen-scope-l$L.json; done` | minutes; < 0.25 GPU-h | does `qwen_scope_sae` find exactly one `.pt` per layer and load the shapes (record file names and shapes); `hf.check_against_library(14, acts)` against SAELens 6.53.0 (codes must agree; if not, the published SAE does not subtract `b_dec` before encoding — record it and fix `qwen_scope_sae`); splice check; FVU, L0, delta loss per layer |
| 17.1 | `python labs/module-17/lesson-01/sae_lab.py --variant main --layer 14 --tokens 4000000 --d-sae 16384 --k 50` | collection 1.1e16 FLOPs + SAE training 6.4e16 FLOPs; < 1 GPU-h; 16 GB host RAM (bf16 activations) + 32 GB GPU (fp32 copy) | course SAE vs Qwen-Scope at the same site: delta loss, FVU, dead fraction; wall-clock of collection |
| 17.2 | `python -m frontierlab.interp.hf ioi --model qwen3-1.7b-base --n 96 --top 10 --random 49 --out runs/m17/ioi-1.7b-base.json` | 6.5e14 FLOPs; ~10 min; < 0.25 GPU-h | candidate heads, ablation effect, random-set distribution, held-out effect, off-target loss; does the claim card pass? |
| 17.3 | env B: `circuit-tracer attribute --prompt "Fact: the capital of the state containing Dallas is" --transcoder_set mwhanna/qwen3-1.7b-transcoders-lowl0 --slug dallas --graph_file_dir runs/m17/graphs`, then the Python steps printed by `python labs/module-17/lesson-03/graph_lab.py --print` | ~1-3 min per prompt on an A100; 10 prompts + 60 interventions ~0.5 GPU-h | does circuit-tracer 0.5.0 run with the Qwen3-1.7B set (plan 12.1 row "17.3 attribution graphs"); graph scores; predicted vs real intervention effects against random features |
| 17.4 | `python labs/module-17/lesson-04/steer_lab.py --model qwen3-1.7b --layers 8 12 16 20 --device cuda --out runs/m17/steer-1.7b.json` | 1.6e15 FLOPs; 10-15 min; < 0.25 GPU-h | chosen layer, dose response on held-out templates, random-direction and two-sided controls, side effects |
| 17.5 | `python labs/module-17/lesson-05/sparse_lab.py --skip-a --hf --model qwen3-1.7b --layers-inject 14 18 22 --alphas 2 4 8 --device cuda` | ~38,000 decode steps; 20-40 min; < 1 GPU-h | detection and false-positive rates with Wilson intervals; transcripts |
| project | `python labs/module-17/project/run_project.py --kind ioi --model qwen3-1.7b-base --device cuda --n 96 --random 39` | as 17.2; < 0.5 GPU-h | the claim card and its check |

Main path total for the module: about 3–4 GPU-hours (PROJECTED), USD 6–12 at USD 2–3 per H100-hour.

## 5. PUBLISHING_WARNING.md

The existing 17.4 paragraph still holds. Suggested addition to it (exact text):

"Lesson 17.4's datasets (`frontierlab/interp/persona.py`) contain only A/B items about simple facts (capitals, planets,
arithmetic) where the trait is agreeing with the user's stated answer; the contrastive system prompts ask the assistant
to agree with the user or to give the correct answer. Lesson 17.5's concept-injection words are everyday nouns
(`introspect.WORDS`). Review any items or words later added to these lists. The steering functions in
`frontierlab/interp/steering.py` are generic; no course dataset, command or lab step extracts or removes a refusal or
safety direction."

## 6. Plan section 14.1 / 14.2 claim checks to record (all checked 2026-10-07 against the primary source)

- V: Toy Models of Superposition (2022-09-14): ReLU output model, importance-weighted MSE, phase change, antipodal pairs
  / pentagons / tetrahedra, D* = m/‖W‖_F² ("The Geometry of Superposition").
- V: Towards Monosemanticity (2023-10-04): 1-layer transformer, 512-neuron MLP, 512 to 131,072 features, unit-norm decoder
  columns, tied pre-bias, L2 + L1; "79% of the log-likelihood loss reduction provided by the MLP layer is recovered"
  (4,096 features, vs zero-ablating the MLP); neuron resampling.
- V: Scaling Monosemanticity (2024-05-21): Claude 3 Sonnet middle-layer residual stream; 1M/4M/34M features; < 300
  active per token; >= 65% variance explained; dead ~2%/35%/65%; Golden Gate feature clamped to 10× max with the error
  term unchanged.
- V: Gao et al. 2406.04093: TopK Eq. 2; 16M latents on GPT-4 for 40B tokens; transpose init, AuxK (alpha 1/32,
  k_aux 512); dead = no activation in 10M tokens; 7% dead at 16M; normalised MSE; metrics section 4.
- V: JumpReLU 2407.14435: Eq. 4; L0 penalty; STEs Eqs. 11–12; rect kernel; epsilon 0.001 with E[x²] = 1; Gemma 2 9B
  layers 9, 20, 31; "at least as good as, and often slightly better than, TopK".
- V: Gemma Scope 2408.05147: all layers and sublayers of Gemma 2 2B and 9B, selected 27B; > 400 SAEs, > 30M features;
  widths 2^14–2^20; delta LM loss primary, FVU secondary; transcoders for 2B. Gemma Scope 2 (2025-12): Gemma 3
  270M–27B, SAEs, transcoders, CLTs for 270M and 1B (HF model card; the technical report PDF was not fetched).
- V: ROME 2202.05262 §2.1 (causal tracing, 3σ noise, mid-layer MLPs at the last subject token); IOI 2211.00593
  (26 heads in 7 classes; path patching; faithfulness/completeness/minimality; pABC mean ablation); Heimersheim &
  Nanda 2404.15255 (§2.3 noising vs denoising, §4.1 logit difference, §5 recommendations); Zhang & Nanda 2309.16042
  (STR over Gaussian noising; logit difference); Makelov et al. 2311.17030 (dormant parallel pathway; §8
  recommendations); Olsson et al. 2022 (induction, K-composition, phase change); Elhage et al. 2021 (Q/K/V
  composition); Nanda 2023 attribution patching (formula and limitations); Syed et al. 2310.10348 (abstract).
- V: Transcoders 2406.11944 (Eqs. 3–5, §3.2.1 factorisation; GPT-2 small, Pythia 410M/1.4B); Circuit Tracing
  (2025-03-27: CLT, local replacement model, error nodes, pruning "factor of 10 ... 20%", 18L CLT matches next-token
  completion on 50%; Haiku CLT 21.7% normalised error, L0 235; limitations list); Biology (Claude 3.5 Haiku; Dallas →
  Texas → Austin with the California swap → Sacramento; "satisfying insight for about a quarter of the prompts");
  Anthropic open-sourcing post (2025-05-29; Gemma-2-2b, Llama-3.2-1b; Neuronpedia).
- V: CAA 2312.06681 (answer-letter mean difference; 7 behaviours incl. sycophancy; Llama 2 7B layer 13; MMLU "does not
  significantly affect"); ActAdd 2308.10248; Persona Vectors 2507.21509 (traits; 5 system-prompt pairs, 40 questions,
  judge; Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct; projection r = 0.75–0.83; inference-time steering degrades
  MMLU at large coefficients; preventative steering preserves capability; data flagging by projection difference);
  RepE 2310.01405.
- V: Weight-sparse transformers 2511.13653 (§2.1 magnitude top-k after each step, ~1 in 1000 nonzero at the sparsest,
  L0 annealed over the first 50%; §2.2 learned masks, target loss 0.15, mean ablation; Fig. 2 "roughly 16-fold smaller";
  closing-quote circuit 12 nodes / 9 edges; bridges §2.3). The OpenAI blog post returned 403 (not used).
- V: Introspection (Lindsey, 2025-10-29): concept vectors, ~20% detection for Claude Opus 4.1 at the best layer and
  strength, 0 false positives in 100 no-injection trials, "highly unreliable and context-dependent".
- NOT VERIFIED and therefore not stated in the lessons: the exact title of the Qwen-Scope paper (arXiv 2605.11887 per
  the model card); the Qwen-Scope `.pt` file names (the loader searches for them; record at the pilot); a Haiku
  replacement-model match rate; the exact circuit-tracer pruning thresholds as stated in the paper (they are the
  library's defaults).
