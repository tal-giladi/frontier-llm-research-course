> **Course guidelines (binding):** before writing or changing any course content, read
> `C:\Users\TalGiladi\OneDrive\repos\tals-academy\docs\new-course-instructions.md`. It wins over
> anything in this file or the course plan unless Tal says otherwise in chat.

# Plan — Frontier LLM Research Engineering (advanced follow-up course)

Status: plan revision 2 (2026-10-03), after an external review (change log: section 13). Nothing built yet.
Parent course: `..\full-ml-engineer-course` ("LLM Research Engineer").
Sibling follow-ups this course links to instead of duplicating: `..\mixture-of-experts-engineering-course`
(deep MoE engineering), `..\ai-inference` (serving operations), `..\llm-evals-course` (application evals).

Research basis: two research passes over primary sources from 2023 to 2026 (arXiv technical reports,
lab blogs, model cards, system cards, safety frameworks), plus a claim-level check of the specific
technical statements this plan relies on (section 14.1). Anything a lab has not disclosed is labelled
INFERENCE here and must stay labelled in the lessons.

**Target graduate:** someone who can design, execute, debug and defend a research experiment on
LLM architecture, training or post-training — not someone who has merely implemented a list of
recent techniques.

---

## 1. Why this course exists

The parent course ends roughly where Stanford CS336 ends: the learner can build, pretrain, scale-plan,
fine-tune, align (PPO/DPO/GRPO) and evaluate a small GPT-style model, and has seen RoPE, GQA, a
DeepSeek-style MoE layer and one lesson on linear/hybrid attention.

Research engineers at frontier labs spend their time on *decisions under uncertainty* that the parent
course does not train: which attention design is worth its complexity at a given context length and
hardware; whether an optimizer or precision recipe that worked for one lab will transfer; which data
change actually caused an improvement; which RL objective is stable for a given model; whether an
interpretability finding supports a causal claim. Published models show many different answers —
open releases from 2024–26 differ in attention (dense GQA, MLA, sliding/global mixes, learned sparse
selection, linear/full hybrids), optimizer (AdamW, Muon), precision (BF16, FP8, FP4 experts), and
post-training recipe. There is no single "frontier block"; there are trade-offs, a few well-replicated
techniques, many promising ones, and many choices that are specific to one model.

This course teaches those decisions. Each module is organised around a research question, uses
published models as case studies, and ends in a controlled experiment whose result — positive or
negative — the learner has to defend.

## 2. Audience, entry point, goal

- **Who:** graduates of the LLM Research Engineer course, or engineers who can already write and
  train a GPT from scratch, profile it, write a basic Triton kernel, use DDP/FSDP and run
  SFT/LoRA/DPO/GRPO.
- **Assumed, not re-taught** (linked instead): parent modules 05–20 and the frontier updates.
  Module 1 opens with a diagnostic whose explanations link back to the parent lessons.
- **Scope:** the required course is **text, reasoning and agents**. Native multimodality is a
  separate elective module (Module 21) and is not a prerequisite for anything.
- **Goal state (observable):** the learner can
  1. turn a vague question ("is MLA better?") into a testable hypothesis with a baseline, controls,
     a stated comparison axis (equal tokens / parameters / training FLOPs / wall-clock), a budget and
     a pre-stated decision rule;
  2. measure noise (seed variance, eval variance) and report effects with uncertainty;
  3. implement a mechanism from its paper and prove it correct (gradient checks, causal-mask tests,
     cached-decode agreement, recurrent-vs-parallel equivalence, agreement with a reference);
  4. profile it and explain the gap between theoretical FLOPs and measured time, then validate one
     improvement;
  5. run and debug training, distributed and post-training experiments at small production scale,
     including failure recovery;
  6. make a causal interpretability claim with controls and stated limits;
  7. design an evaluation that survives contamination and reports uncertainty;
  8. reproduce a recent paper's claim at small scale, extend it, and defend the result in writing.

## 3. Principles

1. **Research questions first, models as case studies.** A module title is a question ("When is
   sub-quadratic attention worth it?"). Models (DeepSeek, Qwen, Kimi, gpt-oss, Gemma, Llama, OLMo,
   GLM, MiniMax) appear as evidence for one answer under one set of constraints.
2. **Maturity tags.** Every technique is tagged in its lesson as **ESTABLISHED** (adopted
   independently by several labs, with ablations in more than one report), **PROMISING** (published
   with evidence, limited independent replication) or **MODEL-SPECIFIC** (one lab's choice, evidence
   mostly from that lab). Tags in section 5 are provisional and are re-checked when each lesson is
   written.
3. **Evidence labels.** PUBLICLY DOCUMENTED (with report section), REASONABLE INDUSTRY PRACTICE,
   INFERENCE/SPECULATION, and (company claim) for vendor numbers. A course experiment at 100M–2B
   scale is never presented as proof of a frontier-scale claim. Closed labs (OpenAI GPT-5, Gemini 3,
   Grok) are used only for what they disclose.
4. **Experimental discipline before experiments.** Module 1 teaches hypotheses, controls, splits,
   comparison axes, tuning budgets, uncertainty and reproducibility. Every later lab uses them.
5. **Every substantial lab has an experiment contract** (section 9) and is graded on soundness, not
   on whether the new method wins. A well-run negative result passes.
6. **Hypotheses are not acceptance criteria.** Where an effect may not appear at course scale
   (stability gains, reward hacking, emergent misalignment, RL asymptotes), the lab states it as a
   hypothesis, is piloted at reduced scale before publishing (section 12.1), and ships the pilot's
   checkpoints or traces so the learner can still investigate it. Analysis of provided artefacts is always labelled
   as analysis, never as reproduction.
7. **Correct, then fast, then compared.** Implementations pass correctness checks before any
   benchmark; benchmarks follow the Module 2 methodology before any comparison.
8. **Mechanism first, framework second** (inherited). Own code in the lab package first, then the
   production tool: torchtitan (pretraining), Megatron-Core (by mapping table), flash-linear-attention
   (linear kernels), verl or slime (RL), vLLM/SGLang (rollouts and serving), SAELens / circuit-tracing
   tools (interpretability). Versions pinned at Phase 0. A framework feature is taught as available
   only if the course has tested it for the architecture in use (tested-capability matrix, 09.3).
9. **Debates are taught as debates**, with the experiment that informs each side: linear hybrids vs
   full attention (MiniMax-M2's return to full attention), whether RLVR adds capability or sharpens
   sampling (Yue et al.), emergence vs metric artefacts, benchmark contamination.
10. **Dated and volatile.** Model facts live in a dated appendix (`references/frontier-models-2026-10.md`).
    Lessons on specific models are `volatility: implementation`; mechanisms are `volatility: concept`.
    New releases are added as new lessons at the end of the relevant module (paths are stable ids).
11. **No duplication of siblings.** Deep MoE routing, balancing, grouped GEMM and expert parallelism
    are in the MoE course; here MoE is a component (copied from `moelab`) with a link. Serving
    operations are in the inference course; here inference appears only where it changes
    architecture, RL or test-time-compute decisions.

## 4. Hardware

Decided with Tal (2026-10-03): production-grade course; **the main path runs on rented GPUs**. Two
free variants exist so nobody is locked out of the mechanism, but they are secondary: they never drive
a curriculum decision, never delay the main path, and are written after the lab's pilot.

- **Main path — rented GPU (the course standard).** Hardware stated per lab, from 1× 48–80 GB GPU
  (L40S/A100/H100) to one 8× H100/H200 node; NVFP4 labs need Blackwell (B200) or use the documented
  fallback. Every lab states GPU type, GPU-hours, unattended runtime and estimated cost at prices
  checked at Phase 0 (dated).
- **Free GPU — Colab/Kaggle T4.** Same lab at reduced scale where it fits a free session; the lab
  says what the learner will not see at this scale.
- **Free CPU — laptop.** The mechanism and its correctness tests at toy scale; emulated low precision
  (labelled); CPU multi-process collectives (real collectives, not interconnect performance).

Where a free variant cannot show the point at all, it analyses the course-provided pilot traces,
labelled as trace analysis.

**Course-side compute (decided 2026-10-03):** the course has no funding for rented GPUs. All course-side
work (pilots, provided artefacts, cost measurements) runs on **one paid Colab account**, limited by its
monthly compute units: single GPU (A100 / L4 / T4; H100 if the account offers it), sessions that can
disconnect, no multi-GPU. Pilots are therefore **scaled pilots** (section 12.1): small samples chosen
to represent the main-path behaviour, with the main-path time and cost projected from them and
labelled as projections.

**Course-provided artefacts** (only what fits that budget): Data-v0 manifests and shards,
the Baseline-0 checkpoint if the ~125M run fits the allowance, scaled-pilot traces for every
hypothesis lab, released third-party checkpoints and model organisms where licences allow. No
course-produced long RL reference runs (14.4 uses published curves instead). Hosted on a free Hugging
Face account (decided 2026-10-03).

## 5. Course map

Stages are taken in order A → B → C → D → E, except for the permitted reorderings in section 6.
R = required, X = extension (keeps its explanation and lab, optional quiz; module quizzes draw only
from required lessons). [E]/[P]/[M] = provisional maturity tag (ESTABLISHED / PROMISING /
MODEL-SPECIFIC). Running lesson numbers are assigned in `_sidebar.md`.

### Stage A — Research foundations

#### Module 1 — How do we know a change helped?
- 01.1 R · Diagnostic and map — the gap from parent 12.x to current open models; diagnostic quiz linked to parent lessons.
- 01.2 R · Reading reports and configs as evidence — structure of a technical report; reconstructing an architecture from `config.json` (DeepSeek-V3, Qwen3, gpt-oss, Gemma 3) with a parameter / active-parameter / FLOPs-per-token / KV-bytes calculator; what closed labs disclose (GPT-5 system card, Gemini 2.5 report, Gemini 3 model card); evidence labels and maturity tags.
- 01.3 R · Designing an experiment — hypothesis, baseline, single changed variable, controls; the four comparison axes (equal tokens, equal parameters, equal training FLOPs, equal wall-clock) and the different questions they answer; tuning budgets (a baseline tuned as hard as the method); pre-stated decision rules; train/validation/test separation for data and for hyperparameter selection.
- 01.4 R · Uncertainty — seed variance and the noise floor, eval sampling variance, paired comparisons, bootstrap confidence intervals, multiple comparisons, when a difference is real, power for a planned ablation.
- 01.5 R · Reproducibility and the experiment record — pinned environment, config, seeds, data hashes, tracker, run card; what has to be identical for two runs to be comparable.
- Project: **Baseline-0** — train and tune the reference model (dense, GQA, RoPE, SwiGLU, ~125M params, Data-v0), measure its noise floor across 3 seeds, and publish Eval Suite v0 (held-out loss, short-context downstream set with intervals). Every later architecture and recipe experiment branches from this artefact.

#### Module 2 — Where does the time go? (performance engineering, required thread)
- 02.1 R · From FLOPs to time — roofline, arithmetic intensity, HBM bandwidth, MFU vs HFU at current shapes; predicting step time before measuring.
- 02.2 R · Profiling a training step — kernel timeline, launch overhead, fusion opportunities, torch.compile, activation memory and memory snapshots, activation checkpointing trade-offs.
- 02.3 R · Communication and overlap — measuring exposed communication with DDP and FSDP2 on 2–8 GPUs; overlap, bucket sizes, what the profiler shows.
- 02.4 R · Validating a performance claim — benchmark methodology (warm-up, synchronisation, repeats, variance, representative shapes), then one improvement to Baseline-0 measured end to end.
- 02.5 R · FlashAttention-2, 3 and 4 — **TODO (added 2026-10-09, not written yet; placeholder page in the sidebar).** What FA-2 (work partitioning, fewer non-matmul FLOPs), FA-3 (Hopper: warp specialisation, async TMA/WGMMA, FP8) and FA-4 (Blackwell) change over FA-1; which one production training uses on which GPU; benchmark on Baseline-0.
- Project: performance report on Baseline-0 — predicted vs measured step time, explanation of the gap, one validated improvement. Required for all later systems comparisons.

### Stage B — Architecture questions

Every Stage B lab branches from Baseline-0 with one change, same Data-v0, same budget unless the
contract says otherwise, and must pass the architecture correctness suite before any comparison:
gradient check (float64, toy size), causal-mask test (perturbing future tokens changes nothing),
full-sequence vs cached-decode agreement, recurrent-vs-parallel equivalence where applicable, and
agreement with a reference implementation where one exists.

#### Module 3 — How should attention spend KV memory?
- 03.1 R · Multi-head Latent Attention [E] — low-rank joint KV compression, the decoupled RoPE key and why RoPE prevents absorbing the up-projection, weight absorption at inference; tests: naive vs absorbed equality, cached decode agreement, gradients.
- 03.2 R · Local/global attention and KV arithmetic [E] — sliding-window and interleaved local/global layers (Gemma 3, gpt-oss) with a rolling KV cache and its decode-agreement test; KV bytes per token for MHA/GQA/MQA/MLA/mixes at 32K–1M.
- 03.3 R · Logit control, sinks and gating [E/P] — QK-norm, attention sinks (StreamingLLM; gpt-oss learned sink logit), head-wise output gating (Qwen gated attention); measuring sink mass and massive activations; claims stated as hypotheses at course scale.
- 03.4 X · Head count and head dimension [M] — Kimi K2's 64-head choice, head dim 256 with partial RoPE (Qwen3-Next).
- Project: controlled comparison of GQA / MLA / local-global vs Baseline-0 at equal parameters **and** at equal training FLOPs, with quality (Eval v0), measured decode memory and latency; a decision memo for a stated serving constraint.

#### Module 4 — Does the model use its context?
- 04.1 R · Long-context evaluation that means something — retrieval is not enough: multi-hop use, distractors, position sensitivity, effective vs advertised length, and short-context regression; builds **Eval Suite v1** (long-context component).
- 04.2 R · Position at long range [E] — position interpolation, NTK-aware scaling, YaRN, partial RoPE; interleaving NoPE layers (Llama 4 iRoPE [M]); worked numbers per frequency.
- 04.3 R · Extending context by continued training [E] — staged extension (Llama 3, DeepSeek-V3.1 as documented), long-document data, document boundaries; measuring the gain and the short-context cost.
- Project: extend Baseline-0 from 2K to 32K; report what it can and cannot use on Eval v1, with short-context regression.

#### Module 5 — When is sub-quadratic attention worth it?
- 05.1 R · Linear and hybrid attention at scale [P] — from parent 12.5 to chunked Gated DeltaNet / Kimi Delta Attention; recurrent vs chunked-parallel equivalence test; hybrid layouts (Qwen3-Next 3 linear : 1 full; Kimi Linear) as case studies.
- 05.2 R · Learned sparse attention [P/M] — NSA (compression, selection, sliding branches) and DeepSeek Sparse Attention: how the indexer is trained (dense warm-up aligning the indexer to the main attention distribution, then sparse training with the indexer optimised by its own objective, separate from the language-modelling loss, as documented in the V3.2 report), top-k selection, and the honest cost model — selected attention is O(L·k) but the all-prefix indexer is still O(L²), so the win depends on the indexer's constant factor, memory traffic and kernel overhead.
- 05.3 R · Measuring cost and quality honestly — profile dense FlashAttention vs the linear hybrid vs DSA-style selection from 8K to 128K: indexing, selection, memory traffic, kernel launches; find the crossover; evaluate on Eval v1; the MiniMax-M2 argument as the counter-hypothesis.
- 05.4 X · Compressed sparse attention (DeepSeek-V4 CSA/HCA) [M] — case study of the V4 design as documented.
- Project: a sub-quadratic decision memo for a stated context length, hardware and quality bar, defended with your measurements, including the case where dense attention wins.

#### Module 6 — Which other block changes earn their complexity?
- 06.1 R · Multi-token prediction [P] — Meta MTP heads vs DeepSeek's sequential MTP modules; loss weighting; reuse as a draft (acceptance rate measured in 15.2).
- 06.2 R · Residual-stream design: hyper-connections and mHC [P] — widening the residual stream; why unconstrained mixing breaks the identity path; Sinkhorn projection onto doubly-stochastic matrices. The stability claim is a hypothesis at course scale; validated pilot traces provided.
- 06.3 R · Combining changes — interaction effects, factorial-lite designs, why "best of each branch" is not a plan; budget for an integration run.
- 06.4 X · Conditional memory and lookup sparsity: Engram [P] — hashed N-gram lookup as a separate sparsity axis next to MoE (not a residual-stream design); the allocation result as reported.
- 06.5 X · Elastic architectures [M] — MatFormer nested FFNs, Per-Layer Embeddings (Gemma 3n/4).
- 06.6 X · Tokenizer-free models [P] — Byte Latent Transformer entropy patching.
- 06.7 X · Non-autoregressive and latent reasoning [P] — LLaDA masked diffusion, Coconut, recurrent depth; Gemini Diffusion only as disclosed.
- Project: **Lineage-F integration experiment** — combine the Module 3–6 branches that passed their own tests (plus the MoE component from the MoE course) at ~350M params; compare against Baseline-0 scaled to the same budget; report interactions and whether the combination beats its best single branch. It is an experiment, not an assumed upgrade.

### Stage C — Training-recipe questions

Stage C experiments run on the architecture the learner chose in Stage B (or Baseline-0), producing
**Recipe-R**: the optimizer, precision and data choices carried into the final pretraining run.

#### Module 7 — Which optimizer and parametrization?
- 07.1 R · Muon from scratch [E in open labs, recent] — momentum + Newton–Schulz orthogonalisation; which parameters stay on AdamW; cost; tests (orthogonality, convergence of the iteration).
- 07.2 R · Muon at scale [P] — weight decay and update-RMS matching to AdamW (Moonlight); attention-logit growth and its fixes: QK-Clip (Kimi K2) vs QK-norm (DeepSeek-V4 as documented).
- 07.3 R · Hyperparameter transfer [E] — µP and µTransfer with worked scaling rules; designing a transfer test; Muon + µP.
- 07.4 R · Schedules [E] — WSD, decay branches, cooldown, continued training.
- 07.5 R · Stability forensics — which logged statistics predict trouble; diagnosing spikes from logs (logit growth, z-loss, soft-capping vs QK-norm, norm and activation clamping); course-provided spike traces plus deliberately induced failures.
- Project: optimizer decision report — Muon vs tuned AdamW at equal tokens **and** equal wall-clock, with tuning budgets reported and a transfer test across two widths.

#### Module 8 — How low can precision go?
- 08.1 R · Number formats and scaling [E] — FP8 E4M3/E5M2, MXFP8/MXFP4, NVFP4, power-of-two scales; block scaling; error analysis with worked numbers.
- 08.2 R · FP8 training [E] — the DeepSeek-V3 fine-grained recipe (tile/block scaling, higher-precision accumulation) and torchao Float8 on H100; what stays in BF16; measured throughput and loss gap.
- 08.3 R · FP4 training and quantisation-aware training [P] — NVFP4 pretraining recipe (NVIDIA), FP4 expert weights (DeepSeek-V4), MXFP4 post-training (gpt-oss), INT4 QAT (Kimi K2 Thinking); B200 main path or labelled emulation.
- 08.4 X · Scaling laws for precision [P] — the reported compute-optimal precision and PTQ-degradation results; fit on provided sweeps.
- Project: precision plan for a stated model and hardware with measured throughput and an error budget.

#### Module 9 — What does the cluster cost, and how does it fail?
- 09.1 R · Parallelism layouts at scale [E] — TP/CP/PP/DP/EP ordering and why (Llama 3, DeepSeek-V3 as documented); context parallelism and Ring Attention; layout planner for memory and communication per device.
- 09.2 R · Pipeline schedules and overlap [E/M] — 1F1B, interleaved, zero-bubble ideas, DualPipe; schedule simulator, then measured bubbles on a real pipeline.
- 09.3 R · A measured multi-GPU investigation — torchtitan on one 8-GPU node: throughput, exposed communication, memory per rank, for two layouts; a **tested-capability matrix** recording which framework features actually composed for the chosen architecture (vs the documented feature list).
- 09.4 R · Failure and recovery — kill and restart a run; verify recovery of model, optimizer, scheduler, RNG and data-stream position; measure checkpoint overhead and useful progress after restart; restore into a different layout; goodput model.
- 09.5 X · TPUs, JAX and hardware co-design — TPU topology and Pathways, the JAX Scaling Book notation; 3FS and DeepSeek's hardware reflections.
- Project: infrastructure plan for a 1T-total / ~40B-active model on a stated cluster, with every assumption that can be measured at node scale grounded in 09.3–09.4 measurements.

#### Module 10 — Which data, in which mix?
Builds on parent 04.3 (packing), 07.3 (resume), 11.1–11.3 (extraction, heuristic and model-based filtering, MinHash dedup, n-gram contamination, mixing budgets) and 14.2 (block-diagonal packing); those are not re-taught.
- 10.1 R · Data integrity at scale [E] — provenance and licence records; deduplication *across* train/validation/test splits; benchmark leakage checks; pretraining packing with and without document-boundary masking; deterministic, resumable mixture sampling; token accounting per source.
- 10.2 R · Model-based quality filtering [E] — FineWeb-Edu, DCLM, Nemotron-CC as case studies; the classifier is evaluated by ablation, not by its accuracy.
- 10.3 R · Synthetic and rephrased data [P] — rephrasing (WRAP, Nemotron-CC, Kimi K2); evaluated on correctness, diversity and downstream utility, with generator compute counted.
- 10.4 R · Mixtures and micro-anneals [P] — data-mixing laws, RegMix; scoring datasets by short anneals (OLMo 2).
- 10.5 R · Continued training and forgetting [E] — a targeted mid-training run measuring the target gain and forgetting elsewhere.
- 10.6 X · Multilingual and code data — per-language pipelines (FineWeb2), repository data.
- Project: Data-v1 recipe, every choice backed by an ablation with intervals, with a provenance and leakage report.

#### Module 11 — What will the big run do?
- 11.1 R · Compute-optimal and over-trained regimes [E] — Kaplan vs Chinchilla reconciled (Porian et al.), data-constrained repetition, why released models train far beyond Chinchilla-optimal.
- 11.2 R · Predicting downstream capability [P] — observational scaling laws; emergence vs metric artefacts.
- 11.3 R · De-risking a run — the ladder: transfer, small-scale fits, held-out extrapolation, go/no-go rules.
- Project: **Recipe-R run** — a pre-registered prediction for the learner's final ~350M–1B pretraining run with Recipe-R and Data-v1, checked after the run.

### Stage D — Post-training questions

Stage D starts from a pinned open base model of 1–2B parameters (chosen at Phase 0) so results are
meaningful; the learner also applies the Module 13 pipeline to their own Recipe-R model to keep one
traceable chain (section 7).

#### Module 12 — Post-training foundations
- 12.1 R · Rewards — verifiable vs learned rewards, reward construction and calibration, reward-model failure modes and over-optimisation (a small reward model trained and over-optimised against a held-out gold signal).
- 12.2 R · Policy-gradient estimators for LLMs — advantage estimation and baselines, group normalisation, zero-variance groups, KL choices (in the reward vs in the loss; estimator variants) and their bias, entropy.
- 12.3 R · Details that change results — length bias, truncation, loss masking, token vs sequence vs prompt-level aggregation, importance ratios and policy staleness.
- 12.4 R · Eval Suite v2 — capability retention and instruction following, run after every post-training stage alongside task reward.
- Project: a correctness-checked RL loop (one unit test per 12.2–12.3 detail) plus a debugging exercise with planted bugs.

#### Module 13 — Which post-training pipeline for which target?
- 13.1 R · Open recipes as case studies — Tülu 3, OLMo 3, Llama 3, Qwen3: what each stage is for and what evidence each report gives.
- 13.2 R · Distillation [E] — SFT on teacher outputs vs on-policy distillation (per-token reverse KL on student samples), with teacher compute counted in the comparison.
- 13.3 R · Specification-driven alignment [E/M] — Constitutional AI / RLAIF, Claude's constitution, OpenAI's Model Spec and deliberative alignment; a mini-spec trained with an AI judge, judge compute counted, adherence and retention measured.
- 13.4 X · Thinking modes and budgets [M] — one model with thinking and non-thinking modes (Qwen3); router-based systems as disclosed (GPT-5).
- Project: SFT → preference → RLVR → distillation on the base model, with Eval v2 after each stage; repeated in short form on Recipe-R.

#### Module 14 — Which RL objective, at what scale?
- 14.1 R · Objectives derived, not switched — GRPO; DAPO and Dr. GRPO (token-level aggregation, clipping and normalisation changes); GSPO (sequence-level, length-normalised importance ratio and sequence-level clipping); CISPO (clipped, stop-gradient importance weights). Shared rollout and logging code, separate loss implementations, each with derivation and unit tests.
- 14.2 R · A controlled objective comparison — same prompts, rollouts per step, KL setting and compute; seeds; tuned per objective within a stated budget; report stability and Eval v2.
- 14.3 R · Rollout systems and staleness [E] — trainer/rollout separation, async RL and bounded staleness, train/inference log-prob mismatch, batch-invariant kernels.
- 14.4 R · RL scaling and the capability debate [P] — compute–performance curves (ScaleRL) and why one short run cannot identify an asymptote (ScaleRL itself fits only after ~1.5k GPU-hours); fitting is practised on published curves, and the learner's own short run is used to show how unidentifiable the asymptote is; pass@k vs pass@1 (Yue et al.).
- Project: reasoning-RL run on the base model with a stability report, Eval v2 retention and a pass@k analysis.

#### Module 15 — How should compute be spent at inference?
- 15.1 R · A test-time compute experiment — longer single reasoning vs independent sampling vs majority vote vs verifier selection vs one guided-search method, under matched total budgets (tokens and wall-clock), verifier cost and latency included; oracle pass@k vs the success of the actual selection procedure.
- 15.2 R · Speculative decoding [E] — rejection-sampling correctness, EAGLE-style drafts, MTP as a draft; measured acceptance and latency.
- 15.3 R · Serving cost of architecture choices — prefill vs decode, disaggregation, KV size for the Module 3–5 designs at 128K; why this matters for RL rollouts.
- 15.4 X · KV and weight quantisation for serving — link to the inference course.
- Project: budget-matched test-time-compute report with a recommendation for a stated latency target.

#### Module 16 — How do we train agents without fooling ourselves?
- 16.1 R · Environments and verifiers — task, tools, state, verifier, reset; verifier tests; reward that resists cheap exploits.
- 16.2 R · Software-engineering tasks [P] — task synthesis (SWE-smith, SWE-Gym), splitting by repository or task family (random splits of related generated tasks do not show generalisation).
- 16.3 R · Multi-turn agentic RL [P] — credit over turns, long trajectories, context management.
- 16.4 R · Reward hacking — catalogue, detection and mitigation; Anthropic's report that production reward hacking generalised to broader misalignment. Whether a student run finds a planted exploit is a hypothesis; validated traces of a run that did are provided for analysis.
- 16.5 X · Computer-use agents — OSWorld, UI-TARS.
- Project: environment pack (3 environments, repository-level split), a hack audit and an RL result.

### Stage E — Understanding, evaluating and communicating

#### Module 17 — What can we claim about a model's internals?
- 17.1 R · Features and sparse autoencoders [E] — superposition; ReLU/TopK/JumpReLU SAEs; reconstruction quality *and* downstream degradation when the SAE is spliced in.
- 17.2 R · Causal interventions [E] — activation patching, path patching, ablations; control interventions (random directions, unrelated features); held-out behaviours.
- 17.3 R · Transcoders and attribution graphs [P] — replacement models and attribution graphs (Circuit Tracing; Biology of an LLM); graph hypotheses validated by intervention.
- 17.4 R · Steering and persona vectors [P] — extraction, steering, side effects, held-out evaluation.
- 17.5 X · Interpretable by design and introspection [P] — weight-sparse transformers; introspection experiments.
- Project: one supported causal claim about an open model's behaviour, with controls, held-out tests and a limitations section.

#### Module 18 — Alignment science, frontier evaluation and governance
- 18.1 R · Model organisms of misalignment — sleeper agents, alignment faking, emergent misalignment and the persona-feature explanation; the lab analyses released organisms where available (analysis, not reproduction) and attempts a small reproduction stated as a hypothesis.
- 18.2 R · Chain-of-thought monitorability — monitors, obfuscation under optimisation pressure, unfaithful CoT; a monitor run on Module 16 traces.
- 18.3 R · Frontier evaluation — benchmark lifecycle and contamination (SWE-bench Verified retirement, SWE-bench Pro), hard evaluations (HLE, FrontierMath, ARC-AGI-2, GPQA), long-horizon time-horizon measurement (METR) and its uncertainty; Eval Suite v3.
- 18.4 R · Safety frameworks and system cards — RSP, Preparedness Framework, Frontier Safety Framework; reading a system card. The lab is a student audit exercise; it is explicitly not evidence that any model meets a lab's deployment thresholds.
- Project: an audit report on the learner's post-trained model with stated limits.

#### Module 19 — How is frontier research chosen, reproduced and written?
- 19.1 R · Research taste and problem choice — goal- vs idea-driven work, cheap proxies, three ranked proposals.
- 19.2 R · Reproducing a paper — scoping, what counts as "reproduced", contacting authors, recording deviations.
- 19.3 R · Writing and defending results — research notes, figures that answer the question, the written defence, reviewing someone else's result.
- Project: capstone proposal with an experiment contract, reviewed against the rubric.

#### Module 20 — Capstone: reproduce, extend, defend
- 20.1 R · Brief and rubric — choose one claim from a 2025–26 report (list provided, e.g. GSPO vs GRPO stability, QK-Clip vs QK-norm, DSA quality and cost vs dense, mHC vs HC stability, micro-anneal data scoring, on-policy vs off-policy distillation at equal compute), reproduce at small scale, extend with one new ablation.
- 20.2 R · Review and defence — reviewer questions, written defence, revision.
- Project: capstone report and code, graded on soundness (section 10).

### Elective

#### Module 21 — Elective: native multimodality
Optional, not a prerequisite for anything. A substantial elective rather than one lesson.
- 21.1 X · Representations — vision encoders, patches vs discrete tokens, audio.
- 21.2 X · Fusion — adapters vs early fusion (Chameleon, Llama 4 as documented), multimodal positions (Qwen3-VL Interleaved-MRoPE), multi-level feature injection.
- 21.3 X · Objectives — next-token on discrete tokens, mixed next-token + diffusion (Transfusion), modality-specific experts (ERNIE 4.5).
- 21.4 X · Data mixtures and stability — interleaved data, modality balance, stability fixes (QK-norm in Chameleon).
- 21.5 X · Multimodal evaluation — what current benchmarks do and do not measure.
- Project: a small early-fusion model with a controlled fusion comparison.

**Counts:** 20 core modules with 72 required lessons and 13 extensions, plus the 5-lesson elective
(85 core + 5 elective = 90 lessons). 20 module projects plus the capstone.

## 6. Dependencies and permitted orders

| Module | Needs | Produces (carried forward) |
|---|---|---|
| 1 | parent course | Baseline-0, Data-v0, Eval v0, experiment record template |
| 2 | 1 | performance methodology, Baseline-0 profile |
| 3 | 1, 2 | attention branches + memo |
| 4 | 1, 2 | Eval v1 (long context), 32K Baseline-0 |
| 5 | 3, 4 | sub-quadratic memo |
| 6 | 3–5 | Lineage-F integration result, chosen architecture |
| 7 | 1, 2 (runs on 6's choice or Baseline-0) | optimizer choice |
| 8 | 2, 7 | precision choice |
| 9 | 2 (8 for FP8 layouts) | measured layout + recovery evidence |
| 10 | 1 | Data-v1 |
| 11 | 7, 8, 10 | Recipe-R checkpoint |
| 12 | 1 | correctness-checked RL loop, Eval v2 |
| 13 | 12 | post-trained base model; short pipeline on Recipe-R (needs 11) |
| 14 | 12, 13 | RL results, reference curves analysis |
| 15 | 3–5 (cost part), 14 (reasoning models) | test-time compute report |
| 16 | 12, 14 | environments, agent traces |
| 17 | 1 | causal claim |
| 18 | 16 (traces), 17 (tools), 13 (model) | Eval v3, audit |
| 19 | 1 | capstone proposal |
| 20 | 19 + the stages its claim uses | capstone |

```text
A: M1 ─▶ M2
          │
B:        ├─▶ M3 ─┐
          ├─▶ M4 ─┴─▶ M5 ─▶ M6 ─┐
C:        ├─▶ M7 ─▶ M8 ─┐       │ (architecture choice)
          ├─▶ M9        ├─▶ M11 ◀┘  (Recipe-R)
          └─▶ M10 ──────┘
D:  M1 ─▶ M12 ─▶ M13 ─▶ M14 ─▶ M15, M16        (M13 short pipeline on Recipe-R needs M11)
E:  M1 ─▶ M17 ;  M13 + M16 + M17 ─▶ M18 ;  M1 ─▶ M19 ─▶ M20
```

Permitted orders:
- **Main path:** A → B → C → D → E.
- **Post-training first:** A → D → M17 → B → C → M18–M20. Valid because Stage D uses the pinned open
  base model and Eval v2, not the learner's pretrained model; the Recipe-R part of the Module 13
  project is done after M11.
- Not supported: Stage C before Stage B (M11 needs the architecture choice), M18 before M16 and M17,
  M5 before M4 (needs Eval v1).

## 7. Artefact chain and lab package

The learner finishes with one traceable chain, recorded in run cards that reference each other:

```text
Data-v0 + Baseline-0 + Eval v0 (M1)
  └─ isolated architecture branches (M3–M6), each vs Baseline-0, same data and budget
       └─ Lineage-F integration experiment → chosen architecture (M6)
            └─ Recipe-R: optimizer (M7) + precision (M8) + Data-v1 (M10) → pre-registered run (M11)
                 └─ post-training: base model pipeline (M13–M14) + short pipeline on Recipe-R (M13)
                      └─ evaluation (Eval v1–v3), interpretability claim, audit (M17–M18)
                           └─ capstone report (M20)
```

- Lab package `frontierlab` under `labs/common/`, one coherent codebase extended module by module
  (`attention/mla.py`, `attention/sliding.py`, `attention/dsa.py`, `attention/hybrid.py`,
  `blocks/mtp.py`, `blocks/mhc.py`, `optim/muon.py`, `precision/`, `data/`, `rl/losses/` with one file
  per objective, `rl/rollout/`, `envs/`, `ttc/`, `interp/`, `evals/`). It copies what it needs from
  the parent's `llmre` and from `moelab` (MoE layer only) and imports neither.
- Shared correctness suite `frontierlab.testing` (gradient, causal, cache, recurrence, reference).
- Experiment record: `run_card.yaml` per run (git hash, config, seeds, data hashes, hardware, budget,
  parent run); `experiments/` directory structure fixed in Module 1.
- Carried forward and pinned: Data-v0/v1 manifests with hashes, Baseline-0 and Recipe-R checkpoints,
  the Stage D base model, Eval v0–v3 with versions. Datasets and checkpoints are downloaded by
  scripts, never committed.

## 8. Workload and budget (estimates, to be re-measured in Phase 0 pilots)

Hours are per learner; required path only. Attended work and unattended GPU runtime are separate.

| Item | Count | Attended hours |
|---|---|---|
| Reading (25–40 min per lesson) | 72 | 30–48 |
| Lesson labs, attended (1–2.5 h each) | 72 | 72–180 |
| Module projects (reuse lab outputs; analysis, extra runs, write-up: 4–8 h) | 20 | 80–160 |
| Capstone | 1 | 30–50 |
| **Total attended** | | **~210–440 h** |

Unattended GPU runtime (main path, estimate): about 650–1,150 H100-equivalent GPU-hours for the
required path, dominated by Stage B ablations with seeds (~200–300), Recipe-R (~50–100), reasoning
and agentic RL (~150–250) and the capstone (~50–150). At USD 2–3 per H100-hour that is roughly
USD 1,300–3,500. Starting from the provided Baseline-0 checkpoint and pilot traces cuts this
somewhat. The per-lab figures in the published cost table are **projections from the scaled pilots**
(section 12.1), labelled as such and with their extrapolation method, not measurements on the main
path; learners record their measured cost in each run card so the projections can be corrected.

Extensions add about 15–30 attended hours; the elective about 15–25 hours plus GPU time.

## 9. Experiment contract (every substantial lab)

Every lab that compares anything states, before the steps:

- **Question** and the decision it informs.
- **Hypothesis** and its status: established effect / reported effect / may not appear at this scale.
- **Baseline** (which run card) and how it was tuned (budget).
- **Changed variable** and **controlled variables** (data, tokens, seed set, eval version).
- **Comparison axis:** equal tokens, equal parameters, equal training FLOPs or equal wall-clock —
  and what the chosen axis does and does not answer.
- **Budget:** GPU type, GPU-hours, wall-clock, cost; teacher, judge, verifier or generator compute
  counted where used.
- **Metrics** with uncertainty (seeds, intervals) and the pre-stated decision rule.
- **Correctness checks** that must pass before results count.
- **Fallback evidence:** the provided checkpoint or trace if the effect does not appear, labelled as
  analysis.
- **Limits of the conclusion:** scale, data, architecture, what would change the answer.

## 10. Assessment

- **Multiple-choice quizzes** (3–5 per lesson, 8–10 scenario questions per module, `.quiz.yaml`,
  70% pass) are checks of understanding, as the binding guide requires.
- **Professional competence is assessed by artefacts**, in each module project and the capstone:
  1. the experiment record (contract, run cards, raw results);
  2. a debugging task (a planted bug or a failed run to diagnose);
  3. a written defence (1–2 pages answering reviewer-style questions about controls, uncertainty and
     limits).
- **Rubric** (in `templates/experiment-rubric.md`): sound question and controls; budget parity and
  tuned baseline; correctness checks passed; uncertainty reported; conclusion matches the evidence;
  limits stated. Negative and null results score the same as positive ones when the experiment is
  sound. Whether the new method wins is never a criterion.
- The Academy grades only the quizzes; artefacts are self-assessed against the rubric with reference
  solutions in `<details>`, and optionally peer-reviewed using the review template.

## 11. Lesson template

Front-matter per the binding guide. Fixed `##` sections in every lesson:

1. `## Why this matters at a frontier lab`
2. `## The idea` (intuition, then the mathematics with every symbol named)
3. `## Worked example` (tiny numbers, by hand)
4. `## Shapes and cost` (shapes/dtype/device, FLOPs, memory, bytes moved)
5. `## Build it` (implementation and its correctness checks)
6. `## What the evidence says` (maturity tag, evidence labels, report sections, open questions)
7. `## Lab` (experiment contract, main path, free GPU, free CPU, pass checks; hints in `<details>`)
8. `## Common mistakes`
9. `## References`
10. `## Next` (link to the following lesson)

## 12. Build phases

- **Phase 0 — decisions, pins and scaled pilots.** Tal's open decisions (section 15); pin versions
  (PyTorch, torchtitan, torchao, flash-linear-attention, vLLM, SGLang, verl, Transformers, TRL, SAELens,
  circuit-tracing tools); choose the Stage D base model; run the scaled pilots in section 12.1 on the
  course's Colab account, in priority order, within each month's compute allowance, and record
  go / redesign / downgrade-to-extension for each; publish the pilot traces and projected costs;
  re-verify every source and claim (section 14); write `BUILD_PROGRESS.md`,
  `curriculum/course-details.md`, `references/frontier-models-2026-10.md`,
  `templates/experiment-contract.md`, `templates/run-card.yaml`, `templates/experiment-rubric.md`.
- **Phase 1:** Stage A (Modules 1–2). **Phase 2:** Stage B (3–6). **Phase 3:** Stage C (7–11).
  **Phase 4:** Stage D (12–16). **Phase 5:** Stage E (17–20), glossary, final QA, `0 problems` from the
  Academy check. **Phase 6 (after the first release, decided 2026-10-03):** Module 21.
- Free variants are written after each lab is piloted, never before.
- At most two writing agents at once, one module each, per the binding guide.

### 12.1 Scaled pilots and feasibility matrix (highest-risk labs)

Course-side compute is one paid Colab account (single GPU, limited compute units, sessions that can
disconnect). Every pilot follows the same method:

1. **Same mechanism, smaller sample.** Same code, data pipeline and experiment contract as the main
   path; only scale changes (width/depth, tokens, context length, rollouts, seeds).
2. **A scale ladder, not a single point.** Each pilot runs at 2–3 sizes (typically ~10M, ~30M, ~70M
   parameters with tokens scaled proportionally, or 1/8 → 1/4 → 1/2 of the main-path context or
   rollout count), so the *trend* of the effect and of the cost is observed, not one number.
3. **What a pilot answers:** (a) the code runs and passes its correctness checks; (b) whether the
   hypothesised effect appears, and whether it grows, shrinks or stays flat along the ladder relative
   to the seed-noise floor; (c) measured throughput and memory at each size.
4. **Projection to the main path.** Main-path GPU-hours = training FLOPs of the main-path run ÷
   (H100 peak × MFU measured in the pilot, adjusted by a stated, sourced A100/L4-to-H100 ratio).
   Memory is extrapolated from the per-size measurements. Every projected figure is labelled
   PROJECTED, with its formula and the ladder it came from.
5. **A projected effect is a hypothesis, not a promise.** If the effect is absent or shrinking along
   the ladder, the lab is redesigned (larger effect, different axis) or becomes an analysis lab on
   published results, and says so.
6. **Checkpoint every session**, so a disconnect costs minutes, not the run. Pilot traces become the
   course-provided traces.

Priority when the month's compute runs out: the Baseline-0 noise floor first (everything depends on
it), then the hypothesis labs, then cost-only pilots. Pilots that need hardware Colab does not offer
are marked **not piloted**; their main-path figures come from published numbers, labelled.

| Lab | Risk | Main-path setup (projected) | Colab scaled pilot | Pilot question | If the pilot fails |
|---|---|---|---|---|---|
| M1 Baseline-0 noise floor | seed variance at ~125M swamps Stage B effects | 1× H100, ~3–4 h per run, ~10 runs | 10M/30M/70M × 3 seeds on A100; the 125M run itself if the allowance permits | how does seed std scale relative to typical architecture effects? | raise main-path tokens or size for Stage B |
| 03 MLA / local-global comparison | effects within noise; decode gains only at long context | 1× H100, 4 variants × 3 seeds | 30M/70M × 4 variants × 2 seeds; decode memory at 8K–32K | quality gap vs noise; memory/latency trend with context | memory/latency primary, quality secondary |
| 05.1 Linear hybrid | kernel/version compatibility | 1× H100 | equivalence tests + 30M training on A100 | do pinned kernels pass equivalence and train? | own chunked reference kernel |
| 05.2 DSA-style indexer | no fused kernel for this setting | 1× H100 | component profiling (indexer, top-k, attention) at 4K–32K on A100 | does measured component cost follow the O(L²) indexer / O(Lk) attention model, and where is the crossover heading? | report component costs; end-to-end speed labelled not measured |
| 06.2 mHC stability | difference may not appear at small scale | 1× H100, ~20–40 h | HC vs mHC at 10M/30M/70M, standard and raised LR | does HC instability appear or grow along the ladder? | analysis of the mHC paper's results + pilot traces |
| 06 Lineage-F integration | 350M runs expensive; interactions noisy | 1× H100, ~25 h per run | integration at 30M/70M, 2 arms × 2 seeds | do interaction effects exceed noise at pilot scale? | integrate fewer components; state limits |
| 07.2 Logit growth / QK-Clip | logit explosion may not reproduce | 1× H100 | Muon at raised LR, 30M/70M, max logit logged | does max logit grow with scale and LR? | induced-failure variant + traces |
| 08.2 FP8 training | fine-grained scaling tooling may be prototype-only | 1× H100 | torchao Float8 on L4 (sm89 has FP8) at 30M/70M | do pinned FP8 paths run; loss gap vs BF16? | tensorwise/rowwise Float8 as main path; fine-grained recipe as kernel benchmark |
| 08.3 NVFP4 | needs Blackwell | 1× B200 | **not piloted** (no Blackwell); emulation only | — | labelled emulation; main-path figures from the NVFP4 report |
| 09.3–09.4 node investigation | multi-GPU; features may not compose | 8× H100 node, ~4–6 h | **multi-GPU not piloted**; full-state recovery (model, optimizer, scheduler, RNG, data stream) piloted on one GPU; layout correctness on CPU processes | does full-state recovery work for the chosen architecture? | multi-GPU figures from published torchtitan / Llama 3 numbers, labelled |
| 10.3 Synthetic data utility | effect too small to detect | 1× H100 | rephrasing with a small open model + 30M/70M ablations | is the effect above noise and trending up? | report as null with a power analysis |
| 12–14 RL loop and objective comparison | differences within noise; long rollouts | 1–8× H100, ~60–100 GPU-h | ~0.5B base model, short responses, 2 objectives × 2 seeds on A100 | do objectives separate on stability metrics at pilot scale? | fewer objectives, more seeds; published comparisons |
| 14.4 RL scaling fit | needs very long runs | — | **not produced** course-side | — | taught on published ScaleRL curves |
| 15.1 Test-time compute | verifier availability; cost | 1× H100 + vLLM | ~0.5–1.5B model on A100 with vLLM, small problem set | which open verifiers work; does the budget-matched ranking hold at both sizes? | outcome verifiers only (exact match / unit tests) |
| 16.3–16.4 Agentic RL and reward hacking | framework maturity; exploit may not be found | 8× H100, ~40–80 GPU-h | single-GPU multi-turn RL on ~0.5B with a planted exploit | does the framework run; is the exploit found within budget? | single-turn tasks; detection analysis on pilot traces |
| 17.3 Attribution graphs | tooling exists only for some models | 1× 48–80 GB GPU | circuit tracing on the smallest supported model on A100 | which pinned model with released transcoders works? | patching-only causal lab |
| 18.1 Emergent misalignment | effect weak in small models | 1× H100 | narrow fine-tune of ~0.5B and ~1.5B models on A100 | any effect, and does it grow from 0.5B to 1.5B? | analysis of released organisms only |

## 13. Change log (revision 1 → 2, 2026-10-03)

Based on an external review; all suggestions were adopted, some in adapted form.

1. **Question-led modules.** Every module is now a research question with models as case studies;
   maturity tags (ESTABLISHED / PROMISING / MODEL-SPECIFIC) added. Removed the "2026 default block"
   framing and the unscoped claim that post-training accounts for most compute or capability gains
   (the available evidence is scoped and partly estimated; see 14.1 item 32). Engram moved out of
   the residual-stream lesson into its own extension on conditional memory.
2. **Discipline first.** Experiment design, uncertainty and reproducibility (old 18.2, 16.4) are now
   Module 1, with Baseline-0 and a measured noise floor before any comparison. Advanced evaluation
   (Module 18) and research writing (Module 19) stay late. The "Stages B–D in any order" claim is
   replaced by an explicit dependency table and two supported orders (section 6).
3. **Experiment contract** for every substantial lab, including the four comparison axes, tuning
   budgets and teacher/judge/verifier/generator compute (section 9). Grading is on soundness; sound
   negative results pass (section 10).
4. **No guaranteed outcomes.** "Show the stability difference", "watch RL find a hack" and "detect
   emergent misalignment" are now hypotheses, piloted before publishing, with pilot traces and
   released checkpoints as fallback and analysis labelled as analysis. RL scaling fits are practised
   on published curves (ScaleRL itself starts fits after ~1.5k GPU-hours).
5. **Architecture correctness suite** (gradient, causal, cache, recurrence, reference) required before
   any comparison. DSA rewritten: indexer training is taught (dense warm-up, then a separate KL
   objective with detached indexer input), and the cost model now states that the indexer is still
   O(L²) while selected attention is O(L·k), with indexing, selection, memory traffic and kernel
   overhead measured.
6. **Performance engineering** is a required Module 2 and a prerequisite for all systems comparisons.
   Long-context evaluation is required (04.1: multi-hop, distractors, position sensitivity,
   short-context regression) and precedes the sub-quadratic module.
7. **Post-training foundations** (Module 12) before algorithm comparisons: rewards and calibration,
   reward-model over-optimisation (required), advantages, KL choices, entropy, length bias,
   truncation, masking, zero-variance groups, staleness. Objectives are derived and implemented
   separately (GSPO's sequence-level ratio and clipping are treated as substantive). Capability
   retention and instruction following evaluated after every stage (Eval v2).
8. **Test-time compute experiment** added (15.1) with matched budgets, verifier cost and latency, and
   oracle pass@k vs actual selection success.
9. **Data module completed** against what the parent already teaches (04.3, 07.3, 11.x, 14.2):
   provenance, cross-split dedup, leakage, document-boundary packing, resumable sampling, token
   accounting, synthetic data judged on correctness/diversity/utility, a continued-training
   improvement-vs-forgetting experiment, repository/task-family splits for agent tasks (16.2).
10. **Measured distributed investigation** (09.3) and a **failure/recovery exercise** (09.4) checking
    model, optimizer, scheduler, RNG and data-stream state, plus a tested-capability matrix.
11. **Causal standards for interpretability**: patching and controls, held-out behaviours,
    reconstruction-induced degradation; project narrowed to one supported causal claim; the audit is
    labelled a student exercise, not deployment-threshold evidence.
12. **Multimodal scope**: the required course is text/reasoning/agents; multimodality is a 5-lesson
    elective (Module 21) covering representation, fusion, objectives, data and evaluation.
13. **Artefact chain**: Baseline-0 → isolated branches → Lineage-F *integration experiment* → Recipe-R
    → post-training → evaluation → capstone, with what carries forward (section 7).
14. **Workload corrected.** Revision 1 actually had 88 lessons (74 required, 14 extensions), not
    ~78/60. Revision 2 consolidates surveys (old 01.2–01.4 into one lesson; post-training recipes
    into one case-study lesson; old Modules 16 and 17 folded into 15 and 18) to 72 required + 13
    extensions + 5 elective. Attended hours, unattended GPU time and project reuse are estimated
    separately (section 8). MCQs remain checks; competence is assessed by artefacts, debugging tasks
    and written defences.

Revision 2.1 (same day, Tal's decision): no course funding for rented GPUs. Pilots become **scaled
pilots** on one paid Colab account (scale ladder, projection to the main path labelled PROJECTED,
section 12.1); multi-GPU and Blackwell pilots are marked not piloted; course-provided artefacts are
limited to what that budget produces, hosted on a free Hugging Face account.

Also corrected from the claim-level check (14.1): Llama 3's 78% hardware share is of *unexpected*
interruptions; Qwen3-Next is reported at 10% (not <10%) of Qwen3-32B's training cost; DeepSeek-V4's
27%/10% figures are single-token inference FLOPs and KV cache at 1M context; on-policy distillation's
9–30× saving is versus off-policy SFT, not RL; NVFP4 keeps the first two and last eight blocks in
BF16; DeepSeek-V4 explicitly does not use QK-Clip.

## 14. Sources

### 14.1 Claim-level checks (2026-10-03)

Each item is checked against the cited section, equation or table, not only title and date.
V = verified at that location; C = corrected (the plan uses the corrected form); P = partially
verified; open gaps are listed and must be closed before the lesson is written.

| # | Claim used in the plan | Status | Where |
|---|---|---|---|
| 1 | DSA: ReLU FP8 lightning indexer; top-2048 keys per query; indexer O(L²) but cheap vs MLA; main attention O(Lk). Dense warm-up: all params frozen except indexer, KL to L1-normalised main attention, 1,000 steps / 2.1B tokens. Sparse stage: indexer input detached, indexer trained only by its KL loss, main model only by LM loss, 15,000 steps / 943.7B tokens | V (gaps: indexer head count; whether the sparse-stage KL is restricted to the selected set) | V3.2 report §2.1, §2.1.1 |
| 2 | NSA: compression, selection, sliding-window branches with MLP+sigmoid gates | V | NSA §3.2 Eq. 5 |
| 3 | DeepSeek-V4: 1.6T/49B and 284B/13B; 1M context; CSA (compressed KV + top-k + uncompressed recent window), HCA (m′ ≫ m, dense); FP8 KV with BF16 RoPE dims (last 64); mHC Sinkhorn t_max = 20; hybrid Newton–Schulz 8 + 2 iterations; no QK-Clip (RMSNorm on Q and KV); hash routing in the first blocks; FP4 routed experts; 27% single-token FLOPs and 10% KV vs V3.2 at 1M | V (gaps: numeric m, m′, k, window; hash-routed layer count) | V4 §2.1–2.4, abstract; HF card |
| 4 | MLA: KV cache −93.3%, 5.76× max generation throughput vs DeepSeek 67B; decoupled RoPE | V (decoupled-RoPE wording to quote from §2.1 when writing) | V2 abstract, §2.1 |
| 5 | Muon: quintic NS (3.4445, −4.7750, 2.0315), 5 steps; embeddings, head and scalar/vector params on AdamW | V | Muon blog |
| 6 | Moonlight: update 0.2·O·√max(A,B) + weight decay; ~2× compute efficiency (~52% FLOPs) | V | §2.2 Eq. 4; abstract; §3.2 |
| 7 | Kimi K2: QK-Clip per-head γ = min(1, τ/S_max); shared rotary key untouched; logits > 1000 without it; 15.5T tokens, zero loss spikes; 64 vs 128 heads (0.5–1.2% loss) | V (gap: numeric τ) | K2 §2.1, §2.3, Fig. 2 |
| 8 | GSPO: length-normalised sequence ratio (geometric mean), sequence-level clipping, ranges 3e-4 / 4e-4; removes need for routing replay in MoE | V | GSPO Eq. 5, Eq. 7 |
| 9 | CISPO: clipped, stop-gradient IS weight; all tokens keep gradients; in practice only ε_high tuned | V | MiniMax-M1 §3.1 Eqs. 4–5 |
| 10 | ScaleRL: R_C − R_0 = (A − R_0)/(1 + (C_mid/C)^B); recipe (PipelineRL 8-step off-policy, interruption length control, FP32 logits, prompt-level aggregation, batch-level advantage norm, CISPO, zero-variance filtering, no-positive-resampling); fits start after ~1.5k GPU-hours | V | ScaleRL |
| 11 | RLVR beats base at small k; base higher pass@k at large k | V | Yue et al. abstract |
| 12 | DAPO: ε_low 0.2 / ε_high 0.28; dynamic sampling 0 < correct < G; token-level loss; soft overlong punishment (L_cache 4,096) and overlong filtering | V (gap: confirm L_max 20,480 = 16,384 + 4,096) | DAPO §3.1–3.4 |
| 13 | Dr. GRPO removes 1/\|o\| and std normalisation | V | §3.2 |
| 14 | gpt-oss: alternating 128-token banded and dense layers; learned per-head bias in softmax denominator; MXFP4 4.25 bits on MoE weights; 116.83B/5.13B and 20.91B/3.61B | V | model card §2.1–2.2, Table 1 |
| 15 | Gemma 3: 5 local per global, 1024 window; QK-norm replaces soft-capping; global RoPE base 1M, local 10k | V | Gemma 3 report |
| 16 | Llama 3: 466 interruptions (47 planned, 419 unexpected), ~78% *of unexpected* hardware; >90% effective training time; [TP, CP, PP, DP] | C (78% scope) | §3.3.2, §3.3.4 |
| 17 | Qwen3-Next: 12 × (3 GDN + 1 gated attention); 512 experts, 10 + 1 shared; partial RoPE 0.25 of head dim 256; zero-centred weight-decayed norm; MTP; 10% of Qwen3-32B training cost | C ("10%", not "<10%"); source is HF card/config, blog did not render | HF model card, config.json |
| 18 | Kimi Linear: channel-wise forgetting in KDA; uniform 3:1 KDA:MLA; up to 75% less KV; up to 6× decode at 1M | V | Kimi Linear |
| 19 | MiniMax-M2: hybrid deficits on multi-hop reasoning at scale, eval difficulty, immature linear-attention infra (precision, prefix caching, speculative decoding) | V | HF blog |
| 20 | Up to 4 epochs of repetition ≈ negligible loss change | V | Muennighoff abstract |
| 21 | Precision: P* ≈ 7–8 bits when jointly optimised (integer-type fits); PTQ degradation grows with data | V | §4.3.2; abstract |
| 22 | Test-time compute can beat a 14× larger model, FLOPs-matched, on problems where the small model has non-trivial success | V (scope stated) | Snell abstract |
| 23 | mHC: doubly-stochastic residual mixing via Sinkhorn-Knopp, t_max 20, n = 4; 3B/9B/27B; 6.7% time overhead | V | mHC |
| 24 | Engram: O(1) N-gram lookup; U-shaped allocation law vs MoE; reported gains incl. BBH +5.0 | V | Engram abstract |
| 25 | Reward hacking in production RL generalised to misalignment; mitigations: prevent hacks, diverse RLHF safety training, inoculation prompting | V | 2511.18397 abstract |
| 26 | OpenAI stopped reporting SWE-bench Verified: flawed tests in an audited subset; contamination; recommends SWE-bench Pro | P (page returned 403; from search/secondary coverage — check the page manually) | OpenAI 2026-02-23 |
| 27 | METR: 50% time horizon doubling ~7 months since 2019, possibly faster in 2024 | V | METR abstract |
| 28 | On-policy distillation: per-token reverse KL on student samples; 9–30× saving is vs off-policy SFT (Qwen3-8B student, 32B teacher, AIME'24); RL comparisons are separate figures | C (framing) | Thinking Machines blog |
| 29 | Inference nondeterminism caused by lack of batch invariance in kernels | V | Thinking Machines blog |
| 30 | NVFP4: 12B hybrid Mamba-Transformer, 10T tokens; 16×16 weight / 1×16 activation scaling; RHT on Wgrad inputs; stochastic rounding on gradients; first 2 + last 8 blocks in BF16 | C (BF16 layer placement) | 2509.25149 |
| 31 | DeepSeek-V3 FP8: 1×128 activation tiles, 128×128 weight blocks, FP32 promotion every 128 elements | V | V3 §3.3.2 |
| 32 | Share of compute in post-training/RL | No official percentage exists. Scoped statements only: xAI (Grok 4 RL "at pretraining scale", relative to prior RL); OpenAI o3 ("an additional order of magnitude" vs o1); Epoch AI estimates (labelled estimates). Taught as speculation. | various |

### 14.2 Source register

Status: **V** = opened and title/date checked on 2026-10-03; **S** = found in search, direct fetch
blocked (re-verify before embedding); **U** = unverified. Claim-level status is in 14.1. All must be
re-verified at Phase 0.

### Architecture — DeepSeek
- V · DeepSeek-V2 (MLA, DeepSeekMoE), 2024-05-07 — https://arxiv.org/abs/2405.04434
- V · Auxiliary-Loss-Free Load Balancing, 2024-08-28 — https://arxiv.org/abs/2408.15664
- V · DeepSeek-V3 Technical Report, 2024-12-27 — https://arxiv.org/abs/2412.19437
- V · Native Sparse Attention, 2025-02-16 — https://arxiv.org/abs/2502.11089
- V · DeepSeek-V3.1 model card, 2025-08 — https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- V · DeepSeek-V3.2, 2025-12-02 — https://arxiv.org/abs/2512.02556
- V · mHC: Manifold-Constrained Hyper-Connections, 2025-12-31 — https://arxiv.org/abs/2512.24880
- V · Conditional Memory via Scalable Lookup (Engram), 2026-01-12 — https://arxiv.org/abs/2601.07372
- V · DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, 2026-04-26 — https://arxiv.org/abs/2606.19348
- V · DeepSeek-V4-Pro model card — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- V · Better & Faster LLMs via Multi-token Prediction (Meta), 2024-04-30 — https://arxiv.org/abs/2404.19737

### Architecture — Moonshot, Qwen, OpenAI, Google, Meta, others
- V · Muon (Keller Jordan blog), 2024-12-08 — https://kellerjordan.github.io/posts/muon/
- V · Muon is Scalable for LLM Training (Moonshot), 2025-02-24 — https://arxiv.org/abs/2502.16982
- V · Practical Efficiency of Muon for Pretraining (Essential AI), 2025-05-04 — https://arxiv.org/abs/2505.02222
- V · Kimi K2: Open Agentic Intelligence, 2025-07-28 — https://arxiv.org/abs/2507.20534
- V · Kimi Linear, 2025-10-30 — https://arxiv.org/abs/2510.26692
- V · Kimi K2.5, 2026-02 — https://arxiv.org/abs/2602.02276
- S · Kimi-K2-Thinking model card — https://huggingface.co/moonshotai/Kimi-K2-Thinking
- V · Qwen3 Technical Report, 2025-05-14 — https://arxiv.org/abs/2505.09388
- V · Gated Attention for LLMs (Qwen), 2025-05-10 — https://arxiv.org/abs/2505.06708
- V · Gated Delta Networks, 2024-12-09 — https://arxiv.org/abs/2412.06464
- V · Qwen3-Next blog (mirror), 2025 — https://www.alibabacloud.com/blog/qwen3-next-towards-ultimate-training-%26-inference-efficiency_602580
- V · Qwen3.5-397B-A17B model card, 2026-02 — https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- S · Qwen3-VL Technical Report, 2025-11 — https://arxiv.org/abs/2511.21631
- V · gpt-oss-120b & gpt-oss-20b Model Card, 2025-08-05 — https://arxiv.org/abs/2508.10925
- V · GPT-5 System Card, 2025-08-13 — https://cdn.openai.com/gpt-5-system-card.pdf
- V · Epoch AI, Why GPT-5 used less training compute than GPT-4.5, 2025-09-26 — https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont
- V · Efficient Streaming LMs with Attention Sinks, 2023-09-29 — https://arxiv.org/abs/2309.17453
- V · Gemma 2, 2024-07-31 — https://arxiv.org/abs/2408.00118
- V · Gemma 3 Technical Report, 2025-03-25 — https://arxiv.org/abs/2503.19786
- S · Introducing Gemma 3n: developer guide, 2025-06 — https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
- U · MatFormer, 2023-10 — https://arxiv.org/abs/2310.07707
- V · Gemma 4 Technical Report, 2026-07-02 — https://arxiv.org/abs/2607.02770
- V · Gemini 1.5, 2024-03-08 — https://arxiv.org/abs/2403.05530
- V · Gemini 2.5 report, 2025-07-07 — https://arxiv.org/abs/2507.06261
- V · Gemini 3 Pro Model Card, 2025-11 — https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf
- V · Gemini Diffusion — https://deepmind.google/models/gemini-diffusion/
- V · The Llama 3 Herd of Models, 2024-07-31 — https://arxiv.org/abs/2407.21783
- V · The Llama 4 herd (Meta blog), 2025-04-05 — https://ai.meta.com/blog/llama-4-multimodal-intelligence/ (do not cite arXiv 2601.11659; withdrawn, not Meta)
- V · Byte Latent Transformer, 2024-12-13 — https://arxiv.org/abs/2412.09871
- V · Chameleon, 2024-05-16 — https://arxiv.org/abs/2405.09818
- V · Transfusion, 2024-08-20 — https://arxiv.org/abs/2408.11039
- V · Mixtral of Experts, 2024-01-08 — https://arxiv.org/abs/2401.04088
- V · Introducing Mistral 3, 2025-12-02 — https://mistral.ai/news/mistral-3
- V · MiniMax-01 (lightning attention), 2025-01-14 — https://arxiv.org/abs/2501.08313
- V · MiniMax-M1 (CISPO), 2025-06-16 — https://arxiv.org/abs/2506.13585
- V · Why Did MiniMax M2 End Up as a Full Attention Model?, 2025-10-30 — https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
- V · GLM-4.5, 2025-08-08 — https://arxiv.org/abs/2508.06471
- S · GLM-5, 2026-02 — https://arxiv.org/abs/2602.15763
- V · ERNIE 4.5 (model card) — https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Base-PT
- V · YaRN, 2023-08-31 — https://arxiv.org/abs/2309.00071
- V · Ring Attention, 2023-10-03 — https://arxiv.org/abs/2310.01889
- V · LLaDA: Large Language Diffusion Models, 2025-02-14 — https://arxiv.org/abs/2502.09992
- V · Coconut: reasoning in continuous latent space, 2024-12-09 — https://arxiv.org/abs/2412.06769
- V · Recurrent-depth latent reasoning (Huginn), 2025-02-07 — https://arxiv.org/abs/2502.05171
- V · The Big LLM Architecture Comparison (Raschka), updated 2026-04-02 — https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

### Optimization, stability, precision, scaling
- V · Tensor Programs V (µTransfer), 2022-03-07 — https://arxiv.org/abs/2203.03466
- V · MiniCPM (WSD), 2024-04-09 — https://arxiv.org/abs/2404.06395
- V · Scaling ViT to 22B (QK-norm), 2023-02-10 — https://arxiv.org/abs/2302.05442
- V · 2 OLMo 2 Furious, 2024-12-31 — https://arxiv.org/abs/2501.00656
- V · Scaling FP8 Training to Trillion-Token LLMs, 2024-09-19 — https://arxiv.org/abs/2409.12517
- V · Pretraining LLMs with NVFP4, 2025-09-29 — https://arxiv.org/abs/2509.25149
- V · FP8 Formats for Deep Learning, 2022-09-12 — https://arxiv.org/abs/2209.05433
- V · Scaling Laws for Precision, 2024-11-07 — https://arxiv.org/abs/2411.04330
- V · Scaling Data-Constrained Language Models, 2023-05-25 — https://arxiv.org/abs/2305.16264
- V · Scaling Laws for Fine-Grained MoE, 2024-02-12 — https://arxiv.org/abs/2402.07871
- V · Joint MoE Scaling Laws, 2025-02-07 — https://arxiv.org/abs/2502.05172
- V · Resolving Discrepancies in Compute-Optimal Scaling, 2024-06-27 — https://arxiv.org/abs/2406.19146
- V · Observational Scaling Laws, 2024-05-17 — https://arxiv.org/abs/2405.10938
- V · Scaling LLM Test-Time Compute Optimally, 2024-08-06 — https://arxiv.org/abs/2408.03314
- V · The Art of Scaling RL Compute for LLMs, 2025-10-15 — https://arxiv.org/abs/2510.13786
- V · Emergent Abilities of LLMs, 2022-06-15 — https://arxiv.org/abs/2206.07682
- V · Are Emergent Abilities a Mirage?, 2023-04-28 — https://arxiv.org/abs/2304.15004

### Infrastructure
- V · Fire-Flyer AI-HPC, 2024-08-26 — https://arxiv.org/abs/2408.14158
- V · Insights into DeepSeek-V3 (hardware), 2025-05-14 — https://arxiv.org/abs/2505.09343
- V · DualPipe — https://github.com/deepseek-ai/DualPipe ; DeepEP — https://github.com/deepseek-ai/DeepEP ; 3FS — https://github.com/deepseek-ai/3FS ; FlashMLA — https://github.com/deepseek-ai/FlashMLA
- V · MegaScale, 2024-02-23 — https://arxiv.org/abs/2402.15627
- V · ByteCheckpoint, 2024-07-29 — https://arxiv.org/abs/2407.20143
- V · Context Parallelism for Million-Token Inference, 2024-11-04 — https://arxiv.org/abs/2411.01783
- V · TorchTitan, 2024-10-09 — https://arxiv.org/abs/2410.06511
- V · Megatron Core User Guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html
- V · TPU v4, 2023-04-04 — https://arxiv.org/abs/2304.01433
- V · Pathways, 2022-03-23 — https://arxiv.org/abs/2203.12533
- V · The Ultra-Scale Playbook (Hugging Face) — https://huggingface.co/spaces/nanotron/ultrascale-playbook
- V · How to Scale Your Model (JAX Scaling Book), 2025-02-04 — https://jax-ml.github.io/scaling-book/

### Data
- V · The FineWeb Datasets, 2024-06-25 — https://arxiv.org/abs/2406.17557
- V · DataComp-LM, 2024-06-17 — https://arxiv.org/abs/2406.11794
- V · Nemotron-CC, 2024-12-03 — https://arxiv.org/abs/2412.02595
- V · FineWeb2, 2025-06-26 — https://arxiv.org/abs/2506.20920
- V · Rephrasing the Web (WRAP), 2024-01-29 — https://arxiv.org/abs/2401.16380
- V · Phi-4 Technical Report, 2024-12-12 — https://arxiv.org/abs/2412.08905
- V · Data Mixing Laws, 2024-03-25 — https://arxiv.org/abs/2403.16952
- V · RegMix, 2024-07-01 — https://arxiv.org/abs/2407.01492
- V · Olmo 3, 2025-12-15 — https://arxiv.org/abs/2512.13961
- V · SWE-smith, 2025-04-30 — https://arxiv.org/abs/2504.21798

### Post-training and RL
- V · Tülu 3, 2024-11-22 — https://arxiv.org/abs/2411.15124
- V · DAPO, 2025-03-18 — https://arxiv.org/abs/2503.14476
- V · Understanding R1-Zero-Like Training (Dr. GRPO), 2025-03-26 — https://arxiv.org/abs/2503.20783
- V · VAPO, 2025-04-07 — https://arxiv.org/abs/2504.05118
- V · Group Sequence Policy Optimization, 2025-07-24 — https://arxiv.org/abs/2507.18071
- V · Kimi k1.5, 2025-01-22 — https://arxiv.org/abs/2501.12599
- V · Does RL Really Incentivize Reasoning Beyond the Base Model?, 2025-04-18 — https://arxiv.org/abs/2504.13837
- V · HybridFlow (verl), 2024-09-28 — https://arxiv.org/abs/2409.19256 ; https://github.com/volcengine/verl
- V · OpenRLHF, 2024-05-20 — https://arxiv.org/abs/2405.11143
- V · slime — https://github.com/THUDM/slime
- V · AReaL, 2025-05-30 — https://arxiv.org/abs/2505.24298
- V · Asynchronous RLHF, 2024-10-23 — https://arxiv.org/abs/2410.18252
- V · Let's Verify Step by Step, 2023-05-31 — https://arxiv.org/abs/2305.20050
- V · Generative Verifiers, 2024-08-27 — https://arxiv.org/abs/2408.15240
- V · RewardBench, 2024-03-20 — https://arxiv.org/abs/2403.13787
- V · SWE-RL, 2025-02-25 — https://arxiv.org/abs/2502.18449
- V · SWE-Gym, 2024-12-30 — https://arxiv.org/abs/2412.21139
- V · OSWorld, 2024-04-11 — https://arxiv.org/abs/2404.07972
- V · UI-TARS, 2025-01-21 — https://arxiv.org/abs/2501.12326
- V · Constitutional AI, 2022-12-15 — https://arxiv.org/abs/2212.08073
- V · Claude's Character, 2024-06-08 — https://www.anthropic.com/research/claude-character
- V · Claude's new constitution, 2026-01-22 — https://www.anthropic.com/news/claude-new-constitution (text: https://www.anthropic.com/constitution)
- V · Deliberative Alignment, 2024-12-20 — https://arxiv.org/abs/2412.16339
- V · OpenAI Model Spec (living; 2026-08-18 version) — https://model-spec.openai.com/
- V · On-Policy Distillation of LMs (GKD), 2023-06-23 — https://arxiv.org/abs/2306.13649
- S · On-Policy Distillation (Thinking Machines), 2025-10-27 — https://thinkingmachines.ai/blog/on-policy-distillation
- S · Defeating Nondeterminism in LLM Inference (Thinking Machines), 2025-09-10 — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

### Interpretability, alignment, governance
- S · Towards Monosemanticity, 2023-10-04 — https://transformer-circuits.pub/2023/monosemantic-features/
- S · Scaling Monosemanticity, 2024-05 — https://transformer-circuits.pub/2024/scaling-monosemanticity/
- V · Scaling and evaluating sparse autoencoders (OpenAI), 2024-06-06 — https://arxiv.org/abs/2406.04093
- V · Gemma Scope, 2024-08-09 — https://arxiv.org/abs/2408.05147
- V · Transcoders Find Interpretable LLM Feature Circuits, 2024-06-17 — https://arxiv.org/abs/2406.11944
- V · Circuit Tracing, 2025-03-27 — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- V · On the Biology of a Large Language Model, 2025-03-27 — https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- S · Emergent Introspective Awareness in LLMs, 2025-10-29 — https://transformer-circuits.pub/2025/introspection/index.html
- S · Weight-sparse transformers have interpretable circuits (OpenAI), 2025-11 — https://arxiv.org/abs/2511.13653
- V · Persona Vectors, 2025-07-29 — https://arxiv.org/abs/2507.21509
- V · Sleeper Agents, 2024-01-10 — https://arxiv.org/abs/2401.05566
- V · Alignment faking in LLMs, 2024-12-18 — https://arxiv.org/abs/2412.14093
- V · Agentic misalignment, 2025-06-20 — https://www.anthropic.com/research/agentic-misalignment
- V · Auditing LMs for hidden objectives, 2025-03-14 — https://arxiv.org/abs/2503.10965
- V · Emergent Misalignment, 2025-02-24 — https://arxiv.org/abs/2502.17424
- V · Persona Features Control Emergent Misalignment (OpenAI), 2025-06-24 — https://arxiv.org/abs/2506.19823
- S · Natural Emergent Misalignment from Reward Hacking in Production RL, 2025-11 — https://arxiv.org/abs/2511.18397
- V · Monitoring Reasoning Models for Misbehavior (OpenAI), 2025-03-14 — https://arxiv.org/abs/2503.11926
- V · Chain of Thought Monitorability, 2025-07-15 — https://arxiv.org/abs/2507.11473
- V · Reasoning Models Don't Always Say What They Think, 2025-05-08 — https://arxiv.org/abs/2505.05410
- V · Stress Testing Deliberative Alignment for Anti-Scheming, 2025-09-19 — https://arxiv.org/abs/2509.15541
- V · Anthropic RSP v3.0, 2026-02-24 — https://www.anthropic.com/news/responsible-scaling-policy-v3
- S · OpenAI Preparedness Framework v2, 2025-04-15 — https://openai.com/index/updating-our-preparedness-framework/
- V · Google DeepMind Frontier Safety Framework (v3, 2025-09-22; v3.1 2026-04-17) — https://deepmind.google/blog/strengthening-our-frontier-safety-framework/
- V · Anthropic system cards index — https://www.anthropic.com/system-cards

### Evaluation and inference
- V · SWE-bench, 2023-10-10 — https://arxiv.org/abs/2310.06770
- S · Introducing SWE-bench Verified, 2024-08-13 — https://openai.com/index/introducing-swe-bench-verified/
- S · Why we no longer evaluate SWE-bench Verified (OpenAI), 2026-02-23 — https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- V · SWE-Bench Pro, 2025-09-21 — https://arxiv.org/abs/2509.16941
- V · Humanity's Last Exam, 2025-01-24 — https://arxiv.org/abs/2501.14249
- V · FrontierMath, 2024-11-07 — https://arxiv.org/abs/2411.04872
- V · ARC-AGI-2, 2025-05-17 — https://arxiv.org/abs/2505.11831
- V · GPQA, 2023-11-20 — https://arxiv.org/abs/2311.12022
- V · Measuring AI Ability to Complete Long Software Tasks (METR), 2025-03-18 — https://arxiv.org/abs/2503.14499
- V · Clarifying limitations of time horizon (METR), 2026-01-22 — https://metr.org/notes/2026-01-22-time-horizon-limitations/
- V · GSM1k, 2024-05-01 — https://arxiv.org/abs/2405.00332
- V · Judging LLM-as-a-Judge (MT-Bench), 2023-06-09 — https://arxiv.org/abs/2306.05685
- V · Epoch AI benchmarks hub — https://epoch.ai/benchmarks
- V · PagedAttention (vLLM), 2023-09-12 — https://arxiv.org/abs/2309.06180
- V · SGLang, 2023-12-12 — https://arxiv.org/abs/2312.07104
- V · DistServe, 2024-01-18 — https://arxiv.org/abs/2401.09670
- V · Splitwise, 2023-11-30 — https://arxiv.org/abs/2311.18677
- V · Mooncake, 2024-06-24 — https://arxiv.org/abs/2407.00079
- V · Fast Inference via Speculative Decoding, 2022-11-30 — https://arxiv.org/abs/2211.17192
- V · Medusa, 2024-01-19 — https://arxiv.org/abs/2401.10774
- V · EAGLE-3, 2025-03-03 — https://arxiv.org/abs/2503.01840
- V · KIVI, 2024-02-05 — https://arxiv.org/abs/2402.02750
- V · GPTQ, 2022-10-31 — https://arxiv.org/abs/2210.17323
- V · AWQ, 2023-06-01 — https://arxiv.org/abs/2306.00978

### Research practice and courses not to duplicate
- V · Research Taste Exercises (Olah) — https://colah.github.io/notes/taste/
- V · Deep Learning Tuning Playbook (Google) — https://github.com/google-research/tuning_playbook
- V · A Recipe for Training Neural Networks (Karpathy), 2019 — https://karpathy.github.io/2019/04/25/recipe/
- U · An Opinionated Guide to ML Research (Schulman) — http://joschu.net/blog/opinionated-guide-ml-research.html
- V · Stanford CS336 (Spring 2026) — https://cs336.stanford.edu/ ; CS25 V6 — https://web.stanford.edu/class/cs25/ ; Berkeley Advanced LLM Agents — https://llmagents-learning.org/sp25

### Added in revision 2
- V · SWE-Bench Pro, 2025-09-21 — https://arxiv.org/abs/2509.16941 (already listed above; now required in 18.3)
- S · Grok 4 announcement (xAI) — https://x.ai/news/grok-4 (scoped RL-compute statement only)
- S · Introducing OpenAI o3 and o4-mini (OpenAI), 2025-04 — https://openai.com/index/introducing-o3-and-o4-mini/ (scoped compute statement only)
- S · Epoch AI, Notes on GPT-5 training compute — https://epoch.ai/gradient-updates (estimate; locate exact URL at Phase 0)

## 15. Decisions

Decided with Tal (2026-10-03):
1. Title "Frontier LLM Research Engineering", repo `frontier-llm-research-course`, slug `frontier-llm-research`.
2. Free course.
3. Main path on rented GPUs (production grade); free Colab GPU and free CPU variants for every lab, secondary.
4. One course.
5. Light `PUBLISHING_WARNING.md` (written).

Open (revision 2):
6. DECIDED 2026-10-03 — per-learner main-path budget accepted (~650–1,150 H100-hours, ~USD 1,300–3,500 before course-provided artefacts).
7. DECIDED 2026-10-03 — no funding; pilots run as scaled pilots on Tal's paid Colab account (section 12.1); no course-produced long RL runs.
7b. DECIDED 2026-10-03 — pilot traces and the Baseline-0 checkpoint are hosted on a free Hugging Face account.
8. DECIDED 2026-10-03 — Module 21 (multimodal elective) is added after the first release.
9. DECIDED 2026-10-07 — Stage D base model: Qwen/Qwen3-1.7B-Base (revision `ea980cb`, Apache-2.0, dense Qwen3 layout as Baseline-0); documented alternative allenai/OLMo-2-0425-1B (open data) for contamination-sensitive lessons; Stage D RL results carry a random- or format-reward control arm.
