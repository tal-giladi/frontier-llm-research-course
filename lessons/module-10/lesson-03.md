---
id: "10.3"
module: 10
minutes: 40
practice_minutes: 150
prerequisites: ["10.1", "10.2"]
objectives:
  - Describe the rephrasing recipes of WRAP, Nemotron-CC and Kimi K2 and the synthetic-data mix of Phi-4 from their reports, with the numbers each one measured.
  - Rephrase a small corpus with a pinned open model on CPU and count the generator's compute exactly from its architecture and token counts.
  - Check synthetic text for fidelity (numbers kept and invented, content recall), diversity (novelty, distinct n-grams, repeated openings) and contamination before training on it.
  - Measure the downstream utility of rephrasing against repetition at equal training tokens, with a probe the arms never train on, and report the result against the generator compute it cost.
volatility: concept
sources:
  - title: "Maini et al., Rephrasing the Web (WRAP) (abstract, sections 3.1, 6.1, 6.2, 7.1)"
    url: https://arxiv.org/abs/2401.16380
  - title: "Su et al., Nemotron-CC (section 2.3: Wikipedia-style rephrasing and four high-quality prompts, Mistral NeMo 12B)"
    url: https://arxiv.org/abs/2412.02595
  - title: "Kimi K2: Open Agentic Intelligence (section 2.2 and Table 1: knowledge and math rephrasing)"
    url: https://arxiv.org/abs/2507.20534
  - title: "Phi-4 Technical Report (section 2.2, Table 3, Table 5, appendix B.1)"
    url: https://arxiv.org/abs/2412.08905
  - title: "Shumailov et al., The Curse of Recursion: Training on Generated Data Makes Models Forget"
    url: https://arxiv.org/abs/2305.17493
  - title: "Qwen/Qwen3-0.6B model card (Apache-2.0; the generator, revision c1899de)"
    url: https://huggingface.co/Qwen/Qwen3-0.6B
last_verified: "2026-10-04"
---

# 10.3 · Synthetic and rephrased data

When good text runs out, labs make more of it: they ask a model to rewrite web pages in a cleaner style, to turn them into questions and answers, or to write new material from seeds. This lesson rephrases a small corpus with a small open model on a laptop, checks what came out for errors, sameness and leaked benchmark text, and then asks the question that decides whether it was worth it: at equal training tokens, does rephrased data beat simply repeating the originals, once the compute spent generating it is on the bill?

## Why this matters at a frontier lab

Synthetic data is now a large share of some frontier mixtures (Phi-4's mix is 40% synthetic by tokens, Table 5), and rephrasing is the cheapest way to stretch a scarce high-quality source. It is also where data quality is hardest to see. A rewritten page can drop a number, invent a date, or turn every document into the same template; a generator can reproduce benchmark items it memorised; and the generation itself can cost more compute than the training run that uses it. The lab's job is to measure all of that before the synthetic source enters a mixture, and to compare against the alternative that costs nothing: repeating the real data.

## The idea

### What the reports document

**WRAP** (Maini et al.; PUBLICLY DOCUMENTED). A "frozen Mistral-7B instruction-tuned model" rewrites C4 passages of at most 300 tokens in four styles: easy, medium ("Wikipedia" quality), hard (terse) and question-answer (section 3.1). Real and synthetic data are sampled "in a 1:1 ratio". The abstract reports that this "speeds up pre-training by ∼3×", "improves perplexity by more than 10% on average across different" Pile subsets and "improves zero-shot question answer accuracy across 13 tasks by more than 2%". Synthetic data alone was worse on some domains (section 6.1), so real data stays in the mix. On cost, section 7.1 reports about 25K A100 GPU-hours (with vLLM) to generate 85B tokens, against about 6K GPU-hours to train a model on 300B tokens: in that setting generation cost more than training.

**Nemotron-CC** (section 2.3; PUBLICLY DOCUMENTED). Rephrasing is split by quality. Low-quality documents are rewritten in Wikipedia style (336.3B tokens generated); high-quality documents get four prompts that produce new forms of the same content: diverse QA pairs (499.5B), distill (157.6B), extract knowledge (303.6B) and knowledge list (203.2B). The generator is Mistral NeMo 12B with FP8 inference, top-p 0.9 and temperature 0.5. The resulting 1.9T synthetic tokens are part of the 6.3T-token dataset of lesson 10.2.

**Kimi K2** (section 2.2; PUBLICLY DOCUMENTED). Knowledge data is rephrased with "style- and perspective-diverse prompting", "chunk-wise autoregressive generation" (long texts are split, rephrased per segment and stitched back) and "fidelity checks that compare the semantic alignment of each rephrased passage with its source". Table 1 is the experiment this lesson's lab copies in miniature, on SimpleQA accuracy:

| Strategy (same total tokens) | SimpleQA accuracy |
|---|---|
| original data, 10 epochs | 23.76 |
| rephrased once, 10 epochs | 27.39 |
| rephrased 10 times, 1 epoch | 28.94 |

Math documents are rewritten into a "learning-note" style following SwallowMath, and high-quality math in other languages is translated into English. The whole pretraining corpus is 15.5T tokens over four domains (web text, code, mathematics, knowledge).

**Phi-4** (section 2.2, Tables 3 and 5; PUBLICLY DOCUMENTED). "50 broad types of synthetic datasets", about 400B unweighted tokens, built from curated seeds with multi-step rewriting, self-revision and, for code, instruction reversal with execution checks. The final 10T-token mix (Table 5): web 15% (1.3T unique tokens, 1.2 epochs), web rewrites 15% (290B unique, 5.2 epochs), synthetic 40% (290B unique, 13.8 epochs), code 20% (820B, 2.4 epochs), acquired sources 10% (580B, 1.7 epochs). Synthetic-only training was much weaker on knowledge benchmarks such as TriviaQA (Table 3), which is why organic web data stays. Decontamination used 13-gram and 7-gram matching (appendix B.1).

Two observations. First, every recipe keeps real data in the mix; rephrasing adds forms of the same knowledge, it does not replace the source. Second, Phi-4 repeats its synthetic data 13.8 times: rephrasing is partly a way to make repetition less repetitive, which is exactly the comparison Kimi K2 reports.

### The risks

- **Fidelity.** A rewrite can drop or change facts. Small generators do it often; the lab measures numbers kept, numbers invented and content-word recall. Kimi K2's fidelity check uses semantic alignment; the course's checks are cheaper proxies.
- **Collapse.** Training on model output narrows the distribution: Shumailov et al. show that "tails of the original content distribution disappear" when models are trained recursively on generated content. One round of rephrasing with real data kept is not recursive training, but the same signal appears as low diversity: every output opening with the same phrase, distinct n-grams falling, pairwise similarity rising.
- **Contamination.** A generator can write text it memorised, including benchmark items, into a corpus that passed decontamination before rephrasing. Run the leakage check of lesson 10.1 on the *outputs*.
- **Licences.** The output carries the licence of the source and the terms of the generator's licence; record both (Qwen3-0.6B: Apache-2.0).

### Generator compute

Rephrasing is generation with a KV cache. For a decoder with $N$ non-embedding parameters, $L$ layers and attention width $d_{\text{attn}}$ (query heads × head dimension), a prompt of $P$ tokens and $G$ generated tokens cost, forward only, with one multiply-add as 2 FLOPs:

$$F = 2N(P + G) + 2 L d_{\text{attn}} P^2 + 4 L d_{\text{attn}} \sum_{c=P}^{P+G-1} c + 2VC\,(P + G),$$

where the second term is the prefill's $P^2/2$ query-key pairs at $4 d_{\text{attn}}$ FLOPs each, the third is each new token attending to its context $c$, and the last is the output head ($V$ vocabulary, $C$ width) when it is tied and therefore not in $N$. The data's cost is $F$ summed over documents, and it belongs in the budget of every run that uses the data (plan section 9).

## Worked example

### Generator FLOPs by hand

Tiny numbers first: $N = 10$, $L = 1$, $d_{\text{attn}} = 2$, $P = 3$, $G = 2$, no head term. $2 \cdot 10 \cdot 5 = 100$; $2 \cdot 1 \cdot 2 \cdot 9 = 36$; contexts $c = 3, 4$ sum to 7, so $4 \cdot 1 \cdot 2 \cdot 7 = 56$. Total 192 FLOPs (the lab test checks this).

Qwen3-0.6B (from its config: 28 layers, 16 query heads × head dimension 128, so $d_{\text{attn}} = 2048$; width 1,024; vocabulary 151,936; tied embeddings; about 440M non-embedding parameters), one document with $P = 300$ and $G = 250$:

- $2N(P+G) = 2 \cdot 4.40 \times 10^8 \cdot 550 = 4.84 \times 10^{11}$
- head: $2 \cdot 151{,}936 \cdot 1{,}024 \cdot 550 = 1.71 \times 10^{11}$
- prefill attention: $2 \cdot 28 \cdot 2048 \cdot 300^2 = 1.03 \times 10^{10}$
- decode attention: $\sum_{c=300}^{549} c = 250 \cdot 300 + 250 \cdot 249/2 = 106{,}125$; $4 \cdot 28 \cdot 2048 \cdot 106{,}125 = 2.43 \times 10^{10}$

$F \approx 6.9 \times 10^{11}$ FLOPs per document. The toy model trained on 819,200 tokens costs about $1.2 \times 10^7$ FLOPs per token ($9.4 \times 10^{12}$ for the run), so rephrasing 128 documents ($8.8 \times 10^{13}$) costs about 9 training runs. At small scale the generator dominates; at frontier scale the ratio depends on how many training tokens each generated token buys.

### Fidelity checks by hand

Source: "In 1999 the council approved 40,000 dollars for 3 new libraries in Springfield." Output: "Springfield's council approved 40000 dollars in 1999 to build 4 libraries, said Mayor Lee in 2001." Numbers in the source $\{1999, 40000, 3\}$; in the output $\{40000, 1999, 4, 2001\}$. Kept $2/3$; invented 2 (4 and 2001); the output also invented a mayor. The number checks catch two of the three errors; none of the cheap checks catches the invented person, which is why Kimi K2 compares meaning.

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| generator input | (batch, P_max) int64, left-padded, CPU | left padding so every sequence's last prompt token is at the same position |
| generator weights | 596M parameters, float32, CPU | about 2.4 GB in memory; bfloat16 halves it on GPU |
| KV cache | 2 (K, V) × 28 layers × 8 KV heads × 128 values per token per sequence | float32 on CPU: 224 KiB per token; a batch of 16 sequences of 600 tokens holds 2.1 GB |
| outputs | JSONL rows: source, output, prompt and generated tokens, FLOPs | `rephrased.jsonl`, appended per batch, so a restart continues |

Generation on CPU is slow: measured 6.7 generated tokens per second (15,039 tokens in 2,228 s for 128 documents; prompt tokens are processed too) for Qwen3-0.6B in float32 with batches of 16 on the build laptop (other jobs running). The main path uses a GPU and a serving engine (vLLM 0.30.0 in the course pins), which batches far more sequences; WRAP generated with vLLM on A100s.

## Build it

```bash
python -m frontierlab.datax.rephrase run --source labs/common/data/m10/web --docs 128 --style wiki \
    --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --out runs/m10/l103/gen-wiki
python -m frontierlab.datax.rephrase evaluate runs/m10/l103/gen-wiki
python -m frontierlab.datax.rephrase build runs/m10/l103/gen-wiki --dest labs/common/data/m10/reph-wiki
```

`run` cuts each source document to 192 Data-v0 tokens at a word boundary, wraps it in the style's instruction with the model's chat template (Qwen3's thinking mode switched off), samples with temperature 0.7 and top-p 0.9 from a fixed seed per batch, and writes one row per document with the exact prompt and generated token counts and $F$ from `generation_flops`. `compute.json` sums them. `evaluate` writes `quality.json` (fidelity, novelty, meta-text, diversity of outputs and of sources). `build` writes a training source in the Data-v0 layout whose provenance names the source dataset and revision, the generator, its revision and licence, the prompt and the sampling settings, and the generator FLOPs. Correctness checks in `labs/common/tests/test_datax.py`: the FLOPs formula against the hand value; the fidelity functions against the worked example.

## What the evidence says

- **Rephrasing web text into cleaner or new forms helps when mixed with real data: PROMISING.** Several labs report it (WRAP, Nemotron-CC, Kimi K2, Phi-4's web rewrites), each with its own evaluation; WRAP's ~3× speed-up and Kimi K2's SimpleQA table are single studies.
- **Rephrasing beats plain repetition of scarce data: PROMISING** (Kimi K2 Table 1, one lab, one benchmark).
- **Synthetic data must not replace real data entirely: ESTABLISHED in the reports cited** (WRAP section 6.1, Phi-4 Table 3), and consistent with the collapse results of Shumailov et al.
- **Large synthetic shares (Phi-4's 40%): MODEL-SPECIFIC**, tied to Phi-4's seeds, generators and evaluation targets.
- **Generation cost can exceed training cost: PUBLICLY DOCUMENTED for WRAP's setting** (section 7.1); in general it depends on how often each synthetic token is reused.
- **Course measurement (free CPU, 2026-10-04, other jobs running):** Qwen3-0.6B rephrased 128 web documents in 37 minutes on CPU (4.95e13 generator FLOPs, 5.3 times one training run); 85% of numbers kept, 3% of outputs with an invented number, 40% novel 4-grams (the small model often copies), no LAMBADA leakage in the outputs. At equal tokens, rephrasings beat repeating the originals on the QA probe by 0.55 nats (95% CI [−0.83, −0.26]), while the originals won on themselves; the probe shares the generator's style, so the gain is partly style (the lab's stated limit).

## Lab

**Folder:** [`labs/module-10/lesson-03/`](../../labs/module-10/) · **Time:** about 2.5 hours (about 2 hours of it unattended: generation, then training) · **Pass check:** `pytest labs/module-10/lesson-03` passes; `rephrase_ablation.py` prints the generation compute, the output checks, the leakage line and both tables; your write-up states the decision and the compute ratio.

### Experiment contract

- **Question:** for a small corpus that must be repeated to fill its share of a mixture, does training on Qwen3-0.6B rephrasings (alone, or 1:1 with the originals) teach its facts in a new form better than repeating the originals, at equal training tokens? Decision informed: whether Data-v1 includes a rephrased source and how to count its cost.
- **Hypothesis:** `reph` and `both` reach lower loss than `orig` on the QA probe (facts in a form no arm trained on), as in Kimi K2 Table 1; `orig` wins on the originals themselves (it trained on them); general held-out loss is unchanged within 0.02. Status: reported effect at large scale (Kimi K2); may not appear at 1.8M parameters with a 0.6B generator.
- **Baseline:** `orig` for the rephrase-vs-repeat question; `none` (the slot filled with more Data-v0) as the reference for whether the small corpus helps at all. Learning rate 3e-3, not tuned per arm.
- **Changed variable:** what fills the 25% slot of a 75% Data-v0 mixture. **Controlled:** 400 steps of 16 × 128 tokens (819,200 tokens), toy preset, warmup 40, cosine, seeds 0–2 (same initial weights per seed across arms), the 128 source documents, the evaluation sets.
- **Comparison axis:** equal training tokens. It does not include the generator's compute, which is reported next to the result as a ratio to training compute.
- **Budget:** free CPU: generation of two styles (about 37 minutes each with other jobs running) plus 12 training runs of about 200 s.
- **Metrics and decision rule:** primary: loss on the QA probe (validation set of the `qa` rephrasings), seed-level paired 95% interval of (arm − `orig`). Adopt rephrasing if the upper bound is below 0; reject if the lower bound is above 0; else inconclusive. Secondary: loss on the originals, Data-v0 and web validation loss, LAMBADA log-probability; generator FLOPs over one training run's FLOPs.
- **Correctness checks:** the output checks are run and reported before training; the leakage check of the outputs against LAMBADA; each arm's `mixture_accounting.json` shows the slot's tokens and epochs; the QA probe was generated with a different seed from the training rephrasings and is never a training source.
- **Fallback evidence:** the Module 10 pilot's 30M/70M runs (plan section 12.1) when published, labelled as analysis of provided traces; otherwise a null result with the MDE.
- **Limits:** 128 documents of 192 tokens; one generator; the probe is itself generated by the same model, so it shares the generator's style (an `reph` advantage on it could be style, not knowledge: compare with the loss on the originals); 1.8M parameters.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100: generation with vLLM 0.30.0 (bf16), training pilot-30m. Not run in this build; part of the Module 10 pilot | `rephrase_ablation.py --variant main` (20,000 documents; 12 runs of 4,000 steps × 64 × 1,024). PROJECTED training: 12 × 2.62e8 tokens × 0.321 GFLOP/token ÷ (989e12 × 0.3) = 0.95 GPU-hours; generation: 20,000 documents × about 6.9e11 FLOPs (worked example) = 1.4e16 FLOPs, a few GPU-minutes at serving MFU, dominated in practice by memory-bound decoding (measure it) |
| Free GPU (Colab/Kaggle T4) | T4 | `rephrase_ablation.py --variant t4` (2,000 documents, generation in fp16 with transformers) |
| Free CPU | laptop; measured: generation 37 + 39 minutes for the two styles (other jobs running), 12 runs of about 200 s, scoring about 1 min per run | the steps below |

### Steps

1. **Implement** the six TODOs in `lab.py`: generator FLOPs, the number check, novelty, distinct n-grams, the cost ratio, the decision rule. Run `pytest labs/module-10/lesson-03`.
2. **Generate** (unattended; the model download is 1.2 GB the first time, measured 5 minutes here): `python labs/module-10/lesson-03/rephrase_ablation.py --generate-only`. Read ten source/output pairs from `runs/m10/l103/gen-wiki/rephrased.jsonl` and mark every changed fact you find.
3. **Check and train** (unattended): `python labs/module-10/lesson-03/rephrase_ablation.py`.
4. **Write up:** the output checks next to your own reading of ten pairs (what did the cheap checks miss?); the decision on the primary metric; the loss on the originals for `orig` vs `reph` and what it says about the probe's limit; the compute ratio and the number of training runs the generation would have paid for.

<details>
<summary>Hint for TODO 1</summary>

The sum of contexts $c = P, \dots, P + G - 1$ is $G \cdot P + G(G - 1)/2$.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04/05 on the build laptop (torch 2.14.1 CPU, transformers 5.18.0; generation ran while other jobs shared the CPU). Generation, Qwen3-0.6B float32, batches of 16, temperature 0.7, top-p 0.9: wiki style 128 documents, 25,266 prompt and 15,039 generated tokens, 2,228 s, 4.95e13 FLOPs by `generation_flops`; QA probe 26,418 prompt and 15,455 generated tokens, 2,325 s, 5.15e13 FLOPs. Outputs were shorter than the worked example assumed (about 117 generated tokens per document), so the measured FLOPs are about half of its estimate. Training: 12 runs of 400 steps × 16 × 128, about 200 s each on a quiet machine; one training run is 9.35e12 FLOPs, so the wiki rephrasings cost 5.3 training runs.

Checks of the 128 wiki outputs: numbers kept 0.849, outputs with at least one invented number 3.1%, content-word recall 0.728, novel 4-grams 0.398, length ratio 0.77, meta-text 0.8%, no repeated openings, distinct bigrams 0.854 (sources 0.828), LAMBADA passages with ≥ 50% 13-gram overlap with the outputs 0 of 5,153. Reading the first pairs shows what the numbers mean: on forum pages the model mostly copied the text (low novelty), on a news digest it merged sentences and kept the facts. The orig slot repeated its 23.6k tokens 8.7 times.

Mean loss per seed (0, 1, 2) and seed-level differences:

| Arm | QA probe (primary) | originals | Data-v0 val | web val | LAMBADA |
|---|---|---|---|---|---|
| none (Data-v0 only) | 6.811, 6.678, 6.795 | 6.679, 6.698, 6.748 | 6.473, 6.496, 6.505 | 6.644, 6.644, 6.680 | −17.41, −17.19, −17.64 |
| orig − none | −0.485 [−0.747, −0.224] | −0.698 [−0.793, −0.602] | −0.092 [−0.125, −0.059] | −0.157 [−0.218, −0.096] | +0.08 [−0.36, +0.52] |
| reph − none | −1.032 [−1.287, −0.777] | −0.525 [−0.605, −0.445] | −0.041 [−0.070, −0.013] | −0.094 [−0.129, −0.058] | +0.07 [−0.60, +0.74] |
| both − none | −0.877 [−1.147, −0.607] | −0.491 [−0.647, −0.334] | +0.011 [−0.143, +0.165] | −0.046 [−0.181, +0.088] | +0.07 [−0.55, +0.70] |

Primary comparison, against repetition: **reph − orig on the QA probe −0.546, 95% CI [−0.830, −0.263]: adopt** under the rule; both − orig −0.392 [−0.806, +0.023]: inconclusive. What it shows. (1) The rephrased arm learned the QA-form facts better than the arm that repeated the originals, but the originals arm was better on the originals themselves (6.011 vs 6.183) and on general text: the probe and the rephrasings come from the same generator, so part of the gain is style matching, and the experiment cannot separate it from knowledge transfer. That limit was stated in the contract and it is the reason the decision is "adopt for the probe", not "rephrasing transfers knowledge". (2) A surprise: every arm with a repeated small slot beat the all-Data-v0 arm even on Data-v0's own validation set (orig −0.092). At 0.8M training tokens a 1.8M-parameter model is far from converged, and a few repeated documents seem to speed up learning generic token statistics (INFERENCE; not tested here). It is a reminder that "none" is not a neutral control at this scale. (3) The generator cost 5.3 training runs of compute for 15k generated tokens: at course scale the data costs more than the training that uses it. For Data-v1 the evidence is not strong enough to include a rephrased source by default; the project's example recipe leaves it out and says why.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-10/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-10/lesson-03`.

</details>

## Common mistakes

- **Leaving the generator's compute out of the comparison.** A synthetic arm that wins at equal training tokens may lose at equal total compute; report both.
- **Judging rephrasing on a probe written in the rephrasing's own style.** Style match lowers loss without any knowledge transfer; use a probe in a different form and the originals.
- **Trusting cheap fidelity checks.** Numbers and content words catch some errors; invented people and changed relations need a reading or a semantic check.
- **Decontaminating before rephrasing only.** Check the outputs too; the generator brings its own memory.
- **Replacing the real data.** Every recipe that reports gains keeps real data in the mix.
- **Sampling with temperature 0.** Greedy rewrites of similar pages converge to the same template; watch the repeated-openings number.

## References

- P. Maini et al., *Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling*, 2024. https://arxiv.org/abs/2401.16380
- D. Su et al., *Nemotron-CC*, 2024, section 2.3. https://arxiv.org/abs/2412.02595
- Moonshot AI, *Kimi K2: Open Agentic Intelligence*, 2025, section 2.2 and Table 1. https://arxiv.org/abs/2507.20534
- Microsoft, *Phi-4 Technical Report*, 2024, section 2.2, Tables 3 and 5, appendix B.1. https://arxiv.org/abs/2412.08905
- I. Shumailov et al., *The Curse of Recursion: Training on Generated Data Makes Models Forget*, 2023. https://arxiv.org/abs/2305.17493
- Qwen team, Qwen3-0.6B model card (revision `c1899de289a04d12100db370d81485cdf75e47ca`). https://huggingface.co/Qwen/Qwen3-0.6B
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[10.4 · Mixtures and micro-anneals](lesson-04.md)
