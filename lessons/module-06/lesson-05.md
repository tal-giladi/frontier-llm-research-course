---
id: "06.5"
module: 6
minutes: 25
practice_minutes: 40
prerequisites: ["06.1", "06.3"]
objectives:
  - Explain MatFormer's nested FFN, its sampled-granularity training and Mix'n'Match, and what "elastic" buys at deployment.
  - Implement the nested FFN and a Mix'n'Match width selection and verify them against the shared module.
  - Train a toy MatFormer and compare each nested submodel's loss, FLOPs and consistency with independently trained models.
  - Describe Per-Layer Embeddings only as far as Google and the Transformers implementation document them, and compute the size of a PLE table from a configuration.
volatility: implementation
sources:
  - title: "Devvrit, Kudugunta et al. — MatFormer: Nested Transformer for Elastic Inference (sections 3.1–3.4, 4.1)"
    url: https://arxiv.org/abs/2310.07707
  - title: "Google — Introducing Gemma 3n: the developer guide (MatFormer, Mix-n-Match, Per-Layer Embeddings), 2025-06"
    url: https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
  - title: "Gemma Team — Gemma 4 Technical Report (section 2: E2B/E4B use per-layer embeddings as in Gemma 3n)"
    url: https://arxiv.org/abs/2607.02770
  - title: "Hugging Face Transformers v5.18.0 — models/gemma3n/modeling_gemma3n.py and configuration_gemma3n.py"
    url: https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/gemma3n/modeling_gemma3n.py
last_verified: "2026-10-04"
---

# 06.5 · Elastic architectures

Extension: most models come in a family of sizes trained separately. An elastic architecture trains one model that contains smaller, usable models inside it. MatFormer does this by ordering the FFN's neurons so that every prefix of them is a working FFN, training several prefix widths at once, and choosing a width per layer at deployment. Google's Gemma 3n uses MatFormer to ship a smaller model extracted from a larger one, and adds Per-Layer Embeddings, a large table that can stay off the accelerator. This lesson builds the nested FFN, trains a toy one and measures its submodels against independently trained ones.

## Why this matters at a frontier lab

Serving targets differ: a phone, a laptop GPU, a datacentre accelerator, a latency tier. Training and maintaining a separate model for each is expensive, and separately trained models of different sizes behave differently, which hurts speculative decoding (the draft disagrees with the verifier) and makes cross-device behaviour drift. MatFormer claims to remove both costs at once; Gemma 3n is a production use: "During the MatFormer training of the 4B effective parameter (E4B) model, a 2B effective parameter (E2B) sub-model is simultaneously optimized within it" (Google developer guide). The research question is narrow and testable: *does one nested model match separately trained models at each size, and at what training cost?*

## The idea

### Nested FFN

An FFN with $d_{\mathrm{ff}}$ hidden neurons; granularity $i$ uses the first $m_i$ of them, $1 \le m_1 < \dots < m_g = d_{\mathrm{ff}}$ (MatFormer section 3.1):

$$\mathrm{FFN}_i(x) = W_{\mathrm{down}}[:, :m_i]\,\big(\mathrm{silu}(W_{\mathrm{gate}}[:m_i]\,x) \odot W_{\mathrm{up}}[:m_i]\,x\big).$$

So the first $m_1$ neurons belong to every submodel, and are "most significant". The paper uses $g = 4$ with FFN ratios $\{0.5, 1, 2, 4\}$, i.e. widths $\{d_{\mathrm{ff}}/8, d_{\mathrm{ff}}/4, d_{\mathrm{ff}}/2, d_{\mathrm{ff}}\}$; attention and embeddings are shared.

### Training: sample one granularity per step

"For each step we randomly sample a MatFormer granularity $i = 1, \dots, g$ and train for it" (section 3.2, Eq. 2), uniformly in most experiments. Each step therefore costs the FLOPs of the sampled width; averaged over steps, the FFN costs the mean width. The paper argues this beats DynaBERT-style joint optimisation of all widths in one step, which gives "fewer gradient updates" for the same compute (section 4.1).

### Mix'n'Match

At inference, any width per layer is allowed, not just the $g$ trained ones. The paper recommends "selecting sub-blocks with minimal granularity changes across layers", sizes that never shrink with depth, and finds the "increasing with minimum slope" configuration best (section 3.3). These configurations "were never explicitly optimized" yet land on the accuracy-vs-compute curve of the trained ones (section 4.1). Gemma 3n exposes the same idea "primarily by adjusting the feed forward network hidden dimension per layer (from 8192 to 16384) and selectively skipping some layers" (developer guide).

### Per-Layer Embeddings (Gemma 3n, Gemma 4 E2B/E4B) — as documented

Google's description: PLE lets "a significant portion of these parameters (the embeddings associated with each layer) to be loaded and computed efficiently on the CPU", so "only the core transformer weights (approximately 2B for E2B and 4B for E4B) need to sit in" accelerator memory, for 5B and 8B total parameters (developer guide). The Gemma 4 report says E2B and E4B "use per-layer embeddings as in Gemma 3n", 2.3B and 4.5B effective out of 5B and 8B (section 2). Google does not publish the equations; the Hugging Face Transformers implementation (v5.18.0, `modeling_gemma3n.py`) shows one: a table `embed_tokens_per_layer` of shape $(V_{\mathrm{ple}}, L \cdot d_{\mathrm{ple}})$, combined with a projection of the ordinary embedding, gives every layer $l$ a $d_{\mathrm{ple}}$-vector per token, which is gated by the layer's hidden state, projected back to $C$ and added. The table is read one row per token — a lookup, like Engram's tables (lesson 06.4) but keyed by the current token alone — which is why it can stream from host memory. The course's `PerLayerEmbedding` follows that code, simplified (labelled; Gemma 3n's AltUp and LAuReL are left out).

## Worked example

### Nested widths of the toy model

Toy $d_{\mathrm{ff}} = 384$: widths $\{48, 96, 192, 384\}$. One FFN layer has $3 C m$ weights: $3 \cdot 128 \cdot 48 = 18{,}432$ at $m = 48$ up to $147{,}456$ at 384. Uniform sampling trains at mean width $(48 + 96 + 192 + 384)/4 = 180$, so a MatFormer step costs 9.53 MFLOP/token against Baseline-0's 11.41 (`blocks.accounting`): MatFormer training is *cheaper* per step than training the full model, and each width gets a quarter of the steps.

### Mix'n'Match for a budget

Four layers, target mean width 240: options at granularities (192, 384): $k$ layers at 192 and $4 - k$ at 384 give means 384, 336, 288, 240, 192. $k = 3$ hits 240 exactly: $[192, 192, 192, 384]$ — widths grow with depth and change by one granularity. Target 84: $[48, 96, 96, 96]$ gives 84.

### A PLE table from a configuration

Transformers' `Gemma3nTextConfig` defaults: `vocab_size_per_layer_input` 262,144, `num_hidden_layers` 35, `hidden_size_per_layer_input` 256. The table has $262{,}144 \cdot 35 \cdot 256 = 2{,}348{,}810{,}240$ parameters — 2.35B values that a token touches only $35 \cdot 256 = 8{,}960$ of. (These are the library's default sizes; we do not claim they are E2B's or E4B's exact configuration.)

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| FFN weights (full) | gate/up (I, C), down (C, I) | fp32 master, bf16 compute | GPU / CPU |
| active slice at width m | (m, C), (C, m) | views, no copy | same |
| hidden activations | (B, T, m) | same | same |
| PLE table | ($V_{\mathrm{ple}}$, L·d) | stored; host memory at inference (Google) | host / CPU |
| per-layer inputs | (B, T, L, d) | bf16 / fp32 | GPU |

Inference FLOPs scale with the chosen widths: for the toy model, 2.77 MFLOP/token (forward) at width 48 everywhere vs 3.80 at 384 — an 8× smaller FFN saves only 27%, because the head ($2VC = 2.1$M forward) and attention do not shrink. Memory: one checkpoint holds every granularity, so co-locating a draft and a verifier costs one model's weights.

## Build it

`labs/common/frontierlab/blocks/matformer.py`: `NestedMLP` (Qwen3 MLP names; full width is an ordinary Baseline-0 FFN), `matformer_widths`, `mix_n_match`, `PerLayerEmbedding`. In a BlockLM:

```python
model = BlockLM(with_blocks(toy(vocab_size=8192), ffn="matformer"))   # widths {48, 96, 192, 384}
model.set_widths(96)                  # evaluate granularity 2 everywhere
model.set_widths([96, 96, 192, 192])  # a Mix'n'Match configuration
```

In training mode `BlockLM` draws one width per forward pass from a generator seeded by a step counter stored in the state dict (`mat_calls`), so a resumed run draws the same widths (tested: stop at step 7, resume, bit-identical weights). Correctness checks, passing: the nested FFN at width $m$ equals a separate SwiGLU built from the first $m$ neurons, exactly; float64 gradcheck; causal check and cached-decode agreement for the MatFormer model; PLE shapes and table size; a PLE model passes causal and cache checks.

## What the evidence says

- **MatFormer — MODEL-SPECIFIC in production (Google), PROMISING as a method.** The paper trains decoders up to 850M parameters: submodels from 582M to 850M "each exhibiting better validation loss and one-shot downstream evaluations than independently trained counterparts"; extracted submodels are "up to 11.5% more consistent" with the XL model than baselines, and speculative decoding with a 393M MatLM draft and 850M verifier is "up to 6% faster than traditional speculative decoding" (abstract, section 4.1, Table 2); loss-vs-compute scaling fits match vanilla Transformers (section 4.1). PUBLICLY DOCUMENTED, one group (Google and collaborators), small scale.
- **Gemma 3n** ships E2B extracted from E4B and Mix-n-Match between them (company claim, developer guide). **PLE** in Gemma 3n and Gemma 4 E2B/E4B is documented as a memory placement technique with effective vs total parameter counts; its quality contribution is not published.
- **Open questions:** the cost to the largest model of training the nested ones (MatFormer's XL vs a separately trained XL), elasticity in attention (the paper sketches it, section 3.1), and whether nesting interacts with MoE or other changes.

## Lab

**Folder:** [`labs/module-06/lesson-05/`](../../labs/module-06/) · **Time:** about 40 minutes · **Pass check:** `pytest labs/module-06/lesson-05` passes; your notes contain the table of granularities, Mix'n'Match configurations and independent baselines with losses and FLOPs, the two consistency numbers, and a paragraph on what a 200-step toy run can say.

### Experiment contract

- **Question:** at toy scale, are MatFormer's nested submodels as good as independently trained models of the same width, and more consistent with the largest one? Decision informed: none for Lineage-F; the lab illustrates elastic training and how to evaluate it.
- **Hypotheses and status:** H1: each granularity is within noise of an independently trained model of its width at equal *per-width* training — reported at 78M–850M (section 4.1), but here each width gets about a quarter of the steps, so **H1 is expected to fail for the full width at 200 steps**; H2: the MatFormer S submodel agrees with its XL more often than an independent S agrees with Baseline-0 — reported (Figure 2c).
- **Baseline:** Baseline-0 at width 384 and an independently trained width-48 model, same steps and data.
- **Changed variable:** the FFN (nested, sampled widths). **Controlled:** data, steps, seed, schedule, attention, embeddings.
- **Comparison axis:** **equal steps and tokens** for every model (the paper's per-granularity token budget differs; stated). MatFormer's training FLOPs are lower (9.53 vs 11.41 MFLOP/token) — report them.
- **Budget:** free CPU about 6 minutes of training.
- **Metrics and decision rule:** held-out loss per configuration (256 windows); inference FLOPs per token; consistency on 64 windows. One seed: differences are reported, not tested.
- **Correctness checks:** `pytest labs/common/tests/test_blocks.py -k "nested or matformer or ple"`.
- **Limits:** one seed, 200 steps, toy width; nothing here transfers to Gemma-scale models.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | optional; not run in this build | `elastic.py --variant main --device cuda` (pilot-30m) |
| Free GPU (Colab/Kaggle T4) | T4 | `elastic.py --variant t4 --device cuda` |
| Free CPU | laptop; measured below | as written |

### Steps

1. **Implement** `nested_ffn`, `mix_n_match`, `consistency` and `ple_params`; run `pytest labs/module-06/lesson-05`.
2. **Run** `python labs/module-06/lesson-05/elastic.py` (trains MatFormer and the width-48 model; reuses Baseline-0 from lesson 06.1).
3. **Report** the table, the consistency numbers and the parameter-storage comparison; say whether each hypothesis holds here and why the step budget matters.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, with another build job sharing the CPU, 2026-10-04):

| Model (toy, 200 steps, seed 0) | Held-out loss | Inference MFLOP/token |
|---|---|---|
| MatFormer, width 48 everywhere (g = 1) | 6.9050 | 2.77 |
| MatFormer, width 96 (g = 2) | 6.8988 | 2.92 |
| MatFormer, width 192 (g = 3) | 6.8981 | 3.21 |
| MatFormer, width 384 (g = 4, the full model) | 6.9020 | 3.80 |
| Mix'n'Match [48, 96, 96, 96] | 6.9076 | 2.88 |
| Mix'n'Match [192, 192, 384, 384] | 6.8992 | 3.51 |
| independent Baseline-0 (width 384) | 6.6857 | 3.80 |
| independent width-48 model | 6.8704 | 2.77 |

- Consistency (greedy next-token agreement, 64 windows): MatFormer S vs its own XL **0.845**; independent width-48 vs Baseline-0 **0.594**.
- Storage: one MatFormer checkpoint (1,836,416 parameters) holds every width; Baseline-0 plus the width-48 model are 3,156,736.
- Runtime: two new training runs 3.5 minutes (MatFormer 9.53 MFLOP/token in training vs Baseline-0's 11.41); evaluation seconds.

What to write about it. H1 fails as expected for this budget: every MatFormer width is about 0.21 nats behind Baseline-0 at full width and 0.03 behind the independent width-48 model at width 48, because each width saw about a quarter of the 200 steps at the steep start of training — and the four widths are almost equally good (6.898–6.905), so nesting has not yet produced the ordered quality the paper reports. H2 holds, but read it carefully: the MatFormer submodels agree with their XL 85% of the time partly because all four share most weights *and* are equally undertrained. The paper's claims are at 78M–850M parameters and per-width token budgets; this run checks the mechanism and the evaluation, not the claim.

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-05`.

</details>

## Common mistakes

- **Slicing the wrong axis.** `gate_proj.weight` and `up_proj.weight` are $(I, C)$: slice rows; `down_proj.weight` is $(C, I)$: slice columns.
- **Comparing MatFormer's XL with Baseline-0 at equal steps and calling the gap "the cost of elasticity".** The XL width trains on a quarter of the steps; compare at equal per-width tokens, or report the step budget next to the number.
- **Counting width savings as model savings.** The output head, attention and embeddings do not shrink; at small width the head dominates.
- **Describing PLE beyond what is documented.** Google publishes the memory placement and the parameter counts; the equations above come from the Transformers code, which is an implementation, not a specification.

## References

- Devvrit, S. Kudugunta et al., *MatFormer: Nested Transformer for Elastic Inference*, sections 3.1–3.4, 4.1, Table 2. https://arxiv.org/abs/2310.07707
- Google, *Introducing Gemma 3n: the developer guide*. https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
- Gemma Team, *Gemma 4 Technical Report*, section 2. https://arxiv.org/abs/2607.02770
- Hugging Face Transformers v5.18.0, `models/gemma3n/modeling_gemma3n.py`, `configuration_gemma3n.py`. https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/gemma3n/modeling_gemma3n.py
- Shared code: `labs/common/frontierlab/blocks/matformer.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[06.6 · Tokenizer-free models: Byte Latent Transformer](lesson-06.md)
