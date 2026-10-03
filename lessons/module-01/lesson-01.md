---
id: "01.1"
module: 1
minutes: 30
practice_minutes: 75
prerequisites: []
objectives:
  - Explain why recent open models made different choices for attention, optimizer, precision and post-training, and why that makes "which choice?" a research question rather than a recipe.
  - Derive Baseline-0's parameter count, training FLOPs per token and KV bytes per token by hand and check them against code.
  - Use the course correctness suite to find and fix a cached-decoding bug that ordinary training would not reveal.
  - Run, interrupt and exactly resume a training run, and verify the resume from its logs.
volatility: concept
sources:
  - title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (abstract: KV cache reduced by 93.3%)"
    url: https://arxiv.org/abs/2405.04434
  - title: "Gemma 3 Technical Report (5 local layers per global layer, 1024-token window, QK-norm)"
    url: https://arxiv.org/abs/2503.19786
  - title: "gpt-oss-120b & gpt-oss-20b Model Card (section 2.2: banded window, learned attention sinks)"
    url: https://arxiv.org/abs/2508.10925
  - title: "Kimi K2: Open Agentic Intelligence (MuonClip; 15.5T tokens with zero loss spikes)"
    url: https://arxiv.org/abs/2507.20534
  - title: "Kimi Linear: An Expressive, Efficient Attention Architecture (3:1 KDA:MLA hybrid)"
    url: https://arxiv.org/abs/2510.26692
  - title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (section 2.1: DeepSeek Sparse Attention)"
    url: https://arxiv.org/abs/2512.02556
  - title: "Why Did MiniMax M2 End Up as a Full Attention Model? (MiniMax, Hugging Face blog)"
    url: https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
  - title: "DeepSeek-V3 Technical Report (section 3.3: FP8 training)"
    url: https://arxiv.org/abs/2412.19437
last_verified: "2026-10-03"
---

# 01.1 · Diagnostic and map

This lesson checks that you have what the course assumes, shows why the course is organised around research questions rather than a recipe, and introduces the three tools you will use in every later lab: the reference model Baseline-0, the correctness suite, and exactly resumable training runs.

## Why this matters at a frontier lab

A research engineer's job is rarely "implement technique X". It is "should we use X here, and how sure are we?" The answer depends on context length, hardware, budget and what else is in the model, and it is usually decided by a small, careful experiment before an expensive run. Two things make that possible: a reference model whose behaviour you know well, and tools that catch bugs before they turn into wrong conclusions. A cached-decoding bug, for example, does not change training loss at all — it only shows up when the model generates, or when someone compares decode memory between two attention designs and gets numbers from code that is quietly wrong.

## The idea

### There is no single frontier block

The parent course ended with a Llama-style block: RoPE, RMSNorm, SwiGLU, grouped-query attention (GQA), and a DeepSeek-style MoE layer. Open models released in 2024–2026 kept most of that and then disagreed with each other, sometimes sharply:

| Choice | Some published answers (PUBLICLY DOCUMENTED) |
|---|---|
| How attention stores the past | GQA (Llama 3, Qwen3); a compressed latent per token, Multi-head Latent Attention (DeepSeek-V2 reports a 93.3% smaller KV cache than DeepSeek 67B); 5 sliding-window layers of 1,024 tokens per global layer (Gemma 3); alternating 128-token banded and dense layers plus a learned "sink" logit per head (gpt-oss) |
| Whether attention is quadratic | full attention everywhere (most models; MiniMax-M2 went *back* to full attention and explained why); 3 linear-attention layers per full layer (Kimi Linear, Qwen3-Next); learned top-k token selection (DeepSeek-V3.2 sparse attention) |
| Optimizer | AdamW (most); Muon with per-head logit clipping (Kimi K2, which reports 15.5T training tokens with zero loss spikes) |
| Training precision | BF16 (most); FP8 with fine-grained scaling (DeepSeek-V3) |

Each row is a trade-off. MLA saves decode memory but adds projections and a more complex cache; linear attention removes the quadratic cost but MiniMax reports that hybrids hid multi-hop reasoning deficits that small benchmarks did not show; Muon needed extra machinery (logit clipping) to stay stable at Kimi K2's scale. Nothing in these reports says "this is best". Each says "this worked for us, under our constraints, with this evidence".

So this course is organised around **questions** — *how should attention spend KV memory? when is sub-quadratic attention worth it? which optimizer?* — and treats each model as a case study: one answer, under stated constraints, with a certain amount of evidence. Throughout the course, each technique carries a maturity tag:

- **ESTABLISHED** — adopted independently by several labs, with ablations in more than one report (GQA, RoPE scaling, QK-norm).
- **PROMISING** — published with evidence, limited independent replication (hyper-connections, conditional memory).
- **MODEL-SPECIFIC** — one lab's choice, mostly evidenced by that lab (a particular hybrid ratio, a particular indexer design).

### Three tools you will use everywhere

1. **Baseline-0** — the course's reference model (`frontierlab.model.baseline0()`): a dense, decoder-only transformer in the Hugging Face Qwen3 layout (pre-norm RMSNorm, GQA with RoPE and QK-norm, SwiGLU, tied embeddings). Every architecture experiment in Stage B is one change to Baseline-0, trained on the same data with the same budget, so differences can be attributed to that change.
2. **The correctness suite** (`frontierlab.testing`) — checks that every new mechanism must pass *before* any comparison counts: gradients against numerical gradients, causality (changing future tokens must not change past outputs), full-sequence versus cached decoding, and agreement with a reference implementation.
3. **Exact resume** (`frontierlab.train.loop`) — checkpoints that contain the model, optimizer, step and every random-number-generator state, so an interrupted run continues as the *same* run. On the main path you will train on rented GPUs and on Colab sessions that disconnect; without exact resume, "the same run" quietly stops meaning anything.

## Worked example

Baseline-0's size, by hand. Symbols: $C$ hidden width, $L$ layers, $H$ query heads, $K$ key/value heads, $d$ head dimension, $I$ SwiGLU inner width, $V$ vocabulary. Baseline-0 has $C=768$, $L=12$, $H=12$, $K=4$, $d=64$, $I=2816$, $V=32768$.

Per layer:

- query and output projections: $2 \cdot C \cdot H d = 2 \cdot 768 \cdot 768 = 1{,}179{,}648$
- key and value projections: $2 \cdot C \cdot K d = 2 \cdot 768 \cdot 256 = 393{,}216$
- QK-norm gains: $2d = 128$
- SwiGLU (gate, up, down): $3 \cdot C \cdot I = 3 \cdot 768 \cdot 2816 = 6{,}488{,}064$
- two RMSNorm gains: $2C = 1{,}536$

Total per layer: $8{,}062{,}592$. Twelve layers plus the final norm ($768$) give the **non-embedding** count $N = 96{,}751{,}872$. The embedding is $V \cdot C = 25{,}165{,}824$; the output head shares it (tied), so the total is $121{,}917{,}696$.

Training FLOPs per token, with one multiply-add counted as 2 FLOPs and training ≈ 3 × forward (parent course lesson 07.4), at context $T = 1024$:

$$\text{forward} \approx 2N + 2 L T (H d) + 2 V C$$

- $2N = 193{,}503{,}744$ — every weight used once per token
- $2LT(Hd) = 2 \cdot 12 \cdot 1024 \cdot 768 = 18{,}874{,}368$ — attention scores and values (a token at position $t$ costs $4 (Hd) t$ per layer; averaged over $t \in [0, T]$ that is $2 (Hd) T$)
- $2VC = 50{,}331{,}648$ — the tied output head, which $N$ does not include

Forward ≈ $262.7$M FLOPs per token, so training ≈ $788.1$M FLOPs per token. Training on 2.5B tokens costs about $1.97 \times 10^{18}$ FLOPs.

Time on one H100 SXM (dense BF16 peak $989 \times 10^{12}$ FLOP/s) at a model-FLOPs utilization (MFU) of 30%: $1.97 \times 10^{18} / (0.30 \cdot 989 \times 10^{12}) \approx 6{,}640$ s $\approx 1.8$ hours. That MFU is an assumption until it is measured — PROJECTED, pending the Module 1 pilot. Small models often reach lower MFU than large ones, because their matrix multiplications are small and fixed overheads weigh more; Module 2 measures this.

KV cache per token in BF16 (2 bytes): $2 \text{ (K and V)} \cdot L \cdot K \cdot d \cdot 2 = 2 \cdot 12 \cdot 4 \cdot 64 \cdot 2 = 12{,}288$ bytes. At 32K tokens that is 384 MiB per sequence — the number Module 3 tries to shrink.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| input ids | (B, T) | int64 | GPU (main path) / CPU |
| hidden states | (B, T, 768) | bf16 under autocast, fp32 on CPU | same |
| queries after RoPE | (B, 12, T, 64) | same | same |
| keys / values after RoPE | (B, 4, T, 64) | same | same |
| cached keys / values (per layer) | (B, 4, S, 64), S = tokens so far | same | same |
| logits | (B, T, 32768) | fp32 (cast before the loss) | same |

The logits tensor dominates activation memory at small width: at B = 32, T = 1024 it is $32 \cdot 1024 \cdot 32768 \cdot 4$ bytes ≈ 4 GiB in fp32. Module 2 measures this and shows how to reduce it.

## Build it

The model's forward pass takes absolute token positions, so the same attention module serves training (positions $0..T-1$, no cache) and decoding (positions continue from the cache length):

```python
from frontierlab.model import LM, baseline0, param_counts
from frontierlab.testing import run_suite

cfg = baseline0()
print(param_counts(cfg)["non_embedding"])        # 96751872, as computed by hand above

toy = LM(cfg.with_(hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=32, intermediate_size=256, vocab_size=97))
print(run_suite(toy, vocab_size=97))
# {'causal_max_abs_diff': 0.0, 'cache_max_abs_diff_step1': 0.0, 'cache_max_abs_diff_chunk5': 0.0}
```

`run_suite` copies the model to float64 so that real bugs are not hidden by rounding: in float64 a correct implementation agrees to within about $10^{-15}$ (often exactly), while a masking bug produces differences of order $10^{-1}$ — the lab's planted bug gives 0.77.

The cache check is the one ordinary training never exercises. During training every query sees keys at positions $\le$ its own via a causal mask aligned at the top-left corner of the score matrix. During cached decoding the score matrix is rectangular — $T$ new queries against $S > T$ keys — and a top-left causal mask lets the first new query see only the first cached key. Training loss is unaffected; generation is wrong. Your lab plants exactly this bug and has you fix it.

## What the evidence says

- The table above is PUBLICLY DOCUMENTED, each row from the cited report section (see References). The trade-off statements about MLA cost and Muon stability are REASONABLE INDUSTRY PRACTICE summaries; the MiniMax-M2 reasons are the company's own account (company claim).
- What closed labs use is mostly undisclosed. The GPT-5 system card describes a system of a fast model, a reasoning model and a router; it does not disclose architecture or size. Any statement about closed models' internals in this course is labelled INFERENCE.
- Baseline-0's MFU and time figures are PROJECTED until the Module 1 pilot measures them.

## Lab

> [!NOTE]
> This lab is a diagnostic and has no comparison, so it has no experiment contract. Every later lab that compares anything starts with one ([template](../../templates/experiment-contract.md)).

**Folder:** [`labs/module-01/lesson-01/`](../../labs/module-01/) · **Time:** about 75 minutes · **Pass check:** `pytest labs/module-01/lesson-01` passes, and step 4's two loss logs are identical.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× GPU (any CUDA GPU; an L4 or A100 is plenty), ~15 min GPU time | steps 1–3 as below; step 4 with `--preset pilot-10m --steps 400 --batch 32 --seq 512 --dtype bf16` on the Data-v0 slice |
| Free GPU (Colab/Kaggle T4) | T4 | same as main path but `--dtype fp32` (the T4 has no BF16 tensor cores) |
| Free CPU | laptop; step 4 measured at about 7 minutes (3 runs of the toy model, ~3,300 tokens/s on a 16-thread laptop, 2026-10-03) | steps 1–4 exactly as written |

1. **Count by hand.** In `lab.py`, implement `count_params(cfg)` from the formulas in the worked example — do not call `param_counts`. The test checks it against the real module for several configurations, including one without QK-norm and one with untied embeddings.
2. **Find the bug.** `lab.py` registers an attention kind `"gqa-lab"` whose `attend()` uses a top-left causal mask. Run the provided check:

   ```bash
   python labs/module-01/lesson-01/diagnose.py
   ```

   It prints the causal check (passes) and the cache check (fails with a large difference). Write down, in one sentence, why training loss would not reveal this bug.
3. **Fix it.** Implement `attend(q, k, v, q_pos, k_pos)` so query $i$ attends to key $j$ exactly when `k_pos[j] <= q_pos[i]`. Re-run `diagnose.py`: both checks must pass (differences of $10^{-15}$ or less; the reference solution gives exactly 0).
4. **Interrupt and resume.** Prepare the data once (`python -m frontierlab.data.prepare --docs 20000 --vocab 8192`, about 3 minutes and 60 MB; measured 2026-10-03 on a laptop), then:

   ```bash
   python -m frontierlab.train.loop --run runs/l11-straight --preset toy --steps 300 --batch 16 --seq 128
   python -m frontierlab.train.loop --run runs/l11-resumed  --preset toy --steps 300 --batch 16 --seq 128 --stop-after 150
   python -m frontierlab.train.loop --run runs/l11-resumed  --preset toy --steps 300 --batch 16 --seq 128
   python labs/module-01/lesson-01/compare_logs.py runs/l11-straight runs/l11-resumed
   ```

   `compare_logs.py` prints the largest difference between the two runs' logged training losses. It must be exactly 0. Then delete `"data_rng"` handling from a *copy* of the loop (or pass a different `--data-seed` on the resume) and see how a run that "resumed" with the wrong data order drifts.

<details>
<summary>Hint for step 3</summary>

Build a boolean matrix of shape `(len(q_pos), len(k_pos))` with `k_pos[None, :] <= q_pos[:, None]` and pass it as `attn_mask` to `F.scaled_dot_product_attention`. With grouped-query attention also pass `enable_gqa=True` when the number of query heads differs from key/value heads.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-01/lesson-01/solution.py` contains a reference implementation. Check it with `LAB_TARGET=solution pytest labs/module-01/lesson-01`.

</details>

## Common mistakes

- **Counting the tied head twice.** With tied embeddings the head *is* the embedding matrix; `sum(p.numel() for p in model.parameters())` counts it once, and so must your formula.
- **Using N with the embedding in `6·N·D`.** Kaplan-style estimates use non-embedding parameters; at Baseline-0's size the embedding is 21% of the total, so mixing the two conventions is a 20% error in every compute estimate.
- **Trusting `is_causal=True` with a cache.** It aligns the mask to the top-left corner; with more keys than queries that is wrong. Always test cached decoding against a full forward.
- **"Resuming" without the data RNG.** The run continues, the loss looks plausible, and it is no longer the run your experiment contract describes.

## References

- DeepSeek-AI, *DeepSeek-V2*, abstract (MLA KV-cache reduction). https://arxiv.org/abs/2405.04434
- Gemma Team, *Gemma 3 Technical Report* (local/global layout, QK-norm). https://arxiv.org/abs/2503.19786
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, section 2.2. https://arxiv.org/abs/2508.10925
- Moonshot AI, *Kimi K2: Open Agentic Intelligence*, abstract and section 2.1. https://arxiv.org/abs/2507.20534
- Moonshot AI, *Kimi Linear*. https://arxiv.org/abs/2510.26692
- DeepSeek-AI, *DeepSeek-V3.2*, section 2.1. https://arxiv.org/abs/2512.02556
- MiniMax, *Why Did MiniMax M2 End Up as a Full Attention Model?* https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, section 3.3. https://arxiv.org/abs/2412.19437
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[01.2 · Reading reports and configs as evidence](lesson-02.md)
