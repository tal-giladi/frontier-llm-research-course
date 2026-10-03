---
id: "03.2"
module: 3
minutes: 35
practice_minutes: 80
prerequisites: ["03.1", "01.2", "02.4"]
objectives:
  - Write the banded causal mask and the rolling KV cache of a sliding-window layer, and show by test that cached decoding agrees with the full forward for one-token and multi-token steps.
  - Compute KV bytes per token for MHA, GQA, MQA, MLA, all-local and interleaved local/global layouts at 32K to 1M tokens, and explain why the per-token figure of a windowed model depends on the context length you quote.
  - Read the local/global layout of Gemma 3 and gpt-oss from their configs and reports, and state what each design keeps exact and what it gives up.
  - Measure decode memory and latency of a local/global model against Baseline-0 with an experiment contract, and separate what the measurement shows from what it cannot.
volatility: concept
sources:
  - title: "Gemma 3 Technical Report (5:1 local/global interleaving, 1024-token local span, RoPE base 10k local / 1M global, KV-cache memory and ablation figures)"
    url: https://arxiv.org/abs/2503.19786
  - title: "gpt-oss-120b & gpt-oss-20b Model Card (section 2.2: alternating banded window of 128 tokens and dense attention)"
    url: https://arxiv.org/abs/2508.10925
  - title: "Mistral 7B (sliding-window attention)"
    url: https://arxiv.org/abs/2310.06825
  - title: "Beltagy, Peters, Cohan — Longformer: The Long-Document Transformer (sliding-window plus global attention)"
    url: https://arxiv.org/abs/2004.05150
  - title: "PyTorch blog — FlexAttention (sliding-window mask_mod and block-sparse masks)"
    url: https://pytorch.org/blog/flexattention/
last_verified: "2026-10-03"
---

# 03.2 · Local/global attention and KV arithmetic

A sliding-window layer lets each token attend only to the last $w$ tokens, so its cache never grows beyond $w$ entries. A model made only of such layers forgets anything older than a few windows; interleaving them with ordinary global layers keeps exact long-range access in a few layers and makes the rest cheap. Gemma 3 uses five local layers of 1,024 tokens per global layer; gpt-oss alternates 128-token banded layers with dense ones. This lesson builds the banded mask and a rolling cache, proves with the correctness suite that they are right, works out the KV arithmetic for every design in the module from 32K to 1M tokens, and measures what a local/global layout does to decode memory and time.

## Why this matters at a frontier lab

Long-context serving is priced in KV bytes: at 128K context, Gemma 3 27B would cache about 62 GiB per sequence if all 62 layers were global; with its 52 local layers it caches about 10.4 GiB (lesson 01.2's calculator, from the config). That is the difference between one sequence and six on an 80 GB GPU. Windowing is cheaper to adopt than MLA — no new projections, the same parameters, the same checkpoint layout — but it changes what the model *can* attend to, so the quality question (does it still use far context?) is real, and Module 4's long-context evaluation is where it gets answered. The implementation is also a common source of silent bugs: off-by-one windows, caches trimmed one step too early, and masks that are right for training but wrong for chunked decoding.

## The idea

### The banded causal mask

A query at position $p$ in a local layer with window $w$ sees keys at positions

$$p - w + 1 \;\le\; j \;\le\; p,$$

$w$ keys including itself. Global layers keep the ordinary causal mask $j \le p$. In matrix form, with $q_i$ and $k_j$ the absolute positions of query $i$ and key $j$, the allowed set is $M_{ij} = [\,k_j \le q_i\,] \wedge [\,k_j > q_i - w\,]$. Using absolute positions, not row and column indices, makes the same mask correct when a decode step has $T$ new queries and $S > T$ keys.

**Conventions differ by one.** Hugging Face's sliding-window masks keep `kv_idx > q_idx - sliding_window` ($w$ keys, our convention, and the one `frontierlab.calc` assumes); the FlexAttention blog's example keeps `q_idx - kv_idx <= SLIDING_WINDOW` ($w + 1$ keys). Before you compare "window 1024" across code bases, check which one you have.

### What a stack of windows can reach

One local layer moves information at most $w - 1$ positions. Stacking $\ell$ local layers lets information travel up to $\ell(w - 1)$ positions, but only through intermediate tokens, a lossy path. A global layer reaches every earlier token in one step. Interleaving (Longformer's local-plus-global idea, now applied per layer rather than per token) keeps a few exact long-range layers and makes the rest cheap. Layer $i$ of a local/global model is global when $(i + 1) \bmod g = 0$: $g = 6$ is Gemma 3's `sliding_window_pattern` (five local, one global), $g = 2$ is gpt-oss's alternation (its `layer_types` start with a sliding layer).

### The rolling cache

A local layer never needs a key older than $w - 1$ positions before its earliest query. After each step it keeps only the last $w$ keys and values; the keys used *in* the step are the stored ones plus the new ones, and the mask removes any that are too old. Trimming *before* attending is wrong for any multi-token step — the prefill, or a chunk of $T > 1$ tokens — because the first queries of the chunk still need keys from before it. The cached-decode test with chunks larger than one catches it; a one-token decode test with a long prefill catches it too, because the prefill is a chunk.

### KV bytes, for any layout

With $e$ cached elements per token and layer ($2Kd$ for GQA, $d_c + d_r$ for MLA), $b$ bytes per element and window $w_\ell$ for local layers:

$$\text{bytes}(S) = b \cdot e \cdot \sum_{\ell=1}^{L} \min(S, w_\ell), \qquad w_\ell = \infty \text{ for global layers}$$

so the average per token is $b \cdot e \cdot (L_{\text{global}} + \sum_{\text{local}} \min(S, w_\ell)/S)$, which falls toward $b \cdot e \cdot L_{\text{global}}$ as $S$ grows. A windowed model has no single "KV bytes per token": always say at which context.

## Worked example

### A window of 2, by hand

Five tokens, $w = 2$. The mask (row = query position, column = key position):

| | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | ✓ | | | | |
| 1 | ✓ | ✓ | | | |
| 2 | | ✓ | ✓ | | |
| 3 | | | ✓ | ✓ | |
| 4 | | | | ✓ | ✓ |

Now decode: prefill positions 0–3, then one chunk with positions 4–5. After the prefill the cache keeps keys 2 and 3 (the last $w = 2$). The chunk attends over $\{2, 3\} \cup \{4, 5\}$; query 4 uses $\{3, 4\}$, query 5 uses $\{4, 5\}$; afterwards the cache keeps $\{4, 5\}$. Had the code trimmed to the last two keys *before* attending, the prefill's queries 0 and 1 would have had no allowed key left at all, while every later one-token step would look fine. So a test that decodes one token at a time after a one-token prefill passes, and any step with more than $w$ new tokens fails — which is why the lab tests chunks of 1, 3 and 7 with $w = 5$ after a 6-token prefill.

### Gemma 3 27B and gpt-oss-120b, from their configs

Gemma 3 27B (`sliding_window` 1024, `sliding_window_pattern` 6, 62 layers, 16 KV heads of width 128): $e = 2 \cdot 16 \cdot 128 = 4{,}096$, 8 KiB per token and layer in BF16. Layers $i$ with $(i + 1) \bmod 6 = 0$ are global: 10 global, 52 local. At $S = 131{,}072$:

$$10 \cdot 131{,}072 \cdot 8 \text{ KiB} + 52 \cdot 1{,}024 \cdot 8 \text{ KiB} = 10.0 \text{ GiB} + 0.41 \text{ GiB} = 10.41 \text{ GiB}$$

against $62 \cdot 131{,}072 \cdot 8$ KiB $= 62$ GiB all-global. Per token that is 83.3 KiB at 128K, 93.0 KiB at 32K, 80.4 KiB at 1M (the lab's table) — the same model, three answers.

gpt-oss-120b (36 layers alternating, window 128, 8 KV heads of width 64): $e = 1{,}024$, 2 KiB per token and layer. At 128K: $18 \cdot 131{,}072 \cdot 2 \text{ KiB} + 18 \cdot 128 \cdot 2 \text{ KiB} = 4.5 \text{ GiB} + 4.5 \text{ MiB}$. The 18 banded layers cost almost nothing; the 18 dense layers are the whole cache.

### Baseline-0 at 32K to 1M

All with Baseline-0's 12 layers and 12 query heads of width 64, BF16, KiB per token (measured by formula, `kv_table.py`):

| Design | 32K | 128K | 1M | GiB per 128K sequence |
|---|---|---|---|---|
| MHA ($K = 12$) | 36.00 | 36.00 | 36.00 | 4.50 |
| GQA $K = 4$ (Baseline-0) | 12.00 | 12.00 | 12.00 | 1.50 |
| GQA $K = 2$ | 6.00 | 6.00 | 6.00 | 0.75 |
| MQA ($K = 1$) | 3.00 | 3.00 | 3.00 | 0.38 |
| MLA $d_c = 256$, $d_r = 32$ | 6.75 | 6.75 | 6.75 | 0.84 |
| all local, $w = 1024$ | 0.38 | 0.09 | 0.01 | 0.01 |
| 5 local : 1 global, $w = 1024$ | 2.31 | 2.08 | 2.01 | 0.26 |
| 1 local : 1 global, $w = 128$ | 6.02 | 6.01 | 6.00 | 0.75 |

The two levers multiply: a 5:1 layout on top of MLA would cache $2 \cdot 288 \cdot 2$ bytes per token plus ten small windows, about 1.2 KiB per token at 1M. Kimi Linear and Qwen3-Next combine a cheap layer type with a few full layers in the same spirit (Module 5).

## Shapes and cost

| Tensor (one local layer, Baseline-0 width) | Shape | dtype / device |
|---|---|---|
| queries | (B, 12, T, 64) | bf16 on GPU (main path), fp32 on CPU |
| cached keys / values | (B, 4, min(S, w), 64) each | same; `LayerCache["k"]`, `["v"]` |
| attended keys this step | (B, 4, min(S, w) + T, 64) | stored plus new, before trimming |
| mask | (T, min(S, w) + T) bool | from absolute positions |

**Training FLOPs.** A query in a local layer of a length-$T$ sequence sees on average about $w - w^2/(2T)$ keys instead of $T/2$ (lesson 01.2). For Baseline-0 at $T = 1024$ with the project's window $w = 256$: $256 - 32 = 224$ keys instead of 512. Ten of twelve layers local: training FLOPs per token fall from 788.1M to 761.6M, 3.4% — at this short context attention is a small part of the cost, so equal-FLOPs runs of the local/global arm get only 3.5% more steps (9,831 instead of 9,500). At $T = 32{,}768$ the attention term would dominate and the saving would be large.

**Decode.** A local layer reads $\min(S, w)$ cached tokens per step instead of $S$. A roofline lower bound for one decode step at $S = 32{,}768$, BF16, H100 SXM, batch 1 (weights plus cache bytes divided by 3.35 TB/s): Baseline-0 193 µs, local/global (5:1, $w = 256$) 94 µs. PROJECTED, pending the Module 3 pilot; at batch 1 the weights (244 MB) are a large share of the bytes, so the gain grows with batch size, where the cache dominates.

**Kernels.** `F.scaled_dot_product_attention` with an arbitrary boolean mask computes every masked score and cannot skip blocks. Production code uses kernels that understand windows (FlashAttention's window option, PyTorch FlexAttention block masks) and skip fully masked blocks; our implementation is for correctness and memory, and its training speed says nothing about theirs.

## Build it

`labs/common/frontierlab/attention/sliding.py` registers two kinds, both Baseline-0's GQA with the same projections and parameter count:

- `"sliding"` — every layer local, `extra["window"]`;
- `"local_global"` — takes `layer_idx`; layer $i$ is global when `(i + 1) % extra["global_every"] == 0` (or per `extra["layer_types"]`); optional `local_rope_theta` / `global_rope_theta` (Gemma 3 uses 10k and 1M) and `sinks` (gpt-oss, lesson 03.3).

The cache update is the one line that matters:

```python
if "k" in cache:
    k = torch.cat((cache["k"], k), dim=2); v = torch.cat((cache["v"], v), dim=2)
    k_pos = torch.cat((cache["pos"], positions))
keep = slice(None) if self.window is None else slice(-self.window, None)
cache["k"], cache["v"], cache["pos"] = k[:, :, keep], v[:, :, keep], k_pos[keep]
return k, v, k_pos                      # attend over ALL of these; the mask drops the old ones
```

Attention itself is `frontierlab.attention.ops.attend(q, k, v, q_pos, k_pos, window=...)`, which builds the banded mask from absolute positions. The correctness suite on toy models passes for `sliding`, `local_global` and `local_global` with sinks: op-level float64 gradient check, causal check, and cached decode with chunks of 1, 3 and 7 tokens across a window of 5 (all differences below $10^{-12}$). Two extra tests: a window of at least $T$ reproduces Baseline-0's logits exactly, and after decoding $S$ tokens every local layer holds exactly $\min(S, w)$ positions. A planted trim-before-attend bug fails the cache test with a logit difference of 0.83 even at one token per step, because the 6-token prefill is itself a chunk.

## What the evidence says

- **ESTABLISHED.** Sliding-window attention (Mistral 7B, Longformer) and interleaved local/global layers are used in Gemma 2 and 3 and in gpt-oss; the layouts are PUBLICLY DOCUMENTED in their reports and configs (Gemma 3: "a pattern of 5 local layers for every global layer", local span 1,024; gpt-oss model card section 2.2: "alternate between banded window and fully dense patterns, where the bandwidth is 128 tokens").
- **Quality evidence is the labs' own, at their scales.** Gemma 3's ablations on 2B models report "minimal impact on perplexity" when changing the local:global ratio (its Figure 3) and that the window "can be reduced significantly without impacting perplexity" (Figure 4); for memory, with a 2B model and a 32K prefill, the global-only configuration "results in a memory overhead of 60%", which local layers with 1,024-token windows reduce to "less than 15%" (Figure 5 and its discussion). Perplexity is not long-range use: whether the model still retrieves and reasons over far context is Module 4's question, and a short-context perplexity match is not evidence for it.
- **MODEL-SPECIFIC choices.** The exact ratio and window (5:1 and 1,024 for Gemma 3; 1:1 and 128 for gpt-oss) and Gemma's separate RoPE bases are single labs' settings.
- **Course-scale hypothesis.** At $T = 1024$ training, a window of 256 should cost little held-out loss on Eval v0 (most next-token information is local); that is a hypothesis for the project, and Eval v0 cannot show what is lost beyond the window.

## Lab

**Folder:** [`labs/module-03/lesson-02/`](../../labs/module-03/) · **Time:** about 80 minutes · **Pass check:** `pytest labs/module-03/lesson-02` passes; `kv_table.py` part 3 prints OK for every kind; your notes state the decision the contract's rule gives.

### Experiment contract

- **Question:** at decode contexts from 1K to 16K tokens (free CPU) or 8K to 32K (main path), how much cache memory and decode time does a local/global layout (5:1 where depth allows, else 3:1; window $T/4$ of the training length) or an all-local layout save against Baseline-0? Decision informed: whether the local/global arm's decode advantage in the project needs its own GPU measurement or can be projected from the formula.
- **Hypotheses and status:** (1) cache bytes equal the formula exactly at every context — established; (2) the cache of the local/global model grows with slope $L_{\text{global}}/L$ of Baseline-0's — established arithmetic; (3) decode time falls relative to Baseline-0 as the context grows — expected where reading the cache dominates the step; may not appear on a CPU at toy size.
- **Baseline:** arm `b0`, random weights, seed 0.
- **Changed variable:** the mask and cache policy only (`local-global`, `sliding`); projections and parameters are identical. **Controlled:** preset, vocabulary, batch 1, dtype, threads, prefill tokens and chunk, warm-up and rounds.
- **Comparison axis:** equal model (same parameters, same FLOPs per token except attention) at equal context.
- **Budget:** free CPU about 2 minutes; main path about 10 GPU-minutes.
- **Metrics and decision rule:** `Cache.nbytes` against the formula; median decode-step time with a 95% interval over 30 interleaved rounds; paired speed-up against `b0`. Rule, stated now: if measured bytes equal the formula at every context, the project may project main-path decode memory from the formula (labelled PROJECTED); decode *time* is reported as measured only on the hardware it was measured on.
- **Correctness checks:** `pytest labs/module-03/lesson-02` (your mask and cache against frontierlab's, causal check, cached decode with chunks 1, 3, 7) passes before any timing.
- **Fallback evidence:** the Module 3 pilot's GPU decode traces, labelled as provided.
- **Limits:** random weights, batch 1, one window; no quality measurement here; the CPU's time profile does not transfer to GPUs.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB, about 10 GPU-minutes. Not run in this build; part of the Module 3 pilot | `python labs/module-03/decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --train-seq 1024 --arms b0 local-global sliding --contexts 8192 16384 32768 --rounds 30 --out runs/l32/decode.json` |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | the same with `--dtype fp32 --preset pilot-10m --train-seq 512 --contexts 4096 8192 16384` |
| Free CPU | laptop; `kv_table.py` about 10 s, `decode_compare.py` about 1 minute (measured 2026-10-03, 16 threads) | the steps below as written |

### Steps

1. **Implement** `window_mask`, `update_window_cache`, `layer_pattern` and `kv_bytes` in `lab.py`. The provided `LabLocalGlobal` attention uses your mask and cache update; `pytest labs/module-03/lesson-02` runs it through the correctness suite.
2. **The KV table:** `python labs/module-03/lesson-02/kv_table.py`. Check two rows of part 1 and one of part 2 by hand.
3. **Measure:**

   ```bash
   python labs/module-03/decode_compare.py --arms b0 local-global sliding --threads 4 --contexts 1024 4096 16384 --rounds 30 --out runs/l32/decode.json
   ```

4. **Report:** the measured cache bytes against the formula; the decode-step medians with intervals; the decision the rule gives; and one paragraph on what this experiment cannot tell you about the local/global model (think about what a 32-token window forgets, and which evaluation would show it).

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, fp32, toy preset: 4 layers, window 32, layer 3 global; `--threads 4`, 30 rounds; another build job was using the CPU): cache bytes equal the formula at every context. At $S = 16{,}384$ the caches hold 32.5 MiB (Baseline-0), 8.17 MiB (local/global) and 65 KiB (all local), each including the int64 position bookkeeping. Decode-step medians with 95% intervals and paired speed-ups against Baseline-0: at 1K, local/global 0.95 [0.80, 1.11] and all-local 1.00 [0.86, 1.85] — no difference shown; at 16K, Baseline-0 10.6 ms [9.2, 13.4], local/global 6.1 ms [5.1, 6.9], speed-up 1.94 [1.67, 2.26], all-local 4.7 ms [4.2, 5.8], 2.25 [1.95, 2.60]. Decision by the rule: the formula matched at every context, so the project projects main-path decode memory from it (labelled PROJECTED); the speed-ups hold for this CPU only. Note that lesson 03.1's run measured Baseline-0 at 16K at 14.4 ms on the same machine minutes earlier: absolute times move between runs on a shared machine, which is why only the paired ratios within one interleaved run are compared.

<details>
<summary>Hint for step 1</summary>

`window_mask` is two comparisons combined with `&`, on `k_pos[None, :]` and `q_pos[:, None]`. In `update_window_cache`, concatenate first, store the trimmed copy in the cache, and return the untrimmed tensors.

</details>

<details>
<summary>Hint for step 4</summary>

Eval v0's held-out loss scores next-token prediction on 128- to 1,024-token windows; most of that signal is within a few dozen tokens. A model that cannot see beyond its window except through three global layers may lose little there and still fail a task whose answer is 10K tokens back. Module 4 builds that evaluation.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-03/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-03/lesson-02`.

</details>

## Common mistakes

- **Trimming the cache before attending.** Right for one-token steps after a one-token prefill, wrong for every chunk; test chunked decoding with chunks larger than the window.
- **An off-by-one window.** $w$ keys including the query, or $w$ before it? Check the convention of every library you compare against.
- **Quoting one "KV bytes per token" for a windowed model.** It depends on the context; quote it at the context you serve.
- **Counting local layers' cache as zero.** It is small, not zero: $w$ tokens per local layer, which matters at short contexts and large batches.
- **Reading a perplexity match as long-context parity.** Short-window losses barely depend on far context; use a long-context evaluation (Module 4).
- **Timing a boolean-mask SDPA and calling it the speed of sliding-window attention.** Production kernels skip masked blocks; ours computes them.

## References

- Gemma Team, *Gemma 3 Technical Report*: architecture (5:1 interleaving, local span 1,024, RoPE bases), ablations (Figures 3–5). https://arxiv.org/abs/2503.19786
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, section 2.2. https://arxiv.org/abs/2508.10925
- A. Q. Jiang et al., *Mistral 7B*. https://arxiv.org/abs/2310.06825
- I. Beltagy, M. Peters, A. Cohan, *Longformer*. https://arxiv.org/abs/2004.05150
- PyTorch blog, *FlexAttention* (sliding-window mask example). https://pytorch.org/blog/flexattention/
- Shared code: `labs/common/frontierlab/attention/sliding.py`, `ops.py`, `accounting.py`; config snapshots in `labs/common/frontierlab/calc/snapshots/`.

## Next

[03.3 · Logit control, sinks and gating](lesson-03.md)
