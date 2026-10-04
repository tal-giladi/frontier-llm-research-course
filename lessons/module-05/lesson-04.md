---
id: "05.4"
module: 5
minutes: 25
practice_minutes: 40
prerequisites: ["05.2", "05.3"]
objectives:
  - Describe DeepSeek-V4's Compressed Sparse Attention and Heavily Compressed Attention using only what section 2.3 of its report documents, and list the hyperparameters it does not state.
  - Implement overlapping (CSA-style) and non-overlapping (HCA-style) KV compression and prove by test which entries a query may use without seeing the future.
  - Compute KV memory and per-query attention work of compressed layouts as functions of the compression rates, top-k and window, and explain why the reported 10% KV figure cannot be reproduced without the unstated values.
volatility: implementation
sources:
  - title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (section 2.3: 2.3.1 CSA, 2.3.2 HCA, 2.3.3 other details, 2.3.4 efficiency discussion; abstract)"
    url: https://arxiv.org/abs/2606.19348
  - title: "DeepSeek-V4-Pro model card"
    url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
  - title: "DeepSeek-V3.2 (section 2.1: DeepSeek Sparse Attention, the indexer CSA reuses)"
    url: https://arxiv.org/abs/2512.02556
last_verified: "2026-10-04"
---

# 05.4 · Compressed sparse attention (DeepSeek-V4 CSA/HCA)

Extension: DeepSeek-V4 goes one step beyond DSA. Instead of keeping one KV entry per token and selecting among them, it first compresses groups of tokens into single entries, then either selects the top-k compressed entries with a lightning indexer (Compressed Sparse Attention, CSA) or attends densely over a much more heavily compressed sequence (Heavily Compressed Attention, HCA), and interleaves the two kinds of layer. This lesson describes the design strictly as section 2.3 of the report documents it, builds both compression rules and their causality test, and works out the cost model as a function of the hyperparameters the report does not publish.

## Why this matters at a frontier lab

DeepSeek-V4 reports that at a 1M-token context, V4-Pro needs "only 27% of single-token inference FLOPs" and "10% of KV cache compared with DeepSeek-V3.2" (section 2.3.4; estimated figures, FLOPs "measured in equivalent FP8 FLOPs"). DSA alone (05.2) cannot reduce the cache, because every token stays selectable. Compression can, and this is the first published frontier design that combines compression, learned selection and a local window in one stack. A research engineer reading it has to separate three things: what the report states, what it implies, and what it leaves out, so that a reproduction or a cost estimate does not silently invent the missing numbers.

## The idea

### What section 2.3 documents

**CSA (2.3.1).** CSA compresses "the KV cache of every m tokens into one entry". Two compression paths produce candidate entries $C^a$ and $C^b$ with compression weights $Z^a$ and $Z^b$; entry $i$ combines path $a$ over its own block of $m$ tokens with path $b$ over the previous block,

$$\text{entry}_i = \sum_{j = mi}^{m(i+1)-1} S^a_j \odot C^a_j \;+\; \sum_{j = m(i-1)}^{mi-1} S^b_j \odot C^b_j,$$

with the softmax weights $S$ normalised over the $2m$ terms, so consecutive entries overlap and the sequence is compressed to $1/m$ of its length. A DSA lightning indexer (05.2), computed from a low-rank compressed query, scores each compressed entry and the top-$k$ are selected. An additional sliding-window branch keeps the $n_{\text{win}}$ most recent tokens uncompressed "for local fine-grained modeling".

**HCA (2.3.2).** HCA compresses every $m'$ entries into one, with $m' \gg m$ and no overlap, and "does not employ sparse attention": it attends densely over all compressed entries.

**Shared details (2.3.3).** Both use shared-KV multi-query attention (each compressed entry is "both attention key and value"); RMSNorm on each query head and on the single compressed KV head before the core attention; RoPE on the last 64 dimensions; a learnable sink logit per head added to the softmax denominator (as gpt-oss, lesson 03.3); a grouped output projection (heads split into groups, each projected through a bottleneck). The model interleaves CSA and HCA layers.

**Precision (2.3.4).** The KV cache uses "BF16 precision ... for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions"; "attention computation within the lightning indexer is performed in FP4 precision".

### What section 2.3 does not state

Not verified in this build, so not used as facts anywhere in the course: the values of $m$, $m'$, $k$ and $n_{\text{win}}$; the indexer's head count and width; the attention head count and head dimension; the exact CSA/HCA interleaving pattern and layer counts. Model-level facts from the abstract and the model card (two sizes, 1.6T total / 49B activated and 284B / 13B parameters, and a 1M-token context) do not depend on these.

### Compression and causality

An entry is a softmax-weighted average, per channel, of the entries of its block(s). It can exist only once every token of its block has been seen. A query at position $t$ (0-based) may therefore use the $\lfloor (t+1)/m \rfloor$ entries of complete blocks; the tokens of the incomplete current block — and, for good local modelling, a few blocks more — must come from somewhere else. In CSA that is the uncompressed sliding window. Getting this wrong (letting a query see the entry of the block it is in) leaks the future during training, the compressed analogue of 01.1's mask bug.

## Worked example

### Two compressions by hand

Tokens with one channel, $c = (1, 3, 10, 20, 99)$, $m = 2$. With equal weights ($z = 0$) HCA-style compression gives $(\tfrac{1+3}{2}, \tfrac{10+20}{2}) = (2, 15)$; token 4 is in an incomplete block and has no entry yet. With $z = (0, \ln 3, \dots)$ the first entry is $\tfrac{1}{4} \cdot 1 + \tfrac{3}{4} \cdot 3 = 2.5$. In the overlapping CSA-style rule, entry 2 is a softmax over the four logits of block 2 (path a) and block 1 (path b): changing a path-b value of block 1 changes entry 2 and nothing else (the lab's test).

### Cost as a function of the unstated values

Per layer, after $S$ tokens, with entry size $e$ bytes:

$$\text{KV}_{\text{CSA}} = e\,(\lfloor S/m \rfloor + \min(S, n_{\text{win}})), \qquad \text{KV}_{\text{HCA}} = e\,\lfloor S/m' \rfloor, \qquad \text{KV}_{\text{dense}} = e\,S.$$

Entries attended per decode query: CSA $\min(k, S/m) + \min(S, n_{\text{win}})$; HCA $S/m'$; and the CSA indexer scores $S/m$ compressed keys, so its $O(L^2)$ term shrinks by $m$ but remains. With ILLUSTRATIVE values (not V4's) $m = 4$, $m' = 128$, $k = 512$, $n_{\text{win}} = 128$ and half the layers of each kind, the cache at 1M tokens is 0.129 of a dense one with the same entry size, and FP8 storage of all but 64 of 512 dimensions multiplies that by 0.56 (`v4_costs.py`). Change $m$ to 8 and the ratio moves to about 0.07 (0.065 with $m' = 256$). The reported 10% compares V4 with V3.2, two different models whose entry sizes and layer counts also differ, so no choice of $m$ and $m'$ can be "checked" against it with the information in section 2.3: that is the honest statement to make in a review.

## Shapes and cost

| Tensor (one layer, B = 1) | Shape | Notes |
|---|---|---|
| per-token candidates $C^a, C^b$ and weights $Z^a, Z^b$ | (1, S, d) each | d = the shared-KV entry width (not stated) |
| CSA compressed entries | (1, ⌊S/m⌋, d) | cached; FP8 except the 64 RoPE dims (BF16) |
| CSA recent window | (1, min(S, n_win), d) | uncompressed |
| CSA indexer keys | (1, ⌊S/m⌋, d^I) | indexer attention in FP4 |
| HCA compressed entries | (1, ⌊S/m'⌋, d) | cached; attended densely |

Compression itself costs a softmax and a weighted sum over $2m$ (CSA) or $m'$ (HCA) entries per output entry, about $4d$ FLOPs per input token per path: negligible next to attention, but it is a new memory-bound kernel on the prefill path.

## Build it

`labs/common/frontierlab/attention/compressed.py` holds a didactic reference: `compress_hca(c, z, m)`, `compress_csa(ca, za, cb, zb, m)`, `usable_entries(t, m)`, and the cost functions `kv_entries` and `attended_entries`; `accounting.compressed_kv_entries` gives the same count for the accounting tables. It is not an attention kind: building a full CSA/HCA layer would need the unstated hyperparameters and design choices (how queries are compressed for the indexer, the grouped projection's sizes), and inventing them would teach a guess as if it were V4. Tests check the hand example, the overlap (a change in block $i-1$'s path-$b$ values changes entry $i$ only), and causality (changing any token after $t$ never changes the $\lfloor (t+1)/m \rfloor$ usable entries).

## What the evidence says

- **CSA/HCA: MODEL-SPECIFIC.** One lab, one model family, PUBLICLY DOCUMENTED at the level of section 2.3; hyperparameters not stated there; no independent replication.
- **Efficiency figures** (27% of single-token FLOPs, 10% of KV cache against V3.2 at 1M): estimated by the lab, in FP8-equivalent FLOPs (company claim).
- **Ingredients with independent support:** token compression (NSA's compression branch, 05.2), learned top-k selection (DSA), sliding windows and sinks (Module 3), low-precision KV (Module 8). The combination and its quality at 1M context are the lab's own evidence.
- Treat any third-party "V4 config" values for $m$, $m'$, $k$ or $n_{\text{win}}$ as unverified until checked against the released model files and code at a stated commit.

## Lab

> [!NOTE]
> This extension lab compares nothing experimentally: it implements two compression rules, tests their causality, and computes costs as functions of hyperparameters. It has no experiment contract.

**Folder:** [`labs/module-05/lesson-04/`](../../labs/module-05/) · **Time:** about 40 minutes · **Pass check:** `pytest labs/module-05/lesson-04` passes; your notes contain the table below for two settings of $m$ and the sentence on what cannot be checked.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | none needed: arithmetic and small tensors | as free CPU |
| Free GPU | not needed | as free CPU |
| Free CPU | laptop; tests about 2 s, `v4_costs.py` under 1 s (measured 2026-10-04) | the steps below |

1. **Implement** `compress_hca`, `compress_csa`, `usable_entries` and `layout_kv_bytes` in `lab.py`; run `pytest labs/module-05/lesson-04`.
2. **Cost tables** for two settings:

   ```bash
   python labs/module-05/lesson-04/v4_costs.py
   python labs/module-05/lesson-04/v4_costs.py --m 8 --m2 256
   ```

3. **Write** (a few sentences): which hyperparameter moves the 1M-token cache most; how many compressed keys the CSA indexer scores per query at 1M for each $m$, and what that means for its $O(L^2)$ term; and why the report's 10% cannot be reproduced from section 2.3.

<details>
<summary>Hint for TODO 2</summary>

Reshape each input to (B, n, m, d) blocks. Build the previous-block tensors by shifting by one block and filling block −1 with logits of `-inf` (so their softmax weight is 0) and values of 0. Concatenate along the block axis to (B, n, 2m, d), softmax over that axis, multiply and sum.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-05/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-05/lesson-04`.

</details>

## Common mistakes

- **Using the current block's entry.** It summarises tokens after the query; training then sees the future.
- **Forgetting the window.** Without the uncompressed recent tokens, a CSA query has no access to its own block or anything not yet compressed.
- **Quoting V4 hyperparameters from blogs.** Section 2.3 does not state them; cite the released files at a commit, or say "not stated".
- **Comparing the 10% figure with a compression rate.** It is a model-to-model ratio, not a property of one layer.

## References

- DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, 2026: section 2.3 (2.3.1–2.3.4), abstract. https://arxiv.org/abs/2606.19348
- DeepSeek-AI, *DeepSeek-V4-Pro* model card. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-AI, *DeepSeek-V3.2*, section 2.1. https://arxiv.org/abs/2512.02556
- Shared code: `labs/common/frontierlab/attention/compressed.py`.

## Next

The [Module 5 project](../../projects/module-05-subquadratic-memo.md): a sub-quadratic decision memo. Module 6 asks which other block changes earn their complexity.
