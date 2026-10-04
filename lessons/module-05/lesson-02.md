---
id: "05.2"
module: 5
minutes: 40
practice_minutes: 80
prerequisites: ["05.1", "03.1", "01.4"]
objectives:
  - Describe NSA's three branches and DeepSeek Sparse Attention's lightning indexer and top-k selection, each with the equations and hyperparameters their reports state.
  - Implement the indexer, top-k selection and the indexer's KL objective, and prove by test that DSA equals dense attention when k covers the context and that the indexer and the main model receive gradients only from their own losses.
  - Run the documented two-stage conversion of a dense model (dense warm-up of the indexer, then sparse training) under an experiment contract and measure how much attention mass the indexer's selection keeps against the oracle and a sliding window.
  - Compute the honest cost model, O(L·k) for the selected attention plus O(L²) for the indexer, for Baseline-0's shape and for DeepSeek-V3.2's config, and say where the indexer dominates.
volatility: concept
sources:
  - title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (section 2.1 DeepSeek Sparse Attention; 2.1.1 continued pre-training; 2.2 parity evaluation; Figure 3 inference cost)"
    url: https://arxiv.org/abs/2512.02556
  - title: "Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (section 3, Eq. 5; compression, selection, sliding window)"
    url: https://arxiv.org/abs/2502.11089
  - title: "DeepSeek-V3.2 config.json (index_n_heads, index_head_dim, index_topk)"
    url: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
last_verified: "2026-10-04"
---

# 05.2 · Learned sparse attention

Linear attention compresses the past into a fixed state; sparse attention keeps every token but lets each query read only a few of them. The question is which few. This lesson covers the two designs DeepSeek published: Native Sparse Attention (NSA), which combines a compressed view of the past, a set of selected blocks and a sliding window, and DeepSeek Sparse Attention (DSA) in DeepSeek-V3.2, where a small "lightning indexer" scores every earlier token and the main attention runs over the top 2,048. You will implement DSA, train its indexer the way the V3.2 report describes, check that the selection is worth having by comparing it with a plain sliding window, and write down the cost model honestly: the attention gets cheaper, the indexer does not.

## Why this matters at a frontier lab

DeepSeek-V3.2's report states that DSA reduces "core attention complexity from O(L²) to O(Lk)", that the lightning indexer "still has a complexity of O(L²)", and that it found no "substantial performance degradation compared with DeepSeek-V3.1-Terminus, on both short- and long-context tasks" (sections 2.1 and 2.2). That is an attractive trade: an existing dense model is converted by continued training, not retrained, and keeps its full KV cache, so nothing is forgotten. But whether it pays depends on numbers the headline hides: how big the indexer is relative to the attention, how well a few billion tokens of indexer training recover the dense model's attention, and what the gather of 2,048 scattered entries per query costs in memory traffic. A research engineer has to be able to put each of these on the table before anyone books the GPUs.

## The idea

### Native Sparse Attention: three views of the past

NSA (arXiv 2502.11089, section 3) computes three attentions per query and mixes them with learned gates:

$$o_t^* = \sum_{c \in \{\text{cmp}, \text{slc}, \text{win}\}} g_t^c \cdot \mathrm{Attn}(q_t, \tilde K_t^c, \tilde V_t^c),$$

with $g_t^c \in [0, 1]$ "derived from input features via an MLP and sigmoid activation" (Eq. 5). The branches:

- **Compression.** Blocks of $l$ consecutive keys, taken with stride $d$, are mapped by a learnable MLP $\varphi$ with intra-block position encoding to one compressed key (and likewise values): a coarse summary of the whole past.
- **Selection.** The compression branch's attention scores $p_t^{\text{cmp}} = \mathrm{Softmax}(q_t^\top \tilde K_t^{\text{cmp}})$ are reused as importance scores for blocks of size $l'$, summed over the query heads of a GQA group so the whole group selects the same blocks, and the top-$n$ blocks are attended to at full resolution.
- **Sliding window** over the last $w$ tokens.

The branches have "independent keys and values" to prevent shortcut learning. Reported settings: $l = 32$, $d = 16$, $l' = 64$, $n = 16$ (including 1 fixed initial and 2 local blocks), $w = 512$. Selection is per *block* and shared across a GQA group because that is what a kernel can load efficiently ("group-centric data loading"). NSA is trained this way from the start ("natively trainable").

### DeepSeek Sparse Attention: score every token, attend to the best k

DSA (V3.2 report, section 2.1) is simpler and token-level. A lightning indexer with $H^I$ heads computes, for query token $t$ and every preceding token $s$,

$$I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}\!\left(q^I_{t,j} \cdot k^I_s\right),$$

where $q^I_{t,j} \in \mathbb{R}^{d^I}$ and $w^I_{t,j} \in \mathbb{R}$ come from the query token's hidden state $h_t$ and $k^I_s \in \mathbb{R}^{d^I}$ from $h_s$ — one indexer key per token, shared by the indexer heads. ReLU is chosen "for throughput consideration", and because the indexer "has a small number of heads and can be implemented in FP8" it is cheap per pair. Then the main attention runs only over the key-value entries with the top-$k$ index scores:

$$u_t = \mathrm{Attn}\big(h_t, \{c_s \mid I_{t,s} \in \text{Top-}k(I_{t,:})\}\big).$$

V3.2 uses $k = 2{,}048$ and instantiates DSA on MLA "in the MQA mode", so each cached latent is shared by all 128 query heads and one selection serves every head. The report does not state $H^I$ or $d^I$; the released `config.json` has `index_n_heads` 64, `index_head_dim` 128 and `index_topk` 2048 (checked 2026-10-04).

### How the indexer is trained (V3.2, section 2.1.1)

Selection is a discrete top-k: the language-modelling loss cannot teach the indexer anything through it. The indexer gets its own objective. For query $t$, the main attention scores are summed over all heads and "L1-normalized along the sequence dimension" to give a target distribution $p_{t,:}$, and the indexer is trained to match it:

1. **Dense warm-up.** Dense attention is kept and "all model parameters except for the lightning indexer" are frozen. Loss: $\mathcal{L}^I = \sum_t D_{\mathrm{KL}}\big(p_{t,:} \,\|\, \mathrm{Softmax}(I_{t,:})\big)$. Reported: learning rate $10^{-3}$, 1,000 steps of 16 sequences of 128K tokens, 2.1B tokens.
2. **Sparse training.** Selection is switched on and everything trains, with two separations: the KL is computed over the selected set $S_t$ only, $\mathcal{L}^I = \sum_t D_{\mathrm{KL}}\big(p_{t,S_t} \,\|\, \mathrm{Softmax}(I_{t,S_t})\big)$, and "we detach the indexer input from the computational graph for separate optimization. The training signal of the indexer is from only $\mathcal{L}^I$, while the optimization of the main model is according to only the language modeling loss." Reported: learning rate $7.3 \times 10^{-6}$, $k = 2{,}048$, 15,000 steps of 480 sequences of 128K tokens, 943.7B tokens.

Both stages start from "a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K". The warm-up costs 0.2% of the sparse stage's tokens: its job is only to give the indexer a sensible starting point before selection depends on it.

### The honest cost model

Per query at context $L$, with $a$ the main attention's FLOPs per (query, key) pair and $c = H^I(2d^I + 2)$ the indexer's:

$$\text{dense: } a \cdot L, \qquad \text{DSA: } \underbrace{c \cdot L}_{\text{indexer, all keys}} + \underbrace{a \cdot \min(k, L)}_{\text{selected attention}} + \text{top-}k + \text{gather}.$$

The selected attention is $O(Lk)$ over a sequence; the indexer is still $O(L^2)$, only with the smaller constant $c$. So DSA wins when $c \ll a$ and $L \gg k$, and the indexer becomes the larger term once $c L > a k$, that is at $L > a k / c$. Two terms are not FLOPs at all: top-$k$ over $L$ scores per query (a memory-bound sort-like operation), and the gather of $k$ scattered cache entries per query, which turns one contiguous read into $k$ small ones unless the kernel is designed for it. Lesson 05.3 measures all four separately.

## Worked example

### An indexer score, a selection and a KL, by hand

One query, two indexer heads with $q^I_1 = (1, 0)$, $q^I_2 = (0, 1)$, weights $w = (0.5, 2)$; two keys $k^I_0 = (2, -1)$, $k^I_1 = (-1, 3)$:

$$I_0 = 0.5\,\mathrm{ReLU}(2) + 2\,\mathrm{ReLU}(-1) = 1.0, \qquad I_1 = 0.5\,\mathrm{ReLU}(-1) + 2\,\mathrm{ReLU}(3) = 6.0.$$

With $k = 1$ the query attends to key 1 only. Selection with the same scores $(3, 1, 2, 9)$ for queries at positions 0–3 and $k = 2$ picks $\{0\}$, $\{0, 1\}$, $\{0, 2\}$ and $\{0, 3\}$: a row with fewer than $k$ allowed keys takes all of them, and no future key is ever selected.

The target: two heads with attention rows $(0.5, 0.5, 0)$ and $(0, 0.5, 0.5)$ sum to $(0.5, 1, 0.5)$, L1-normalised $p = (0.25, 0.5, 0.25)$. A uniform indexer ($I = 0$) gives $\mathrm{Softmax}(I) = (1/3, 1/3, 1/3)$ and

$$D_{\mathrm{KL}} = 0.25 \ln\tfrac{0.25}{1/3} + 0.5 \ln\tfrac{0.5}{1/3} + 0.25 \ln\tfrac{0.25}{1/3} = -0.0719 + 0.2027 - 0.0719 = 0.0589 \text{ nats}.$$

If the selected set is keys $\{1, 2\}$, the sparse-stage KL uses $p$ renormalised over them, $(2/3, 1/3)$, against a softmax over two scores.

### The cost model for Baseline-0 and for DeepSeek-V3.2

**Baseline-0 shape** (12 heads of 64, so $a = 4 \cdot 12 \cdot 64 = 3{,}072$), with the course's small indexer ($H^I = 4$, $d^I = 32$, $c = 264$) and $k = 2{,}048$: in training, dense mixing costs $a L/2$ per token and DSA $c L/2 + a(k - k^2/2L)$; they are equal at $L = 2{,}897$ (`accounting.dsa_crossover_length`). Below about 2.9K tokens DSA does more arithmetic than dense attention — before counting top-$k$ and gather.

**DeepSeek-V3.2 shape** (INFERENCE from `config.json`, absorbed MLA in MQA mode: 128 heads, scores over $d_c + d_r = 576$ latent dimensions and values over $d_c = 512$): $a = 2 \cdot 128 \cdot 576 + 2 \cdot 128 \cdot 512 = 278{,}528$ FLOPs per pair; indexer $c = 64 \cdot (2 \cdot 128 + 2) = 16{,}512$, so $c/a = 5.9\%$. One decode step at $S = 131{,}072$:

| Term | FLOPs | Share of dense |
|---|---|---|
| dense attention, $a S$ | $3.65 \times 10^{10}$ | 100% |
| indexer, $c S$ | $2.16 \times 10^{9}$ | 5.9% |
| selected attention, $a k$ | $5.70 \times 10^{8}$ | 1.6% |

The indexer is the larger part of the remaining arithmetic above $S = a k / c = 34{,}500$ tokens, and at 128K it is almost four times the selected attention. That is why the report stresses FP8 and few heads for the indexer: they lower its time per FLOP, not its $O(L^2)$ growth.

## Shapes and cost

| Tensor (one DSA layer, course defaults) | Shape | dtype / device |
|---|---|---|
| main q / k / v | (B, H, T, d) / (B, KV, S, d) | bf16 on GPU, fp32 on CPU |
| indexer queries $q^I$ | (B, $H^I$, T, $d^I$) | same (V3.2: FP8) |
| indexer keys $k^I$, cached in `LayerCache["idx_k"]` | (B, S, $d^I$) | same |
| indexer weights $w^I$ | (B, T, $H^I$) | same |
| scores $I$ | (B, T, S) per query block | fp32 |
| selection | (B, T, k) indices or (B, T, S) bool | int64 / bool |
| gathered keys and values (gather path) | (B, KV, block, k, d) | same as k, v |

**Memory.** DSA keeps the whole KV cache plus $d^I$ indexer-key elements per token and layer: Baseline-0 with $d^I = 32$ caches 1,088 bytes per token and layer instead of 1,024 in BF16 (+6%). DSA saves compute and bandwidth per step, never cache size; if the cache is the constraint, Module 3's levers or 05.1's hybrids apply.

**Training FLOPs.** `accounting.m05_flops_per_token(cfg, T)` counts the indexer over all $T/2$ average keys and the selected attention over $k - k^2/(2T)$; at $T = 256$ with $k = 32$ the toy DSA model does 2.6% more training FLOPs per token than toy Baseline-0 (its projections and the indexer outweigh the saved attention). Our default mask path computes every score anyway, so its real cost is the dense one plus the indexer (`dsa_impl="mask"`).

## Build it

`labs/common/frontierlab/attention/dsa.py` registers `"dsa"`: Baseline-0's GQA (same projections, RoPE, QK-norm) plus the indexer (`idx_q`, `idx_k`, `idx_w`, with RoPE on the indexer as this implementation's choice). Settings in `cfg.extra`: `index_topk`, `index_heads`, `index_head_dim`, `dsa_mode` ("sparse" or "dense" for the warm-up), `dsa_path` ("mask" or "gather").

```python
from frontierlab.attention import dsa
from frontierlab.model import LM, toy

m = LM(toy().with_(attention="dsa", extra={"index_topk": 32}))
dsa.set_dsa(m, collect=True)                 # forward also computes L^I in every DSA layer
out = m(x, labels=x)
loss = out.loss + dsa.indexer_loss(m)        # safe to add: the gradients cannot cross (tests below)
```

The separation is enforced by construction rather than by optimizer bookkeeping: the indexer reads `x.detach()`, the target $p$ is computed under `no_grad`, and the top-$k$ mask is boolean, so $\mathcal{L}^I$ can only reach indexer weights and the LM loss can only reach the rest. `labs/module-05/train_arm.py` adds the two as `lm + (L_I - L_I.detach())`, whose *value* is the LM loss (so the loop's logs stay comparable with dense runs) and whose *gradient* contains both; in the warm-up stage it freezes every non-indexer parameter.

Correctness, measured in this build (`test_attention_m05.py`, float64): gradcheck, causal check and cached decode in 1-, 3- and 7-token steps for DSA in sparse, dense and gather modes and inside a hybrid; with $k \ge T$ the logits equal Baseline-0's (with the same weights) to $10^{-12}$, while $k = 6$ changes them by more than $10^{-3}$; the gather path equals the mask path to $10^{-12}$; and in both stages the indexer loss gives non-zero gradients to every indexer parameter and none to any other, while the LM loss gives none to the indexer.

## What the evidence says

- **DSA: MODEL-SPECIFIC.** One lab, one model family, documented in detail (sections 2.1–2.2). Quality: "no substantial performance degradation" against V3.1-Terminus on short- and long-context tasks, with AA-LCR four points higher in reasoning mode (section 2.2; company claim, against the lab's own predecessor). Cost: Figure 3 plots cost per million tokens against token position for prefill and decode, "estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour" (company claim, their kernels). The report also notes that "for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA": at short context, the sparse path is not the fast one — consistent with the crossover above.
- **NSA: PROMISING.** Published with ablations and kernels by DeepSeek researchers; block-level selection is what hardware-aligned sparse kernels need. Limited independent replication at scale.
- **Training the selector by distillation from dense attention** (KL to the head-summed attention) is PUBLICLY DOCUMENTED for DSA; whether the sparse-stage target $p_{t,S_t}$ is renormalised over $S_t$ is not stated (we renormalise; INFERENCE).
- **At course scale.** The lab converts a 1.8M-parameter model trained at 256 tokens with $k = 32$ (1/8 of the context; V3.2 uses 1/64 at 128K). It can show whether the indexer learns where attention goes and whether the converted model keeps its held-out loss; it cannot show anything about cost at long context (05.3) or about reasoning quality.

## Lab

**Folder:** [`labs/module-05/lesson-02/`](../../labs/module-05/) · **Time:** about 80 minutes (about 32 of them unattended) · **Pass check:** `pytest labs/module-05/lesson-02` passes; `dsa_stages.py` prints the three tables; your notes apply the contract's decision rule.

### Experiment contract

- **Question:** after the documented two-stage conversion, does a DSA-style model at $k = T/8$ keep the held-out loss of the same dense model trained on for the same tokens, and does its learned selection keep more attention mass than a sliding window of the same size? Decision informed: whether DSA is a candidate in the 05.3 cost study and the module memo.
- **Hypotheses and status:** (1) DSA with $k \ge T$ equals dense attention exactly — established (tested); (2) after the warm-up the indexer's top-$k$ keeps more attention mass than the $k$ most recent keys — reported in spirit (V3.2's parity), may not hold at toy scale where attention is mostly local; (3) the sparse model's held-out loss is within 0.03 nats of the control — reported at scale (company claim), may not appear here; (4) skipping the warm-up hurts — expected (the sparse stage would start from random selections), not separately reported by DeepSeek.
- **Baseline:** `control`: the dense parent (`runs/m04/base-cpu`, the Module 4 base model) trained for the same steps, tokens, learning rate and schedule as the sparse stage.
- **Changed variable:** the attention (selection with $k = 32$ of 256 keys) and the indexer it needs; for the ablation, also the warm-up. **Controlled:** parent checkpoint, Data-v0, seed 0, 400 steps × 16 × 256 tokens, learning rate $3 \times 10^{-4}$ with the same warm-up and cosine schedule, evaluation windows.
- **Comparison axis:** equal LM-training tokens from the same parent. The warm-up's 150 steps train only the indexer and are not counted as LM training; say so, and note that the DSA arm used about 37% more forward passes in total.
- **Budget:** free CPU about 32 minutes (measured below); main path about 2 GPU-hours, PROJECTED.
- **Metrics and decision rule:** (a) attention-mass recall at $k$ of the indexer, the oracle top-$k$ and the $k$ most recent keys, 16 validation windows, dense mode; (b) held-out loss on 256 validation windows, paired bootstrap 95% CI of sparse − control. Rule, stated now: the conversion "keeps quality" if the CI of (b) lies inside ±0.03 nats; the selection "is worth it" if the indexer's recall after the warm-up exceeds the window's.
- **Correctness checks:** `pytest labs/module-05/lesson-02` and the DSA tests in `test_attention_m05.py` pass before any training.
- **Fallback evidence:** the Module 5 pilot's component profiles and 30M-parameter conversion, labelled as provided.
- **Limits:** toy scale, 256 tokens, one seed, $k/T = 1/8$, a much smaller indexer than V3.2's relative to the model; recall is measured on the dense attention of the same model, which itself changes during sparse training.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100 80 GB; about 2 GPU-hours, PROJECTED: Baseline-0 at $T = 1024$, $k = 128$; warm-up 500 steps, sparse and control 2,000 steps each × 64 sequences; sparse and control together $2 	imes 788	ext{M} 	imes 131	ext{M tokens} pprox 2.1 	imes 10^{17}$ FLOPs, about 0.3 h at an assumed 20% MFU on 989 TFLOP/s, budgeted at 2 h because our mask path computes every score and the indexer adds its own work. Not run in this build; part of the Module 5 pilot | `python labs/module-05/lesson-02/dsa_stages.py --variant main --device cuda --ablation` (parent: your Module 1 Baseline-0 seed-0 run) |
| Free GPU (Colab/Kaggle T4) | T4, fp32, about 1 hour (PROJECTED) | `dsa_stages.py --variant t4 --device cuda` (parent: the 04.1 T4 base) |
| Free CPU | laptop; measured below | the steps below |

### Steps

1. **Implement** `index_scores`, `select_topk`, `indexer_target`, `indexer_kl` and `trainable` in `lab.py`; run `pytest labs/module-05/lesson-02`. The provided `LabDSA` uses your functions; one test checks that your indexer loss sends gradients only to the indexer.
2. **Convert the Module 4 base model** (needs `runs/m04/base-cpu`; if you skipped 04.1, run `python labs/module-04/lesson-01/train_base.py` first, about 30 minutes):

   ```bash
   python labs/module-05/lesson-02/dsa_stages.py --ablation
   ```

   It runs warm-up, sparse, control and no-warmup, then prints the recall table, the paired held-out losses and the sparse model at other values of $k$. The indexer's KL and recall during training are in `runs/m05/l52/cpu/*/indexer.jsonl`.
3. **Report** (half a page): the recall table and what the oracle column bounds; whether the learned selection beats the recent-$k$ window, and by how much; the paired loss differences and the rule's decision; what the $k$ sweep says about how the model depends on the $k$ it was trained with; and the cost-model sentence for Baseline-0 at 128K: which term dominates?

Measured in this build (free CPU, Windows 11, 16-thread laptop, torch 2.14.1+cpu, fp32; parent `runs/m04/base-cpu`, toy preset trained at 256 tokens; $k = 32$; seed 0; another build job was using the CPU): 32 minutes in total — warm-up 2.4 minutes (150 steps, about 4,300 tokens/s with the main model frozen), sparse 11.7 minutes (about 2,350 tokens/s), control 7.0 minutes (about 3,700 tokens/s), no-warmup 9.1 minutes, report 2 minutes.

| Attention-mass recall at $k = 32$ of 256 keys | indexer | oracle top-$k$ | 32 most recent |
|---|---|---|---|
| random indexer (parent converted, before training) | 0.382 | 0.770 | 0.713 |
| after the dense warm-up | 0.759 | 0.770 | 0.713 |
| after sparse training | 0.753 | 0.762 | 0.705 |
| no-warmup arm after its sparse stage | 0.739 | 0.753 | 0.693 |

The warm-up's KL fell from 4.33 to 0.27 nats in 150 steps, and the indexer went from half the window's recall to within 0.011 of the oracle; it beats the recent-32 window by 0.046, so by the contract the selection "is worth it" at this setting — modestly, because at 256 tokens most of this model's attention is local anyway (the window alone keeps 71%). Held-out loss on 256 validation windows, paired against `control` (5.0025): `sparse` 5.0049, difference $+0.0024$, 95% CI $[+0.0016, +0.0032]$ — inside the ±0.03 margin, so the conversion "keeps quality" by the rule, though the interval excludes zero: the sparse model is very slightly worse. `no-warmup` $+0.0054$ $[+0.0045, +0.0064]$: skipping the warm-up roughly doubled the gap, and its indexer started from a KL of 3.17. The $k$ sweep on the sparse model: $k = 8$: $+0.039$; 16: $+0.011$; 32: $+0.0024$; 64: $+0.0017$; 256 (dense): $+0.0033$ — the model has adapted to sparse attention, so giving it all keys back is slightly worse than $k = 64$. One seed, one parent, toy scale; nothing here bears on cost (05.3).

<details>
<summary>Hint for TODO 2</summary>

Fill the disallowed scores with `-inf`, take `topk(min(k, S))` indices, scatter `True` into a zero boolean tensor, and AND the result with `allowed`: when a row has fewer than $k$ allowed keys, `topk` returns some `-inf` positions, and the AND removes them.

</details>

<details>
<summary>Hint for TODO 4</summary>

Mask `p` to the support and renormalise it; take `log_softmax` of the scores with the outside of the support set to `-inf`; use `torch.where(p > 0, p * (log p - log q), 0)` so that $0 \cdot \log 0$ and $0 \cdot (-\infty)$ never produce `nan`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-05/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-05/lesson-02`.

</details>

## Common mistakes

- **Letting the LM loss "train" the indexer.** It cannot through a top-$k$ mask; if you see gradients on the indexer from the LM loss, the selection is not really discrete (a soft mask) or the indexer shares parameters with the main model.
- **Letting the KL train the main model.** Without detaching the indexer input and the target, $\mathcal{L}^I$ pulls the main attention toward whatever the indexer can represent.
- **Skipping the warm-up.** A random indexer selects random tokens; the sparse stage then trains the model around a bad selector.
- **Selecting future keys in short rows.** `topk` always returns $k$ indices; without the causal AND, early queries attend to the future.
- **Quoting O(Lk) as the cost of DSA.** It is the cost of the selected attention. The indexer is $O(L^2)$; say which term dominates at your context.
- **Expecting a smaller cache.** DSA keeps every token's KV entry plus an indexer key.

## References

- DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, 2025: section 2.1 (DSA), 2.1.1 (continued pre-training), 2.2 (parity evaluation, Figure 3). https://arxiv.org/abs/2512.02556
- J. Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, 2025, section 3. https://arxiv.org/abs/2502.11089
- DeepSeek-AI, *DeepSeek-V3.2* `config.json` (indexer fields; checked 2026-10-04). https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- Shared code: `labs/common/frontierlab/attention/dsa.py`, `accounting.py` (Module 5 section); `labs/module-05/train_arm.py`.

## Next

[05.3 · Measuring cost and quality honestly](lesson-03.md)
