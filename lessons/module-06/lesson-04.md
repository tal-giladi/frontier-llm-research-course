---
id: "06.4"
module: 6
minutes: 30
practice_minutes: 45
prerequisites: ["06.2", "06.3"]
objectives:
  - Explain conditional memory as a sparsity axis separate from MoE — lookup by token context instead of computation selected by a router — and why it is not a residual-stream design.
  - Implement Engram's multiplicative-XOR N-gram hash and its context-aware gate, and verify them against the shared module, including causality and cached decoding.
  - Compute routed-expert counts from the paper's allocation ratio and check them against the reported 46, 43 and 55 experts.
  - Measure N-gram counts and hash collisions on Data-v0 and run a small iso-parameter MoE vs MoE + Engram comparison, stating what one tiny run can and cannot say about the reported U-shaped allocation law.
volatility: concept
sources:
  - title: "DeepSeek-AI — Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram), sections 2–4, 6"
    url: https://arxiv.org/abs/2601.07372
  - title: "DeepSeek-AI — DeepSeekMoE (fine-grained and shared experts)"
    url: https://arxiv.org/abs/2401.06066
last_verified: "2026-10-04"
---

# 06.4 · Conditional memory and lookup sparsity: Engram

Extension: a Mixture-of-Experts layer makes a model larger without making each token more expensive by computing only a few experts per token. Engram does the same with *memory*: it looks up a few rows of a very large embedding table, indexed by a hash of the last few tokens, and adds a gated, filtered version of them to the residual at a couple of layers. DeepSeek presents it as a second sparsity axis next to MoE and reports that, at a fixed parameter and compute budget, the best models split the sparse parameters between the two. This lesson builds the module, checks its hashing on Data-v0, reproduces the paper's expert counts from its allocation ratio, and runs one small iso-parameter comparison.

## Why this matters at a frontier lab

A model's parameter count is bounded by accelerator memory and by FLOPs per token. MoE relaxes the FLOP bound; the parameters still sit in HBM. Engram's tables are indexed by token ids alone, so the rows a forward pass will need are known before the pass starts; DeepSeek offloads a 100B-parameter table to host memory with "negligible overhead (< 3%)" by prefetching (section 1, section 6.4). If that holds, memory capacity becomes a separate scaling axis that costs PCIe bandwidth, not HBM or FLOPs. The research question for a lab is the paper's own: *given a fixed total and active parameter budget, how much of the sparse budget should go to experts and how much to lookup memory?*

## The idea

### Lookup instead of routing

For position $t$ with token ids $x_{\le t}$ and hidden state $h_t \in \mathbb{R}^C$ at the layer where Engram sits (paper section 2, Eqs. 1–5):

1. **Suffix N-grams.** $g_{t,n} = (x_{t-n+1}, \dots, x_t)$ for $n = 2, \dots, N$. The paper first maps token ids through a *tokenizer compression* $\mathcal{P}: V \to V'$ that merges ids differing only by case or Unicode form ("a 23% reduction in the effective vocabulary size for a 128k tokenizer", section 2.2); the course module skips it.
2. **Multi-head hashing.** $K$ hash heads per order: $z_{t,n,k} = \varphi_{n,k}(g_{t,n}) \in [0, M_{n,k})$ with prime table sizes $M_{n,k}$; $\varphi$ is "a lightweight multiplicative-XOR hash". The retrieved rows are concatenated: $e_t = \Vert_{n,k} E_{n,k}[z_{t,n,k}] \in \mathbb{R}^{d_{\mathrm{mem}}}$.
3. **Context-aware gate.** $k_t = W_K e_t$, $v_t = W_V e_t$, and
$$\alpha_t = \sigma\!\left(\frac{\mathrm{RMSNorm}(h_t)^\top \mathrm{RMSNorm}(k_t)}{\sqrt{C}}\right), \qquad \tilde v_t = \alpha_t v_t .$$
"If the retrieved memory $e_t$ contradicts the current context $h_t$, the gate $\alpha_t$ tends toward zero" — this is how hash collisions and ambiguous N-grams are suppressed.
4. **Filter and add.** $Y = \mathrm{SiLU}(\mathrm{Conv1D}(\mathrm{RMSNorm}(\tilde V))) + \tilde V$, a depthwise causal convolution with kernel $w = 4$ and dilation $N$; then $H \leftarrow H + Y$ before the layer's attention and FFN.

Everything is causal: the N-gram at $t$ uses tokens up to $t$, the convolution looks back only. The course module keeps the last $N - 1$ ids and the last $(w - 1) N$ filtered values in the decode cache, and passes cached-decode agreement.

### Why it is not a residual-stream design

Lesson 06.2 changed *how* every layer reads and writes the residual. Engram changes *what* is added at one or two layers: a lookup result, gated by the current state. It composes with any residual design; DeepSeek's own experiments run it on an mHC backbone with $M = 4$ streams, sharing one table and $W_V$ across streams and giving each stream its own $W_K$ (section 2.4). In MoE terms: an expert is selected by a learned router from the hidden state and costs FLOPs; an Engram row is selected by a fixed hash of token ids and costs a memory read.

### The allocation problem

Section 3.1 fixes total parameters $P_{\mathrm{tot}}$ and activated parameters $P_{\mathrm{act}}$ (so FLOPs), and splits the *inactive* budget $P_{\mathrm{sparse}} = P_{\mathrm{tot}} - P_{\mathrm{act}}$:

$$P_{\mathrm{MoE}}^{(\mathrm{sparse})} = \rho\, P_{\mathrm{sparse}}, \qquad P_{\mathrm{Engram}} = (1 - \rho)\, P_{\mathrm{sparse}} .$$

$\rho = 1$ is pure MoE. As reported: validation loss against $\rho$ is U-shaped at both compute budgets tested ($2 \times 10^{20}$ and $6 \times 10^{20}$ FLOPs; $P_{\mathrm{tot}} \approx 5.7$B and 9.9B, $P_{\mathrm{act}} = 568$M and 993M), with the optimum "stable across regimes ($\rho \approx 75\%$–80%)"; in the 10B regime loss improves "from 1.7248 (at $\rho = 100\%$) to 1.7109 near the optimum of $\rho \approx 80\%$ ($\Delta = 0.0139$)", and Engram matches pure MoE even at $\rho \approx 40\%$ (section 3.1, Figure 3). Two budgets, one sparsity ratio ($P_{\mathrm{tot}}/P_{\mathrm{act}} \approx 10$), one lab: a law in the sense of a fitted curve, not yet a replicated one.

## Worked example

### Hashing a trigram by hand

Course hash (`blocks/engram.py`): $h \leftarrow ((h \cdot m) \bmod (2^{31} - 1)) \oplus (x + 1)$ over the n-gram's tokens, then $\bmod M$. Trigram $(5, 9, 2)$ with $m = 48{,}271$, $M = 4{,}093$:

- $h = (0 \cdot m) \oplus 6 = 6$
- $h = (6 \cdot 48{,}271 = 289{,}626) \oplus 10 = 289{,}616$ (binary: XOR flips bits 1 and 3 of 289,626, both set)
- $h = (289{,}616 \cdot 48{,}271 \bmod 2{,}147{,}483{,}647) \oplus 3$; $289{,}616 \cdot 48{,}271 = 13{,}980{,}053{,}936$, minus $6 \cdot 2{,}147{,}483{,}647 = 12{,}884{,}901{,}882$ gives $1{,}095{,}152{,}054$; XOR 3 gives $1{,}095{,}152{,}053$
- bucket $= 1{,}095{,}152{,}053 \bmod 4{,}093 = 1{,}095{,}152{,}053 - 267{,}567 \cdot 4{,}093 = 322$

The lab's test checks your hash bit for bit against the module.

### Collisions: why several heads

With $N$ distinct n-grams uniformly hashed into $M$ buckets, an n-gram is alone in its bucket with probability $(1 - 1/M)^{N - 1} \approx e^{-N/M}$. Data-v0's first 2M training tokens contain 507,261 distinct bigrams: with $M = 1{,}000{,}003$ that is $e^{-0.507} = 0.60$ alone — 40% collide. With $K$ independent heads an n-gram is uncollided in at least one of them with probability $1 - (1 - 0.60)^K$: 0.84 for $K = 2$, 0.97 for $K = 4$. The gate can then lean on the clean head. This is the reason for multi-head hashing (section 2.2: "to mitigate collisions, we employ $K$ distinct hash heads").

### The paper's expert counts from $\rho$

Routed experts at $\rho$, with $E_1$ experts at $\rho = 1$ of which $k$ are active: $E(\rho) = k + \rho (E_1 - k)$ (the active experts are not part of the sparse budget). With $k = 6$ (the top-$k$ of the paper's 27B models, section 4): 5.7B model, $E_1 = 106$: $6 + 0.4 \cdot 100 = 46$; 9.9B model, $E_1 = 99$: $6 + 0.4 \cdot 93 = 43.2 \to 43$ — exactly the "46 experts for the 5.7B model and 43 experts for the 9.9B model" of section 3.1. Engram-27B: $E_1 = 72$, $\rho = 74.3\%$: $6 + 0.743 \cdot 66 = 55.0$ — the reported 72 → 55. (That the 5.7B and 9.9B runs also used top-6 is INFERENCE from these numbers; section 3.1 does not state $k$.)

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| token ids, history kept in the cache | (B, T), (B, N−1) | int64 | GPU / CPU |
| hash bucket per table | (B, T) | int64 | same |
| tables $E_{n,k}$ | ($M_{n,k}$, $d_{\mathrm{head}}$) each | fp32 master; DeepSeek shards them (training) or keeps them in host memory (inference) | GPU, or host memory |
| retrieved memory $e_t$ | (B, T, $d_{\mathrm{mem}}$) | bf16 / fp32 | GPU |
| gate $\alpha$ | (B, T) (or (B, T, n) with n streams) | same | same |
| conv history in the cache | (B, $(w-1)N$, C) | same | same |

Cost per token: $(N - 1)K$ row reads ($d_{\mathrm{head}}$ values each) and no FLOPs for the lookup; $2 d_{\mathrm{mem}} C$ multiply-adds each for $W_K$ and $W_V$; a $w$-tap depthwise convolution. The tables' parameters count in $P_{\mathrm{tot}}$ but not in $P_{\mathrm{act}}$ (`frontierlab.blocks.accounting` reports them as `engram_tables`).

The paper's 27B configuration (section 4): Engram at layers 2 and 15, maximum N-gram size 3, 8 heads, dimension 1,280, 5.7B embedding parameters; the tables trained with Adam at 5× the learning rate and no weight decay, the convolution initialised to zero.

## Build it

`labs/common/frontierlab/blocks/engram.py`: `hash_ngrams`, `EngramModule` (tables, gate, causal convolution, decode cache, multi-branch support), `collision_rate`. In a `BlockLM`:

```python
cfg = with_blocks(toy(vocab_size=8192), engram={"layers": [1], "max_n": 3, "heads": 2, "table": 1465})
```

```bash
python -m frontierlab.blocks.train --ffn moe --moe-experts 6 --engram-layers 1 --engram-table 1465 \
    --run runs/m06/try-engram --preset toy --steps 200 --batch 16 --seq 128 --lr 1.5e-3
```

The wrapper keeps Engram tables out of weight decay (as the paper does) but uses the loop's learning rate for them (the paper's 5× is not applied; a stated difference). Correctness checks, passing in `test_blocks.py`: the hash is deterministic, in range, and causal (changing tokens from position 12 leaves buckets before 12 unchanged); float64 gradcheck of the module with respect to the hidden state and every parameter including the tables; causal check and cached-decode agreement for Engram alone and combined with mHC, MoE and MTP; `collision_rate` counts exactly on a hand-made sequence.

## What the evidence says

- **Conditional memory as a sparsity axis — PROMISING, one lab.** PUBLICLY DOCUMENTED by DeepSeek (this paper); no independent replication found. Related lookup ideas are older (N-gram embeddings; hash embeddings; BLT's encoder hash embeddings, lesson 06.6), but the allocation result against MoE is this paper's.
- **Reported results** (section 4, Table 1; all models 262B tokens, 3.8B activated parameters): Engram-27B against an iso-parameter, iso-FLOPs MoE-27B — BBH +5.0, ARC-Challenge +3.7, DROP +3.3, HumanEval +3.0, MATH +2.4, GSM8K +2.2, MMLU-Pro +1.8, CMMLU +4.0. MMLU is +3.0 in Table 1 (57.4 → 60.4) and in section 4's text; the abstract and introduction say "MMLU +3.4", which matches the table's MMLU-Redux row (60.6 → 64.0) — the course uses the table. Long context (section 5): Multi-Query NIAH 84.2 → 97.0 at matched pre-training loss. Mechanistic claim (section 6.1): early Engram layers make the backbone's representations at layer 5 look like the MoE baseline's at about layer 12 — "effectively deepening the network".
- **Placement** (section 6.2): a single module works best at layer 2 among the layers swept; the system design wants it deeper to hide prefetch latency — a modelling/systems trade-off the paper states explicitly.
- **The U-shape** is reported at two budgets with one sparsity ratio; whether the optimum moves at other ratios or scales is open.

## Lab

**Folder:** [`labs/module-06/lesson-04/`](../../labs/module-06/) · **Time:** about 45 minutes · **Pass check:** `pytest labs/module-06/lesson-04` passes; your notes contain the collision table with measured and predicted shares, the three expert counts, and the iso-parameter result with its interval and the limits paragraph.

### Experiment contract

- **Question:** at toy scale and equal total parameters, does moving a third of the inactive expert budget into an Engram table change held-out loss? Decision informed: none for Lineage-F (Engram is not in its main design); this is an illustration of the allocation question and of how to set up an iso-parameter comparison.
- **Hypothesis and status:** a hybrid allocation beats pure MoE — reported at 5.7B/9.9B (section 3.1); **may not appear at 2.4M parameters**, where the tables are tiny (four of ~1,455 rows against 507k distinct bigrams in 2M tokens: almost every lookup collides).
- **Baseline:** the MoE arm (8 routed experts, top-2, width 96, one shared expert, MoE in layers 1–3).
- **Changed variable:** the allocation: 6 routed experts plus an Engram module at layer 1. **Controlled:** total parameters (2,390,080 vs 2,392,448, −0.1%), data, steps, seed, schedule, Eval v0 windows.
- **Comparison axis:** **equal total parameters** (the paper's iso-parameter axis), approximately equal FLOPs: the Engram projections add 1.8% training FLOPs per token (`blocks.accounting`), stated as a limit.
- **Budget:** free CPU about 8 minutes; no main-path run (extension).
- **Metrics and decision rule:** held-out loss paired over 256 windows; one seed, so only an interval over windows — no seed noise — and therefore no adoption claim: the result is reported as "difference [interval] on one seed".
- **Correctness checks:** `pytest labs/common/tests/test_blocks.py -k engram` passes.
- **Limits:** one seed, one $\rho$, tiny tables, 200 steps, no tokenizer compression, no 5× table learning rate.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× GPU, optional. Not run in this build | `iso_param.py --variant main --device cuda` (pilot-30m with the same arm definitions; the table size should then be re-derived for the 30M model's expert width — the arm names fix toy-scale sizes, so treat this as an exercise) |
| Free GPU (Colab/Kaggle T4) | T4 | `iso_param.py --variant t4 --device cuda` |
| Free CPU | laptop; measured below | every step as written |

### Steps

1. **Implement** `ngram_hash`, `engram_gate`, `experts_at` and `collision_free_share`; run `pytest labs/module-06/lesson-04`.
2. **Collisions:** `python labs/module-06/lesson-04/collisions.py`. Compare measured and predicted collision shares; if they disagree badly, suspect the hash (see Common mistakes).
3. **Iso-parameter run:** `python labs/module-06/lesson-04/iso_param.py`. Record the difference, its interval, and the gate statistics, and write the limits paragraph.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, with another build job sharing the CPU, 2026-10-04):

| Measurement | Result |
|---|---|
| distinct n-grams in the first 2M training tokens | bigrams 507,261; trigrams 1,339,944 |
| collision share, one head, 1,000,003-row table (measured / uniform prediction) | bigrams 0.384 / 0.398; trigrams 0.738 / 0.738 |
| same, 4,093- and 65,521-row tables | 1.000 for both orders (every n-gram shares its bucket) |
| bigrams alone in ≥ 1 of K heads at 1,000,003 rows (prediction) | K = 2: 0.842, K = 4: 0.975, K = 8: 0.999 |
| `moe` (8 experts) vs `moe6+engram1465`, toy, 200 steps, seed 0 | held-out 6.6640 vs 6.6792; difference **+0.0152 [+0.0111, +0.0192]**; total parameters 2,392,448 vs 2,390,080; training 10.77 vs 10.96 MFLOP/token |
| Engram gate at layer 1, 16 held-out windows | mean 0.635, 10th–90th percentile 0.11–0.98 |
| runtimes | `collisions.py` 37 s; `iso_param.py` 4.4 minutes (two runs of about 2 minutes) |

What to write about it. At this scale moving a third of the inactive budget into lookup tables costs 0.015 nats on one seed — the opposite direction from the paper's 5.7B/9.9B result, and unsurprising: four tables of about 1,455 rows hold half a million distinct bigrams, so every row is shared by hundreds of contexts, while the two removed experts were real capacity. The gate is neither closed nor open (mean 0.64, spread 0.11–0.98): the model uses some lookups and suppresses others. One seed, one ρ, tiny tables: this is one point at the wrong end of the U-curve's axis of table size, not a test of the law.

<details>
<summary>Hint for TODO 1</summary>

Build the padded matrix once with `torch.cat([torch.full((B, n - 1), pad), ids], 1)` and loop over the $n$ token offsets, not over positions: at offset `j` the tokens are `padded[:, j:j + S]`. Do the arithmetic in `torch.int64`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-04`.

</details>

## Common mistakes

- **A hash multiplier that shares structure with the table size.** This build's first version used a multiplier equal to the 1,000,003-row table size: products of small ids were 0 modulo the table, and 98% of bigrams collided where uniform hashing predicts 40%. Measure collisions against the uniform prediction before trusting a hash.
- **Counting the tables as active parameters.** They are read, not multiplied; a FLOP estimate that includes them overstates Engram's cost by the table size.
- **Reading the gate as "memory use".** $\alpha$ near 1 means the retrieved vector agrees with the context, not that the model depends on it; ablate the module to measure dependence.
- **Comparing at equal *active* parameters only.** The allocation question holds total *and* active parameters fixed; adding a table to a fixed MoE is a capacity increase, not a reallocation.
- **Placing Engram where only the system wants it.** Deeper placement hides prefetch latency; the paper finds early placement models better. State which constraint you optimised.

## References

- DeepSeek-AI, *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*, sections 2.1–2.5, 3.1, 4, 5, 6.1–6.4. https://arxiv.org/abs/2601.07372
- D. Dai et al., *DeepSeekMoE*. https://arxiv.org/abs/2401.06066
- Shared code: `labs/common/frontierlab/blocks/engram.py`, `moe.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[06.5 · Elastic architectures](lesson-05.md)
