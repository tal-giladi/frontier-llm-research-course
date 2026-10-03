---
id: "04.1"
module: 4
minutes: 35
practice_minutes: 90
prerequisites: ["01.3", "01.4", "01.5"]
objectives:
  - Explain why a needle-in-a-haystack score and an advertised context length do not show that a model uses its context, with the published evidence for each failure.
  - Build and validate Eval Suite v1's long-context components (position loss, context gain, synthetic retrieval and multi-hop tasks with controlled depth and distractors) with an oracle and an evidence-ablation check.
  - Measure a model's context gain and evidence effect with paired intervals, and state its effective length under a pre-stated threshold.
  - Compute how many evaluation items a planned long-context comparison needs before running it.
volatility: concept
sources:
  - title: "Hsieh et al., RULER: What's the Real Context Size of Your Long-Context Language Models? (abstract; section 3 task categories; section 4 effective length threshold)"
    url: https://arxiv.org/abs/2404.06654
  - title: "Liu et al., Lost in the Middle: How Language Models Use Long Contexts (abstract)"
    url: https://arxiv.org/abs/2307.03172
  - title: "Yen et al., HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly (abstract)"
    url: https://arxiv.org/abs/2410.02694
  - title: "Modarressi et al., NoLiMa: Long-Context Evaluation Beyond Literal Matching (abstract)"
    url: https://arxiv.org/abs/2502.05167
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.4.2: long-context pre-training criteria)"
    url: https://arxiv.org/abs/2407.21783
  - title: "Why Did MiniMax M2 End Up as a Full Attention Model? (MiniMax, Hugging Face blog)"
    url: https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
last_verified: "2026-10-03"
---

# 04.1 · Long-context evaluation that means something

A model that accepts 128K tokens has not shown that it uses them. This lesson builds the evaluation the rest of the module, and Module 5, depends on: Eval Suite v1, which measures whether far context lowers the loss on real documents, whether the model can retrieve and combine facts placed at controlled positions among distractors, and what the long context cost the model at short context. You will validate the evaluation itself before trusting any number it produces.

## Why this matters at a frontier lab

Every long-context decision in this course is made with this evaluation: which RoPE rule to use (04.2), how many tokens of long data to spend (04.3), and whether a sub-quadratic attention design keeps quality at 32K–128K (Module 5). If the evaluation only checks that a model can copy a fact back, every design that can copy passes, and the expensive mistakes (a hybrid that cannot do multi-hop reasoning over its context, a context extension that quietly damaged short-context quality) are found after the big run, not before.

The published record shows that this happens. RULER reports that "despite achieving nearly perfect accuracy in the vanilla NIAH test, almost all models exhibit large performance drops as the context length increases", and that of the models it tested that claim 32K tokens or more, "only half of them can maintain satisfactory performance at the length of 32K" (RULER, abstract). NoLiMa removes the word overlap between the question and the hidden fact and finds that at 32K "11 models drop below 50% of their strong short-length baselines" (NoLiMa, abstract). MiniMax explains its return to full attention partly by deficits in multi-hop reasoning that its hybrid model showed only at scale and only on harder evaluations (MiniMax blog, company claim).

## The idea

### Five ways a long-context evaluation lies

1. **It only measures retrieval.** Needle-in-a-haystack (NIAH) hides one sentence and asks for it back. A model with a working copy mechanism (an induction head that finds the query words earlier in the context and copies what followed) solves it without understanding anything. Using a fact (combining it with another, following a chain) is a different capability.
2. **The question shares words with the answer.** If the query repeats the needle's wording, retrieval reduces to string matching. NoLiMa is built to remove that shortcut.
3. **The fact is always in the same place.** Liu et al. found that performance "is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle" (Lost in the Middle, abstract). An evaluation that always puts the needle at the end measures the easy case.
4. **There are no distractors.** One fact in irrelevant text is easier than one fact among several similar ones.
5. **It ignores what the long context cost.** Extending context changes the weights. If short-context quality dropped, the extension is a trade, not a gain; Llama 3's long-context stages advanced only when "model performance on short-context evaluations has recovered completely" (Llama 3, section 3.4.2).

A sixth, quieter one: **averaged perplexity**. The mean loss over a long document is dominated by tokens that are predictable from the last few hundred tokens. A model can ignore everything further back and lose almost nothing on the average. Loss has to be split by position, and compared against the same model with the far context removed.

### What Eval Suite v1 measures

Eval v1 (`frontierlab/evals/suite_v1.py`, version `eval-v1.0`) has four components. Each is scored per item, so two models can be compared with the paired bootstrap of lesson [01.4](../module-01/lesson-04.md).

**1. Natural-text loss by document position.** Take the first $L$ tokens of every held-out document with at least $L$ tokens, so position $t$ really is position $t$ in a document. Report the mean loss in position buckets $[0, 64), [64, 128), [128, 256), \dots$. If the model uses its context, loss keeps falling with position. If it extrapolates badly past its trained length, loss rises there.

**2. Context gain.** For a target token at document position $t$, let $\ell_t^{\text{full}}$ be its loss when the model sees document tokens $0..t-1$ and $\ell_t^{W}$ its loss when the context is cut so that it starts at position $\text{lo}-W$ (at least $W$ tokens of context). For targets $S = [\text{lo}, \text{hi})$ of one document $j$:

$$G_j(W) = \frac{1}{|S|} \sum_{t \in S} \left( \ell^{W}_{t} - \ell^{\text{full}}_{t} \right)$$

Here $\ell$ is next-token cross-entropy in nats, $W$ the number of tokens kept, $S$ the target positions (Eval v1 uses $[L/2, L)$), and the result is averaged over documents with a bootstrap interval. $G > 0$: tokens more than $W$ back lowered the loss, so the model uses them. $G \approx 0$: it ignores them. $G < 0$: the far context *hurts*, the signature of a model running past the positions it was trained on.

**3. Synthetic retrieval and multi-hop tasks.** A haystack of held-out natural text with short statements inserted, built from single-token words so no tokenizer merge crosses a boundary:

```text
... natural text ... Remember: river candle. ... Ignore: river planet. ... Remember: orbit fabric. ...
... natural text ... Remember: river
```

The model must rank the right value (`candle`) first among the candidate values. The generator controls the variables that the five failures above depend on: total length $L$, the depth of the answer statement (0 = start, 1 = just before the query), the number of other `Remember` statements (distractors), *lexical decoys* (`Ignore: river planet` shares the key with the query, so a model that only matches the key picks the decoy half the time), and the number of hops: `Remember: k1 k2. Remember: k2 k3. Remember: k3 value.` with the statements shuffled through the haystack and the query `Remember: k1`. One hop is retrieval; two or more require using what was retrieved.

**4. Short-context regression.** Eval v0 (held-out loss on the fixed windows at the original length and LAMBADA target log-probability, [Module 1 project](../../projects/module-01-baseline0.md)) on the extended model against the model it came from, paired item by item.

### Scores, chance and the evidence effect

For an item with $K$ candidates $c_1..c_K$ and logits $z$ at the last position:

$$\text{logp}_{\text{cand}} = z_{a} - \log \sum_{k=1}^{K} e^{z_{c_k}}, \qquad \text{correct} = [\, z_a > z_{c_k} \ \forall c_k \neq a \,]$$

where $a$ is the answer. Chance accuracy is $1/K$ and chance $\text{logp}_{\text{cand}}$ is $\ln(1/K)$. Accuracy against chance is a weak test at small scale: a model has prior preferences among the candidate tokens (frequent words get more probability), which can put it below or above chance without reading anything. So every item has an **evidence-ablated twin**: the same item with the target chain replaced by filler text. The **evidence effect** is the paired mean of $\text{logp}_{\text{cand}}(\text{item}) - \text{logp}_{\text{cand}}(\text{twin})$. The prior is identical in both, so it cancels; what remains is how much the statement itself moved the model toward the answer.

### Is the evaluation valid?

Before any model is scored, four checks (they are this lesson's lab):

1. **Solvable.** An oracle that reads the statements answers 100% of items. A generator bug that drops the answer statement fails here.
2. **Not solvable without the evidence.** On the ablated twin, the oracle finds nothing.
3. **No prior leak.** A model's accuracy on the twins is consistent with $1/K$. If it is clearly above chance, the answer can be guessed (for example, the answer value is always the most frequent word).
4. **Same items for every model.** Items are generated from pinned inputs (tokenizer hash, filler split, seed); `compare` refuses to pair results whose pins differ.

### Effective versus advertised length

The **advertised length** is what a model accepts (its `max_position_embeddings` or the length it was trained at). The **effective length** is the longest length at which a stated score stays above a stated threshold at that length *and every shorter one*. RULER uses Llama-2-7B's score at 4K as the threshold and defines "the effective context length" as "the maximum length passing this threshold" (RULER, section 4 and Table 3). In this course the threshold is written in the experiment contract before the runs; a result that passes at 2,048 after failing at 1,024 does not count, because the failure at 1,024 is real.

## Worked example

### Context gain by hand

Two documents, three target tokens each, losses in nats:

| Document | $\ell^{\text{full}}$ | $\ell^{W}$ | $G_j$ |
|---|---|---|---|
| 1 | 4.0, 3.0, 5.0 | 4.3, 3.1, 5.2 | $(0.3 + 0.1 + 0.2)/3 = 0.200$ |
| 2 | 2.0, 6.0, 4.0 | 2.0, 5.9, 4.1 | $(0.0 - 0.1 + 0.1)/3 = 0.000$ |

Mean gain $0.100$ nats. Document 1 benefits from its far context; document 2 does not. With only two documents no interval means anything; Eval v1 uses up to 100 documents per length and bootstraps over documents, because tokens in one document are not independent.

### How many items does a synthetic comparison need?

Accuracy on $n$ items has standard error $\sqrt{p(1-p)/n}$. With $K = 4$ candidates (chance $p_0 = 0.25$) and a model you expect to score $p_1 = 0.35$, the items needed to detect the difference with 80% power at $\alpha = 0.05$ (two-sided, normal approximation) are

$$n = \left( \frac{1.96\sqrt{p_0(1-p_0)} + 0.84\sqrt{p_1(1-p_1)}}{p_1 - p_0} \right)^2 = \left( \frac{1.96 \cdot 0.433 + 0.84 \cdot 0.477}{0.10} \right)^2 = 12.49^2 \approx 157.$$

Forty items per cell (the CPU default) can only separate $p_1 = 0.50$ from chance ($n = 26$) or larger effects. If you expect a 10-point effect, plan 157 items per cell or pool cells, and say which before you look.

### What a key-matching model scores

With one decoy (`Ignore: river planet`) and three other keys, a model that finds the query key and copies the next token chooses between `candle` and `planet` at random: expected accuracy $0.5$ among $K = 5$ candidates, against chance $0.2$. A score near $0.5$ on items with one decoy is the fingerprint of string matching; a score near 1 means the model also reads the prefix.

### Effective length from a sweep

Illustrative numbers, not a measurement: retrieval accuracy at depth 0.5 (95% intervals) at 4K, 8K, 16K, 32K: $0.92\,[0.86, 0.96]$, $0.88\,[0.81, 0.93]$, $0.71\,[0.62, 0.79]$, $0.90\,[0.84, 0.95]$. With the threshold $0.80$ on the lower bound, the effective length is 8K: 16K fails ($0.62 < 0.80$), so 32K does not count even though it passes on its own.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| synthetic items | (n, L) token ids, grouped by length | int64 | GPU (main path) / CPU |
| last-position logits | (n, V) | fp32 | same |
| document windows | (docs, L) | int64 | same |
| per-token losses | (docs, L − 1) | fp32, kept on CPU as NumPy | CPU |
| context-gain cut windows | (docs, hi − lo + W) | int64 | GPU / CPU |

Cost, with $F(T)$ the forward FLOPs per token at context $T$ from `frontierlab.flops.flops_per_token(cfg, T, training=False)`: one synthetic item of length $L$ costs $L \cdot F(L)$; for Baseline-0 at $L = 32{,}768$ that is $32{,}768 \times 847.8\text{M} = 2.78 \times 10^{13}$ FLOPs, about 0.09 s on an H100 at an assumed 30% MFU (PROJECTED, pending the Module 4 pilot). The suite at one length runs $2n$ synthetic forwards per cell (item and twin) and three forwards per document (position loss, full and cut context for the gain).

Memory is the trap. The model returns logits for every position, $(B, L, V)$ in fp32: at $L = 32{,}768$ and $V = 32{,}768$ that is $32{,}768^2 \times 4$ bytes $= 4$ GiB *per sequence*, before the loss makes its copy. Score long items with batch 1, or compute the loss with the chunked cross-entropy of Module 2 ([02.2](../module-02/lesson-02.md)). The KV cache is not involved: every item is one full forward.

## Build it

```python
from frontierlab.data.loader import TokenData
from frontierlab.evals import suite_v1 as v1

vocab = v1.vocab_from_tokenizer("labs/common/data/v0/tokenizer.json", TokenData("val"))
items = v1.make_items(vocab, length=1024, n=40, hops=1, depth=0.5, distractors=3, hard=1, seed=0)
assert all(v1.oracle_answer(it, vocab) == it["answer"] for it in items)          # check 1
assert all(v1.oracle_answer(it, vocab, "ids_ablated") is None for it in items)   # check 2

model = v1.load_model("runs/m04/base-cpu")
cell = {"scores": v1.score_items(model, items), "scores_ablated": v1.score_items(model, items, "ids_ablated")}
print(v1.summarize_scores(cell["scores"])["acc"], v1.evidence_effect(cell))
```

The command line runs all components at several lengths and writes one JSON per model; `compare` pairs two of them:

```bash
python -m frontierlab.evals.suite_v1 run runs/m04/base-cpu --train-len 256 --lengths 256 512 1024 2048 --out runs/m04/base-cpu/eval_v1.json
python -m frontierlab.evals.suite_v1 compare runs/m04/base-cpu/eval_v1.json runs/m04/other/eval_v1.json
```

What the code does, in order: `make_items` draws keys, values and decoys from disjoint word lists, cuts filler from the validation stream, moves each insertion point back to the nearest sentence end within 48 tokens, inserts the statements (the answer statement at the requested depth), and builds the ablated twin by replacing the target chain with the next filler tokens, so both have exactly $L$ tokens. `score_items` batches items of equal length, takes the logits at the last position, and scores the answer against the candidates only. `context_gain` runs each document twice, once from position 0 and once from $\text{lo} - W$, and lines up the loss columns of the same target tokens. Held-out documents come from `labs/common/data/v0-long` if you prepared it (below), otherwise from Data-v0's validation split; in both cases the split is decided by the same text hash as Data-v0, so no training document can be a held-out document.

The suite's own tests (`labs/common/tests/test_longctx.py`) check that items are deterministic, exactly $L$ long, solvable by the oracle and unsolvable without the evidence; that an untrained model's candidate log-probability is near $\ln(1/K)$; that the context gain is exactly zero when nothing is cut ($W = \text{lo}$) and equals a hand computation otherwise; and that `short_context_regression` refuses unpaired results.

## What the evidence says

- **Retrieval-only evaluation overstates usable context: ESTABLISHED.** Independent benchmarks from different groups reach the same conclusion with different designs: RULER (synthetic retrieval, multi-hop tracing, aggregation, QA; section 3), NoLiMa (no literal overlap), HELMET ("synthetic tasks like NIAH do not reliably predict downstream performance", abstract), Lost in the Middle (position). PUBLICLY DOCUMENTED, each in its abstract.
- **Position sensitivity (lost in the middle): ESTABLISHED** for the models those papers tested; how it changes with newer training recipes is an open question you can measure with the depth sweep.
- **Short-context regression as a gate: PUBLICLY DOCUMENTED** for Llama 3 (section 3.4.2, with NIAH as the other gate); REASONABLE INDUSTRY PRACTICE generally.
- **Hybrid attention hiding multi-hop deficits until scale:** MiniMax's account (company claim); Module 5 tests it with this suite.
- **At course scale.** Measured on the CPU base model of this module (toy preset, 1.8M parameters, trained at 256 tokens; 2026-10-03, 16-thread laptop): its context gain *within* its trained length is $+0.0014$ nats, 95% CI $[-0.0004, +0.0030]$ (L = 256, W = 64): it barely uses anything more than 64 tokens back. Its retrieval accuracy is at chance at every length and depth, yet the evidence effect is positive and resolvable when the statement sits just before the query ($+0.066$ nats, CI $[+0.016, +0.117]$ at L = 256) and shrinks toward zero in the middle of long contexts ($+0.002$, CI $[-0.002, +0.005]$ at L = 2,048, depth 0.5). That is position sensitivity in a model far too small to retrieve, and it is why Eval v1 reports the evidence effect next to accuracy. Whether Baseline-0 retrieves above chance is a hypothesis for the main path.

## Lab

**Folder:** [`labs/module-04/lesson-01/`](../../labs/module-04/) · **Time:** about 90 minutes (30 of them unattended) · **Pass check:** `pytest labs/module-04/lesson-01` passes; `check_eval.py` prints four PASS lines for your `lab.py`; your write-up states the base model's effective length under the contract's rule, with the evidence for it.

### Experiment contract

- **Question:** up to what length does the Module 4 base model use its context, and is Eval v1 a valid measurement on it? Decision informed: which lengths and which components the 04.2 and 04.3 comparisons use as their primary metrics.
- **Hypothesis:** context gain is positive inside the trained length and turns negative beyond it (the model extrapolates badly); synthetic accuracy is near chance at this scale. Status: the first is a reported effect for RoPE models run past their trained length without scaling (Position Interpolation paper, abstract; lesson 04.2); the second may not hold on the main path, where Baseline-0 has about 70 times more parameters.
- **Baseline:** the base model, `runs/m04/base-cpu` (main path: your Baseline-0 seed-0 run from the Module 1 project). No tuning: this lab measures one model.
- **Changed variable:** evaluation length (256, 512, 1,024, 2,048; main path 1K–32K). **Controlled:** the same checkpoint, the same pinned items (`eval-v1.0`, seed 0, Data-v0 tokenizer hash, filler from Data-v0 validation), the same held-out documents per length.
- **Comparison axis:** not a training comparison; the same model at different evaluation lengths.
- **Budget:** free CPU about 30 minutes to train the base, 3–4 minutes for Eval v1, 1 minute for `check_eval.py` (measured, below).
- **Metrics and decision rule:** primary: context gain at each length with W = 256 (the trained length) and W = 64, 95% bootstrap CI over documents. Secondary: evidence effect per cell, retrieval accuracy against chance, loss by position. Rule, stated now: the effective length on natural text is the longest swept length whose gain CI at W = 64 has a lower bound $\geq 0$ at that length and every shorter one.
- **Correctness checks:** the four validity checks of `check_eval.py` pass; `pytest labs/common/tests/test_longctx.py` passes; the eval JSON's `pins` match between any two results you compare.
- **Fallback evidence:** none needed; a model that uses nothing beyond 64 tokens is a valid result.
- **Limits:** one 1.8M-parameter model on web text; held-out documents of at most 20K tokens (CPU); 40 items per synthetic cell, enough to detect only large accuracy effects (worked example).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100 80 GB, about 30 GPU-minutes. Not run in this build; part of the Module 4 pilot | no base training (use your Module 1 Baseline-0 seed-0 run); `prepare_long` with `--skip 2600000 --docs 2000000 --min-tokens 8192 --splits val test --out labs/common/data/v0-long` (vocab 32,768 Data-v0); then `python -m frontierlab.evals.suite_v1 run <run> --train-len 1024 --lengths 1024 2048 4096 8192 16384 32768 --n 100 --device cuda --bf16 --out runs/m04/b0-eval_v1.json` |
| Free GPU (Colab/Kaggle T4) | T4, about 30 minutes for the base and 5 for the eval | `train_base.py --variant t4`, then the suite with `--train-len 512 --lengths 512 1024 2048 4096 --device cuda` (no `--bf16` on a T4) |
| Free CPU | laptop; measured 2026-10-03, 16 threads, other jobs running: base about 30 minutes (3,550 tokens/s), Eval v1 at four lengths 200 s, `check_eval.py` 64 s, `prepare_long` 6.7 minutes and about 1 GB of download | the steps below |

### Steps

1. **Train the base model** (unattended, about 30 minutes; rerun the same command if it stops):

   ```bash
   python labs/module-04/lesson-01/train_base.py
   ```

   While it trains, prepare more held-out long documents (optional but recommended: Data-v0's validation split has only 61 documents of at least 1,024 tokens and 18 of at least 2,048):

   ```bash
   python -m frontierlab.longctx.prepare_long --skip 20000 --docs 200000 --min-tokens 1024 --splits val test --out labs/common/data/v0-long
   ```

   In the build this gave 648 validation documents of at least 1,024 tokens, 219 of at least 2,048 and 57 of at least 4,096.

2. **Implement** `oracle_answer`, `candidate_score`, `context_gain`, `effective_length` and `items_needed` in `lab.py`; run `pytest labs/module-04/lesson-01`.
3. **Validate the evaluation** on the trained base: `python labs/module-04/lesson-01/check_eval.py --run runs/m04/base-cpu`. All four checks must pass before step 4 counts.
4. **Run Eval v1** at four lengths:

   ```bash
   python -m frontierlab.evals.suite_v1 run runs/m04/base-cpu --train-len 256 --lengths 256 512 1024 2048 --out runs/m04/base-cpu/eval_v1.json
   ```

5. **Write up** (half a page): (a) the context gain at each length for W = 64 and W = 256 and the effective length by the contract's rule; (b) loss by position at 2,048: where does it start to rise, and why there? (c) retrieval accuracy against chance, and the evidence effect by depth: is there position sensitivity, and how do you know it is not noise? (d) how many items per cell you would need to detect a 10-point accuracy difference at the accuracy you measured.

<details>
<summary>Hint for TODO 3</summary>

`per_token_loss[:, p]` is the loss of predicting token $p + 1$. Target token $t$ is therefore column $t - 1$ of the full window and column $t - (\text{lo} - W) - 1$ of the cut window; for $t = \text{lo}$ that is column $W - 1$.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Base model (toy, 1,500 steps × 16 × 256 tokens, final validation loss 4.975), 100 documents per length from `v0-long`:

| L | gain W = 64 | gain W = 256 | loss at [L/2, L) |
|---|---|---|---|
| 256 | +0.0014 [−0.0004, +0.0030] | — | 5.024 ([128, 255)) |
| 512 | −0.0475 [−0.0542, −0.0411] | 0.0000 (nothing cut) | 5.128 |
| 1,024 | −0.1184 [−0.1289, −0.1092] | −0.0657 [−0.0724, −0.0599] | 5.206 |
| 2,048 | −0.1762 [−0.1862, −0.1668] | −0.1326 [−0.1402, −0.1254] | 5.406 |

No length passes the rule (256 fails: its lower bound is $-0.0004$), so the natural-text effective length is "below 256": the model uses at most about 64 tokens of context. Past 256 the far context actively hurts and loss rises with position. Retrieval accuracy is at chance everywhere (between 0.10 and 0.35; every interval contains chance, 0.2 for retrieval with $K = 5$ and 0.25 for two hops), and `check_eval.py` reports 0.195 against a chance of 0.225 on 200 items. The evidence effect is clearly positive only at depth 1.0 (+0.066, +0.087, +0.036, +0.025 at 256, 512, 1,024, 2,048) and near zero in the middle at long lengths.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-04/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-04/lesson-01` and run `LAB_TARGET=solution python labs/module-04/lesson-01/check_eval.py --run runs/m04/base-cpu`.

</details>

## Common mistakes

- **Reporting NIAH as "context length".** It shows the model can find one literal string. Report context gain, multi-hop and short-context regression next to it, or say that only retrieval was measured.
- **Averaging loss over the whole window.** Near tokens dominate; a model that ignores everything far away loses almost nothing. Split by position and compare with the cut context.
- **Reading accuracy below chance as "worse than random".** It is usually the model's prior over the candidate words. Use the ablated twin: the evidence effect cancels the prior.
- **Comparing models on different items or documents.** Regenerated items with another seed or tokenizer are not paired; `compare` refuses results whose pins differ.
- **Taking eval documents from the training split.** Long documents are scarce, and it is tempting to reuse them. The hash-split rule of `prepare_long` keeps held-out documents held out.
- **Scoring 32K items with batch 8.** The fp32 logits alone are 4 GiB per sequence at vocabulary 32,768; the run dies or silently falls back to a slower path.

## References

- C.-P. Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, 2024, abstract, sections 3–4, Table 3. https://arxiv.org/abs/2404.06654
- N. F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, 2023, abstract. https://arxiv.org/abs/2307.03172
- H. Yen et al., *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly*, 2024, abstract. https://arxiv.org/abs/2410.02694
- A. Modarressi et al., *NoLiMa: Long-Context Evaluation Beyond Literal Matching*, 2025, abstract. https://arxiv.org/abs/2502.05167
- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, section 3.4.2. https://arxiv.org/abs/2407.21783
- MiniMax, *Why Did MiniMax M2 End Up as a Full Attention Model?* https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[04.2 · Position at long range](lesson-02.md)
