---
id: "04.3"
module: 4
minutes: 40
practice_minutes: 90
prerequisites: ["04.1", "04.2", "02.1"]
objectives:
  - Describe the staged long-context extensions documented by Llama 3, DeepSeek-V3 and DeepSeek-V3.1, and derive their token budgets from the stated numbers.
  - Measure how much same-document context random training windows actually contain, and train on within-document windows from long documents instead.
  - Continue training a checkpoint at a longer length with a fresh optimizer and exact resume, and project the GPU-hours of a staged extension from FLOPs per token at each length.
  - Separate the gain of an extension from the effect of extra training tokens with an equal-token control, and judge it against a pre-stated short-context regression budget.
volatility: concept
sources:
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.2: document attention mask, RoPE base 500,000; section 3.4.2: long-context pre-training)"
    url: https://arxiv.org/abs/2407.21783
  - title: "DeepSeek-AI, DeepSeek-V3 Technical Report (section 4.3: long context extension)"
    url: https://arxiv.org/abs/2412.19437
  - title: "DeepSeek-V3.1 model card (extension phases: 630B tokens at 32K, 209B at 128K)"
    url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
  - title: "Gao et al., How to Train Long-Context Language Models (Effectively) (ProLong; abstract)"
    url: https://arxiv.org/abs/2410.02660
  - title: "Fu et al., Data Engineering for Scaling Language Models to 128K Context (abstract)"
    url: https://arxiv.org/abs/2402.10171
  - title: "Peng et al., YaRN (v3, section 4.1: training budget)"
    url: https://arxiv.org/abs/2309.00071
last_verified: "2026-10-03"
---

# 04.3 · Extending context by continued training

A RoPE rule (04.2) only stops far positions from hurting; it does not teach a model to use them. This lesson extends context the way published models do it, by continued training at longer lengths on long documents in stages, and measures the two things that decide whether it worked: the context gain at the new length and the cost at the old one. The data question turns out to be as important as the position question, because most documents are short and most "long" training windows are several unrelated documents glued together.

## Why this matters at a frontier lab

Long-context extension is a small fraction of pretraining compute, and it is where the context length on the model card is actually produced. The recipes are documented in enough detail to copy, and they differ in ways that matter: how many stages, how many tokens at each length, which data, and what gate decides that a stage is done. A lab that extends its own models needs to know which of these choices its evaluation can see, and whether a gain is from the long context or simply from more training tokens. The second question is the one most often skipped.

## The idea

### What the reports document

- **Llama 3** (section 3.4.2): in 405B pre-training the context length was "increased ... gradually in six stages, starting from the original 8K context window and ending in the final 128K context window", with "approximately 800B training tokens" for this stage (the 405B model). A stage is done when "(1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves 'needle in a haystack' tasks up to that length." The RoPE base was raised to 500,000 from the start of pretraining (section 3.2). PUBLICLY DOCUMENTED.
- **DeepSeek-V3** (section 4.3): YaRN applied "exclusively to the decoupled shared key" with $s = 40$, $\alpha = 1$, $\beta = 32$ and $\sqrt{t} = 0.1 \ln s + 1$; "two additional training phases, each comprising 1000 steps, to progressively expand the context window from 4K to 32K and then to 128K", the first at sequence length 32K and batch 1,920, the second at 128K and batch 480, learning rate $7.3 \times 10^{-6}$, "matching the final learning rate from the pre-training stage". PUBLICLY DOCUMENTED.
- **DeepSeek-V3.1** (model card): "The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens." PUBLICLY DOCUMENTED (company model card; no ablation published).
- **YaRN** itself fine-tuned Llama 2 for 400 steps at batch 64 on 64K-token chunks, then 200 more steps for $s = 32$ (section 4.1).

The V3 and V3.1 numbers are consistent: $32{,}768 \times 1{,}920 \times 1{,}000 = 62.9$B tokens in V3's first phase, and ten times that is 629B, the card's 630B; $131{,}072 \times 480 \times 1{,}000 = 62.9$B in the second, and $3.3 \times 62.9 = 208$B, the card's 209B. (The token counts are this lesson's arithmetic from the stated sequence lengths, batches and steps; the reports state the inputs.)

### What the research on data says

Two studies isolate the data. Fu et al. find that "500 million to 5 billion tokens are enough to enable the model to retrieve information anywhere within the 128K context", and that the data needs both "domain balance and length upsampling": "naively upsampling longer data on certain domains like books ... gives suboptimal performance" (abstract). ProLong (Gao et al.) evaluates with downstream tasks rather than perplexity or NIAH and reports that "code repositories and books are excellent sources of long data, but it is crucial to combine them with high-quality short-context data", and that "training with a sequence length beyond the evaluation length boosts long-context performance"; ProLong-8B was trained "on 40B tokens" at 128K (abstract). PUBLICLY DOCUMENTED; each is one group's study, so the specific mixtures are PROMISING rather than ESTABLISHED.

### The document-boundary problem

The course loop draws training windows at random positions of one stream in which documents are separated by an end-of-text token. On Data-v0 (median document 726 tokens, measured) a random window of 2,048 tokens usually spans three or more documents. Let $c_t$ be the **same-document context** of the token at window position $t$: how many earlier tokens of the window belong to its own document. Inside one document $c_t = t$. Measured on Data-v0's CPU training split (`python -m frontierlab.longctx.data`):

| Window $T$ | tokens with $c_t \geq T/2$, random windows | inside one document | documents $\geq T$ tokens | their share of tokens |
|---|---|---|---|---|
| 256 | 43.7% | 50% | 89.4% | 98.4% |
| 1,024 | 30.1% | 50% | 34.2% | 72.1% |
| 2,048 | 21.2% | 50% | 12.1% | 46.7% |
| 8,192 | 6.6% | 50% | 1.3% | 15.8% |
| 32,768 | 1.4% | 50% | 0.04% | 1.7% |

At 32K almost no token in a random window has 16K tokens of its own document behind it. A model trained on such windows learns that context more than a few hundred tokens back is a different document, which is exactly the wrong lesson.

There are two honest fixes. **Document masking** keeps packed windows but forbids attention across a boundary; Llama 3 uses "an attention mask that prevents self-attention between different documents within the same sequence" and found it "had limited impact during in standard pre-training, but ... important in continued pre-training on very long sequences" (section 3.2; the "during in" is in the original). It needs a per-sequence block mask, which the course attention interface does not carry. **Within-document windows** draw each window from inside one document of at least $T$ tokens; `frontierlab.longctx.data.LongDocData` does this, uniformly over all valid windows, with a `long_fraction` to mix ordinary windows back in. Either way, the long documents must exist: masking short documents gives *no* long-range signal at all.

### Where long documents come from

FineWeb-Edu is web text. In Data-v0's CPU slice only 8 of 19,599 documents have 32,768 tokens or more (0.04% of documents, 1.7% of tokens; measured). `python -m frontierlab.longctx.prepare_long` streams later documents of the same pinned dataset, keeps those above a length threshold, and assigns splits with Data-v0's hash rule, so a held-out document can never be a training document. For the main path, projected from the CPU slice's distribution: reading the full `sample-10BT` (about 10B tokens) and keeping documents of at least 32K tokens yields on the order of 100–170M tokens, PROJECTED, and fewer at vocabulary 32,768 (fewer tokens per document). That covers a 32K stage of a few hundred million tokens with one to two passes over the long documents. Published recipes add books and code repositories because web text runs out; this course does not, to keep one pinned dataset, and the project states it as a limit.

### Continued training, mechanically

`python -m frontierlab.longctx.extend --init-from CKPT ...` wraps `frontierlab.train.loop` (the exact loop change is proposed in `curriculum/inbox/module-04-loop-changes.md`). It loads only the model weights from the checkpoint; the optimizer starts fresh (new AdamW moments, warmup from step 0), because the sequence length and data distribution change and old second-moment estimates describe the old ones. It sets the RoPE rule in `cfg.extra` (the parameters are Baseline-0's, so the weights load with `strict=True`), raises `max_position_embeddings`, swaps the training data for `LongDocData` if asked, and records the initial checkpoint's SHA-256, its step and the data mode in the run card, with `parent_run` set to the base run. Exact resume still holds: rerunning the command after an interruption loads `<run>/checkpoint.pt` over the initial weights (tested: a 6-step run stopped at step 3 and resumed ends bit-identical to an uninterrupted one).

### The control that is usually missing

An extended model has seen more tokens than its base. If it improves on anything, including short-context evaluations, the extra tokens are a candidate explanation. The **equal-token control** continues the same base at the *original* length for the same number of tokens. "The extension helped long context" is supported only by the difference between the extended arm and the control; "the extension cost short-context quality" is measured against the control too, not only against the base.

## Worked example

### A staged plan for Baseline-0, by hand

Baseline-0 was trained at 1,024 tokens. Plan: 400M tokens at 8,192, then 200M at 32,768. Forward FLOPs per token at length $T$ are $2N + 2LT(Hd) + 2VC$ with $N = 96{,}751{,}872$, $L = 12$, $Hd = 768$, $V C = 32{,}768 \cdot 768$ (lesson [01.1](../module-01/lesson-01.md)):

- $T = 8{,}192$: $193.5\text{M} + 2 \cdot 12 \cdot 8{,}192 \cdot 768 + 50.3\text{M} = 193.5 + 151.0 + 50.3 = 394.8$M; training $\times 3 = 1.184$ GFLOP per token. Stage: $1.184 \times 10^9 \times 4 \times 10^8 = 4.74 \times 10^{17}$ FLOPs.
- $T = 32{,}768$: $193.5 + 604.0 + 50.3 = 847.8$M; training $2.543$ GFLOP per token, $2.15\times$ an 8K token. Stage: $5.09 \times 10^{17}$ FLOPs.
- Total $9.83 \times 10^{17}$ FLOPs. On one H100 SXM (989 TFLOP/s dense BF16) at an assumed 30% MFU: $9.83 \times 10^{17} / (0.3 \times 989 \times 10^{12}) / 3600 = 0.92$ GPU-hours. PROJECTED, pending the Module 4 pilot; at 32K the attention term is 71% of the forward FLOPs, so the MFU of the attention kernel decides the real number.

Compare the base run: 2.49B tokens at 1,024 for about 1.8 projected GPU-hours (lesson 01.1). The extension is 24% of the base's tokens and about half its compute. DeepSeek-V3's two phases (125.8B tokens) are about 0.8% of its 14.8T pretraining tokens (abstract); V3.1's (839B) about 5.7%; Llama 3's (800B of the 405B model's 15.6T text tokens, section 2) about 5%.

### Same-document context, tiny numbers

Documents start at offsets 0, 5 and 7 of the stream. A window of 6 tokens starting at offset 3 covers tokens 3–8, which belong to documents 0, 0, 1, 1, 2, 2. Same-document context: $(0, 1, 0, 1, 0, 1)$. Inside one document it would have been $(0, 1, 2, 3, 4, 5)$. Lab test 1 checks exactly this.

### Within-document sampling, tiny numbers

Documents of lengths 5, 2 and 4 at offsets 0, 5, 7, window $T = 3$. Valid starts: document 0 has $5 - 3 + 1 = 3$ (offsets 0, 1, 2), document 1 none, document 2 has 2 (offsets 7, 8). Cumulative counts $(3, 5)$. Drawing $u$ uniformly in $\{0..4\}$: $u = 3$ falls in document 2 (first cumulative count above 3 is 5, the one before is 3), offset $7 + (3 - 3) = 7$. Every valid window has probability $1/5$, so long documents are drawn in proportion to their length.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| training batch | (B, T): CPU (4, 1,024); main path (4, 8,192) × 8 accumulation steps | int64 | GPU / CPU |
| logits | (B, T, V): (4, 8,192, 32,768) | fp32 after the upcast: 4 GiB | GPU |
| attention scores | not materialised by the SDPA flash path; with the explicit mask path (cached decode only) (B, H, T, S) | bf16 | GPU |
| document index for sampling | cumulative valid starts per document, (docs,) | int64 | CPU (NumPy) |

At the main path's 8,192 tokens and batch 4 the fp32 logits are 4 GiB and the loss keeps a second copy; use `--loss chunked` (lesson 02.4) if memory is tight, and lower the batch with more accumulation rather than shortening the sequence. Activation memory of the rest grows linearly in $T$ with flash attention. FLOPs per token grow with $T$ through the attention term, as in the worked example.

## Build it

```python
from frontierlab.longctx.data import LongDocData, same_doc_context

d = LongDocData("train", long_fraction=1.0)
print(d.stats(1024))                 # documents >= 1024 tokens, valid window starts
print((same_doc_context(d, 1024) >= 512).mean())     # random windows: about 0.30
```

```bash
python -m frontierlab.longctx.extend --init-from runs/m04/base-cpu/checkpoint.pt --rope yarn --factor 4 --original 256 \
    --long-fraction 1.0 --run runs/m04/cpu-yarn-within --preset toy --seq 1024 --batch 4 --steps 200 --lr 1e-3 --warmup 20
```

Every argument the loop knows is passed through. `--theta` changes the RoPE base instead of applying a rule; `--short-data DIR` draws the ordinary windows from another prepared folder (long documents in `--data`, Data-v0 in `--short-data`). Correctness checks in `labs/common/tests/test_longctx.py`: within-document windows never contain an end-of-text token except as their last token, ordinary windows do; the same generator state gives the same batch; the same-document context grows by one and resets at boundaries; the extension run's weights after stop-and-resume equal the uninterrupted run's bit for bit, and its run card carries the initial step, the rule and `parent_run`.

## What the evidence says

- **Staged extension by continued training at increasing lengths: ESTABLISHED** (Llama 3, DeepSeek-V3, DeepSeek-V3.1, Qwen and others document it; PUBLICLY DOCUMENTED in each report).
- **Short-context recovery as a stage gate: PUBLICLY DOCUMENTED** for Llama 3; the size of the short-context cost at a given budget is rarely published.
- **More extension tokens:** DeepSeek increased both phases (V3.1 card) without publishing the ablation behind it (company claim of the change, not of its effect).
- **Long data needs both long documents and short high-quality data: PROMISING** (ProLong; Fu et al.), consistent across the two studies.
- **Document masking matters for long continued training:** Llama 3's statement (company claim, no numbers given).
- **At course scale (measured, CPU):** every extension arm removed the far-context penalty but none made far context useful (context gain at 1,024 within $\pm 0.001$ of zero), the equal-token control improved short-context loss by 0.007 nats on its own, and within-document windows did not beat random windows. The lab's reference results give the numbers; none of them is evidence about larger models.

## Lab

**Folder:** [`labs/module-04/lesson-03/`](../../labs/module-04/) · **Time:** about 90 minutes (20 of them unattended) · **Pass check:** `pytest labs/module-04/lesson-03` passes; all arms trained with run cards whose `longctx.init_sha256` matches the base checkpoint; `compare_arms.py` output and the decision for each arm in your write-up.

### Experiment contract

- **Question:** does continued training at 1,024 tokens from the 256-token base, with YaRN and within-document windows, make the model use context beyond 256 tokens, and at what short-context cost? Decision informed: the data mode and rule for the module project's extension.
- **Hypothesis:** the extended arms remove the far-context penalty (gain at 1,024 no longer negative) and the within-document arm gains more far context use than the random-window arm; short-context loss at 256 changes by less than 0.02 nats. Status: reported effects at 7B+ scale (Llama 3 section 3.2 for masking; ProLong for data); may not appear at 1.8M parameters, which barely used 64 tokens before.
- **Baseline:** the base model (`runs/m04/base-cpu`), plus two references: `base+yarn` (YaRN switched on, no training) and `ctrl-short` (the equal-token control at 256).
- **Changed variables (one per arm, against `ctrl-short`):** length and rule (`yarn-within`: 1,024 with YaRN, $s = 4$, within-document windows); data mode (`yarn-random` vs `yarn-within`); rule (`pi-within` vs `yarn-within`). **Controlled:** the base checkpoint (by SHA-256), seed 0, learning rate $10^{-3}$ with 20 warmup steps and cosine decay, a fresh optimizer, 200 steps × 4,096 tokens = 819,200 tokens per arm, Eval v1 pins, Eval v0 windows and LAMBADA file.
- **Comparison axis:** equal training tokens. It does not answer which arm is cheaper: a 1,024-token step costs more FLOPs per token than a 256-token step (the attention term), measured in the run cards' `budget.train_flops`.
- **Budget:** free CPU, about 3.5 minutes per arm, 4 arms; scoring about 3 minutes per model (measured below).
- **Metrics and decision rule:** primary: context gain at 1,024 with $W = 256$ (95% CI over 100 documents); guard: held-out loss at 256 on Eval v0's fixed windows minus the base's, paired by window. Rule, as in `decide`: adopt if the guard's upper bound is $\leq 0.02$ nats and the gain's lower bound is $\geq 0$; reject if the guard's lower bound exceeds 0.02 or the gain's upper bound is below 0; otherwise inconclusive. Secondary: far-position loss and LAMBADA log-probability, each against the base and against `ctrl-short`; the evidence effect at 1,024.
- **Correctness checks:** `pytest labs/common/tests/test_longctx.py` (within-document windows, exact resume of the wrapper); every arm's run card names the same base SHA-256; `python -m frontierlab.record runs/m04/cpu-yarn-within runs/m04/cpu-yarn-random` shows `longctx.long_fraction` as the only difference besides bookkeeping (the record tool flags it as WARN, unclassified, so read the `longctx` lines yourself).
- **Fallback evidence:** none; a null result is a valid result at this scale.
- **Limits:** one seed (the Module 1 project measured a seed std of 0.016 nats on held-out loss for the toy recipe, so differences of that size between arms are not resolvable); one small model; 0.8M tokens per arm; FineWeb-Edu web text only.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB; 4 arms of 524M tokens (1K → 8K, batch 4 × 8 accumulation); PROJECTED $\approx 0.4$–$0.6$ GPU-hours per arm, about 2 GPU-hours in all. Not run in this build; part of the Module 4 pilot | `extend_arms.py --variant main --base <your Module 1 seed-0 run>`, then `compare_arms.py --base <run> --prefix runs/m04/main- --original 1024 --new 8192 --device cuda` |
| Free GPU (Colab/Kaggle T4) | T4, about 40 minutes | `extend_arms.py --variant t4` (base from `train_base.py --variant t4`), then `compare_arms.py --base runs/m04/base-t4 --prefix runs/m04/t4- --original 512 --new 2048 --device cuda` |
| Free CPU | laptop; measured 2026-10-03 (16 threads, other jobs running): arms 193–220 s each, about 14 minutes for four; `compare_arms.py` 729 s the first time (six models, Eval v1 at two lengths and Eval v0 with all 5,153 LAMBADA passages) | the steps below |

### Steps

1. **Implement** `same_doc_context`, `within_doc_start`, `stage_plan` and `decide` in `lab.py`; run `pytest labs/module-04/lesson-03`.
2. **Measure the boundary problem** on your data: `python -m frontierlab.longctx.data --split train`. Compare with the table in the lesson.
3. **Project the main-path plan** with your `stage_plan`: Baseline-0, 400M tokens at 8,192 and 200M at 32,768, H100 SXM, MFU 0.30. Check the total against the worked example (0.92 GPU-hours).
4. **Train the arms** (unattended, about 14 minutes; rerun if interrupted): `python labs/module-04/lesson-03/extend_arms.py`.
5. **Score and decide:** `python labs/module-04/lesson-03/compare_arms.py`. It uses your `decide`.
6. **Write up** (one page): the decision for each arm under the contract's rule; what `ctrl-short` shows about the extra tokens; whether within-document windows beat random windows at this scale and how confident you can be with one seed; the short-context cost of static YaRN alone (`base+yarn`) against after training; and what you would change for the module project.

<details>
<summary>Hint for TODO 2</summary>

Keep only documents with `lengths >= T`; their counts of valid starts are `lengths - T + 1`; `np.cumsum` of those, then `np.searchsorted(cum, u, side="right")` is the index of $u$'s document among the kept ones, and $u$ minus the count before it is the offset inside that document.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-03 on the build laptop (torch 2.14.1 CPU; arms 193 s, 218 s, 239 s and 215 s; `compare_arms.py` 729 s for six models). Held-out = Eval v0 loss at 256 minus the base's (positive = worse); LAMBADA = target log-probability minus the base's (negative = worse); far loss = positions $[512, 1023)$ at 1,024 minus the base's; all paired, 95% CI.

| Model | context gain at 1,024, W = 256 | held-out at 256 | LAMBADA | far loss | decision |
|---|---|---|---|---|---|
| base | −0.0657 [−0.0722, −0.0599] | — | — | — | — |
| base+yarn (no training) | −0.0005 [−0.0014, +0.0004] | +0.0378 [+0.0349, +0.0405] | +0.084 [+0.076, +0.091] | −0.120 [−0.133, −0.109] | reject |
| ctrl-short | −0.0654 [−0.0715, −0.0601] | −0.0071 [−0.0116, −0.0027] | −0.002 [−0.021, +0.018] | −0.003 [−0.011, +0.006] | reject |
| yarn-within | +0.0003 [−0.0006, +0.0011] | +0.0400 [+0.0323, +0.0477] | −0.124 [−0.145, −0.103] | −0.137 [−0.151, −0.124] | reject |
| yarn-random | +0.0003 [−0.0003, +0.0008] | +0.0068 [+0.0018, +0.0115] | −0.106 [−0.126, −0.084] | −0.149 [−0.163, −0.135] | inconclusive |
| pi-within | −0.0009 [−0.0017, −0.0001] | +0.0597 [+0.0524, +0.0672] | −0.138 [−0.160, −0.117] | −0.118 [−0.131, −0.105] | reject |

What it shows. (1) Every extended arm removed the far-context *penalty* (far loss about 0.12–0.15 nats better than the base, gain from −0.066 to about 0) but none made the far context *useful*: every gain interval covers or lies below zero, so no arm passes the primary criterion. At 1.8M parameters, a model that used about 64 tokens before extension does not learn to use 256+ tokens from 0.8M tokens of training; that is a valid null result, not a failed experiment. (2) The control matters: `ctrl-short` improved short-context loss by 0.007 with the same tokens, so against the control `yarn-random`'s short-context cost is about +0.014, not +0.007. (3) Most of the short-context cost of `yarn-within` (+0.040) is already there in `base+yarn` (+0.038): it is the static rule, evaluated at 256 with the factor on. Training on ordinary windows (`yarn-random`) recovered most of it; training only inside long documents did not, plausibly because long FineWeb-Edu documents are a different distribution from the average window (INFERENCE; ProLong's finding that long data must be mixed with short data points the same way). (4) The hypothesis that within-document windows beat random windows is not supported at this scale: the two gains are identical within $\pm 0.001$. (5) PI is worst again, as in 04.2. (6) LAMBADA moves the other way from held-out loss for `base+yarn` and is worse for every trained arm; with one seed, report it and do not explain it away. The mean retrieval evidence effect at 1,024 rose from +0.010 (base) to +0.018 (`yarn-within`) and +0.020 (`pi-within`), small and without a pooled interval.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-04/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-04/lesson-03`.

</details>

## Common mistakes

- **No equal-token control.** Any improvement is then confounded with more training. Train `ctrl-short` with the same tokens, seed and schedule.
- **Measuring "long-context gain" on random windows.** Most of the context in them is other documents; use document-start windows (Eval v1) for evaluation and long documents for training.
- **Resuming the base run's optimizer state.** Its moments and schedule belong to the old run; continued training starts a fresh optimizer with its own warmup, and the run card says so.
- **Pointing `--init-from` at the run folder you are writing to.** On the second launch the loop loads the run's own checkpoint (exact resume) and ignores `--init-from`; that is intended, but a fresh run needs a fresh folder.
- **Comparing arms on budgets that differ.** Equal tokens is not equal FLOPs: a token at 8K costs 1.5× a token at 1K for Baseline-0 (1.184 vs 0.788 GFLOP), 3.2× at 32K. State the axis and report both.
- **Evaluating long documents that were in training.** `prepare_long` uses Data-v0's hash split for this reason; do not pool train documents into the evaluation set because long ones are scarce.

## References

- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, sections 3.2 and 3.4.2. https://arxiv.org/abs/2407.21783
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024, section 4.3. https://arxiv.org/abs/2412.19437
- DeepSeek-AI, *DeepSeek-V3.1* model card, 2025. https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- T. Gao et al., *How to Train Long-Context Language Models (Effectively)*, 2024, abstract. https://arxiv.org/abs/2410.02660
- Y. Fu et al., *Data Engineering for Scaling Language Models to 128K Context*, 2024, abstract. https://arxiv.org/abs/2402.10171
- B. Peng et al., *YaRN*, v3, section 4.1. https://arxiv.org/abs/2309.00071
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The module project extends Baseline-0 from 2K to 32K and reports what it can and cannot use: [Module 4 project](../../projects/module-04-context-extension.md). Module 5 then asks when sub-quadratic attention is worth it, judged with Eval v1. Context parallelism and Ring Attention, which split one long sequence across GPUs, are Module 9's subject.
