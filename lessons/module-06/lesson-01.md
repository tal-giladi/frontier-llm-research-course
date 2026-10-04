---
id: "06.1"
module: 6
minutes: 40
practice_minutes: 90
prerequisites: ["01.3", "01.4", "01.5", "02.4"]
objectives:
  - Explain the difference between Meta's parallel multi-token heads and DeepSeek-V3's sequential MTP modules, including what each extra prediction may see and why DeepSeek calls its design a complete causal chain.
  - Implement DeepSeek-style MTP modules, their shifted targets and the published loss-weight schedule, and verify them against a reference and a causality test.
  - Compute by hand the training FLOPs MTP adds to Baseline-0 and to the CPU toy model, and say why the extra output head dominates at small width.
  - Run an MTP comparison at equal training FLOPs with seeds and a pre-stated decision rule, and report a null or negative result as one.
  - Measure the acceptance rate of the MTP modules used as a draft for self-speculative decoding, check that greedy speculative decoding is lossless, and project its decode speed-up with a stated formula.
volatility: concept
sources:
  - title: "Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (sections 2, 3.1, 3.2)"
    url: https://arxiv.org/abs/2404.19737
  - title: "DeepSeek-AI — DeepSeek-V3 Technical Report (section 2.2 MTP; 4.2 hyper-parameters; 4.5.1 ablation; 5.4.3 MTP evaluation)"
    url: https://arxiv.org/abs/2412.19437
  - title: "Qwen — Qwen3-Next-80B-A3B-Instruct model card (MTP listed; vLLM speculative config qwen3_next_mtp)"
    url: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
  - title: "Gemma Team — Gemma 4 Technical Report (section 2.6: Multi-Token Prediction Drafter)"
    url: https://arxiv.org/abs/2607.02770
last_verified: "2026-10-04"
---

# 06.1 · Multi-token prediction

A next-token model learns from one target per position. Multi-token prediction (MTP) adds targets further ahead — the token after next, and so on — through extra prediction modules that are trained with the model. Two published designs differ in what those modules may look at: Meta's parallel heads predict every future token from the same hidden state, while DeepSeek-V3's sequential modules each see the token in between. This lesson builds the DeepSeek design on Baseline-0, counts what it costs, compares it with next-token training at equal training FLOPs, and then reuses the modules as a draft for speculative decoding and measures how often the draft is accepted.

## Why this matters at a frontier lab

MTP is in recent frontier open models for two different reasons. DeepSeek-V3 trains with one MTP module "to improve training" and notes it "may enable the model to pre-plan its representations" (section 2.2); at inference it can drop the module or reuse it as a speculative-decoding draft, with a reported 85–90% acceptance of the second token and 1.8 times the tokens per second (section 5.4.3). Qwen3-Next lists MTP as a feature that "boosts pretraining model performance and accelerates inference" and ships a vLLM speculative-decoding configuration for it (model card). Gemma 4 trains a separate small MTP drafter head for speculative decoding (report section 2.6). So the question a lab faces is concrete: *should our next pretraining run carry MTP modules — for quality, for decode speed, or not at all?* The training objective costs real FLOPs (26.8% more per token for Baseline-0, below), so the answer needs a comparison on the right axis, and the decode benefit needs an acceptance rate, not an assumption.

## The idea

### Two ways to predict further ahead

Write the input tokens $t_1, \dots, t_T$, the model's final hidden state at position $i$ as $h_i \in \mathbb{R}^C$ (before the final norm), the embedding $\mathrm{Emb}(\cdot)$ and the shared output head $\mathrm{OutHead}(\cdot)$ (final norm, then the tied unembedding $W_E^\top$ of shape $C \times V$).

**Meta's parallel heads** (Gloeckle et al., section 2). A shared trunk computes $z_i$ from $t_{\le i}$; $n$ independent heads, each a transformer layer, predict $t_{i+1}, \dots, t_{i+n}$ from $z_i$ alone, through a shared unembedding. The loss is the unweighted sum over heads,

$$L_n = -\sum_i \sum_{k=1}^{n} \log P(t_{i+k} \mid z_i) \qquad \text{(Meta Eq. 2)}.$$

Head $k$ must predict $t_{i+k}$ without knowing $t_{i+1}, \dots, t_{i+k-1}$. To keep parameters equal, Meta moves layers: "when we add $n - 1$ layers in future prediction heads, we remove $n - 1$ layers from the shared model trunk" (section 3.1). In the course code head 1 *is* Baseline-0's last layer, so the trunk is layers $0..L-2$ and the next-token path, including its decode cache, is unchanged.

**DeepSeek-V3's sequential modules** (section 2.2, Eqs. 21–25). $D$ modules; module $k$ has its own projection $M_k \in \mathbb{R}^{C \times 2C}$ and transformer block $\mathrm{TRM}_k$, and shares the embedding and output head with the main model:

$$h'^{\,k}_i = M_k\big[\mathrm{RMSNorm}(h^{k-1}_i);\ \mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))\big], \qquad h^k_{1:T-k} = \mathrm{TRM}_k\big(h'^{\,k}_{1:T-k}\big),$$

$$P^k_{i+k+1} = \mathrm{OutHead}(h^k_i), \qquad \mathcal{L}^k = \mathrm{CE}\big(P^k, t_{\cdot + k + 1}\big), \qquad \mathcal{L}_{\mathrm{MTP}} = \frac{\lambda}{D}\sum_{k=1}^{D} \mathcal{L}^k,$$

with $h^0_i = h_i$. Module $k$ at position $i$ sees the real token $t_{i+k}$, so its prediction of $t_{i+k+1}$ is conditioned on every token before it: DeepSeek keeps "the complete causal chain for the prediction of each token at each depth" (Figure 3 caption), and compares the principle to EAGLE's drafter. Each depth's sequence is $k$ positions shorter.

The total training loss is $\mathcal{L}_{\mathrm{main}} + \mathcal{L}_{\mathrm{MTP}}$. DeepSeek-V3 uses $D = 1$ and "$\lambda$ is set to 0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens" (section 4.2). Nothing is detached: MTP gradients reach the main model's hidden states, embedding and head — that is the point (a denser training signal), and also the risk (the main model's representation is pulled towards what helps the extra predictions).

### Reuse as a draft

At inference the modules are either dropped — "the main model can function independently and normally" (section 2.2) — or used to draft. With $D = 1$, greedy self-speculative decoding is:

1. The main model, at the current last position $i$, picks $y_1 = \arg\max P(\cdot \mid t_{\le i})$.
2. The MTP module drafts $d_2 = \arg\max \mathrm{OutHead}(\mathrm{TRM}_1(M_1[\mathrm{RMSNorm}(h_i); \mathrm{RMSNorm}(\mathrm{Emb}(y_1))]))$.
3. One main-model forward over $t_{\le i}, y_1, d_2$ gives the main model's own choice $y_2$ after $y_1$. If $d_2 = y_2$ the draft is accepted and the round produced two tokens; otherwise one.

The output is exactly what plain greedy decoding produces — speculative decoding changes speed, never the tokens — so a correct implementation passes an equality test. With acceptance probability $p$ per draft and $D$ chained drafts, a round yields $1 + p + \dots + p^D$ tokens. Its cost is one main step plus $D$ draft steps of relative cost $c$; in the memory-bound decode regime the extra verified tokens are nearly free, so

$$\text{speed-up} \approx \frac{1 + p + \dots + p^D}{1 + D c}.$$

This is a projection, not a latency measurement. Module 15 measures latency with a serving engine; this lesson measures $p$.

## Worked example

### Targets and loss with six tokens

Tokens $t_1..t_6$, $D = 1$. The main head at positions 1..5 predicts $t_2..t_6$ (5 targets). Depth 1 runs on positions 1..5 of the *shifted* sequence — it combines $h_i$ with $\mathrm{Emb}(t_{i+1})$ — and predicts $t_{i+2}$; only positions 1..4 have a target ($t_3..t_6$), so 4 targets. If the main cross-entropy averages 2.0 nats and depth 1's averages 2.5, the training loss with $\lambda = 0.3$ is $2.0 + 0.3 \cdot 2.5 = 2.75$; evaluation still reports 2.0, the main head's loss. In code (0-based): depth $k$'s logit at index $i$ is compared with `labels[i + k + 1]`. Getting this shift wrong by one trains the module to predict a token it can see — the loss drops fast and means nothing; the lab's causality test catches it.

### The λ schedule for a 9,500-step run

DeepSeek switches at $10/14.8 = 0.6757$ of training. For Baseline-0's 9,500 steps that is step $\lceil 0.6757 \cdot 9500 \rceil = 6{,}419$: $\lambda = 0.3$ for steps 0–6,418 and $0.1$ after. (The schedule is defined on tokens; with a constant batch, steps are proportional.)

### What MTP costs Baseline-0

Course convention: forward FLOPs per token = $2\times$ the weights used + attention scores; training = $3\times$ forward. One DeepSeek module on Baseline-0 ($C = 768$, $V = 32{,}768$, $L = 12$, $T = 1024$):

- weights: one block (8,062,592, lesson 01.1) + $M_1$ ($2C \cdot C = 1{,}179{,}648$) + three norm gains ($3C = 2{,}304$) = 9,244,544; forward $2 \times$ that = 18,489,088
- the shared head applied again: $2VC = 50{,}331{,}648$
- one more layer of attention scores: $2 T (H d) = 2 \cdot 1024 \cdot 768 = 1{,}572{,}864$

Forward 70,393,600, training $3\times$ = 211.2M FLOPs per token, against Baseline-0's 788.1M: **+26.8%**. Most of it is the output head, not the block. On the CPU toy model ($C = 128$, $V = 8{,}192$, $L = 4$) the head is $2VC = 2.1$M of a 3.8M-FLOP forward, and MTP adds **+68%** (11.41 → 19.18 MFLOP per token, `frontierlab.blocks.accounting`). Meta's head 2 costs about the same (one block plus the head, without $M_1$): +25.9% for Baseline-0, +66% for toy.

Memory: each extra prediction materialises another $(B, T, V)$ logits tensor (4 GiB in fp32 at $B = 32$, $T = 1024$ for Baseline-0, lesson 01.1). Meta avoids holding them all by running the heads' forward and backward one at a time and accumulating the trunk gradient, reducing peak memory "from $O(nV + d)$ to $O(V + d)$" (section 2). The course code runs them together, which is fine at course scale and is what `--loss chunked` does not support (the wrapper refuses the combination).

### Acceptance into speed

DeepSeek reports $p \in [0.85, 0.90]$ and 1.8× tokens per second. With $D = 1$, $1 + p = 1.85$ tokens per round at $p = 0.85$; $1.85/(1 + c) = 1.8$ gives $c \approx 0.03$ — consistent with one extra layer against 61 (INFERENCE from the formula; DeepSeek does not publish its draft cost).

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| main hidden state $h$ (before final norm) | (B, T, C) | bf16 under autocast, fp32 on CPU | GPU (main path) / CPU |
| embedding of the token ahead, depth $k$ | (B, T−k, C) | same | same |
| concatenation into $M_k$ | (B, T−k, 2C) | same | same |
| depth-$k$ hidden $h^k$ | (B, T−k, C) | same | same |
| depth-$k$ logits | (B, T−k, V) | fp32 (cast before the loss) | same |
| Meta head-$k$ logits | (B, T, V) | fp32 | same |

| Model, $T$ | Training MFLOP/token, next-token | + DeepSeek $D = 1$ | + Meta $n = 2$ | Extra parameters |
|---|---|---|---|---|
| CPU toy, 128 | 11.41 | 19.18 (+68.1%) | 18.98 (+66.4%) | 230,080 / 197,056 |
| Baseline-0, 1024 | 788.1 | 999.3 (+26.8%) | 992.2 (+25.9%) | 9.24M / 8.06M |

Decode with the modules dropped costs exactly Baseline-0. Decode with drafting adds one block and one head per draft step, and changes how many main-model steps are needed.

## Build it

`labs/common/frontierlab/blocks/` holds Module 6's code. `model.py` defines `BlockLM`, a subclass of `frontierlab.model.LM`: with every switch off it is Baseline-0 — same modules, same Qwen3 names, same initialisation draws, bit-identical logits (tested). Switches live in `cfg.extra["blocks"]`, so they are recorded in every run card and checkpoint:

```python
from frontierlab.blocks import BlockLM, with_blocks
from frontierlab.model import toy

cfg = with_blocks(toy(vocab_size=8192), mtp="deepseek", mtp_depth=1)   # or mtp="meta"
model = BlockLM(cfg)
out = model(idx, labels=idx)
out.loss                     # main + lambda * MTP loss (what the optimizer sees)
out.per_token_loss           # main head only: what Eval v0 reports, comparable across arms
out.extras                   # {"main_loss": ..., "mtp_loss": ..., "mtp_losses": [...]}
```

`mtp.py` has `DeepSeekMTP` (modules under `mtp.layers.{k}` with `enorm`, `hnorm`, `eh_proj`, a block and `norm` — DeepSeek-V3's checkpoint names), `MetaHeads`, the shifted losses, `lambda_at` (the V3 schedule), `speculative_greedy` and `teacher_forced_acceptance`. Training goes through the unmodified course loop with a wrapper, like Modules 4 and 7:

```bash
python -m frontierlab.blocks.train --mtp deepseek --mtp-depth 1 --mtp-schedule deepseek \
    --run runs/m06/try --preset toy --steps 200 --batch 16 --seq 128 --lr 1.5e-3 --blocks-log
```

The wrapper swaps in `BlockLM`, its exact parameter and FLOP accounting (`blocks/accounting.py`, so `budget.train_flops` in the run card includes the MTP modules), sets $\lambda$ each step, and logs main and MTP losses to `blocks.jsonl`. Exact resume holds (tested: stop at step 7, resume, bit-identical weights and losses).

Correctness checks, all in `labs/common/tests/test_blocks.py` and passing: causal check and cached-decode agreement (1 and 5 tokens per step, differences below $10^{-14}$ in float64) for Meta and DeepSeek variants — for MTP models this checks the main head's decode path, the only one decoding uses; float64 gradcheck of a DeepSeek module (combine, block) with respect to inputs and all parameters; speculative greedy decoding equals plain greedy decoding token for token for both designs; the loss equals main + 0.3 × MTP exactly.

## What the evidence says

- **MTP as a training objective — PROMISING.** Two labs publish ablations with gains, both at scale and both their own designs. Meta: at 13B, "+12% more problems on HumanEval and 17% more on MBPP" than next-token models; $n = 4$ best for tokens, $n = 8$ for bytes (section 3); and, important for this course, "multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale" (section 3.1, Figure 3; sizes 0.3B–13B). DeepSeek-V3 (section 4.5.1, Table 4): on a 15.7B-total MoE with 1.33T tokens and a 228.7B-total MoE with 540B tokens, MTP improves most benchmarks (HumanEval 20.7 → 26.8 and 44.5 → 53.7). Both are PUBLICLY DOCUMENTED; neither is an independent replication, and DeepSeek's ablation compares at equal training tokens with the MTP module discarded at evaluation, not at equal training FLOPs.
- **MTP modules as a built-in draft — PROMISING, increasingly common, designs MODEL-SPECIFIC.** DeepSeek-V3: 85–90% second-token acceptance, 1.8× TPS (section 5.4.3, company claim). Meta: self-speculative decoding "speedup of 3.0× on code with an average of 2.5 accepted tokens out of 3 suggestions" (section 3.2). Qwen3-Next ships MTP with a vLLM speculative config (model card). Gemma 4 trains a separate "small autoregressive MTP drafter head" whose 4-layer block cross-attends to the main model's KV cache (section 2.6) — a third design, built for drafting rather than for the training signal.
- **Open questions.** Whether the training gain survives an equal-FLOPs comparison; whether it exists below ~1B parameters (Meta says no); what $\lambda$ and $D$ should be for a given size — DeepSeek gives one schedule, no sweep.

## Lab

**Folder:** [`labs/module-06/lesson-01/`](../../labs/module-06/) · **Time:** about 90 minutes (about 30 of them unattended training) · **Pass check:** `pytest labs/module-06/lesson-01` passes; your notes contain the filled contract, the comparison table with paired intervals and the noise floor, the decision by the rule, the measured acceptance rate with its lossless check, and the projected speed-up with its formula.

### Experiment contract

- **Question:** at this scale, does training with one DeepSeek-style MTP module lower the main head's held-out loss compared with next-token training *at equal training FLOPs*? Decision informed: whether Lineage-F (the module project) carries an MTP module as a training objective, and whether it is kept as a draft.
- **Hypothesis and status:** H1: MTP lowers held-out loss at equal tokens. Reported effect at ≥1B (DeepSeek Table 4; Meta at scale); Meta reports the opposite for small models, so **may not appear — or reverse — at this scale**. H2: at equal training FLOPs MTP is no better than Baseline-0 trained for 68% more steps (CPU) / 27% more (Baseline-0). H3: the MTP module is a usable draft — acceptance clearly above chance (reported 85–90% at 671B; unknown at course scale).
- **Baseline:** Baseline-0 at the preset, same learning rate (1.5e-3 on CPU, the Module 3 CPU choice) and cosine schedule; tuning budget zero for every arm (stated as a limit).
- **Changed variable:** the training objective (MTP module and its loss; $\lambda$ schedule fixed in advance). **Controlled:** Data-v0 and its hashes, tokenizer, preset, tokens per step, seeds {0, 1}, data order, the 256 fixed validation windows of Eval v0, software versions.
- **Comparison axis:** **equal training FLOPs** decides, because MTP adds FLOPs per token (+68% / +27%) without adding any at inference: an equal-tokens comparison gives MTP a larger budget and would credit it for compute, not for the objective. The baseline gets $\lceil 200 \cdot 19.18/11.41 \rceil = 336$ steps on CPU. Equal tokens is reported as secondary (it is the axis of the published ablations). Neither axis answers wall-clock or decode latency.
- **Budget:** free CPU about 30 minutes for 8 runs (measured below); main path PROJECTED below.
- **Metrics and decision rule:** held-out loss of the main head on 256 windows, seed-averaged, paired by window (95% bootstrap); per-seed differences; b0's seed std and the MDE. Rule, stated now: adopt MTP as an objective if, at equal FLOPs, the upper bound of the paired interval of `mtp-ds − b0@336` is below 0 and both per-seed differences are negative; reject if the lower bound is above 0; otherwise inconclusive — then the decision falls back to the simpler choice (no MTP objective) and the report says so. Secondary: acceptance rate (teacher-forced and generative), tokens per round.
- **Correctness checks:** `pytest labs/common/tests/test_blocks.py -k "mtp or speculative or deepseek or causal"` and `pytest labs/module-06/lesson-01` pass first; `python -m frontierlab.record runs/m06/cpu/b0/s0 runs/m06/cpu/mtp-ds/s0 --changed config.extra.blocks.mtp config.extra.blocks.mtp_depth` shows no other INVALIDATES line (budget and parameter differences are expected).
- **Fallback evidence:** a null or negative result at this scale is the expected outcome under Meta's small-model finding; cite Meta Figure 3 and DeepSeek Table 4 as the published evidence at scale, labelled as such.
- **Limits:** one tiny model, 0.41M training tokens, two seeds, no tuning of $\lambda$ or learning rate; results say nothing about ≥1B models.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100. Not run in this build; part of the Module 6 pilot | `train_arms.py --variant main` (`pilot-30m`, 4,000 steps × 64 × 1,024 = 262M tokens per arm; 4 arms × 2 seeds) and `--variant main70` (`pilot-70m`), then `compare.py --variant main --device cuda` and `acceptance.py --variant main --device cuda`. PROJECTED cost: training FLOPs per token at $T = 1024$ from `blocks.accounting`: $3.21 \times 10^8$ (next-token) and $4.47 \times 10^8$ (MTP) for pilot-30m; two seeds of b0, mtp-ds, mtp-meta and the equal-FLOPs b0 = $2 \cdot (3.21 + 4.47 + 4.44 + 4.47) \times 10^8 \cdot 2.62 \times 10^8$ tokens $\approx 8.7 \times 10^{17}$ FLOPs; at an assumed 25% MFU on 989 TFLOP/s that is $8.7 \times 10^{17} / (0.25 \cdot 989 \times 10^{12}) \approx 3{,}500$ s ≈ 1.0 GPU-hour; pilot-70m ($6.14$ and $7.75 \times 10^8$) ≈ 1.7 GPU-hours |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `train_arms.py --variant t4` (`pilot-10m`, 2,000 steps × 32 × 512), `--max-minutes 80` and rerun after disconnects |
| Free CPU | laptop; measured below | every step as written |

### Steps

1. **Implement** `mtp_chain`, `mtp_loss`, `lambda_schedule`, `accepted_prefix` and `mtp_extra_flops` in `lab.py`; run `pytest labs/module-06/lesson-01`. The causality test changes tokens from position 9 on and requires depth-$k$ logits before position $9 - k$ to stay identical: think about why the bound is $9 - k$ and not 9.
2. **Count:** `python labs/module-06/lesson-01/train_arms.py --print-only`. Check the three FLOP numbers and the equal-FLOPs step count against the worked example.
3. **Train** (unattended, resumable): `python labs/module-06/lesson-01/train_arms.py`.
4. **Compare:** `python labs/module-06/lesson-01/compare.py`. Fill in the table and apply the rule.
5. **Draft:** `python labs/module-06/lesson-01/acceptance.py`. Record both acceptance numbers, the lossless check and the projected speed-up.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, with another build job sharing the CPU, 2026-10-04):

| Arm (toy, 200 steps × 16 × 128 unless stated) | Held-out loss, seeds 0 / 1 | Training MFLOP/token | Wall-clock per run |
|---|---|---|---|
| b0 | 6.6857 / 6.6762 | 11.41 | 87–138 s |
| mtp-ds ($D = 1$, V3 λ schedule) | 6.6746 / 6.6630 | 19.18 | 133–286 s |
| mtp-meta ($n = 2$) | 6.8129 / 6.8310 | 18.98 | 154–211 s |
| b0, 336 steps (equal FLOPs to mtp-ds) | 6.3656 / 6.3158 | 11.41 | 133–165 s |

Noise floor: b0's seed std 0.0067 nats; MDE with two seeds per arm 0.019 nats (a rough estimate from two seeds). Paired by window, seed-averaged, 95% bootstrap:

- equal tokens: mtp-ds − b0 = **−0.0122 [−0.0155, −0.0089]**, per seed −0.0111, −0.0132 — a small gain, below the MDE; mtp-meta − b0 = **+0.1409 [+0.1366, +0.1452]** — Meta's parallel head hurts at this size, as Meta reports for small models.
- equal training FLOPs (the decision axis): mtp-ds − b0@336 = **+0.3281 [+0.3194, +0.3371]**, per seed +0.309, +0.347. Decision by the rule: **reject the MTP objective at this scale**.
- drafting (`acceptance.py`, seed 0): DeepSeek module — teacher-forced first-draft agreement 0.643 over 8,128 positions (0.595 where the main head's choice was right); real greedy speculative decoding on 8 prompts × 32 new tokens: output **identical** to greedy, 62 of 188 drafts accepted (**$p = 0.33$**), 1.32 tokens per verification round, PROJECTED speed-up $1.33/(1 + 0.25) = 1.06\times$ with the toy model's draft cost of one block in four. Meta head: agreement 0.414, $p = 0.34$, 1.33 tokens per round.
- runtimes: 8 training runs 22 minutes in all; `compare.py` 41 s; `acceptance.py` 31 s.

What to write about it. At equal tokens the DeepSeek module gives a gain smaller than the detectable effect; at equal FLOPs it loses badly, because 200 steps is the steep start of training, where 136 extra steps of next-token training are worth far more than a denser signal. Both facts fit the published picture — Meta finds MTP hurts small models, DeepSeek's gains are at 15.7B+ and compared at equal tokens — and neither says anything about MTP at scale. The draft numbers are the more transferable result: the module works as a draft (losslessly), its acceptance is far below DeepSeek's 85–90% at 671B, and teacher-forced agreement overstates generative acceptance (0.64 vs 0.33) because on a toy model both heads mostly predict the same frequent tokens. Run-card check: `python -m frontierlab.record runs/m06/cpu/b0/s0 runs/m06/cpu/mtp-ds/s0 --changed config.extra.blocks.mtp config.extra.blocks.mtp_depth` prints COMPARABLE (the wrapper's own `blocks.*` record shows as unclassified WARN lines; for seed replicates add `--changed config.extra.blocks.seed`, the seed MatFormer's sampler uses).

<details>
<summary>Hint for TODO 1</summary>

Keep `h` as the running hidden state: depth 1 starts from `h0`, depth 2 from depth 1's output. Slice before combining: depth $k$ uses `h[:, :S]` with `S = T - k` and `idx[:, k:k + S]`. The block takes positions `torch.arange(S)`.

</details>

<details>
<summary>Hint for TODO 5</summary>

`attn_acc.param_counts(cfg)["attention_per_layer"]` is a list (layers may differ); any entry works for Baseline-0. Do not forget that the module has *three* norms (`hnorm`, `enorm`, `norm`) and the block two more.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-01`.

</details>

## Common mistakes

- **Comparing at equal tokens only.** MTP spends 27–68% more training FLOPs per token here; at equal tokens it gets more compute. Report equal tokens if you like, but decide on equal FLOPs (or wall-clock).
- **Shifting the targets by $k$ instead of $k + 1$.** Depth $k$ already sees $t_{i+k}$; asking it to predict $t_{i+k}$ is copying. The loss collapses and the causality test fails.
- **Reporting the training loss.** It contains $\lambda \cdot \mathcal{L}_{\mathrm{MTP}}$, so it is larger for MTP arms and not comparable. Compare `per_token_loss` (the main head) on the fixed windows.
- **Forgetting the head.** At small width the shared output head, not the block, is most of MTP's cost; a FLOP estimate that counts only "one more layer" is off by 3× on the toy model.
- **Measuring acceptance with teacher forcing and calling it the decode number.** On real text the draft is conditioned on the *real* next token; in decoding it is conditioned on the model's own choice. They agree only where the model's choice was right — the script reports that subset separately and runs real speculative decoding on a few prompts.
- **A speculative decoder that changes the output.** Greedy speculative decoding must reproduce greedy decoding exactly; check equality before you count speed.

## References

- F. Gloeckle et al. (Meta), *Better & Faster Large Language Models via Multi-token Prediction*, sections 2, 3.1, 3.2. https://arxiv.org/abs/2404.19737
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, sections 2.2, 4.2, 4.5.1, 5.4.3. https://arxiv.org/abs/2412.19437
- Qwen, *Qwen3-Next-80B-A3B-Instruct* model card. https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Gemma Team, *Gemma 4 Technical Report*, section 2.6. https://arxiv.org/abs/2607.02770
- Shared code: `labs/common/frontierlab/blocks/mtp.py`, `model.py`, `accounting.py`, `train.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[06.2 · Residual-stream design: hyper-connections and mHC](lesson-02.md)
