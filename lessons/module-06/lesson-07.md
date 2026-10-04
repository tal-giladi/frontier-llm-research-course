---
id: "06.7"
module: 6
minutes: 30
practice_minutes: 40
prerequisites: ["06.1"]
objectives:
  - Explain masked diffusion language modelling as LLaDA defines it — forward masking, the 1/t-weighted loss and why it bounds the likelihood — and contrast its sampling cost with autoregressive decoding.
  - Implement the LLaDA loss terms, its lower-variance likelihood estimator and the low-confidence remasking schedule, and verify that both estimators have the same expectation.
  - Describe Coconut's continuous thoughts and recurrent-depth models, and compute what each costs per generated answer.
  - Train a tiny masked-diffusion model and report its likelihood bound next to an autoregressive loss without presenting them as the same quantity; state what Gemini Diffusion discloses and nothing more.
volatility: concept
sources:
  - title: "Nie et al. — Large Language Diffusion Models (LLaDA), sections 2.1–2.4, 3"
    url: https://arxiv.org/abs/2502.09992
  - title: "Hao et al. (Meta) — Training Large Language Models to Reason in a Continuous Latent Space (Coconut), sections 3–5"
    url: https://arxiv.org/abs/2412.06769
  - title: "Geiping et al. — Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach, sections 3–4"
    url: https://arxiv.org/abs/2502.05171
  - title: "Google DeepMind — Gemini Diffusion (model page: experimental demo, reported benchmarks and sampling speed)"
    url: https://deepmind.google/models/gemini-diffusion/
last_verified: "2026-10-04"
---

# 06.7 · Non-autoregressive and latent reasoning

Extension: Baseline-0 writes one token per forward pass, left to right, and does all its "thinking" either inside one pass or in tokens it writes out. Three lines of work change that. Masked diffusion models (LLaDA) generate by unmasking many positions in parallel over a few refinement steps. Coconut lets a model reason in its hidden state, feeding the last hidden vector back as the next input instead of a token. Recurrent-depth models loop a block of layers a variable number of times per token. This lesson builds LLaDA's objective and sampler at toy scale, checks its likelihood estimators exactly, and puts all three on one cost scale.

## Why this matters at a frontier lab

Decode speed and reasoning compute are where inference money goes (Module 15). Each idea here moves a different knob: diffusion trades many cheap sequential steps for a few expensive parallel ones; Coconut and recurrent depth spend compute on reasoning without emitting tokens. Google lists Gemini Diffusion as "our state-of-the-art, experimental text diffusion model", with a reported average sampling speed of 1,479 tokens per second excluding overhead (model page; company claim). The questions a lab asks are concrete: *what does each method cost per answer, what does its training objective actually optimise, and how would we evaluate it fairly against an autoregressive model?*

## The idea

### Masked diffusion (LLaDA)

LLaDA (section 2.1) trains a *mask predictor* $p_\theta(\cdot \mid x_t)$: a Transformer with no causal mask that sees a partially masked sequence and predicts every masked token at once.

- **Forward process:** draw $t \sim U[0, 1]$; replace each token of $x_0$ by the mask token $M$ independently with probability $t$.
- **Loss (Eq. 3):** $\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\big[\tfrac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = M] \log p_\theta(x_0^i \mid x_t)\big]$.
- **Bound (Eq. 4):** $-\mathbb{E}[\log p_\theta(x_0)] \le \mathcal{L}(\theta)$. The $1/t$ weight is what makes it a likelihood bound; with it, LLaDA is "a principled generative model"; MaskGIT's objective "misses the $1/t$ term" (section 2.1). BERT masks a fixed 15%; LLaDA's rate varies over $(0, 1]$.
- **Lower-variance estimate (Eq. 6):** draw $l \sim U\{1, \dots, L\}$, mask exactly $l$ positions uniformly, weight the masked cross-entropies by $L/l$. Same expectation as Eq. 3 (the lab checks it exactly by enumerating every mask of a 3-token sequence).

**Sampling (section 2.4):** start from a fully masked response of chosen length; at each of a fixed number of steps predict all masked tokens, keep the most confident, and remask the rest ("low-confidence remasking"). LLaDA cannot use a KV cache: every position attends to every other and the masked positions change at every step (section 2.2). So each step is a full forward pass over prompt + response; the gain is that a step can commit several tokens.

Scale as reported: LLaDA 8B, pretrained from scratch on 2.3T tokens "using 0.13 million H800 GPU hours", then SFT on 4.5M pairs (section 1); "competitive with strong LLMs like LLaMA3 8B in in-context learning" (abstract), and better than GPT-4o on a reversal poem-completion task (section 3.3).

### Continuous thoughts (Coconut)

Coconut (section 3) switches between "language mode" and "latent mode". In latent mode the last hidden state — a *continuous thought* — is fed back as the next input embedding instead of being decoded to a token. Training uses language chains of thought as a curriculum: stage $k$ replaces the first $k$ reasoning steps by $k \cdot c$ continuous thoughts; the loss is on the remaining tokens only. "We perform $n + 1$ forward passes when $n$ latent thoughts are scheduled" — training is sequential in the thoughts. Reported (GPT-2 base model): on GSM8k, 6 continuous thoughts give 34.1% vs 16.5% without chain of thought, but "Coconut does not surpass CoT on GSM8k"; on the planning-heavy ProsQA and on ProntoQA it beats CoT with fewer generated tokens; trained without the curriculum it does no better than no-CoT (section 5.3). The analysis reads a continuous thought as encoding several next steps at once, "akin to breadth-first search" (section 4).

### Recurrent depth

Geiping et al. (section 3) build a prelude $P$, a core block $R$ and a coda $C$; the core is iterated $r$ times on a state initialised from noise, re-injecting the prelude's output each time:

$$e = P(x), \quad s_0 \sim \mathcal{N}(0, \sigma^2 I), \quad s_i = R(A[s_{i-1}; e]), \quad p = C(s_r).$$

Training samples $r$ per step from a log-normal Poisson distribution with mean $\bar r = 32$ and backpropagates only through the last $k = 8$ iterations, so memory does not grow with $r$ (section 3.3). The reported model has shape $(l_P, l_R, l_C) = (2, 4, 2)$, hidden size 5,280, about 1.5B parameters in prelude and coda, 1.5B in the core and 0.5B in the tied embedding, trained on about 800B tokens (section 4); at test time it improves "up to a computation load equivalent to 50 billion parameters" (abstract).

### Gemini Diffusion — what is disclosed

The model page discloses: it is "experimental"; diffusion models "learn to generate outputs by refining noise, step-by-step" and can correct errors during generation; a benchmark table against Gemini 2.0 Flash-Lite (e.g. HumanEval 89.6% vs 90.2%, LiveCodeBench v6 30.9% vs 28.5%, GPQA Diamond 40.4% vs 56.5%, AIME 2025 23.3% vs 20.0%); and "sampling speed excluding overhead 1479 tokens / sec", "overhead 0.84 sec", averaged across the reported evals. Architecture, size, training data and hardware are not disclosed; anything beyond the page is INFERENCE.

## Worked example

### The 1/t weight, by hand

$L = 4$ tokens, $t = 0.5$, positions 2 and 4 masked with cross-entropies 2.0 and 3.0: Eq. 3's sample value is $\frac{1}{0.5}(2.0 + 3.0) = 10$ nats per sequence, 2.5 nats per token after dividing by $L$. At $t = 0.25$ one position is masked on average; the $1/t = 4$ weight scales one masked token's loss up to "per sequence". Without it, low-$t$ samples (most tokens visible, easy predictions) would dominate and the objective would no longer bound the likelihood.

### The two estimators agree

For a mask set $S$ of size $s$, Eq. 3 weights it by $\int_0^1 \frac{1}{t} t^s (1 - t)^{L - s}\,dt = \frac{(s-1)!\,(L-s)!}{L!}$; Eq. 6 by $P(l = s) \cdot P(S \mid l = s) \cdot \frac{L}{s} = \frac{1}{L} \cdot \binom{L}{s}^{-1} \cdot \frac{L}{s}$, the same number. For $L = 3$, $s = 1$: $\frac{0! \, 2!}{3!} = \frac{1}{3}$ and $\frac{1}{3} \cdot \frac{1}{3} \cdot 3 = \frac{1}{3}$.

### Steps per answer

A 256-token answer: autoregressive decoding takes 256 steps, each over one new token with a cache. LLaDA with 64 sampling steps takes 64 full passes over prompt + 256 positions, committing 4 tokens per step. With a 512-token prompt that is $64 \cdot 768 = 49{,}152$ token-positions processed against about $256$ for cached AR decoding plus the prompt's 512 once — fewer *sequential* steps, far more total compute; whether it is faster depends on whether decode was latency-bound (Module 15). Coconut with $k = 6$ thoughts adds 6 decode steps. A recurrent-depth model at $r = 32$ runs $2 + 4 \cdot 32 + 2 = 132$ layers per token: about $2 \cdot (1.5 + 32 \cdot 1.5) = 99$ GFLOPs per token, the "50B-parameter" equivalent.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| clean tokens $x_0$, masked $x_t$ | (B, L) | int64 | GPU / CPU |
| mask rates $t$ | (B,) | fp32 | same |
| mask predictor logits | (B, L, V+1) (one extra id for $M$) | fp32 for the loss | same |
| continuous thought | (B, 1, C) per thought | the model's dtype | same |
| recurrent state $s_i$ | (B, T, C) | same | same; only the last $k$ iterations kept for backward |

Training cost: a masked-diffusion step costs the same forward/backward as an AR step of the same model, but bidirectional attention scores cost 2× the causal ones (every query sees all $L$ keys; `blocks.accounting` counts it). It learns from only the masked positions — on average half of them.

## Build it

`labs/common/frontierlab/blocks/diffusion.py`: the attention kind `"gqa-bidir"` (Baseline-0's GQA without a mask, no cache), `mask_tokens`, `diffusion_loss_terms`, `DiffusionLM` (a BlockLM whose last vocabulary id is the mask; in training the masks come from torch's RNG, which the loop saves, so exact resume holds — tested), `nll_bound` (Eq. 6), `exact_bound` (enumeration of Eqs. 3 and 6) and `sample` (low-confidence remasking). `latent.py`: `continuous_thoughts` (Coconut's latent mode over the decode cache, using `BlockLM(inputs_embeds=...)`) and `RecurrentDepthLM` with `sample_recurrence` and truncated backpropagation.

```bash
python -m frontierlab.blocks.train --objective diffusion --run runs/m06/try-diff --preset toy --steps 200 \
    --batch 16 --seq 128 --lr 1.5e-3
```

Correctness checks, passing in `test_blocks.py`: Eq. 3 and Eq. 6 have exactly the same expectation by enumeration (difference below $10^{-10}$) and the Monte-Carlo `nll_bound` agrees within 5%; the bidirectional model *fails* the causal check, as it must, while every autoregressive variant passes; the mask rate matches $t$; the sampler leaves no mask; $k$ continuous thoughts over the cache equal a full recompute with the same input embeddings ($10^{-10}$); truncated backpropagation with $k \ge r$ gives exactly the full gradient and with $k < r$ a different one; the effective depth of a $(1, 2, 1)$ model at $r = 32$ is 66.

## What the evidence says

- **Masked diffusion LMs — PROMISING.** LLaDA (PUBLICLY DOCUMENTED, 8B, one lab) is competitive with same-size AR baselines on many benchmarks; its authors' AR baselines were "self-constructed", and the comparison is at their data and scale. Gemini Diffusion's numbers are a company claim without architecture disclosure.
- **Continuous thoughts (Coconut) — PROMISING, small scale.** GPT-2-sized evidence on synthetic logic and GSM8k; the method needs a language-CoT curriculum, and training is sequential in the number of thoughts.
- **Recurrent depth — PROMISING, one model.** A 3.5B proof of concept; gains concentrate on reasoning tasks (GSM8k improves with $r$, OpenBookQA saturates early; Figure 1).
- **Open questions:** evaluation parity (likelihood bounds vs exact NLL; reasoning compute matched in FLOPs), serving support (no KV cache for diffusion; variable depth per token), and whether these mix with the rest of a production stack.

## Lab

**Folder:** [`labs/module-06/lesson-07/`](../../labs/module-06/) · **Time:** about 40 minutes · **Pass check:** `pytest labs/module-06/lesson-07` passes; your notes contain the two held-out numbers *labelled as what they are*, one diffusion sample and one AR sample, and the steps-per-answer arithmetic for a 256-token answer with your own prompt length.

This lab trains one model and reports two numbers that are not on the same axis; it decides nothing, so it has no experiment contract. Its point is evaluation hygiene: an AR loss and a diffusion bound look alike (nats per token) and are not interchangeable.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | optional; not run in this build | `nonar.py --variant main --device cuda` (pilot-30m, 4,000 steps) |
| Free GPU (Colab/Kaggle T4) | T4 | `nonar.py --variant t4 --device cuda` |
| Free CPU | laptop; measured below | as written |

### Steps

1. **Implement** `masked_terms`, `eq6_estimate`, `commit_count` and `effective_depth`; run `pytest labs/module-06/lesson-07`.
2. **Run** `python labs/module-06/lesson-07/nonar.py` (trains the diffusion model; reuses Baseline-0 from 06.1).
3. **Write** what each held-out number measures, why the bound is expected to be higher at this scale and training length, and what a fair comparison would need (same tokens *and* the same quantity: an AR model's NLL of all $T$ tokens, or both models' bounds).

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, with another build job sharing the CPU, 2026-10-04):

| Measurement | Result |
|---|---|
| Baseline-0, next-token loss (exact NLL of tokens 2..128), 256 windows | **6.6857** nats/token |
| masked diffusion, Eq. 6 upper bound on the NLL of all 128 tokens, 8 draws | **7.3223** nats/token |
| diffusion sample after a 16-token prompt, 24 tokens in 8 steps (3 committed per step) | 24 periods: `........................` |
| Baseline-0 greedy, 24 tokens | `, and the the same a the same a the same is the same is ...` |
| runtime | `nonar.py` 1.9 minutes (one diffusion training run plus evaluation; Baseline-0 reused) |

What to write about it. The diffusion model's bound is 0.64 nats above the AR loss — but the two are different quantities (a bound over all positions vs an exact conditional NLL), and the diffusion model learned from only the masked half of the positions in 200 steps. Neither sample is text yet; the diffusion one shows a typical failure of an undertrained mask predictor with confidence-ordered unmasking — the most frequent token wins every position. The arithmetic is the transferable part: 8 full passes over 40 positions for 24 tokens here, versus 24 cached decode steps.

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-07/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-07`.

</details>

## Common mistakes

- **Dropping the $1/t$ weight.** The objective then over-weights easy, lightly masked samples and is no longer a likelihood bound (LLaDA's point against MaskGIT).
- **Comparing a diffusion bound with an AR loss as if they were the same number.** One is an upper bound on all $L$ tokens, the other an exact NLL of $L - 1$ tokens given the first.
- **Counting diffusion sampling steps as decode steps.** Each step is a full pass over the whole sequence without a cache; compare compute and latency, not step counts.
- **Training Coconut without the curriculum.** The paper's own ablation finds it no better than no chain of thought.
- **Reading recurrent-depth FLOPs from the parameter count.** A 3.5B model at $r = 32$ costs like a ~50B one per token.
- **Attributing an architecture to Gemini Diffusion.** Google discloses benchmarks and speed, not design.

## References

- S. Nie et al., *Large Language Diffusion Models*, sections 1, 2.1–2.4, 3.3. https://arxiv.org/abs/2502.09992
- S. Hao et al. (Meta), *Training Large Language Models to Reason in a Continuous Latent Space*, sections 3–5. https://arxiv.org/abs/2412.06769
- J. Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, sections 3–4. https://arxiv.org/abs/2502.05171
- Google DeepMind, *Gemini Diffusion*. https://deepmind.google/models/gemini-diffusion/
- Shared code: `labs/common/frontierlab/blocks/diffusion.py`, `latent.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

The module project: [Lineage-F integration experiment](../../projects/module-06-lineage-f.md). Module 7 asks which optimizer and parametrization to train the chosen architecture with: [07.1 · Muon from scratch](../module-07/lesson-01.md).
