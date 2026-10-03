---
id: "03.3"
module: 3
minutes: 40
practice_minutes: 75
prerequisites: ["03.2", "01.4"]
objectives:
  - Explain how attention logits grow, why that destabilises training, and how QK-norm bounds them, with a numerical bound for a given head width.
  - Explain attention sinks and massive activations, and implement a softmax with a learned per-head sink logit (gpt-oss form) that passes the correctness suite.
  - Implement head-specific sigmoid output gating after attention (Qwen gated attention) and state where the gate sits and what it costs in parameters.
  - Measure first-token attention mass, learned-sink mass, maximum logit and massive-activation ratio on trained models, and report them as course-scale observations about stated hypotheses, not as confirmations of published claims.
volatility: concept
sources:
  - title: "Xiao et al. — Efficient Streaming Language Models with Attention Sinks (StreamingLLM)"
    url: https://arxiv.org/abs/2309.17453
  - title: "Sun et al. — Massive Activations in Large Language Models"
    url: https://arxiv.org/abs/2402.17762
  - title: "Qiu et al. (Qwen) — Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (section 2.2, Table 1, Table 4, Figure 2)"
    url: https://arxiv.org/abs/2505.06708
  - title: "gpt-oss-120b & gpt-oss-20b Model Card (section 2.2: learned bias in the softmax denominator)"
    url: https://arxiv.org/abs/2508.10925
  - title: "Dehghani et al. — Scaling Vision Transformers to 22 Billion Parameters (QK-norm against logit growth)"
    url: https://arxiv.org/abs/2302.05442
  - title: "Gemma 3 Technical Report (QK-norm replaces Gemma 2's soft-capping)"
    url: https://arxiv.org/abs/2503.19786
  - title: "Team OLMo — 2 OLMo 2 Furious (QK-norm for training stability)"
    url: https://arxiv.org/abs/2501.00656
last_verified: "2026-10-03"
---

# 03.3 · Logit control, sinks and gating

Softmax attention has two habits that matter in practice. Its logits can grow without bound as training proceeds, which saturates the softmax and destabilises training; and because each row must sum to one, heads that have nothing useful to attend to dump their probability on a few tokens — usually the first — which then carry enormous activations. This lesson covers three controls that recent models use: QK-norm (Baseline-0, Gemma 3, OLMo 2, Qwen3), a learned "sink" logit per head (gpt-oss), and a sigmoid gate on each head's output (Qwen's gated attention). You implement the sink and the gate, run them through the correctness suite, and measure sinks, logits and massive activations on small trained models — stating plainly what a course-scale experiment can and cannot show about claims made at billions of parameters.

## Why this matters at a frontier lab

A loss spike at step 200,000 of a frontier run costs days. Growing attention logits are one of the documented causes; Kimi K2 reports logits above 1,000 without its clipping (Module 7), and Dehghani et al. added QK-norm to train a 22B vision transformer stably. Attention sinks and massive activations are less dramatic but just as practical: they decide whether a model survives having its first tokens evicted from a sliding cache (StreamingLLM), they create the outliers that make low-precision inference hard (Module 8), and they confuse interpretability analyses that read attention patterns as meaning. A research engineer needs to know which control a model uses, what it costs, how to measure the behaviour it targets, and how weak the evidence from a small experiment is.

## The idea

### Logit growth and QK-norm

A head's logit is $\ell_{ij} = q_i \cdot k_j / \sqrt{d}$ with $q_i = W_Q x_i$, $k_j = W_K x_j$. Nothing in the loss stops $\|W_Q\|$ and $\|W_K\|$ from growing; the logit scales with their product. Large logits make the softmax nearly one-hot: its gradient $\partial p_i / \partial \ell_j = p_i(\delta_{ij} - p_j)$ vanishes for every entry except one, and small parameter changes swing which entry wins — a recipe for spikes.

**QK-norm** applies RMSNorm to each head's query and key before RoPE (Baseline-0's `q_norm`, `k_norm`, with learned gains $g_q, g_k \in \mathbb{R}^d$):

$$\hat q = g_q \odot \frac{q}{\mathrm{rms}(q)}, \qquad \hat k = g_k \odot \frac{k}{\mathrm{rms}(k)}, \qquad \mathrm{rms}(u) = \sqrt{\tfrac{1}{d}\textstyle\sum_c u_c^2}$$

Each normalised vector (before its gain) has Euclidean norm exactly $\sqrt d$, and RoPE is a rotation, so by Cauchy–Schwarz

$$|\ell_{ij}| \le \frac{\|g_q\|_\infty \sqrt d \cdot \|g_k\|_\infty \sqrt d}{\sqrt d} = \|g_q\|_\infty \|g_k\|_\infty \sqrt d .$$

The logit is bounded by the gains, which grow slowly, instead of by the weight norms. Gemma 3 "replace[s] the soft-capping of Gemma 2 with QK-norm"; soft-capping ($\ell \mapsto c \tanh(\ell / c)$) bounds logits too but bends every large score. Kimi K2's QK-Clip (Module 7) rescales the weights after an update instead.

### Attention sinks and massive activations

Each softmax row sums to one: "the nature of the SoftMax function prevents all attended tokens from having zero values" (StreamingLLM). A head whose pattern is irrelevant for the current token must still put its weight somewhere, and trained models learn to put it on a token whose value vector does little — typically the first token of the sequence, whatever it is. Xiao et al. call such tokens **attention sinks**: window attention that evicts them "collapses once the sequence length exceeds the cache size, i.e., even just evicting the KV of the first token", and keeping four initial tokens plus a rolling window restores it. Sun et al. find that a few hidden-state activations become "100,000 times larger" than the rest, on specific tokens and channels, stay nearly constant across inputs, and act as bias terms that drive this attention concentration.

### A learned sink logit (gpt-oss)

Give each head $h$ a learned scalar $s_h$ that joins the denominator and has no value:

$$p_{ij} = \frac{e^{\ell_{ij}}}{e^{s_h} + \sum_{j'} e^{\ell_{ij'}}}, \qquad y_i = \sum_j p_{ij} v_j, \qquad \sum_j p_{ij} = 1 - \frac{e^{s_h}}{e^{s_h} + \sum_{j'} e^{\ell_{ij'}}} .$$

A head can now attend to "nothing" ($y_i \to 0$) without needing a token to dump on. gpt-oss: "Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks, which enables the attention mechanism to pay no attention to any tokens" (model card section 2.2). StreamingLLM's pretraining variant uses a learnable placeholder *token* instead, and reports that pretraining with such a sink token gives stable streaming perplexity with fewer initial tokens kept.

### Head-specific output gating (Qwen)

Qiu et al. multiply each head's attention output by a sigmoid gate computed from the same normalised hidden state $X$ that produced the query ("We adopt the hidden states after pre-normalization as X", section 2.2), at position G1 — after scaled dot-product attention, before the output projection:

$$Y' = Y \odot \sigma(X W_\theta)$$

Their best variant is head-specific and elementwise ($W_\theta$ of shape $C \times H d$); a headwise variant uses one scalar per head ($C \times H$). They give two reasons it helps: it adds a non-linearity between $W_V$ and $W_O$, whose product is otherwise a low-rank linear map per head, and the gate is query-dependent and sparse (mean gate score 0.116 for the elementwise SDPA gate, Table 4), so a head can switch itself off for a token — which removes the need for a sink.

## Worked example

### The QK-norm bound

Baseline-0's heads have $d = 64$. With gains at their initial value 1, $|\ell| \le \sqrt{64} = 8$: the largest probability ratio between two keys is $e^{16} \approx 8.9 \times 10^6$ — peaked, but finite and controlled. If the gains grow to 2, the bound is $4 \cdot 8 = 32$. Without QK-norm, $q$ and $k$ of norm 30 each (plausible for unconstrained weights late in training) allow $|\ell|$ up to $900/8 = 112.5$.

### A sink by hand

One head, one query, two keys with logits $\ell = (\ln 3, 0)$ and a sink $s = 0$: $e^{\ell} = (3, 1)$, $e^{s} = 1$, denominator $5$. Probabilities $(0.6, 0.2)$ and sink mass $0.2$; the output is $0.6 v_1 + 0.2 v_2$, 20% shorter than without the sink. With $s = -\infty$ the sink vanishes and the row is the ordinary $(0.75, 0.25)$ — our test checks that a sink of $-10^4$ reproduces Baseline-0's logits exactly.

### A gate by hand

A head outputs $y = (2, -1)$; the gate input gives $X W_\theta = (0, -3)$ for its two channels, so $\sigma = (0.5, 0.047)$ and $y' = (1.0, -0.047)$. A gate that learns large negative inputs for a token switches that head off for it, as a sink would, but per token and per channel, and without consuming probability mass.

### First-token mass under uniform attention

If every query at position $t$ spread its attention uniformly over its $t + 1$ keys, the mass on key 0 would be $1/(t + 1)$. Averaged over $t = 8 \ldots 127$ (what the lab's probe uses) that is $(H_{128} - H_8)/120 = 0.023$, with $H_n$ the $n$-th harmonic number. A measured value far above this baseline is a sink.

## Shapes and cost

| Piece | Parameters per layer (Baseline-0 width) | Shape | Extra FLOPs per token |
|---|---|---|---|
| QK-norm gains | $2d = 128$ | (64,) each | $O(Hd)$, negligible |
| learned sinks (`sinks`) | $H = 12$ | (12,) | one extra logit per (query, head) |
| elementwise gate (`gate_proj`) | $C \cdot Hd = 589{,}824$ | (768, 768) | $2 C H d \approx 1.2$M (+0.15% of 788M) |
| headwise gate | $C H = 9{,}216$ | (768, 12) | $2CH$, negligible |

The elementwise gate adds 7.1M parameters to Baseline-0 (+7.3% non-embedding) — the paper's Table 1 gives 201M added parameters for its elementwise G1 gate against 1.6M for headwise, at its model sizes. An equal-parameters comparison must shrink something else, as the project does for MLA.

Tensor shapes inside the sink and gated kinds: queries (B, H, T, d), cached keys/values (B, K, S, d); logits (B, H, T, S) concatenated with the sink to (B, H, T, S + 1) before the softmax, the sink column dropped after it; gate (B, H, T, d) or (B, H, T, 1), multiplied into the attention output (B, H, T, d). dtype: the softmax runs in at least float32 even under BF16 autocast (`ops.sink_softmax`). The sink path cannot use `F.scaled_dot_product_attention` (which has no sink input), so it materialises the (T, S) logits: memory $O(T S)$ per head, which our CPU runs can afford; at scale you need an attention kernel that supports a sink term (check what your serving engine supports before choosing this design).

## Build it

`labs/common/frontierlab/attention/gated.py` registers two kinds, each Baseline-0's GQA with QK-norm plus one mechanism:

- `"sink"` — `self.sinks`, an `nn.Parameter` of shape (H,), initialised to 0, used by `ops.attend(..., sink=self.sinks)`; combine with `extra["window"]`, or use `local_global` with `extra["sinks"] = True` for gpt-oss's layout;
- `"gated"` — `self.gate_proj`, `nn.Linear(C, H·d)` (elementwise, default) or `nn.Linear(C, H)` (`extra["gate"] = "headwise"`); `forward` computes `heads(x) * sigmoid(gate_proj(x))` before `o_proj`.

Both pass the correctness suite (op-level float64 gradient check with respect to the input and every parameter, including the sink logits and gate weights; causal check; cached decode in steps of 1, 3 and 7 tokens). Two limit tests tie them to Baseline-0: a sink of $-10^4$ gives Baseline-0's logits to $10^{-12}$, and a gate with zero weights (every gate $= 0.5$) gives exactly Baseline-0 with its output projection halved. Weight decay: the loop puts 1-D parameters (the sinks, all norm gains) in the no-decay group.

The probes (`frontierlab/attention/probes.py`) recompute each layer's logits and probabilities from its own projections on fixed inputs, through forward pre-hooks, so they work for every kind without changing the model:

```python
from frontierlab.attention import probes
rep = probes.report(model, idx)        # idx (8, 128): fixed validation windows
rep["first_token_mass"], rep["sink_mass"], rep["max_logit"], rep["massive_ratio"]
```

## What the evidence says

- **QK-norm — ESTABLISHED.** Used by Gemma 3, OLMo 2, Qwen3 and GLM-4.5 (PUBLICLY DOCUMENTED in their reports or configs), introduced at scale for ViT-22B. Its stability benefit is well supported at large scale; at our scale logits are small either way, so the lab tests the *mechanism* (bounded logits at a raised learning rate), not the stability claim.
- **Learned sinks — PROMISING / MODEL-SPECIFIC.** gpt-oss is the main production example, documented in one sentence of its model card; StreamingLLM's sink-token pretraining experiments are evidence for the idea.
- **Output gating — PROMISING.** Qiu et al. report, for a 15B MoE model (2.54B activated) trained on 3.5T tokens, that the gate cut the average attention on the first token from 46.7% to 4.8% (Figure 2) and the maximum activation from 1,053 to 94 (Table 4), improved perplexity and benchmarks, and allowed larger learning rates; Qwen3-Next uses gated attention in its full-attention layers (model card). Independent replication at other labs is limited so far.
- **Course-scale hypotheses** (lab): (H1) trained toy models show a first-token mass well above the uniform baseline of 0.023; (H2) the sink and gated arms show lower first-token mass than Baseline-0; (H3) without QK-norm the maximum logit is larger, and grows more at a raised learning rate. Each may not appear at 1.8M parameters and 0.6M tokens. If they do not, the conclusion is "not shown at this scale", and the pilot's 30M/70M traces are the fallback evidence.

## Lab

**Folder:** [`labs/module-03/lesson-03/`](../../labs/module-03/) · **Time:** about 75 minutes (20 of them unattended) · **Pass check:** `pytest labs/module-03/lesson-03` passes; `probe.py` runs on your six runs; your notes state, for each hypothesis, "observed", "not observed" or "not shown at this scale", with the numbers.

### Experiment contract

- **Question:** at toy scale, do learned sinks or output gates reduce attention sinks and massive activations relative to Baseline-0, and does removing QK-norm raise the maximum logit, more so at a raised learning rate? Decision informed: which probes the module project and Module 7 should log, and whether these effects are visible enough at course scale to use as lab outcomes.
- **Hypotheses and status:** H1–H3 above; reported effects at 1.7B–15B (Qwen) and in large models (Sun et al.; StreamingLLM); may not appear at this scale.
- **Baseline:** `b0` (toy preset), learning rate 3e-3, 300 steps; not tuned further. Every arm gets the same zero tuning budget.
- **Changed variable:** one mechanism per arm (`no-qknorm`, `sink`, `gated`); for H3, the learning rate (×3.3, arms `b0@hi`, `no-qknorm@hi`). **Controlled:** Data-v0 at the CPU size, 300 steps × 16 × 128 tokens, seed 0 (same initialisation draw order and data order), the same probe windows and the same 256 held-out windows.
- **Comparison axis:** equal tokens. Parameters differ slightly (sinks +48; elementwise gate +65,536, +8.3% non-embedding at toy size): say so when you compare losses.
- **Budget:** free CPU, 6 runs, 23 minutes measured; main path 1× H100 or A100, PROJECTED $3.2 \times 10^8$ FLOPs per token (`accounting.flops_per_token`, `pilot-30m`, $T = 1024$) × $4{,}000 \times 64 \times 1{,}024 = 2.6 \times 10^8$ tokens $= 8.4 \times 10^{16}$ FLOPs per run; at an assumed 20% MFU on an H100 SXM, $8.4 \times 10^{16} / (0.2 \cdot 989 \times 10^{12}) / 3600 \approx 0.12$ GPU-hours per run, under 1 GPU-hour for 6 runs (the sink arm's explicit softmax will run at lower MFU).
- **Metrics and decision rule:** first-token mass (mean over layers, heads and queries $t \ge 8$), learned-sink mass, maximum logit, massive-activation ratio, held-out loss over 256 windows with a paired bootstrap against `b0`. Rules stated now: H1 observed if every trained arm's first-token mass is at least 3× the uniform baseline; H2 observed if the sink or gated arm's first-token mass is at most half of `b0`'s; H3 observed if `no-qknorm@hi`'s maximum logit is at least 2× `b0@hi`'s. One seed, so each is an observation on one run, not an estimate of an effect.
- **Correctness checks:** `pytest labs/common/tests/test_attention_m03.py -k "sink or gated"` and `pytest labs/module-03/lesson-03` pass before any probe counts.
- **Fallback evidence:** the Module 3 pilot's probe traces at 30M and 70M, labelled as analysis of provided traces.
- **Limits:** tiny model, short training, one seed, $T = 128$; sinks and massive activations in the literature are reported in models trained far longer.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 SXM or A100, under 1 GPU-hour (PROJECTED, see the contract). Not run in this build; part of the Module 3 pilot | `python labs/module-03/lesson-03/train_arms.py --variant main`, then `python labs/module-03/lesson-03/probe.py runs/m03/l33/main --device cuda` |
| Free GPU (Colab/Kaggle T4) | T4, fp32, `pilot-10m`, 2,000 steps | `train_arms.py --variant t4 --max-minutes 80` (rerun after disconnects), then `probe.py runs/m03/l33/t4 --device cuda` |
| Free CPU | laptop; 23 minutes for the 6 runs, 41 seconds for `probe.py` (measured 2026-10-03, 16 threads) | the steps below as written |

### Steps

1. **Implement** `sink_softmax`, `first_token_mass`, `max_logit` and `massive_ratio` in `lab.py`; run `pytest labs/module-03/lesson-03`.
2. **Train** (unattended): `python labs/module-03/lesson-03/train_arms.py`. While it runs, write down your prediction for H1–H3.
3. **Probe:** `python labs/module-03/lesson-03/probe.py runs/m03/l33/cpu`. The script uses your functions where they are implemented.
4. **Report** each hypothesis with its numbers and the rule's verdict, and one paragraph on why the gated arm's loss difference against `b0` is not a test of the gate (count its parameters).

Measured in this build (free CPU, Windows 11, 16 threads, torch 2.14.1+cpu, toy preset, seed 0):

| Run | Learning rate | Held-out loss | First-token mass | Learned-sink mass | Max logit | Massive ratio |
|---|---|---|---|---|---|---|
| `b0` | 3e-3 | 6.302 | 0.013 | 0 | 6.79 | 18.6 |
| `no-qknorm` | 3e-3 | 6.258 | 0.015 | 0 | 14.17 | 22.9 |
| `sink` | 3e-3 | 6.287 | 0.014 | 0.011 | 6.99 | 17.2 |
| `gated` | 3e-3 | 6.217 | 0.014 | 0 | 7.06 | 10.7 |
| `b0@hi` | 1e-2 | 6.409 | 0.012 | 0 | 11.22 | 30.7 |
| `no-qknorm@hi` | 1e-2 | 6.709 | 0.014 | 0 | 172.80 | 23.8 |

Uniform attention would give a first-token mass of 0.023 (8 windows × 128 tokens, queries $t \ge 8$). Paired held-out-loss differences over 256 windows against `b0` at the same learning rate (one seed, so evaluation noise only): `gated` −0.085 [−0.091, −0.079], `no-qknorm` −0.044 [−0.049, −0.039], `sink` −0.014 [−0.019, −0.009], `no-qknorm@hi` +0.300 [+0.287, +0.313]. Training the six runs took 23 minutes, `probe.py` 41 seconds.

Verdicts by the rules. **H1 not observed:** every arm's first-token mass (0.012–0.015) is *below* the uniform baseline; at 1.8M parameters, 300 steps and windows that start mid-document, these models attend locally and have formed no sink. **H2 not testable:** there is no sink to remove; the learned sink took 1.1% of the probability mass, a small "attend to nothing" option the model barely used. **H3 observed:** without QK-norm at the raised learning rate the maximum logit reached 172.8 against 11.2 with QK-norm (15×), and that run's loss was 0.30 nats worse — the only arm that clearly degraded. Two observations the rules did not ask about: at the default learning rate the `no-qknorm` run had a *lower* loss than `b0` (one seed: a reason to measure, not a conclusion), and the gated run had the lowest massive-activation ratio (10.7 against 18.6) and the lowest loss, with 8.3% more parameters. The consistency check of the QK-norm bound holds: with $d = 32$ the bound is $\sqrt{32}\, g_q g_k = 5.66\, g_q g_k$, so `b0`'s 6.79 implies a gain product of at least 1.2, and `b0@hi`'s 11.22 at least 2.0 — the gains, not the weights, grew.

<details>
<summary>Hint for step 1</summary>

Concatenate the sink as an extra column, subtract the row maximum (it can be the sink) before `exp`, normalise, and drop the last column. For `max_logit`, select the finite entries first: masked positions are $-\infty$.

</details>

<details>
<summary>Hint for step 4</summary>

At toy size the elementwise gate adds 16,384 parameters per layer to a 0.79M-parameter model. A lower loss could come from capacity, not from the gating mechanism; an equal-parameters arm (headwise gate, or a narrower SwiGLU) would separate the two.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-03/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-03/lesson-03`.

</details>

## Common mistakes

- **Softmax with a sink but without subtracting the maximum.** Logits of 1,000 overflow `exp` in float32; include the sink in the row maximum.
- **Forgetting that sink probabilities do not sum to one.** Code that renormalises rows "to be safe" removes the mechanism.
- **Putting the gate after the output projection.** The paper's best position is after attention and before $W_O$, per head; the paper's Table 1 compares positions, and G1 is the one it recommends.
- **Reading attention to the first token as meaning.** A sink is where idle heads park probability, not evidence that the first token matters.
- **Calling a one-seed toy run a replication.** Report what you saw, at what scale, and the published claim it does or does not echo.

## References

- G. Xiao et al., *Efficient Streaming Language Models with Attention Sinks*. https://arxiv.org/abs/2309.17453
- M. Sun et al., *Massive Activations in Large Language Models*. https://arxiv.org/abs/2402.17762
- Qiu et al. (Qwen), *Gated Attention for Large Language Models*, section 2.2, Tables 1 and 4, Figure 2. https://arxiv.org/abs/2505.06708
- OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, section 2.2. https://arxiv.org/abs/2508.10925
- M. Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*. https://arxiv.org/abs/2302.05442
- Gemma Team, *Gemma 3 Technical Report*. https://arxiv.org/abs/2503.19786
- Team OLMo, *2 OLMo 2 Furious*. https://arxiv.org/abs/2501.00656
- Shared code: `labs/common/frontierlab/attention/gated.py`, `ops.py`, `probes.py`.

## Next

[03.4 · Head count and head dimension](lesson-04.md) (extension), or go straight to the module project: [Attention memo](../../projects/module-03-attention-memo.md).
