---
id: "17.1"
module: 17
minutes: 40
practice_minutes: 75
prerequisites: ["01.4"]
objectives:
  - Explain superposition with the toy model of Elhage et al., and measure how many features a 5-dimensional model represents as feature sparsity grows.
  - Implement ReLU (L1), TopK and JumpReLU (L0, straight-through) sparse autoencoders from scratch and check them against hand-worked values.
  - Evaluate an SAE by FVU, L0, dead latents and the increase in next-token loss when it is spliced into the model, with a paired interval, and run the splice check first.
  - Compare the three SAE kinds at matched sparsity and say which numbers support a choice and which do not.
volatility: concept
sources:
  - title: "Elhage et al. — Toy Models of Superposition (Anthropic, 2022): sections Demonstrating Superposition, Superposition as a Phase Change, The Geometry of Superposition"
    url: https://transformer-circuits.pub/2022/toy_model/index.html
  - title: "Bricken et al. — Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Anthropic, 2023)"
    url: https://transformer-circuits.pub/2023/monosemantic-features/index.html
  - title: "Templeton et al. — Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (Anthropic, 2024)"
    url: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
  - title: "Gao et al. — Scaling and evaluating sparse autoencoders (OpenAI, 2024): sections 2.1-2.4, 4, appendix A"
    url: https://arxiv.org/abs/2406.04093
  - title: "Rajamanoharan et al. — Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders (2024): Eq. 4, Eqs. 11-12"
    url: https://arxiv.org/abs/2407.14435
  - title: "Lieberum et al. — Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (2024)"
    url: https://arxiv.org/abs/2408.05147
  - title: "Gemma Scope 2 (Google DeepMind, December 2025): SAEs and transcoders for Gemma 3"
    url: https://huggingface.co/google/gemma-scope-2
  - title: "Qwen-Scope SAEs for Qwen3-1.7B-Base (TopK, 32K latents, k = 50), revision ce1a79d"
    url: https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
  - title: "SAELens v6.53.0 (pretrained_saes.yaml lists the Qwen-Scope releases)"
    url: https://github.com/decoderesearch/SAELens/blob/v6.53.0/sae_lens/pretrained_saes.yaml
last_verified: "2026-10-07"
---

# 17.1 · Features and sparse autoencoders

A transformer's residual stream has a few thousand dimensions and the model appears to use far more concepts than that. This lesson starts from the toy model that explains how — superposition, many sparse features sharing few dimensions — and then builds the tool the field uses to pull those features back out: the sparse autoencoder. You implement the three kinds in current use (ReLU with an L1 penalty, TopK, and JumpReLU with an L0 penalty), train them on a small language model's residual stream, and judge them the way a lab must: not by how well they reconstruct activations, but by how much next-token loss the model loses when the reconstruction replaces the real activation.

## Why this matters at a frontier lab

Sparse autoencoders are the main source of named features in published interpretability work: Anthropic's 34-million-feature dictionary on Claude 3 Sonnet, OpenAI's 16-million-latent dictionary on GPT-4, Google DeepMind's Gemma Scope, and Qwen-Scope for Qwen3. Features from them are used to explain behaviours, build monitors and steer models (lessons 17.3–17.4). Every one of those uses inherits the SAE's errors. An SAE that reconstructs 98% of the variance can still remove the 2% the model depends on, and then any "the model uses feature X" claim is a claim about a damaged model. The habit this lesson builds — splice it in, measure the loss, report the interval — is what separates a dictionary you can reason with from a pretty list of features.

## The idea

### Superposition

Suppose a model must track $n$ features in $m < n$ dimensions. If features are **dense** (often active together) it can only store about $m$ of them without interference, so it keeps the most important ones and drops the rest. If features are **sparse** (each rarely active), two features can share a direction with little cost, because they are rarely active at the same time. Elhage et al. (2022) show this with a toy model:

$$h = W x, \qquad x' = \text{ReLU}(W^\top h + b), \qquad L = \mathbb{E}_x \sum_{i=1}^{n} I_i (x_i - x'_i)^2$$

Symbols: $x \in \mathbb{R}^n$ the features, each 0 with probability $S$ (the sparsity) and otherwise uniform on $[0, 1]$; $W \in \mathbb{R}^{m \times n}$ the embedding, column $W_i$ the direction of feature $i$; $b \in \mathbb{R}^n$ a bias; $I_i$ the importance of feature $i$. The ReLU lets the model filter out small interference. Their finding ("Superposition as a Phase Change"): sparsity is necessary for superposition, and whether a feature is stored in superposition changes abruptly as sparsity and importance vary. The features organise into geometric structures — antipodal pairs, triangles, pentagons, tetrahedra ("The Geometry of Superposition").

Two measures from the paper: **dimensions per feature** $D^* = m / \lVert W \rVert_F^2$, and each feature's **dimensionality**

$$D_i = \frac{\lVert W_i \rVert^4}{\sum_j (\hat W_i \cdot W_j)^2},$$

1 for a feature with a dimension to itself, $1/2$ for each member of an antipodal pair, $2/5$ for a pentagon.

### Undoing superposition: sparse autoencoders

If a model's activation $x \in \mathbb{R}^d$ is a sparse sum of many feature directions, a dictionary of $d_{\text{sae}} \gg d$ directions with a sparse code should recover them:

$$\text{pre} = (x - b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}}, \qquad z = \sigma(\text{pre}), \qquad \hat x = z W_{\text{dec}} + b_{\text{dec}}, \qquad x = \hat x + e$$

Symbols: $W_{\text{enc}} \in \mathbb{R}^{d \times d_{\text{sae}}}$ the encoder, $W_{\text{dec}} \in \mathbb{R}^{d_{\text{sae}} \times d}$ the decoder whose rows are the feature directions (kept at unit norm), $b_{\text{enc}}, b_{\text{dec}}$ biases, $z \ge 0$ the sparse code (one entry per **latent**), $e$ the reconstruction error. The three activation functions $\sigma$:

- **ReLU + L1** (Bricken et al. 2023): $z = \text{ReLU}(\text{pre})$, loss $\lVert x - \hat x \rVert^2 + \lambda \sum_i z_i$. The L1 penalty also shrinks the codes of features that are active, so reconstructions are biased low (**shrinkage**).
- **TopK** (Gao et al. 2024, Eq. 2): keep the $k$ largest pre-activations, zero the rest; loss $\lVert x - \hat x \rVert^2$. Sparsity is set directly ($L_0 = k$) and there is no shrinkage. Dead latents are fought with an auxiliary loss (AuxK) that reconstructs the error with the top dead latents, weighted $1/32$, and with the encoder initialised to the transpose of the decoder (section 2.4, appendix A).
- **JumpReLU + L0** (Rajamanoharan et al. 2024, Eq. 4): $z_i = \text{pre}_i \cdot H(\text{pre}_i - \theta_i)$ with a learned threshold $\theta_i > 0$ per latent and $H$ the step function; loss $\lVert x - \hat x \rVert^2 + \lambda \sum_i H(\text{pre}_i - \theta_i)$. The step has zero gradient almost everywhere, so training uses **straight-through estimators** (Eqs. 11–12): with $K(u) = 1$ for $|u| < 1/2$, else 0, and bandwidth $\varepsilon$,

$$\frac{\partial z_i}{\partial \theta_i} \approx -\frac{\theta_i}{\varepsilon} K\Big(\frac{\text{pre}_i - \theta_i}{\varepsilon}\Big), \qquad \frac{\partial H(\text{pre}_i - \theta_i)}{\partial \theta_i} \approx -\frac{1}{\varepsilon} K\Big(\frac{\text{pre}_i - \theta_i}{\varepsilon}\Big).$$

The paper uses $\varepsilon = 0.001$ with inputs scaled so that $\mathbb{E}[x_j^2] = 1$; the course scales inputs the same way.

### How to judge an SAE

| Measure | Definition | What it misses |
|---|---|---|
| FVU (fraction of variance unexplained) | $\sum \lVert x - \hat x \rVert^2 / \sum \lVert x - \bar x \rVert^2$ | whether the lost variance mattered to the model |
| $L_0$ | mean number of active latents per token | whether the latents mean anything |
| dead latents | fraction that never fire on held-out tokens | — |
| **delta loss** (spliced) | next-token loss with $\hat x$ in place of $x$, minus the clean loss | interpretability of the latents |
| loss recovered | $(L_{\text{ablated}} - L_{\text{spliced}}) / (L_{\text{ablated}} - L_{\text{clean}})$ | depends on the ablation chosen |

Gemma Scope's primary metric is delta LM loss, with FVU secondary (section 4); Gao et al. report downstream loss alongside probe, explainability and ablation-sparsity metrics (section 4). Towards Monosemanticity reports "79% of the log-likelihood loss reduction provided by the MLP layer is recovered" for its 4,096-feature run, where the reference is zero-ablating the MLP. For a residual-stream site zero ablation destroys the model, so "loss recovered" against it is always near 1 and says little: the course reports the raw delta loss with an interval, and loss recovered against **mean** ablation.

Before any of these, the **splice check**: put $\hat x + (x - \hat x)$ back and confirm the loss is unchanged. If it is not, the hook is in the wrong place, the dtype is wrong, or the SAE expects normalised inputs it is not getting — and every number after it is meaningless.

## Worked example

**TopK, $k = 2$.** $\text{pre} = (0.5, -1.0, 2.0, 0.1)$: the two largest are 2.0 and 0.5, so $z = (0.5, 0, 2.0, 0)$. For $\text{pre} = (-0.3, -0.2, -0.1, -0.4)$ the two largest are negative and the ReLU after TopK gives $z = 0$: a token can have fewer than $k$ active latents.

**JumpReLU and its estimator.** $\theta = 0.5$ for all four latents, $\varepsilon = 0.1$, $\text{pre} = (0.10, 0.52, 0.48, 1.00)$. Forward: $z = (0, 0.52, 0, 1.00)$ — the 0.48 is cut although it is close to the threshold, which is the jump. Backward with upstream gradient 1: with respect to $\text{pre}$, $(0, 1, 0, 1)$; with respect to $\theta$, only entries with $|\text{pre} - \theta| < \varepsilon/2 = 0.05$ count (0.52 and 0.48), each $-\theta/\varepsilon = -5$. The L0 penalty's estimator gives $-1/\varepsilon = -10$ for the same two entries. These are the exact values `test_interp.py` checks.

**FVU.** $x = \{(1, 0), (3, 2)\}$, mean $(2, 1)$, total variance $1 + 1 + 1 + 1 = 4$. A reconstruction $\hat x = x + (1, 1)$ has squared error $2 + 2 = 4$: FVU = 1, as bad as predicting the mean, although every coordinate is only 1 off.

**Loss recovered.** The lab's clean loss is 5.736, mean ablation of the site gives 8.031, and the ReLU SAE with $\lambda = 1$ gives 5.784: $(8.031 - 5.784)/(8.031 - 5.736) = 0.979$. The same SAE's delta loss is $+0.048$ nats.

**Superposition dimensions.** Two features sharing one dimension as an antipodal pair: $W_1 = 1$, $W_2 = -1$. Then $D_1 = 1^4 / (1^2 + (-1)^2) = 1/2$. The lab measures 0.45–0.55 for the represented features at sparsity 0.7: pairs.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| toy-model weights $W$ | (5, 20) | float32 | CPU |
| lab activations at `resid_post.1` | (65,024, 128) training, (8,128, 128) held out | float32 | CPU |
| lab SAE: $W_{\text{enc}}$ / $W_{\text{dec}}$ | (128, 1,024) / (1,024, 128) | float32 | CPU |
| Qwen3-1.7B-Base residual stream | (B, T, 2,048) | bf16 | GPU |
| Qwen-Scope SAE per layer: $W_{\text{enc}}$ / $W_{\text{dec}}$ | (2,048, 32,768) / (32,768, 2,048) | bf16 / fp32 | GPU |

FLOPs per token for one SAE: encoder $2 d\, d_{\text{sae}}$, decoder $2 k d$ with a sparse decode or $2 d\, d_{\text{sae}}$ dense. For Qwen-Scope on Qwen3-1.7B-Base: $2 \cdot 2048 \cdot 32768 = 1.34 \times 10^8$ FLOPs to encode, against about $2 \cdot 1.41 \times 10^9 = 2.8 \times 10^9$ for the model's forward pass (non-embedding parameters): one SAE adds about 5% per token when spliced in. Parameters: $2 \cdot 2048 \cdot 32768 = 1.34 \times 10^8$ per layer, 0.27 GB in bf16; all 28 layers 7.5 GB. Training data dominates memory: 4M tokens of activations at $d = 2048$ in bf16 are 16 GB, which is why production trainers stream activations through a buffer instead of storing them.

## Build it

```python
from frontierlab.interp import sae as S, superposition as SP, tasks as T

SP.stats(SP.train(20, 5, sparsity=0.9))                    # represented features, D*, D_i
model = T.m17_model()                                      # 4 layers, width 128, trained on Data-v0
acts = S.collect(model, T.data_windows("train", 512, 128, 0), "resid_post.1")
sae, log = S.train_sae(acts, "jumprelu", d_sae=1024, coeff=0.5, steps=2000)
S.evaluate(sae, held_out_acts)                             # fvu, l0, dead
S.spliced_losses(model, val_windows, "resid_post.1", sae, "sae+error")   # the splice check: equals clean
S.spliced_losses(model, val_windows, "resid_post.1", sae, "sae")         # per-window loss with the SAE in
```

`frontierlab/interp/sae.py` implements the three kinds in one class: decoder rows renormalised to unit norm after every step (so the L1 penalty cannot be dodged by shrinking $z$ and growing $W_{\text{dec}}$), the encoder initialised to the decoder's transpose, $b_{\text{dec}}$ initialised to the data mean, inputs divided by one scalar so that $\mathbb{E}[x_j^2] = 1$, AuxK on latents that have not fired for 200 steps (TopK only), and JumpReLU's two estimators as `torch.autograd.Function`s. Correctness checks (`labs/common/tests/test_interp.py`): the JumpReLU forward and both pseudo-gradients against the hand-worked values above; TopK's $L_0 \le k$; unit-norm decoder rows after renormalising; an SAE trained on synthetic activations made from 32 known sparse directions recovers most of them (cosine above 0.9); the splice check holds to $10^{-10}$ in float64 and an untrained SAE does change the loss.

> [!NOTE]
> The first coefficients tried in this build ($\lambda = 0.002$ to $0.2$ for ReLU and JumpReLU) produced codes with 130–330 active latents out of 1,024 — reconstructions near perfect and no sparsity. The penalty is relative to the reconstruction term, which here is a sum over 128 normalised dimensions; the lab's values ($\lambda = 1$ and 4 for L1, 0.5 and 2 for L0) were chosen to reach $L_0$ between 5 and 30. A JumpReLU threshold stored as a logarithm also barely moved in 2,000 steps; the course stores $\theta$ directly. Always look at $L_0$ before reading anything else.

## What the evidence says

- **Superposition: ESTABLISHED in toy models**, PUBLICLY DOCUMENTED (Elhage et al. 2022). How much of a real model's computation is in superposition is still debated; the toy is evidence that it *can* happen and what its signature looks like.
- **Sparse autoencoders: ESTABLISHED as a tool**, with open questions about what their latents mean. Published scale (PUBLICLY DOCUMENTED): Towards Monosemanticity trained 512 to 131,072 features on the 512-neuron MLP of a one-layer transformer; Scaling Monosemanticity trained 1M, 4M and 34M features on Claude 3 Sonnet's middle-layer residual stream, with fewer than 300 active per token, at least 65% variance explained, and about 2%, 35% and 65% dead features at the three sizes (company claims about a closed model). Gao et al. trained a 16M-latent TopK SAE on GPT-4 residual activations for 40B tokens and report only 7% dead latents with their fixes (company claim).
- **TopK vs JumpReLU vs ReLU: PROMISING evidence that TopK and JumpReLU beat L1-ReLU at matched sparsity.** Rajamanoharan et al. report JumpReLU "at least as good as, and often slightly better than" TopK and better than Gated SAEs on Gemma 2 9B (abstract); Gao et al. report TopK improving the reconstruction-sparsity frontier over ReLU. These are each lab's own comparisons on its own models.
- **Open dictionaries:** Gemma Scope (JumpReLU on every layer and sublayer of Gemma 2 2B and 9B and selected layers of 27B; more than 400 SAEs in the main release, widths 16K to 1M; delta LM loss as the primary metric); Gemma Scope 2 (December 2025: SAEs and transcoders for Gemma 3 270M–27B, plus cross-layer transcoders for 270M and 1B); Qwen-Scope for Qwen3-1.7B-Base (TopK, 32,768 latents, $k = 50$, residual stream after each of the 28 layers; the model card says using them on the post-trained Qwen3-1.7B is "also reasonable", an untested claim you can check). MODEL-SPECIFIC: each dictionary belongs to one model and one site.
- **Course measurement (free CPU, 2026-10-07; torch 2.14.1, 16-thread laptop with other jobs running).** Toy model, 20 features in 5 dimensions: at sparsity 0, 0.7, 0.9 and 0.97 it represents 5, 10, 12 and 16 features, with $D^* = 1.00, 0.50, 0.37, 0.25$; represented features have dimensionality 1.0 (dense), about 0.5 (pairs) and 0.28–0.45 (larger structures). SAEs with 1,024 latents on `resid_post.1` of the Module 17 model (clean loss 5.736 on 64 validation windows; mean ablation 8.031; zero ablation 9.011), splice check $9.5 \times 10^{-7}$ in float32:

| SAE | FVU | $L_0$ | dead | delta loss [95% CI] | recovered vs mean |
|---|---|---|---|---|---|
| ReLU, $\lambda = 1$ | 0.055 | 27.5 | 0.02 | +0.048 [+0.041, +0.056] | 0.979 |
| ReLU, $\lambda = 4$ | 0.217 | 4.9 | 0.27 | +0.232 [+0.213, +0.250] | 0.899 |
| TopK, $k = 16$ | 0.022 | 16.0 | 0.18 | +0.017 [+0.013, +0.020] | 0.993 |
| TopK, $k = 32$ | 0.013 | 32.0 | 0.03 | +0.009 [+0.007, +0.011] | 0.996 |
| JumpReLU, $\lambda = 0.5$ | 0.033 | 28.3 | 0.11 | +0.024 [+0.020, +0.028] | 0.989 |
| JumpReLU, $\lambda = 2$ | 0.043 | 22.4 | 0.02 | +0.034 [+0.029, +0.039] | 0.985 |

  At about 28 active latents, JumpReLU loses half the loss ReLU loses (0.024 against 0.048), and TopK with only 16 loses less than either: the published ordering, on a 1.8M-parameter model trained for 2.5M tokens. Every SAE costs a loss increase whose interval excludes zero — none is a lossless view of the model. What the most frequently used TopK latents read is in the lab output; in this build two of them fired on the first piece of a word that continues ("po", "investig", "transl"): sub-word structure, the kind of feature a small model trained on 2.5M tokens can have.

## Lab

**Folder:** [`labs/module-17/lesson-01/`](../../labs/module-17/) · **Time:** about 75 minutes · **Pass check:** `pytest labs/module-17/lesson-01` passes; `sae_lab.py` prints all five parts with a splice check below $10^{-5}$; your write-up picks one SAE for lessons 17.2–17.3 and justifies it with delta loss at matched $L_0$, not FVU alone.

### Experiment contract

- **Question:** at matched sparsity, which SAE kind keeps the most of what the model uses at `resid_post.1`? Decision informed: the dictionary kind used for features in the rest of the module (and on the main path, whether the published Qwen-Scope SAE is good enough or the learner should train one).
- **Hypothesis:** TopK and JumpReLU have lower delta loss than ReLU at the same $L_0$. Status: reported effect (Gao et al.; Rajamanoharan et al.); may not appear in a 4-layer model.
- **Baseline:** the ReLU (L1) SAE, the original recipe.
- **Changed variable:** the activation function and its sparsity setting. **Controlled:** the model checkpoint, the site, the 65,024 training and 8,128 held-out tokens, 1,024 latents, 2,000 steps of batch 512, Adam at $10^{-3}$, seed 0.
- **Comparison axis:** equal latents, data and steps; compare at matched $L_0$ (interpolate between settings when $L_0$ differ).
- **Budget:** free CPU, measured 3.1 minutes once the model exists (8 minutes more to train it on an idle laptop); main path under 1 GPU-hour (PROJECTED).
- **Metrics and decision rule:** delta loss on 64 fixed validation windows with a paired bootstrap 95% interval, FVU, $L_0$ and dead fraction on held-out tokens; a kind "wins" only if its delta-loss interval lies below the other's at an $L_0$ no higher.
- **Correctness checks:** `pytest labs/common/tests/test_interp.py -k "sae or jumprelu or splice"`; the splice check in every row.
- **Fallback evidence:** the Gemma Scope and Gao et al. comparisons, labelled as published.
- **Limits:** one site of one tiny model; one seed per setting; 2,000 steps; held-out tokens from the same distribution; latent interpretability not measured.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× L40S/A100/H100. Not run in this build; part of the Module 17 pilot | `sae_lab.py --variant main --print` lists them: `python -m frontierlab.interp.hf sae-eval --layer L` for the published Qwen-Scope SAE at layers 7, 14, 21 of Qwen3-1.7B-Base (FVU, $L_0$, delta loss, splice check), then a course TopK SAE (16K latents, 4M tokens) on layer 14 compared with it. **PROJECTED:** under 1 GPU-hour |
| Free GPU (Colab/Kaggle T4) | T4 | `sae-eval` for one layer of Qwen3-1.7B-Base in float32 (6.9 GB of weights fits); training your own SAE needs fewer tokens (1M) |
| Free CPU | laptop; measured 3.1 min after the model exists, plus about 8 min to train it | `python labs/module-17/lesson-01/sae_lab.py` |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-17/lesson-01`.
2. **Run** `python labs/module-17/lesson-01/sae_lab.py`. In part 2, find the sparsity at which the toy model first represents more features than dimensions, and which geometry the represented features form.
3. **Splice check.** Confirm every row's check is below $10^{-5}$. Then break it on purpose: splice a reconstruction computed from *unscaled* inputs (call `sae.decode(sae.code(sae.pre(x)))`, skipping the scale) and see what the loss does.
4. **Matched sparsity.** Plot delta loss against $L_0$ for the six SAEs. At $L_0 \approx 20$, which kind is best, and is the difference outside the intervals?
5. **FVU versus loss.** Find two SAEs whose FVU ranking and delta-loss ranking disagree, or argue from the table why they might at a different site.
6. **Read three latents** in part 5. Write one sentence per latent saying what it fires on, and one sentence on what would make you trust that reading (lesson 17.2's interventions).

<details>
<summary>Hint for TODO 2</summary>

Save `pre` and `theta` with `ctx.save_for_backward`, and `eps` on `ctx`. In backward, the gradient for `theta` must be summed over every dimension except the last, because `theta` has one entry per latent: `.sum(dim=tuple(range(pre.dim() - 1)))`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-17/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-17/lesson-01`. The build's measured numbers are in "What the evidence says".

</details>

## Common mistakes

- **Reporting FVU without delta loss.** Variance the model ignores and variance it relies on count the same in FVU.
- **Skipping the splice check.** A hook on the wrong site or a missing input scale gives plausible-looking, meaningless numbers.
- **Loss recovered against zero ablation of the residual stream.** Zero ablation destroys the model, so almost any SAE "recovers" 99%; use mean ablation and the raw delta loss.
- **Comparing SAEs at different $L_0$.** A denser code reconstructs better; compare at matched sparsity.
- **Letting the decoder norm grow.** Without renormalisation the L1 penalty is beaten by small codes and large decoder rows.
- **Using a published SAE on the wrong model or site.** Qwen-Scope was trained on Qwen3-1.7B-Base's residual stream after each layer; on another checkpoint or site, measure delta loss before trusting it.
- **Reading a latent from its top activations alone.** That is a hypothesis about the latent, not evidence that the model uses it.

## References

- N. Elhage et al., *Toy Models of Superposition*, Anthropic, 2022. https://transformer-circuits.pub/2022/toy_model/index.html
- T. Bricken et al., *Towards Monosemanticity*, Anthropic, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html
- A. Templeton et al., *Scaling Monosemanticity*, Anthropic, 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- L. Gao et al., *Scaling and evaluating sparse autoencoders*, 2024. https://arxiv.org/abs/2406.04093
- S. Rajamanoharan et al., *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*, 2024. https://arxiv.org/abs/2407.14435
- T. Lieberum et al., *Gemma Scope*, 2024. https://arxiv.org/abs/2408.05147
- Google DeepMind, *Gemma Scope 2*, 2025. https://huggingface.co/google/gemma-scope-2
- Qwen, *SAE-Res-Qwen3-1.7B-Base-W32K-L0_50* (revision `ce1a79d9c5163932d65c417380e53230e1086370`; Qwen licence). https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50
- SAELens v6.53.0, `pretrained_saes.yaml`. https://github.com/decoderesearch/SAELens/blob/v6.53.0/sae_lens/pretrained_saes.yaml
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[17.2 · Causal interventions](lesson-02.md)
