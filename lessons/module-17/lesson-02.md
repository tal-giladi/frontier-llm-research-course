---
id: "17.2"
module: 17
minutes: 40
practice_minutes: 80
prerequisites: ["17.1", "01.4"]
objectives:
  - Run activation patching in both directions (denoising and noising) on clean/corrupt prompt pairs, normalise the effect, and say what each direction can and cannot show.
  - Implement head-level patching, attribution patching and path patching from scratch with forward hooks, and check them against identities that must hold exactly.
  - Recover a known circuit (a previous-token head feeding induction heads through their keys) and state which experiment establishes each part of that claim.
  - Test a candidate component against random directions and random components of the same size, on held-out prompts, and on off-target behaviour, and report each effect with a 95% interval.
volatility: concept
sources:
  - title: "Meng et al. — Locating and Editing Factual Associations in GPT (section 2.1: causal tracing)"
    url: https://arxiv.org/abs/2202.05262
  - title: "Wang et al. — Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (path patching; faithfulness, completeness, minimality)"
    url: https://arxiv.org/abs/2211.00593
  - title: "Heimersheim and Nanda — How to use and interpret activation patching (sections 2.3, 4.1, 5)"
    url: https://arxiv.org/abs/2404.15255
  - title: "Zhang and Nanda — Towards Best Practices of Activation Patching in Language Models: Metrics and Methods"
    url: https://arxiv.org/abs/2309.16042
  - title: "Makelov, Lange and Nanda — Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching"
    url: https://arxiv.org/abs/2311.17030
  - title: "Olsson et al. — In-context Learning and Induction Heads (Anthropic, 2022)"
    url: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
  - title: "Elhage et al. — A Mathematical Framework for Transformer Circuits (Anthropic, 2021): Q-, K- and V-composition"
    url: https://transformer-circuits.pub/2021/framework/index.html
  - title: "Nanda — Attribution Patching: Activation Patching At Industrial Scale (2023)"
    url: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
  - title: "Syed, Rager and Conmy — Attribution Patching Outperforms Automated Circuit Discovery"
    url: https://arxiv.org/abs/2310.10348
  - title: "Chan et al. (Redwood Research) — Causal Scrubbing: a method for rigorously testing interpretability hypotheses"
    url: https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
last_verified: "2026-10-07"
---

# 17.2 · Causal interventions

An activation tells you what a model computed; only an intervention tells you what the model uses. This lesson builds the interventions interpretability claims rest on — activation patching in both directions, head-level and path patching, the cheap attribution-patching screen, and ablations — and then the controls that turn an effect into a claim: random directions and random components of the same size, held-out prompts, and off-target behaviour. You first run the whole pipeline on a two-layer model whose mechanism is known in advance, so you can see the methods find it, and then on indirect-object identification in Qwen3-0.6B, where nobody hands you the answer.

## Why this matters at a frontier lab

Interpretability results feed decisions: which component to monitor, which feature to steer, whether a safety-relevant behaviour has a findable cause. A result that only says "this head's activation correlates with the behaviour" cannot support any of those decisions, because many components correlate with any behaviour. The claims that survive review are interventional, scoped and controlled: "on this distribution, replacing this component's output with its value on a contrast prompt removes 90% [85%, 95%] of the behaviour, while 19 random sets of the same size remove at most 12%, and the effect holds on templates we never looked at." The plan's causal standard for this module is exactly that: patching with controls, held-out behaviours, and effects reported with uncertainty. The methods are cheap; the discipline is the hard part.

## The idea

### Clean, corrupt, and a metric

A patching experiment needs two prompts that differ in one controlled way and a scalar metric of the behaviour. For induction, the clean prompt is a random token segment repeated (`… A B … A`) and the model should continue with `B`. Two corruptions change one token each:

- **value corruption:** the first `B` becomes `C`; the right continuation becomes `C`;
- **key corruption:** the first `A` becomes `X`; the second `A` no longer has an earlier match, so the model loses its reason to say `B`.

Corruptions should keep the prompt in distribution — replacing tokens symmetrically, not adding Gaussian noise to embeddings (Zhang and Nanda, section 3.2, show that Gaussian noising can put the model off distribution and give inconsistent localisation; they recommend symmetric token replacement). The metric should be roughly linear in the logits: the **logit difference** $m = \ell_{\text{good}} - \ell_{\text{bad}}$ at the prediction position (Heimersheim and Nanda, section 4.1; probabilities saturate and hide negative components).

### Two directions, two different questions

With per-prompt metrics $m_{\text{clean}}$, $m_{\text{corrupt}}$ and the gap $g = \overline{m}_{\text{clean}} - \overline{m}_{\text{corrupt}}$ (means over the prompt set):

$$\text{denoising: run corrupt, copy in the clean activation:}\quad e = \frac{m_{\text{patched}} - m_{\text{corrupt}}}{g}$$

$$\text{noising: run clean, copy in the corrupt activation:}\quad e = \frac{m_{\text{clean}} - m_{\text{patched}}}{g}$$

Both are 0 for no effect and 1 for a full effect. Denoising asks whether a component is **sufficient** (together with everything downstream) to carry the difference; noising asks whether it is **necessary** on this distribution. They are not symmetric (Heimersheim and Nanda, section 2.3): with redundant components, each can be sufficient and none necessary. Dividing by the mean gap rather than each prompt's own gap avoids blowing up prompts where the gap is near zero.

### Heads, paths and composition

A transformer's residual stream is a sum: every attention head and MLP adds its output to it, and every later component reads the sum. Head $h$ of layer $l$ writes $z_h W_O^{h\top}$, where $z_h$ is its slice of the input to `o_proj`. That makes three finer interventions possible:

- **head patching** — copy one head's $z_h$ slice from the other run;
- **path patching** (Wang et al. 2022) — the effect of a sender head along the direct path into one receiver only. Because the stream is additive, the sender's change $\Delta = (z_h^{\text{other}} - z_h^{\text{run}}) W_O^{h\top}$ can be added to the *input of one receiver* while every other component reads the unchanged residual. Receivers can be a later head's query, key or value input, an MLP input, or the logits directly;
- **composition** (Elhage et al. 2021) — a later head can read an earlier head's output through its queries (Q-composition), keys (K-composition) or values (V-composition). The induction mechanism (Olsson et al. 2022) is K-composition: a *previous-token head* writes "the token before me was `A`" into the position of `B`; an *induction head* at the second `A` matches its query against that key, attends to `B` and copies it.

### The cheap screen: attribution patching

Patching every head costs one forward pass per head. **Attribution patching** (Nanda 2023) replaces that with a first-order estimate from one forward and one backward pass of the corrupt run:

$$\widehat{\Delta m}_h = \sum_{b,t,i \in h} \big(z^{\text{clean}} - z^{\text{corrupt}}\big)_{b,t,i}\; \frac{\partial m}{\partial z_{b,t,i}}\Big|_{\text{corrupt}}$$

Syed et al. (2023) report that edge attribution patching recovers circuits better than earlier automated circuit discovery at a fraction of the cost. It is still a linearisation: it fails for large patches, across normalisation layers and saturated nonlinearities (Nanda lists these). The course uses it only to choose what to patch for real.

### Ablations

**Ablation** removes a component instead of swapping it between two prompts: **zero** ablation (replace with 0, which puts the next layers off distribution), **mean** ablation (its mean over a reference distribution — Wang et al. use the "ABC" distribution of prompts with three unrelated names) and **resample** ablation (its value on another random prompt; causal scrubbing, Chan et al. 2022, builds a whole hypothesis test from resampling). An ablation effect is a statement about necessity relative to a reference: always name the reference.

### Controls: what makes an effect a claim

1. **Random components.** Ablate $n$ random sets of heads of the same size as the candidate. If the candidate's effect is inside that distribution, the claim "these heads matter" says nothing specific. Report $p_{\text{random}} = (1 + \#\{r \ge e\}) / (1 + n)$; with 19 draws the smallest possible value is 0.05.
2. **Random directions.** For a direction-level claim ("the model represents X along $u$"), project out $n$ random unit directions in the same site. A random direction in a 1,024-dimensional stream carries about $1/1024$ of the variance on average; a real direction must do much better.
3. **Held-out prompts.** Components are chosen on one prompt family; the effect is measured on another (other templates, other tokens, other distances). Selection on the same prompts you report is the interpretability version of tuning on the test set (lesson 01.3).
4. **Off-target behaviour.** A component that "explains" the behaviour by breaking the whole model is not specific. Measure next-token loss on ordinary text with the same intervention.
5. **Subspace illusions.** A subspace patch can change behaviour by activating a dormant pathway rather than the one in use (Makelov et al. 2023, section 3). Their recommendation: intervene in bottlenecks such as the residual stream and validate beyond the end-to-end metric.

Wang et al. add three criteria for a circuit as a whole: **faithfulness** (the circuit alone reproduces the model's metric), **completeness** (removing any subset from the circuit and from the model has similar effects) and **minimality** (every component matters for some subset). The module project asks for one component-level claim, not a full circuit, and holds it to controls 1, 3 and 4.

## Worked example

**Normalised effects.** Two prompts: clean metrics $(4, 6)$, corrupt $(0, -2)$, so $g = 5 - (-1) = 6$. A patch gives $(2, 1)$. Denoising effects: $(2 - 0)/6 = 0.33$ and $(1 - (-2))/6 = 0.50$. Noising effects of the same patched values: $(4 - 2)/6 = 0.33$ and $(6 - 1)/6 = 0.83$.

**Why redundancy breaks noising.** Suppose the behaviour needs *at least one* of two heads, each fully sufficient. Denoising either head alone restores everything: 1.0 and 1.0. Noising either alone loses nothing, because the other still works: 0.0 and 0.0. Only noising both shows 1.0. The lab's induction model has four layer-1 heads that share the induction work, and its numbers have exactly this shape (below).

**A direct path.** Let a sender head's output change by $\Delta = (0.2, -0.1)$ in a 2-dimensional stream, a receiver key projection be $W_K = \begin{pmatrix}1 & 0\\ 0 & 2\end{pmatrix}$, and ignore the norm. The receiver's key changes by $W_K \Delta = (0.2, -0.2)$; no other component sees $\Delta$. Path patching measures how much that change alone moves the metric.

**The control p-value.** A candidate removes 0.91 of the gap; 19 random sets remove at most 0.30. Then $p_{\text{random}} = (1 + 0)/(1 + 19) = 0.05$. With 7 draws the smallest possible value is $1/8 = 0.125$, so 7 draws can never give $p \le 0.05$ — plan the number of draws before running.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| clean / corrupt ids (induction model) | (64, 32): 12 tokens, gap 4, the 12 again, 4 filler | int64 | CPU |
| residual stream (induction model) | (64, 32, 64) | float32 | CPU |
| `z.L` (input of o_proj), induction model | (64, 32, 4·16) | float32 | CPU |
| `z.L`, Qwen3-0.6B | (B, T, 16·128 = 2,048); 16 query heads, 8 key/value heads | float32 on CPU / bf16 on GPU | CPU / GPU |
| sender delta $\Delta$ | (B, T, C) | as the stream | same |
| attribution map | (layers, heads): (2, 4) and (28, 16) | float64 (NumPy) | CPU |

Cost per experiment, counted in forward passes of the prompt batch: head patching $L \cdot H$ (8 for the toy, 448 for Qwen3-0.6B); attribution patching 2 forwards + 1 backward ≈ 4 forward-equivalents for *all* heads; one path-patching receiver 3; a component control with $n$ draws $n$. For Qwen3-0.6B at 48 prompts × ~20 tokens a forward is about $2 \cdot 0.44\text{B} \cdot 960 \approx 8.5 \times 10^{11}$ FLOPs, so the full head sweep would be $3.8 \times 10^{14}$ FLOPs — minutes on an idle laptop, hours on a busy one — while the screen plus 16 real patches and 19 controls is about 40 forward-equivalents. With grouped-query attention, a key/value head serves two query heads in Qwen3, so "the key input of head 5" is a shared projection; the course's path patching handles this by patching the whole key projection or a key/value head.

## Build it

```python
from frontierlab.interp import hooks as HK, patching as P, tasks as T

model = T.train_induction_model(path="runs/m17/induction.pt")           # about a minute
pk = T.induction_pairs(64, kind="key", seed=1)                           # clean/corrupt + metric position
P.head_sweep(model, pk, "denoise"), P.head_sweep(model, pk, "noise")     # (layers, heads)
P.attribution_heads(model, pk)                                           # the one-backward screen
P.path_patch(model, pk, sender_layer=0, sender_heads=[2], receiver=("k", 1))
P.component_control(model, pk, {0: [2]}, "mean", n_random=19)
P.direction_control(model, pk, "resid_pre.1", direction, n_random=32, positions=[7])
```

`frontierlab/interp/hooks.py` addresses activations by site name (`resid_pre.L`, `resid_post.L`, `attn_out.L`, `z.L`, `mlp_in.L`, `mlp_out.L`, `final`) on both the course's models and Hugging Face Qwen3 models, and removes every hook on exit, also on an exception. `patching.py` builds everything above on those hooks. Correctness checks (`labs/common/tests/test_interp.py`), each an identity that must hold exactly in float64:

- every site agrees with the forward pass (`attn_out = o_proj(z)`, `mlp_in = norm(resid_pre + attn_out)`, the layer boundary), and no hook outlives its call;
- patching the whole last-layer residual stream restores the clean metric exactly (effect 1);
- patching all heads of a layer one slice at a time equals patching the whole `z` site;
- with the last MLP switched off, path patching all last-layer heads into the logits equals ordinary noising of that layer; with the MLP on it does not — the difference is the path through the MLP;
- the control p-value cannot be smaller than $1/(1+n)$.

The induction data varies the distance between the two copies (a random gap of 0–8 tokens per sequence during training). The first version of this lab repeated the segment at a fixed offset, and the trained model solved the task with heads that attend a fixed number of positions back: the key corruption then barely changed its prediction (clean 13.1, corrupt 12.9), because no content matching was needed. Varying the offset is what forces induction — a reminder that a "known mechanism" is only known if the training distribution requires it.

## What the evidence says

- **Activation patching and its variants: ESTABLISHED.** Causal tracing located factual recall in mid-layer MLPs at the last subject token of GPT-2 XL (Meng et al., section 2.1 and Fig. 2, by restoring hidden states into a run with noised subject embeddings). The IOI circuit in GPT-2 small has 26 heads in 7 classes, found with path patching and checked with faithfulness, completeness and minimality (Wang et al., abstract and section 3). Both are PUBLICLY DOCUMENTED; both are on small models, and their reported numbers are about those models.
- **Best-practice guidance** (PUBLICLY DOCUMENTED): prefer patching to ablation, keep a metric that is roughly linear in the logits, distinguish denoising from noising, prefer in-distribution corruptions (Heimersheim and Nanda, section 5; Zhang and Nanda, section 6).
- **Attribution patching: ESTABLISHED as a screen**, with known failure modes (Nanda 2023). Syed et al. report better circuit recovery than ACDC with two forward passes and one backward pass (abstract).
- **Course measurement (free CPU, 2026-10-07, the induction model, 64 prompts each; torch 2.14.1 CPU, other jobs running):**
  - value corruption: clean logit difference 11.82, corrupt −11.78; every layer-1 head restores about 0.2 of the gap by denoising (0.18–0.21) and loses about 0.2 by noising — the copying is shared across four heads;
  - key corruption: clean 11.70, corrupt 2.64; denoising head 0.2 alone restores 0.97 of the gap and noising it loses 1.01; layer-1 heads each restore 0.43–0.47 by denoising but lose only 0.07–0.09 by noising — redundancy, exactly as in the worked example;
  - attribution patching ranked the same top head (0.2) but estimated 0.64 for its real 0.97 effect: right ranking, wrong size;
  - path patching from head 0.2 into the layer-1 *keys* gives 0.985 of the gap; into the queries 0.000, into the values 0.009, directly into the logits 0.000; the other layer-0 heads give 0.000 everywhere. That is the K-composition claim, measured;
  - controls: mean-ablating 0.2 loses 1.005 [0.906, 1.103] of the gap; the 7 other single heads lose 0.027 on average (at most 0.054), $p_{\text{random}} = 0.125$, the floor for 7 draws, so this table is not by itself a significance claim (the model has only 8 heads). Projecting the mean clean-minus-corrupt direction out of the residual stream at the first `B` removes only 0.030 [0.013, 0.052] of the gap — more than 32 random directions (95th percentile 0.009, $p = 0.03$) and still almost nothing: the information is token-specific, so no single mean direction carries it. Significant and small are not opposites;
  - held out: an unseen gap of 12 tokens (training used 0–8), and query positions 2 and 10 — ablating 0.2 loses 1.057, 0.965 and 0.930 of the gap, all intervals well above 0.8.
- **The Qwen3-0.6B part** (`--hf`) runs the same pipeline on IOI and feeds the module project. In this build (13.7 minutes CPU) the screen and real patches selected 8 heads in layers 17–27; mean-ablating them removed 0.80 [0.72, 0.87] of the logit difference, 0.95 [0.82, 1.09] on held-out templates and names, at +0.047 nats of ordinary-text loss. Against 19 random 8-head sets drawn from the same layers the effect was the largest ($p = 0.05$); against 19 sets drawn from any layer it was not ($p = 0.25$: four sets, three of them containing the same layer-1 head, removed more than the whole gap). The pre-stated control decides which claim you are making. Whether the IOI heads of GPT-2 small have counterparts in Qwen3 is an open question you answer for one model, at one scale, with your own controls — not a fact to assume.

## Lab

**Folder:** [`labs/module-17/lesson-02/`](../../labs/module-17/) · **Time:** about 80 minutes · **Pass check:** `pytest labs/module-17/lesson-02` passes; `patch_lab.py` prints all six parts; your write-up states the K-composition claim with the experiment behind each part of it, and one claim about Qwen3-0.6B with its controls (or says why the evidence does not support one).

### Experiment contract

- **Question:** which components of the induction model carry the matching step, and do the same methods with controls support a component-level claim about IOI in Qwen3-0.6B? Decision informed: which methods and controls the module project relies on.
- **Hypothesis:** a single layer-0 previous-token head feeds the layer-1 heads through their keys (K-composition); the layer-1 copying is shared. Status: established mechanism (Olsson et al.); its exact form in this toy is the thing measured.
- **Baseline:** the clean and corrupt runs of the same model and prompts (the two ends of every normalised effect).
- **Changed variable:** which activation is patched or ablated. **Controlled:** model checkpoint, prompt set and seed, metric, corruption type, reference distribution for mean ablation.
- **Comparison axis:** the same prompts in every arm; effects normalised by the same gap.
- **Budget:** free CPU, measured 7.4 minutes for parts 1–6 including training the model, with other jobs running; part 7 adds Qwen3-0.6B (measured 14.5 minutes CPU). Main path under 0.25 GPU-hours (PROJECTED).
- **Metrics and decision rule:** mean normalised effect with a 95% bootstrap interval over prompts; a component is "the" carrier of a step if its path-patching effect exceeds 0.8 with every other sender of its layer below 0.05; a claim needs $p_{\text{random}} \le 0.05$ (at least 19 random draws), a held-out interval above 0 and at least half the selection effect, and an off-target loss change under 0.05 nats.
- **Correctness checks:** `pytest labs/common/tests/test_interp.py`; `pytest labs/module-17/lesson-02`; the identities in "Build it".
- **Fallback evidence:** Olsson et al. and Wang et al., labelled as published, if your Qwen3 result is null.
- **Limits:** a 2-layer synthetic model; one task family; mean ablation over one reference distribution; IOI on one small model; logit difference only.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× A100/H100 80 GB (any 24 GB+ GPU works). Not run in this build; part of the Module 17 pilot | `python -m frontierlab.interp.hf ioi --model qwen3-1.7b-base --n 96 --top 10 --random 49 --out runs/m17/ioi-1.7b-base.json` on the Stage D base model; optional cross-check of one patch with TransformerLens 4.0.0 (`patch_lab.py --print`). **PROJECTED:** about 100 forward passes of 96 × 20 tokens, $6.5 \times 10^{14}$ FLOPs, minutes; under 0.25 GPU-hours |
| Free GPU (Colab/Kaggle T4) | T4 | the same command with `--model qwen3-1.7b-base --n 64` in float32 (fits in 16 GB); what you lose: nothing but time |
| Free CPU | laptop; parts 1–6 measured 7.4 min (16-thread laptop, other jobs running), part 7 14.5 min | `python labs/module-17/lesson-02/patch_lab.py` then `--hf` |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-17/lesson-02`.
2. **Run** `python labs/module-17/lesson-02/patch_lab.py`. Before reading part 3, write down which heads you expect to matter under each corruption, in each direction.
3. **Redundancy.** Explain, from part 3, why the layer-1 heads look important by denoising and unimportant by noising under the key corruption. Which experiment would show that they are jointly necessary? Run it (ablate all four with `P.ablation_effect`).
4. **Screen versus truth.** In part 4, where is attribution patching wrong, and would it have misled you about *which* head matters or only about *how much*?
5. **The claim.** Write the K-composition claim as one sentence and list the experiment behind each part: "previous-token" (which output?), "read through keys" (part 5), "necessary" (part 6), "on held-out distances" (part 6). Then write the sentence the direction control in part 6 does *not* support.
6. **Qwen3-0.6B.** Run `--hf`. Does the candidate pass every control of the contract? Write the claim card fields (`frontierlab.interp.claims.ClaimCard`) for it, or the reason it fails.

<details>
<summary>Hint for TODO 3</summary>

Build a mask of length H·d with ones on the sender heads' slices, multiply `(z_to - z_from)` by it, and multiply by `W_O.T` (W_O is `o_proj.weight`, shape (C, H·d)). With every head in the list, the result must equal the whole change of the attention output.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-17/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-17/lesson-02`. The build's measured numbers are in "What the evidence says".

</details>

## Common mistakes

- **Reporting one direction as if it were both.** Denoising shows sufficiency, noising necessity; with redundant components they disagree, and both are true.
- **Normalising per prompt.** A prompt whose clean and corrupt metrics are almost equal produces huge normalised effects; divide by the mean gap.
- **Gaussian noise as the corruption.** It moves the model off distribution and can localise the wrong place; swap tokens instead.
- **Choosing heads and reporting effects on the same prompts.** Selection inflates the effect; report on a held-out family.
- **Too few random draws.** With 7 draws $p_{\text{random}}$ cannot go below 0.125; decide the number of draws from the significance level before running.
- **Treating a significant direction as an explanation.** The lab's mean direction beats every random direction and removes 3% of the behaviour.
- **Forgetting that GQA shares keys.** In Qwen3 a key/value head serves two query heads; "this head's key" is shared.
- **Leaking a hook.** A hook left registered silently changes every later measurement; register hooks inside a context manager and assert none remain.

## References

- K. Meng et al., *Locating and Editing Factual Associations in GPT*, 2022, section 2.1. https://arxiv.org/abs/2202.05262
- K. Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, 2022. https://arxiv.org/abs/2211.00593
- S. Heimersheim and N. Nanda, *How to use and interpret activation patching*, 2024. https://arxiv.org/abs/2404.15255
- F. Zhang and N. Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*, 2023. https://arxiv.org/abs/2309.16042
- A. Makelov, G. Lange and N. Nanda, *Is This the Subspace You Are Looking for?*, 2023. https://arxiv.org/abs/2311.17030
- C. Olsson et al., *In-context Learning and Induction Heads*, 2022. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021. https://transformer-circuits.pub/2021/framework/index.html
- N. Nanda, *Attribution Patching: Activation Patching At Industrial Scale*, 2023. https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
- A. Syed, C. Rager and A. Conmy, *Attribution Patching Outperforms Automated Circuit Discovery*, 2023. https://arxiv.org/abs/2310.10348
- L. Chan et al. (Redwood Research), *Causal Scrubbing*, 2022. https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[17.3 · Transcoders and attribution graphs](lesson-03.md)
