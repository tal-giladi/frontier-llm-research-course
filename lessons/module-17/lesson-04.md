---
id: "17.4"
module: 17
minutes: 35
practice_minutes: 60
prerequisites: ["17.2"]
objectives:
  - Extract a steering vector by difference of means from contrastive A/B items, and remove the confound between the trait and a correlated property before extracting.
  - Choose the layer and strength on a selection split only, then measure a dose response with 95% intervals on held-out facts and a held-out template family.
  - Test a vector against random directions of the same norm (one- and two-sided) and a label-shuffled vector, and measure side effects on factual accuracy, on items where the trait is correct, and on ordinary-text loss and KL.
  - State what the evidence supports about a steering vector, including when the random-direction control fails.
volatility: concept
sources:
  - title: "Panickssery et al. — Steering Llama 2 via Contrastive Activation Addition (CAA)"
    url: https://arxiv.org/abs/2312.06681
  - title: "Turner et al. — Steering Language Models With Activation Engineering (ActAdd)"
    url: https://arxiv.org/abs/2308.10248
  - title: "Chen et al. — Persona Vectors: Monitoring and Controlling Character Traits in Language Models (Anthropic, 2025): sections 2, 3, 5, 6"
    url: https://arxiv.org/abs/2507.21509
  - title: "Zou et al. — Representation Engineering: A Top-Down Approach to AI Transparency"
    url: https://arxiv.org/abs/2310.01405
  - title: "Templeton et al. — Scaling Monosemanticity (feature steering, section Influence on Behavior)"
    url: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
last_verified: "2026-10-07"
---

# 17.4 · Steering and persona vectors

If a trait is represented along a direction in the residual stream, adding that direction should change the trait. Steering vectors are the simplest intervention interpretability offers — a difference of two means, added with a coefficient — and they are used to monitor and control character traits in deployed models. This lesson builds them from scratch on a harmless trait, sycophancy about simple facts in Qwen3-0.6B, and holds them to the module's standard: layer and strength chosen on one split, effects measured on held-out facts and templates, random directions of the same norm and a label-shuffled vector as controls, and side effects on everything the vector was not supposed to touch. The lab's honest result is mixed, and the lesson is about reading it correctly.

> [!NOTE]
> The course steers harmless traits only: style, topic, and agreeing with a user's stated answer. It does not extract or remove refusal or safety directions; that topic belongs to a separate course with its own review.

## Why this matters at a frontier lab

Persona vectors (Anthropic, 2025) turn a trait description into a direction that predicts when a model will show the trait, can steer it, can be added during fine-tuning to prevent the model drifting along it, and can flag training data that would push it. Each of those uses rests on the direction being specific: if a "sycophancy" vector is really a "be less certain" vector, monitoring with it flags the wrong conversations and preventative steering suppresses the wrong thing. The engineering question is never "does adding the vector change the behaviour?" — almost any large perturbation does — but "does it change *this* behaviour more than a random direction would, without changing what it should not?"

## The idea

### Extraction: a difference of means

ActAdd (Turner et al. 2023) takes the activation difference of one prompt pair ("Love" − "Hate") at one layer. Contrastive activation addition (CAA, Panickssery et al. 2023) averages over hundreds of A/B multiple-choice pairs: the same question answered with the behaviour's letter and with the other letter, activations read at the answer letter:

$$v_L = \frac{1}{N}\sum_{i=1}^{N} h_L\big(q_i, a_i^{\text{trait}}\big) - \frac{1}{N}\sum_{i=1}^{N} h_L\big(q_i, a_i^{\text{other}}\big)$$

Symbols: $h_L(\cdot) \in \mathbb{R}^d$ the residual stream after layer $L$ at the last token, $q_i$ the item, $a_i$ the appended answer letter. Persona vectors (Chen et al. 2025, section 2) generate the contrast automatically: from a trait name and description, a model writes contrastive system prompts (5 pairs), questions (40, split into extraction and evaluation) and a judge rubric; the vector is the difference of mean response-token activations between responses that do and do not show the trait, filtered by the judge. Representation engineering (Zou et al. 2023) is the broader programme of reading and controlling such population-level directions.

**Steering** adds $\alpha v_L$ to the residual stream after layer $L$ at every position. **Monitoring** projects activations on $\hat v_L = v_L / \lVert v_L \rVert$.

### The confound you must design out

A difference of means encodes *every* way the two sets differ. In the course's first version of the lab, every user belief was wrong, so "picks the user's answer" and "picks a false answer" were the same contrast; the extracted vector steered in the *opposite* direction to its label on held-out items. The fix is in the data: half the extraction items state the right answer, half the wrong one, and answer letters are random, so agreeing with the user is the only systematic difference. Design the contrast before computing the mean.

### Evaluation: splits, dose response, controls, side effects

1. **Splits.** Extraction items (facts of half 0, template family "train"); a selection split (the same facts without the answer appended) chooses the layer and $\alpha$; evaluation uses facts never seen (half 1) in the train family and in a held-out family of phrasings.
2. **Dose response.** The change in trait log-odds $\Delta(\alpha) = \overline{\ell(\alpha)} - \overline{\ell(0)}$, with $\ell = \log p(\text{trait letter}) - \log p(\text{other letter})$, for $\alpha \in \{-4, -2, -1, 1, 2, 4\}$, with a bootstrap interval over items.
3. **Controls.** Random directions with the same norm as $v$ (one-sided: is $\Delta(+2)$ larger than random?; two-sided: is $\Delta(+2) - \Delta(-2)$ larger than random? — damage tends to move a behaviour one way, a trait direction moves it both ways), and the same estimator with the trait labels shuffled.
4. **Side effects.** Factual errors on the same questions with no user opinion; behaviour on items where the user's answer is *correct* (steering toward agreement should not hurt here, steering away should); next-token loss and KL on ordinary text. Persona vectors report that inference-time steering with large coefficients degrades MMLU, and that preventative steering during fine-tuning preserves capability better (section 5).

## Worked example

**A two-item mean difference.** In a 2-dimensional stream, trait-letter activations $(1.0, 2.0)$ and $(3.0, 4.0)$, other-letter activations $(0, 0)$, $(0, 2)$, $(0, 1)$: $v = (2.0, 3.0) - (0, 1.0) = (2.0, 2.0)$, $\lVert v \rVert = 2.83$.

**Strength relative to the stream.** In the lab, $\lVert v_{12} \rVert = 3.70$ while the mean residual norm at layer 12 is 43.9, so $\alpha = 2$ adds a vector of norm 7.4, 17% of the stream's norm, at every position. At layer 16 the vector is 14.7 against 73.2; the same $\alpha$ is a larger relative push and produced 8× the ordinary-text KL.

**Log-odds and flips.** An item with $p(\text{trait}) = 0.3$, $p(\text{other}) = 0.6$ has $\ell = \log(0.3/0.6) = -0.69$: the model picks the other answer. A steering change of $+1.0$ gives $\ell = +0.31$: it now picks the trait answer, a flip. The mean change and the flip rate answer different questions; report both.

**The two-sided control.** Suppose a random direction gives $\Delta(+2) = +0.9$ and $\Delta(-2) = +0.7$ (both directions push the log-odds toward zero: damage). Its spread is $0.2$. A trait direction with $\Delta(+2) = +1.6$, $\Delta(-2) = -0.8$ has spread 2.4. The spread separates a direction from damage better than either side alone.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| extraction items with the letter appended | 72 sequences of 90–110 tokens, left-padded in batches of 8 | int64, attention mask int64 | CPU |
| residual stream after layer 12, Qwen3-0.6B | (8, T, 1,024) | float32 | CPU |
| steering vector $v_{12}$ | (1,024,) | float32 | CPU |
| logits at the last position | (8, 151,936) | float32 | CPU |
| Qwen3-1.7B (main path) residual stream | (16, T, 2,048) | bf16 | GPU |

Cost: one forward pass per item per setting. The lab runs about 3,700 item-forwards of ~100 tokens on Qwen3-0.6B (0.44B non-embedding parameters): $3{,}700 \cdot 100 \cdot 2 \cdot 4.4 \times 10^8 \approx 3.3 \times 10^{14}$ FLOPs, 26 minutes on a laptop CPU, dominated by the 2 × 19 random-direction controls. Left padding with an attention mask gives the same last-token logits as unpadded runs (a test checks it), because RoPE depends only on relative positions and padded keys are masked.

## Build it

```python
from frontierlab.interp import persona as PE, steering as ST, hf

model, tok = hf.load("qwen3-0.6b")
ext = PE.items("train", half=0, belief="mixed")                         # agreement is the only systematic contrast
acts_t = ST.read(model, [x["ids"] for x in PE.encode(tok, ext, "trait")], [8, 12, 16])
acts_o = ST.read(model, [x["ids"] for x in PE.encode(tok, ext, "other")], [8, 12, 16])
v = ST.mean_diff(acts_t[12], acts_o[12])
ev = PE.encode(tok, PE.items("heldout", half=1, belief="wrong"))
ST.dose_response(model, ev, 12, v, [-4, -2, -1, 1, 2, 4])               # change, interval, trait rate, flips
ST.random_like(v, 19); ST.shuffled_control(acts_t[12], acts_o[12])
ST.side_effects(model, windows, 12, v, 2.0)                             # loss increase with interval, KL
```

`frontierlab/interp/steering.py` and `persona.py`. Correctness checks: the mean difference by hand; steering at $\alpha = 0$ leaves loss and KL exactly unchanged; random directions have exactly the vector's norm; the projection monitor separates two shifted clouds; left-padded batches match single-item runs to $10^{-4}$; extraction and evaluation facts are disjoint and answer letters balanced.

## What the evidence says

- **Steering by difference of means: ESTABLISHED as a technique, PROMISING as a specific control.** CAA steered seven behaviours (including sycophancy) in Llama 2 7B and 13B Chat at layers around 13–15, generalised to open-ended generation, and "does not significantly affect MMLU performance" (PUBLICLY DOCUMENTED). Persona vectors (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct; traits evil, sycophancy, hallucination and four more): the projection of the last prompt token predicts trait expression with r = 0.75–0.83; inference-time steering with large coefficients degrades MMLU; preventative steering during fine-tuning preserves capabilities better; projection differences flag problematic training data, also on LMSYS-Chat-1M (sections 3, 5, 6). Scaling Monosemanticity steered with single SAE features (the Golden Gate Bridge feature clamped to 10× its maximum, error term unchanged), and notes that interesting effects typically need large clamps (company claims about a closed model).
- **Course measurement (free CPU, 2026-10-07; Qwen3-0.6B `c1899de`, float32, 26 minutes).**
  - Baseline: with a wrong user belief the model picks the user's answer 44–46% of the time; with no opinion stated it errs 12% of the time; with a right belief it agrees 100%. The trait is present: a stated wrong belief nearly quadruples the error rate.
  - Selection (selection split only): layer 8 barely moves ($-0.17$ / $+0.10$); layer 12 gives $-0.99$ [−1.65, −0.34] at $\alpha = -2$ and $+0.80$ [+0.27, +1.32] at $+2$ with ordinary-text KL 0.028; layer 16 is not monotonic and has KL 0.20–0.23 (excluded by the 0.1 limit). Chosen: layer 12.
  - Held-out dose response (facts never seen): train family $-0.82, -0.79, +0.63, +0.91, +1.08$ at $\alpha = -2, -1, +1, +2, +4$; held-out family $-0.78, -0.84, +0.98, +1.65, +1.90$, with the rate of picking the user's wrong answer going from 0.23 ($\alpha = -1$) to 0.60 ($\alpha = +4$) against 0.44 unsteered. At $\alpha = -4$ the effect reverses sign in both families (+0.18, +0.34, intervals including 0): past that strength the vector is damage, not steering.
  - Controls at $\alpha = +2$ on the held-out family: the vector $+1.65$ [+1.15, +2.16]; 19 random directions of the same norm: mean $+0.32$, maximum $+1.80$, $p_{\text{random}} = 0.15$; two-sided spread $+2.43$ against random mean $-0.29$, maximum $+3.30$, $p_{\text{random}} = 0.10$; label-shuffled vector $+0.22$ [−0.49, +0.89].
  - Side effects: at $+2$, factual errors without an opinion 0.12 → 0.17, agreement with correct beliefs 1.00 → 0.98, ordinary-text loss +0.025 [+0.010, +0.041] nats, KL 0.028. At $-2$, errors 0.12 → 0.25 and agreement with *correct* beliefs 1.00 → 0.67: the negative direction makes the model contrarian, not more accurate.
  - Monitoring: the projection of the last prompt token on $v$ does not predict the unsteered answer (r = −0.19, n = 48), unlike the r = 0.75–0.83 reported for persona vectors in 7–8B models.
  - What this supports: on Qwen3-0.6B, the layer-12 vector moves agreement with a user's wrong answer in the direction of the sign of $\alpha$ over $\alpha \in [-2, +4]$ (rising with $\alpha$ on the positive side, flat at about $-0.8$ on the negative side), on facts and phrasings it was not extracted on, at a small ordinary-text cost. It does **not** yet support "this is a sycophancy direction": by the contract's rule (random-direction $p \le 0.05$) it fails, because random directions of the same norm move this small model's A/B log-odds a lot — 2 of 19 moved it as much. More random draws would not rescue the claim; a larger model, a larger contrast set or a better-conditioned metric might, and that is the main-path question.

## Lab

**Folder:** [`labs/module-17/lesson-04/`](../../labs/module-17/) · **Time:** about 60 minutes (26 minutes unattended) · **Pass check:** `pytest labs/module-17/lesson-04` passes; `steer_lab.py` prints all eight parts; your write-up states, in one sentence each, what the vector does on held-out items, whether it passes the random-direction control, and its side effects — and ends with the claim the evidence supports.

### Experiment contract

- **Question:** does a difference-of-means vector for agreeing with the user change that behaviour specifically, on facts and phrasings it was not extracted on? Decision informed: whether to use such a vector as a monitor or a steering control for the Stage D model (main path), and how strong it may be.
- **Hypothesis:** adding the vector raises, and subtracting lowers, the rate of agreeing with a user's wrong answer, more than random directions of the same norm, with small side effects at moderate strength. Status: reported effect (CAA, persona vectors) in 7–13B chat models; may not appear in a 0.6B model.
- **Baseline:** the unsteered model on the same items.
- **Changed variable:** the steering vector and $\alpha$. **Controlled:** model and revision, the item splits (fixed in `persona.py`), the chat template with thinking disabled, the read position, the metric.
- **Comparison axis:** equal norm for every direction; the same items for every setting.
- **Budget:** free CPU, measured 26 minutes; main path under 0.25 GPU-hours (PROJECTED).
- **Metrics and decision rule:** mean change in trait log-odds with a 95% bootstrap interval over items, trait rate and flip rate. Layer and $\alpha$ are chosen on the selection split only (largest |change| at $|\alpha| \le 2$ with ordinary-text KL below 0.1). The vector "is a trait direction" if, on the held-out family, its one-sided and two-sided random-direction p-values are both $\le 0.05$ (19 draws), the shuffled-label vector's interval includes 0, and agreement with correct beliefs drops by less than 0.05 at $+2$.
- **Correctness checks:** `pytest labs/common/tests/test_interp.py -k "steering or persona or left_padded or ab_logodds"`; `pytest labs/module-17/lesson-04`.
- **Fallback evidence:** CAA's and the persona-vectors paper's figures, labelled as published.
- **Limits:** one 0.6B model; 24 facts per half and five phrasings; A/B log-odds only, no open-ended judging; one vector construction (CAA); greedy generations shown only qualitatively.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× L40S/A100/H100. Not run in this build; part of the Module 17 pilot | `python labs/module-17/lesson-04/steer_lab.py --model qwen3-1.7b --layers 8 12 16 20 --device cuda` on Qwen/Qwen3-1.7B (`70d244cc`) or your Module 13/14 post-trained Stage D model. **PROJECTED:** $1.6 \times 10^{15}$ FLOPs, 10–15 minutes, under 0.25 GPU-hours. Optional: the persona-vectors pipeline (github.com/safety-research/persona_vectors) with its judge, harmless traits only, judge calls counted in the budget |
| Free GPU (Colab/Kaggle T4) | T4 | the same with `--model qwen3-1.7b --device cuda` in float32 (6.9 GB of weights, fits in 16 GB) |
| Free CPU | laptop; measured 26 min | `python labs/module-17/lesson-04/steer_lab.py` |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-17/lesson-04`.
2. **Predict, then run** `python labs/module-17/lesson-04/steer_lab.py`. Before part 4 prints, write down which layer you expect to be chosen and why the selection criterion includes KL.
3. **Read the dose response.** Where does the curve stop being monotonic, and what does that tell you about the usable range of $\alpha$?
4. **Controls.** Compute what fraction of random directions beat the vector at $+2$. Why do random directions move this model's log-odds in the *positive* direction on average? (Look at the baseline log-odds and at what damage does to a log-odds near zero.)
5. **Side effects.** Explain the $-2$ row: what does "agreement with correct beliefs drops to 0.67" say about what the negative direction does?
6. **The confound.** Rerun extraction with `belief="wrong"` only (edit the `ext = ...` line) and compare the vector's direction and held-out effect with the mixed version. Write two sentences on what the first version of the lab got wrong.
7. **Write the claim** the evidence supports, in the claim-card fields of `frontierlab.interp.claims`.

<details>
<summary>Hint for TODO 2</summary>

Clone the activation, then add `alpha * vec` to `y[:, start:]`. `vec` has shape (C,) and broadcasts over batch and positions. Cast it to the activation's dtype so the edit also works under bf16 autocast.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-17/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-17/lesson-04`. The build's measured numbers are in "What the evidence says".

</details>

## Common mistakes

- **A contrast with two differences.** If every positive example is also false (or longer, or always letter A), the vector encodes that too. Balance the extraction set.
- **Choosing the layer on the evaluation items.** Report the evaluation once, after choosing on a selection split.
- **Unmatched controls.** A random direction must have the vector's norm; a weaker random direction makes any vector look specific.
- **One-sided controls only.** Damage pushes a near-zero log-odds toward zero from either side; check that $-\alpha$ moves the behaviour the other way.
- **Ignoring items where the trait is correct.** A "less sycophantic" direction that makes the model disagree with correct beliefs is a contrarian direction.
- **Large $\alpha$ as proof.** Past the monotonic range, every direction changes everything.
- **Reading generations as measurements.** The two generations in part 8 illustrate; the A/B log-odds on held-out items measure.

## References

- N. Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition*, 2023 (v4, 2024). https://arxiv.org/abs/2312.06681
- A. M. Turner et al., *Steering Language Models With Activation Engineering*, 2023. https://arxiv.org/abs/2308.10248
- R. Chen et al., *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*, 2025. https://arxiv.org/abs/2507.21509
- A. Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency*, 2023. https://arxiv.org/abs/2310.01405
- A. Templeton et al., *Scaling Monosemanticity*, Anthropic, 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[17.5 · Interpretable by design and introspection](lesson-05.md) (extension), or go straight to the [Module 17 project](../../projects/module-17-causal-claim.md).
