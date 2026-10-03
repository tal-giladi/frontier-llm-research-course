---
id: "04.2"
module: 4
minutes: 40
practice_minutes: 90
prerequisites: ["04.1"]
objectives:
  - Explain, per RoPE frequency, why a model fails past its trained length, using wavelengths and rotations over the trained length.
  - Derive Position Interpolation, NTK-aware scaling and YaRN (frequency ramp and attention temperature) and compute their per-frequency factors by hand.
  - Implement the three rules and match Hugging Face Transformers' YaRN frequencies and attention factor numerically.
  - Measure what each rule buys past the trained length and costs below it, without training, with paired intervals on Eval v1.
  - Distinguish rules applied to a trained model (PI, NTK, YaRN) from architecture choices that need training from the start (partial RoPE, interleaved NoPE layers as in Llama 4's iRoPE).
volatility: concept
sources:
  - title: "Chen et al., Extending Context Window of Large Language Models via Positional Interpolation (abstract)"
    url: https://arxiv.org/abs/2306.15595
  - title: "Peng et al., YaRN: Efficient Context Window Extension of Large Language Models (v3: sections 3.1-3.4, Eqs. 8-15 and 19; section 4.1)"
    url: https://arxiv.org/abs/2309.00071
  - title: "DeepSeek-AI, DeepSeek-V3 Technical Report (section 4.3: long context extension with YaRN)"
    url: https://arxiv.org/abs/2412.19437
  - title: "Qwen3-8B model card (YaRN to 131,072 tokens; static YaRN note)"
    url: https://huggingface.co/Qwen/Qwen3-8B
  - title: "gpt-oss-20b config.json (rope_scaling: yarn, factor 32, original 4,096)"
    url: https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
  - title: "Qwen3-Next-80B-A3B-Instruct config.json and model card (partial_rotary_factor 0.25, head_dim 256)"
    url: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
  - title: "Barbero et al., Round and Round We Go! What makes Rotary Positional Encodings useful? (p-RoPE)"
    url: https://arxiv.org/abs/2410.06205
  - title: "Kazemnejad et al., The Impact of Positional Encoding on Length Generalization in Transformers (NoPE)"
    url: https://arxiv.org/abs/2305.19466
  - title: "Yang et al., Rope to Nope and Back Again: A New Hybrid Attention Strategy (section 3)"
    url: https://arxiv.org/abs/2501.18795
  - title: "Meta AI, The Llama 4 herd (blog, 2025-04-05: iRoPE)"
    url: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
  - title: "Hugging Face Transformers 5.18.0, src/transformers/modeling_rope_utils.py (_compute_yarn_parameters, _compute_llama3_parameters)"
    url: https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/modeling_rope_utils.py
last_verified: "2026-10-03"
---

# 04.2 · Position at long range

RoPE encodes position as rotations at many frequencies, and a model trained at length $L$ has only seen part of the rotation range of its slow frequencies. This lesson explains that failure one frequency at a time, derives the three rules used to stretch a trained model's positions (Position Interpolation, NTK-aware scaling and YaRN), checks your implementation against Hugging Face Transformers, and measures on Eval v1 what each rule does to a model past and below its trained length. It ends with two design choices that cannot be applied to a trained model: partial RoPE and interleaved layers without positions.

## Why this matters at a frontier lab

Every open long-context model in 2025–2026 states its position scheme in its config, and the choices differ: Qwen3 extends 32,768 native tokens to 131,072 with YaRN factor 4 (model card); gpt-oss's config uses YaRN with factor 32 from an original 4,096 (config.json); DeepSeek-V3 applies YaRN with $s = 40$ to its decoupled RoPE key only (V3 report, section 4.3); Qwen3-Next rotates only a quarter of each 256-dimensional head (`partial_rotary_factor` 0.25); Llama 4 interleaves layers with no positional embedding at all (Meta blog). The rule decides how much continued training an extension needs (04.3), how much short-context quality it costs, and whether a released model can be served past its native length. Reading these configs and predicting their effect per frequency is a daily task.

## The idea

### RoPE as a bank of clocks

RoPE (parent course lesson 12.1) splits each head's $D$ rotated channels into $D/2$ pairs. Pair $i$ of the query at position $m$ is rotated by the angle $m\theta_i$, the key at position $n$ by $n\theta_i$, with

$$\theta_i = b^{-2i/D}, \qquad i = 0, \dots, D/2 - 1,$$

where $b$ is the base (`rope_theta`, 10,000 for Baseline-0). The score between them depends on $m - n$ only, through $\cos((m-n)\theta_i)$ and $\sin((m-n)\theta_i)$. Each pair is a clock with **wavelength**

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi b^{2i/D} \quad \text{tokens}$$

(YaRN Eq. 8). Over the trained length $L$ the pair turns

$$r_i = \frac{L}{\lambda_i}$$

times (YaRN Eq. 10). Fast pairs ($i$ small) turn many times: the model has seen every angle, at every relative distance up to $L$. Slow pairs ($r_i < 1$) never completed a turn: at relative distance $L$ their angle was $2\pi r_i$, and every larger distance produces an angle the model never saw. That is the failure: past $L$, the slow pairs report positions outside their training range, attention scores on those channels take values the model never learned to handle, and loss rises (the base model in 04.1 lost 0.13 nats on far targets at 2,048 when its full context was used).

Symbols used below: $s = L'/L$ the scale factor (new length over trained length), $\theta'_i$ the new frequency of pair $i$, and the **divisor** $\theta_i / \theta'_i$ (1 = unchanged, $s$ = stretched fully).

### Position Interpolation (PI): stretch every clock

Divide every position by $s$, which is the same as dividing every frequency by $s$:

$$\theta'_i = \theta_i / s \quad \text{for all } i.$$

Now distance $L'$ produces exactly the angles distance $L$ used to, so no pair leaves its range. Chen et al. extend LLaMA models "to up to 32768 with minimal fine-tuning (within 1000 steps)" and show that "the upper bound of interpolation is at least ~600 × smaller than that of extrapolation" (PI, abstract). The cost: fast pairs are slowed too. Two neighbouring tokens that pair 0 used to separate by 1 radian are now separated by $1/s$ radian, so the model's sense of *local* order is blurred until it is retrained.

### NTK-aware scaling: stretch the slow clocks more

Change the base instead, $b' = b \cdot s^{D/(D-2)}$ (YaRN Appendix A.2, Eq. 19), and recompute $\theta'_i = b'^{-2i/D}$. The divisor of pair $i$ is $s^{2i/(D-2)}$: exactly 1 for the fastest pair ($i = 0$) and exactly $s$ for the slowest ($i = D/2 - 1$), geometric in between. Local resolution is mostly kept. The cost: middle pairs are stretched by less than $s$, so some of them still see angles past their training range at length $L'$.

### YaRN: decide per clock by how many turns it made

YaRN's "NTK-by-parts" rule (section 3.2) uses $r_i$ directly. With two thresholds $\alpha < \beta$:

$$\gamma(r) = \begin{cases} 0 & r < \alpha \\ 1 & r > \beta \\ \dfrac{r - \alpha}{\beta - \alpha} & \text{otherwise} \end{cases} \qquad \theta'_i = (1 - \gamma(r_i))\,\frac{\theta_i}{s} + \gamma(r_i)\,\theta_i$$

(YaRN Eqs. 11 and 13). Pairs that turned more than $\beta$ times are left alone (they already saw all angles); pairs that turned fewer than $\alpha$ times are interpolated like PI; pairs in between are blended. The paper recommends "$\alpha = 1$ and $\beta = 32$" for the Llama family (section 3.2).

YaRN adds an **attention temperature** (section 3.3): $\text{softmax}(q_m^\top k_n / (t\sqrt{D}))$ with

$$\sqrt{1/t} = 0.1 \ln s + 1$$

(Eqs. 14–15). It is implemented by multiplying $q$ and $k$ (in practice, cos and sin) by $\sqrt{1/t}$, so the logits are multiplied by $(0.1\ln s + 1)^2$. Interpolation packs more keys into the same angle range and flattens the attention distribution; the temperature sharpens it back. The paper's training: Llama 2 7B and 13B fine-tuned with $s = 16$ "for 400 steps with global batch size 64" on 64K-token PG19 chunks, then $s = 32$ for "only an additional 200 steps" (section 4.1), "less than ~0.1% of the original pre-training data" (section 1).

### The paper's ramp and the code's ramp are not the same function

The YaRN authors' code, and Transformers 5.18.0 (`_compute_yarn_parameters`), do not evaluate $\gamma$ at $r_i$. They compute the real-valued pair indices where $r = \beta$ and $r = \alpha$,

$$d(n) = \frac{D \ln\!\big(L / (2\pi n)\big)}{2 \ln b},$$

round them outwards ($\lfloor d(\beta) \rfloor$, $\lceil d(\alpha) \rceil$, unless `truncate` is false, as in gpt-oss's config), and ramp *linearly in the pair index* between them. Both agree at the ends; in the middle the code interpolates less. Released checkpoints run the code's version, so `frontierlab.longctx.rope` uses it by default and keeps the paper's as `ramp="paper"`. In Transformers, the rule also applies to all heads' rotated channels only; channels that partial RoPE leaves unrotated are not scaled by the temperature.

### Base-frequency adjustment and Llama 3.1's rope type

Two related practices. **Raise the base** before or during training: Llama 3 "increase[s] the RoPE base frequency hyperparameter to 500,000" (Llama 3, section 3.2), which makes every clock slower so that more of them cover a long length. And Llama 3.1's checkpoints use a rope type that Transformers implements as a wavelength threshold version of the same per-frequency idea (`_compute_llama3_parameters`: unchanged below wavelength $L/4$, divided by $s$ above $L$, smooth in between; the code comments name factor 8 and original length 8,192 as the original implementation's values). `frontierlab.longctx.rope.llama3_inv_freq` reproduces it and is tested against Transformers.

### Choices that need training from the start

**Partial RoPE** rotates only some channels of each head and leaves the rest without any position signal. Two conventions are in use, and configs do not always say which. In the GPT-NeoX convention, which Transformers applies to `partial_rotary_factor` and Qwen3-Next uses (0.25 with head dimension 256, so 64 channels rotate), the rotated channels get frequencies $b^{-2i/(pD)}$ recomputed over the $pD$ rotated channels: fewer clocks, but still spanning the full range from fast to slow. In **p-RoPE** (Barbero et al.), the standard frequencies are kept for the fastest fraction $p$ of pairs and the slowest $1 - p$ are set to zero, so exactly the clocks that break past $L$ are removed (Transformers' `proportional` type; `RopeScaling(type="proportional")` here). Barbero et al. report that Gemma 7B uses RoPE's lowest frequencies as "semantic" channels and propose p-RoPE as a useful modification (abstract and the body). Either way the unrotated channels behave like position-free (NoPE) channels, which cannot break at any length. And either way it changes which channels carry position, so a model trained with full RoPE cannot be switched to partial RoPE without retraining; `frontierlab.longctx.extend` refuses to.

**Interleaved NoPE layers** remove positional encoding from some layers entirely; the causal mask alone still lets a decoder infer order. Kazemnejad et al. found that with no positional encoding (NoPE), decoder-only models generalised to longer inputs better than with explicit encodings on their reasoning and arithmetic tasks (abstract). Yang et al. interleave three sliding-window RoPE layers with one full-attention NoPE layer and report that "a 1:3 ratio strikes an optimal balance" (section 3). Meta's Llama 4 blog describes "interleaved attention layers without positional embeddings" plus "inference time temperature scaling of attention to enhance length generalization", calls it iRoPE, and says the "i" stands for "interleaved" with "the long-term goal of supporting 'infinite' context length". The blog gives neither the ratio nor the formula. Hugging Face's Llama 4 port defaults to a NoPE layer every fourth layer (`no_rope_layer_interval=4`) and multiplies NoPE layers' queries by $1 + 0.1\ln(1 + \lfloor (m+1)/8192 \rfloor)$; the course's `"gqa-irope"` kind follows those defaults and labels them as one implementation's choices.

## Worked example

### Four clocks, by hand

Take $D = 8$ (four pairs), $b = 10{,}000$, trained at $L = 256$, extended to $L' = 1{,}024$ ($s = 4$). Then $\theta = (1, 0.1, 0.01, 0.001)$, $\lambda = (6.3, 62.8, 628, 6{,}283)$ tokens and $r = L/\lambda = (40.7, 4.07, 0.41, 0.04)$ turns.

- **Unscaled at 1,024:** pair 3's largest angle in training was $256 \times 0.001 = 0.256$ rad; at distance 1,000 it is 1.0 rad, four times past anything seen. Pair 2: 2.56 rad in training, 10 rad now (it has wrapped, but the pattern of angles across pairs is new).
- **PI:** divisors $(4, 4, 4, 4)$. Pair 0 now separates neighbours by 0.25 rad instead of 1.
- **NTK-aware:** $b' = 10{,}000 \cdot 4^{8/6} = 63{,}496$. Divisors $4^{2i/6} = (1, 1.587, 2.520, 4)$.
- **YaRN, paper ramp** ($\alpha = 1$, $\beta = 32$): $\gamma = (1,\ (4.07 - 1)/31 = 0.099,\ 0,\ 0)$. Pair 1: $\theta' = (1 - 0.099) \cdot 0.1/4 + 0.099 \cdot 0.1 = 0.0324$, divisor $3.08$. Divisors $(1, 3.08, 4, 4)$.
- **YaRN, code ramp:** $d(32) = 8 \ln(256/(2\pi \cdot 32)) / (2 \ln 10^4) = 0.105 \to 0$; $d(1) = 8 \ln(256/2\pi)/(2 \ln 10^4) = 1.610 \to 2$. Ramp over indices 0..2: $\gamma = (1, 0.5, 0, 0)$, divisors $(1, 1.6, 4, 4)$.
- **Temperature:** $0.1 \ln 4 + 1 = 1.1386$; logits $\times 1.2965$.

`python labs/module-04/lesson-02/freq_table.py --dim 8` prints exactly these columns. For Baseline-0 ($D = 64$, $L = 1{,}024$, $s = 32$) the code ramp leaves pairs 0–5 unchanged (more than 32 turns), interpolates pairs 18–31 fully (less than one turn), blends pairs 6–17, and multiplies logits by $1.3466^2 = 1.813$.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| `inv_freq` ($\theta'$) | (D/2,) = (32,) for Baseline-0 | computed in float64, stored float32 | buffer on the model's device |
| angles $m\theta'_i$ | (T, D/2) | float32 | same |
| cos, sin (× attention factor) | (T, D) | float32, cast to q's dtype in `apply_rope` | same |
| q, k after RoPE | (B, H or KV, T, hd) | bf16 on GPU, fp32 / fp64 on CPU | same |

Partial RoPE alone (no scaling rule) is also Module 3's `"gqa_partial"` kind, introduced in [03.4](../module-03/lesson-04.md) as a head-shape choice; here it is one option among the long-range rules.

The rules cost nothing at run time: they change a 32-element vector once. Parameters and their names are identical (`inv_freq` is a non-persistent buffer), so a Baseline-0 checkpoint loads into `"gqa-rope-scaled"` with `strict=True`. Two numerical points: angles are computed in float32, so at position 32,768 the angle of pair 0 has an absolute rounding error of about $32{,}768 \times 2^{-24} \approx 0.002$ rad (Transformers computes them in float32 too); and with bf16 autocast, q and k are rotated in bf16, which is the course's main-path setting and the same as production code.

## Build it

```python
import frontierlab.longctx                            # registers "gqa-rope-scaled" and "gqa-irope"
from frontierlab.longctx import convert, rope_extra
from frontierlab.evals.suite_v1 import load_model

base = load_model("runs/m04/base-cpu")                # trained at 256 with plain RoPE
yarn = convert(base, "gqa-rope-scaled", **rope_extra("yarn", factor=4, original=256))
print(yarn.model.layers[0].self_attn.rope.inv_freq[:4], yarn.model.layers[0].self_attn.rope.attention_factor)
```

`ScaledRotaryEmbedding` returns $\cos \cdot a$ and $\sin \cdot a$; `apply_rope` (unchanged, from `attention/base.py`) rotates the first `rot_dim` channels of q and k with them, so both are scaled by $a$ and their dot product by $a^2$. The iRoPE kind takes `layer_idx`; layer $i$ is NoPE when $(i + 1) \bmod 4 = 0$, and a NoPE layer skips the rotation and, if `temperature` is on, multiplies its queries by the position-dependent factor above, which depends only on each query's absolute position and so is identical in a full forward and in cached decoding.

Correctness checks (`labs/common/tests/test_longctx.py`): YaRN frequencies and attention factor match Transformers within $10^{-6}$ relative for five configurations (including gpt-oss's `truncate: false` and Qwen3's 32,768 × 4) and for partial RoPE; PI equals Transformers' `linear`; static NTK-aware equals Transformers' `dynamic` evaluated at $L' = sL$; Llama 3.1's type matches; for every rule the score between rotated q and k depends only on $m - n$ (within $10^{-6}$ in float64) and equals $a^2 q^\top k$ at distance 0; every kind passes the causal and cached-decode checks to $10^{-9}$ in float64 and a float64 gradient check; and the default rule reproduces Baseline-0's logits bit for bit.

## What the evidence says

- **Interpolating positions instead of extrapolating them: ESTABLISHED.** PI and YaRN are in published papers with ablations, and YaRN with stated parameters is in the configs of DeepSeek-V3 (section 4.3), Qwen3 (model card) and gpt-oss (config.json), all PUBLICLY DOCUMENTED.
- **YaRN's per-frequency rule and temperature: ESTABLISHED in practice** (several labs ship it); the 0.1 ln s + 1 form is the paper's fit for Llama models (Eq. 15), and DeepSeek-V3 uses the same form (section 4.3). Whether the same constants are best for other models is an open question.
- **Static scaling costs short-context quality.** Qwen's card warns that static YaRN means "the scaling factor remains constant regardless of input length, potentially impacting performance on shorter texts" (company claim). The CPU measurement below shows the same direction.
- **Partial RoPE and p-RoPE: PROMISING.** Partial RoPE is in Qwen3-Next's config (PUBLICLY DOCUMENTED); p-RoPE is argued for by Barbero et al.; few controlled comparisons at scale, and almost none that separate the two conventions.
- **Interleaved NoPE layers (iRoPE): MODEL-SPECIFIC.** Llama 4 as described in Meta's blog (company claim, no ablations published there); one independent study (Yang et al.) supports the 1:3 layout under its setup.
- **Measured at course scale** (CPU base model of 04.1, trained at 256, evaluated zero-shot at 1,024 on 200 held-out documents; 2026-10-03): see the lab's reference results. The direction matches the papers; the size says nothing about larger models.

## Lab

**Folder:** [`labs/module-04/lesson-02/`](../../labs/module-04/) · **Time:** about 90 minutes · **Pass check:** `pytest labs/module-04/lesson-02` passes (including the Transformers comparison); `zero_shot.py` runs with your `lab.py`; your write-up applies the contract's rule and names the rule 04.3 will use.

### Experiment contract

- **Question:** applied to the trained base model without any training, which RoPE rule gives the lowest loss on document positions past the trained length, and what does each cost below it? Decision informed: the rule used for continued training in 04.3.
- **Hypothesis:** PI costs most at short range; NTK-aware costs little at short range but helps less far away; YaRN helps most far away at a small short-range cost; YaRN without its temperature is worse than with it. Status: reported effects (YaRN paper, sections 3 and 4) at 7B–13B scale; may differ at 1.8M parameters.
- **Baseline:** the unchanged model ("none"), `runs/m04/base-cpu`.
- **Changed variable:** the RoPE rule (none, PI, NTK-aware, YaRN, YaRN without temperature), all with $s = 4$ (1,024 / 256). **Controlled:** the same weights (no training), the same 200 held-out documents (first 1,024 tokens of each), the same position buckets.
- **Comparison axis:** equal everything; the rules are free at run time, so no budget axis applies.
- **Budget:** free CPU, about 3 minutes per evaluation length (measured below).
- **Metrics and decision rule:** primary: mean loss on positions $[512, 1023)$ minus the unchanged model's, paired by document, 95% bootstrap CI. Guard: the same difference on positions $[0, 256)$ (inside the trained length) must have an upper bound $\leq 0.05$ nats. Rule: choose the rule with the most negative primary difference among those that pass the guard; if no rule's primary interval lies below zero, keep "none" and say so.
- **Correctness checks:** all lab tests pass, including the match with Transformers; the default rule reproduces the base model's logits exactly (common tests).
- **Fallback evidence:** none needed.
- **Limits:** one small model, one scale factor, no training: a rule that loses here can win after the continued training of 04.3 (PI's original claim is about fine-tuned models).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100, about 20 GPU-minutes. Not run in this build; part of the Module 4 pilot | `zero_shot.py --run <Baseline-0 seed-0 run> --train-len 1024 --eval-len 8192 --device cuda --bf16 --data labs/common/data/v0-long --out runs/m04/b0-zero-shot-8192.json`, and again with `--eval-len 32768 --max-docs 50` |
| Free GPU (Colab/Kaggle T4) | T4, about 10 minutes | `--run runs/m04/base-t4 --train-len 512 --eval-len 2048 --device cuda` |
| Free CPU | laptop; measured 2026-10-03: 193 s at `--eval-len 1024`, 498 s at 2,048 (16 threads, other jobs running); optional architecture runs 15–18 minutes each | the steps below |

### Steps

1. **Implement** `ntk_inv_freq`, `rotations`, `yarn_gamma`, `yarn_inv_freq` and `yarn_attention_factor` in `lab.py`; run `pytest labs/module-04/lesson-02`.
2. **Print the frequency table** for the worked example and for the base model, and check one row of each by hand:

   ```bash
   python labs/module-04/lesson-02/freq_table.py --dim 8
   python labs/module-04/lesson-02/freq_table.py --dim 32 --train-len 256 --factor 4
   ```

3. **Run the zero-shot comparison** with your frequencies:

   ```bash
   python labs/module-04/lesson-02/zero_shot.py --run runs/m04/base-cpu --train-len 256 --eval-len 1024 --out runs/m04/zero-shot-1024.json
   ```

   Then repeat with `--eval-len 2048` ($s = 8$).
4. **Write up:** apply the contract's rule at $s = 4$; compare with $s = 8$; explain PI's short-range cost from the frequency table (which pairs did it slow that it did not need to?); explain what the temperature ablation shows; say what you expect continued training to change.
5. **Optional, about 35 minutes unattended: architectures that need training.** Train two more base models with the same recipe, one with partial RoPE ($p = 0.5$) and one with a NoPE layer every fourth layer, then compare their zero-shot behaviour at 1,024 with the base model's:

   ```bash
   python -m frontierlab.longctx.extend --partial 0.5 --run runs/m04/arch-partial --preset toy --seq 256 --batch 16 --steps 1500 --lr 3e-3 --warmup 50
   python -m frontierlab.longctx.extend --nope-every 4 --run runs/m04/arch-irope --preset toy --seq 256 --batch 16 --steps 1500 --lr 3e-3 --warmup 50
   python labs/module-04/lesson-02/zero_shot.py --run runs/m04/arch-partial --train-len 256 --eval-len 1024 --rules none yarn
   python labs/module-04/lesson-02/zero_shot.py --run runs/m04/arch-irope --train-len 256 --eval-len 1024 --rules none yarn
   ```

   These are separate trained models (one seed each), so differences between them include seed noise of the size measured in lesson 01.4; treat them as a pilot, not a result.

   <details>
   <summary>What the build's optional runs gave</summary>

   Measured 2026-10-03/04 (training 18 and 14.5 minutes; each `zero_shot.py` call under a minute with two rules). Validation loss at 256 after 1,500 steps: base 4.975, partial RoPE 4.942, NoPE every fourth layer 4.942. Mean loss on positions $[512, 1023)$ at 1,024, unscaled / with YaRN ($s = 4$): base 5.221 / 5.105; partial RoPE 5.522 / 5.063; NoPE interleave 5.316 / 5.076. So in this one-seed pilot neither design extrapolated better *without* a rule (both were worse than the base past 256), and both did slightly better than the base once YaRN was applied, with the same short-range cost pattern (+0.02 to +0.04). Two lessons: partial RoPE in the NeoX convention still has slow clocks among its rotated channels, so it is not a substitute for a scaling rule; and a NoPE layer among three RoPE layers does not by itself make a 1.8M-parameter model length-generalise. Differences of 0.03 nats between separately trained models are within plausible seed noise at this size.

   </details>

<details>
<summary>Hint for TODO 3</summary>

Compute $d(\beta_{\text{fast}})$ and $d(\beta_{\text{slow}})$ with `math.log`, floor the first and ceil the second, clamp to $[0, D-1]$, then `torch.arange(D // 2)` gives the pair indices for the ramp. $\beta_{\text{fast}} = 32$ gives the *smaller* index, because fast pairs have small $i$.

</details>

<details>
<summary>What the build's run gave at s = 4 (compare after your write-up)</summary>

Base model trained at 256, evaluated at 1,024 on 200 documents from `v0-long` (measured 2026-10-03, 193 s). Mean loss by document position, then the paired difference to "none" (95% CI):

| Rule | [0, 64) | [64, 128) | [128, 256) | [256, 512) | [512, 1023) | gain W = 256 |
|---|---|---|---|---|---|---|
| none | 5.229 | 5.023 | 5.054 | 5.114 | 5.221 | −0.063 |
| PI | +0.197 [+0.183, +0.211] | +0.237 | +0.247 | +0.197 | +0.097 [+0.089, +0.104] | −0.002 |
| NTK-aware | +0.010 [+0.007, +0.013] | +0.018 | +0.020 | −0.027 | −0.037 [−0.041, −0.033] | −0.067 |
| YaRN | +0.025 [+0.021, +0.029] | +0.037 | +0.037 [+0.033, +0.041] | −0.014 | −0.116 [−0.124, −0.108] | −0.001 |
| YaRN, no temperature | +0.020 [+0.016, +0.025] | +0.047 | +0.064 [+0.059, +0.069] | +0.023 | −0.071 [−0.077, −0.065] | −0.006 |

By the contract's rule YaRN is chosen: the most negative far-position difference ($-0.116$) with a short-range cost whose upper bound (0.043 on $[128, 256)$) is under the 0.05 guard. PI fails the guard by a wide margin; without training it is worse than doing nothing at every position. The temperature is worth 0.045 nats far away and 0.027 at $[128, 256)$. No rule makes the far context *useful* yet: YaRN's context gain is $-0.001$ (the far tokens stop hurting) against $-0.063$ unscaled. That is what continued training (04.3) has to change.

At $s = 8$ (evaluated at 2,048, 200 documents, 498 s) the trade-off sharpens. YaRN's far-position gain grows ($-0.221$, CI $[-0.232, -0.210]$ on $[1024, 2047)$) but so does its short-range cost ($+0.054$, CI $[+0.049, +0.060]$ on $[128, 256)$, and $+0.058$ with upper bound $0.066$ on $[64, 128)$), so it fails the 0.05 guard. NTK-aware passes the guard (upper bound 0.047) with a far gain of $-0.031$, and the rule as written picks it. Whether that is the right decision depends on what the guard protects; the point of writing the guard down first is that you cannot move it after seeing this table. PI costs $+0.36$ at short range. The temperature ablation again favours the temperature ($-0.221$ against $-0.116$ far away).

</details>

<details>
<summary>Reference solution</summary>

`labs/module-04/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-04/lesson-02`.

</details>

## Common mistakes

- **Setting `original_max_position_embeddings` to the new length.** Then $r_i$ is computed over the wrong length and YaRN interpolates the wrong pairs. It is the length the model was trained at.
- **Comparing YaRN implementations without saying which ramp.** The paper's and the code's ramp give different frequencies for the middle pairs (the table's last two columns).
- **Forgetting the temperature, or applying it twice.** Transformers folds it into cos and sin; if you also scale the logits, the effective factor is cubed.
- **Switching on partial RoPE for an extension.** It changes what the trained channels mean; it is a from-scratch architecture choice.
- **Reading `partial_rotary_factor` as p-RoPE.** In Transformers' default convention the rotated channels still include slow frequencies; p-RoPE (`proportional`) is the variant that drops them.
- **Using a static factor and reporting short-context scores from the unscaled model.** The served model is the scaled one; measure its short-context quality (Eval v0) with the factor on.
- **Assuming Llama 4's iRoPE layout and temperature formula are documented by Meta.** The blog names the idea; the ratio and formula in this course come from Hugging Face's implementation.

## References

- S. Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation*, 2023, abstract. https://arxiv.org/abs/2306.15595
- B. Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, v3, sections 3.1–3.4 (Eqs. 8–15), 4.1 and Appendix A (Eqs. 16–19). https://arxiv.org/abs/2309.00071
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, section 4.3. https://arxiv.org/abs/2412.19437
- Qwen, *Qwen3-8B model card* (processing long texts). https://huggingface.co/Qwen/Qwen3-8B
- OpenAI, *gpt-oss-20b config.json* (`rope_scaling`). https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
- Qwen, *Qwen3-Next-80B-A3B-Instruct* config.json and model card. https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- F. Barbero et al., *Round and Round We Go! What makes Rotary Positional Encodings useful?*, 2024. https://arxiv.org/abs/2410.06205
- A. Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers*, 2023. https://arxiv.org/abs/2305.19466
- B. Yang et al., *Rope to Nope and Back Again: A New Hybrid Attention Strategy*, 2025, section 3. https://arxiv.org/abs/2501.18795
- Meta AI, *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation*, 2025-04-05. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Llama Team, Meta, *The Llama 3 Herd of Models*, section 3.2. https://arxiv.org/abs/2407.21783
- Hugging Face Transformers 5.18.0, `modeling_rope_utils.py` and `models/llama4/`. https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/modeling_rope_utils.py

## Next

[04.3 · Extending context by continued training](lesson-03.md)
