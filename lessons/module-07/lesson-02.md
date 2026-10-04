---
id: "07.2"
module: 7
minutes: 40
practice_minutes: 90
prerequisites: ["07.1", "03.1", "03.3"]
objectives:
  - Derive the RMS of an orthogonalised update for a matrix of any shape, and the factor that matches it to AdamW's update RMS (Moonlight Eq. 4).
  - Explain why Muon needed weight decay at scale and what the matched scaling lets a team reuse.
  - Implement QK-Clip per head for the course's GQA and MLA kinds, including which MLA parts are scaled and which are left alone, and verify that the clipped logits equal the threshold.
  - Compare QK-Clip with QK-norm as fixes for attention-logit growth (Kimi K2 vs DeepSeek-V4), and state what a course-scale run can and cannot show about either.
volatility: concept
sources:
  - title: "Moonshot AI — Muon is Scalable for LLM Training (Moonlight), section 2.2 (Eq. 3, Eq. 4, Lemma 1), section 3.2, appendix D"
    url: https://arxiv.org/abs/2502.16982
  - title: "Moonshot AI — Kimi K2: Open Agentic Intelligence, section 2.1 (MuonClip, QK-Clip, tau = 100, Figure 2)"
    url: https://arxiv.org/abs/2507.20534
  - title: "DeepSeek-AI — DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, sections 2.3.3 and 2.4"
    url: https://arxiv.org/abs/2606.19348
  - title: "DeepSeek-AI — DeepSeek-V2, section 2.1 (Multi-head Latent Attention)"
    url: https://arxiv.org/abs/2405.04434
last_verified: "2026-10-04"
---

# 07.2 · Muon at scale

Muon at 300 steps on a laptop and Muon at 15 trillion tokens are not the same optimizer problem. At scale, two things went wrong for the teams that tried it: the update size depended on each matrix's shape, so AdamW's carefully tuned learning rate and weight decay could not be reused, and attention logits grew until training became unstable. Moonshot fixed the first with weight decay and update-RMS matching (Moonlight), and the second with QK-Clip, a per-head rescaling of the query and key weights after every step (Kimi K2). DeepSeek-V4 took a different route for the second: it normalises queries and keys instead. This lesson derives the scaling, implements QK-Clip for the course's GQA and MLA attention, and runs the logit-growth experiment at course scale — where, as you will see, the effect does not look the way it does at a trillion parameters.

## Why this matters at a frontier lab

Re-tuning a learning rate at scale costs a run. Moonlight's update-RMS matching exists so that "Muon can directly reuse the learning rate and weight decay tuned for AdamW" (section 2.2) — the difference between "try Muon in the next run" and "start a tuning campaign". Logit growth is the failure that decides whether a run finishes: Kimi K2 reports that in mid-scale experiments (9B activated, 53B total parameters) logits "exceed 1000" without intervention, that logit explosion "occurs more frequently with Muon but less with AdamW" in their experiments, and that with QK-Clip the 15.5T-token pretraining had zero loss spikes (section 2.1, Figure 2, abstract). When you propose Muon for Recipe-R, a reviewer will ask both questions: which hyperparameters carry over, and what stops the logits.

## The idea

### The RMS of an orthogonal update

For a full-rank matrix $O \in \mathbb{R}^{A \times B}$ with all $\min(A, B)$ singular values equal to 1, $\|O\|_F^2 = \sum_i s_i^2 = \min(A, B)$, so

$$\mathrm{RMS}(O) = \sqrt{\frac{\min(A,B)}{A B}} = \frac{1}{\sqrt{\max(A,B)}}$$

(Moonlight Lemma 1). A $768 \times 768$ attention projection gets updates of RMS $1/\sqrt{768} = 0.036$, a $2816 \times 768$ SwiGLU matrix $1/\sqrt{2816} = 0.019$ — a different effective learning rate for every shape, and neither close to AdamW's.

### Weight decay and update-RMS matching

Moonlight made two changes (section 2.2). **Weight decay** (Eq. 3): $W_t = W_{t-1} - \eta_t (O_t + \lambda W_{t-1})$, $\lambda = 0.1$. Without it, in an 800M model trained on 100B tokens, "both the weight and the layer output's RMS keep growing to a large scale, exceeding the high-precision range of bf16". **Matching** (Eq. 4):

$$W_t = W_{t-1} - \eta_t\big(0.2 \cdot O_t \cdot \sqrt{\max(A, B)} + \lambda W_{t-1}\big),$$

which makes the update RMS $0.2$ for every shape, because "AdamW's update RMS is usually around 0.2 to 0.4". One $\eta$ and one $\lambda$ now mean the same thing for Muon's matrices and for the AdamW groups (embedding, norms), which is also why one learning-rate schedule can drive the whole `MuonAdamW` optimizer. Jordan's original scaling $\sqrt{\max(1, A/B)}$ instead normalises the *spectral* size of the update (lesson 07.3); it needs its own learning rate (0.02 in his code).

### Why logits grow, and why Muon might make it worse

A head's logit $\ell_{ij} = x_i^\top W_Q^{h\top} W_K^{h} x_j / \sqrt d$ is bilinear in the two weight matrices; nothing in the loss stops their norms from growing together (lesson 03.3). Muon's update has *all* singular values near 1, so it grows every direction of $W_Q$ and $W_K$ at a similar rate, including directions AdamW would barely move. Kimi K2 gives its observation, not a proof: explosion "occurs more frequently with Muon but less with AdamW in our experiments". Moonlight's appendix D also reports that with Muon the maximum attention logit "exhibited a distinct upward trajectory … exceeding a threshold of 100", though such extreme logits were sparse (about $10^{-4}$ of them).

### QK-Clip

Kimi K2 (section 2.1) uses the maximum logit per head on the current batch, "already computed during forward",

$$S_{\max}^h = \frac{1}{\sqrt d} \max_{X \in \text{batch}} \max_{i, j} q_i^h \cdot k_j^h, \qquad \gamma_h = \min\left(1, \frac{\tau}{S_{\max}^h}\right),$$

and after the optimizer step rescales the weights of every head with $\gamma_h < 1$ ("we … opt to apply per-head QK-Clip"). The forward and backward pass are unchanged; only the weights are rescaled. For multi-head attention $W_q^h$ and $W_k^h$ are each scaled by $\sqrt{\gamma_h}$, so the logit scales by $\gamma_h$. For **MLA** (lesson 03.1) the key has a head-specific part and a part shared by all heads:

| MLA component | QK-Clip factor | Why |
|---|---|---|
| $q^C$, non-rotary query of head $h$ | $\sqrt{\gamma_h}$ | head-specific, pairs with $k^C$ |
| $k^C = W_{UK}^h c$, non-rotary key of head $h$ | $\sqrt{\gamma_h}$ | head-specific |
| $q^R$, rotary query of head $h$ | $\gamma_h$ | head-specific; its partner $k^R$ is shared |
| $k^R$, the shared rotary key | untouched | scaling it would change every head's logits |

Kimi K2 used $\tau = 100$; its Figure 2 shows the max logit held at 100 early in training and then decaying on its own after roughly 30% of the steps, at which point the clip stops acting.

**GQA** is not covered by the paper: several query heads share one key head. The course follows the MLA rule for shared parts — scale query head $h$ by $\gamma_h$ and leave the shared key alone (INFERENCE, our choice; scaling the shared key would also shrink the other heads' logits).

### Why not QK-norm? And why DeepSeek-V4 chose it anyway

QK-norm bounds the logit by the norm gains (lesson 03.3) — but it needs the per-head keys. Kimi K2: "QK-Norm is not applicable to multi-head latent attention (MLA), because its Key matrices are not fully materialized during inference" (with weight absorption, MLA attends over the latent). Soft-capping is not a substitute either: "the dot products between queries and keys can still grow excessively before capping is applied". DeepSeek-V4's attention (CSA/HCA) keeps a single compressed KV entry per token, and V4 "perform[s] an additional RMSNorm operation on each head of the queries and the only head of the compressed KV entries, just before the core attention operation" (section 2.3.3), and therefore states it does not need QK-Clip: this RMSNorm "effectively prevents attention logits from exploding" (section 2.4). Same problem, two answers, each matched to its attention design. V4 also runs its 10-step hybrid Newton–Schulz (lesson 07.1) and rescales the update RMS "for reutilization of [its] AdamW hyper-parameters", as Moonlight does.

## Worked example

### Matching for Baseline-0's shapes

| Matrix (A, B) | RMS of $O$ | Jordan's scale $\sqrt{\max(1, A/B)}$ | update RMS (Jordan) | Moonlight's scale $0.2\sqrt{\max(A,B)}$ | update RMS (Moonlight) |
|---|---|---|---|---|---|
| q_proj (768, 768) | 0.0361 | 1 | 0.0361 | 5.54 | 0.2 |
| k_proj (256, 768) | 0.0361 | 1 | 0.0361 | 5.54 | 0.2 |
| up_proj (2816, 768) | 0.0188 | 1.915 | 0.0361 | 10.61 | 0.2 |
| down_proj (768, 2816) | 0.0188 | 1 | 0.0188 | 10.61 | 0.2 |

Jordan's scale equalises update RMS only for tall matrices; Moonlight's equalises it for every shape.

### QK-Clip on one head, by hand

$\tau = 100$, $S_{\max}^h = 400$: $\gamma_h = 0.25$. In MHA, $W_q^h$ and $W_k^h$ are each multiplied by $\sqrt{0.25} = 0.5$: the logit that was 400 becomes $0.5 \cdot 0.5 \cdot 400 = 100$. In MLA, if that logit was $q^C \cdot k^C = 320$ plus $q^R \cdot k^R = 80$ (before the $1/\sqrt d$, say), the new parts are $0.5 \cdot 0.5 \cdot 320 = 80$ and $0.25 \cdot 1 \cdot 80 = 20$: total 100 again, with the shared $k^R$ unchanged. A head with $S_{\max}^h = 60$ has $\gamma_h = 1$ and is not touched.

## Shapes and cost

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| GQA `q_proj.weight` | (H·d, C) | fp32 | rows $hd \ldots (h+1)d - 1$ belong to query head $h$ |
| GQA `k_proj.weight` | (KV·d, C) | fp32 | key head $g$ serves query heads $g \cdot H/KV \ldots$ |
| MLA `q_proj.weight` (or `q_b_proj`) | (H·(d_n + d_r), C or r_q) | fp32 | per head: $d_n$ rows of $q^C$, then $d_r$ rows of $q^R$ |
| MLA `kv_b_proj.weight` | (H·(d_n + d_v), d_c) | fp32 | per head: $d_n$ rows of $W_{UK}$, then $d_v$ rows of $W_{UV}$ (untouched) |
| MLA `kv_a_proj_with_mqa.weight` | (d_c + d_r, C) | fp32 | the last $d_r$ rows make the shared $k^R$ (untouched) |
| per-head $S_{\max}$ | (H,) | fp32 | max over the batch, all layers kept separately |

Cost: the rescale is $O(\text{weights})$ per step, negligible. Getting $S_{\max}$ is the real cost: in production it comes from the attention kernel's running maximum; here a forward pre-hook recomputes $q k^\top$ one sequence at a time, $2 B H T^2 d$ FLOPs per layer (67M per layer for the toy model at $B = 16$, $T = 128$; $3.4 \times 10^{10}$ per layer for `pilot-30m` at $B = 32$, $T = 1{,}024$, so $3.4 \times 10^{11}$ for its 10 layers — about 3% of that step's $1.05 \times 10^{13}$ training FLOPs, PROJECTED from the formula; on the main path, log it every few steps if that matters).

## Build it

`labs/common/frontierlab/optim/qkclip.py`:

- `head_max_logits(mod, x, positions)` recomputes one layer's per-head maximum from its own projections (GQA, the Module 3 windowed/sink/gated kinds, MLA), with the same mask and scale the layer uses.
- `LogitMonitor(model)` registers the pre-hooks; only training forwards count (evaluation never clips), and with gradient accumulation the maximum is over all micro-batches of the step.
- `clip_module(mod, gamma)` applies the table above; `QKClip(model, tau).attach(opt)` runs it as a post-step hook of `MuonAdamW`. It refuses a model with QK-norm: the RMSNorm after the projections would undo the rescale.

From the command line: `python -m frontierlab.optim.train --optimizer muon --qk-norm off --qk-clip 100 ...`.

Correctness checks (`tests/test_optim.py`, all passing): for MHA, GQA, MLA and MLA with a query latent, in float64, after the clip the recomputed $S_{\max}^h$ on the same layer inputs equals $\min(S_{\max}^h, \tau)$ to $10^{-9}$; the monitor's maximum matches the Module 3 probe (`probes.report`) on the same batch; exact resume holds with QK-Clip and the stability log on; update-RMS matching gives $0.2 \pm 0.005$ for three shapes.

## What the evidence says

- **Weight decay and update-RMS matching — ESTABLISHED in open Muon recipes.** PUBLICLY DOCUMENTED by Moonlight (section 2.2), adopted by Kimi K2 (Algorithm 1) and DeepSeek-V4 (section 2.4); `torch.optim.Muon` ships the option (`adjust_lr_fn="match_rms_adamw"`).
- **QK-Clip — PROMISING / MODEL-SPECIFIC.** One lab's mechanism (Kimi K2), documented with a mid-scale ablation and one very large run. $\tau = 100$ is their value for their model; the right $\tau$ for another architecture is not published.
- **RMSNorm on queries and KV entries (DeepSeek-V4) — ESTABLISHED mechanism** (QK-norm, lesson 03.3), MODEL-SPECIFIC placement in V4's compressed attention.
- **"Muon makes logits grow more" — company claim**, observed by Moonshot in their experiments; not, as far as we found, measured independently.
- **Course-scale hypotheses** (lab): (H1) without QK-norm, Muon's max logit grows at least 3× higher than with QK-norm; (H2) QK-Clip holds the max logit at about $\tau$ when the unclipped run passes $\tau$, without hurting held-out loss by more than 0.02 nats; (H3) with QK-norm the max logit stays bounded as in lesson 03.3. Kimi's effect appeared at 9B activated parameters; at 1.8M it may not appear at all, and Kimi's $\tau = 100$ may never be reached.

## Lab

**Folder:** [`labs/module-07/lesson-02/`](../../labs/module-07/) · **Time:** about 90 minutes (about 45 of them unattended) · **Pass check:** `pytest labs/module-07/lesson-02` passes; `rms_check.py` and `compare_arms.py` run; your notes give each hypothesis a verdict with numbers, and say which conclusions need the pilot's 30M/70M traces.

### Experiment contract

- **Question:** at toy scale and a raised learning rate (1e-2), do attention logits grow under Muon without QK-norm, does QK-Clip hold them at $\tau$, and how do QK-Clip and QK-norm compare on max logit and held-out loss? Decision informed: whether Recipe-R with Muon needs QK-Clip, given that Baseline-0 has QK-norm; and which statistics the project logs.
- **Hypotheses and status:** H1–H3 above; reported effects at 9B-activated scale (Kimi K2) and in Moonlight's runs; may not appear at this scale.
- **Baseline:** `muon-noqk` (Muon, QK-norm off) for the clip arms; `adamw-noqk` as a reference for "does Muon grow logits more than AdamW". No arm is tuned: all use learning rate 1e-2 (3.3× the toy default, where lesson 03.3 saw AdamW's logits grow) — the same zero tuning budget for every arm.
- **Changed variable:** one per arm (optimizer; QK-Clip $\tau$; QK-norm; MLA). **Controlled:** toy preset, 300 steps of 16 × 128 tokens, cosine schedule with 50 warmup steps, weight decay 0.1, clipping 1.0, seed 0 and data order, the same per-step stability log, 256 fixed held-out windows.
- **Comparison axis:** equal tokens. Parameters: MLA has fewer attention parameters than GQA at this size; say so if you compare MLA's loss with GQA's.
- **Budget:** free CPU, measured below; main path PROJECTED $7 \times 4{,}000 \times 6.55 \times 10^4 \times 3.2 \times 10^8 = 5.9 \times 10^{17}$ FLOPs (`pilot-30m`), about 0.8 H100-hours at 20% MFU plus about 3% for the logit probe.
- **Metrics and decision rule:** max attention logit per step (`stability.jsonl`), its maximum over the run; clipped head-updates; held-out loss with a paired bootstrap against `muon-noqk`. Rules (stated now): H1 observed if `muon-noqk`'s run maximum is at least 3× `muon-qknorm`'s; H2 observed for an arm if the unclipped arm passed its $\tau$, the clipped arm's run maximum stays within 10% of $\tau$, and its held-out loss interval against `muon-noqk` has an upper bound below +0.02; H2 is *not testable* for an arm whose unclipped counterpart never reached $\tau$. The second $\tau$ (15) is fixed in advance as about 2× the max logit of lesson 03.3's QK-norm arm at the raised learning rate (11.2), so that a clip can bind at this scale — an induced test of the mechanism, labelled as such.
- **Correctness checks:** `pytest labs/common/tests/test_optim.py -k "qkclip or rms or monitor"` and `pytest labs/module-07/lesson-02` pass first.
- **Fallback evidence:** the Module 7 pilot's traces at 30M and 70M (Muon at a raised learning rate, max logit logged), and Kimi K2's Figure 2, labelled as analysis of provided or published results.
- **Limits:** one seed, 1.8M parameters, 0.6M tokens; the published effect is at 9B activated parameters and trillions of tokens.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100; PROJECTED about 0.8 GPU-hours. Not run in this build; part of the Module 7 pilot | `python labs/module-07/lesson-02/train_arms.py --variant main`, then `compare_arms.py runs/m07/l72/main --device cuda` |
| Free GPU (Colab/Kaggle T4) | T4, fp32, `pilot-10m`, 2,000 steps | `train_arms.py --variant t4 --max-minutes 80` (rerun after disconnects), then `compare_arms.py runs/m07/l72/t4 --device cuda` |
| Free CPU | laptop; measured below | the steps below as written |

### Steps

1. **Implement** `orthogonal_update_rms`, `rms_matched_scale`, `head_max_logits`, `qk_clip_gamma`, `clip_gqa_weights` and `clip_mla_weights` in `lab.py`; run `pytest labs/module-07/lesson-02`.
2. **Update RMS:** `python labs/module-07/lesson-02/rms_check.py`. Compare each column with the RMS formula and with AdamW's.
3. **Train the arms** (unattended): `python labs/module-07/lesson-02/train_arms.py`. Write down your prediction for H1–H3 while it runs.
4. **Compare:** `python labs/module-07/lesson-02/compare_arms.py`. Apply the rules.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04):

`rms_check.py` (1 minute 46 seconds): median RMS of the update divided by the learning rate, steps 21–40, no weight decay.

| Shape (A, B) | $1/\sqrt{\max(A,B)}$ | AdamW | Muon, `none` | Muon, `original` | Muon, `match_rms` |
|---|---|---|---|---|---|
| (64, 128) | 0.088 | 0.104 | 0.082 | 0.082 | 0.182 |
| (128, 128) | 0.088 | 0.119 | 0.075 | 0.075 | 0.177 |
| (128, 384) | 0.051 | 0.165 | 0.050 | 0.050 | 0.194 |
| (384, 128) | 0.051 | 0.124 | 0.049 | 0.085 | 0.193 |

Unscaled Muon follows $1/\sqrt{\max(A,B)}$ (a little below it for the square attention matrices, whose gradients are ill-conditioned, so five quintic steps leave some singular values below 1 — lesson 07.1); Jordan's scaling lifts only the tall SwiGLU matrix (×1.73); Moonlight's matching gives 0.18–0.19 for every shape. AdamW's update RMS here, 0.10–0.17, is below the 0.2–0.4 Moonlight quotes for long runs; matching to 0.2 makes Muon's steps somewhat larger than this AdamW's at the same learning rate.

`train_arms.py`: 7 runs in 29.7 minutes (3.5–5.4 minutes each); `compare_arms.py` 27 seconds.

| Arm | max logit @50 | @150 | end | run max | clipped head-updates | train loss (last 20) | held-out | vs `muon-noqk` (paired, 256 windows) |
|---|---|---|---|---|---|---|---|---|
| `adamw-noqk` | 35.6 | 203.9 | 186.7 | 261.9 | 0 | 6.759 | 6.709 | +0.875 [+0.853, +0.899] |
| `muon-noqk` | 1.4 | 18.0 | 21.7 | 26.7 | 0 | 5.904 | 5.834 | — |
| `muon-clip` (τ = 100) | 1.4 | 18.0 | 21.7 | 26.7 | 0 | 5.904 | 5.834 | +0.000 [+0.000, +0.000] |
| `muon-clip-lo` (τ = 15) | 1.4 | 14.7 | 14.5 | 17.5 | 201 | 5.902 | 5.832 | −0.002 [−0.002, −0.001] |
| `muon-qknorm` | 5.2 | 10.7 | 13.2 | 13.5 | 0 | 5.934 | 5.862 | +0.029 [+0.021, +0.036] |
| `mla-muon` | 0.8 | 12.7 | 21.9 | 23.9 | 0 | 5.928 | 5.862 | +0.028 [+0.021, +0.036] |
| `mla-clip-lo` (τ = 15) | 0.8 | 12.7 | 14.8 | 17.5 | 60 | 5.928 | 5.861 | +0.028 [+0.021, +0.035] |

Verdicts by the rules. **H1 not observed:** Muon without QK-norm reached a run maximum of 26.7 against 13.5 with QK-norm — 2.0×, below the 3× rule. The striking row is AdamW: at the same learning rate and without QK-norm its logits reached 262, ten times Muon's, and it lost 0.88 nats. At this scale the published ordering ("more frequently with Muon") is reversed; that is a course-scale observation on one seed, not evidence against Kimi K2's report at 9B activated parameters. **H2 with τ = 100 not testable:** the unclipped run never reached 100, the clip never fired, and `muon-clip` is bit-identical to `muon-noqk` (difference exactly 0). **H2 with τ = 15 (induced), not observed by the letter of the rule:** the clip fired on 201 head-updates (60 for MLA), held the end-of-run maximum at 14.5 (MLA 14.8) and cost nothing in loss (−0.002 nats), but the logged *run maximum* is 17.5, 16% above τ. The logged statistic is the maximum of each step's forward pass *before* that step's clip — one optimizer step of growth after the previous clip — so a rule on it needs a tolerance larger than one step's growth; a post-clip measurement is at most τ by construction (the unit test). Write the rule you would use next time. **H3 observed:** with QK-norm the maximum stayed at 13.5, inside the lesson 03.3 bound $\|g_q\|_\infty \|g_k\|_\infty \sqrt d$, which the final checkpoint's gains put at 8.4, 16.1, 19.1 and 14.7 for layers 0–3 (the last step's per-layer maxima: 7.0, 11.9, 13.2, 10.0). MLA and QK-norm both cost about 0.03 nats against the unclipped GQA run here; MLA has fewer attention parameters at this size, so that difference is not a test of either mechanism.

<details>
<summary>Hint for TODO 6</summary>

Walk the heads with two strides: `dq = d_n + d_r` rows per head in the query weight, `dkv = d_n + d_v` rows per head in `kv_b_proj`. Within a head the non-rotary rows come first. Scale in place with `mul_` on row slices, and skip heads with $\gamma_h \ge 1$.

</details>

<details>
<summary>Hint for step 4</summary>

If the `muon-clip` arm (τ = 100) has exactly the same numbers as `muon-noqk`, that is not a bug: the clip never fired because the logits never reached 100. "Not testable at this scale" is the honest verdict for that arm.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-07/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-07/lesson-02`.

</details>

## Common mistakes

- **QK-Clip on a model with QK-norm.** The RMSNorm after `q_proj` and `k_proj` divides the rescale away; the clip silently does nothing. Use QK-norm *or* QK-Clip.
- **Scaling MLA's shared rotary key.** It is shared by every head; scaling it changes all heads' logits, not the one being clipped.
- **Clipping before the optimizer step, or adjusting the gradients.** Kimi K2 rescales the *weights after* the update and leaves the forward and backward pass unchanged.
- **Comparing Muon with Jordan's scaling at AdamW's learning rate.** Without matching, a 768 × 768 matrix's update RMS is 0.036 per unit learning rate instead of 0.2: a 5.5× smaller step.
- **Reading "Muon makes logits grow" as a law.** It is one lab's observation at its scale; check it in your own logs.

## References

- J. Liu et al. (Moonshot AI), *Muon is Scalable for LLM Training*, section 2.2 (Eq. 3, Eq. 4, Lemma 1), section 3.2, appendix D. https://arxiv.org/abs/2502.16982
- Moonshot AI, *Kimi K2*, section 2.1, Algorithm 1, Figure 2. https://arxiv.org/abs/2507.20534
- DeepSeek-AI, *DeepSeek-V4*, sections 2.3.3 and 2.4. https://arxiv.org/abs/2606.19348
- DeepSeek-AI, *DeepSeek-V2*, section 2.1. https://arxiv.org/abs/2405.04434
- Shared code: `labs/common/frontierlab/optim/qkclip.py`, `muon.py`, `stability.py`.

## Next

[07.3 · Hyperparameter transfer](lesson-03.md)
