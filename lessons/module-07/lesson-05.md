---
id: "07.5"
module: 7
minutes: 40
practice_minutes: 90
prerequisites: ["07.2", "03.3", "01.5"]
objectives:
  - Choose the per-step statistics a training run should log to catch instability early, and say what each one costs to compute.
  - Detect loss spikes with a rolling median and median absolute deviation, and find which statistic crossed its threshold first.
  - Tell three failure kinds apart from their logs — attention-logit growth, an optimizer or learning-rate fault, and a bad data batch — and match each to the fixes that address it.
  - State what QK-norm, QK-Clip, attention soft-capping, QKV clamping and z-loss each bound, and measure their effect on an induced logit-growth failure.
volatility: concept
sources:
  - title: "Chowdhery et al. — PaLM: Scaling Language Modeling with Pathways (section 5: z-loss; section 5.1: training instability, restart and skip)"
    url: https://arxiv.org/abs/2204.02311
  - title: "Team OLMo — 2 OLMo 2 Furious (sections 2.1, 3: QK-norm, z-loss, repeated n-gram filtering, gradient-norm spikes)"
    url: https://arxiv.org/abs/2501.00656
  - title: "Gemma Team — Gemma 2: Improving Open Language Models at a Practical Size (section 2: logit soft-capping, 50 attention / 30 final)"
    url: https://arxiv.org/abs/2408.00118
  - title: "Moonshot AI — Kimi K2: Open Agentic Intelligence (section 2.1: why soft-capping and QK-norm were not enough; QK-Clip)"
    url: https://arxiv.org/abs/2507.20534
  - title: "Allen AI — OLMo-1.7-7B config.json (clip_qkv: 8.0)"
    url: https://huggingface.co/allenai/OLMo-1.7-7B-hf/blob/main/config.json
  - title: "Dehghani et al. — Scaling Vision Transformers to 22 Billion Parameters (attention-logit growth and QK-norm)"
    url: https://arxiv.org/abs/2302.05442
last_verified: "2026-10-04"
---

# 07.5 · Stability forensics

When a large run spikes, the question is not only "how do we get past this?" but "what caused it, and would we have seen it coming?". The answer is in the logs, if the run logged the right statistics. This lesson picks those statistics — loss, gradient norm, the update-to-weight ratio of every matrix, the largest attention logit per layer, the output softmax normaliser — builds a spike detector, and then makes three different things go wrong on purpose: attention logits that grow until the softmax saturates, a restart with the wrong learning rate, and a few batches of degenerate data. You diagnose each one from its logs alone, then test the published fixes on the first and see which ones bound what.

## Why this matters at a frontier lab

PaLM 540B "observed spikes in the loss roughly 20 times during training, despite the fact that gradient clipping was enabled", at irregular intervals, and handled them by restarting "from a checkpoint roughly 100 steps before the spike started" and skipping "roughly 200–500 data batches" (section 5.1). OLMo 2 traced its spikes to two sources it could fix — architecture and initialisation choices that let norms and logits grow, and "long, repeated n-gram sequences" in the data — and changed both (sections 3.1–3.3). Kimi K2 trained 15.5T tokens "with zero loss spikes" after adding QK-Clip (abstract). Every one of those results started with someone reading logs. A research engineer is expected to look at a spike and say, with evidence, which kind it is — because the fixes are different, and a wrong fix (skipping data when the logits are growing) only postpones the next spike.

## The idea

### What to log, per step

| Statistic | What it watches | Cost |
|---|---|---|
| loss | the symptom | free |
| gradient norm (before clipping) | size of the raw gradient | one reduction (the loop already computes it to clip) |
| update / weight RMS, per matrix (median and max) | how far one step moves each weight relative to its size; jumps when the learning rate or the optimizer state is wrong | one subtraction and two norms per matrix; the course computes it only on logged steps |
| max attention logit per layer (and head) | growth of $q \cdot k/\sqrt d$ toward softmax saturation | recomputing $q k^\top$ per layer: $O(T^2)$ — production systems take it from the attention kernel (Kimi K2 computes $S_{\max}$ "during forward") |
| output $\log Z$ (mean, max) and max $|\text{logit}|$ | drift of the output softmax normaliser (what z-loss controls) | one logsumexp over (B, T, V) |
| learning rate | schedule and restart bugs | free |

`frontierlab.optim.stability.StabilityLogger` writes all of them to `<run>/stability.jsonl`; the loop's `metrics.jsonl` has loss, gradient norm and learning rate (use `--log-every 1` when you want per-step losses).

### Detecting a spike

A loss spike is a step whose loss is far above the recent level, where "far" must adapt to the run's own noise. With the previous $w$ losses, median $\tilde\ell$ and median absolute deviation $\mathrm{MAD} = \operatorname{median}|\ell - \tilde\ell|$:

$$\text{spike at } t \iff \ell_t - \tilde\ell > \max\big(k \cdot 1.4826 \cdot \mathrm{MAD},\; \rho\, \tilde\ell\big),$$

with $1.4826 \cdot \mathrm{MAD}$ an estimate of the standard deviation that ignores earlier spikes, $k = 6$ and a relative floor $\rho = 0.05$ so that a near-constant loss does not flag tiny wiggles. Consecutive flagged steps are one spike. Single-batch losses are noisy (±0.15 nats per step on the toy model), so the course runs the detector on a centred running median of 5 steps: it keeps a spike of three or more steps and a sustained jump, and removes one-step noise. A one-step spike is then invisible — look for it in the gradient norm, which is logged per step too.

### Three failure signatures

1. **Attention-logit growth.** Nothing in the loss stops $\|W_Q\|$ and $\|W_K\|$ from growing; the logits grow with their product, the softmax approaches one-hot, its gradient $p_i(\delta_{ij} - p_j)$ vanishes for all but one entry, and small weight changes swing which entry wins (lesson 03.3). *Signature:* the max logit climbs over tens to hundreds of steps **before** the spike; the update ratio does not jump; the loss may spike and then plateau above where a healthy run would be.
2. **Optimizer or learning-rate fault** (a restart with the wrong learning rate, a missing warmup, a corrupted optimizer state). *Signature:* the update-to-weight ratio jumps **at the same step** the fault begins — by about the factor of the learning-rate error — and the loss rises right after; logits need not move first.
3. **Bad data.** A few batches of degenerate text (long repeated n-grams, as OLMo 2 found). *Signature:* nothing moves before the spike; the loss and the raw gradient norm jump **on the bad steps only**; gradient clipping keeps the update ratio near normal; the loss comes back within tens of steps. PaLM's caveat matters here: it did "not believe that the spikes were caused by 'bad data' per se", because the same batches did not cause a spike from another checkpoint — the data and the model state interact.

### The fixes, and what each one bounds

- **QK-norm** (RMSNorm on each head's $q$ and $k$, Baseline-0's default): $|\ell| \le \|g_q\|_\infty \|g_k\|_\infty \sqrt d$ (lesson 03.3). Bounds logits by the norm gains.
- **QK-Clip** (Kimi K2): rescales $W_q$, $W_k$ per head after each step so the batch's max logit is at most $\tau$ (lesson 07.2). Bounds logits *at the weights*.
- **Attention soft-capping** (Gemma 2: $\ell \leftarrow c \tanh(\ell / c)$, $c = 50$): bounds the logit *values* fed to the softmax, but not $q \cdot k$ itself; Kimi K2's objection is that "the dot products between queries and keys can still grow excessively before capping is applied". Its gradient $\mathrm{sech}^2(\ell/c)$ also vanishes for capped logits. Gemma 3 replaced it with QK-norm.
- **QKV clamping** (OLMo-1.7-7B's `"clip_qkv": 8.0`, applied to the q, k and v projections in Hugging Face's `OlmoAttention`): entries at most $c$, so $|\ell| \le c^2 \sqrt d$ — loose unless $c$ is small. OLMo 2 replaced it with QK-norm (Table 1).
- **z-loss** (PaLM: $10^{-4} \log^2 Z$ "to encourage the softmax normalizer log(Z) to be close to 0"; OLMo 2 uses $10^{-5}$): acts on the **output** softmax, not on attention. It is a control for the attention-logit failure, not a fix for it.
- **Gradient clipping** bounds the gradient norm. For SGD that bounds the step; for Adam it mostly does not — Adam divides by $\sqrt{\hat v}$, so scaling every gradient by a constant leaves the update unchanged once the moments have adapted — and Muon normalises its update by construction. Clipping changes Adam's and Muon's steps mainly by changing how much a large gradient counts in the moving averages.
- **Restart and skip** (PaLM): operational; it addresses the trigger, not the cause, unless the cause is the data.

## Worked example

### A spike threshold, by hand

Previous 8 losses: 6.10, 6.08, 6.12, 6.05, 6.09, 6.07, 6.11, 6.06. Median 6.085; deviations 0.015, 0.005, 0.035, 0.035, 0.005, 0.015, 0.025, 0.025, MAD 0.02. $6 \cdot 1.4826 \cdot 0.02 = 0.178$ and $0.05 \cdot 6.085 = 0.304$, so the threshold is 0.304: a loss of 6.50 is a spike (excess 0.415), 6.30 is not.

### Soft-capping at $c = 50$

$\ell = 30 \to 50 \tanh(0.6) = 26.85$ (gradient factor $\mathrm{sech}^2(0.6) = 0.71$). $\ell = 200 \to 49.97$ (gradient factor $\mathrm{sech}^2(4) = 0.0013$): the head's preference is frozen at the cap and almost no gradient flows back to reduce it.

### The clamping bound

$c = 8$, head dimension $d = 32$: $|q \cdot k| \le 32 \cdot 64 = 2{,}048$, so $|\ell| = |q \cdot k|/\sqrt{32} \le 362$ — above the logits the lab's failure reaches, so `clip_qkv 8` may not bind at all on the toy model.

### z-loss on the toy model

Uniform logits over 8,192 tokens give $\log Z = \ln 8192 = 9.01$ (with logits near 0). With $\log Z \approx 8.3$, the term is $10^{-4} \cdot 8.3^2 = 0.0069$ nats against a loss near 6 — negligible, but its gradient $2 \cdot 10^{-4} \log Z$ on the normaliser pushes steadily toward $\log Z = 0$.

### Why clipping does not shrink an Adam step

One weight, gradients $g = 0.1$ every step: $\hat m = 0.1$, $\hat v = 0.01$, update $\eta \cdot 0.1 / 0.1 = \eta$. Gradients $g = 10$ (100×): $\hat m = 10$, $\hat v = 100$, update $\eta$. Clip the 100× gradients to 0.1 and the update is still $\eta$.

## Shapes and cost

| Statistic | Tensors touched | Extra compute per logged step |
|---|---|---|
| max logit per layer | q (B, H, T, d), k (B, KV, T, d) recomputed from the layer input; logits (1, H, T, T) one sequence at a time | $2 B H T^2 d$ per layer plus the q/k projections; at toy size (B = 16, H = 4, T = 128, d = 32) 67M FLOPs per layer, small next to the 23 GFLOP step |
| update / weight RMS | one (A, B) difference per matrix | $O(N)$ |
| log Z | logits (B, T, V) fp32 | one logsumexp, $O(BTV)$ |

The probes run inside forward pre-hooks, under the step's autocast, without gradients; they change no training number (straight and stop-and-resume runs with the logger on are bit-identical, `tests/test_optim.py`). On the main path, recomputing the logits at $T = 1{,}024$ costs about $2 \cdot 32 \cdot 8 \cdot 1024^2 \cdot 64 = 3.4 \times 10^{10}$ FLOPs per layer and batch of 32 — log every 10–50 steps, or read the max from a fused attention kernel.

## Build it

- `frontierlab/optim/stability.py`: `StabilityLogger(model, opt, path, every, qkclip=)`; `detect_spikes(steps, losses, window=20, k=6, min_rel=0.05)`; `precursor(...)`; `smooth(values, 5)` (a centred running median: single-batch losses are noisy); `diagnose(metrics_rows, stability_rows)`, which detects spikes on the smoothed losses, applies the decision rules in `RULES` (including a no-spike branch for logit growth that stalls the loss) and returns the evidence with a verdict.
- `frontierlab/optim/stabilizers.py`: `OptLM` (z-loss, final-logit soft-cap, log Z tracking) and the attention kind `"gqa-softcap"` (`extra["attn_softcap"]`, `extra["clip_qkv"]`), which passes the causal and cached-decode checks and reduces to Baseline-0's GQA when the cap is huge (tested to $10^{-5}$).
- `frontierlab/optim/train.py` flags: `--stability-log`, `--qk-norm off`, `--attn-softcap`, `--clip-qkv`, `--z-loss`, `--qk-clip`, `--inject-bad-steps`, `--branch-from`.

## What the evidence says

- **QK-norm against logit growth — ESTABLISHED** (ViT-22B, OLMo 2, Gemma 3, Qwen3; DeepSeek-V4 normalises queries and KV entries for the same reason, section 2.4).
- **z-loss — ESTABLISHED** for the output softmax (PaLM section 5; OLMo 2 Table 1).
- **Soft-capping — MODEL-SPECIFIC and receding** (Gemma 2; replaced by QK-norm in Gemma 3). **QKV clamping — MODEL-SPECIFIC** (OLMo-1.7; replaced by QK-norm in OLMo 2). **QK-Clip — PROMISING** (Kimi K2; one lab's evidence at scale).
- **Data-triggered spikes — PUBLICLY DOCUMENTED** (OLMo 2 section 3.1; PaLM section 5.1, with its caveat that data and model state interact).
- **Which statistic predicts trouble.** PUBLICLY DOCUMENTED cases tie spikes to growing logits (Kimi K2 Figure 2; ViT-22B) and to slowly growing gradient norms (OLMo 2 section 3). That a rising max logit *precedes* a spike by a usable margin is REASONABLE INDUSTRY PRACTICE; we found no published study measuring lead times systematically.
- **Course-scale hypotheses** (lab): (H1) each induced failure has the signature above and `diagnose` labels it correctly; (H2) QK-norm, QK-Clip and soft-capping each keep the f1 run's max logit at least 4× lower than without them, and z-loss does not; (H3) clamping at 8 does not bind on this model. They are about these toy runs; the published failures happened at billions of parameters.

## Lab

**Folder:** [`labs/module-07/lesson-05/`](../../labs/module-07/) · **Time:** about 90 minutes (about 50 of them unattended) · **Pass check:** `pytest labs/module-07/lesson-05` passes; `diagnose.py` runs on your runs and on the three provided traces; your notes give, for each failure, the evidence that identifies it and the verdict, and H2–H3 with numbers.

### Experiment contract

- **Question:** can three induced failure kinds be told apart from the per-step logs alone, and which published fixes bound the attention-logit failure at toy scale? Decision informed: which statistics the module project and the Recipe-R run (Module 11) log, at what cadence, and which fix Recipe-R uses if logits grow.
- **Hypotheses and status:** H1–H3 above; the mechanisms are established (lesson 03.3, PaLM, OLMo 2, Kimi K2); the specific signatures at this scale may differ.
- **Baseline:** `healthy` (QK-norm on, learning rate 3e-3), and for the fix arms `f1-logits` (QK-norm off, learning rate 1e-2).
- **Changed variable:** one per arm (the failure injected, or one fix on top of `f1-logits`). **Controlled:** toy preset, `MuonAdamW` in its AdamW configuration, 300 steps of 16 × 128 tokens, cosine with 30 warmup steps, clipping at 1.0, seed 0 and data order, the same per-step logging.
- **Comparison axis:** equal tokens.
- **Budget:** free CPU, measured below; main path (`pilot-30m`, 4,000 steps of 64 × 1,024): PROJECTED $9 \times 4{,}000 \times 6.55 \times 10^4 \times 3.2 \times 10^8 = 7.5 \times 10^{17}$ FLOPs, about 1.1 H100-hours at 20% MFU, plus the logit probes.
- **Metrics and decision rule:** spikes by `detect_spikes`; the verdict of `diagnose` against the known cause of each failure (H1 observed if all three are labelled correctly); max logit over the run per fix arm against `f1-logits` (H2 observed for a fix if it is at most a quarter); for H3, whether `f1+clipqkv`'s logits differ from `f1-logits`'s at all. Held-out loss with a paired bootstrap against `f1-logits` for the fix arms, as a secondary.
- **Correctness checks:** `pytest labs/common/tests/test_optim.py -k "softcap or zloss or spikes or resume"` passes; the `gqa-softcap` kind passes the causal and cached-decode checks; the `healthy` run has no detected spike.
- **Fallback evidence:** the three provided traces in `labs/module-07/lesson-05/traces/` (CPU runs from this build) and the published spike descriptions (PaLM, OLMo 2, Kimi K2 Figure 2), labelled as analysis.
- **Limits:** one seed, a 1.8M-parameter model, 300 steps; failures were induced, not found; at scale, spikes have more causes (hardware faults, precision, MoE routing) than the three studied here.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100; PROJECTED about 1.1 GPU-hours. Not run in this build; part of the Module 7 pilot | `python labs/module-07/lesson-05/induce.py --variant main`, then `diagnose.py runs/m07/l75/main` |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `--variant cpu` with `--device cuda` added in `VARIANTS` |
| Free CPU | laptop; measured below | the steps below as written |

### Steps

1. **Implement** `z_loss`, `softcap`, `find_spikes`, `first_crossing` and `classify` in `lab.py`; run `pytest labs/module-07/lesson-05`.
2. **Diagnose blind first:** `python labs/module-07/lesson-05/diagnose.py labs/module-07/lesson-05/traces`. The three traces A, B and C are one failure kind each, unlabelled. Write your verdict and the evidence for each *before* step 3.
3. **Induce** (unattended): `python labs/module-07/lesson-05/induce.py`.
4. **Diagnose your runs:** `python labs/module-07/lesson-05/diagnose.py runs/m07/l75/cpu`. Compare with what you know caused each, and with your blind verdicts.
5. **Fixes:** read the fix table `diagnose.py` prints for the `f1+...` arms and apply the H2–H3 rules.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04):

`induce.py`: 9 runs in 27.9 minutes (about 2.5–3.5 minutes each); `diagnose.py` 32 seconds.

| Run | Detected spike (running median) | Evidence before / at the spike | Verdict by the rules | Known cause |
|---|---|---|---|---|
| `healthy` | none | max logit 6.2 over the run; held-out 6.358 | none | — |
| `f1-logits` | none | max logit 0.4 → 188 (by step 24 it passed 20); loss stalled: held-out 6.597 | logit growth (no spike) | QK-norm off, lr 1e-2 |
| `f2-lr-bug` | steps 153–155, peak 7.23 vs median 6.78 | logits 5.8 (flat); update/weight ratio ×11.4 at the spike | optimizer | resumed at step 150 with lr 3e-2 |
| `f3-data` | steps 150–152, peak 8.71 vs median 6.79 | logits 5.8 (flat); ratio ×2.2; back within 20 steps | data | 3 batches of a repeated 4-token pattern |

| Fix on top of `f1-logits` | max logit over the run | first step ≥ 20 | end | held-out | vs `f1-logits` (paired) |
|---|---|---|---|---|---|
| none (`f1-logits`) | 188.4 | 24 | 160.3 | 6.597 | — |
| `f1+qknorm` | 11.7 | never | 11.6 | 6.522 | −0.076 [−0.084, −0.067] |
| `f1+qkclip` (τ = 15) | 21.5 | 43 | 15.2 | 6.428 | −0.169 [−0.180, −0.158] |
| `f1+softcap` (50) | 259.7 | 24 | 254.3 | 6.565 | −0.032 [−0.041, −0.023] |
| `f1+clipqkv` (8) | 162.4 | 24 | 147.3 | 6.611 | +0.014 [+0.005, +0.023] |
| `f1+zloss` (1e-4) | 134.8 | 25 | 107.4 | 6.645 | +0.048 [+0.038, +0.057] |

Verdicts. **H1 observed, with one correction to the lesson's picture:** all three failures were labelled correctly, but at this scale attention-logit growth did not produce a loss *spike* — the logits passed 20 within 24 steps and reached 188, and the loss simply stalled (held-out 6.597 against `healthy`'s 6.358: +0.239 [+0.228, +0.251]). The other two failures left little lasting damage: `f2-lr-bug` ended +0.010 [+0.001, +0.018] and `f3-data` +0.019 [+0.017, +0.021] above `healthy`. The rules therefore need the no-spike branch ("logit growth (no spike)"), and a real monitoring system needs a logit alarm, not only a spike detector. The learning-rate fault was visible on the first resumed step (ratio ×11); the data fault had no precursor and recovered. Per-step single-batch losses on this model vary by ±0.15 nats, so the detector runs on a 5-step running median; on raw losses the learning-rate fault (+0.4–0.6 nats for 5 steps) is not flagged. **H2:** QK-norm (16× lower) and QK-Clip (8.7× lower; its logged pre-clip maximum overshoots τ = 15 by one step's growth, as in lesson 07.2) are observed; soft-capping is *not* — the raw $q \cdot k/\sqrt d$ grew even larger (260) behind the cap, exactly Kimi K2's objection, though the capped run's loss was 0.03 better; z-loss is not, as expected for a control that acts on the output softmax (and it cost 0.05 nats here). **H3 not observed:** the clamp at 8 did bind — at the end of training the pre-clamp q and k projections reached 10–25 — yet the logits still reached 162, because the bound $c^2\sqrt d = 362$ is far above where trouble starts. Loss differences between fix arms are one seed each (toy seed std 0.029): QK-Clip's −0.17 is well outside it, the soft-cap and clamp rows are not clearly.

<details>
<summary>Hint for step 2</summary>

Look at three things around the first spike: what the max logit did over the previous 50–100 steps, what the update/weight ratio did *at* the spike step compared with the 20 steps before, and whether the loss came back down within 20 steps.

</details>

<details>
<summary>Reference diagnosis of the three traces (open after step 2)</summary>

- **A — bad data.** Spike at steps 150–152 (peak 8.7 against a median of 6.8) with nothing moving before it: max logit 5.8 and flat, update/weight ratio only ×2.2 at the spike (gradient clipping held it), loss back within 20 steps. It is `f3-data`: three batches of one repeated 4-token pattern.
- **B — attention-logit growth without a spike.** No spike is detected, but the max logit climbs from 0.4 to 188 (past 20 by step 24) while the loss stalls about 0.24 nats above a healthy run. It is `f1-logits`: QK-norm off at learning rate 1e-2.
- **C — optimizer / learning-rate fault.** Spike at steps 153–155 right after a restart at 150; the update/weight ratio jumps ×11 on the first resumed step and the `lr` column jumps from 0.0019 to 0.0188. It is `f2-lr-bug`: resumed with `--lr 3e-2`. Reading the learning-rate column is the fastest check of all.

</details>

<details>
<summary>Hint for TODO 3</summary>

Use `statistics.median` for both the median and the MAD; compare against the window *before* the step (not including it), and start at index `window`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-07/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-07/lesson-05`.

</details>

## Common mistakes

- **Logging only the loss.** By the time the loss spikes, the cause has usually been visible elsewhere for a while (logits) or is visible only on the same step (update ratio). Without those columns, every spike looks like "bad data".
- **Logging statistics every 100 steps.** A three-step data spike or a one-step learning-rate jump falls between the samples. Log the cheap statistics every step and the expensive ones (logits) often enough to see a trend.
- **Treating z-loss as a fix for attention.** It regularises the output softmax normaliser; attention logits need QK-norm, QK-Clip or a cap.
- **Trusting gradient clipping to bound an Adam or Muon step.** Both normalise the update; clipping changes the mix of gradients in the moments, not the step length.
- **Skipping data for a logit-growth spike.** The run will spike again later; fix the cause.
- **Comparing logit statistics across a soft-capped and an uncapped run without saying which logit.** The logger records the raw $q \cdot k/\sqrt d$ before any cap, which is what grows.

## References

- A. Chowdhery et al., *PaLM*, section 5 (z-loss) and section 5.1 (spikes, restart and skip). https://arxiv.org/abs/2204.02311
- Team OLMo, *2 OLMo 2 Furious*, sections 2.1, 3, 3.1 and Table 1. https://arxiv.org/abs/2501.00656
- Gemma Team, *Gemma 2*, section 2 (soft-capping 50 / 30). https://arxiv.org/abs/2408.00118
- Moonshot AI, *Kimi K2*, section 2.1 and Figure 2. https://arxiv.org/abs/2507.20534
- Allen AI, OLMo-1.7-7B `config.json` (`clip_qkv`). https://huggingface.co/allenai/OLMo-1.7-7B-hf/blob/main/config.json
- M. Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*. https://arxiv.org/abs/2302.05442
- Shared code: `labs/common/frontierlab/optim/stability.py`, `stabilizers.py`, `qkclip.py`.

## Next

The module project: [Optimizer decision report](../../projects/module-07-optimizer-decision.md).
