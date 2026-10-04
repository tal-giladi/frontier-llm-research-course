---
id: "07.1"
module: 7
minutes: 40
practice_minutes: 75
prerequisites: ["01.5", "02.4"]
objectives:
  - Derive what one Newton–Schulz step does to each singular value, and explain why Jordan's quintic coefficients leave singular values in a band around 1 instead of at 1.
  - Implement Muon (momentum, Nesterov, Newton–Schulz orthogonalisation, decoupled weight decay, shape scaling) and test it against an exact SVD, a converged cubic iteration and torch.optim.Muon.
  - Decide which parameters of a transformer go to Muon and which stay on AdamW, and say why.
  - Compute Muon's optimizer FLOPs per step, compare them with the training FLOPs of the step, and measure the optimizer's wall-clock share with the Module 2 benchmark harness.
  - Train with Muon through the course loop and show that stop-and-resume is still bit-identical.
volatility: concept
sources:
  - title: "K. Jordan — Muon: An optimizer for hidden layers in neural networks (blog post, 2024-12-08)"
    url: https://kellerjordan.github.io/posts/muon/
  - title: "K. Jordan — Muon reference implementation (muon.py)"
    url: https://github.com/KellerJordan/Muon
  - title: "Moonshot AI — Muon is Scalable for LLM Training (Moonlight), sections 2.1–2.2"
    url: https://arxiv.org/abs/2502.16982
  - title: "PyTorch 2.14 — torch.optim.Muon"
    url: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
  - title: "DeepSeek-AI — DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, section 2.4"
    url: https://arxiv.org/abs/2606.19348
last_verified: "2026-10-04"
---

# 07.1 · Muon from scratch

AdamW updates every weight entry on its own, scaled by a running estimate of that entry's gradient size. Muon treats a weight matrix as a matrix: it takes the momentum of the gradient and replaces it with the nearest matrix whose singular values are all about 1, so that every direction the gradient points in moves by a similar amount. This lesson builds Muon from its parts — momentum, a Newton–Schulz iteration that orthogonalises without an SVD, the shape scaling, and the split of parameters between Muon and AdamW — tests each part against an exact reference, counts what it costs, and trains Baseline-0's toy model with it through the course loop with exact resume intact.

## Why this matters at a frontier lab

Kimi K2 (1T parameters, 15.5T tokens) and Moonlight were trained with Muon, and DeepSeek-V4 uses it "for the majority of modules" (V4 section 2.4). Moonlight reports that Muon needs about 52% of AdamW's training FLOPs to reach the same loss in its compute-optimal scaling-law runs (section 3.2). If that holds for your model, it is worth more than most architecture changes; if it does not, switching optimizers costs a re-tuning campaign, new failure modes (lesson 07.2) and an optimizer step that is harder to shard. The question for this module is the one a lab actually faces: *should our next run use Muon, and with which settings?* Answering it starts with an implementation you trust bit for bit and a cost model you can defend in a design review.

## The idea

### From a gradient to an orthogonal update

Write a weight matrix's gradient (or its momentum) as $G = U S V^\top$ (singular value decomposition), with $U \in \mathbb{R}^{A \times r}$, $V \in \mathbb{R}^{B \times r}$ orthonormal columns, $S = \operatorname{diag}(s_1, \dots, s_r)$, $r = \min(A, B)$. Gradients of transformer weights are typically dominated by a few large singular values: a step along $G$ moves the weight a lot in a few directions and barely at all in the rest. Muon replaces $G$ by

$$O = U V^\top,$$

the orthogonal matrix closest to $G$ (in Frobenius norm): same singular *directions*, every singular *value* set to 1. Jordan's post motivates it as amplifying "rare directions" that have small magnitude in the update but matter for learning; Bernstein and Newhouse (arXiv 2409.20325) show that such methods, with the moving averages switched off, are steepest descent under a particular norm — for this update, the spectral norm. Both views agree on the operation.

### Newton–Schulz instead of an SVD

An SVD per matrix per step is too slow on a GPU. The Newton–Schulz iteration needs only matrix products. Let $X_0 = G / \|G\|_F$; since $\|G\|_F \ge \|G\|_2$, every singular value of $X_0$ is in $(0, 1]$. One step is

$$A = X X^\top, \qquad B = b A + c A^2, \qquad X \leftarrow a X + B X .$$

Because $X X^\top = U S^2 U^\top$, the step keeps $U$ and $V$ and acts on each singular value separately:

$$s \;\mapsto\; \varphi(s) = a s + b s^3 + c s^5 .$$

Two choices of $(a, b, c)$:

- **Classic cubic** $(1.5, -0.5, 0)$: $\varphi(1) = 1$ and $\varphi'(1) = 0$, so 1 is a stable fixed point and every singular value in $(0, 1]$ converges to exactly 1 — slowly for small $s$, because $\varphi'(0) = 1.5$.
- **Jordan's quintic** $(3.4445, -4.7750, 2.0315)$, 5 steps, in bfloat16. The slope at zero is $a = 3.4445$, so small singular values grow fast. The price: $\varphi(1) = 3.4445 - 4.7750 + 2.0315 = 0.701$, so 1 is *not* a fixed point; singular values end up oscillating in a band around 1 (Jordan's code comment: roughly $U S' V^\top$ with $S'_{ii}$ spread between about 0.5 and 1.5), which "turns out not to hurt model performance". The iteration is not converging to $UV^\top$; it is a fast, approximate whitening.
- **DeepSeek-V4's hybrid**: 8 quintic steps "to drive rapid convergence", then 2 steps with $(2, -1.5, 0.5)$ "to stabilize the singular values precisely at 1" (V4 section 2.4). For that polynomial $\varphi(1) = 1$ and $\varphi'(1) = 2 - 4.5 + 2.5 = 0$: a stable fixed point with quadratic convergence.

Working on the wide orientation (transpose if $A > B$) makes $X X^\top$ the small $r \times r$ Gram matrix.

### The whole update

For a weight $W$ of shape $(A, B) = (\text{fan\_out}, \text{fan\_in})$, gradient $G_t$, momentum $\mu = 0.95$, learning rate $\eta$, weight decay $\lambda$:

$$M_t = \mu M_{t-1} + G_t, \qquad U_t = G_t + \mu M_t \;\;(\text{Nesterov}), \qquad O_t = \mathrm{NS}(U_t),$$

$$W_t = W_{t-1} - \eta\big(s(A,B)\, O_t + \lambda W_{t-1}\big).$$

Jordan's code uses an exponential moving average ($M \leftarrow \mathrm{lerp}(M, G, 1 - \mu)$); that buffer is $(1 - \mu)$ times the sum form above, and the Newton–Schulz normalisation divides the scale out, so both give the same $O_t$. The shape factor $s(A, B)$ comes in two published forms: Jordan's $\sqrt{\max(1, A/B)}$, and Moonlight's $0.2\sqrt{\max(A, B)}$ (Eq. 4), which makes the update's RMS about 0.2 — the range Moonlight observed for AdamW — so that AdamW's learning rate and weight decay can be reused. Lesson 07.2 derives that factor.

### Which parameters stay on AdamW

Muon is defined for 2-D matrices that map one hidden representation to another. The token embedding is a lookup table (each step updates only the rows of tokens that appeared, so "the nearest orthogonal matrix" of its gradient is not a meaningful step), the output head is the same matrix in Baseline-0 (tied), and norm gains, biases and learned sinks are vectors. All of them stay on AdamW: "Scalar and vector parameters of the network, as well as the input and output layers, should be optimized by a standard method such as AdamW" (Jordan); Moonlight uses AdamW for "RMSNorm, LM head, and embedding parameters"; DeepSeek-V4 keeps AdamW for "the embedding module, the prediction head module, the static biases and gating factors of mHC modules, and the weights of all RMSNorm module[s]". In Baseline-0 the q, k, v and o projections and the three SwiGLU matrices of every layer go to Muon — each as its own matrix. (Implementations differ on whether to orthogonalise q, k and v separately or as one fused matrix, and per head or not; the course keeps the separate `nn.Linear` matrices of the model.)

## Worked example

### Newton–Schulz on a 2 × 2 matrix, by hand

$G = \operatorname{diag}(3, 1)$, so $U = V = I$ and $UV^\top = I$. $\|G\|_F = \sqrt{10} = 3.162$, so $X_0$ has singular values $(0.9487, 0.3162)$.

Quintic step 1 on $s = 0.3162$: $3.4445 \cdot 0.3162 - 4.7750 \cdot 0.0316 + 2.0315 \cdot 0.0032 = 1.0893 - 0.1510 + 0.0064 = 0.9447$. On $s = 0.9487$: $3.2678 - 4.0773 + 1.5613 = 0.7518$. Five steps give

| step | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| quintic, larger $s$ | 0.7518 | 1.0484 | 0.6819 | 1.1343 | 0.7530 |
| quintic, smaller $s$ | 0.9447 | 0.7568 | 1.0414 | 0.6825 | 1.1337 |
| cubic, smaller $s$ | 0.4585 | 0.6396 | 0.8286 | 0.9584 | 0.9974 |

The quintic lifts the small singular value from 0.32 to 0.94 in one step (the cubic needs four), then both values bounce inside $[0.68, 1.13]$ — the band. Two V4 finishing steps from 0.68 give $2(0.68) - 1.5(0.68)^3 + 0.5(0.68)^5 = 0.8281$, then 1.0006; from 1.13: 1.0169, then 1.0002.

### Momentum and the scale that disappears

Step 1 with $M_0 = 0$: $M_1 = G_1$, $U_1 = G_1 + 0.95 G_1 = 1.95\,G_1$. After the Frobenius normalisation $U_1$ and $G_1$ give the same $X_0$: Muon's step size is set by $\eta$ and $s(A, B)$, never by the gradient's magnitude. This is why gradient clipping changes Muon's *direction* only through the momentum mix, not its step length.

### Cost of one Muon step for Baseline-0

Per Newton–Schulz step on a wide $r \times n$ matrix: $2r^2n$ for $XX^\top$, $2r^3$ for $A^2$, $2r^2n$ for $BX$, so $4r^2n + 2r^3$. A $768 \times 768$ projection: 5 steps $\times\, 6 \cdot 768^3 = 1.36 \times 10^{10}$ FLOPs. Over Baseline-0's 84 Muon matrices (12 layers × q, k, v, o, gate, up, down), `frontierlab.optim.cost` gives $1.71 \times 10^{12}$ FLOPs per optimizer step. The training step at the main-path batch (16 × 32,768 = 524,288 tokens, $7.881 \times 10^8$ FLOPs per token) costs $4.13 \times 10^{14}$: Newton–Schulz is **0.41%**. Jordan's bound $T m / B = 5 \cdot 768 / 524{,}288 = 0.73\%$ is the square-matrix version of the same ratio ($k \cdot 6m^3$ against $6m^2 B$).

At a CPU batch of 2,048 tokens the same model's Newton–Schulz work is 134% of the training FLOPs: the overhead is per *step*, so it is amortised only by large batches.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| weight $W$, gradient $G$, momentum $M$ | (A, B), e.g. (768, 768), (2816, 768), (768, 2816) | fp32 master weights | GPU (main path) / CPU |
| Newton–Schulz working copy $X$ | (r, n), r = min(A, B) | bf16 (Jordan, torch.optim.Muon); fp64 in tests | same |
| Gram matrices $A$, $B$ | (r, r) | bf16 | same |
| update $O$ | (A, B) | cast back to fp32 | same |

Optimizer **memory**: Muon keeps one buffer per matrix (momentum), AdamW two (first and second moments). For Baseline-0's 84 Muon matrices (96.7M of its 121.9M parameters) that saves 96.7M × 4 bytes ≈ 0.39 GB of fp32 state; the embedding stays on AdamW with both moments.

Optimizer **time** is not the FLOP share. Newton–Schulz is 15 small dependent matmuls per matrix, run one matrix after another; in a sharded run (FSDP, Module 9) each matrix must first be gathered whole on one device, because orthogonalisation needs the entire matrix. That gather is the main engineering cost Moonlight's distributed implementation addresses (section 2.3). On one device, `cost_table.py` measures it.

## Build it

`labs/common/frontierlab/optim/muon.py` has the pieces, each small enough to read:

- `newton_schulz(G, schedule="quintic5", dtype=torch.bfloat16)` — the iteration above; `NS_SCHEDULES` holds `quintic5`, `v4-hybrid`, `cubic5`, `cubic20`.
- `split_params(model)` — Muon matrices vs AdamW parameters (embedding/tied head, norms and vectors).
- `MuonAdamW` — *one* `torch.optim.Optimizer` whose parameter groups are `kind="muon"` or `kind="adamw"`. The AdamW groups are written out with the same arithmetic as `torch.optim.AdamW(foreach=False)`, so both arms of a Muon-vs-AdamW comparison run the same code. Being one optimizer, it drops into the course loop unchanged: the loop sets `group["lr"]` for every group, and `state_dict()` carries momentum buffers, Adam moments and the step counter, so the checkpoint resumes exactly.
- `frontierlab/optim/train.py` — `python -m frontierlab.optim.train --optimizer muon <loop args>` swaps the loop's `make_optimizer` for one `loop.main()` call (the loop file is not edited; the native `--optimizer` flag is proposed in the Module 7 inbox).

Correctness checks in `labs/common/tests/test_optim.py`, all of which pass:

```python
G = torch.randn(32, 32, dtype=torch.float64)
O = newton_schulz(G, [CUBIC] * 40, dtype=torch.float64)
(O - orthogonal_polar(G)).abs().max()          # < 1e-10: the cubic converges to U V^T (exact SVD)
torch.linalg.svdvals(newton_schulz(G, "quintic5", dtype=torch.float64))   # all in (0.6, 1.25): the band
```

Further tests: the singular vectors are unchanged (U^T O V is diagonal to $10^{-9}$); one step equals the polynomial on singular values; the AdamW groups equal `torch.optim.AdamW` to $10^{-12}$ in float64; Muon groups equal `torch.optim.Muon` (PyTorch 2.14, both `adjust_lr_fn` options) within bf16 rounding; and stop-and-resume through the wrapper is bit-identical with Muon on.

## What the evidence says

- **Muon for hidden matrices — ESTABLISHED in open labs, recent.** PUBLICLY DOCUMENTED in production by Moonshot (Moonlight; Kimi K2, section 2.1) and DeepSeek (V4, section 2.4), plus Essential AI's study (arXiv 2505.02222, up to 4B parameters). Jordan's own results are speed records on small models (CIFAR-10, NanoGPT: a 1.35× training-speed improvement in his post) — evidence of sample efficiency, not of frontier-scale behaviour.
- **Efficiency claims (company claim).** Moonlight: ~2× compute efficiency, "only requires about 52% training FLOPs to match the performance of AdamW under compute-optimal setting" (section 3.2). Essential AI: 10–15% fewer tokens than AdamW to reach the same loss, with the advantage persisting at large batch sizes (sections 2.4, 5). The two numbers measure different things (FLOPs at compute-optimal size vs tokens at fixed size); neither is a guarantee for your model. The module project tests it at course scale, as a hypothesis.
- **Exactness of the orthogonalisation.** Jordan states the quintic's inexact singular values do not hurt; DeepSeek-V4 chose to make them exact with two extra steps. No published ablation that we found isolates the effect of that choice — treat it as MODEL-SPECIFIC.
- **Cost.** The FLOP share is small at real batch sizes (PUBLICLY DOCUMENTED bound, verified by our arithmetic); the wall-clock share depends on implementation and sharding, and must be measured (lesson 02.4).

## Lab

**Folder:** [`labs/module-07/lesson-01/`](../../labs/module-07/) · **Time:** about 75 minutes (15 of them unattended) · **Pass check:** `pytest labs/module-07/lesson-01` passes; `ns_explore.py` shows your `newton_schulz` matching the reference; the straight and resumed Muon runs are IDENTICAL; your notes give the measured Muon/AdamW step-time ratio with its interval and the projected FLOP share for Baseline-0.

### Experiment contract

This lab's one comparison is the step-time cost of Muon against AdamW (step 3).

- **Question:** how much longer is a training step with Muon than with AdamW for the toy and pilot-10m models on this machine, and how does that compare with the FLOP share predicted by the formula? Decision informed: the step-time ratio the module project uses to set its equal-wall-clock budget.
- **Hypothesis and status:** the measured ratio exceeds 1 + (FLOP share), because Newton–Schulz runs as many small dependent matmuls. Established effect (Jordan; Moonlight section 2.3 on distributed cost), measured here on one device.
- **Baseline:** `MuonAdamW` with every parameter on AdamW (same code path, same model, same batch).
- **Changed variable:** the optimizer of the hidden matrices (AdamW → Muon, `match_rms`, quintic × 5, bf16). **Controlled:** model weights at the start, batch (16 × 128 random tokens, fixed seed), threads, process, warm-up 3 calls.
- **Comparison axis:** equal work per step (one forward, backward and optimizer step on the same batch). It does not say which optimizer reaches a given loss sooner — that needs the project's equal-wall-clock training runs.
- **Budget:** free CPU, about 1 minute measured; main path on the GPU, a few minutes (not run in this build).
- **Metrics and decision rule:** median step time of each, the paired per-round ratio with a 95% bootstrap interval (`frontierlab.perf.speedup`), and the optimizer step alone. Rule: use the ratio's point estimate for the project's equal-wall-clock step count, and report its interval; if the interval spans more than ±20%, rerun with more rounds on a quieter machine.
- **Correctness checks:** `pytest labs/module-07/lesson-01` and `pytest labs/common/tests/test_optim.py` pass first.
- **Fallback evidence:** none needed; this is a measurement, not an effect that may fail to appear.
- **Limits:** one machine, one batch size; CPU bf16 matmuls are slow and not representative of a GPU; the share falls as the batch grows (see the table).

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100, under 15 GPU-minutes. Not run in this build; part of the Module 7 pilot | steps 1–2 as written; step 3 with `python labs/module-07/lesson-01/cost_table.py --device cuda --presets pilot-30m baseline0 --batch 32 --seq 1024 --vocab 32768`; step 4 with `--preset pilot-10m --steps 400 --batch 32 --seq 512 --dtype bf16` |
| Free GPU (Colab/Kaggle T4) | T4 | as the main path with `--dtype fp32 --presets pilot-10m pilot-30m --seq 512` |
| Free CPU | laptop; measured below | steps 1–4 exactly as written |

### Steps

1. **Implement** `ns_step`, `newton_schulz`, `muon_direction`, `muon_apply`, `muon_param_names` and `ns_flops` in `lab.py`; run `pytest labs/module-07/lesson-01`.
2. **Look at real gradients:** `python labs/module-07/lesson-01/ns_explore.py`. For the two gradients it prints, write down the condition number, how many quintic steps the smallest singular value needs to enter the band, and the final relative distance to $UV^\top$ for each schedule.
3. **Cost:** `python labs/module-07/lesson-01/cost_table.py`. Record the projected Newton–Schulz share at your CPU batch and at the main-path batch, and the measured step-time ratio with its interval.
4. **Train and resume with Muon:**

   ```bash
   python -m frontierlab.optim.train --optimizer muon --run runs/m07/l71/muon-straight --preset toy --steps 300 --batch 16 --seq 128 --lr 3e-3
   python -m frontierlab.optim.train --optimizer muon --run runs/m07/l71/muon-resumed  --preset toy --steps 300 --batch 16 --seq 128 --lr 3e-3 --stop-after 150
   python -m frontierlab.optim.train --optimizer muon --run runs/m07/l71/muon-resumed  --preset toy --steps 300 --batch 16 --seq 128 --lr 3e-3
   python labs/module-01/lesson-01/compare_logs.py runs/m07/l71/muon-straight runs/m07/l71/muon-resumed
   ```

   It must print IDENTICAL. Open `runs/m07/l71/muon-straight/run_card.yaml` and find the `optim` block and `budget.optimizer_flops`.

Measured in this build (free CPU: Windows 11, 16-thread Intel laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04):

| Measurement | Result |
|---|---|
| `ns_explore.py`, attention `q_proj` gradient (128 × 128) | condition number 14,600; after 5 quintic steps the smallest singular value is only 0.02 (largest 1.20), relative distance to $UV^\top$ 0.22; V4 hybrid reaches [1.00, 1.00] in float64 and [0.99, 1.01] in bf16; cubic × 5 is still at 0.00 |
| `ns_explore.py`, SwiGLU `up_proj` gradient (384 × 128) | condition number 45.6; quintic × 5 band [0.68, 1.13], distance 0.18; V4 hybrid [1.00, 1.00] (float64) |
| Newton–Schulz share of training FLOPs per step | toy at 2,048 tokens: 10%; Baseline-0 at 2,048 tokens: 134%; Baseline-0 at 524,288 tokens: 0.41% (PROJECTED arithmetic) |
| step time, toy, 16 × 128 | AdamW 290 ms, Muon 333 ms; ratio 1.32 [1.00, 1.55]; optimizer alone 8 ms vs 79 ms |
| step time, pilot-10m, 16 × 128 | AdamW 1,084 ms, Muon 1,798 ms; ratio 1.62 [1.44, 1.71]; optimizer alone 37 ms vs 726 ms |
| `ns_explore.py` / `cost_table.py` runtime | 5 s / 68 s |
| Muon toy run, 300 steps (step 4) | 5.2 minutes straight (about 2,000 tokens/s), validation loss 5.979 at step 300 (64 windows); stopped at 150 and resumed: 4.5 minutes in all; `compare_logs.py`: IDENTICAL, largest loss difference 0.0 over 15 logged steps |

The ill-conditioned attention gradient is the interesting row: five quintic steps do not lift its smallest directions into the band, so "Muon orthogonalises the update" is only approximately true for exactly the matrices whose small directions Muon is meant to amplify. Whether that matters for training is an empirical question nobody has settled publicly; the V4 hybrid is one answer, at twice the Newton–Schulz cost.

<details>
<summary>Hint for TODO 2</summary>

Keep track of whether you transposed: `tall = G.shape[0] > G.shape[1]`. Normalise *after* transposing (the Frobenius norm does not care) and transpose back at the end. Do not add `eps` inside the square root; Jordan's code divides by `X.norm() + eps`.

</details>

<details>
<summary>Hint for TODO 5</summary>

Use the parameter names of `frontierlab.model.LM`: matrices are `ndim == 2`; the embedding is `model.embed_tokens.weight` (the tied head does not appear separately in `named_parameters()`); norm gains have `norm` in their name.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-07/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-07/lesson-01`.

</details>

## Common mistakes

- **Putting the embedding (and a tied head) on Muon.** An embedding gradient has nonzero rows only for tokens in the batch; orthogonalising it spreads the update over every row. Keep it on AdamW, as every published recipe does.
- **Orthogonalising the tall orientation.** Correct either way, but $X X^\top$ is then the large Gram matrix: a 2816 × 768 SwiGLU matrix costs 9.1× more per step than its transpose ($4r^2n + 2r^3$ with $r = 2816$ instead of 768).
- **Expecting exact orthogonality from the quintic.** Singular values stay in a band (about 0.68–1.13 for well-conditioned inputs, and lower for ill-conditioned ones). A test that asserts $O^\top O = I$ for five quintic steps is wrong; test the band, and test convergence with the cubic.
- **Calling the FLOP share the wall-clock share.** At small batches Newton–Schulz can cost more than the forward and backward pass; on a sharded cluster it needs whole matrices. Measure it.
- **Changing the optimizer and the code path at once.** Compare Muon with AdamW inside the same optimizer class (here `MuonAdamW`), or check that the two implementations agree, before you attribute a difference to the algorithm.

## References

- K. Jordan, *Muon: An optimizer for hidden layers in neural networks* (coefficients, 5 steps, bf16, which parameters, overhead bound $Tm/B$). https://kellerjordan.github.io/posts/muon/
- K. Jordan, Muon reference implementation, `muon.py` (EMA momentum, $\sqrt{\max(1, A/B)}$ scaling). https://github.com/KellerJordan/Muon
- J. Liu et al. (Moonshot AI), *Muon is Scalable for LLM Training*, sections 2.1–2.3 and 3.2. https://arxiv.org/abs/2502.16982
- PyTorch, `torch.optim.Muon` (PyTorch 2.14). https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- DeepSeek-AI, *DeepSeek-V4*, section 2.4 (Muon for most modules, AdamW list, hybrid Newton–Schulz). https://arxiv.org/abs/2606.19348
- Essential AI, *Practical Efficiency of Muon for Pretraining*, sections 2.3–2.4. https://arxiv.org/abs/2505.02222
- J. Bernstein, L. Newhouse, *Old Optimizer, New Norm: An Anthology*. https://arxiv.org/abs/2409.20325
- Shared code: `labs/common/frontierlab/optim/muon.py`, `cost.py`, `train.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[07.2 · Muon at scale](lesson-02.md)
