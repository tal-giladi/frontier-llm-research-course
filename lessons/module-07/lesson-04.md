---
id: "07.4"
module: 7
minutes: 35
practice_minutes: 75
prerequisites: ["07.1", "01.4", "01.5"]
objectives:
  - Write the warmup-stable-decay (WSD) schedule with a decay start, a decay length and a shape, and compute its learning rate at any step by hand.
  - Branch several decays off one stable run from saved full checkpoints, and compute how many training steps that saves compared with one cosine run per length.
  - Compare decay branches with cosine runs of the same length on paired held-out loss, and state which differences are above the noise.
  - Plan continued training from a stable checkpoint versus from a finished (decayed) run, and explain what the re-warm costs.
volatility: concept
sources:
  - title: "Hu et al. — MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (sections 4.1–4.3: Cosine(T), WSD, decay length, data scaling with WSD)"
    url: https://arxiv.org/abs/2404.06395
  - title: "Hägele et al. — Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (section 3.2: cooldown length and 1-sqrt shape)"
    url: https://arxiv.org/abs/2405.18392
  - title: "DeepSeek-AI — DeepSeek-V3 Technical Report (section 4.2: constant learning rate to 10T tokens, then cosine decay over 4.3T)"
    url: https://arxiv.org/abs/2412.19437
  - title: "Moonshot AI — Kimi K2: Open Agentic Intelligence (section 2.5: WSD schedule)"
    url: https://arxiv.org/abs/2507.20534
last_verified: "2026-10-04"
---

# 07.4 · Schedules

A cosine schedule ties the learning rate to the planned length of the run: to know what a model would reach at 600 steps instead of 300, you train a second run from scratch. Warmup-stable-decay (WSD) separates the two: hold the learning rate constant for as long as you like, and decay only when you want a finished model — from any saved checkpoint of the stable run. This lesson writes the schedule precisely, branches several decays off one stable run (the exact-resume machinery of Module 1 is what makes a branch continue the stable run bit for bit), compares the branches with cosine runs of the same lengths on paired held-out loss, and counts the steps saved; it ends with continued training, the use that made WSD popular.

## Why this matters at a frontier lab

Scaling-law fits need models trained to several lengths; data ablations (Module 10) need finished models from the same trunk; and a pretraining run is rarely the last word — mid-training, long-context extension (Module 4) and data refreshes continue it. With cosine, each length is a separate run, and a finished run is at its learning-rate floor, so continuing it means re-warming and partly undoing the decay. DeepSeek-V3 kept its learning rate constant at $2.2 \times 10^{-4}$ until 10T tokens and then decayed over 4.3T (section 4.2); Kimi K2 trained "the first 10T tokens … with a constant learning rate of 2e-4 after a 500-step warm-up, followed by 5.5T tokens with a cosine decay" (section 2.5). Both are WSD-shaped. Knowing what the decay buys, how long it must be and what a branch costs is a planning skill for every run you will design.

## The idea

### Cosine and its length dependence

The course loop's cosine (lesson 01.1, `frontierlab.train.loop.lr_at`), for peak $\eta$, warmup $W$, total $S$ and floor ratio $r = 0.1$:

$$\eta_t = \eta\left(r + (1 - r)\,\tfrac12\big(1 + \cos(\pi\,\tfrac{t - W}{S - W})\big)\right), \qquad W \le t < S.$$

Every $\eta_t$ depends on $S$. MiniCPM (section 4.1) finds that a cosine run reaches its lowest loss at step $S$ when its period equals $S$ ("Cosine(T) where T = S") — a cosine planned for a longer run is worse at $S$, and one planned for a shorter run cannot be extended without a new schedule.

### WSD

$$\eta_t = \begin{cases} \eta\,(t+1)/W & t < W \\ \eta & W \le t < S_0 \\ \eta\big(r + (1 - r)\, f(u)\big), \quad u = \min\!\big(1, \tfrac{t - S_0 + 1}{D}\big) & t \ge S_0 \end{cases}$$

with decay start $S_0$, decay length $D$, floor ratio $r$ and shape $f$: linear $1 - u$; **1-sqrt** $1 - \sqrt u$ (Hägele et al., section 3.2, who find it beats linear); cosine $\tfrac12(1 + \cos \pi u)$. MiniCPM defines WSD with an exponential decay and reports three findings (sections 4.2–4.3): the loss "experiences a significant rapid decline" during the decay; a decay over about 10% of the tokens is enough in their setting; and one can "reuse the model before decay and continue training with the previous high learning rate", decaying from any stable checkpoint. Hägele et al. report that a constant learning rate with a cooldown "scales predictably and reliably similar to cosine", with benefits of longer cooldowns plateauing around 20% in their runs.

In `frontierlab.optim.schedules.lr_at` the stable phase returns exactly `lr` — the same floating-point value as the constant schedule — so a branch started from the stable run's checkpoint at $S_0$ is the stable run up to $S_0$, bit for bit, and diverges only through the decay.

### Branching off one stable run

To get finished models at lengths $S_1 < S_2 < \dots$: train one stable run to the last decay start, keep a full checkpoint (weights, optimizer moments, step, every RNG state — lesson 01.5) at each $S_k - D_k$, and run each decay as a branch from its checkpoint. Cost in steps:

$$\text{WSD} = \max_k (S_k - D_k) + \sum_k D_k, \qquad \text{cosine} = \sum_k S_k .$$

### Continued training

From a stable checkpoint, more training is simply more of the stable run, followed by a new decay. From a finished cosine run at its floor $r\eta$, continuing at a useful learning rate needs a re-warm, which first *raises* the loss (the decay's gain is partly given back) before it improves. Whether the re-warmed run catches up with the WSD continuation at the same total tokens is an empirical question the lab measures.

## Worked example

### One WSD learning rate, by hand

$\eta = 3 \times 10^{-3}$, $W = 30$, $S_0 = 540$, $D = 60$, linear, $r = 0.1$. At $t = 539$: stable, $3 \times 10^{-3}$. At $t = 540$: $u = 1/60$, $\eta_t = 3 \times 10^{-3}(0.1 + 0.9 \cdot 59/60) = 2.955 \times 10^{-3}$. At $t = 569$: $u = 30/60 = 0.5$, $\eta_t = 3 \times 10^{-3}(0.1 + 0.45) = 1.65 \times 10^{-3}$. At $t = 599$: $u = 1$, $\eta_t = 3 \times 10^{-4}$, the floor. With 1-sqrt at $t = 569$: $f = 1 - \sqrt{0.5} = 0.293$, $\eta_t = 3 \times 10^{-3}(0.1 + 0.9 \cdot 0.293) = 1.09 \times 10^{-3}$ — 1-sqrt drops faster at first and lingers near the floor.

### What branching saves

Lengths 300 and 600 with 10% decays: stable to 540, decays of 30 and 60 steps: $540 + 30 + 60 = 630$ steps, against $300 + 600 = 900$ for two cosine runs — 30% fewer. Lengths 1,000, 2,000 and 4,000: $3{,}600 + 100 + 200 + 400 = 4{,}300$ against $7{,}000$ — 39% fewer. The saving grows with the number of lengths, which is why scaling-law fits use it (Hägele et al.).

### Decay length in production

DeepSeek-V3: 4.3T decay tokens of a 14.8T run, 29%; Kimi K2: 5.5T of 15.5T, 35%. Both decayed far longer than MiniCPM's 10% — a reminder that "10% is enough" was measured on MiniCPM's models and data, and that labs with a single expensive run may choose a long, conservative decay.

## Shapes and cost

A schedule changes no tensor. It changes **when you pay** and **what you must store**: every branch point is a full checkpoint (for Baseline-0 in fp32: 122M weights × 4 bytes = 0.49 GB, plus AdamW's two moments, 0.98 GB, about 1.5 GB per kept checkpoint; with Muon one moment for the hidden matrices, about 1.1 GB). The compute arithmetic is in the worked example. On the main path (lengths 3,000 and 6,000 steps of 64 × 1,024 tokens of `pilot-30m`, 10% decays, plus the continuation to 9,000) the lab is 23,700 steps (stable 8,100, decays 3,600, cosine runs 9,000, re-warmed continuation 3,000) of $6.55 \times 10^4$ tokens; at $3.2 \times 10^8$ training FLOPs per token (`accounting.flops_per_token`, `pilot-30m`, $T = 1{,}024$) that is $5.0 \times 10^{17}$ FLOPs, about 0.7 H100-hours at an assumed 20% MFU — PROJECTED, pending the Module 7 pilot.

## Build it

`labs/common/frontierlab/optim/schedules.py` has `lr_at(step, steps, lr, warmup, schedule, decay_start=, decay_steps=, shape=, min_ratio=)` (its `"cosine"` is the loop's cosine to the last bit, tested), `branch_plan` and `branch_cost`. The Module 7 wrapper installs it for one `loop.main()` call and adds `--branch-from CKPT`, which copies a full checkpoint into the new run's folder before the loop starts, so the loop "resumes" there:

```bash
# stable run to 540 (constant), keep its checkpoint, branch a 60-step decay off it
python -m frontierlab.optim.train --run runs/stable --steps 810 --schedule constant --stop-after 540 ...
cp runs/stable/checkpoint.pt runs/stable/step540.pt        # keep a copy next to the run
python -m frontierlab.optim.train --run runs/wsd-T600 --steps 600 --schedule wsd --decay-start 540 \
    --decay-steps 60 --min-lr-ratio 0.1 --branch-from runs/stable/step540.pt ...
```

`--init-from CKPT` is the other kind of continuation: weights only, a fresh optimizer and step 0, so the schedule re-warms. Correctness checks (`tests/test_optim.py`): WSD equals the constant schedule before $S_0$ for every step; the decay is monotone and ends at the floor; the branch's first logged step is $S_0 + 1$ and its run card names the stable run as `parent_run`.

## What the evidence says

- **WSD and decay branches — ESTABLISHED.** PUBLICLY DOCUMENTED by MiniCPM (sections 4.2–4.3) and Hägele et al. (sections 3–5, which also report that cooldowns give scaling-law fits at reduced cost); production runs with a long constant phase followed by a decay are documented for DeepSeek-V3 (section 4.2) and Kimi K2 (section 2.5).
- **Decay length and shape — PROMISING / setting-dependent.** "10% is enough" (MiniCPM) and "plateau around 20%", 1-sqrt better than linear (Hägele et al.) are measured on those papers' models; production decays of 29–35% show labs do not treat 10% as a rule.
- **Course-scale hypotheses** (lab): (H1) every decay branch ends within the noise of, or below, the cosine run of the same length; (H2) the training loss falls sharply during the decay; (H3) continuing from the stable checkpoint and decaying at 900 beats re-warming the finished cosine-600 run for 300 steps. H2 is robust; H1 and H3 may be within evaluation noise at 0.6–1.8M tokens.

## Lab

**Folder:** [`labs/module-07/lesson-04/`](../../labs/module-07/) · **Time:** about 75 minutes (about 40 of them unattended) · **Pass check:** `pytest labs/module-07/lesson-04` passes; `compare_schedules.py` runs on your runs; your notes state H1–H3 as observed / not observed / within noise, with the paired intervals, and the step counts of both ways.

### Experiment contract

- **Question:** at toy scale, do WSD decay branches off one stable run match cosine runs of the same length, at fewer total steps, and is continuing from a stable checkpoint better than re-warming a finished run? Decision informed: whether Module 11's scaling-law ladder and Module 10's data ablations should branch decays off stable runs.
- **Hypotheses and status:** H1–H3 above; reported effects (MiniCPM, Hägele et al.); may be within noise at this scale.
- **Baseline:** cosine runs of the same total length (`cos-T300`, `cos-T600`), and the cosine-way continuation `cos-T600+300`. No re-tuning of either schedule: both use the peak learning rate 3e-3 from lesson 01.1's toy runs (equal tuning budget: zero for both).
- **Changed variable:** the schedule. **Controlled:** toy preset, AdamW (the loop's own), peak 3e-3, warmup 30, floor 0.1 × peak for every schedule (so the final learning rate is not a second variable), 16 × 128 tokens per step, seed 0 and data order (a branch continues the stable run's data stream), 256 fixed held-out windows.
- **Comparison axis:** equal tokens at each length; the step counts of the two ways of producing all lengths are the cost comparison.
- **Budget:** free CPU, measured below; main path PROJECTED 0.7 H100-hours (formula above).
- **Metrics and decision rule:** held-out loss with a paired bootstrap over windows (lesson 01.4) of each WSD arm against the cosine arm of the same length. Rule: H1 observed if every WSD arm's interval has its upper bound at or below +0.01 nats; H3 observed if the interval of `wsd-T900-d10` minus `cos-T600+300` lies below 0. One seed, so the intervals cover evaluation noise only; compare effects with the Module 1 seed noise floor before calling them real.
- **Correctness checks:** `pytest labs/common/tests/test_optim.py -k "wsd or branch"` passes; each branch's first logged step is its $S_0 + 1$; `python -m frontierlab.record runs/m07/l74/cpu/cos-T600 runs/m07/l74/cpu/wsd-T600-d10 --changed args.schedule optim.decay_start optim.decay_steps optim.min_lr_ratio optim.branch_from` prints only CHANGED lines and COMPARABLE (measured in this build).
- **Fallback evidence:** MiniCPM's Figures for WSD vs cosine and Hägele et al.'s cooldown sweeps, labelled as published results.
- **Limits:** one seed, tiny model, 0.6–1.8M tokens (far from the regimes the papers study), one peak learning rate.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100; PROJECTED 0.7 GPU-hours. Not run in this build; part of the Module 7 pilot | `python labs/module-07/lesson-04/run_schedules.py --variant main`, then `compare_schedules.py runs/m07/l74/main --device cuda` |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant cpu` with `--device cuda` added in `VARIANTS` |
| Free CPU | laptop; measured below | the steps below as written |

### Steps

1. **Implement** `wsd_lr`, `branch_plan`, `branch_cost` and `decay_drop` in `lab.py`; run `pytest labs/module-07/lesson-04`.
2. **Train** (unattended): `python labs/module-07/lesson-04/run_schedules.py`. Run `--print` first and check, from the printed arguments alone, that every branch's `--decay-start` equals the step of the checkpoint it starts from.
3. **Compare:** `python labs/module-07/lesson-04/compare_schedules.py`. Apply the rules to H1–H3.
4. **Plan:** you need finished models at 1,000, 2,000, 4,000 and 8,000 steps with 10% decays. Compute both step counts with your `branch_cost`, and the storage for the kept checkpoints if the model were Baseline-0.
5. **Extension (optional, about 20 minutes):** repeat with `--lr 1e-2` for every arm (edit `VARIANTS`, and use a new `--out`), and compare the two peak learning rates for each schedule. Does WSD still win by 0.12 nats when each schedule gets its better peak?

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, other jobs running, 2026-10-04):

`run_schedules.py`: all 10 runs in 13 minutes with per-step logging (19 minutes in a first, busier session; the decays took 15–53 s each); `compare_schedules.py` 24 seconds. The training-loss drop is the mean of the 10 logged steps before the decay start minus the mean of the run's last 10 steps. Held-out loss, 256 windows of 128 tokens:

| Model | Steps trained | Held-out loss |
|---|---|---|
| stable run at 270 / 480 / 540 / 810 (no decay) | — | 6.324 / 6.002 / 5.924 / 5.719 |
| `cos-T300` | 300 | 6.358 |
| `cos-T600` | 600 | 5.958 |
| `cos-T600+300` (re-warmed 300 more) | 900 | 5.761 |

| WSD arm | Held-out | vs cosine of the same length (paired) | Training-loss drop during the decay |
|---|---|---|---|
| `wsd-T300-d10` | 6.234 | −0.123 [−0.132, −0.115] vs `cos-T300` | +0.052 |
| `wsd-T600-d10` | 5.823 | −0.135 [−0.143, −0.126] vs `cos-T600` | +0.069 |
| `wsd-T600-d20` | 5.815 | −0.144 [−0.151, −0.136] vs `cos-T600` | +0.127 |
| `wsd-T600-d10-sqrt` | 5.821 | −0.138 [−0.146, −0.129] vs `cos-T600` | +0.076 |
| `wsd-T900-d10` | 5.585 | −0.176 [−0.188, −0.164] vs `cos-T600+300` | +0.163 |

Steps for lengths 300 and 600: 630 with branches, 900 with separate cosine runs (30% fewer).

Verdicts by the rules. **H1 observed — by a margin that should make you suspicious.** Every branch beats the cosine run of its length by 0.12–0.14 nats, far more than MiniCPM or Hägele et al. report (they find WSD about equal to a well-tuned cosine), and *even the undecayed stable run at 540 steps (5.924) beats the finished 600-step cosine run (5.958)*. The cause is in the contract: the peak learning rate, 3e-3, was not tuned for either schedule, and lesson 07.3's sweep put this model's optimum at or above it. A schedule that spends more steps near the peak wins when the peak is too low — the comparison measures the untuned baseline, not WSD. With equal tuning budgets (a peak-learning-rate sweep for each schedule), the gap would shrink; that is the extension below. **H2 observed** in the held-out numbers: decaying from 540 for 60 steps gained 0.10 nats (5.924 → 5.823), while the stable run gained 0.078 over the previous 60 steps (6.002 → 5.924), with the training-loss drop in the last column (+0.07 for the 60-step decays, +0.13 for the 120-step one). **H3 observed:** continuing the stable trunk to 810 and decaying beat re-warming the finished cosine run by 0.18 nats, with the same 900 steps of data in each (inherits the same untuned-peak caveat). The shapes and lengths (linear vs 1-sqrt, 10% vs 20%) are within 0.01 of each other: not resolved by one seed (toy seed std 0.029, lesson 01.4).

<details>
<summary>Hint for TODO 1</summary>

Write the three phases in order and return early. The decay variable is $u = (t - S_0 + 1)/D$, clipped to 1, so that the last decay step already sits at the floor; when $u = 1$ return `lr * min_ratio` without evaluating the shape.

</details>

<details>
<summary>Hint for step 3</summary>

The stable run's kept checkpoints are evaluated too. The gap between "stable at 540" and the decayed branch at 600 is what the decay bought; the gap between the stable checkpoint and the cosine run at a similar step shows what a constant learning rate costs *before* the decay.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-07/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-07/lesson-04`.

</details>

## Common mistakes

- **Branching from a weights-only checkpoint.** Without the optimizer moments, step and data RNG, the branch is a new run with a fresh optimizer, not a decay of the stable one. `--branch-from` copies the full checkpoint; `--init-from` deliberately does not.
- **Comparing schedules that end at different learning rates.** A WSD branch decayed to 0 against a cosine with a 0.1 floor changes two things.
- **Judging a WSD run before its decay.** The stable run's loss is higher than a cosine run's at the same step; that is the point, not a defect. Compare finished models.
- **Keeping too few checkpoints.** You can only branch where a full checkpoint exists. Decide the lengths before the stable run, or keep checkpoints at a regular cadence.
- **Taking "10% decay" as a constant.** It is one paper's measurement; production runs decayed for about a third of their tokens.

## References

- S. Hu et al., *MiniCPM*, sections 4.1–4.3 (Cosine(T), WSD, decay length, reuse of stable checkpoints). https://arxiv.org/abs/2404.06395
- A. Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, section 3.2 (1-sqrt cooldown, cooldown length). https://arxiv.org/abs/2405.18392
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, section 4.2. https://arxiv.org/abs/2412.19437
- Moonshot AI, *Kimi K2*, section 2.5. https://arxiv.org/abs/2507.20534
- Shared code: `labs/common/frontierlab/optim/schedules.py`, `train.py`.

## Next

[07.5 · Stability forensics](lesson-05.md)
