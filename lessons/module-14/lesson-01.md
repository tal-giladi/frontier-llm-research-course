---
id: "14.1"
module: 14
minutes: 45
practice_minutes: 110
prerequisites: ["12.2", "12.3"]
objectives:
  - Derive the per-token gradient weight of GRPO, DAPO, Dr. GRPO, GSPO and CISPO from their objectives and compute it by hand for a small batch.
  - Explain which samples each objective silences once the policy has moved (token clipping, sequence clipping, capped weights) and predict its clip fraction.
  - Show that GRPO and GSPO, and DAPO and CISPO, give identical gradients on an on-policy minibatch, and say what that implies for experiments with one update per batch.
  - Implement the five losses behind one interface, pass float64 gradient tests against the derivations, and run each inside the Module 12 loop.
  - Map each objective onto its verl v0.9.1 configuration without relying on the framework for its definition.
volatility: concept
sources:
  - title: "Shao et al., DeepSeekMath (section 4.1.1, Eq. 3: GRPO; Eq. 4: KL estimator; section 4.2 hyperparameters)"
    url: https://arxiv.org/abs/2402.03300
  - title: "Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Eq. 8; section 4.1)"
    url: https://arxiv.org/abs/2503.14476
  - title: "Liu et al., Understanding R1-Zero-Like Training: A Critical Perspective (sections 3.1-3.2: Dr. GRPO)"
    url: https://arxiv.org/abs/2503.20783
  - title: "Zheng et al., Group Sequence Policy Optimization (sections 4.1-4.3, Eqs. 5, 7, 10, 12-14; sections 5.1-5.3)"
    url: https://arxiv.org/abs/2507.18071
  - title: "MiniMax, MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (section 3.1: CISPO)"
    url: https://arxiv.org/abs/2506.13585
  - title: "Schulman et al., Proximal Policy Optimization Algorithms (section 3, Eq. 7)"
    url: https://arxiv.org/abs/1707.06347
  - title: "Khatri et al., The Art of Scaling Reinforcement Learning Compute for LLMs (section 4: the ScaleRL recipe uses CISPO)"
    url: https://arxiv.org/abs/2510.13786
  - title: "verl v0.9.1 (verl/trainer/ppo/core_algos.py, policy loss registry)"
    url: https://github.com/volcengine/verl/tree/v0.9.1
last_verified: "2026-10-07"
---

# 14.1 · Objectives derived, not switched

GRPO, DAPO, Dr. GRPO, GSPO and CISPO are usually compared as names in a config file. This lesson treats each one as a formula for how much each sampled token's log-probability gradient counts, derives that weight from the paper's objective, and shows that the five differ in only three places: how tokens are weighted across responses of different lengths, what happens to a sample once the policy has moved away from the one that generated it, and whether the importance ratio is taken per token or per sequence. You then implement all five behind one interface, check them against the derivations in float64, and run each in the Module 12 loop.

## Why this matters at a frontier lab

Each of these objectives comes from a lab that used it for a released model or a large published run: GRPO for DeepSeekMath and, per the Qwen3 report (section 4.2), Qwen3's reasoning RL; DAPO for ByteDance Seed's open 32B run; GSPO, which the Qwen team credits for improvements in its latest Qwen3 models; CISPO for MiniMax-M1 and as the loss of Meta's ScaleRL recipe. A team that switches objectives by flipping `loss_mode` in a framework inherits that framework's reading of the paper, its aggregation default and its clip defaults, and usually changes three things at once without knowing it. Knowing the gradient each objective produces lets you say what a switch changes, design the comparison of lesson 14.2 so that only that changes, and spot an implementation that does not match its paper. Several do not: lesson 12.3 showed that TRL and verl disagree on default aggregation.

## The idea

### One gradient, five weightings

Every objective here is built from the same pieces as lesson 12.2: for a prompt $x$, $G$ responses $y_1, \dots, y_G$ are sampled from the behaviour policy $\pi_{\text{old}}$, scored, and given advantages $A_i$ (the group-normalised reward, one per response). The token importance ratio is

$$r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})},$$

with $\pi_\theta$ the policy being trained, $y_{i,t}$ the $t$-th token of response $i$ and $|y_i|$ its length. Since $\nabla_\theta r_{i,t} = r_{i,t}\,\nabla_\theta \log \pi_\theta(y_{i,t})$, the gradient of every objective below has the form

$$\nabla_\theta J = \sum_i \sum_t w_{i,t}\, \nabla_\theta \log \pi_\theta(y_{i,t} \mid x, y_{i,<t}),$$

and the objectives differ only in the **weight** $w_{i,t}$. That weight is what `frontierlab.rlscale.objectives.gradient_weights` computes by autograd, and what you derive by hand below.

### GRPO

GRPO (DeepSeekMath section 4.1.1, Eq. 3) is PPO's clipped surrogate averaged per response, then over the group, with the KL to a reference model inside the sum:

$$J_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \Big\{ \min\big(r_{i,t} A_i,\ \mathrm{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\, A_i\big) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \Vert \pi_{\text{ref}}] \Big\}.$$

The KL term uses the $k_3$ estimator (Eq. 4; lesson 12.2) and is a separate design choice; this module holds it fixed across objectives (off, $\beta = 0$, as DAPO and many later recipes do) so that the comparison is about the policy term. DeepSeekMath's own run used $\beta = 0.04$, $G = 64$, learning rate $10^{-6}$ (section 4.2); it does not state $\epsilon$.

Differentiate the min. Where the unclipped branch is the smaller one, $\nabla \min(\cdot) = A_i \nabla r_{i,t} = A_i r_{i,t} \nabla\log\pi_\theta$. Where the clipped branch is smaller, it is a constant and the gradient is 0. That happens when $A_i > 0$ and $r_{i,t} > 1+\epsilon$ (the token's probability already rose enough) or $A_i < 0$ and $r_{i,t} < 1-\epsilon$. So

$$w^{\text{GRPO}}_{i,t} = \frac{1}{G}\,\frac{1}{|y_i|}\; r_{i,t}\, A_i \;\cdot\; \mathbb{1}[\text{not clipped}].$$

### DAPO and Dr. GRPO: the same surrogate, other normalisers

DAPO (Eq. 8) drops the KL term, raises the upper clip bound (clip-higher, $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$, section 4.1) and replaces the per-response mean by one mean over every token of the group:

$$J_{\text{DAPO}} = \frac{1}{\sum_{i}|y_i|}\sum_{i}\sum_{t} \min\big(r_{i,t} A_i,\ \mathrm{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\, A_i\big), \qquad w^{\text{DAPO}}_{i,t} = \frac{r_{i,t} A_i}{\sum_j |y_j|}\,\mathbb{1}[\text{not clipped}].$$

It is subject to the dynamic-sampling constraint $0 < |\{i : y_i \text{ correct}\}| < G$: groups that are all right or all wrong are dropped and more prompts are sampled to fill the batch (section 3.2). That is a batch rule, not part of the loss. In the course loop it is `zero_var="filter"`, which drops those groups without refilling.

Dr. GRPO (section 3.2) removes the two terms its section 3.1 calls biased: the $1/|y_i|$ (response-level length bias) and the division by the group's reward standard deviation (question-level difficulty bias). Its code divides by a constant `MAX_TOKENS`, so with $L$ the generation budget:

$$w^{\text{Dr.GRPO}}_{i,t} = \frac{1}{G\,L}\; r_{i,t}\, A_i\,\mathbb{1}[\text{not clipped}], \qquad A_i = R_i - \mathrm{mean}(R).$$

Lesson 12.3 already measured the consequence of each normaliser: who gets more weight, short or long responses. Here the point is that DAPO and Dr. GRPO keep GRPO's **per-token clipped ratio**. All three silence a token as soon as its own ratio leaves the trust region in the advantage's direction.

### GSPO: one ratio per sequence

GSPO (section 4.1) argues that a per-token ratio is the wrong unit, because the reward belongs to the whole response. A token-level weight $r_{i,t}$ comes from a single sample of that token's distribution, so it is noisy, and the noise grows with response length. GSPO uses the **length-normalised sequence ratio** (Eq. 7),

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}\right)^{1/|y_i|} = \exp\Big(\frac{1}{|y_i|}\sum_{t} \log r_{i,t}\Big),$$

the geometric mean of the token ratios, and clips it per sequence (Eq. 5):

$$J_{\text{GSPO}} = \frac{1}{G}\sum_i \min\big(s_i A_i,\ \mathrm{clip}(s_i, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\, A_i\big).$$

Because $\nabla s_i = s_i \cdot \frac{1}{|y_i|}\sum_t \nabla\log\pi_\theta(y_{i,t})$ (the paper's Eq. 10),

$$w^{\text{GSPO}}_{i,t} = \frac{1}{G}\,\frac{1}{|y_i|}\; s_i\, A_i \;\cdot\; \mathbb{1}[\text{sequence } i \text{ not clipped}].$$

Every token of a response gets the same weight, and a clipped response contributes nothing. Without the length normalisation, one ratio over a 10,000-token response would be a product of 10,000 noisy factors, and no clip range would make sense. With it, $s_i$ stays near 1, which is why GSPO's ranges are tiny: $3 \times 10^{-4}$ below and $4 \times 10^{-4}$ above, against 0.2 and 0.27 for its GRPO baseline (section 5.1). GSPO reports clipping about two orders of magnitude more tokens than GRPO while training more efficiently (section 5.2). That is a reminder that "clip fraction" means different things under different objectives. GSPO-token (section 4.3, Eq. 14) defines $s_{i,t} = \mathrm{sg}[s_i]\,\pi_\theta(y_{i,t})/\mathrm{sg}[\pi_\theta(y_{i,t})]$, where sg is stop-gradient. Its value is $s_i$ and its gradient is $s_i\nabla\log\pi_\theta(y_{i,t})$, so with one advantage per response it equals GSPO exactly. It exists to allow per-token advantages.

GSPO's other argument is about mixture-of-experts models (section 5.3). After an update, the same token can be routed to different experts, so per-token ratios jump. Qwen's GRPO runs on MoE needed "Routing Replay" (caching the old policy's expert choices) to converge. The sequence likelihood moves much less, and GSPO trains without it. The paper also reports that the sequence ratio tolerates the training/inference precision gap better (section 5.4; lesson 14.3).

### CISPO: clip the weight, keep the token

CISPO (MiniMax-M1 section 3.1) starts from a different complaint. In MiniMax's runs the tokens that PPO-style clipping silenced included rare, high-ratio "fork" tokens ("However", "Recheck", "Wait", "Aha") that start a reflective step, and they are exactly the tokens a reasoning policy needs to learn. So CISPO keeps REINFORCE's gradient for every token and clips only the importance weight, under a stop-gradient:

$$J_{\text{CISPO}} = \frac{1}{\sum_i |y_i|}\sum_i\sum_t \mathrm{sg}\big(\hat r_{i,t}\big)\, A_i \log\pi_\theta(y_{i,t}), \qquad \hat r_{i,t} = \mathrm{clip}\big(r_{i,t}, 1-\epsilon^{\text{IS}}_{\text{low}}, 1+\epsilon^{\text{IS}}_{\text{high}}\big),$$

$$w^{\text{CISPO}}_{i,t} = \frac{\hat r_{i,t}\, A_i}{\sum_j |y_j|} \quad\text{for every token.}$$

A token whose ratio passed the upper bound keeps its gradient at the capped weight $1+\epsilon_{\text{high}}$, instead of losing it. MiniMax "did not impose a lower bound": it set $\epsilon^{\text{IS}}_{\text{low}}$ large and tuned only $\epsilon^{\text{IS}}_{\text{high}}$. The paper's text does not give its value. The course default of 0.28 is DAPO's upper range, a starting point to tune (lesson 14.2), not MiniMax's number. Without a trust region the update size is controlled only by the cap, the learning rate and the number of reuses of each batch. ScaleRL's recipe uses this loss in the form $\mathrm{sg}(\min(\rho_{i,t}, \epsilon_{\max}))$, a cap with no lower bound (its section 4).

### What the derivations say before any experiment

Put the five weights side by side on an **on-policy** minibatch, where every $r_{i,t} = s_i = 1$ and nothing is clipped:

- GRPO and GSPO both give $\frac{1}{G}\frac{1}{|y_i|}A_i$: identical gradients.
- DAPO and CISPO both give $A_i/\sum_j|y_j|$: identical gradients.
- Dr. GRPO gives the same token-uniform weight as DAPO, times the constant $\sum_j|y_j| / (G L)$.

So in a loop with **one** gradient step per batch, GSPO is GRPO, CISPO is DAPO and Dr. GRPO is DAPO at a smaller learning rate, apart from the advantage normalisation and the batch rules. The objectives start to differ only on the second and later minibatches of a batch, on stale batches from an asynchronous sampler, and when the sampler's probabilities differ from the trainer's: the regimes of lesson 14.3. An objective comparison that uses one minibatch, one epoch and no staleness compares aggregation and advantage normalisation, whatever the objectives are called. The lab and lesson 14.2 therefore reuse each batch for several minibatches and epochs.

| Objective | Ratio | What clipping does | Token weight across lengths | Paired settings (paper) |
|---|---|---|---|---|
| GRPO | per token | zero gradient for a token past $1\pm\epsilon$ | $1/(G \lvert y_i\rvert)$ | group-std advantage, $k_3$ KL in the loss |
| DAPO | per token | same, asymmetric range | $1/\sum_j \lvert y_j\rvert$ | clip-higher, dynamic sampling, overlong shaping, no KL |
| Dr. GRPO | per token | same | $1/(G L)$, constant | no std in the advantage |
| GSPO | per sequence, length-normalised | zero gradient for a whole response past $1+\epsilon$ (tiny ranges) | $1/(G \lvert y_i\rvert)$ | group-std advantage |
| CISPO | per token, detached | caps the weight, keeps the gradient | $1/\sum_j \lvert y_j\rvert$ | no lower bound, tune $\epsilon_{\text{high}}$ |

## Worked example

One group of $G = 2$, all numbers chosen to be easy. Response 1 is correct ($A_1 = +1$), 2 tokens, with token ratios $(1.5, 0.9)$. Response 2 is wrong ($A_2 = -1$), 3 tokens, ratios $(0.7, 1.1, 1.0)$. $\sum_j |y_j| = 5$; for Dr. GRPO take $L = 4$.

**GRPO** ($\epsilon = 0.2$). Token $(1,1)$: $A > 0$ and $1.5 > 1.2$, so it is clipped and $w = 0$. Token $(1,2)$: $\frac{1}{2}\cdot\frac{1}{2}\cdot 0.9 \cdot 1 = 0.225$. Token $(2,1)$: $A < 0$ and $0.7 < 0.8$, so it is clipped and $w = 0$. Token $(2,2)$: $\frac{1}{2}\cdot\frac{1}{3}\cdot 1.1 \cdot (-1) = -0.1833$. Token $(2,3)$: $\frac{1}{6}\cdot 1.0 \cdot (-1) = -0.1667$.

**DAPO** ($0.2$, $0.28$). Same tokens clipped ($1.5 > 1.28$; $0.7 < 0.8$); the others are divided by 5: $0.9/5 = 0.18$, $-1.1/5 = -0.22$, $-1.0/5 = -0.2$.

**Dr. GRPO.** Divided by $G L = 8$: $0.1125$, $-0.1375$, $-0.125$.

**GSPO.** $s_1 = \sqrt{1.5 \cdot 0.9} = \sqrt{1.35} = 1.162$, far above $1 + 4\times10^{-4}$ with $A > 0$, so the whole response is clipped. $s_2 = (0.7 \cdot 1.1 \cdot 1.0)^{1/3} = 0.77^{1/3} = 0.917$, far below $1 - 3\times10^{-4}$ with $A < 0$, so it is also clipped. Every weight is 0. With GSPO's ranges, ratios this large never occur in a healthy run. Move the policy a little instead, with token ratios $(1.0002, 1.0004)$ and $(0.9999, 1.0001, 1.0)$: $s_1 = 1.0003$ lies inside the range, so both tokens get $\frac{1}{2}\cdot\frac{1}{2}\cdot 1.0003 = 0.2501$. $s_2 = 1.0000$ gives $-0.1667$ each.

**CISPO** ($\epsilon_{\text{high}} = 0.28$, no lower bound). No token loses its gradient. $(1,1)$: $\min(1.5, 1.28)/5 = 0.256$. $(1,2)$: $0.18$. $(2,1)$: $0.7 \cdot (-1)/5 = -0.14$, kept, where PPO-style clipping dropped it. Then $-0.22$ and $-0.2$.

| Objective | resp. 1 tokens | resp. 2 tokens | tokens silenced |
|---|---|---|---|
| GRPO | 0, 0.225 | 0, −0.1833, −0.1667 | 2 of 5 |
| DAPO | 0, 0.18 | 0, −0.22, −0.2 | 2 of 5 |
| Dr. GRPO | 0, 0.1125 | 0, −0.1375, −0.125 | 2 of 5 |
| GSPO | 0, 0 | 0, 0, 0 | 5 of 5 |
| CISPO | 0.256, 0.18 | −0.14, −0.22, −0.2 | 0 of 5 |

`objectives_lab.py --part table` prints this table from your code, next to these values.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| `logp` (current policy, with gradient) | (B, R) = (P·G, R); toy (128, 8), main path (256, 512) | float32 (float64 in the tests) | trainer's GPU / CPU |
| `old_logp` (behaviour policy, no gradient) | (B, R) | float32 | same |
| `adv` | (B,) | float32 | same |
| `mask` | (B, R) | float32 | same |
| GSPO $\log s_i$ | (B,) | float32 | same |
| CISPO weight $\hat r$ | (B, R), detached | float32 | same |

The five losses cost the same: a few elementwise operations on (B, R) tensors, negligible next to the forward and backward pass of the policy ($6N$ FLOPs per trained token). They also need the same inputs: the behaviour log-probabilities (one extra forward pass per batch, $2N$ FLOPs per token, which the course loop always computes) and nothing else. No objective here needs a value model. Memory is the same too. The choice of objective is not a cost decision. Its cost shows up indirectly, through how many samples a run needs and how often it has to be restarted.

## Build it

```python
import torch
from frontierlab.rlscale import objectives as O

old = torch.zeros(2, 3, dtype=torch.float64)
logp = torch.log(torch.tensor([[1.5, 0.9, 1.0], [0.7, 1.1, 1.0]], dtype=torch.float64))
mask = torch.tensor([[1.0, 1, 0], [1, 1, 1]], dtype=torch.float64)
adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
print(O.gradient_weights("cispo", logp, old, adv, mask))     # [[0.256, 0.18, 0], [-0.14, -0.22, -0.2]]
hook = O.as_hook("gspo")                                     # policy_loss hook for frontierlab.posttrain.rl.train
```

`frontierlab/rlscale/objectives.py` has one function per objective (`grpo_loss`, `dapo_loss`, `drgrpo_loss`, `gspo_loss`, `gspo_token_loss`, `cispo_loss`), all with the signature `(logp, old_logp, adv, mask, **params) -> (loss, diagnostics)`. `OBJECTIVES` records each objective's default parameters and the loop settings its paper pairs with it (advantage scale, aggregation for any KL or entropy term, zero-variance filtering). `runner.train_objective(cfg, name)` runs Module 12's loop unchanged with the objective as its `policy_loss` hook. Rollout, reward, advantage and logging code is shared, and only the loss differs.

Correctness checks (`labs/common/tests/test_rlscale.py`), all in float64:

- each loss's autograd gradient equals its closed-form weight above, on random off-policy batches, including which tokens are clipped;
- the on-policy equivalences (GRPO = GSPO, DAPO = CISPO, Dr. GRPO = DAPO × constant);
- GSPO-token equals GSPO in value and gradient when advantages are per sequence;
- $s_i$ is the geometric mean, and it does not change when a response is extended with tokens of the same ratio;
- CISPO keeps a non-zero gradient exactly where DAPO's is zero;
- changing log-probabilities at masked positions changes no loss and no gradient;
- the hook runs every objective for two steps inside the real loop, and a truncated-IS weight scales the loss as it should.

**Mapping onto verl v0.9.1** (read from `verl/trainer/ppo/core_algos.py` and `trainer/config/actor/actor.yaml` at the tag). The objective is `actor_rollout_ref.actor.policy_loss.loss_mode`, one of `vanilla` (PPO/GRPO clipping), `gspo`, `cispo`, `gpg`, `clip_cov`, `kl_cov`, `geo_mean`, `dro`, `sapo`, `dppo_tv`, `dppo_kl`, `bypass_mode`. The clip ranges are `clip_ratio_low` / `clip_ratio_high` (both 0.2 by default), and verl's CISPO clamps with the same two keys, so it has a lower bound unless you set `clip_ratio_low` large. The aggregation is `loss_agg_mode`, default `token-mean`, which is DAPO's normaliser even when `loss_mode=vanilla`. "GRPO in verl" with defaults is therefore not DeepSeekMath's GRPO: write down every setting, because the name says less than it seems to.

## What the evidence says

- **GRPO: ESTABLISHED** as a baseline. It is used by DeepSeekMath and DeepSeek-R1 and in Qwen3's reasoning RL (Qwen3 report section 4.2: GRPO on 3,995 query-verifier pairs), and it is the default in TRL and verl (PUBLICLY DOCUMENTED).
- **Token-level normalisation and clip-higher (DAPO): PROMISING, widely adopted.** DAPO reports 50 points on AIME 2024 from Qwen2.5-32B-Base, above DeepSeek-R1-Zero-Qwen-32B's 47 with half its training steps (abstract, Figure 1; PUBLICLY DOCUMENTED). That is a whole-recipe result, not an ablation of the loss alone.
- **Removing the length and std normalisers (Dr. GRPO): PROMISING.** The two biases follow from the formulas (ESTABLISHED as mechanisms; lesson 12.3). The reported benefit (Oat-Zero-7B, 43.3% on AIME 2024, Table 4) is one recipe at 7B.
- **Sequence-level ratios (GSPO): PROMISING**, with its strongest evidence on MoE, where it removes the need for Routing Replay (section 5.3). The evidence is from Qwen's own runs on Qwen3-30B-A3B, so for dense models it is closer to **MODEL-SPECIFIC**.
- **Capped stop-gradient weights (CISPO): PROMISING.** MiniMax-M1 reports a 2× speed-up over DAPO on Qwen2.5-32B zero-RL, matching DAPO with 50% of the steps (section 3.1, Figure 2; company claim, PUBLICLY DOCUMENTED). ScaleRL's leave-one-out study found CISPO and GSPO reach a higher asymptote than DAPO, with CISPO marginally ahead of GSPO late in training (its Figure 5a; one lab's runs at up to 16,000 GPU-hours each). Lesson 14.4 reads those curves.
- **The on-policy equivalences** are a mathematical fact (ESTABLISHED). That published comparisons differ mainly through their off-policy regime is INFERENCE from it. Check the number of minibatches and epochs before reading any objective comparison.
- **Course measurement (free CPU, 2026-10-07, 1 seed, untuned, 80 steps with 2 epochs × 4 minibatches):** every objective trained from the SFT start's 0.21 held-out accuracy to 0.35–0.38. GSPO's clip fraction was 0.45 against 0.02–0.05 for the others, the paper's "two orders of magnitude" pattern at toy scale. One seed shows nothing about which objective is better, and these runs have no control arm of their own. Lesson 14.2 runs the comparison with a random-reward control.

## Lab

> [!NOTE]
> This lab checks implementations and runs a single-seed smoke test; it compares nothing, so it has no experiment contract. Lesson 14.2's lab is the controlled comparison ([template](../../templates/experiment-contract.md)).

**Folder:** [`labs/module-14/lesson-01/`](../../labs/module-14/) · **Time:** about 110 minutes (about 7 minutes unattended) · **Pass check:** `pytest labs/module-14/lesson-01` passes; `objectives_lab.py --part table` prints "matches hand values" for all five; your write-up explains each row of the loop table from the derivations.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 14 pilot | `python labs/module-14/lesson-01/objectives_lab.py --variant main --print` prints 5 runs of `frontierlab.rlscale.hf_rl` (Qwen3-1.7B-Base, GSM8K, 50 steps each). **PROJECTED:** 5 × 50 steps × 30–50 s per step (lesson 12.2's per-step range for P = 32, G = 8, 512 new tokens) = 2.1–3.5 GPU-hours, USD 4–11 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --print`: Qwen3-0.6B-Base (revision `da87bfb`), 8 prompts, 256 new tokens, fp32. You will see the same clip-fraction pattern on real text, but no accuracy differences worth reading |
| Free CPU | laptop; measured 6.9 minutes for the 5 runs (16-thread laptop, 8 threads, another module's jobs running) | the steps below |

### Steps

1. **Derive first.** Before writing code, fill in the worked-example table for a third group of your own (three responses, one of them long). For each objective, predict which tokens are silenced.
2. **Implement** the five TODOs in `lab.py`: `grpo_loss`, `dapo_loss`, `drgrpo_loss`, `gspo_loss`, `cispo_loss`. Run `pytest labs/module-14/lesson-01`. The tests compare values and gradients with the reference at ratio spreads of 0, $10^{-4}$ and 0.4, check the worked example, check that no gradient reaches `old_logp`, and check that padding cannot change anything.
3. **Print the table:** `python labs/module-14/lesson-01/objectives_lab.py --part table`. Explain the near-on-policy rows: why GRPO's and GSPO's are the same, and DAPO's and CISPO's.
4. **Run the loop:** `python labs/module-14/lesson-01/objectives_lab.py --part loop`. Each of your losses trains for 80 steps from the Module 12 SFT start, reusing each batch for 2 epochs × 4 minibatches.
5. **Write up** (half a page): for each objective, the clip fraction you would predict from its derivation and the one you measured; why GSPO's is an order of magnitude higher while its updates are not smaller; what CISPO's `ratio_max` means when its weights are capped at 1.28.

<details>
<summary>Hint for TODO 1</summary>

`torch.minimum(r * a, torch.clamp(r, 1 - eps, 1 + eps) * a)` with `a = adv[:, None]` and `r = torch.exp(logp - old_logp.detach())`. Multiply by the mask before any sum, and divide each row by `mask.sum(-1)`, not by R.

</details>

<details>
<summary>Hint for TODO 4</summary>

Compute $\log s_i$ first: `((logp - old_logp) * mask).sum(-1) / mask.sum(-1)`, shape (B,). Exponentiate, then apply the min and clip on (B,) tensors, and average over B. Do not average the token ratios themselves: the arithmetic mean of 2 and 0.5 is 1.25, while their geometric mean is 1.

</details>

<details>
<summary>Hint for TODO 5</summary>

The weight must be a constant for autograd: compute the ratio from `logp.detach()`. Then the objective is `w * a * logp`, whose gradient is `w * a * grad(logp)`. If you write `torch.clamp(torch.exp(logp - old), ...) * a`, you have rebuilt a PPO surrogate without the min, and the tests for capped tokens fail.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, another module's jobs sharing the CPU), `LAB_TARGET=solution`, seed 0, 80 steps, 2 epochs × 4 minibatches, SFT start at 0.21 held-out sampled accuracy:

| Objective | held-out sampled accuracy | clip fraction | max ratio | entropy (last step) | exact KL to start | seconds |
|---|---|---|---|---|---|---|
| GRPO | 0.357 | 0.046 | 18.3 | 0.246 | 0.277 | 96 |
| DAPO | 0.383 | 0.051 | 4.9 | 0.268 | 0.345 | 66 |
| Dr. GRPO | 0.367 | 0.050 | 7.0 | 0.292 | 0.291 | 72 |
| GSPO | 0.353 | 0.453 | 1.8 | 0.357 | 0.239 | 69 |
| CISPO | 0.367 | 0.025 (capped) | 13.6 | 0.272 | 0.285 | 71 |

All five learned. The accuracy spread (0.35–0.38) is within what one seed of this task varies by (lesson 12.3 measured seed pairs 0.03–0.05 apart), so no ranking follows. The diagnostics differ in kind. Under GSPO almost half the tokens sat in sequences outside its $\pm 4 \times 10^{-4}$ range, and its largest sequence ratio was 1.8. A geometric mean moves far less than individual tokens, whose ratios reached 18 under GRPO. CISPO's clip fraction counts tokens whose weight was capped, which still got gradient, so it is not comparable with the others' zero-gradient fraction. A log that reports "clip_frac" for all five under one name invites a wrong comparison.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-14/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-14/lesson-01`.

</details>

## Common mistakes

- **Switching `loss_mode` and calling it an objective comparison.** The framework's aggregation, clip and advantage defaults come along. Write down all of them for every arm.
- **Comparing objectives with one gradient step per batch.** On-policy, GSPO = GRPO and CISPO = DAPO. Any difference you see then comes from the normalisers and batch rules.
- **Averaging token ratios for GSPO** (arithmetic mean), or not normalising by length (product of ratios). Both break the clip range.
- **Letting gradient flow through CISPO's weight.** Without the stop-gradient it is a PPO surrogate without the min: tokens past the bound get a gradient proportional to the weight, and nothing caps the step.
- **Using GRPO's clip range with GSPO, or GSPO's with GRPO.** $0.2$ on a sequence ratio clips almost nothing. $4 \times 10^{-4}$ on token ratios clips almost everything.
- **Reading `clip_frac` across objectives as one quantity.** For GRPO/DAPO/Dr. GRPO it is the share of tokens with zero gradient, for GSPO the share of tokens in clipped sequences, for CISPO the share of capped weights.

## References

- Z. Shao et al., *DeepSeekMath*, 2024, section 4.1 (Eqs. 3–4) and section 4.2. https://arxiv.org/abs/2402.03300
- Q. Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025, Eq. 8, sections 3.2 and 4.1. https://arxiv.org/abs/2503.14476
- Z. Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, 2025, sections 3.1–3.2. https://arxiv.org/abs/2503.20783
- C. Zheng et al., *Group Sequence Policy Optimization*, 2025, sections 4.1–4.3 and 5.1–5.4. https://arxiv.org/abs/2507.18071
- MiniMax, *MiniMax-M1*, 2025, section 3.1. https://arxiv.org/abs/2506.13585
- J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, section 3. https://arxiv.org/abs/1707.06347
- D. Khatri et al., *The Art of Scaling Reinforcement Learning Compute for LLMs*, 2025, section 4. https://arxiv.org/abs/2510.13786
- Qwen Team, *Qwen3 Technical Report*, 2025, section 4.2. https://arxiv.org/abs/2505.09388
- verl v0.9.1, `verl/trainer/ppo/core_algos.py`. https://github.com/volcengine/verl/tree/v0.9.1
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[14.2 · A controlled objective comparison](lesson-02.md)
