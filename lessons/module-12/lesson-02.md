---
id: "12.2"
module: 12
minutes: 45
practice_minutes: 120
prerequisites: ["12.1", "01.4"]
objectives:
  - Derive the outcome-reward policy gradient for a sequence and show why subtracting a baseline that does not depend on the sampled response leaves it unbiased.
  - Compute group-mean, leave-one-out and standardised advantages by hand, and predict the fraction of zero-variance groups from the pass rate and group size.
  - Implement the k1, k2 and k3 KL estimators and prove by exact enumeration which divergence each one's gradient follows when it is used as a loss term.
  - Place the KL penalty in the reward or in the loss and measure the exact reverse KL each placement actually controls.
  - Compare advantage estimators in a controlled RL experiment with seeds, paired intervals and a pre-stated decision rule, and map each choice to the TRL and verl options that implement it.
volatility: concept
sources:
  - title: "Shao et al., DeepSeekMath (section 4.1, Eq. 3-4; section 4.1.2: GRPO)"
    url: https://arxiv.org/abs/2402.03300
  - title: "Ahmadian et al., Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (section 2.3: RLOO)"
    url: https://arxiv.org/abs/2402.14740
  - title: "Liu et al., Understanding R1-Zero-Like Training: A Critical Perspective (sections 3.1-3.2: Dr. GRPO)"
    url: https://arxiv.org/abs/2503.20783
  - title: "Yu et al., DAPO (sections 2.3, 3.2: no KL, dynamic sampling)"
    url: https://arxiv.org/abs/2503.14476
  - title: "Khatri et al., The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL, section 3.2)"
    url: https://arxiv.org/abs/2510.13786
  - title: "Schulman, Approximating KL Divergence (blog, 2020)"
    url: http://joschu.net/blog/kl-approx.html
  - title: "Tang and Munos, On a few pitfalls in KL divergence gradient estimation for RL (section 3)"
    url: https://arxiv.org/abs/2506.09477
  - title: "Shah et al., A Comedy of Estimators: On KL Regularization in RL Training of LLMs"
    url: https://arxiv.org/abs/2512.21852
  - title: "Zhang et al., On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning"
    url: https://arxiv.org/abs/2505.17508
  - title: "Cui et al., The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models (Eq. 6, Theorem 2)"
    url: https://arxiv.org/abs/2505.22617
  - title: "Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation (Eq. 16)"
    url: https://arxiv.org/abs/1506.02438
  - title: "He et al., Skywork Open Reasoner 1 Technical Report (section 3.1)"
    url: https://arxiv.org/abs/2505.22312
last_verified: "2026-10-07"
---

# 12.2 · Policy-gradient estimators for LLMs

Every RL recipe for language models is the same gradient estimate with different choices bolted on: what to subtract from the reward, whether to divide by a standard deviation, what to do with prompts where every sample got the same reward, where to put the KL penalty and which estimator of it to use, and whether to reward entropy. This lesson derives the estimate, implements each choice in a few lines, proves what two of them really do by exact enumeration, and then compares them in controlled runs of the course's RL loop.

## Why this matters at a frontier lab

GRPO, RLOO, Dr. GRPO, DAPO and ScaleRL differ in a handful of lines of the advantage and loss code, and each paper reports that its lines matter. A research engineer has to know which lines change the *expected* gradient (the objective you are optimising) and which only change its *variance* or *scale* (how fast and how stably you get there), because the two kinds of change need different evidence. The KL term is the clearest case: GRPO's widely copied KL-in-the-loss with the k3 estimator has a gradient that regularises toward the reference in the *forward* KL direction, not the reverse KL that the objective writes down. That is not an opinion; it is a three-line calculation, and this lesson checks it to 12 decimal places.

## The idea

### The policy gradient for a whole response

A prompt $x$, a response $y = (y_1, \dots, y_T)$ sampled from the policy $\pi_\theta$, and one scalar reward $R(x, y)$ at the end (an outcome reward). The objective is $J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(x, y)]$, and the score-function identity $\nabla_\theta \pi = \pi \nabla_\theta \log \pi$ gives

$$\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta}\Big[R(x, y) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\Big].$$

Every token of the response gets the same weight $R$: the sequence's log-probability is the sum of its tokens' log-probabilities. Here $\theta$ are the policy's parameters, $y_{<t}$ the tokens before $t$, and the expectation is over responses sampled from the current policy.

### Baselines

Subtract any $b(x)$ that does not depend on the sampled $y$:

$$\mathbb{E}_y\big[(R - b)\,\nabla \log \pi(y)\big] = \mathbb{E}_y[R\,\nabla \log \pi(y)] - b\,\underbrace{\mathbb{E}_y[\nabla \log \pi(y)]}_{= \nabla \sum_y \pi(y) = \nabla 1 = 0}.$$

The expected gradient is unchanged; the variance falls when $b$ is close to $\mathbb{E}[R \mid x]$. The quantity $A = R - b$ is the **advantage**. LLM recipes sample a **group** of $G$ responses per prompt and build $b$ from the group:

| Estimator | Advantage of response $i$ in a group of rewards $r_1..r_G$ | Bias |
|---|---|---|
| REINFORCE | $r_i$ | none (high variance) |
| group mean (GRPO) | $r_i - \bar r$ | $\bar r$ contains $r_i$: expected gradient scaled by $(G-1)/G$, same direction |
| leave-one-out (RLOO) | $r_i - \frac{1}{G-1}\sum_{j \ne i} r_j = \frac{G}{G-1}(r_i - \bar r)$ | none |
| GRPO standardised | $(r_i - \bar r)/(\mathrm{std}(r) + \epsilon)$ | std contains $r_i$: not a constant rescaling |
| batch standardised (ScaleRL) | $(r_i - \bar r)/(\mathrm{std}_{\text{batch}} + \epsilon)$ | one scale for the whole batch |

GRPO's outcome supervision sets $\hat A_{i,t} = (r_i - \mathrm{mean}(r))/\mathrm{std}(r)$ for every token $t$ of response $i$ (DeepSeekMath section 4.1.2). RLOO's estimator is $\frac{1}{k}\sum_i [R(y_i) - \frac{1}{k-1}\sum_{j\ne i} R(y_j)]\,\nabla \log \pi(y_i)$ (Ahmadian et al. section 2.3). Dr. GRPO argues the std division gives a *question-level difficulty bias*: groups with small std (nearly all right or nearly all wrong) get their advantages inflated, so very easy and very hard prompts get more weight (Liu et al. section 3.1), and removes it (section 3.2). ScaleRL compares prompt-level, batch-level and no standardisation and uses batch-level (Khatri et al. section 3.2).

PPO-style RLHF instead learns a **value model** $V(s_t)$ and uses per-token advantages from generalised advantage estimation, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ and $\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$ (Schulman et al. 2015, Eq. 16), with $\gamma$ the discount and $\lambda$ the bias-variance knob. The group baselines exist to drop that second model; `frontierlab.posttrain.advantages.gae` is there for comparison and for Module 16's multi-turn credit assignment.

### Zero-variance groups

With a binary verifier and pass rate $p$ on a prompt, a group of $G$ is all-correct with probability $p^G$ and all-wrong with $(1-p)^G$. In both cases every advantage is 0 under any group baseline: the prompt contributes no policy gradient, only its KL and entropy terms, and the effective batch shrinks. DAPO over-samples and keeps only groups with $0 < |\text{correct}| < G$ ("dynamic sampling", section 3.2, Eq. 11); ScaleRL excludes zero-variance prompts from the effective batch and permanently drops prompts with pass rate $\ge 0.9$ ("no-positive-resampling", section 3.2); Skywork-OR1 rejects zero-advantage groups because they "do not contribute to the policy loss but may influence the KL loss or entropy loss" (section 3.1). The fraction of zero-variance groups is a first-class training metric: it tells you whether your prompt set is too easy or too hard for the current policy.

### KL to the reference: where and which estimator

The regularised objective is $\mathbb{E}[R] - \beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$, with $\pi_{\text{ref}}$ the starting (SFT) policy and $\beta \ge 0$. Per token, with $\delta_t = \log \pi_\theta(y_t) - \log \pi_{\text{ref}}(y_t)$ on a sampled token, three estimators of the KL (Schulman's blog, there written with $r = \pi_{\text{ref}}/\pi_\theta$):

$$k_1 = \delta_t, \qquad k_2 = \tfrac{1}{2}\delta_t^2, \qquad k_3 = e^{-\delta_t} - 1 + \delta_t.$$

$k_1$ is unbiased but negative on some samples; $k_2$ is biased with low variance; $k_3$ is unbiased, low-variance and always $\ge 0$. GRPO uses $k_3$ (DeepSeekMath Eq. 4, $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$).

**Placement 1, in the reward** (InstructGPT Eq. 2; PPO-style). The detached value $\beta \sum_t k_{1,t}$ is subtracted from the reward and flows through the advantage. Then the policy-gradient machinery differentiates it correctly: the gradient of $-\beta\,\mathrm{KL}(\pi \| \pi_{\text{ref}})$ is $-\beta\,\mathbb{E}[(\log \pi - \log \pi_{\text{ref}})\,\nabla \log \pi]$, which is exactly "reward $-\beta k_1$ times the score".

**Placement 2, in the loss** (GRPO Eq. 3). The estimator is added to the loss and *differentiated directly*, with the sample treated as fixed. Now the question is not whether its value is unbiased but what its gradient is in expectation over $y \sim \pi$. Write $\nabla$ for $\nabla_\theta$ and $\delta = \log\pi - \log\pi_{\text{ref}}$, so $\nabla\delta = \nabla\log\pi$:

- $\nabla k_1 = \nabla \log \pi$, and $\mathbb{E}_\pi[\nabla\log\pi] = 0$: **zero-mean noise**, no regularisation.
- $\nabla k_2 = \delta\,\nabla \log \pi$, whose expectation $\mathbb{E}_\pi[(\log\pi - \log\pi_{\text{ref}})\nabla\log\pi]$ is $\nabla\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$: **the intended reverse-KL gradient**.
- $\nabla k_3 = (1 - e^{-\delta})\nabla\log\pi = (1 - \pi_{\text{ref}}/\pi)\nabla\log\pi$, with expectation $0 - \sum_y \pi_{\text{ref}}(y)\nabla\log\pi(y) = \nabla\,\mathrm{KL}(\pi_{\text{ref}}\|\pi)$: **the forward-KL gradient**.

Tang and Munos derive exactly these three results (section 3) and add a second pitfall: a per-token KL loss ignores the effect of token $y_t$ on the KL of later tokens, so it gives only part of the sequence-level gradient. Zhang et al. show that the $k_3$ penalty is exactly an *unnormalised* KL and that GRPO's KL term has an off-policy importance-weighting error; Shah et al. report that estimator configurations with unbiased gradients work better in and out of domain and biased ones can destabilise training. In practice the forward and reverse KL gradients point in similar directions when $\pi$ is close to $\pi_{\text{ref}}$ (both are second-order in the difference), so $k_3$-in-loss *does* keep the policy near the reference; it just optimises a different regulariser than the objective states. DAPO removes the KL term entirely (section 2.3); Tülu 3 keeps it ($\beta = 0.05$ for the final 8B model).

### Entropy

The entropy of the policy at a position, $H = -\sum_v \pi(v)\log\pi(v)$ over the vocabulary, measures how much the policy still explores. RL on verifiable rewards drives it down fast. Cui et al. fit $R = -a\,e^{H} + b$ between validation reward and entropy across runs (Eq. 6): when entropy is spent, reward stops improving. Their Theorem 2 says that, under natural policy gradient, the entropy change is approximately $-\eta\,\mathrm{Cov}(\log\pi(a\mid s), A(s,a))$: entropy falls when the policy raises tokens it already likes (high log-probability, positive advantage). An **entropy bonus** adds $-c\,H$ to the loss; Clip-Cov and KL-Cov act on the high-covariance tokens only (section 4.2).

## Worked example

**Advantages for one group** of $G = 4$ with rewards $(1, 0, 0, 0)$:

- group mean: $\bar r = 0.25$, so $A = (0.75, -0.25, -0.25, -0.25)$;
- leave-one-out: $(1 - 0, 0 - 1/3, 0 - 1/3, 0 - 1/3) = (1, -0.333, -0.333, -0.333)$, which is $\frac{4}{3}\times$ the group-mean advantages;
- GRPO: the sample std (ddof 1) is $\sqrt{(0.75^2 + 3 \cdot 0.25^2)/3} = 0.5$, so $A = (1.5, -0.5, -0.5, -0.5)$;
- a group $(1, 1, 1, 1)$: $A = (0, 0, 0, 0)$ under all of them; with std normalisation $0/(0 + 10^{-6}) = 0$.

**Zero-variance fraction.** $G = 8$. At pass rate $p = 0.2$: $0.2^8 + 0.8^8 = 0.0000026 + 0.168 = 0.168$, one group in six carries no signal. At $p = 0.9$: $0.9^8 + 0.1^8 = 0.430$. As training succeeds, more of the batch goes silent.

**KL estimators on one token.** $\pi = 0.5$, $\pi_{\text{ref}} = 0.25$, $\delta = \ln 2 = 0.693$: $k_1 = 0.693$, $k_2 = 0.240$, $k_3 = 0.5 - 1 + 0.693 = 0.193$.

**Their gradients, exactly.** A 3-outcome policy with logits $(0.5, -0.2, 1.0)$ and reference logits $(0, 0.3, -0.4)$. Enumerating every outcome (the lab's TODO 4) in float64:

| | expected gradient w.r.t. logits |
|---|---|
| $k_1$ as loss | $(0, 0, 0)$ to $10^{-16}$ |
| $k_2$ as loss | $(-0.0998, -0.2074, 0.3072)$ |
| true $\nabla\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ (value 0.2733) | $(-0.0998, -0.2074, 0.3072)$ |
| $k_3$ as loss | $(-0.0132, -0.2891, 0.3022)$ |
| true $\nabla\,\mathrm{KL}(\pi_{\text{ref}}\|\pi)$ (value 0.2878) | $(-0.0132, -0.2891, 0.3022)$ |

**Entropy.** $\pi = (0.5, 0.25, 0.25)$: $H = 0.5\ln 2 + 2 \cdot 0.25 \ln 4 = 0.347 + 0.693 = 1.040$ nats; uniform over the 35-token vocabulary would be $\ln 35 = 3.56$.

**GAE.** Rewards $(0, 0, 1)$, values $(0.2, 0.4, 0.5)$ then 0, $\gamma = 1$, $\lambda = 0.5$: $\delta = (0 + 0.4 - 0.2,\ 0 + 0.5 - 0.4,\ 1 + 0 - 0.5) = (0.2, 0.1, 0.5)$; $\hat A_3 = 0.5$, $\hat A_2 = 0.1 + 0.5 \cdot 0.5 = 0.35$, $\hat A_1 = 0.2 + 0.5 \cdot 0.35 = 0.375$.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| rewards | (P, G) = (16, 8) | float32 | CPU |
| advantages | (P, G), computed in float64, returned float32, then flattened to (P·G,) | float32 | CPU |
| policy, old, reference log-probs | (P·G, R) = (128, 8) | float32 (logits cast before the softmax) | same as the model |
| response mask | (128, 8) | float32 | same |
| exact token KL (toy only) | (128, 8): a sum over the 35-token vocabulary | float32 | same |

Cost per RL step, in forward-pass units of $2N$ FLOPs per token for a policy of $N$ parameters: sampling $P G R$ tokens (memory-bound decoding), one forward of the reference model and one of the old policy over the $P G (P_{\text{len}} + R)$ prompt-plus-response tokens ($2 \times 2N$ per token), and the update's forward and backward ($6N$ per token, times the number of epochs). KL in the loss and KL in the reward cost the same: one reference forward. The entropy needs the full softmax, which the loss already computes.

Main path (PROJECTED, pending the Module 12 pilot): Qwen3-1.7B-Base, $P = 32$, $G = 8$, a 4-shot GSM8K prompt of about 700 tokens and responses of about 300. Training-side FLOPs per step $\approx 256 \cdot 1{,}000 \cdot (6 + 2 + 2) \cdot 1.72 \times 10^9 = 4.4 \times 10^{15}$, about 11 s at 40% of 989 TFLOP/s; generation of $256 \times 300$ tokens with `generate` at a few thousand tokens per second is 20–40 s. So about 30–50 s per step, 100 steps in 1–1.5 GPU-hours per run.

## Build it

```python
import torch
from frontierlab.posttrain import advantages as A, kl as K

R = torch.tensor([[1., 0, 0, 0], [1, 1, 1, 1]])
print(A.group_advantages(R, "mean", "group"))   # [[1.5, -.5, -.5, -.5], [0, 0, 0, 0]]
print(A.group_advantages(R, "loo", "none"))     # [[1, -1/3, -1/3, -1/3], [0, 0, 0, 0]]
print(A.zero_variance(R))                       # [False, True]

out = K.exact_categorical(torch.tensor([0.5, -0.2, 1.0]), torch.tensor([0.0, 0.3, -0.4]))
print(out["k2"], out["grad_reverse_kl"])        # equal
print(out["k3"], out["grad_forward_kl"])        # equal
```

The RL loop `python -m frontierlab.posttrain.rl` exposes every choice as a flag: `--baseline none|mean|loo`, `--scale none|group|batch`, `--zero-var keep|filter`, `--kl-place none|reward|loss`, `--kl-kind k1|k2|k3`, `--kl-beta`, `--entropy-coef`. It logs per step the pass rate, the zero-variance fraction (and how many of those groups were all wrong), the entropy, the mean $k_3$ and the **exact** per-token reverse KL to the reference (`kl_exact`, a sum over the vocabulary, affordable only because the vocabulary has 35 tokens), clip fraction, maximum ratio, lengths and truncation. Correctness checks (`labs/common/tests/test_posttrain.py`): the advantages against the hand values above; the three gradient identities by enumeration to $10^{-12}$; GAE against the hand values; the loop resumes exactly from a checkpoint (stopped at step 2 and restarted gives identical metrics at steps 3–4).

### Mapping to the frameworks

You will meet the same choices under other names in TRL (Module 13) and verl (Module 14). The names below were read from the source at the pinned tags, TRL v1.14.1 (`trl/trainer/grpo_config.py`, `grpo_trainer.py`, `rloo_trainer.py`) and verl v0.9.1 (`verl/trainer/config/actor/actor.yaml`, `verl/trainer/config/algorithm.py`, `verl/trainer/ppo/core_algos.py`), on 2026-10-07. Defaults move between releases; check the installed version's config before relying on one.

| Course flag | TRL v1.14.1 | verl v0.9.1 |
|---|---|---|
| `--group` | `GRPOConfig.num_generations` (default 8) | `actor_rollout_ref.rollout.n` |
| `--baseline mean --scale group` | `scale_rewards="group"` (the default) | `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=True` (the default) |
| `--scale none` | `scale_rewards="none"` | `algorithm.norm_adv_by_std_in_grpo=False` |
| `--scale batch` | `scale_rewards="batch"` | — |
| `--baseline loo` | `RLOOTrainer` (always leave-one-out; KL in the reward as $\beta\sum_t k_1$, `beta` default 0.05) | `algorithm.adv_estimator=rloo` |
| `--kl-place loss --kl-kind k3` | `beta > 0` (default 0.0: no reference model loaded). The per-token term is $k_3$; with `use_bias_correction_kl=True` (default) it is also multiplied by the importance ratio | `actor_rollout_ref.actor.use_kl_loss=True`, `kl_loss_type=low_var_kl` (or `k3`), `kl_loss_coef` |
| `--kl-kind k2` / `k1` | — | `kl_loss_type=mse` (`k2`) / `kl` (`k1`); a `+` suffix (`k3+`) keeps the $k_3$ value with the $k_2$ gradient |
| `--kl-place reward` | — (RLOO only) | `algorithm.use_kl_in_reward=True`, `algorithm.kl_penalty`, `algorithm.kl_ctrl.kl_coef` |
| `--entropy-coef` | `entropy_coef` | `actor_rollout_ref.actor.entropy_coeff` |
| `gae` | — | `algorithm.adv_estimator=gae`, `compute_gae_advantage_return`, `algorithm.gamma`, `algorithm.lam` |

verl's `+` variants exist precisely because of the gradient identities above: they report the low-variance $k_3$ value but differentiate $k_2$.

## What the evidence says

- **Group baselines instead of a value model: ESTABLISHED** for outcome-reward RL on LLMs (GRPO in DeepSeekMath; RLOO; most open reasoning recipes since). Ahmadian et al. report REINFORCE-style RLOO beating PPO by 3.2% to 20.3% in win rate in their RLHF settings (PUBLICLY DOCUMENTED, their benchmarks).
- **Standardising by the group std: debated.** GRPO does it; Dr. GRPO argues it biases toward very easy and very hard prompts and removes it; ScaleRL measures prompt-level vs batch-level vs none and chooses batch-level. Each is PUBLICLY DOCUMENTED for its own setup; which is best in general is not settled (PROMISING for each alternative).
- **Filtering zero-variance groups: ESTABLISHED as practice** (DAPO dynamic sampling, ScaleRL zero-variance filtering, Skywork-OR1 rejection sampling); its benefit is a larger *effective* batch for the same compute, not a change of objective.
- **KL estimators as loss terms: the gradient identities are mathematics** (Tang and Munos section 3; checked here exactly). **Whether a KL term helps reasoning RL at all: debated.** DAPO removes it; Tülu 3 keeps $\beta = 0.05$ for 8B; Gao et al. found a KL penalty acts like early stopping rather than improving the reward-KL frontier (lesson 12.1). REASONABLE INDUSTRY PRACTICE: if you keep a KL term in the loss, use an estimator whose gradient is the one you intend ($k_2$ for reverse KL), or put $k_1$ in the reward.
- **Entropy collapse limits RL gains: PROMISING** (Cui et al.'s fit across their runs). Entropy bonuses are old practice in RL; for LLM reasoning RL their benefit is reported inconsistently (INFERENCE from the spread of recipes that do and do not use one).
- **Course measurement (free CPU, 2026-10-07):** no-baseline REINFORCE beat GRPO on held-out sampled accuracy by +0.033 [+0.017, +0.050] at one untuned learning rate (3 seeds), RLOO and Dr. GRPO were within noise of GRPO; $k_1$ in the loss left the exact KL unchanged while $k_2$, $k_3$ and $k_1$-in-reward cut it by 0.12–0.17 nats. Details and caveats in the lab's results box. A 308k-parameter toy says nothing about the size of these effects in a 1.7B model.

## Lab

**Folder:** [`labs/module-12/lesson-02/`](../../labs/module-12/) · **Time:** about 2 hours (about 20 minutes of it unattended) · **Pass check:** `pytest labs/module-12/lesson-02` passes; `estimator_lab.py` prints the three comparisons; the write-up applies the decision rule and explains every KL arm's `kl_exact` with the identities above.

### Experiment contract

- **Question:** on the toy verifiable task, from the same SFT start and at equal samples, which advantage estimator (REINFORCE, GRPO, RLOO, group mean without std) gives the highest held-out accuracy after 120 steps; and which KL placement and estimator actually limits the reverse KL to the SFT policy? Decision informed: the default estimator and KL setting of the Module 12 project loop and Module 14's comparison.
- **Hypothesis:** every baseline beats REINFORCE without one (variance); GRPO, RLOO and Dr. GRPO are within noise of each other at this scale (their differences are second-order); $k_1$-in-loss gives the same reverse KL as no KL (its gradient is zero in expectation); $k_2$-in-loss and $k_1$-in-reward reduce reverse KL; $k_3$-in-loss reduces it too, through the forward KL. Status: the KL predictions follow from the mathematics; the estimator ranking is a reported effect that may not appear with 3 seeds at 308k parameters.
- **Baseline:** `grpo` (group mean and std, the most copied recipe), lr $3 \times 10^{-4}$, not tuned per arm.
- **Changed variable:** the advantage estimator (Part 1); the KL placement and estimator at $\beta = 0.5$ (Part 2); an entropy bonus of 0.02 (Part 3). **Controlled:** the SFT checkpoint, 16 prompts × 8 samples per step, 8 new tokens, temperature 1, 2 minibatches, token-mean aggregation, clip 0.2, strict verifier, 120 steps, seeds 0–2 (0–1 for KL), evaluation on 300 held-out problems with a fixed sampling seed.
- **Comparison axis:** equal samples (steps × prompts × group). Every arm costs the same rollouts; KL arms add one reference forward that the others also compute (for logging).
- **Budget:** free CPU, about 20 minutes for 25 runs of 120 steps (measured).
- **Metrics and decision rule:** primary: held-out sampled accuracy (pass@1 at temperature 1) at step 120; paired by seed against `grpo`, 95% t-interval over the per-seed differences. "Better" if the interval is above 0, "worse" if below, else "inconclusive" (TODO 6). Secondary: mean training pass rate over the run (sample efficiency), zero-variance fraction, entropy, `kl_exact`.
- **Correctness checks:** `test_posttrain.py` (advantages, KL identities, resume); your TODO tests; the KL arms' `kl_exact` is computed from the full distributions, not from an estimator.
- **Fallback evidence:** a null result reported with the seed standard deviation; published comparisons (Dr. GRPO section 3, ScaleRL section 3.2) labelled as published.
- **Limits:** 308k parameters, 2-digit addition, responses of 2–4 tokens (length effects are lesson 12.3's), 120 steps, 2–3 seeds.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 12 pilot | `python labs/module-12/lesson-02/estimator_lab.py --variant main --print` prints the 8 runs (4 estimators × 2 seeds, 100 steps of `frontierlab.posttrain.hf` on Qwen3-1.7B-Base and GSM8K). **PROJECTED:** 8 × 1–1.5 GPU-hours = 8–12 GPU-hours (formula in "Shapes and cost"), USD 16–36 |
| Free GPU (Colab/Kaggle T4) | T4 | the main-path commands with `--model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --prompts 8 --max-new 256`, fp32; you will not see 1.7B-scale learning curves |
| Free CPU | laptop; measured 20 minutes for the 25 runs (other jobs running) | the steps below |

### Steps

1. **Implement** the six TODOs in `lab.py` and run `pytest labs/module-12/lesson-02`.
2. **Predict** before running: write down, for each KL arm, whether its `kl_exact` at step 120 will be above, near or below `kl-none`, and why.
3. **Run** `python labs/module-12/lesson-02/estimator_lab.py` (your `group_advantages` drives every run).
4. **Write up:** the decision for each estimator against GRPO with its interval; the zero-variance fraction and what it says about the prompt difficulty; the KL table against your prediction; whether the entropy bonus changed entropy and accuracy.

<details>
<summary>Hint for TODO 4</summary>

Make `theta` a float64 leaf with `requires_grad_(True)`, compute `logp = torch.log_softmax(theta, -1)` once, then for each outcome `y` call `torch.autograd.grad(k[y], theta, retain_graph=True)` and add `p[y] * grad`, where `p = logp.exp().detach()`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, another module's jobs sharing the CPU): 25 runs of 120 steps, 20 minutes in all (about 45 s per run). SFT start: 0.21 sampled held-out accuracy.

Part 1, held-out sampled accuracy at step 120 (seeds 0, 1, 2) and the paired difference against `grpo` with its 95% t-interval:

| Arm | per seed | mean | vs grpo | decision |
|---|---|---|---|---|
| grpo | 0.343, 0.310, 0.333 | 0.329 | — | — |
| reinforce | 0.370, 0.350, 0.367 | 0.362 | +0.033 [+0.017, +0.050] | better |
| rloo | 0.347, 0.333, 0.320 | 0.333 | +0.004 [−0.041, +0.050] | inconclusive |
| drgrpo | 0.367, 0.297, 0.370 | 0.344 | +0.016 [−0.049, +0.080] | inconclusive |

The hypothesis that every baseline beats no baseline was **wrong here**, and the reason is worth more than the ranking. With 0/1 rewards and no baseline, REINFORCE only ever pushes correct responses up; wrong ones get advantage 0. It lowered entropy the most (0.327 against 0.357 for GRPO), and sampled accuracy at temperature 1 rewards a sharper distribution. The comparison was also not at equal step size: GRPO's standardised advantages are up to about 2.6 in magnitude, REINFORCE's at most 1, with one untuned learning rate for all arms. So the result says "at this learning rate and 120 steps", not "baselines do not help"; a tuned comparison (a small learning-rate sweep per arm, the plan's budget rule) is the honest next step. The variance argument for baselines is about large reward scales and long runs, where an all-positive REINFORCE gradient wanders. RLOO and Dr. GRPO were within noise of GRPO, as predicted. Mean training pass rates differed by less than 0.02 between all arms; the zero-variance fraction was 0.14–0.15 in every arm (about one group in seven silent, most of them all wrong at this pass rate).

Part 2, KL at $\beta = 0.5$ (2 seeds), exact per-token reverse KL to the SFT policy over the last 20 steps:

| Arm | kl_exact | vs kl-none | held-out accuracy vs kl-none |
|---|---|---|---|
| kl-none | 0.210, 0.205 | — | — |
| kl-loss-k1 | 0.205, 0.223 | +0.006 [−0.137, +0.150] | −0.010 [−0.052, +0.032] |
| kl-loss-k2 | 0.081, 0.088 | −0.123 [−0.192, −0.054] | −0.018 [−0.167, +0.130] |
| kl-loss-k3 | 0.057, 0.049 | −0.154 [−0.180, −0.129] | −0.075 [−0.181, +0.031] |
| kl-reward-k1 | 0.034, 0.037 | −0.172 [−0.213, −0.131] | −0.100 [−0.185, −0.015] |

Exactly as the identities predict: $k_1$ in the loss did nothing to the KL (its logged $k_3$ mean was even slightly *higher* than without a KL term). $k_2$ in the loss and $k_1$ in the reward both reduced the reverse KL; at the same $\beta$ they are not equally strong (the reward form puts the penalty through the advantage, whose scale differs), so compare placements at matched KL, not at matched $\beta$. $k_3$ in the loss reduced the reverse KL too: it optimises the forward KL, but close to the reference the two move together. Every arm that held the KL down also cost held-out accuracy at 120 steps (significantly only for the reward placement), which is the trade-off lesson 12.4 measures on retention.

Part 3: the entropy bonus of 0.02 raised entropy by 0.011 [+0.002, +0.020] and held-out accuracy by +0.024 [−0.038, +0.087], inconclusive.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-12/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-12/lesson-02`.

</details>

## Common mistakes

- **Using the group mean of *all* groups as the baseline** (or reshaping rewards so that groups mix prompts). The baseline must come from the same prompt's samples, or it adds the variance of prompt difficulty.
- **Calling a KL estimator "unbiased" and using it as a loss.** Unbiasedness of the value says nothing about the gradient; $k_1$ in the loss does nothing in expectation, $k_3$ in the loss regularises the forward KL.
- **Logging only the estimator you optimise.** Log a quantity you can trust (here the exact KL; on a real model, $k_1$ and $k_3$ means over a held-out batch) next to it.
- **Ignoring zero-variance groups.** A batch that is 60% silent is a smaller batch than its size says; report the effective fraction.
- **Comparing arms at one seed.** Two seeds of the same arm often differ more than two arms do at this scale.

## References

- Z. Shao et al., *DeepSeekMath*, 2024, section 4.1. https://arxiv.org/abs/2402.03300
- A. Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, 2024, section 2.3. https://arxiv.org/abs/2402.14740
- Z. Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, 2025, sections 3.1–3.2. https://arxiv.org/abs/2503.20783
- Q. Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025, sections 2.3 and 3.2. https://arxiv.org/abs/2503.14476
- D. Khatri et al., *The Art of Scaling Reinforcement Learning Compute for LLMs*, 2025, section 3.2. https://arxiv.org/abs/2510.13786
- J. Schulman, *Approximating KL Divergence*, 2020. http://joschu.net/blog/kl-approx.html
- Y. Tang, R. Munos, *On a few pitfalls in KL divergence gradient estimation for RL*, 2025, section 3. https://arxiv.org/abs/2506.09477
- V. Shah et al., *A Comedy of Estimators: On KL Regularization in RL Training of LLMs*, 2025. https://arxiv.org/abs/2512.21852
- Y. Zhang et al., *On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning*, 2025. https://arxiv.org/abs/2505.17508
- G. Cui et al., *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*, 2025, Eq. 6 and Theorem 2. https://arxiv.org/abs/2505.22617
- J. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015, Eq. 16. https://arxiv.org/abs/1506.02438
- J. He et al., *Skywork Open Reasoner 1 Technical Report*, 2025, section 3.1. https://arxiv.org/abs/2505.22312
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[12.3 · Details that change results](lesson-03.md)
