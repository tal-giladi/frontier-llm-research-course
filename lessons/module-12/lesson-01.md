---
id: "12.1"
module: 12
minutes: 40
practice_minutes: 120
prerequisites: ["01.3", "01.4"]
objectives:
  - Distinguish verifiable from learned rewards, and measure a verifier's false-positive and false-negative rates against its specification with a probe set before any RL run uses it.
  - Derive the Bradley-Terry reward-model loss, train a small reward model on preference pairs, and report its held-out accuracy and expected calibration error.
  - Compute the exact KL of best-of-n sampling and an unbiased best-of-n estimate from one sample pool, and use them to draw proxy-reward and gold-reward curves against KL.
  - Detect reward-model over-optimisation against a held-out gold signal, fit Gao et al.'s best-of-n form, and state how reward-model data size moved the over-optimisation point.
  - Choose and pin the Stage D base model with its licence and revision, and say what its pretraining makes uncertain about RL results.
volatility: concept
sources:
  - title: "Gao, Schulman, Hilton, Scaling Laws for Reward Model Overoptimization (sections 1, 2, 2.1, 3.6)"
    url: https://arxiv.org/abs/2210.10760
  - title: "Ouyang et al., Training language models to follow instructions with human feedback (section 3.5, Eq. 1 and 2)"
    url: https://arxiv.org/abs/2203.02155
  - title: "Lambert et al., Tülu 3: Pushing Frontiers in Open Language Model Post-Training (section 6, Eq. 7-8)"
    url: https://arxiv.org/abs/2411.15124
  - title: "Lightman et al., Let's Verify Step by Step"
    url: https://arxiv.org/abs/2305.20050
  - title: "Shao et al., Spurious Rewards: Rethinking Training Signals in RLVR (sections 2.2, 3)"
    url: https://arxiv.org/abs/2506.10947
  - title: "Singhal et al., A Long Way to Go: Investigating Length Correlations in RLHF"
    url: https://arxiv.org/abs/2310.03716
  - title: "Skalse et al., Defining and Characterizing Reward Hacking"
    url: https://arxiv.org/abs/2209.13085
  - title: "Malik et al., RewardBench 2: Advancing Reward Model Evaluation"
    url: https://arxiv.org/abs/2506.01937
  - title: "Qwen Team, Qwen3 Technical Report"
    url: https://arxiv.org/abs/2505.09388
  - title: "Qwen/Qwen3-1.7B-Base model card (Apache-2.0; revision ea980cb)"
    url: https://huggingface.co/Qwen/Qwen3-1.7B-Base
last_verified: "2026-10-07"
---

# 12.1 · Rewards

Reinforcement learning on a language model optimises whatever number the reward function returns, not what you meant by it. This lesson builds both kinds of reward a frontier lab uses — a verifier that checks an answer, and a reward model learned from preferences — measures how wrong each one is, and then pushes a policy against learned reward models of different quality to watch the moment when a higher reward stops meaning a better answer.

## Why this matters at a frontier lab

Stage D starts here because every later post-training result is only as good as its reward. A math verifier that accepts "the last number in the response" can be satisfied by a response that lists several numbers. A preference reward model trained on a few thousand comparisons ranks typical responses well and extreme ones badly, and RL is very good at finding extreme responses. Both failures look like success in the training curve: reward goes up. They show up only against a signal the optimiser never saw — a stricter verifier, held-out human judgement, or (in this lesson, as in Gao et al.) a hidden "gold" reward. Measuring a reward before trusting it is ordinary engineering at a lab; skipping it is how a run spends a week of GPUs teaching a model to game a checker.

## The idea

### Two kinds of reward

A **verifiable reward** is computed by a program from the response and some ground truth: exact-match on a final answer, unit tests passing, a format rule. Tülu 3 names this RLVR and defines it as $v(x, y) = \alpha$ if the response is correct and $0$ otherwise, with $\alpha = 10$ "based on pilot experiments", optimised as $\max \mathbb{E}[v(x,y)] - \beta\,\mathrm{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}]$ (Tülu 3 section 6, Eq. 7–8; PUBLICLY DOCUMENTED). Its training prompts are GSM8K, MATH and instruction-following constraints, all of which a program can check.

A **learned reward** is a model $r_\phi(x, y)$ trained to predict which of two responses a person (or a stronger model) prefers. It is needed wherever no program can check the answer: helpfulness, tone, safety judgements. A learned reward model can also stand in for a verifier: an *outcome reward model* (ORM) scores the final answer, a *process reward model* (PRM) scores each step. Lightman et al. report that process supervision beats outcome supervision on MATH, with a PRM trained on PRM800K (800,000 step-level human labels) solving 78% of a representative MATH test subset (PUBLICLY DOCUMENTED).

Neither kind is "the truth". A verifier implements a *specification*, and the specification can be looser than what you meant. A reward model is a *statistical estimate* of preferences, accurate where its training data was and unreliable elsewhere. The working definition of the danger is Skalse et al.'s: a proxy reward is *unhackable* if increasing the expected proxy return can never decrease the expected true return. Almost no useful proxy is unhackable; the question is how far you can optimise before it bites.

### Reward construction: the verifier is code, so test it

A verifier has a specification ("finished with EOS, and the text is exactly the answer") and an implementation. The course's toy task has three implementations of increasing leniency:

| Verifier | Accepts | Typical source |
|---|---|---|
| `strict` | finished, text equals the formatted answer | the specification |
| `last_number` | the last run of digits equals the answer | "extract the final answer" scripts |
| `lenient` | the answer's digits appear anywhere | a quick substring check |

The two numbers that describe a verifier are its **false-positive rate** (wrong responses it accepts, a reward-hacking opportunity) and its **false-negative rate** (correct responses it rejects, a lost learning signal), both measured against the specification on (a) a probe set of hand-made responses that try the obvious loopholes and (b) the policy's own samples. A loophole that the current policy never produces costs nothing *today*: RL can only exploit what it can sample. That is why the probe set matters. It shows the loophole before the policy finds it.

### Learned rewards: the Bradley-Terry model

Given a prompt $x$ and two responses, the Bradley-Terry model says the probability that $y_c$ (chosen) is preferred to $y_r$ (rejected) is

$$P(y_c \succ y_r \mid x) = \sigma\big(r_\phi(x, y_c) - r_\phi(x, y_r)\big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}},$$

where $r_\phi$ is a scalar score from the reward model with parameters $\phi$. Training maximises the likelihood of the observed preferences, so the loss for one pair is

$$\mathcal{L}(\phi) = -\log \sigma\big(r_\phi(x, y_c) - r_\phi(x, y_r)\big).$$

Only *differences* of scores matter: adding a constant to every score changes nothing, so a reward model's absolute values carry no meaning. InstructGPT's reward-model loss is this one averaged over all $\binom{K}{2}$ pairs among $K = 4$ to $9$ ranked responses per prompt, with all pairs from one prompt kept in a single batch element to avoid overfitting (InstructGPT section 3.5, Eq. 1; PUBLICLY DOCUMENTED). Its RL reward is the reward model's score minus a per-token KL penalty to the SFT model, $r_\theta(x,y) - \beta \log(\pi^{\text{RL}}/\pi^{\text{SFT}})$ (Eq. 2): lesson 12.2 takes that penalty apart.

The reward model in this course is the policy architecture with its language-model head replaced by a linear layer on the final hidden state at the EOS position. Reading the score at EOS means the score has seen the whole response.

### Is the reward model any good? Accuracy and calibration

Two held-out measurements, both on pairs the reward model never trained on:

- **Pair accuracy**: the fraction of pairs ranked the same way as the labels (or, where you have it, the same way as the gold reward). With noisy labels the ceiling is below 1.
- **Calibration**: when the model says $P(y_c \succ y_r) = 0.8$, is it right 80% of the time? The **expected calibration error** bins the predictions and averages the gap:

$$\mathrm{ECE} = \sum_{b} \frac{n_b}{n}\,\big|\bar p_b - \bar y_b\big|,$$

where $n_b$ is the number of pairs in bin $b$, $\bar p_b$ their mean predicted probability and $\bar y_b$ the fraction actually preferred. A reward model that is over-confident (large score gaps for small true differences) has a large ECE, and its large score gaps are exactly what an optimiser chases.

### Over-optimisation: pushing on a proxy

**Best-of-n** (BoN) is the simplest optimiser: sample $n$ responses from the base policy and keep the one the reward model scores highest. Its distribution has an exact KL from the base policy (Gao et al. section 2, after Stiennon et al.):

$$\mathrm{KL}_{\text{bon}}(n) = \log n - \frac{n-1}{n} \quad \text{nats}.$$

So BoN gives an optimisation curve with a known x-axis and no training noise. Gao et al.'s setup, which this lab copies in miniature, uses a *synthetic gold* reward model (6B parameters) to label 100,000 comparisons "deterministically by always marking the trajectory with the higher gold RM score as preferred"; proxy reward models (3M to 3B parameters) are trained on those labels, the policy is optimised against a proxy, and the gold reward is measured on what comes out (section 2.1). They fit, with $d = \sqrt{\mathrm{KL}}$ and rewards measured relative to the initial policy,

$$R_{\text{bon}}(d) = d\,(\alpha_{\text{bon}} - \beta_{\text{bon}}\, d), \qquad R_{\text{RL}}(d) = d\,(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d),$$

(section 1). The proxy reward keeps rising with $d$; the gold reward rises, peaks at $d^\ast = \alpha/(2\beta)$ for BoN, and falls. Larger proxy models and more data push the peak further out. One finding the next lessons use: a KL penalty in RL "does not affect the KL_RL-gold reward frontier" in their setting; it makes the gold score peak earlier, like early stopping, rather than giving a better trade-off (section 3.6; PUBLICLY DOCUMENTED, for their setup).

### The lab's gold world

On a laptop there is no 6B gold model, so the lab builds a world where the gold reward is known exactly. A base sampler writes strings of letters, mostly 3 to 8 long and rarely longer. The hidden gold reward of a string $y$ of length $L$ is

$$\mathrm{gold}(y) = \sum_{c \in \text{distinct}(y)} v_c + 0.4 \min(L, 8) - 0.8 \max(0, L - 8),$$

with a fixed random value $v_c \in [-0.5, 1]$ per letter. A noisy annotator prefers $y_1$ with probability $\sigma((\mathrm{gold}(y_1) - \mathrm{gold}(y_2))/0.5)$. Where the data lives, "longer is better" is nearly true; past 8 letters it is false, and the reward model has almost no data there. BoN with large $n$ goes exactly there.

## Worked example

**Bradley-Terry loss.** Scores $r_c = 1.2$, $r_r = 0.4$: $\sigma(0.8) = 0.690$, loss $= -\ln 0.690 = 0.371$. If the model swaps them ($r_c = 0.4$, $r_r = 1.2$): $\sigma(-0.8) = 0.310$, loss $1.171$. Adding 5 to both scores changes neither number.

**BoN KL.** $n = 4$: $\ln 4 - 3/4 = 1.386 - 0.750 = 0.636$ nats, $d = 0.798$. $n = 4096$: $8.318 - 1.000 = 7.318$ nats. Doubling $n$ adds a little under $\ln 2 = 0.693$ nats.

**Unbiased BoN from one pool.** Drawing fresh samples for every $n$ is wasteful and noisy. Instead, take one pool of $N$ samples, sort it by proxy score, and note that the $i$-th smallest (1-based) is the winner of a uniformly random $n$-subset with probability $\binom{i-1}{n-1}/\binom{N}{n}$ (it must be in the subset, and the other $n-1$ members must be among the $i-1$ smaller ones). Pool of $N = 4$ with proxy scores $(0.1, 0.5, 0.3, 0.9)$ and gold scores $(1, 2, 0, -1)$, $n = 2$. Sorted by proxy: gold $(1, 0, 2, -1)$ with weights $(0, 1, 2, 3)/6$. Expected gold of BoN-2 $= (0 \cdot 1 + 1 \cdot 0 + 2 \cdot 2 + 3 \cdot (-1))/6 = 1/6$. Check by enumeration: the six pairs pick gold values $0, 2, -1, 2, -1, -1$, mean $1/6$. The proxy's favourite (0.9) has the worst gold score, so BoN-2 is already worse on gold than the base average $(1 + 2 + 0 - 1)/4 = 0.5$: a tiny over-optimisation.

**ECE.** Ten pairs predicted at 0.9, of which 9 were right; ten at 0.6, of which 3 were right. Bin gaps $|0.9 - 0.9| = 0$ and $|0.6 - 0.3| = 0.3$; ECE $= 0.5 \cdot 0 + 0.5 \cdot 0.3 = 0.15$.

**Where the gold peaks.** If a fit gives $\alpha = 1.5$, $\beta = 0.4$: $d^\ast = 1.5/0.8 = 1.875$, so KL$^\ast = 3.5$ nats, about $n = e^{3.5 + 1} \approx 90$ (from $\log n \approx \mathrm{KL} + 1$ for large $n$).

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| reward-model input ids | (B, 18): BOS, up to 16 letters, EOS, PAD | int64 | CPU (free path) / GPU |
| EOS index | (B,) | int64 | same |
| final hidden states | (B, 18, 96) | float32 | same |
| hidden at EOS | (B, 96) | float32 | same |
| scores $r_\phi$ | (B,) | float32 | same |
| BoN pool scores (proxy, gold) | (N,) = (20,000,) | float64 (NumPy) | CPU |

The free-path reward model is the 308,400-parameter policy body plus a 97-parameter head. Training it on $P$ pairs for $S$ steps of batch $b$ costs about $6 \cdot 3.1 \times 10^5 \cdot 2 \cdot 18 \cdot b \cdot S$ FLOPs (two sequences per pair): $b = 64$, $S = 400$ gives $1.7 \times 10^{12}$ FLOPs, which took 40–76 s on the build laptop (shared with another job). BoN costs $n$ generations and $n$ reward-model scores *per prompt*, which is why labs use it as an analysis tool and as a baseline, and why Gao et al. use the unbiased estimator over a fixed pool.

Main path (PROJECTED, pending the Module 12 pilot): a Qwen3-0.6B-Base reward model on responses of about 400 tokens. Training 16,000 pairs is $16{,}000 \cdot 2 \cdot 400 \cdot 6 \cdot 0.6 \times 10^9 = 4.6 \times 10^{16}$ FLOPs, about 2 minutes at 40% of an H100's 989 TFLOP/s; scoring 64,000 responses with the 8B gold model is $64{,}000 \cdot 400 \cdot 2 \cdot 8 \times 10^9 = 4.1 \times 10^{17}$ FLOPs, about 17 minutes. Generation dominates (see the lab table).

## Build it

```python
import numpy as np
from frontierlab.posttrain.tasks import LetterWorld
from frontierlab.posttrain import reward as RW

world = LetterWorld()                                  # hidden gold reward, noisy annotator
rm, _ = RW.train_rm(world, n_pairs=1000, steps=400)    # Bradley-Terry on labelled pairs
print(RW.preference_metrics(rm, world))                # acc_vs_labels, acc_vs_gold, nll, ece, reliability

pool = world.sample(20000, np.random.default_rng(5))
proxy = RW.score_strings(rm, pool, world.max_len)
gold = np.array([world.gold(s) for s in pool])
for n in (1, 16, 256, 4096):
    print(n, RW.bon_kl(n), RW.bon_expected(proxy, gold, n) - gold.mean())
```

`frontierlab/posttrain/reward.py` holds the reward model, the loss, calibration, `bon_kl`, `bon_expected` and `fit_gao`; `frontierlab/posttrain/tasks.py` holds the verifiers and the letter world. Correctness checks in `labs/common/tests/test_posttrain.py`: `bon_expected` equals brute-force enumeration over every $n$-subset of a 9-element pool to $10^{-10}$; the reward model's score does not change when tokens after EOS change; the verifiers give the expected verdicts on a probe table; `fit_gao` recovers $\alpha$ and $\beta$ from noise-free data.

### Choosing the Stage D base model

Stage D needs one pinned open base model of 1–2B parameters (plan section 5). The course pins **Qwen3-1.7B-Base**, revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (checked on the Hub 2026-10-06), Apache-2.0, 1,720,574,976 parameters in BF16 (the safetensors header; the model card gives 1.4B non-embedding). Reasons:

- **Same layout as Baseline-0.** Its `config.json` is a dense `Qwen3ForCausalLM`: 28 layers, 16 query and 8 key/value heads of dimension 128, SwiGLU 6,144, QK-norm, tied embeddings. Everything the course built for Baseline-0 (parameter and FLOP accounting, the correctness suite's ideas) carries over, and rollouts need no special kernels.
- **A real base model.** Qwen3 publishes base checkpoints separately (the report describes 36T pretraining tokens in 119 languages; Qwen3 technical report), so Stage D can run SFT, preference tuning and RL itself instead of inheriting someone's post-training.
- **Comparability.** Qwen base models are the most common starting point in published RLVR work, so results can be set next to published curves.

Considered and not chosen: **Qwen3.5-2B-Base** (Apache-2.0, released 2026) is a hybrid of gated-delta-net and attention layers with a vision encoder (`Qwen3_5ForConditionalGeneration`, `layer_types` alternating three linear-attention layers per full-attention layer): rollout engines and training kernels for it add a train/inference mismatch the course does not want in its first RL module. **OLMo-2-0425-1B** (Apache-2.0, fully open data) is the better choice for contamination studies and is the course's documented alternative; it is weaker at GSM8K-style math at this size.

> [!WARNING]
> Pinning a Qwen base has a known risk. Shao et al. report that RLVR on Qwen2.5-Math-7B improved MATH-500 by 21.4 points with *random* rewards and 24.1 with *incorrect* rewards, against 29.1 with ground-truth rewards, and that such spurious rewards "generally fail to yield any gains" on Llama 3 and OLMo 2 (sections 2.2 and 3; PUBLICLY DOCUMENTED, for Qwen2.5-Math). Whether Qwen3-1.7B-Base behaves the same is not established (INFERENCE: plausible, untested here). Every Stage D RL result in this course therefore includes a control run that a reviewer would ask for: a random-reward or format-only arm, or a repeat on OLMo-2-0425-1B.

## What the evidence says

- **Verifiable rewards for math and code: ESTABLISHED.** Tülu 3 (section 6), DeepSeekMath and most open reasoning recipes use them. The failure mode is the verifier's specification, which is why it gets unit tests like any other code (REASONABLE INDUSTRY PRACTICE).
- **Bradley-Terry reward models from pairwise preferences: ESTABLISHED** (InstructGPT section 3.5 and nearly every RLHF report since).
- **Reward-model over-optimisation: ESTABLISHED as a phenomenon.** Gao et al. measure it systematically; Singhal et al. show that much of the reward gain of RLHF in their settings is explained by length, and that "a purely length-based reward reproduces most downstream RLHF improvements" — a reward model with a length bias is one that an optimiser over-optimises along the cheapest direction. **The functional forms are PROMISING**: one study, one gold model family.
- **Reward-model benchmarks predict downstream use only partly.** RewardBench 2 reports models scoring about 20 points lower than on the first RewardBench and scores that correlate with best-of-n and PPO results (PUBLICLY DOCUMENTED). A high benchmark score is not the same as being safe to optimise against.
- **Process reward models: PROMISING** (Lightman et al.'s 78% is one study with expensive human step labels).
- **Spurious-reward gains on Qwen math models: MODEL-SPECIFIC** (Shao et al.), and the reason for the control arm above.
- **Course measurement (free CPU, 2026-10-07, build laptop with another module's jobs running; torch 2.14.1 CPU).** Reward models on 250, 1,000 and 4,000 pairs (2 seeds each) reached gold-order pair accuracy 0.76–0.79, 0.82–0.83 and 0.90–0.93, and ECE 0.20–0.22, 0.11–0.14 and 0.02. Best-of-n gold reward (relative to the base mean; gold-oracle BoN in brackets):

| n | KL (nats) | 250 pairs | 1,000 pairs | 4,000 pairs | [oracle] |
|---|---|---|---|---|---|
| 16 | 1.84 | 1.08, 1.34 | 1.43, 1.50 | 1.75, 1.73 | [1.85] |
| 256 | 4.55 | 1.25, 1.82 | 1.92, 1.90 | 2.44, 2.50 | [2.79] |
| 512 | 5.24 | **1.27, 1.89** | 1.98, 1.93 | 2.53, 2.61 | [2.96] |
| 4096 | 7.32 | 1.20, 1.60 | 2.38, 1.97 | 2.89, 3.00 | [3.43] |

  With 250 pairs, both seeds peak at $n = 512$ and fall after it (by 0.07 and 0.29 at $n = 4096$): the over-optimisation curve. With 1,000 pairs one seed flattens and one keeps rising; with 4,000 pairs gold reward still rises at $n = 4096$ and stays within 0.5 of the oracle. More data moved the peak beyond the range BoN could reach here, as Gao et al. report for larger proxies. This is a designed toy world, so it shows the mechanism, not the size of the effect in an LLM.
- Part A of the lab (verifiers) is reported in the lab's results box below.

## Lab

**Folder:** [`labs/module-12/lesson-01/`](../../labs/module-12/) · **Time:** about 2 hours (about 15 minutes of it unattended) · **Pass check:** `pytest labs/module-12/lesson-01` passes; `reward_lab.py` prints both parts; your write-up states the verifier you would ship, the over-optimisation point of each reward model with its interval across seeds, and the decision below.

### Experiment contract

- **Question:** how far can best-of-n optimise a learned reward model before the hidden gold reward stops improving, and how does that point move with the reward model's training data (250, 1,000, 4,000 pairs)? Decision informed: how much KL budget to allow when Module 13 optimises against a learned reward of a given quality, and whether to spend on more preference data instead.
- **Hypothesis:** gold reward peaks and declines for the smallest reward model; the peak moves to larger $n$ (or out of range) as data grows; the proxy reward rises monotonically for every model. Status: reported effect (Gao et al. sections 1 and 3), designed into the toy world by the gold function's long-string penalty; may not appear for the larger reward models within $n \le 4096$.
- **Baseline:** the gold-oracle BoN curve (selection by the gold reward itself), which is the best any reward model could do with the same samples.
- **Changed variable:** reward-model training pairs (250 / 1,000 / 4,000). **Controlled:** the base sampler and its 20,000-sample pool (seed 5), the annotator temperature 0.5, the reward-model architecture, 400 steps of batch 64, learning rate $2 \times 10^{-3}$, seeds 0–1, the held-out pairs (seed 99).
- **Comparison axis:** equal KL from the base policy (each $n$ has an exact KL). It does not answer the cost question: a bigger $n$ costs more samples, a better reward model costs more labels.
- **Budget:** free CPU, about 9 minutes for Part B (6 reward models of about a minute, BoN estimates in seconds) plus about 4 minutes for Part A's two RL runs.
- **Metrics and decision rule:** primary: gold reward of BoN at each $n$, its peak $n$ and the drop from the peak to $n = 4096$ (TODO 6, tolerance 0.05) per seed; secondary: pair accuracy against gold, ECE, the gap to the oracle, mean BoN length. Rule, stated now: a reward model is *safe to optimise to KL k* if both seeds' gold reward is still within 0.05 of its running maximum at that KL. Report the largest safe KL per data size.
- **Correctness checks:** `test_posttrain.py` (the estimator against enumeration, the EOS read-out, the verifier probes); your TODO tests; the oracle curve rises monotonically (a check on the estimator and the pool).
- **Fallback evidence:** if no reward model over-optimises, report the null with the gap to the oracle and use Gao et al.'s published curves (labelled as published results).
- **Limits:** a designed gold reward and a known base sampler, not an LLM; BoN, not RL (Gao et al. find RL over-optimises along a different curve); 2 seeds; reward models of one size.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 12 pilot | `python labs/module-12/lesson-01/rm_main.py --out runs/m12/l121-main`: Qwen3-1.7B-Base answers 2,000 + 500 UltraFeedback prompts (16 and 64 samples), `Skywork-Reward-V2-Qwen3-8B` is the gold reward, Qwen3-0.6B-Base proxies on 1,000 / 4,000 / 16,000 pairs × 2 seeds, BoN to $n = 64$. **PROJECTED:** generation $1.6 \times 10^7$ tokens at about 2–3k tokens/s with `generate` (1.5–2.2 h; vLLM 0.30.0 is several times faster), gold scoring $4.1 \times 10^{17}$ FLOPs (about 17 min at 40% MFU), proxies minutes: 2–3 GPU-hours, USD 5–10 |
| Free GPU (Colab/Kaggle T4) | T4 16 GB | `rm_main.py --gold 1.7b --train-prompts 500 --pool-prompts 100 --pairs 1000,4000 --out runs/m12/l121-t4` (the 8B gold model does not fit next to the others; a 1.7B gold model is a weaker stand-in for a human) |
| Free CPU | laptop, measured 537 s (9 minutes) for the whole script with `--rl-lenient`, other jobs running | the steps below |

### Steps

1. **Implement** the six TODOs in `lab.py` (Bradley-Terry loss, BoN KL, the unbiased BoN estimator, ECE, verifier error rates, the peak) and run `pytest labs/module-12/lesson-01`.
2. **Audit the verifiers** and **over-optimise**: `python labs/module-12/lesson-01/reward_lab.py --rl-lenient`. It trains the SFT warm start first if `runs/m12/sft` does not exist (about a minute).
3. **Read Part A.** Which probes does each verifier accept? How often does the policy produce them? Did RL against `lenient` find its loophole in 200 steps, and why (look at `len` in `runs/m12/l121/rl-lenient/metrics.jsonl`)?
4. **Read Part B.** For each data size and seed: the peak $n$, the drop after it, the largest safe KL under the rule, the ECE. Plot gold and proxy against $d = \sqrt{\mathrm{KL}}$ from `runs/m12/l121/results.json` and fit $d(\alpha - \beta d)$ with `frontierlab.posttrain.reward.fit_gao`.
5. **Write up** (one page): the verifier you would use for RL and the probe that convinced you; the KL budget you would allow for each reward model and what you would buy instead (more pairs or a larger reward model) if you needed more.

<details>
<summary>Hint for TODO 3</summary>

`np.argsort(proxy)` gives the ascending order. With `i = np.arange(1, N + 1)`, the log-weight is `gammaln(i) - gammaln(n) - gammaln(i - n + 1) - logC(N, n)` for `i >= n` and `-inf` otherwise; `np.exp` of `-inf` is 0. Test it against `itertools.combinations` on a pool of 8.

</details>

<details>
<summary>Hint for TODO 4</summary>

`np.linspace(0, 1, 11)` gives the edges. Use `p >= lo` and `p < hi` for every bin except the last, which uses `p <= hi` so that a prediction of exactly 1.0 is counted.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Part B is in the table in "What the evidence says". Under the rule, the largest KL at which *both* 250-pair seeds were safe was 5.93 nats ($n = 1024$; seed 1 fell more than 0.05 below its peak at $n = 2048$, seed 0 only at $n = 4096$); for 1,000 pairs, seed 0 never left its running maximum and seed 1 stayed within 0.05 of it throughout (it flattened after $n = 256$, so "safe" there means "no further gain", not "still improving"); for 4,000 pairs every $n$ was safe. ECE fell from about 0.21 to 0.02 with data, and the over-confident reward models were the ones that over-optimised.

Part A (probe set, 20 problems × 10 probes; truth = the strict specification): `strict` 0 false positives, `last_number` 0.44, `lenient` 0.78, all with 0 false negatives. `last_number` accepts the unfinished answer (no EOS), the zero-padded answer, and the answer followed by `!` or wrapped in `#` (4 of the 9 wrong probes); `lenient` additionally accepts the answer repeated, embedded in a longer number, or written after a wrong one (7 of 9). On the policy's own samples (100 problems × 64 samples; the SFT start passes 19.4% strict at temperature 1) the false-positive rates were tiny: `last_number` 0.06% and `lenient` 0.10% of wrong responses at temperature 1, 0.16% and 0.41% at temperature 2. The SFT policy almost always writes a short number and stops, so the loopholes exist but the policy does not visit them. The 200-step RL runs at temperature 1.5 did not find the lenient loophole either: the run trained against `lenient` ended at 0.31 strict and 0.37 lenient held-out accuracy with mean training length 3.6 tokens, the run against `strict` at 0.37 and 0.41 with 3.5 tokens. Here the hackable verifier cost a little accuracy (0.06, one seed, so within noise) rather than producing a hack; the probe-set numbers are what tell you it *could*. The RL numbers depend on the SFT start; the probe-set numbers do not.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-12/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-12/lesson-01`.

</details>

## Common mistakes

- **Trusting a verifier because training reward went up.** Reward going up is what RL does to *any* verifier; only the probe set and a stricter check tell you whether the gain is real.
- **Calling a loophole harmless because the policy never uses it.** It is harmless until a better policy, a higher temperature or a longer run samples it.
- **Reading meaning into absolute reward-model scores.** Bradley-Terry fixes only differences; normalise per prompt or compare within a prompt.
- **Judging a reward model by pair accuracy alone.** Accuracy says nothing about the tails, which is where optimisation goes; check calibration and run a BoN curve against a held-out signal.
- **Comparing optimisers at equal steps instead of equal KL.** Gao et al.'s curves are against KL; two runs at the same step can be at very different distances from the base policy.
- **Running fresh samples for each n.** It is slower and noisier than the unbiased estimator over one pool.

## References

- L. Gao, J. Schulman, J. Hilton, *Scaling Laws for Reward Model Overoptimization*, 2022, sections 1, 2, 2.1, 3.6. https://arxiv.org/abs/2210.10760
- L. Ouyang et al., *Training language models to follow instructions with human feedback*, 2022, section 3.5. https://arxiv.org/abs/2203.02155
- N. Lambert et al., *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, 2024, section 6. https://arxiv.org/abs/2411.15124
- H. Lightman et al., *Let's Verify Step by Step*, 2023. https://arxiv.org/abs/2305.20050
- R. Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, 2025, sections 2.2 and 3. https://arxiv.org/abs/2506.10947
- P. Singhal et al., *A Long Way to Go: Investigating Length Correlations in RLHF*, 2023. https://arxiv.org/abs/2310.03716
- J. Skalse et al., *Defining and Characterizing Reward Hacking*, 2022. https://arxiv.org/abs/2209.13085
- S. Malik et al., *RewardBench 2: Advancing Reward Model Evaluation*, 2025. https://arxiv.org/abs/2506.01937
- R. A. Bradley, M. E. Terry, *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*, Biometrika 39(3/4), 1952.
- Qwen Team, *Qwen3 Technical Report*, 2025. https://arxiv.org/abs/2505.09388
- Qwen/Qwen3-1.7B-Base, revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. https://huggingface.co/Qwen/Qwen3-1.7B-Base
- Main-path models and data (pinned in `rm_main.py`): Skywork/Skywork-Reward-V2-Qwen3-8B `6f19fdef`, Qwen/Qwen3-0.6B-Base `da87bfb6`, HuggingFaceH4/ultrafeedback_binarized `3949bf5f` (MIT).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[12.2 · Policy-gradient estimators for LLMs](lesson-02.md)
