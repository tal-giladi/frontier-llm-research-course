---
id: "19.1"
module: 19
minutes: 35
practice_minutes: 90
prerequisites: ["01.3", "01.4", "11.3"]
objectives:
  - Classify a research plan as goal-driven or idea-driven and state the decision its answer would change.
  - Estimate the probability that a planned comparison gives a decisive answer from its seed noise, seed count and the chance that a small-scale answer transfers, and compute it by hand.
  - Score three proposals by value × probability of a decisive answer ÷ cost, and measure how robust the ranking is when every estimate is off by a factor of 2 or 3.
  - Run a two-size proxy ladder on CPU and read it as below noise, sign flips, grows, shrinks or flat before trusting it as a cheap proxy.
volatility: concept
sources:
  - title: "John Schulman, An Opinionated Guide to ML Research (posted 2020-01-24; sections Choosing Problems, Making Continual Progress)"
    url: http://joschu.net/blog/opinionated-guide-ml-research.html
  - title: "Chris Olah, Research Taste Exercises (rough note, 2021-01-09)"
    url: https://colah.github.io/notes/taste/
  - title: "Andrej Karpathy, A Recipe for Training Neural Networks (2019-04-25)"
    url: https://karpathy.github.io/2019/04/25/recipe/
  - title: "Godbole, Dahl, Gilmer, Shallue, Nado, Deep Learning Tuning Playbook (sections: incremental tuning strategy; exploration vs exploitation; scientific, nuisance and fixed hyperparameters)"
    url: https://github.com/google-research/tuning_playbook
  - title: "Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities (abstract; Figure 1; section 3.1.1)"
    url: https://arxiv.org/abs/2309.14322
last_verified: "2026-10-07"
---

# 19.1 · Research taste and problem choice

Most research time is lost before the first run, by choosing a question whose answer would change nothing, could not be detected at the planned scale, or costs more than it is worth. This lesson turns problem choice into something you can write down and check: whether the work is goal-driven or idea-driven, what decision the answer changes, how likely the planned experiment is to give a decisive answer, what it costs, and what a cheap proxy can and cannot tell you. The lab scores three capstone proposals, stress-tests the ranking, and runs one real proxy ladder on CPU.

## Why this matters at a frontier lab

A frontier lab has far more ideas than GPU-hours. Every proposal competes for the same cluster, and the people who choose well are the ones whose experiments end in a decision: adopt, reject, or "not at this scale, here is why". Taste is the name for doing this well, and both Schulman and Olah describe it as the skill that matters most and is hardest to train, because the feedback on a choice arrives months later. You cannot speed up the months. You can make your choices explicit, so that each one is checked against the evidence when it arrives, and you can buy faster, noisier feedback with cheap proxies, as long as you check the proxy before you trust it.

## The idea

### Goal-driven and idea-driven work

Schulman's guide separates two ways of choosing what to do next. **Idea-driven**: read a paper, have an idea to do X better, test it. **Goal-driven**: hold a concrete capability you want to reach and solve the problems in its way, trying existing methods first. He recommends goal-driven research for most people: idea-driven work risks being scooped because everyone reads the same papers, while a goal gives a differentiated perspective and lets a team split the problem. He also warns that incremental work must be cheap to adopt: a method that gives a small gain "better be very simple", or nobody will use it.

In this course the goal is fixed by the artefact chain: a chosen architecture, Recipe-R, a post-training pipeline, an audited model. A good capstone question is goal-driven in that sense: its answer changes a choice in that chain (or in your team's equivalent). "Is GSPO better than GRPO?" is idea-driven. "Should our reasoning-RL recipe switch from GRPO to GSPO, given that our 1.7B runs collapse about once in four seeds?" is goal-driven: it names the decision and the observation that makes it worth asking.

### What a proposal must state

A proposal is worth running when you can write five things down before any GPU time:

1. **The question**, one sentence ending in a question mark.
2. **The decision** it informs: what you will do differently for each answer. If both answers lead to the same action, stop here.
3. **The value** of the answer, 1–10: how much that decision matters. A score, not a measurement; write why.
4. **The probability of a decisive answer**, $p_{\text{decisive}}$, from the design (below).
5. **The cost** in main-path GPU-hours, every arm, seed, tuning run and evaluation included, labelled PROJECTED with its formula.

Plus two that protect you later: the **cheap proxy** you will run first, and the **kill criterion**, the result that makes you stop. Olah's first exercise (write ideas down, have a mentor rate each 1–10, discuss the disagreements) and Preetum Nakkiran's suggestion he quotes (write a short project proposal before committing a month: why the question is interesting, why the answer is valuable, what you expect to find) are both this list in another form.

### The score

$$\text{score} = \frac{v \cdot p_{\text{decisive}}}{c}$$

with $v$ the value, $c$ the cost in GPU-hours. It is expected decision value per GPU-hour. The interesting term is $p_{\text{decisive}}$, which most proposals silently set to 1. Two things make an experiment indecisive:

- **It cannot see the effect.** With seed standard deviation $\sigma$ of the metric, $n$ seeds per arm and a true difference $\delta$, a two-arm comparison at $\alpha = 0.05$ detects it with probability (normal approximation, lesson 01.4)
  $$\text{power}(\delta) = \Phi\!\left(\frac{|\delta|}{\sigma\sqrt{2/n}} - 1.96\right) + \Phi\!\left(-\frac{|\delta|}{\sigma\sqrt{2/n}} - 1.96\right).$$
  Choose $\delta$ as the *smallest* difference that would change the decision. If the design cannot see that, a null result is not an answer.
- **The answer does not transfer.** A comparison at 30M parameters answers the question at 30M. If the decision is about a 1B run, multiply by $p_{\text{transfer}}$, your probability that the small-scale answer is the large-scale answer. This is where cheap proxies and published scale ladders enter.

So $p_{\text{decisive}} = \text{power}(\delta) \cdot p_{\text{transfer}}$. It is crude; its use is that it makes the two most common failures (underpowered and non-transferable) visible on the page where the proposal is ranked.

### Estimates are guesses: test the ranking

Value, transfer probability and cost are all guesses. A ranking you would reverse after one conversation is not a decision. Perturb every estimate independently by a random factor between $1/f$ and $f$ (log-uniform) and count how often the top proposal stays on top. Near 100%: the choice does not hinge on the guesses. Near $1/k$ for $k$ proposals: it is a coin toss, and the cheapest way forward is to buy information (run the proxies) rather than think harder.

### Cheap proxies and when they mislead

A cheap proxy is a small, fast experiment whose result you expect to predict the expensive one. Three examples from the sources:

- **A small-scale ladder.** Wortsman et al.'s thesis is that training instabilities reported at large scale "also appear in small models when training at high learning rates", and that the same mitigations work there (abstract, section 3.1). They also find that "LR sensitivity still increases with model scale" (Figure 1 caption): the proxy shows the mechanism, but its *size* changes along the ladder. Lesson 19.2 reproduces their headline claim on CPU.
- **A dumb baseline and an overfit check.** Karpathy's recipe is a sequence of cheap proxies for "my training code is right": verify the loss at initialisation, train a model you could not have broken, overfit one batch, before any expensive run ("neural net training fails silently").
- **An exploration round.** The tuning playbook spends most of its rounds on insight rather than on the validation number ("most of the time, our primary goal is to gain insight into the problem"), separating the *scientific* hyperparameter under study from *nuisance* hyperparameters that must be re-tuned per setting and *fixed* ones. A proxy that fixes a nuisance hyperparameter (the learning rate, typically) can rank methods by which one happens to like that value.

Proxies mislead in recognisable ways. Read the same comparison at two or three sizes (plan section 12.1) and classify it:

| Reading | Rule (noise $\sigma_d$ = seed std of one difference, $k = 2$) | What it means |
|---|---|---|
| below noise | every $\lvert e\rvert < k\sigma_d$ | the proxy says nothing; more seeds or a larger effect, not a conclusion |
| sign flips | effects above noise point both ways | the small scale is not a proxy for this question |
| grows / shrinks | $\lvert e_{\text{largest}}\rvert - \lvert e_{\text{smallest}}\rvert$ beyond $\pm k\sigma_d$ | a trend to extrapolate cautiously, with the ladder's sizes stated |
| flat | otherwise | the effect is stable over this range; the range is the limit |

The course has already met each failure. In lesson 07.2 the published ordering "Muon grows logits more than AdamW" was reversed at toy scale (one seed); in lesson 03.3 removing QK-norm *lowered* the loss at the default learning rate and raised it by 0.30 nats at a 3.3× higher one: the proxy's answer depends on a nuisance hyperparameter. In lesson 11.3 the learning-rate sensitivity doubled from one ladder rung to the next.

## Worked example

Three proposals (the reference solution's; the noise floors are assumptions for illustration):

| | value $v$ | smallest effect $\delta$ | seed std $\sigma$ | seeds $n$ | $p_{\text{transfer}}$ | cost $c$ (GPU-h) |
|---|---|---|---|---|---|---|
| micro-anneal ranking | 7 | 0.05 | 0.02 | 2 | 0.6 | 6 |
| QK-Clip vs QK-norm | 6 | 0.03 | 0.02 | 3 | 0.5 | 4 |
| GSPO vs GRPO collapses | 8 | 0.15 | 0.10 | 3 | 0.4 | 40 |

QK-Clip vs QK-norm, by hand. Standard error of the difference $\sigma\sqrt{2/n} = 0.02 \cdot \sqrt{2/3} = 0.0163$. $z = 0.03 / 0.0163 = 1.84$. Power $= \Phi(1.84 - 1.96) + \Phi(-1.84 - 1.96) = \Phi(-0.12) + \Phi(-3.80) = 0.451 + 0.0001 = 0.451$. $p_{\text{decisive}} = 0.451 \cdot 0.5 = 0.226$. Score $= 6 \cdot 0.226 / 4 = 0.338$.

Micro-anneal: $\sigma\sqrt{2/2} = 0.02$, $z = 2.5$, power $= \Phi(0.54) = 0.705$, $p_{\text{decisive}} = 0.423$, score $= 7 \cdot 0.423 / 6 = 0.494$. GSPO: $z = 0.15/0.0816 = 1.84$, power 0.451, $p_{\text{decisive}} = 0.180$, score $= 8 \cdot 0.180 / 40 = 0.036$.

Ranking: micro-anneal (0.49), QK-Clip (0.34), GSPO (0.04). Two lessons in these numbers. First, all three designs are underpowered at their stated effect: the minimum detectable effect $\text{MDE} = 2.80\,\sigma\sqrt{2/n}$ is 0.056, 0.046 and 0.229 against effects of 0.05, 0.03 and 0.15. A power of 0.45 means a coin flip between an answer and "inconclusive"; the honest fix is more seeds or a larger effect before ranking, and the lab prints this. Second, robustness: perturbing every estimate by up to 2× keeps micro-anneal on top in 65% of draws, by up to 3× in 59%. The first two proposals are close enough that the guesses decide; running both proxies (a few GPU-hours) is worth more than refining the numbers.

A proxy reading. Effects of removing a stabiliser at three sizes: $+0.03, +0.06, +0.10$ nats, seed noise of one difference $\sigma_d = 0.01$. All are above $2\sigma_d = 0.02$, all positive, and $0.10 - 0.03 = 0.07 > 0.02$: **grows**. With $+0.05, +0.01, -0.06$: the first and last are above noise with opposite signs: **sign flips**, and the small sizes are not a proxy.

## Shapes and cost

| Item | Size | Cost |
|---|---|---|
| a proposal | 9 numbers and 4 sentences | minutes to write; the cost is in being honest about $\sigma$ and $p_{\text{transfer}}$ |
| `rank_robustness` | $n = 4000$ draws × 3 factors × $k$ proposals (float64 NumPy) | milliseconds |
| proxy ladder, free CPU | 8 runs: 2 widths (toy preset at hidden size 64 and 128) × QK-norm on/off × 2 seeds, 200 steps of 16 × 128 tokens | measured 7.8 minutes |
| proxy ladder, main path | the same at `pilot-30m` widths 256 and 512, 2,000 steps of 32 × 1,024 tokens, bf16 | PROJECTED 0.13 H100-hours: $1.14 \times 10^{17}$ FLOPs (`flops_per_token` × tokens, summed over runs) ÷ ($989 \times 10^{12}$ × 0.25 assumed MFU) |

## Build it

`frontierlab.research.proposals` holds the pieces:

```python
from frontierlab.research.proposals import CAPSTONE_CLAIMS, Proposal, power, rank, rank_robustness, proxy_trend

p = Proposal("qkclip-vs-qknorm", "Does QK-Clip match QK-norm within 0.02 nats at pilot-30m?",
             "keep QK-norm or allow MLA with QK-Clip in Recipe-R", value=6,
             p_decisive=power(0.03, 0.02, 3) * 0.5, cost_gpu_h=4.0,
             proxy="07.2 arms at toy size, 2 widths", kill="the clip never binds at pilot-30m")
print(round(p.p_decisive, 3))                     # 0.226
print(proxy_trend([1, 2, 3], [0.03, 0.06, 0.10], noise=0.01))   # grows
```

`CAPSTONE_CLAIMS` is the Module 20 claim list (GSPO vs GRPO stability, QK-Clip vs QK-norm, DSA quality and cost vs dense, mHC vs HC stability, micro-anneal data scoring, on-policy vs off-policy distillation at equal compute), each with its source, the course lessons that built it, the PROJECTED main-path cost of the course lab (one arm set, one seed), a suggested proxy, and the proxy's known risk. `check_proposals` refuses a set that is not three distinct proposals, uses no capstone claim, has a question that is not a question, or leaves the decision, proxy or kill criterion empty. `power` is the inverse of `frontierlab.stats.min_detectable_effect` (at the MDE it returns 0.80). Tests: `pytest labs/common/tests/test_research.py -k "score or power or proxy or proposals"`.

## What the evidence says

- **Problem choice as the main skill: REASONABLE INDUSTRY PRACTICE**, argued from experience by Schulman ("Your ability to choose the right problems to work on is even more important than your raw technical skill") and Olah. These are essays, not studies; no controlled evidence compares goal-driven with idea-driven research.
- **Small-scale proxies for training instabilities: PROMISING** (Wortsman et al.: attention-logit growth and output-logit divergence reproduced in small models at high learning rates, and the mitigations work there; whether every large-scale instability has a small-scale proxy is open). Lesson 19.2 tests the headline claim.
- **Proxies that fix a nuisance hyperparameter can rank methods wrongly: ESTABLISHED** as a tuning principle (the tuning playbook; Dodge et al., lesson 19.3) and seen twice in this course (lessons 03.3 and 07.2).
- **The score formula: the course's formalisation**, REASONABLE INDUSTRY PRACTICE in spirit (expected value per cost), not a published method. Its numbers are only as good as the estimates; the robustness check is there to say how much that matters.

## Lab

**Folder:** [`labs/module-19/lesson-01/`](../../labs/module-19/) · **Time:** about 90 minutes (about 8 of them unattended on CPU) · **Pass check:** `pytest labs/module-19/lesson-01` passes (it checks your three proposals are complete and listed best first); `choose_lab.py` prints your ranking, its robustness and the proxy reading.

### Experiment contract

- **Question:** for the capstone claim "QK-Clip vs QK-norm", is the cost of removing QK-norm at a raised learning rate (1e-2) growing, shrinking or flat between hidden width 64 and 128, relative to the seed noise? Decision informed: the $p_{\text{transfer}}$ you write for that proposal, and whether its proxy is worth running at `pilot-30m`.
- **Hypothesis:** the cost grows with width (Wortsman et al.: sensitivity increases with scale). **Status:** reported effect at much larger scale; may not appear between two toy widths.
- **Baseline:** QK-norm on (Baseline-0's default) at each width.
- **Changed variable:** QK-norm off. **Controlled:** toy preset widened with `--width` (head dimension 32 fixed, heads and SwiGLU scaled), AdamW (the loop's own), learning rate 1e-2 with 20 warm-up steps and cosine, 200 steps × 16 × 128 tokens, seeds 0 and 1 (initialisation and data order), the same 128 validation windows.
- **Comparison axis:** equal tokens. It says nothing about wall-clock cost (QK-norm adds two small norms per layer).
- **Budget:** free CPU, measured 7.8 minutes; main path PROJECTED 0.13 H100-hours (formula in Shapes and cost).
- **Metrics and decision rule:** effect per width = mean over seeds of (held-out loss without QK-norm − with), noise = pooled seed std of the paired differences; `proxy_trend` with $k = 2$. Rule: write $p_{\text{transfer}} \ge 0.5$ for the QK-Clip proposal only if the reading is "grows" or "flat" above noise.
- **Correctness checks:** `pytest labs/module-19/lesson-01`; `pytest labs/common/tests/test_research.py`; both arms of a width share the run card except the QK-norm switch (`python -m frontierlab.record runs/m19/l191/cpu/qknorm-w64-s0 runs/m19/l191/cpu/noqk-w64-s0 --changed config.qk_norm optim.qk_norm`).
- **Fallback evidence:** lesson 03.3's measured single-seed result at width 128 (+0.30 nats at 1e-2) and Wortsman et al.'s Figure 1.
- **Limits:** two widths both under 1M non-embedding parameters; two seeds give a crude noise estimate; one learning rate (a nuisance hyperparameter fixed for both arms); 0.4M tokens per run.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 or A100. Not run in this build; part of the Module 19 pilot | `python labs/module-19/lesson-01/choose_lab.py --variant main --print` prints the 8 runs at `pilot-30m` widths 256 and 512; run them, then `--variant main --part proxy`. PROJECTED 0.13 H100-hours |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | the CPU commands with `--device cuda` added by hand; at this size the T4 is only a few times faster than a laptop |
| Free CPU | laptop; measured 7.8 minutes for the proxy | the steps below |

### Steps

1. **Write before you compute.** For each of your three proposals write, on paper: goal-driven or idea-driven, and the decision it changes. Strike any proposal whose two answers lead to the same action.
2. **Implement** the three functions in `lab.py` (`score`, `p_decisive`, `proxy_trend`) and fill `PROPOSALS`: at least one claim from the capstone list, every field. For $\sigma$ use a noise floor you measured in an earlier lab, or label it an assumption in a comment.
3. **Rank:** `python labs/module-19/lesson-01/choose_lab.py --part rank`. Order `PROPOSALS` best first and rerun `pytest labs/module-19/lesson-01`. Read the MDE column: any design marked underpowered gets more seeds or a larger effect, or you accept the lower $p_{\text{decisive}}$ and say so.
4. **Proxy:** `python labs/module-19/lesson-01/choose_lab.py --part proxy` (about 8 minutes). Apply the contract's rule and update $p_{\text{transfer}}$ for the QK-Clip proposal if you have one.
5. **Write up** (half a page): your ranking, its robustness at 2× and 3×, which estimate the ranking is most sensitive to, the proxy reading with its noise, and what you would run next with 5 GPU-hours.

<details>
<summary>Hint for TODO 3</summary>

Sort the effects by size with `np.argsort(sizes)` first. "Below noise" and "sign flips" use only the effects whose magnitude is at least `k * noise`; "grows" and "shrinks" compare the magnitudes at the smallest and largest size.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (Windows 11, Python 3.12.13, torch 2.14.1+cpu, 16 threads, another module's jobs sharing the CPU): `--part rank` in under a second; `--part proxy` 7.8 minutes for the 8 runs.

**Ranking** (the reference proposals of the worked example): micro-anneal 0.494, QK-Clip vs QK-norm 0.338, GSPO vs GRPO 0.036 per GPU-hour; all three flagged underpowered at their stated effect; the top proposal stays on top in 65% of draws at 2× and 59% at 3×.

**Proxy ladder** (held-out loss after 200 steps at lr 1e-2, QK-norm on / off):

| Width | Parameters | Seed 0 | Seed 1 | Cost of removing QK-norm |
|---|---|---|---|---|
| 64 | 721,728 | 6.719 / 6.747 | 6.893 / 6.969 | +0.052 |
| 128 | 1,836,416 | 6.771 / 7.013 | 6.854 / 6.978 | +0.182 |

Pooled seed noise of one difference: 0.064 (2 degrees of freedom). Reading: **grows**, because $0.182 - 0.052 = 0.130$ exceeds $2 \times 0.064 = 0.128$, by 0.002. That is the honest headline: the rule fires, barely, on a noise estimate from four differences, and the effect at width 64 alone is below the noise. Under the contract's rule you may write $p_{\text{transfer}} \ge 0.5$ for the QK-Clip proposal; a careful author would also note that one more seed per width could flip the reading to "flat" or "below noise", and that both widths are far below the decision's scale. The direction agrees with Wortsman et al.'s Figure 1 (sensitivity grows with scale); it is a two-point toy ladder, not evidence about their sizes.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-19/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-19/lesson-01`.

</details>

## Common mistakes

- **No decision.** "It would be interesting to know" is not a decision; if both outcomes leave the recipe unchanged, the experiment has no value however clean it is.
- **Setting $p_{\text{decisive}}$ to 1.** Compute the power at the smallest effect that matters. Most first drafts are underpowered by a factor of two in seeds.
- **Ranking without a robustness check.** Two proposals within a factor of 2 in score are tied until a proxy separates them.
- **Trusting a proxy at one size.** One point cannot show a trend or a sign flip. Run at least two sizes, ideally three, and compare the change with the noise.
- **Fixing a nuisance hyperparameter in the proxy.** Removing QK-norm helps at one learning rate and hurts at another (lesson 03.3). Sweep it, or state it as a limit.
- **Switching problems too often.** Schulman notes that abandoning promising ideas early is the more common failure; the kill criterion, written in advance, is what lets you stop for a reason rather than a mood.

## References

- J. Schulman, *An Opinionated Guide to ML Research*, posted 2020-01-24. http://joschu.net/blog/opinionated-guide-ml-research.html
- C. Olah, *Research Taste Exercises*, 2021-01-09. https://colah.github.io/notes/taste/
- A. Karpathy, *A Recipe for Training Neural Networks*, 2019-04-25. https://karpathy.github.io/2019/04/25/recipe/
- V. Godbole, G. E. Dahl, J. Gilmer, C. J. Shallue, Z. Nado, *Deep Learning Tuning Playbook*. https://github.com/google-research/tuning_playbook
- M. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, 2023, abstract, Figure 1, section 3.1. https://arxiv.org/abs/2309.14322
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[19.2 · Reproducing a paper](lesson-02.md)
