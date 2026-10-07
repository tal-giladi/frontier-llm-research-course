---
id: "13.1"
module: 13
minutes: 40
practice_minutes: 110
prerequisites: ["12.1", "12.4", "01.4"]
objectives:
  - Describe the post-training stages of Tülu 3, OLMo 3, Llama 3 and Qwen3, say what each stage is for, and name the evidence each report gives for it (per-stage tables, ablations, seeds, held-out suites).
  - Derive the DPO loss from the KL-regularised reward objective and implement it, with Tülu 3's length normalisation and Llama 3's NLL term, from scratch.
  - Build preference pairs from ratings with Tülu 3's binarisation rule and train DPO arms with a random-label control, judged by Eval Suite v2.
  - Compute the item noise of a benchmark (binomial standard error, minimum detectable difference) and decide which published stage gains it can resolve.
volatility: implementation
sources:
  - title: "Lambert et al., Tülu 3: Pushing Frontiers in Open Language Model Post-Training (sections 2.3, 3.2, 4.3.1, 5.1.2, 5.2.1, 6; Tables 6, 7, 14, 20, 21)"
    url: https://arxiv.org/abs/2411.15124
  - title: "Team Olmo, Olmo 3 (sections 4.2-4.4, Eq. 1-2, Table 22)"
    url: https://arxiv.org/abs/2512.13961
  - title: "Llama Team, The Llama 3 Herd of Models (sections 4.1.2-4.1.6, 4.2.2)"
    url: https://arxiv.org/abs/2407.21783
  - title: "Qwen Team, Qwen3 Technical Report (sections 4.1-4.5, Tables 21-22)"
    url: https://arxiv.org/abs/2505.09388
  - title: "Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model (section 4, Eq. 4-7)"
    url: https://arxiv.org/abs/2305.18290
  - title: "Geng et al., The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains"
    url: https://arxiv.org/abs/2507.06187
  - title: "OLMo 2 1B stage checkpoints: allenai/OLMo-2-0425-1B, -SFT, -DPO, -Instruct (Apache-2.0)"
    url: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
last_verified: "2026-10-07"
---

# 13.1 · Open recipes as case studies

Four open post-training recipes cover most of what the field publicly knows about turning a base model into an assistant: Tülu 3, OLMo 3, Llama 3 and Qwen3. This lesson reads each one as a sequence of stages with a purpose and a piece of evidence behind it, builds the stage they all share after supervised fine-tuning (preference optimisation with DPO) from scratch, measures three published variants of it on the course's toy policy with Eval Suite v2 and a random-label control, and then audits the reports' own stage tables against the item noise of the benchmarks they use.

## Why this matters at a frontier lab

A post-training team does not invent a pipeline; it starts from the best documented one and changes what its evidence says it can change. That only works if you can read a recipe the way a reviewer reads an experiment: which stage buys which capability, what the stage costs, and how strong the evidence is. The reports differ a lot on the last point. Tülu 3 publishes a per-stage table, ablations for each stage and a seed study; OLMo 3 publishes every checkpoint and dataset, and one-run stage ablations; Llama 3 describes its choices with the observations behind them; Qwen3 reports company-run comparisons with GPU-hours. A gain of 0.4 points on an average, or 12 points on a 30-problem benchmark, can be real or noise, and the report alone often cannot tell you which. Knowing that before you copy a stage saves the GPU-weeks of finding out.

## The idea

### What each stage is for

| Stage | What it changes | Training signal | Typical cost driver |
|---|---|---|---|
| Supervised fine-tuning (SFT) | format, chat behaviour, skills shown by demonstrations | next-token loss on (prompt, response) pairs, loss on response tokens only | data curation; tokens |
| Preference optimisation (DPO or RLHF) | which of the model's own kinds of answer it prefers | pairs (chosen, rejected) from humans, an AI judge or a stronger/weaker model | producing and rating responses |
| RL with verifiable rewards (RLVR) | success on checkable tasks (math, code, format constraints) | a verifier's reward on the policy's samples (Module 12) | sampling |
| Distillation | transfers a stronger model's behaviour to a smaller one | the teacher's outputs or per-token probabilities (lesson 13.2) | teacher compute |

### Four recipes

**Tülu 3** (Lambert et al. 2024; PUBLICLY DOCUMENTED). Four stages (section 2.3): curate prompts and decontaminate them against the evaluations (8-gram matching, section 3.2); SFT on a 939,344-prompt mixture (Table 7); **length-normalised DPO** on 354,192 preference pairs for the 8B model, built from on-policy Tülu 3 SFT completions and off-policy completions of other models, each rated 1–5 by GPT-4o on helpfulness, instruction following, honesty and truthfulness, then binarised: the highest mean rating is chosen, the rejected response is drawn at random from the lower-rated ones (section 5.2.1); **RLVR** with PPO and a reward of $\alpha = 10$ for a verified answer, on GSM8K, MATH and IFEval-style constraints, $\beta = 0.05$ for the final 8B (section 6, Table 21). The 8B model moves from 60.6 (SFT) to 64.7 (DPO) to 65.1 (RLVR) on the development-suite average (Table 6).

**OLMo 3** (Team Olmo 2025; PUBLICLY DOCUMENTED). The Think models go SFT → DPO → RLVR. The DPO stage is built on the **delta-learning** idea: pair a chosen response from a strong model (Qwen3 32B, thinking) with a rejected response from a much weaker one (Qwen3 0.6B, thinking), because "the quality of preference data depends primarily on the quality of the delta between chosen and rejected responses" (section 4.3). The report also states that further SFT on Qwen3 32B traces "outright hurts" its SFT model, which is why the same completions are used as *chosen* responses in DPO instead. RL uses OlmoRL: token-level loss, truncated importance sampling against vLLM's probabilities, clip-higher and no standard-deviation normalisation of the advantage (section 4.4.1, Eq. 1–2): the Module 12 pieces, chosen. All data and every stage checkpoint are released.

**Llama 3** (Llama Team 2024; PUBLICLY DOCUMENTED). Six rounds of the same loop (section 4.1.6): train a reward model on human preferences (4.1.2); sample $K$ = 10 to 30 responses per prompt from the latest model and keep the reward model's best (**rejection sampling**, 4.2.2); SFT on that plus other data (4.1.3); DPO with $\beta = 0.1$, learning rate $10^{-5}$, formatting tokens masked out of the loss and an NLL term with coefficient 0.2 on the chosen response "to stabilize DPO training ... and preventing the decrease of log probability of chosen responses" (4.1.4); average the models from different data or hyperparameter versions (4.1.5).

**Qwen3** (Qwen Team 2025; PUBLICLY DOCUMENTED, company-run comparisons). The flagship models go through four stages (section 4): a small long-chain-of-thought cold start, reasoning RL with GRPO on 3,995 query–verifier pairs, **thinking-mode fusion** (SFT on data with and without reasoning, switched by `/think` and `/no_think`; lesson 13.4), and general RL. The smaller models (0.6B to 14B and 30B-A3B) are not trained that way: they are **distilled** from the large ones, first off-policy (teacher outputs) and then on-policy, matching the teacher's logits on the student's own samples (section 4.5; lesson 13.2).

### The shared stage: DPO from the RLHF objective

RLHF maximises reward with a KL leash to a reference policy $\pi_{\text{ref}}$ (the SFT model):

$$\max_{\pi}\; \mathbb{E}_{x,\, y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big[\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big],$$

with $r$ the reward, $\beta > 0$ the strength of the leash. Its exact solution (Rafailov et al. Eq. 4) is

$$\pi^\ast(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}, \qquad Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, e^{r(x,y)/\beta}.$$

Solve for the reward: $r(x, y) = \beta \log \frac{\pi^\ast(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$. Put that into the Bradley-Terry model of lesson 12.1, $P(y_c \succ y_r) = \sigma(r(x,y_c) - r(x,y_r))$: the intractable $\beta \log Z(x)$ appears in both terms and cancels. Replacing $\pi^\ast$ by the trainable policy $\pi_\theta$ gives the DPO loss (Eq. 7):

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\big(\hat r_\theta(x, y_c) - \hat r_\theta(x, y_r)\big), \qquad \hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)},$$

where $y_c$ is the chosen response, $y_r$ the rejected one and $\hat r_\theta$ the *implicit reward*. Its gradient is $-\beta\, \sigma(\hat r_\theta(y_r) - \hat r_\theta(y_c))\,[\nabla \log \pi_\theta(y_c) - \nabla \log \pi_\theta(y_r)]$: each pair is weighted by how wrongly the implicit reward orders it (section 4). Nothing in the loss says that $\log \pi_\theta(y_c)$ must *rise*; only the difference of the two log-ratios must. It can be satisfied by pushing both down, the rejected one faster. That is the failure Llama 3's NLL term guards against.

Two variants from the recipes:

- **Length-normalised DPO** (Tülu 3 Eq. 6): each log-ratio is divided by its response length $|y|$, so a long response does not get a larger implicit reward for being long. The per-token ratios are small, so $\beta$ is larger (5 for Tülu 3 8B, Table 20, against 0.1 for Llama 3).
- **DPO + NLL** (Llama 3 4.1.4): $\mathcal{L} = \mathcal{L}_{\text{DPO}} + \lambda \cdot \big(-\tfrac{1}{|y_c|}\log \pi_\theta(y_c \mid x)\big)$ with $\lambda = 0.2$.

### How strong is a stage table?

A score on $n$ items, each right with probability $p$, has a binomial standard error $\sqrt{p(1-p)/n}$. Two independent scores near $p$ differ significantly at the 95% level only if they differ by more than the **minimum detectable difference** $1.96\sqrt{2p(1-p)/n}$. A paired comparison over the same items (lesson 01.4) does better, but needs per-item results, which reports rarely publish. When a model is scored by averaging many samples per item, item-to-item variation still remains, so the binomial figure is a conservative bound, not an exact one. Seeds add a second source of variation the item formula does not see: Tülu 3's five 8B SFT seeds averaged 59.8 to 60.1 (Table 14).

## Worked example

**The DPO optimum.** Three responses with $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$, rewards $(1, 0, 0)$, $\beta = 1$. Then $\pi^\ast \propto (0.5e, 0.3, 0.2) = (1.359, 0.3, 0.2)$, so $\pi^\ast = (0.731, 0.161, 0.108)$. The implicit rewards $\log(\pi^\ast/\pi_{\text{ref}})$ are $(0.380, -0.620, -0.620)$: the differences equal the reward differences (1 and 0), and the constant $\log Z = -0.620$ is what cancelled.

**One DPO pair.** $\beta = 0.1$, the chosen response's log-ratio is $+2.0$ and the rejected one's $-1.0$: $h = 0.1 \cdot (2.0 - (-1.0)) = 0.3$, loss $= -\log \sigma(0.3) = -\log 0.5744 = 0.554$, gradient weight $\sigma(-0.3) = 0.426$. At the start of training $\pi_\theta = \pi_{\text{ref}}$, so $h = 0$ and the loss is $\log 2 = 0.693$ for every pair.

**Length normalisation.** Same log-ratios, chosen 4 tokens and rejected 2: per-token ratios $+0.5$ and $-0.5$, so with $\beta = 0.5$, $h = 0.5$.

**Is RLVR's gain real?** Tülu 3 8B's GSM8K goes 84.3 → 87.6 (Table 6) on 1,319 test problems. MDE at $p = 0.843$: $1.96\sqrt{2 \cdot 0.843 \cdot 0.157/1319} = 0.0278$, i.e. 2.8 points: the +3.3 is beyond item noise. IFEval 81.1 → 82.4 on 541 prompts: MDE 4.7 points, the +1.3 is not. On one AIME year (30 problems) at $p = 0.5$ the MDE is $1.96\sqrt{2 \cdot 0.25/30} = 0.253$: 25 points.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| pair batch: chosen rows, then rejected rows | (2B, T) = (128, ≤ 16) | int64 | CPU (free path) / GPU |
| response mask | (2B, T) | float32 | same |
| policy sequence log-probabilities | (2B,) | float32 (logits upcast before the softmax) | same |
| reference log-probabilities, computed once | (P,) and (P,) for P pairs | float32 | same |
| implicit-reward margin $h$ | (B,) | float32 | same |

Cost of one DPO epoch over $P$ pairs of $T_c + T_r$ tokens: the policy's forward and backward $6N(T_c + T_r)P$, and the reference's forward $2N(T_c + T_r)P$ once (cache it; the frozen reference never changes). With no sampling and no reward model in the loop, DPO is the cheapest preference stage; the price is that its pairs are fixed, so it cannot explore beyond them. Producing the pairs costs extra: sampling $k$ responses per prompt and rating them. Free CPU, measured: one arm's 150 steps of 64 pairs is $4.3 \times 10^{11}$ FLOPs of training, $0.7 \times 10^{11}$ for the reference and $2.4 \times 10^{11}$ for sampling the 32,000 candidates (`Ledger` in the lab output), about 20 seconds. Main path (PROJECTED, pending the Module 13 pilot): Eval v2 on a 1.5B checkpoint costs 0.3–0.5 GPU-hours (lesson 12.4's estimate), so the 7 evaluations of the lab are 2–3.5 H100-hours; the project projects the DPO stage itself.

## Build it

```python
import copy
from frontierlab.posttrain.sft import load_policy
from frontierlab.pipeline import dpo, toy
from frontierlab.pipeline.compute import Ledger

base = load_policy("runs/m12/sft/policy.pt")                # Module 12's SFT checkpoint is the reference
probs = toy.train_problems(8000, seed=11)                    # training triples only
cands = toy.sample_texts(base, probs, 4, temperature=1.0)    # 4 responses per prompt, as in Tülu 3
pairs, _ = toy.build_pairs(probs, cands, toy.aspect_score)   # two-aspect rating, Tülu 3 binarisation
policy, led = copy.deepcopy(base), Ledger()
dpo.train_dpo(policy, base, pairs, steps=150, beta=0.1, nll_coef=0.2, lr=5e-5, ledger=led)
print(led.line(), toy.eval_v2(policy)["summary"])
```

`frontierlab/pipeline/dpo.py` holds the loss (plain, length-normalised, with NLL, with soft labels), the cached reference pass and the training loop; `seqs.py` the padding and masks shared by every stage; `toy.py` the rating and binarisation; `compute.py` the ledger. Correctness checks in `labs/common/tests/test_pipeline.py`: the loss equals $\log 2$ when policy and reference agree; its gradient equals $-\beta\,\sigma(-h)$ on the chosen and $+\beta\,\sigma(-h)$ on the rejected log-probability; on a categorical toy, minimising DPO on all pairs of a finite response set converges to $\pi_{\text{ref}}\, e^{r/\beta}/Z$; changing tokens after EOS does not change any sequence log-probability.

> [!NOTE]
> TRL as a mapping, read from the source at tag v1.14.1 (the course pin; `trl/trainer/dpo_config.py` and `dpo_trainer.py`): `DPOConfig(beta=..., loss_type=["sigmoid"])` is plain DPO; `loss_type=["sigmoid_norm"]` divides each log-ratio by its response length (Tülu 3's form); `loss_type=["sigmoid", "sft"]` with `loss_weights=[1.0, 0.2]` adds an NLL term on the chosen responses (TRL averages it over all chosen tokens of the batch, the course per response, so long responses weigh more in TRL's); `precompute_ref_log_probs=True` caches the reference pass. The course's own code is the reference implementation; TRL is how you would run it at scale.

## What the evidence says

- **SFT → preference optimisation → RL with verifiable rewards: ESTABLISHED** as the open-recipe shape (Tülu 3, OLMo 2 and 3, and, with rejection sampling in place of RLVR, Llama 3). PUBLICLY DOCUMENTED in each report's stage section.
- **DPO as the preference stage: ESTABLISHED** (Tülu 3, OLMo 3, Llama 3). Tülu 3 found that only length-normalised DPO beat its starting checkpoint among DPO, SimPO and length-normalised DPO on UltraFeedback (section 5.4.1, Table 18): **length normalisation is PROMISING**, one lab's ablation. **The NLL term is MODEL-SPECIFIC**, justified in Llama 3 by an observation rather than a published ablation table.
- **Delta learning: PROMISING** (Geng et al.; used in OLMo 3). OLMo 3's Table 22 is "from one run only": SFT 70.1, +DPO 72.7, +RLVR 74.1, against SFT+RLVR 71.9.
- **Distillation for small models instead of the full pipeline: MODEL-SPECIFIC but strong** (Qwen3 Table 21: from the same off-policy-distilled 8B checkpoint, RL reached 67.6 on AIME'24 in 17,920 GPU-hours, on-policy distillation 74.4 in 1,800; company-run, one comparison). Lesson 13.2.
- **Reading the tables.** The Tülu 3 8B RLVR stage adds 0.4 points on the average (65.1 against 64.7), close to the 0.3-point range of its own SFT seeds. Per benchmark, GSM8K (+3.3 on 1,319 items) is beyond item noise and IFEval (+1.3 on 541) is not. The final RLVR checkpoint was "picked ... with best overall performance on MATH and IFEval" from evaluations every 100 steps (section 6.4): selection on the development suite, which is why Tülu 3 also reports an *unseen* suite (section 7). On AIME (30 problems) no single published difference in this lesson's table exceeds the unpaired 95% bound; the reports give no per-item results to pair. This is an INFERENCE about evidence strength, not a claim that the gains are absent.
- **Course measurement (free CPU, 2026-10-07, torch 2.14.1 CPU, 8 threads, 3 min 35 s for both parts).** From Module 12's SFT checkpoint, 4,735 and 4,583 pairs (seeds 0, 1) from 8,000 prompts; 150 DPO steps; Eval v2 against the SFT start:

| Arm | add_pass1 Δ (s0, s1) | sft_nll Δ (nats) | if_correct Δ | chosen log-prob change | Eval v2 |
|---|---|---|---|---|---|
| `dpo` ($\beta$ 0.1) | −0.058, −0.102 | −0.21, −0.26 | −0.10, −0.14 | −1.63, −1.70 | FAIL, FAIL |
| `dpo-norm` ($\beta$ 0.5) | −0.036, −0.085 | −0.14, −0.16 | −0.05, −0.08 | −1.28, −1.34 | FAIL, FAIL |
| `dpo-nll` ($\beta$ 0.1, $\lambda$ 0.2) | **+0.039, +0.043** (both improved) | +0.03, +0.04 | +0.13, +0.16 | +0.27, +0.27 | FAIL (sub_greedy interval), PASS |
| `random-nll` control | +0.004, +0.012 | +0.01, +0.01 | +0.03, −0.01 | +0.04, +0.02 | PASS, FAIL (if_correct interval) |

  Plain and length-normalised DPO *lowered* the probability of the chosen responses by 1.3–1.7 nats while raising the margin, and every task and retention component got worse in both seeds: the failure Llama 3's NLL term names. With the NLL term the chosen probability rose and the task improved in both seeds; the random-label control with the same NLL term did not move the task, so the gain comes from the preference signal, not from the NLL term's self-training. Two "FAIL"s are interval failures, not losses: `sub_greedy`'s 200-item interval is about ±0.08, wider than the 0.02 guard, so the guard there cannot be met by any arm, which is something to fix in the suite (more items), not in the model. A 308k-parameter toy is not evidence about DPO at 8B; it shows the mechanism.

## Lab

**Folder:** [`labs/module-13/lesson-01/`](../../labs/module-13/) · **Time:** about 1 hour 50 minutes (about 4 minutes unattended) · **Pass check:** `pytest labs/module-13/lesson-01` passes; `recipe_lab.py` prints Part A (8 Eval v2 comparisons) and Part B (the audit); your write-up gives the decision below and one paragraph per recipe on what its evidence does and does not show.

### Experiment contract

- **Question:** which form of the DPO loss (plain, length-normalised, with an NLL term) improves the toy task over the SFT start without failing Eval v2, on the same pairs? Decision informed: the preference-stage loss for the module project's pipeline (SFT → DPO → RLVR → distillation).
- **Hypothesis:** plain DPO lowers the chosen responses' log-probability (Llama 3 4.1.4 reports the effect and adds an NLL term against it); the NLL arm improves `add_pass1` and holds retention; the random-label control does not improve the task. Status: reported effect (Llama 3 section 4.1.4, which cites Pang et al. 2024 for the NLL term); the size at toy scale is unknown.
- **Baseline:** the SFT checkpoint (`runs/m12/sft`), the starting point and reference of every arm.
- **Changed variable:** the loss (4 arms). **Controlled:** the same pairs per seed (8,000 prompts × 4 samples at temperature 1, the two-aspect rating, your binarisation), 150 steps of 64 pairs, lr $5 \times 10^{-5}$, seeds 0–1, all Eval v2 pins.
- **Comparison axis:** equal pairs and steps (equal training FLOPs, $4.3 \times 10^{11}$ per arm).
- **Budget:** free CPU, measured 3 min 35 s for Parts A and B.
- **Metrics and decision rule:** Eval v2 against the SFT start, guards 0.02. Adopt the cheapest arm whose `add_pass1` is "improved" in both seeds and that has no component with a mean loss larger than the guard; report the intervals that are too wide to meet the guard separately.
- **Correctness checks:** your TODO tests; the script stops if your `dpo_loss` disagrees with the course's on a real batch; `test_pipeline.py`; the loss starts at $\log 2$.
- **Fallback evidence:** Tülu 3 Table 18 and Llama 3 4.1.4 (labelled as published).
- **Limits:** a tiny model and a two-aspect programmatic rating instead of a judge; 2 seeds; item intervals within a seed.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB (any 24 GB+ GPU works). Not run in this build; part of the Module 13 pilot | `python labs/module-13/lesson-01/recipe_lab.py --variant main --print` lists 7 Eval v2 runs of `frontierlab.pipeline.hf_eval` on the released OLMo 2 1B stage checkpoints (base, SFT, DPO, Instruct = RLVR) at pinned revisions, plain format for all four, chat format for the three post-trained ones, and the stage comparisons. **PROJECTED:** 0.3–0.5 GPU-hours per checkpoint, 2–3.5 GPU-hours, USD 4–11 |
| Free GPU (Colab/Kaggle T4) | T4 16 GB | the same commands with `--n-gsm8k 200 --n-lambada 300` (OLMo 2 1B fits in fp16 or fp32) |
| Free CPU | laptop; measured 3 min 35 s | the steps below |

### Steps

1. **Implement** the five TODOs in `lab.py` and run `pytest labs/module-13/lesson-01`.
2. **Run** `python labs/module-13/lesson-01/recipe_lab.py`. Part A writes `runs/m13/l131/<arm>-s<seed>/result.json`; Part B writes `runs/m13/l131/audit.json`.
3. **Read Part A.** For each arm: the task verdicts, the retention and instruction verdicts, the chosen and rejected log-probability changes. Which arm passes the decision rule? Which "regressed" verdicts are wide intervals rather than losses?
4. **Read Part B.** For each recipe, which reported stage gains are beyond item noise, which are not, and what evidence the report would need to publish to settle the rest (per-item results, seeds, a held-out suite).
5. **Write up** (one page): the loss you would use for the project's preference stage and why; for each of the four recipes, one sentence on what each stage is for and one on how strong its evidence is.

<details>
<summary>Hint for TODO 1</summary>

Compute the two log-ratios, normalise them if asked, then `-F.logsigmoid(beta * (lc - lr)).mean()`. The NLL term uses the *policy's* chosen log-probability divided by its length, not the log-ratio.

</details>

<details>
<summary>Hint for TODO 5</summary>

The scores are in percent; `mde_unpaired` takes and returns fractions. Multiply by 100 before comparing with the delta.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Part A is the table in "What the evidence says". By the decision rule, `dpo-nll` is the arm to adopt: "improved" on `add_pass1` in both seeds (+0.039 [+0.021, +0.058] and +0.043 [+0.021, +0.064]), no guarded component's mean fell. Its one "regressed" verdict is `sub_greedy` in seed 0 with interval [−0.040, +0.120] around a mean of +0.040: a wide interval, not a loss. Plain DPO fails clearly in both seeds (subtraction accuracy fell from 0.38 to 0.105 and 0.125): the margin grew while the chosen log-probability fell by 1.6–1.7 nats.

Part B, the 95% unpaired item bound against each published change: Tülu 3 8B GSM8K SFT→DPO +8.1 (MDE 3.3) and DPO→RLVR +3.3 (2.8) are beyond item noise; IFEval SFT→DPO +8.3 (5.3) beyond, DPO→RLVR +1.3 (4.7) within; MMLU SFT→DPO +2.8 (1.1) beyond, DPO→RLVR −0.5 within. The averages have no item count; against Tülu 3's SFT seed range of 0.3, SFT→DPO +4.1 is far outside it and DPO→RLVR +0.4 barely. OLMo 3 Think 7B IFEval −2.0 after DPO is within noise and +6.4 after RLVR is beyond; AIME 2025 +5.1 and +1.5 are within (MDE about 25 on 30 problems). Qwen3-8B AIME'24 +12.6 (RL) and +19.4 (on-policy distillation) are also within the conservative unpaired bound; the report's pass@64 and GPU-hour columns and its other benchmarks (MATH500 92.4 → 94.8 and 97.0 on 500 problems) carry more of the weight.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-13/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-13/lesson-01`.

</details>

## Common mistakes

- **Copying a stage without its evidence.** A stage that adds 0.4 points on an average may be within seed noise; check the per-benchmark numbers and the item counts first.
- **Reading a rising DPO margin as success.** The margin can grow while the chosen response becomes *less* likely; log the chosen and rejected log-probability changes, not only the loss and the reward accuracy.
- **Recomputing the reference every step.** It is frozen; compute its log-probabilities once.
- **Comparing $\beta$ across DPO forms.** A length-normalised log-ratio is a per-token quantity; its $\beta$ is on a different scale (Tülu 3 uses 5).
- **Selecting the checkpoint on the evaluation you report.** Tülu 3 says it did so on its development suite and reports an unseen suite as well; do the same or say you did not.
- **Treating "within noise" as "no effect".** It means the benchmark cannot tell; a larger or paired evaluation might.

## References

- N. Lambert et al., *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, 2024, sections 2.3, 3.2, 4.3.1, 5.1.2, 5.2.1, 5.4.1, 6, 7; Tables 6, 7, 14, 18, 20, 21. https://arxiv.org/abs/2411.15124
- Team Olmo, *Olmo 3*, 2025, sections 4.2–4.4 (Eq. 1–2), Table 22. https://arxiv.org/abs/2512.13961
- Llama Team, *The Llama 3 Herd of Models*, 2024, sections 4.1.2–4.1.6 and 4.2.2. https://arxiv.org/abs/2407.21783
- Qwen Team, *Qwen3 Technical Report*, 2025, sections 4.1–4.5, Tables 21–22. https://arxiv.org/abs/2505.09388
- R. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023, section 4 (Eq. 4–7). https://arxiv.org/abs/2305.18290
- S. Geng et al., *The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains*, 2025. https://arxiv.org/abs/2507.06187
- OLMo 2 1B stage checkpoints (Apache-2.0): `allenai/OLMo-2-0425-1B` `a1847dff`, `-SFT` `0d85a3d0`, `-DPO` `c4b04859`, `-Instruct` `48d788ec` (full hashes in `recipe_lab.py`). https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[13.2 · Distillation](lesson-02.md)
