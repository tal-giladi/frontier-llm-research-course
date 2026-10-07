---
id: "13.2"
module: 13
minutes: 35
practice_minutes: 110
prerequisites: ["13.1", "12.2", "12.4"]
objectives:
  - Distinguish off-policy distillation (SFT on teacher outputs) from on-policy distillation (per-token reverse KL on the student's own samples), and say which train/inference mismatch each one has.
  - Show that the sampled per-token reverse-KL surrogate is an unbiased estimate of the exact reverse-KL gradient at a position, and implement both forms.
  - Count teacher compute (sampling vs scoring, one-off teacher training) in a distillation comparison and decide whether it changes the ranking.
  - Run SFT-on-teacher, on-policy distillation, RL and a self-teacher control at equal steps and judge them with Eval Suite v2.
volatility: concept
sources:
  - title: "Agarwal et al., On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD; sections 3-4)"
    url: https://arxiv.org/abs/2306.13649
  - title: "Lu and Thinking Machines Lab, On-Policy Distillation (2025-10-27)"
    url: https://thinkingmachines.ai/blog/on-policy-distillation/
  - title: "Qwen Team, Qwen3 Technical Report (section 4.5, Table 21)"
    url: https://arxiv.org/abs/2505.09388
  - title: "Kim and Rush, Sequence-Level Knowledge Distillation"
    url: https://arxiv.org/abs/1606.07947
  - title: "Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network"
    url: https://arxiv.org/abs/1503.02531
  - title: "Gu et al., MiniLLM: On-Policy Distillation of Large Language Models"
    url: https://arxiv.org/abs/2306.08543
  - title: "Qwen/Qwen3-8B model card (teacher; Apache-2.0; revision b968826d)"
    url: https://huggingface.co/Qwen/Qwen3-8B
last_verified: "2026-10-07"
---

# 13.2 · Distillation

Distillation trains a small model on a larger model's behaviour, and it is how most small open models are now post-trained: Qwen3's models up to 14B are distilled from the 32B and 235B ones rather than taken through the full pipeline. There are two ways to do it. The student can be fine-tuned on text the teacher wrote, or it can write its own text and be corrected token by token by the teacher. This lesson derives both, builds them from scratch, and compares them with RL and a control at equal training steps, with the teacher's compute counted.

## Why this matters at a frontier lab

A lab that has a strong model does not post-train its small models from scratch. It distils, and the choice between the two forms of distillation decides the compute bill and which mistakes the student keeps. Qwen3 reports that on-policy distillation of its 8B model reached 74.4 on AIME'24 in 1,800 GPU-hours, against 67.6 in 17,920 GPU-hours for RL from the same checkpoint (Table 21). Thinking Machines report 9–30× lower cost than SFT on teacher outputs for reaching a target score. Both numbers depend on what is counted (teacher training, teacher sampling, teacher scoring) and on the task. A comparison that leaves the teacher out of the bill, or that runs on a task where the student never leaves the teacher's distribution, can rank the methods the wrong way round.

## The idea

### Two kinds of distillation

Let $\pi_T$ be the teacher, $\pi_s$ the student, $x$ a prompt and $y = (y_1, \dots, y_L)$ a response.

**Off-policy (sequence-level) distillation.** Sample $y \sim \pi_T(\cdot \mid x)$ and minimise the SFT loss $-\sum_t \log \pi_s(y_t \mid x, y_{<t})$ on it (Kim and Rush 2016; Qwen3's "off-policy distillation", section 4.5). In expectation this is the *forward* KL $\mathrm{KL}(\pi_T \,\|\, \pi_s)$ over teacher sequences, which is mass-covering: the student must put probability everywhere the teacher does. Its weakness is the one GKD names: the student trains on the teacher's prefixes $y_{<t}$, but at inference it conditions on its *own*, and after its first mistake it is in prefixes it never trained on (Agarwal et al. section 1).

**On-policy distillation.** Sample $y \sim \pi_s$ and lower, at every position $t$ of the student's own response, the per-token *reverse* KL

$$\mathrm{KL}_t = \sum_{v} \pi_s(v \mid c_t)\,\big[\log \pi_s(v \mid c_t) - \log \pi_T(v \mid c_t)\big], \qquad c_t = (x, y_{<t}),$$

where $v$ runs over the vocabulary and $c_t$ is the context. The student is corrected exactly where it goes, including after its own mistakes. Reverse KL is mode-seeking: it pushes the student towards one of the teacher's high-probability behaviours rather than spreading over all of them (GKD section 3; MiniLLM uses reverse KL for the same reason). GKD generalises the recipe: a student-data fraction $\lambda$ (1 = fully on-policy) and a choice of divergence, forward KL, reverse KL or the generalised Jensen–Shannon divergence between them.

### Two estimators of the on-policy gradient

**Exact.** Compute $\mathrm{KL}_t$ over the whole vocabulary from both models' logits and differentiate. This needs the teacher's full distribution at every position: free for the toy's 35-token vocabulary, a $(B, R, 151{,}936)$ tensor for a Qwen3 teacher.

**Sampled** (Thinking Machines). Use only the sampled token. With $d_t = \log \pi_s(y_t \mid c_t) - \log \pi_T(y_t \mid c_t)$, set the advantage $A_t = -d_t$ (detached) and use the policy-gradient surrogate $-A_t \log \pi_s(y_t \mid c_t)$. Because $\mathbb{E}_{y_t \sim \pi_s}[\nabla \log \pi_s(y_t)] = 0$,

$$\mathbb{E}_{y_t \sim \pi_s}\big[d_t\, \nabla \log \pi_s(y_t \mid c_t)\big] = \nabla \mathrm{KL}_t,$$

so the surrogate is unbiased at each position. The blog uses "a discount factor of zero": token $t$ is credited only with its own KL, not with the KL of the tokens it leads to. That is a choice, not a consequence: the gradient of the *sequence-level* reverse KL also contains those future terms. The surrogate needs one teacher forward pass over the student's tokens and the teacher's log-probability of each sampled token, nothing more, which is why it fits an RL stack: it is RL with a dense, per-token reward $-d_t$.

Both forms compare the two models' probabilities of *the same tokens*, so teacher and student must share a tokenizer (INFERENCE from the definition; Qwen3's family shares one, which is part of why Qwen3 can distil inside it).

### Counting the teacher

| Arm | Student FLOPs | Teacher FLOPs | What dominates on a GPU |
|---|---|---|---|
| SFT on teacher outputs | $6 N_s$ per trained token | $2 N_T$ per *generated* token | teacher generation (decode is memory-bound) |
| On-policy distillation | $2 N_s$ per sampled token + $6 N_s$ per trained token | $2 N_T$ per *scored* token | student sampling; teacher scoring is one parallel forward pass |
| RL with a verifier | $2 N_s$ sampled + $6 N_s$ trained (+ reference passes) | none | sampling |

The teacher's own training is a one-off cost; whether a comparison charges it depends on whether the teacher exists anyway. Thinking Machines report 9× when "the SFT dataset is given", about 18× in GPU-hours when the on-policy run pays for the teacher's log-probabilities and the off-policy run does not, and about 30× "if we include the full cost of the teacher model in off-policy distillation", i.e. its sampling (company blog).

## Worked example

**Two divergences by hand.** One position, two tokens, $\pi_s = (0.5, 0.5)$, $\pi_T = (0.9, 0.1)$. Reverse KL $= 0.5 \ln(0.5/0.9) + 0.5 \ln(0.5/0.1) = -0.294 + 0.805 = 0.511$ nats. Forward KL $= 0.9 \ln(0.9/0.5) + 0.1 \ln(0.1/0.5) = 0.529 - 0.161 = 0.368$ nats. They differ, and so do their gradients: the reverse KL is dominated by the token the teacher thinks unlikely and the student does not.

**The sampled estimator.** Same distributions. Sampled token 1: $d = \ln 0.5 - \ln 0.9 = -0.588$, advantage $+0.588$: raise it. Token 2: $d = \ln 0.5 - \ln 0.1 = +1.609$, advantage $-1.609$: lower it. With logits $z$, $\nabla_z \log \pi_s(y) = e_y - \pi_s$, with $e_y$ the one-hot vector of token $y$, so the expected gradient of the surrogate $-A \log \pi_s(y)$ is $\mathbb{E}[d\,(e_y - \pi_s)] = 0.5 \cdot (-0.588)\,(0.5, -0.5) + 0.5 \cdot 1.609\,(-0.5, 0.5) = (-0.549, +0.549)$. The exact gradient of the reverse KL with respect to the logits is $\pi_s \odot (\log\pi_s - \log\pi_T - \mathrm{KL}) = 0.5\,(-0.588 - 0.511,\; 1.609 - 0.511) = (-0.549, +0.549)$. They agree, and a descent step moves the student towards token 1.

**Counting the teacher.** Toy student $N_s = 308{,}400$, teacher $N_T = 1{,}174{,}880$. On-policy, one step of 128 responses of about 10 tokens (prompt included): student sampling $2 N_s \cdot 1{,}280 = 7.9 \times 10^8$, teacher scoring $2 N_T \cdot 1{,}280 = 3.0 \times 10^9$, student update $6 N_s \cdot 1{,}280 = 2.4 \times 10^9$: the teacher is 49% of the step's FLOPs because it is 3.8 times larger than the student. The teacher's one-off training was $1.37 \times 10^{13}$ FLOPs, eleven times the lab's whole 200-step off-policy arm.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| student rollout tokens | (P·G, 7 + R) = (128, ≤ 15) | int64 | CPU (free path) / GPU |
| student logits on response positions | (128, R, 38) toy; (128, R, 151,936) Qwen3 | float32 | same |
| teacher logits (exact form only) | same shape as the student's | float32 (bf16 on GPU, upcast) | same |
| sampled log-probabilities (student, teacher) | (128, R) | float32 | same |
| per-token reverse KL, mask | (128, R) | float32 | same |

Main path (PROJECTED, pending the Module 13 pilot): student Qwen3-1.7B-Base ($N_s = 1.72 \times 10^9$), teacher Qwen3-8B ($N_T = 8.19 \times 10^9$, 16.4 GB in bf16), 100 steps of 32 GSM8K prompts × 4 samples, about 700 prompt and 300 response tokens per sequence: $1.28 \times 10^7$ tokens per arm. On-policy: student sampling $4.4 \times 10^{16}$, teacher scoring $2.1 \times 10^{17}$, student update $1.3 \times 10^{17}$ FLOPs, $3.9 \times 10^{17}$ in all, about 17 minutes at 40% of an H100's 989 TFLOP/s; generating $3.8 \times 10^6$ response tokens at 2–3k tokens/s with `generate` adds 20–30 minutes. Off-policy: the 8B teacher generates the same $3.8 \times 10^6$ tokens at roughly 1–1.5k tokens/s (40–60 minutes), plus the student's training. Memory: the student's float32 weights and AdamW state are $16 \times 1.72 \times 10^9 = 27.5$ GB, the teacher 16.4 GB, so both fit on one 80 GB GPU with activation memory to spare at batch 32.

## Build it

```python
from frontierlab.pipeline import distill, toy
from frontierlab.pipeline.compute import Ledger
from frontierlab.posttrain.sft import load_policy
from frontierlab.posttrain.tasks import make_problems, split_problems

teacher = load_policy(distill.ensure_teacher("runs/m13/teacher"))      # 1.17M parameters, same tokenizer
student = load_policy("runs/m12/sft/policy.pt")
_, held = split_problems(2)
prompts = lambda step: make_problems(16, 2, "+", ".", seed=step, exclude=held)
led = Ledger()
distill.train_opd(student, teacher, prompts, steps=200, kind="sampled", ledger=led)   # or kind="exact"
print(led.line(), toy.eval_v2(student)["summary"])
```

`frontierlab/pipeline/distill.py` holds the teacher, both divergences, `opd_loss` (sampled and exact), `train_opd`, `teacher_examples` for the off-policy arm and `rl_ledger` to count a Module 12 RL run on the same terms. `hf_stages.py distill --mode offline|onpolicy` is the main-path version on Hugging Face models. Correctness checks in `labs/common/tests/test_pipeline.py`: the sampled surrogate's expected gradient equals the exact reverse-KL gradient by enumeration over a 6-token vocabulary (float64, $10^{-12}$); both divergences by hand; teacher scoring is charged at $N_T/N_s$ times student sampling; teacher samples are charged as teacher, not student, compute.

> [!NOTE]
> TRL as a mapping (trl 1.14.1 in the course pins): GKD lives in `trl.experimental.gkd` at tag v1.14.1: `GKDConfig(lmbda=..., beta=...)`, where `lmbda` is the student-data fraction $\lambda$ (default 0.5) and `beta` the generalised JSD coefficient; the config's docstring says `beta=0.0` gives the KL divergence and `beta=1.0` the "Inverse KL" (this lesson's reverse KL). It is experimental: pin it and read its loss before trusting it. The sampled per-token form is what Thinking Machines run through an RL loss with "advantages = −reverse KL"; in the course it is `opd_loss(kind="sampled")`.

## What the evidence says

- **On-policy distillation beats SFT on teacher outputs for long reasoning: PROMISING, two strong reports.** GKD (summarisation, translation, GSM8K with T5 students): "on-policy and mixed variants consistently outperform supervised variants" (section 4). Thinking Machines (company blog): student Qwen3-8B-Base, teacher Qwen3-32B, from a 400k-example SFT checkpoint at 60% AIME'24, about 150 on-policy steps reached 70%, where extrapolated SFT would need about 2M examples: the 9–30× range above, depending on which teacher costs are counted.
- **Distillation instead of RL for small models: MODEL-SPECIFIC but documented** (Qwen3 Table 21, company-run: RL 67.6 AIME'24 at 17,920 GPU-hours, on-policy distillation 74.4 at 1,800, from the same off-policy-distilled 8B; RL left pass@64 at 90.0, distillation raised it to 93.3). The comparison needs a teacher that already solves the task; RL does not.
- **Distillation to recover lost behaviour: PROMISING.** Thinking Machines' personalisation experiment: mid-training Qwen3-8B on internal documents dropped IF-eval from 85% to 79% (70% documents) or 45% (100%); on-policy distillation from the original model brought it back to 83% while keeping most of the knowledge gain (41% vs 36% on internal QA). This is the retention use of distillation the module project uses as its final stage.
- **Reverse KL is mode-seeking: ESTABLISHED** as a property of the divergence; whether it is the right choice is task-dependent (GKD finds JSD variants best for small T5 students and the gap shrinking for larger ones).
- **Course measurement (free CPU, 2026-10-07, torch 2.14.1 CPU, 8 threads; 511 s for 10 runs, plus about 15 minutes once for the teacher, with another module's jobs sharing the CPU).** Student = Module 12's SFT checkpoint; 200 steps × 16 prompts of plain addition; Eval v2 against the SFT start (paired 95% intervals over 200 items):

| Arm | add_pass1 Δ (s0, s1) | add_greedy Δ | sft_nll Δ (nats) | sub_greedy Δ | FLOPs per arm (teacher share) |
|---|---|---|---|---|---|
| `sft-teacher` | **+0.346, +0.352** | +0.505, +0.500 | +0.003, −0.000 | −0.035, −0.095 | $1.23 \times 10^{12}$ (56%, teacher sampling) |
| `opd` (sampled) | +0.167, +0.169 | +0.220, +0.225 | −0.022, −0.034 | +0.000, −0.030 | $1.97 \times 10^{12}$ (49%, teacher scoring) |
| `opd-exact` | +0.194, +0.201 | +0.275, +0.240 | −0.036, +0.027 | −0.030, +0.025 | $1.97 \times 10^{12}$ (49%) |
| `rl` (GRPO, strict verifier) | +0.176, +0.174 | +0.160, +0.235 | −0.095, −0.069 | −0.040, +0.095 | $1.16 \times 10^{12}$ (0%) |
| `opd-self` (control) | 0.000, 0.000 | 0.000, 0.000 | 0.000, 0.000 | 0.000, 0.000 | $1.26 \times 10^{12}$ |

  Here SFT on teacher outputs won clearly, with the least compute among the distillation arms: the *opposite* of the long-reasoning results. The likely reason (INFERENCE): responses are 2–4 tokens and the teacher is 97–100% right, so there is almost no prefix mismatch for on-policy training to fix, while the off-policy arm gets full, correct sequences as targets. The exact estimator beat the sampled one in both seeds (+0.03; lower variance). On-policy distillation matched RL's pass@1 gain while losing a third as much of the SFT data's likelihood (−0.02 to −0.04 nats against −0.07 to −0.10). The self-teacher control is exactly zero: a teacher identical to the student gives zero KL and zero gradient, which checks the plumbing rather than measuring an effect. Counting the teacher's one-off training ($1.37 \times 10^{13}$ FLOPs) would multiply the distillation arms' costs by 8 to 12; it only makes sense to distil from a teacher that exists for other reasons.

## Lab

**Folder:** [`labs/module-13/lesson-02/`](../../labs/module-13/) · **Time:** about 1 hour 50 minutes (about 25 minutes unattended, teacher included) · **Pass check:** `pytest labs/module-13/lesson-02` passes; `distill_lab.py` prints 10 rows; your write-up ranks the arms at equal steps and at equal FLOPs (with and without the teacher's training) and states the decision below.

### Experiment contract

- **Question:** at equal training steps and prompts, which of SFT on teacher outputs, on-policy distillation (sampled or exact reverse KL) and RL improves the toy student most, and does the ranking change when every arm's FLOPs, the teacher's included, are counted? Decision informed: the final stage of the module project (distilling the RLVR model) and its loss.
- **Hypothesis:** on-policy distillation beats SFT on teacher outputs (GKD, Thinking Machines, Qwen3); both beat RL per step; the exact estimator is at least as good as the sampled one. Status: reported effect for long reasoning sequences; may not appear for 3-token answers, where the student barely leaves the teacher's distribution.
- **Baseline:** the SFT checkpoint (`runs/m12/sft`); RL with Module 12's defaults is the non-distillation baseline.
- **Changed variable:** the training signal (5 arms). **Controlled:** the student start, the teacher, 200 steps × 16 prompts (plain addition, training triples, the same prompt stream per seed), 8 samples per prompt (on-policy arms and RL) or 8 teacher samples per prompt (SFT arm), lr $3 \times 10^{-4}$ for all (no per-arm tuning: the same zero tuning budget for every arm), seeds 0–1, Eval v2 pins.
- **Comparison axis:** equal steps and prompts (primary); equal FLOPs (secondary, from each arm's ledger).
- **Budget:** free CPU, measured 511 s for the 10 runs, about 15 minutes once for the teacher.
- **Metrics and decision rule:** `add_pass1` difference against the SFT start with its paired interval; `sft_nll` and `sub_greedy` as retention. Pick the arm with the largest `add_pass1` gain in both seeds; prefer the cheaper arm when two arms' intervals overlap; report the retention cost of the chosen arm.
- **Correctness checks:** your TODO tests (including the unbiasedness test); the script checks your functions against the course's on real logits; the self-teacher control must give exactly zero change.
- **Fallback evidence:** Qwen3 Table 21 and the Thinking Machines blog (labelled as published).
- **Limits:** 3-token answers; a near-perfect teacher; one learning rate for every arm; 2 seeds.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 13 pilot | `python labs/module-13/lesson-02/distill_lab.py --variant main --print`: student Qwen3-1.7B-Base, teacher Qwen3-8B (non-thinking, same tokenizer), 100 steps of 32 GSM8K prompts: `hf_stages distill --mode offline`, `--mode onpolicy`, the self-teacher control, and `posttrain.hf` RL; then `hf_eval` on each. **PROJECTED:** 0.7–1.0 GPU-hours per on-policy arm, about 1–1.3 for the off-policy arm, 0.8–1.4 for RL, 0.3–0.5 per Eval v2: about 5–7 GPU-hours, USD 10–21 |
| Free GPU (Colab/Kaggle T4) | T4 16 GB | student Qwen3-0.6B-Base, teacher Qwen3-1.7B (post-trained, same tokenizer), `--prompts 8 --max-new 256`, fp32 student |
| Free CPU | laptop; measured 511 s + teacher | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-13/lesson-02`.
2. **Run** `python labs/module-13/lesson-02/distill_lab.py`. The first run trains the teacher (`runs/m13/teacher`, about 4–15 minutes depending on load).
3. **Read** each arm's Eval v2 rows and its FLOPs line. Re-rank the arms by `add_pass1` gain per $10^{12}$ FLOPs, once without and once with the teacher's training cost.
4. **Explain** the ranking you got against the hypothesis, using the response length, the teacher's accuracy and the per-step logs (`history` in `runs/m13/l132/<arm>-s0/result.json`: `rkl_sampled` and `rkl_exact`).
5. **Write up** (one page): the arm you would use for the project's distillation stage and why; what would have to change in the task for on-policy distillation to win.

<details>
<summary>Hint for TODO 3</summary>

`adv = -(logp - teacher_logp).detach()`, `rho = torch.exp(logp - logp.detach())`, then `-(rho * adv * mask).sum() / mask.sum()`. The ratio is 1 in value; it exists so that the gradient is $\nabla \log \pi_s$.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

The table in "What the evidence says". By the decision rule, `sft-teacher` wins outright: +0.35 `add_pass1` in both seeds with intervals [+0.31, +0.38], far from every other arm's, and it is also the cheapest distillation arm ($1.23 \times 10^{12}$ FLOPs against $1.97 \times 10^{12}$). Per $10^{12}$ FLOPs: `sft-teacher` +0.28, `rl` +0.15, `opd-exact` +0.10, `opd` +0.09. With the teacher's one-off training charged to each distillation arm, those three fall to about +0.023, +0.013 and +0.011 per $10^{12}$, below RL. Its retention cost is small on the SFT data (`sft_nll` +0.003 and −0.000) and uncertain on subtraction (−0.035 and −0.095, intervals of about ±0.09). Every arm "fails" the guard only through `sub_greedy`'s wide interval or small `sft_nll` losses; RL lost the most likelihood on the SFT data. The hypothesis that on-policy distillation wins was not supported here, for the reason in the evidence section: 3-token answers from a near-perfect teacher leave nothing for on-policy correction to do. On the main path the responses are hundreds of tokens and the comparison is open.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-13/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-13/lesson-02`.

</details>

## Common mistakes

- **Leaving the teacher out of the bill**, or charging it to one arm and not the other. State which teacher costs each arm pays, and whether the teacher's training is amortised.
- **Calling SFT on teacher samples "on-policy" because the samples are fresh.** On-policy means the *student's* samples.
- **Distilling across tokenizers with per-token log-probabilities.** The two models must score the same tokens.
- **Computing the exact reverse KL on a large vocabulary without thinking about memory.** A (B, R, 151,936) float32 tensor at B = 128 and R = 512 is 40 GB; use the sampled form or chunk the computation.
- **Treating the self-distillation control as an effect size.** Zero is guaranteed when the teacher equals the student; it checks the code, not the method.
- **Generalising from short answers to long reasoning.** Exposure bias grows with sequence length; measure on the length you will deploy.

## References

- R. Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD), 2023, sections 3–4. https://arxiv.org/abs/2306.13649
- K. Lu and Thinking Machines Lab, *On-Policy Distillation*, 2025-10-27 (company blog). https://thinkingmachines.ai/blog/on-policy-distillation/
- Qwen Team, *Qwen3 Technical Report*, 2025, section 4.5, Table 21. https://arxiv.org/abs/2505.09388
- Y. Kim, A. M. Rush, *Sequence-Level Knowledge Distillation*, 2016. https://arxiv.org/abs/1606.07947
- G. Hinton, O. Vinyals, J. Dean, *Distilling the Knowledge in a Neural Network*, 2015. https://arxiv.org/abs/1503.02531
- Y. Gu et al., *MiniLLM: On-Policy Distillation of Large Language Models*, 2023. https://arxiv.org/abs/2306.08543
- Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` (8,190,735,360 parameters, Apache-2.0). https://huggingface.co/Qwen/Qwen3-8B
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[13.3 · Specification-driven alignment](lesson-03.md)
