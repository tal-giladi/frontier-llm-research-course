---
id: "13.3"
module: 13
minutes: 40
practice_minutes: 110
prerequisites: ["13.1", "12.1", "12.4"]
objectives:
  - Describe how Constitutional AI (critique and revision, then RL from AI feedback), Claude's constitution, OpenAI's Model Spec and deliberative alignment turn a written specification into training signal, and label which parts are publicly documented and which are company claims.
  - Write a mini-spec as code, train an AI judge from imperfect AI feedback, and measure the judge's false-accept and false-reject rates against the spec itself.
  - Train a policy against the judge (DPO and SFT forms) with an oracle arm and a random-ranking control, counting the judge's compute.
  - Report adherence (refusal where required, over-refusal where not) together with Eval v2 retention, and trace a policy's failure back to a judge or labeller bias.
volatility: implementation
sources:
  - title: "Bai et al., Constitutional AI: Harmlessness from AI Feedback (sections 1, 3, 4.1-4.4)"
    url: https://arxiv.org/abs/2212.08073
  - title: "Anthropic, Claude's new constitution (2026-01-22; text at anthropic.com/constitution, CC0)"
    url: https://www.anthropic.com/news/claude-new-constitution
  - title: "OpenAI, Model Spec (version 2026-08-18; CC0)"
    url: https://model-spec.openai.com/2026-08-18.html
  - title: "Guan et al., Deliberative Alignment: Reasoning Enables Safer Language Models (sections 2.3, 2.4, 4.1)"
    url: https://arxiv.org/abs/2412.16339
  - title: "OpenAI, GPT-5 System Card (2025-08-13; section 3.1, from hard refusals to safe-completions)"
    url: https://cdn.openai.com/gpt-5-system-card.pdf
  - title: "Brahman et al., The Art of Saying No: Contextual Noncompliance in Language Models (CoCoNot)"
    url: https://arxiv.org/abs/2407.12043
  - title: "allenai/coconot dataset (ODC-By; revision 2cbe16aa)"
    url: https://huggingface.co/datasets/allenai/coconot
last_verified: "2026-10-07"
---

# 13.3 · Specification-driven alignment

Labs increasingly write down how their models should behave, as a constitution or a model spec, and train against that document with an AI judge in place of human labels for most of the work. This lesson reads the four public versions of that idea (Constitutional AI, Claude's constitution, OpenAI's Model Spec and deliberative alignment), then builds the whole loop on the toy model: a four-clause mini-spec written as code, a simulated AI labeller with a realistic bias, a judge trained on its labels, and policies trained against the judge, measured on adherence and on Eval v2 retention, with the judge's compute on the bill.

## Why this matters at a frontier lab

Human preference labels are slow, expensive and hard to change. A written specification plus an AI judge makes the objective editable: change a clause, relabel overnight. That is the promise of Constitutional AI and its successors, and it is also the risk. The policy is optimised against the *judge*, not the document, so every error the judge makes (a clause it misreads, a boundary it draws in the wrong place, a quality it ignores) becomes a behaviour of the model. Measuring the judge against the specification before using it, keeping an independent check of adherence, and watching what the spec does not mention (here, correctness) are the parts of this work that decide whether it helped.

## The idea

### Four public versions

**Constitutional AI** (Bai et al. 2022; PUBLICLY DOCUMENTED). Two stages. *Supervised*: sample responses to red-team prompts from a helpful-only model, ask the model to critique each response against a randomly drawn principle and to revise it, repeat (16 harmlessness principles, 4 revisions per prompt), and fine-tune on the revisions together with helpful responses to ordinary prompts, "in order to retain helpfulness as much as possible" (section 3). *RL from AI feedback*: a feedback model is shown a prompt, two responses and a principle as a multiple-choice question; its normalised probabilities of (A) and (B) are the preference labels (section 4.1). Soft labels worked "much better" than hard 0/1 ones, and chain-of-thought labels, which are over-confident, were clamped to 40–60% (section 4.3). A preference model trained on those labels (mixed with human helpfulness labels) is the RL reward. The report also names the failure to watch: over-trained RL-CAI models showed "Goodharting behavior", overly harsh or boilerplate responses (section 4.3). The stated aim was a model that is "harmless but non-evasive" (section 4.4).

**Claude's constitution** (Anthropic, 2026-01-22; company document, CC0). Four core properties in priority order, "broadly safe", "broadly ethical", "compliant with Anthropic's guidelines" and "genuinely helpful", which Claude "should generally prioritize ... in the order in which they're listed" when they conflict; a small set of hard constraints; and a preference for explaining *why* over rigid rules, so the model can "apply broad principles rather than mechanically following specific rules". Anthropic says the constitution is used to "construct many kinds of synthetic training data", including "rankings of possible responses", growing out of the Constitutional AI methods used since 2023 (company claim: the training details are not published).

**OpenAI's Model Spec** (version 2026-08-18; company document, CC0). A *chain of command* of authority levels: Root (rules no message can override, "mostly prohibitive"), System, Developer, User and Guideline, where "instructions with higher authority override those with lower authority"; defaults lower in the chain can be changed by developers and users. Principles such as "assume best intentions" argue against over-refusal. OpenAI states "we are training our models to align to the principles in the Model Spec" and that production models "do not yet fully reflect" it (company claim).

**Deliberative alignment** (Guan et al. 2024; PUBLICLY DOCUMENTED, OpenAI authors). The model learns to reason *about the spec text* in its chain of thought. SFT data: a reasoning model is prompted with the relevant safety specification, its completions are judged $k$ times by a "judge" reasoning model $\mathcal{G}_{RM}$ that is also given the spec, the *minimum* score is kept "because any single run may overlook an issue", and the spec is then removed from the prompt so the model must recall it (section 2.3). RL (section 2.4): the same judge, with the spec, gives reward on safety prompts; the chain of thought is hidden from it during RL "to reduce the chance of encouraging deceptive CoTs". An ablation finds that safety data in SFT only or RL only gives "intermediate results" between none and both (section 4.1). The GPT-5 system card later reports moving "from hard refusals to safe-completions" (section 3.1): a binary comply/refuse is itself a simplification.

### The loop, abstractly

$$\text{spec } S \xrightarrow{\text{labeller}} \text{labels } \ell \xrightarrow{\text{train}} \text{judge } J_\phi \xrightarrow{\text{score candidates}} \text{pairs or filtered SFT data} \xrightarrow{\text{DPO / SFT / RL}} \pi_\theta$$

Each arrow can lose information. Write $c(x, y) \in \{0, 1\}$ for "response $y$ to prompt $x$ complies with $S$" (the ground truth, which in practice only careful review approximates) and $J_\phi(x, y) \in [0, 1]$ for the judge's probability. The judge's errors are its **false-accept rate** $P(J > 0.5 \mid c = 0)$ (violations it rewards) and **false-reject rate** $P(J \le 0.5 \mid c = 1)$ (compliant responses it punishes). Random label noise largely averages out when a judge is trained on many labels; a *systematic* labeller error does not: where the labeller is wrong more often than right, the judge learns the error, and the policy learns it from the judge.

### Spec-T, the toy mini-spec

| Clause | Level | Rule |
|---|---|---|
| 1. Restricted requests | root | first operand ≥ 90: the only compliant response is `nil` |
| 2. No over-refusal | root | `nil` on any other request violates the spec |
| 3. Format | user | follow the format the tag asks for |
| 4. Correctness | guideline | not judged; Eval v2 measures it |

The simulated labeller flips 5% of labels at random and, on first operands 85–89, applies the restricted rule 80% of the time (it reads "90 or more" as "about 90"): a labeller that is over-cautious near the boundary. Clause 4 is deliberately left to Eval v2, as many real judges leave factual accuracy to other checks.

## Worked example

**Judge error rates.** Six held-out responses, judge probabilities $(0.9, 0.8, 0.2, 0.7, 0.1, 0.6)$, ground truth $(1, 0, 1, 0, 0, 1)$. Violations: responses 2, 4, 5; the judge passes 2 and 4: false-accept $2/3$. Compliant: 1, 3, 6; it fails 3: false-reject $1/3$. Accuracy $3/6$.

**Why the systematic error survives.** On prompts with first operand 85–89, an answer is labelled compliant with probability $0.2 \cdot 0.95 + 0.8 \cdot 0.05 = 0.23$ and `nil` with probability $0.8 \cdot 0.95 + 0.2 \cdot 0.05 = 0.77$. A judge trained to predict the labels converges towards $P(\text{label} = 1)$: it prefers `nil` there, and DPO against it prefers `nil` there. On prompts with first operand below 80 the 5% noise leaves the majority label right, so the judge is right.

**Judge plus verifier.** A ranking that adds 0.5 when a verifier says the number is right changes which candidate wins wherever the judge's margin is smaller than 0.5. With judge probabilities 0.62 for `nil` and 0.30 for a correct answer, the combined scores are 0.62 and 0.80: the correct answer becomes the chosen response.

**Judge compute.** Judge $N_J = 308{,}497$ parameters, trained 400 steps of 128 sequences of about 12 tokens: $6 \cdot 308{,}497 \cdot 614{,}000 \approx 1.1 \times 10^{12}$ FLOPs (the lab's ledger: $1.15 \times 10^{12}$). Scoring 30,000 candidates of about 9 tokens: $2 \cdot 308{,}497 \cdot 270{,}000 = 1.7 \times 10^{11}$. The judge costs more than the DPO run it serves ($0.86 \times 10^{12}$).

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| judge input: BOS, prompt, response, EOS, PAD | (B, 18) | int64 | CPU (free path) / GPU |
| index of the last real token | (B,) | int64 | same |
| judge logit, probability | (B,) | float32 | same |
| AI-feedback labels | (8,130,) | float64 (NumPy) | CPU |
| DPO pair batch | (128, ≤ 16) | int64 | same |

Main path (PROJECTED, pending the Module 13 pilot): policy Qwen3-1.7B-Base after the project's SFT stage; training judge Qwen3-8B in non-thinking mode, prompted with the main-path spec (`SPEC_MAIN` in `hf_stages.py`) and read as $P(\text{Yes})$ normalised over Yes/No (Constitutional AI's multiple-choice form); 4,000 prompts (half CoCoNot "should not comply" prompts, half ordinary Tülu prompts) × 4 samples of up to 256 tokens. Sampling $1.6 \times 10^4 \times 256 = 4.1 \times 10^6$ tokens at 2–3k tokens/s: 0.4–0.6 hours. Judge scoring about $1.6 \times 10^4 \times 500$ tokens $\times\, 2 \times 8.19 \times 10^9 = 1.3 \times 10^{17}$ FLOPs, about 6 minutes at 40% MFU (15–20 minutes with batched `transformers` forward passes). DPO 200 steps: minutes. Evaluation on CoCoNot's 1,001 + 379 test prompts with a *different* judge (Qwen3-4B, asked only "did it decline?") plus a 50-response human audit, and Eval v2 twice: about 1–1.5 hours. In all 2–3 GPU-hours, USD 4–9.

## Build it

```python
from frontierlab.pipeline import judge as J, toy, dpo
from frontierlab.pipeline.compute import Ledger
from frontierlab.posttrain.sft import load_policy

base, led = load_policy("runs/m12/sft/policy.pt"), Ledger()
jp = J.spec_problems(3000, seed=21)                                   # 25% restricted, 25% borderline
texts = toy.sample_texts(base, jp, 2, ledger=led)
P = [p for p, ts in zip(jp, texts) for _ in ts + [0]]
T = [t for ts in texts for t, _ in ts + [(J.REFUSAL, True)]]
F = [f for ts in texts for _, f in ts + [(J.REFUSAL, True)]]
labels = J.ai_feedback(P, T, F)                                      # 5% noise, misreads 85-89
judge, _ = J.train_judge(P, T, F, labels, ledger=led)
print(J.judge_report(J.judge_probs(judge, P, T, F), P, T, F))        # against Spec-T itself
print(J.adherence(base))                                             # refusal / over-refusal / format
```

`frontierlab/pipeline/judge.py` holds Spec-T (`SPEC_T`, `spec_check`), the balanced and held-out prompt sets, the simulated labeller, the judge model, its training and scoring (charged to the ledger), the judge report and the adherence measurement. The main-path version is `hf_stages.py spec` (judge with a prompted 8B model, pairs, DPO) and `spec-eval` (CoCoNot refusal rates with a separate evaluation judge and an audit file). Correctness checks in `labs/common/tests/test_pipeline.py`: every Spec-T clause on hand-made cases; the labeller's bias and noise rates within 1.5 points of their settings on 20,000 draws; a judge trained on clean labels exceeds 90% accuracy and its compute is counted.

## What the evidence says

- **AI feedback can replace most human harmlessness labels: ESTABLISHED in principle** (Constitutional AI; RLAIF is now standard in open recipes, e.g. Tülu 3's GPT-4o preference ratings). The size of the benefit depends on the judge; Constitutional AI's Figure 2 shows RL-CAI less harmful at a given helpfulness than human-feedback RLHF in their crowdworker comparisons (PUBLICLY DOCUMENTED, one lab).
- **Training to reason over a written spec (deliberative alignment): PROMISING.** PUBLICLY DOCUMENTED by its authors, with an ablation of the SFT and RL stages; reported as a Pareto improvement on under- and over-refusal for o1; not independently replicated at scale.
- **Constitutions and model specs as training targets: MODEL-SPECIFIC, company claims.** The documents are public (CC0); how exactly they enter training is not, beyond the statements quoted above.
- **Judge errors become policy behaviour: ESTABLISHED as a mechanism** (reward-model over-optimisation, lesson 12.1; CAI's Goodharting observation), and the reason to measure a judge against the spec before using it.
- **Course measurement (free CPU, 2026-10-07, torch 2.14.1 CPU, 8 threads, 452 s, another module's jobs sharing the CPU).** The labeller agreed with Spec-T on 85.9% of 8,130 labels; the judge trained on them reached 83.8% accuracy on held-out candidates: 100% on restricted, 99.8% on ordinary prompts, 51% on borderline ones, where it accepted *every* refusal on 85–89 (false-accept 13.6%, false-reject 18.4% overall). Arms from the SFT start (which never refuses: 66.7% compliance), seeds 0 and 1:

| Arm | Compliance | Refuses restricted | Over-refuses 85–89 | add_pass1 Δ | sft_nll Δ (nats) | FLOPs ($10^{12}$) |
|---|---|---|---|---|---|---|
| `judge-dpo` | 0.823, 0.840 | 0.98, 0.98 | **0.91, 0.81** | −0.043, −0.039 | −0.11, −0.13 | 2.48 |
| `judge+verifier-dpo` | 0.938, 0.913 | 0.95, 0.955 | 0.18, 0.25 | +0.014, +0.007 | −0.02, −0.05 | 2.48 |
| `judge-sft` | 0.838, 0.830 | 0.98, 0.98 | 0.88, 0.92 | −0.022, −0.019 | −0.11, −0.09 | 1.97 |
| `oracle-dpo` | **0.995, 0.997** | 1.00, 1.00 | 0.00, 0.00 | −0.027, −0.012 | −0.06, −0.02 | 1.13 |
| `random-dpo` (control) | 0.667, 0.667 | 0.00, 0.00 | 0.00, 0.00 | −0.013, −0.008 | −0.02, −0.05 | 1.13 |

  No arm over-refused below 85 (at most 3% on 80–84 for `judge-sft`). Training against the judge learned the hard rule almost perfectly and learned the labeller's misreading just as well: 81–92% refusals on 85–89, where the spec says answer. The oracle arm shows the spec itself was learnable at little cost. Adding a verifier for the clause the judge does not check cut the inherited over-refusal by about three quarters and was the only judge arm whose task score did not fall. The random control did nothing to adherence and lost a little retention (−0.02 and −0.05 nats), which is the cost of 300 DPO+NLL steps on the model's own samples; the judge-only arms lost two to five times more. The judge's training and scoring ($1.3 \times 10^{12}$ FLOPs) more than doubled the cost of each judge arm. This is a designed toy world with a planted labeller bias; it shows the mechanism, not its size in an LLM.

## Lab

**Folder:** [`labs/module-13/lesson-03/`](../../labs/module-13/) · **Time:** about 1 hour 50 minutes (about 8 minutes unattended) · **Pass check:** `pytest labs/module-13/lesson-03` passes; `spec_lab.py` prints the judge report and 10 arm rows; your write-up applies the decision rule and traces the over-refusal on 85–89 to its source.

### Experiment contract

- **Question:** does training against an AI judge trained from imperfect AI feedback make the policy follow Spec-T, at what retention cost and compute, and which judge errors does it inherit? Decision informed: whether the project's pipeline uses a judge-only ranking or a judge-plus-verifier ranking for any spec-following stage.
- **Hypothesis:** every judge arm learns clause 1 (restricted → `nil`); the judge arms also learn the labeller's misreading on 85–89, the oracle arm does not; the judge-only arms lose more retention than the random control because the judge rewards wrong answers in the right format; adding a verifier reduces that. Status: mechanism established (12.1); the effect sizes here are designed into the toy.
- **Baseline:** the SFT checkpoint; the random-ranking control (same pipeline, no signal) and the oracle arm (Spec-T as the ranking) bracket the judge arms.
- **Changed variable:** the ranking signal and, for `judge-sft`, the training form. **Controlled:** the judge (trained once, seed 0), 6,000 prompts per seed with 4 samples plus `nil`, DPO $\beta$ 0.1 with NLL 0.2, lr $3 \times 10^{-4}$, 300 steps of 64, seeds 0–1, Eval v2 pins, the 600 held-out adherence prompts.
- **Comparison axis:** equal prompts, candidates and steps; FLOPs reported per arm, the judge's included.
- **Budget:** free CPU, measured 452 s.
- **Metrics and decision rule:** adherence (compliance; refusal on restricted; over-refusal on 85–89, 80–84 and below 80) and Eval v2 against the SFT start. Adopt a ranking for the project only if, in both seeds, compliance is at least 0.9, over-refusal on 85–89 at most 0.3 and `add_pass1` does not fall by more than 0.02.
- **Correctness checks:** your TODO tests; the script stops if your `spec_check` disagrees with Spec-T; the judge report is computed against Spec-T, not against the labels.
- **Fallback evidence:** Constitutional AI sections 4.3–4.4 and deliberative alignment section 4.1 (labelled as published).
- **Limits:** a designed spec with one planted labeller bias; a binary refusal instead of safe completions; one judge; 2 seeds.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 13 pilot | `python labs/module-13/lesson-03/spec_lab.py --variant main --print`: `hf_stages spec` (policy = the project's Qwen3-1.7B-Base SFT checkpoint, judge Qwen3-8B with `SPEC_MAIN`), `hf_stages spec-eval` with Qwen3-4B as the separate evaluation judge on CoCoNot original/test (1,001, should decline) and contrast/test (379, should comply), a human audit of the 50 responses in `audit.jsonl`, and Eval v2. **PROJECTED:** 2–3 GPU-hours, USD 4–9 |
| Free GPU (Colab/Kaggle T4) | T4 16 GB | the same with Qwen3-0.6B-Base as the policy and Qwen3-1.7B (post-trained) as the training judge, `--n 1000 --k 4` |
| Free CPU | laptop; measured 452 s | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-13/lesson-03`.
2. **Run** `python labs/module-13/lesson-03/spec_lab.py`.
3. **Read the judge report** before any arm: which group does the judge get wrong, and in which direction?
4. **Read the arms.** For each, adherence by group, the Eval v2 changes and the FLOPs with and without the judge. Compare every judge arm with both the oracle and the random control.
5. **Write up** (one page): the decision under the rule; where the over-refusal on 85–89 comes from (labeller, judge, or training) and the evidence; what you would change first in a real pipeline (relabel the boundary, a second judge, a verifier, an audit) and what each costs.

<details>
<summary>Hint for TODO 1</summary>

Check `finished` first, then the restricted case, then the refusal, then the format. The format patterns are `\d+`, `\d{3}` (digits + 1 for two-digit problems), `#\d+#` and `\d+!`, matched against the whole text with `re.fullmatch`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

The table in "What the evidence says". Under the rule `judge+verifier-dpo` qualifies: compliance 0.938 and 0.913, over-refusal on 85–89 of 0.18 and 0.25 (under the 0.3 limit), `add_pass1` +0.014 and +0.007. `oracle-dpo` is the best on adherence (0.995 and 0.997, no over-refusal) but misses the task condition in seed 0 by 0.007 (−0.027), well within the interval of a 200-item task component: say so rather than rounding it in, and note that the oracle is not available in practice anyway. The judge-only arms fail on over-refusal (81–92%) and on the task.

The trace: the labeller agreed with the spec on 85.9% of labels, but on 85–89 its majority label was wrong; the judge then accepted every refusal there (`accepts_refusal_85_89` 1.0), while it was right on 99.8% of ordinary prompts; DPO and SFT against it produced the refusals. The oracle arm, trained on the same prompts and candidates, did not. So the source is the labeller's systematic error, transmitted by the judge. Relabelling the boundary region (or auditing the judge there) is the cheap fix; a verifier helps only with what it checks.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-13/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-13/lesson-03`.

</details>

## Common mistakes

- **Measuring adherence with the training judge.** The policy has been optimised against it; use the spec itself, a different judge, or human review.
- **Reading high judge accuracy as a safe judge.** 83.8% overall hid a 51% accuracy exactly at the boundary the spec cares about; report accuracy per group.
- **Forgetting what the spec does not say.** A judge for behaviour rewards a confidently wrong answer in the right format; pair it with a verifier or a capability check.
- **Counting the policy's compute only.** The judge's training and scoring more than doubled each judge arm's FLOPs here; on the main path the judge is 8B against a 1.7B policy.
- **Hard labels from an over-confident judge.** Constitutional AI found soft (or clamped) labels more robust; use probabilities when the judge gives them.
- **Treating refusal as the only safe behaviour.** Over-refusal is a spec violation too (Model Spec "assume best intentions", CAI's non-evasiveness, GPT-5's safe-completions).

## References

- Y. Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, 2022, sections 1, 3, 4.1–4.4. https://arxiv.org/abs/2212.08073
- Anthropic, *Claude's new constitution*, 2026-01-22 (text at https://www.anthropic.com/constitution, CC0). https://www.anthropic.com/news/claude-new-constitution
- OpenAI, *Model Spec*, version 2026-08-18 (CC0). https://model-spec.openai.com/2026-08-18.html
- M. Y. Guan et al., *Deliberative Alignment: Reasoning Enables Safer Language Models*, 2024, sections 2.3, 2.4, 4.1. https://arxiv.org/abs/2412.16339
- OpenAI, *GPT-5 System Card*, 2025-08-13, section 3.1. https://cdn.openai.com/gpt-5-system-card.pdf
- F. Brahman et al., *The Art of Saying No: Contextual Noncompliance in Language Models* (CoCoNot), 2024. https://arxiv.org/abs/2407.12043
- allenai/coconot, revision `2cbe16aabf9069f17e48c8daad8aeabc29469eb7`, ODC-By. https://huggingface.co/datasets/allenai/coconot
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[13.4 · Thinking modes and budgets](lesson-04.md)
