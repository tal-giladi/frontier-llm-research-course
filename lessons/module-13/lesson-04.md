---
id: "13.4"
module: 13
minutes: 30
practice_minutes: 90
prerequisites: ["13.2", "12.4"]
objectives:
  - Describe the three disclosed ways to give one product a fast and a slow mode (one hybrid model with a switch, Qwen3 and DeepSeek-V3.1; separate models behind a router, GPT-5; separate releases, Qwen3-2507) and label each claim by its evidence.
  - Train one toy model with thinking and non-thinking modes, and measure accuracy and generated tokens in each.
  - Implement budget forcing, measure accuracy against a thinking budget, and test whether budget control emerges from mode fusion or has to be trained.
  - Train a router on a correctness signal and compare it with random routing at equal cost and with an oracle, on the cost-accuracy frontier.
volatility: implementation
sources:
  - title: "Qwen Team, Qwen3 Technical Report (sections 4.3, 4.4; Table 9; Figure 2)"
    url: https://arxiv.org/abs/2505.09388
  - title: "Qwen/Qwen3-1.7B model card (enable_thinking, /think and /no_think, sampling settings; revision 70d244cc)"
    url: https://huggingface.co/Qwen/Qwen3-1.7B
  - title: "OpenAI, GPT-5 System Card (2025-08-13; section 1)"
    url: https://cdn.openai.com/gpt-5-system-card.pdf
  - title: "OpenAI, GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum (2025-11)"
    url: https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/
  - title: "deepseek-ai/DeepSeek-V3.1 model card (hybrid thinking via the chat template)"
    url: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
  - title: "openai/gpt-oss-20b model card (reasoning effort low / medium / high in the system prompt)"
    url: https://huggingface.co/openai/gpt-oss-20b
  - title: "The Register, Alibaba admits Qwen3's hybrid-thinking mode was dumb (2025-07-31; reports the Qwen team's statement)"
    url: https://www.theregister.com/2025/07/31/alibaba_qwen3_hybrid_thinking/
last_verified: "2026-10-07"
---

# 13.4 · Thinking modes and budgets

Extension: a reasoning model spends tokens on thinking before it answers, and not every question needs them. This lesson looks at how labs expose that choice (one model with a switch, a router in front of two models, or two separate models), builds a toy model with both modes, forces it to answer under a thinking budget, and trains a router that decides per prompt whether to think, with every token counted.

## Why this matters at a frontier lab

Thinking tokens are the most expensive part of serving a reasoning model: they multiply latency and cost per answer, often by an order of magnitude, for questions that may not need them. The post-training pipeline decides what the product can do about it. Qwen3 trains one model that can think or not and can be cut off at a budget; OpenAI's GPT-5 puts a router in front of a fast model and a thinking model; the Qwen team later shipped separate instruct and thinking models instead of the hybrid. Each choice is made in post-training (what data, which modes, which signals train the router), and each needs the same measurement: accuracy against tokens, at the operating point the product will use.

## The idea

### Three designs, as disclosed

**One model, two modes (Qwen3; DeepSeek-V3.1).** Qwen3's third post-training stage, *thinking mode fusion*, fine-tunes the reasoning model on data with and without reasoning, marked by `/think` and `/no_think` in the user turn; a non-thinking response keeps an *empty* think block (Table 9), and thinking is the default (section 4.3; PUBLICLY DOCUMENTED). In the released checkpoints the switch is `enable_thinking` in `apply_chat_template`, with the soft switches working when thinking is enabled (Qwen3-1.7B model card). DeepSeek-V3.1's card states that "one model supports both thinking mode and non-thinking mode by changing the chat template" (PUBLICLY DOCUMENTED, model card). gpt-oss exposes three levels, set in the system prompt as "Reasoning: low / medium / high" (model card).

**Budget forcing (Qwen3).** "When the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process and insert the stop-thinking instruction", a fixed sentence ending in `</think>`, and the model answers from what it has; the report says this ability "is not explicitly trained but emerges naturally as a result of applying Thinking Mode Fusion" and that accuracy rises smoothly with the budget for Qwen3-235B-A22B (section 4.3, Figure 2; company-run).

**Two models behind a router (GPT-5).** "A smart and fast model that answers most questions, a deeper reasoning model for harder problems, and a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent"; the router "is continuously trained on real signals, including when users switch models, preference rates for responses, and measured correctness" (GPT-5 system card, section 1; company claim). The GPT-5.1 addendum adds that the fast model "can decide when to think" and the thinking model "adapts thinking time" per question, with the router still choosing between them (company claim, read via secondary coverage because the page refused automated access).

**Separate models (Qwen3-2507).** In July 2025 the Qwen team released separate Instruct-2507 and Thinking-2507 models and said, as reported, that it had stopped using the hybrid mode "to get the best quality possible" (company statement, secondary source: The Register, 2025-07-31). The Instruct-2507 card states that it "supports only non-thinking mode". Whether hybrid training costs quality is therefore *disputed by its own originator*; the evidence on either side is company-internal.

### What to measure

For a prompt $x$, mode $m \in \{\text{think}, \text{no-think}\}$ gives correctness $c_m(x) \in \{0,1\}$ and generated tokens $n_m(x)$. A routing rule $r(x) \in \{\text{think}, \text{no-think}\}$ has accuracy $\frac{1}{|X|}\sum_x c_{r(x)}(x)$ and cost $\frac{1}{|X|}\sum_x n_{r(x)}(x)$. A router that sends a fraction $f$ of prompts to thinking must beat **random routing at the same $f$**, whose accuracy is $f\,\bar c_{\text{think}} + (1 - f)\,\bar c_{\text{no-think}}$; the **oracle** thinks exactly when no-thinking is wrong and thinking is right. Plot every rule as a (tokens, accuracy) point and keep the non-dominated ones: the cost-accuracy frontier. A budget $B$ gives another family of points.

### The toy

Three-digit addition. Thinking mode writes the column sums from the units up with carries, then `#` (the toy's `</think>`), then the answer: 478 + 365 → `a13b14c08#843`, 14.5 generated tokens on average with EOS. Non-thinking mode writes `#843`, 5.5 tokens. One 308k-parameter model is trained on both modes, half each, for 1,000 steps: mode fusion in miniature. The `fusion+budget` variant also cuts 30% of the thinking traces at a random point before `#`, so the model sees answers written after partial thinking. The router is the judge architecture of lesson 13.3 reading only the prompt, trained on 4,000 training prompts to predict whether the non-thinking answer is right: one of GPT-5's disclosed signals, measured correctness.

## Worked example

**Random routing.** No-think accuracy 0.92 at 5.5 tokens, think 0.98 at 14.5. Send $f = 0.4$ at random: accuracy $0.4 \cdot 0.98 + 0.6 \cdot 0.92 = 0.944$, cost $0.4 \cdot 14.5 + 0.6 \cdot 5.5 = 9.1$ tokens. A router at $f = 0.4$ is useful only if it beats 0.944 at 9.1 tokens.

**The oracle.** If 7.7% of prompts fail without thinking and all of those succeed with it, the oracle thinks on exactly those: accuracy 1.0, cost $0.077 \cdot 14.5 + 0.923 \cdot 5.5 = 6.2$ tokens. The gap between 6.2 and a real router's cost at equal accuracy is what a better difficulty signal would buy.

**The frontier.** Points (5.5, 0.92), (9.3, 0.96), (10.0, 0.95), (14.5, 0.98): (10.0, 0.95) is dominated by (9.3, 0.96) (fewer tokens, higher accuracy); the frontier is the other three.

**A budget in tokens and FLOPs.** For the 308k-parameter toy, each generated token costs about $2N = 6.2 \times 10^5$ FLOPs, so thinking adds about $9 \times 6.2 \times 10^5 = 5.6 \times 10^6$ FLOPs per answer. For Qwen3-1.7B, 1,000 thinking tokens cost $2 \cdot 2.03 \times 10^9 \cdot 1{,}000 = 4 \times 10^{12}$ FLOPs per answer, and on a GPU the decode time matters more than the FLOPs: at 50–100 tokens/s for one sequence, 10–20 seconds of latency.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| prompts `h478+365=` / `n478+365=` with BOS | (B, 10) | int64 | CPU (free path) / GPU |
| greedy continuation, cached | (B, B_think + 7) | int64 | same |
| forced continuation after `#` | (B_forced, 10 + B + 1) → (B_forced, 6) | int64 | same |
| router logit per prompt | (B,) | float32 | same |
| per-prompt correctness and tokens, both modes | (1,000,) each | bool / float64 (NumPy) | CPU |

Training one toy model: 1,000 steps × 128 sequences of about 24 tokens, $6 \cdot 308{,}400 \cdot 3.1 \times 10^6 = 5.7 \times 10^{12}$ FLOPs (the ledger: $5.3$–$5.5 \times 10^{12}$ including the router). Main path (PROJECTED, pending the Module 13 pilot): Qwen3-1.7B (2,031,739,904 parameters in the released safetensors, which store the tied output matrix separately) on the first 500 GSM8K test questions: non-thinking (about 250 tokens each), thinking at budgets 0, 256, 512, 1,024, 2,048 (each plus up to 512 answer tokens) and unlimited, then the cascade router: about $4 \times 10^6$ generated tokens at 1.5–3k tokens/s with batched `generate`, 0.4–0.8 GPU-hours on an H100, about 1–1.5 GPU-hours with the cascade run, USD 2–5.

## Build it

```python
from frontierlab.pipeline import thinking as TH
from frontierlab.posttrain.sft import load_policy

model = load_policy("runs/m13/l134/fusion+budget-s1/policy.pt")
_, held = TH.split()
probs = TH.problems(held, 1000, seed=1, mode="h")
for B in (0, 3, 6, 9, None):                                # None = think until '#'
    rows = TH.answer_with_budget(model, probs, B)
    print(B, sum(r["correct"] for r in rows) / len(rows), sum(r["total_tokens"] for r in rows) / len(rows))
```

`frontierlab/pipeline/thinking.py` holds the task (trace, targets, the train/held-out split), batched budget forcing, and the routing curve, random-routing baseline and oracle. The main-path version is `labs/module-13/lesson-04/think_main.py`: Qwen3's chat template with `enable_thinking`, budget forcing with the report's stop-thinking sentence, and a cascade router (answer without thinking; think again if the answer's mean token log-probability is below a threshold; both passes counted). Correctness checks in `labs/common/tests/test_pipeline.py`: traces and targets by hand (including 999 + 1); forced answers never exceed the budget; the routing curve's endpoints equal "never think" and "always think" and the oracle by hand.

## What the evidence says

- **Hybrid thinking/non-thinking in one model: MODEL-SPECIFIC** (Qwen3, DeepSeek-V3.1, gpt-oss's effort levels; PUBLICLY DOCUMENTED as designs). Qwen3 Table 22 shows thinking-mode fusion and general RL raising instruction following and ThinkFollow (88.7 → 98.9 for the mode-switch test) while thinking-mode AIME'24 went 83.8 → 81.9 → 81.4 (company-run, one model). The originator's later move to separate models is a company statement, not a published ablation.
- **Budget forcing: PROMISING.** Qwen3 reports smooth gains with budget for its largest model and says the ability emerges from fusion; no independent replication in this lesson's sources.
- **Routing between a fast and a thinking model: MODEL-SPECIFIC, company claim** (GPT-5, GPT-5.1). The router's training signals are disclosed in one sentence; its accuracy is not.
- **Course measurement (free CPU, 2026-10-07, torch 2.14.1 CPU, 8 threads, 704 s for 4 models).** 1,000 held-out problems, greedy:

| Model, seed | Think acc. (14.5 tokens) | No-think acc. (5.5 tokens) | Budget B = 0 / 3 / 6 / 8 | Router at about 40% thinking (acc., tokens) vs random | Oracle (acc., tokens) |
|---|---|---|---|---|---|
| `fusion` s0 | 0.988 | 0.022 | 0.000 / 0.002 / 0.001 / 0.012 | router sends everything to thinking | 0.988, 14.2 |
| `fusion` s1 | 0.984 | 0.923 | 0.000 / 0.000 / 0.001 / 0.103 | 0.963, 9.3 vs 0.950 | 1.000, 6.2 |
| `fusion+budget` s0 | 0.943 | 0.856 | 0.330 / 0.257 / 0.670 / 0.932 | 0.914, 9.1 vs 0.893 | 0.968, 6.5 |
| `fusion+budget` s1 | 0.962 | 0.938 | 0.656 / 0.887 / 0.865 / 0.940 | 0.957, 7.8 (26%) vs 0.938 | 0.994, 6.0 |

  Four findings. (1) *Budget control did not emerge from fusion here*: the plain fusion model answered almost nothing correctly when its thinking was cut before the last column (B ≤ 8), in both seeds, even though the same model answered directly in non-thinking mode at 0.92 in seed 1. A truncated trace followed by `#` is a context it never saw. Training on 30% truncated traces made forcing work (0.66–0.94 at B = 0–8 in seed 1), at a cost of 2–4 points of full-thinking accuracy. Qwen3's emergence claim is about a 235B model trained on far more varied data; this toy says it is not automatic. (2) *The non-thinking skill arrives late and suddenly*: at 1,000 steps seed 0 had 0.02 and seed 1 0.92 without thinking, while both were at 0.98 with it. Thinking was learned first. (3) *The router beats random routing by 1.3–2.1 points at equal cost, but stays far from the oracle* (which needs only 6.0–6.5 tokens for 0.97–1.00): failures without thinking were not predictable from the prompt (no-think accuracy by number of carries: 0.88–0.98 in seed 1, nearly flat). (4) Seed variance is larger than every routing effect: report seeds before any claim. Pilot note (one seed, same script before the variants): the plain fusion model's no-think accuracy went 0.71, 0.97, 0.98 at 1,000, 1,500 and 2,500 steps, so the gap between the modes on this task closes with training.

## Lab

**Folder:** [`labs/module-13/lesson-04/`](../../labs/module-13/) · **Time:** about 1 hour 30 minutes (about 12 minutes unattended) · **Pass check:** `pytest labs/module-13/lesson-04` passes; `think_lab.py` prints the four model summaries; your write-up gives the decision below.

### Experiment contract

- **Question:** (a) does budget forcing work on a model trained by mode fusion alone, or must partial thinking be trained? (b) does a router trained on non-thinking correctness beat random routing at equal token cost? Decision informed: whether a hybrid model in the course needs budget-aware training data, and whether routing is worth a classifier.
- **Hypothesis:** (a) accuracy rises with B for both models (Qwen3's emergence claim); (b) the router beats random by more than the seed-to-seed spread. Status: (a) reported for a 235B model, may not appear at this scale; (b) company claim without published numbers.
- **Baseline:** for (a) the unforced thinking accuracy and no-think accuracy; for (b) random routing at the router's thinking fraction.
- **Changed variable:** the training data (with or without truncated traces); the routing rule. **Controlled:** the architecture, 1,000 steps of 128, lr $3 \times 10^{-3}$, 60,000 training problems, seeds 0–1, the 1,000 held-out problems, greedy decoding.
- **Comparison axis:** equal mean generated tokens (routing); equal budget (forcing).
- **Budget:** free CPU, measured 704 s.
- **Metrics and decision rule:** accuracy against B; router accuracy minus random accuracy at matched thinking fraction, per seed. (a) Budget training is needed if forced accuracy at B = 6 is below the no-think accuracy for the fusion model in both seeds. (b) A router is worth keeping if it beats random by at least 1 point in both seeds of the model that has both skills.
- **Correctness checks:** your TODO tests; the script checks your trace against the course's; forced answers never use more than B thinking tokens (`test_pipeline.py`).
- **Fallback evidence:** Qwen3 Figure 2 and Table 22 (labelled as published).
- **Limits:** a 308k-parameter model on one algorithmic task; greedy decoding; a router from one signal; 2 seeds; training stopped at a point where one seed had not yet learned the direct skill.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB (a 24 GB GPU suffices for a 1.7B model in bf16). Not run in this build; part of the Module 13 pilot | `python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-1.7B --n 500 --budgets 0,256,512,1024,2048,none --out runs/m13/l134-main`, then `--route cascade`. **PROJECTED:** 1–1.5 GPU-hours, USD 2–5 |
| Free GPU (Colab/Kaggle T4) | T4 16 GB | the same with `--model Qwen/Qwen3-0.6B --revision <pin at use> --n 200 --budgets 0,256,1024,none` in fp16 |
| Free CPU | laptop; measured 704 s | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-13/lesson-04`.
2. **Run** `python labs/module-13/lesson-04/think_lab.py`.
3. **Plot** for each model accuracy against tokens: the budget points, the router curve, random routing and the oracle (`runs/m13/l134/results.json`).
4. **Write up** (one page): the answers to (a) and (b) under the rules; what the per-carry accuracies say about why the router is far from the oracle; which of the three product designs you would choose for a model that must answer most questions fast, and what you would measure first on the main path.

<details>
<summary>Hint for TODO 1</summary>

Loop over the three columns from the units: digit $i$ of $a$ is `(a // 10**i) % 10`. The column sum includes the carry from the previous column, is written with two digits (`f"{s:02d}"`) after the column letter, and the next carry is `s // 10`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

The table in "What the evidence says". (a) Budget training is needed: the plain fusion model's forced accuracy at B = 6 was 0.001 in both seeds, below its no-think accuracy (0.022 and 0.923); with truncated traces in training it was 0.670 and 0.865. (b) The router clears the 1-point bar: in the `fusion` model with both skills (seed 1) it beat random by 1.3 points at 42% thinking; in `fusion+budget` by 2.1 and 1.9 points. It cannot reach the oracle because the cheap signal it sees, the prompt, barely predicts failure: no-think accuracy hardly varies with the number of carries. A real router has richer signals (conversation type, explicit requests, user switching), which is why GPT-5's is trained on them. For a product that must answer most questions fast, the toy suggests a hybrid model with budget-aware training plus a router, measured first on how predictable the fast mode's failures are.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-13/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-13/lesson-04`.

</details>

## Common mistakes

- **Comparing modes at unequal cost.** A thinking mode that is more accurate and three times as long is not "better" until you fix the token budget or the latency target.
- **Comparing a router with "always fast" instead of random routing at the same thinking fraction.** Any rule that thinks more will look better than never thinking.
- **Assuming budget forcing works.** Measure accuracy at each budget; a model may never have learned to answer from a truncated trace.
- **Greedy decoding on Qwen3's thinking mode.** The model card says not to; use its sampling settings (temperature 0.6, top-p 0.95, top-k 20).
- **Counting only the second pass of a cascade.** A cascade pays for the fast answer on every prompt and for the slow one on every escalated prompt.

## References

- Qwen Team, *Qwen3 Technical Report*, 2025, sections 4.3–4.4, Table 9, Table 22, Figure 2. https://arxiv.org/abs/2505.09388
- Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Apache-2.0), model card. https://huggingface.co/Qwen/Qwen3-1.7B
- OpenAI, *GPT-5 System Card*, 2025-08-13, section 1. https://cdn.openai.com/gpt-5-system-card.pdf
- OpenAI, *GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum*, 2025-11 (company claim; quoted from secondary coverage). https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/
- deepseek-ai/DeepSeek-V3.1, model card. https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- openai/gpt-oss-20b, model card. https://huggingface.co/openai/gpt-oss-20b
- The Register, *Alibaba admits Qwen3's hybrid-thinking mode was dumb*, 2025-07-31 (secondary source for the Qwen team's statement). https://www.theregister.com/2025/07/31/alibaba_qwen3_hybrid_thinking/
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The module project: [a post-training pipeline, stage by stage](../../projects/module-13-pipeline.md).
