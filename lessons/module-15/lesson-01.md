---
id: "15.1"
module: 15
minutes: 40
practice_minutes: 120
prerequisites: ["14.4", "13.4", "12.4", "02.4"]
objectives:
  - Compare a longer single chain, independent sampling, majority vote, verifier best-of-N, weighted vote and PRM-guided beam search at matched budgets in policy-token equivalents and in wall-clock, with the verifier's tokens and latency counted.
  - Separate oracle pass@N (coverage) from the success of a selection procedure, and attribute the gap to the verifier's false positives.
  - Convert a strategy's structure (batch width, sequential steps, verifier calls on the critical path) into latency with a model built from measured step times, and check the model against end-to-end timings.
  - Apply a pre-stated decision rule (budget, latency target, paired interval) to choose a strategy, and say what would change the choice.
volatility: concept
sources:
  - title: "Snell, Lee, Xu, Kumar — Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (abstract; sections 3, 5.2-5.3, 6.2, 7)"
    url: https://arxiv.org/abs/2408.03314
  - title: "Brown et al. — Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (abstract; sections 2.1, 3.1, 4.1)"
    url: https://arxiv.org/abs/2407.21787
  - title: "Wang et al. — Self-Consistency Improves Chain of Thought Reasoning in Language Models (abstract)"
    url: https://arxiv.org/abs/2203.11171
  - title: "Cobbe et al. — Training Verifiers to Solve Math Word Problems (section 5.1, Figure 7; conclusion)"
    url: https://arxiv.org/abs/2110.14168
  - title: "Lightman et al. — Let's Verify Step by Step (section 3; abstract)"
    url: https://arxiv.org/abs/2305.20050
  - title: "Wang et al. — Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (section 3.3, Eqs. 3-4)"
    url: https://arxiv.org/abs/2312.08935
  - title: "Muennighoff et al. — s1: Simple test-time scaling (section 3.1; section 4.2; Figure 4)"
    url: https://arxiv.org/abs/2501.19393
  - title: "Stroebl, Kapoor, Narayanan — Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers (sections 3-4)"
    url: https://arxiv.org/abs/2411.17501
  - title: "Wu et al. — Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models (section 3.1.2; section 4.2; Table 1)"
    url: https://arxiv.org/abs/2408.00724
  - title: "Zhang et al. — The Lessons of Developing Process Reward Models in Mathematical Reasoning (Qwen2.5-Math-PRM)"
    url: https://arxiv.org/abs/2501.07301
  - title: "Beeching, Tunstall, Rush — Scaling test-time compute with open models (Hugging Face, 2024)"
    url: https://huggingfaceh4-blogpost-scaling-test-time-compute.hf.space/
last_verified: "2026-10-07"
---

# 15.1 · A test-time compute experiment

A reasoning model can spend extra inference compute in several ways: think longer in one chain, sample many chains and vote, let a verifier pick one, or search step by step with a process verifier. Papers compare these on different axes, often against an oracle that knows the right answer. This lesson sets up the comparison the way a deployment decision needs it: every method gets the same budget in policy-token equivalents, the verifier's tokens and latency are charged to the method that uses it, wall-clock is modelled from measured step times, and oracle pass@N is reported next to what each procedure actually selects. You run it on a toy world where every number can be checked, then on the Stage D reasoning model with vLLM.

## Why this matters at a frontier lab

Inference compute is now a design axis. OpenAI's o-series, DeepSeek-R1 and Qwen3's thinking mode spend thousands of tokens before they answer, and serving teams choose between a longer chain, more samples and a verifier for every product surface. The published evidence pulls in different directions. Snell et al. report that a compute-optimal mix of search and revisions is "more than 4x" more efficient than best-of-N, and that FLOPs-matched test-time compute on a small model can "outperform a 14x larger model" (abstract), but only on problems where the small model already has non-trivial success. Brown et al. show that coverage, the chance that at least one of k samples is right, keeps rising over four orders of magnitude of samples, while majority vote and reward models plateau (abstract). Stroebl et al. argue that a verifier with false positives caps what resampling can reach, whatever the budget.

All three statements can be true at once, because they measure different things. Coverage is not accuracy, a verifier is not free, and a method that wins at equal tokens can lose at equal latency. The common failure in internal reports is a plot of pass@k labelled "best-of-k" next to a single chain, with the verifier's cost and the latency left out. This lesson builds the comparison that survives a reviewer.

## The idea

### Five ways to spend a budget

| Strategy | What it spends per problem | Sequential steps (latency) | Needs |
|---|---|---|---|
| Longer single chain | one chain of length $L$; budget forcing caps or extends it | $L$ decode steps at batch 1 | a model whose accuracy grows with reasoning length |
| Independent samples, majority vote | $N$ chains of length $L$, one shared prompt | $L$ steps at batch $N$ | answers that can be compared (numbers, labels) |
| Best-of-N with an outcome verifier (ORM) | $N$ chains plus $N$ verifier passes | $L$ steps at batch $N$, then one verifier pass | a verifier that ranks right above wrong |
| Weighted vote | as best-of-N; sum verifier scores per distinct answer | as best-of-N | both of the above |
| PRM-guided beam search | $w \cdot m$ step candidates per step, a process-verifier call per step | one verifier call *between* every pair of steps | a process verifier (PRM) and a step format |

Self-consistency (Wang et al.) is the majority-vote row. Cobbe et al.'s GSM8K verifiers are the best-of-N row. The weighted vote and PRM-guided beam search are the strongest methods in Snell et al. (section 5.2) and in the Hugging Face reproduction with open models. "Longer single reasoning" is what s1's budget forcing controls: to cap a chain, the end-of-thinking delimiter is inserted; to extend it, the delimiter is suppressed and "Wait" is appended (s1 section 3.1). Qwen3 caps thinking the same way (lesson 13.4).

### The oracle and the procedure

From $n$ samples per problem with $c$ correct, the unbiased estimate of coverage at $k$ is

$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}$$

(Chen et al. 2021, the estimator of lessons 12.4 and 14.4). pass@k is an **oracle** number: it counts a problem as solved if any of the $k$ samples is right, which only someone holding the answer can know. A deployable procedure must pick one answer without the gold. Write $s_N$ for the success of a procedure applied to $N$ fresh samples. Then $s_N \le \text{pass@}N$ for every procedure, and the difference

$$\text{gap}(N) = \text{pass@}N - s_N$$

is the part of the coverage the procedure cannot turn into accuracy. For majority vote the gap comes from problems where the most frequent answer is wrong: more samples make the vote more certain, not more correct. For a verifier it comes from false positives.

A simple model makes the verifier's ceiling concrete. Suppose each sample is right with probability $p$, and a verifier accepts a right answer with probability $t$ (true-positive rate) and a wrong one with probability $f$ (false-positive rate). A procedure that returns a random accepted candidate succeeds, as $N \to \infty$, with probability

$$s_\infty = \frac{p\,t}{p\,t + (1-p)\,f}.$$

If $f > 0$, $s_\infty < 1$ however many samples are drawn. That is Stroebl et al.'s argument in one line. Best-of-N with the *highest* score is worse than this model suggests, because the argmax over many candidates selects exactly the wrong answers the verifier is most confident about. This is the over-optimisation Cobbe et al. saw: performance rose up to about 400 completions per problem and then fell, as the search found "adversarial solutions that fool the verifier" (section 5.1).

### Matched budgets: tokens, FLOPs and wall-clock

A token that passes through a model of $N$ parameters costs about $2N$ FLOPs, whether it is prefilled, generated with a cache or read by a verifier (Module 13's ledger). Divide by $2N_\pi$, the cost of one policy token, to put every strategy on one axis, the **policy-token equivalents** (pte):

$$\text{pte} = P + D + \frac{N_v}{N_\pi}\,V,$$

with $P$ the prefilled prompt tokens, $D$ the generated tokens, $V$ the tokens the verifier reads and $N_v$ its parameters. Two conventions must be stated. First, $N$ samples of one prompt share it. An engine with prefix caching (vLLM's automatic prefix caching, SGLang's RadixAttention) prefills it once, so $P$ is counted once. Second, a verifier reads the whole candidate, prompt included, and is usually *larger* than the policy: Qwen2.5-Math-PRM-7B scoring Qwen3-1.7B samples costs about four policy tokens per token it reads.

Equal FLOPs is not equal latency. $N$ chains of length $L$ decode as one batch, in $L$ sequential steps, and a decode step at small batch costs almost the same as at batch 1, because it is bound by reading the weights (lesson 15.3 works this out on the roofline). A chain $N$ times longer takes $N$ times as many steps. PRM-guided search puts a verifier call between every pair of steps. To compare strategies at a latency target, this lesson builds a latency model from measured parts: seconds per decode step at batch 1, 4, 16, 64 and 256, seconds per verifier pass at the same batch sizes, and seconds per prefilled token. A strategy's latency is the sum over its sequential structure. The model is checked against end-to-end timings, because a model nobody checked is a guess.

### The decision rule

Fixed before any run: among the procedures within the budget and the latency target, take the one with the highest mean success. Compare it, paired over the same problems, with every cheaper procedure, and if the 95% bootstrap interval of the difference contains 0, recommend the cheaper one (`frontierlab.ttc.report.recommend`). Oracle rows are reported and never recommended.

## Worked example

**Spend.** A prompt of $P = 12$ tokens, samples of 19 tokens, a verifier of the same size reading 31 tokens per candidate ($N_v/N_\pi = 1$).

- One short chain: $12 + 19 = 31$ pte.
- Four samples with a shared prefix: $12 + 4 \cdot 19 = 88$ pte. Without prefix caching: $4 \cdot 12 + 76 = 124$.
- Best-of-4 with the verifier: $88 + 4 \cdot 31 = 212$ pte. The verifier costs more than the samples it ranks, because it rereads the prompt and has as many parameters as the policy.
- One long chain of 43 tokens: $12 + 43 = 55$ pte.

**Latency.** Say a decode step takes 1.2 ms at batch 1 and 1.4 ms at batch 4, and a verifier pass over 4 candidates 2 ms. Four parallel samples: $19 \cdot 1.4 = 26.6$ ms; best-of-4: $26.6 + 2 = 28.6$ ms. The long chain: $43 \cdot 1.2 = 51.6$ ms. At equal latency the samples are cheap; at equal tokens the long chain is cheap.

**Oracle vs procedure.** Four samples of one problem give the answers 8442, 8442, 8432, 8442, with the third correct. Coverage: $c = 1$, $n = 4$, pass@4 = 1. Majority vote returns 8442 and fails. A verifier that scores the four candidates 0.71, 0.66, 0.64, 0.70 also fails: its false positive at 0.71 outranks the right answer. Only the oracle "solved" this problem.

**The ceiling.** With $p = 0.665$ (one sample's accuracy), $t = 0.94$ and $f = 0.515$ (the lab's outcome verifier at threshold 0.5), $s_\infty = 0.625 / (0.625 + 0.172) = 0.784$. Taking the highest score instead of a random accepted one can only do worse once the verifier's most confident errors dominate.

**Main-path costs.** Policy Qwen3-1.7B ($N_\pi \approx 2.03 \times 10^9$ counting its output matrix), PRM Qwen2.5-Math-PRM-7B ($N_v \approx 7.6 \times 10^9$, ratio 3.7). One search step scores 16 candidates of about 400 tokens: $16 \cdot 400 \cdot 3.7 = 23{,}700$ pte of verifier reading, against about $16 \cdot 60 = 960$ generated tokens for the step itself.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| prompts, repeated per sample | (P·N, 12) | int64 | CPU (toy) / GPU |
| policy logits per step | (P·N, 38) toy; (P·N, 151,936) Qwen3 | float32 | same |
| KV cache per layer | (P·N, 2, S, 24) toy; (P·N, 8, S, 128) Qwen3 | float32 toy / bf16 | same |
| verifier inputs | (M, ≤ 55) toy; (M, ≤ 2,048) main | int64 | same |
| verifier scores | (M,) | float32 | CPU after the pass |
| sample pool | P problems × n candidates (answer, correct, tokens, score) | Python records | CPU |

Toy cost: 300 problems × (32 + 16) samples ≈ 14,400 sequences of 19–43 tokens, under a minute of sampling on a laptop; the verifiers cost more to train (16,000 labelled samples each) than to run. Main path (PROJECTED, pending the pilot): 500 GSM8K questions × 32 samples × ~350 tokens ≈ $5.6 \times 10^6$ generated tokens in each sampling mode, plus single chains at five budgets (~$4 \times 10^6$ tokens), ORM scoring of 32,000 candidates (~$1.3 \times 10^7$ tokens read) and PRM search on 200 questions. At a few thousand generated tokens per second for a 1.7B model on one H100 with vLLM (lesson 15.3's roofline bound is about 6,800 tokens/s at 4K context and full batch), that is 2–4 GPU-hours, USD 4–12 at USD 2–3 per hour.

## Build it

```python
from frontierlab.ttc import world as W, select as SE, report as R
from frontierlab.ttc.budget import LatencyModel

pol = load_policy(W.ensure_policy("runs/m15/l151/policy.pt", steps=400))
orm, _ = W.ensure_verifier(pol, "orm", "runs/m15/l151/orm.pt")       # trained on training-problem samples
pool = W.sample(pol, probs, n=32, temperature=0.7)                     # per problem: answer, correct, gen_tokens
SE.oracle_pass_at_k(correct, 8).mean(), SE.subset_success(answers, gold, 8, "majority").mean()
rows = R.pool_rows("mode m", pool, gold, (1, 2, 4, 8, 16, 32), policy_params=Np, verifier_params=Nv,
                   prompt_tokens=12, latency=lm)
R.recommend(rows, budget=160, latency_target=0.07)
```

`frontierlab/ttc/` holds the pieces: `select.py` (votes, best-of-N, subset success, oracle pass@k), `budget.py` (spend and the latency model), `world.py` (the toy task, budget-forced sampling, ORM and PRM training, PRM beam search, step-time measurement), `report.py` (rows, families, the decision rule) and `hf_ttc.py` (the main path). The toy world is four-digit addition with a thinking trace in three reasoning efforts: none (`n`), a short column trace (`m`, `a12b14c14d08#8442`) and a long one that writes every digit and carry (`h`, `a3+9+0=12b8+5+1=14...`). The process verifier's labels are one-rollout Monte Carlo estimates in the sense of Math-Shepherd (section 3.3): every prefix of a sampled trajectory inherits its final correctness. Correctness checks (`labs/common/tests/test_ttc.py`): ties never break on correctness, a procedure never beats the oracle, $N = 1$ success equals the sample accuracy, pass@k against enumeration, budget forcing never exceeds its budget, spend arithmetic, the latency model's batching.

## What the evidence says

- **Majority vote (self-consistency): ESTABLISHED.** Wang et al. report +17.9% on GSM8K, +11.0% SVAMP, +12.2% AQuA (abstract), and the method is used in many later reports. PUBLICLY DOCUMENTED.
- **Verifier best-of-N: ESTABLISHED, with over-optimisation.** Cobbe et al.: a 6B model with verification "slightly outperforms a finetuned 175B model", about a 30× size increase (conclusion), improving up to 400 completions and then declining (section 5.1). Lightman et al.: process supervision reaches 78.2% on a 500-problem MATH subset with best-of-1860, trained on PRM800K's 800K step labels (section 3; abstract).
- **Coverage keeps rising; selection plateaus: PUBLICLY DOCUMENTED.** Brown et al.: SWE-bench Lite with DeepSeek-Coder-V2-Instruct goes from 15.9% with one sample to 56% with 250 (section 2.1), where unit tests act as a strong verifier; on GSM8K and MATH, majority vote and reward models plateau "beyond several hundred samples" (abstract; the section 4.1 text says around 100).
- **Imperfect verifiers cap resampling: PROMISING.** Stroebl et al. (sections 3–4) on HumanEval+ and MBPP+, with the optimal number of samples "often fewer than 10".
- **PRM-guided search vs best-of-N: PROMISING, budget-dependent.** Snell et al. (section 5.3): beam search "significantly outperforms best-of-N" at small budgets and often underperforms it at large ones, where it over-optimises on easy questions. Wu et al. (REBASE, section 3.1.2): Llemma-7B with tree search matches Llemma-34B at about half the FLOPs (section 4.2). The Hugging Face study reports that Llama 3.2 1B and 3B with PRM search beat Llama 3.1 8B and 70B on MATH-500 "given enough time to think" (blog, company claim, one benchmark). Zhang et al. report that Monte Carlo step labels, the kind this lab uses, make worse PRMs than LLM-judge or human labels.
- **Sequential scaling (budget forcing): PUBLICLY DOCUMENTED, one recipe.** s1-32B goes from 50% to 57% on AIME24 when extended with "Wait" (section 4.2). The authors argue sequential beats parallel because later tokens build on earlier ones, and show majority vote on the base model failing to match it (Figure 4).
- **The 14× claim is scoped.** Snell et al.'s FLOPs-matched win over a 14× larger model holds on problems where the small model has non-trivial success; on the hardest problems pretraining compute was the better buy (section 7). Plan item 22 records this scope.
- **Course measurement (free CPU, 2026-10-07; see the lab's results box):** on the toy world, majority vote never beat the greedy short chain, the learned outcome verifier made best-of-N *worse* as N grew, and only a programmatic checker turned coverage into accuracy. This is one toy task with one small verifier. It shows how to run the comparison; it says nothing about which method wins on GSM8K.

## Lab

**Folder:** [`labs/module-15/lesson-01/`](../../labs/module-15/) · **Time:** about 120 minutes (about 10 minutes unattended) · **Pass check:** `pytest labs/module-15/lesson-01` passes; `ttc_lab.py` prints all four tables; your write-up states the recommendation at your latency target with its paired interval, and the selection gap at N = 16 with its cause.

### Experiment contract

- **Question:** at a fixed budget per problem and a latency target, which strategy for spending inference compute gives the highest success, with the verifier's cost and latency counted? Decision informed: the default strategy for the Stage D reasoning model in Module 16's agent environments and in the project.
- **Hypothesis:** with a weak learned verifier, majority vote or a single longer chain beats best-of-N at every budget, and the selection gap grows with N; with a reliable verifier, best-of-N wins once the budget allows a few samples. Status: reported effects (Brown et al., Stroebl et al., Cobbe et al.); which side holds depends on the verifier, and may differ between the toy and the main path.
- **Baseline:** the greedy single chain in short-reasoning mode (`mode m`), the cheapest strategy with non-zero success.
- **Changed variable:** the strategy and its size (N, reasoning effort, beam width and expansion). **Controlled:** the policy checkpoint and its training seed, the 300 held-out problems (pairs never seen in training), temperature 0.7 for every sampled method, the verifiers (trained once on training-problem samples), the sampling seed.
- **Comparison axis:** equal policy-token equivalents (FLOPs, verifier included, prompt prefilled once) and, separately, equal latency from the measured latency model. Neither axis answers throughput at high load (lesson 15.3).
- **Budget:** free CPU, about 10 minutes the first time; main path 2–4 H100-hours (PROJECTED). Verifier training compute is reported separately (16,000 policy samples and 400 steps each); it is a fixed cost amortised over all queries, not a per-problem cost.
- **Metrics and decision rule:** success per problem, 95% bootstrap interval over problems; oracle pass@N beside every N; the decision rule above at budgets 20–640 pte and a latency target of 2× one greedy short chain.
- **Correctness checks:** `pytest labs/module-15/lesson-01`; `pytest labs/common/tests/test_ttc.py`; the latency model within a stated factor of end-to-end timings; your selection-gap table equal to the course's (`report.pool_rows`).
- **Fallback evidence:** Snell et al.'s Figures (beam search vs best-of-N by budget and difficulty), Brown et al.'s coverage curves, labelled as published.
- **Limits:** one policy seed and one task; a 308k-parameter verifier trained on Monte Carlo labels; CPU latency, where Python overhead is a large share of a step; the programmatic checker exists only because the toy task allows one.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB, vLLM 0.30.0. Not run in this build; part of the Module 15 pilot | `--variant main --print`: Qwen/Qwen3-1.7B (`70d244cc`) on 500 GSM8K test questions; 32 non-thinking samples and 32 thinking samples at a 512-token budget; single thinking chains at 256–4,096 tokens; ORM Skywork-Reward-V2-Qwen3-1.7B (`e51ea3e`); PRM beam search with Qwen2.5-Math-PRM-7B (`0610740`, Qwen licence, `trust_remote_code`: read the remote code at that revision first) on 200 questions; latency one question at a time. **PROJECTED:** 2–4 GPU-hours, USD 4–12 |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --print`: Qwen3-0.6B (pin its revision at the pilot) on 100 questions, 8 samples, budgets to 1,024; the 7B PRM does not fit next to it, so no search arm |
| Free CPU | laptop; measured about 10 minutes the first time (policy and two verifiers trained, other jobs running), about 30 s from the cache | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` (majority vote, weighted vote, policy-token equivalents, subset success) and run `pytest labs/module-15/lesson-01`.
2. **Run** `python labs/module-15/lesson-01/ttc_lab.py`. Read the verifier line first: AUC, false-positive and false-negative rates on held-out samples.
3. **The selection gap.** From your table, compute the gap at N = 16 for majority vote and for best-of-N. Use the ceiling formula with the measured rates to predict where best-of-N should level off, and explain why it falls below that.
4. **The full table.** Find the cheapest strategy that beats the greedy short chain beyond its interval. Does a longer chain (`mode h`) beat the short one? What does budget forcing at 9, 18 or 27 thinking tokens buy in this task, and why?
5. **The recommendation.** Rerun with `--latency-x 1.2` and `--latency-x 4`. Write down the recommendation at each target and budget, and the one sentence that explains each change.
6. **The latency model.** Compare the three model predictions with the end-to-end timings. Name one cost the model leaves out.

<details>
<summary>Hint for TODO 1</summary>

A plain `dict` keeps insertion order, and `max(d, key=d.get)` returns the first of several equal maxima. Skip `None` when counting.

</details>

<details>
<summary>Hint for TODO 4</summary>

Loop over problems in order, and for each one draw `resamples` index sets with `rng.choice(n, size=N, replace=False)` from one `rng` created before the loop. When `N == n` use `np.arange(n)` once. A pick of `None` is a failure even if the gold is somehow `None`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (Windows 11, torch 2.14.1 CPU, 8 threads, other modules' jobs running). Policy 308,400 parameters after 400 SFT steps; ORM and PRM 308,497 each.

Greedy single chains on 300 held-out problems: no thinking 0.000 (18.5 pte), short trace 0.797 [0.750, 0.840] (30.6 pte), long trace 0.813 [0.767, 0.857] (54.6 pte). Budget forcing inside the long trace at 9, 18 and 27 thinking tokens: 0.000, 0.017, 0.073. In this task an answer forced before the last column is a guess, so the sequential axis is all or nothing, and the long trace buys +0.016 for 1.8× the tokens and 2.3× the latency.

ORM on held-out short-mode samples: AUC 0.798, false-positive rate 0.515 and false-negative rate 0.061 at 0.5.

| N | oracle pass@N | majority | best-of-N (ORM) | weighted (ORM) | best-of-N (rule checker) |
|---|---|---|---|---|---|
| 1 | 0.667 | 0.665 | 0.665 | 0.665 | 0.665 |
| 4 | 0.862 | 0.759 | 0.693 | 0.768 | 0.844 |
| 16 | 0.917 | 0.793 | 0.587 | 0.796 | 0.899 |
| 32 | 0.930 | 0.793 | 0.527 | 0.797 | 0.910 |

Sampling at temperature 0.7 costs accuracy per sample (0.665 against greedy 0.797), and majority vote only wins it back: about 20% of problems have a wrong modal answer, so the vote saturates at 0.79 while coverage reaches 0.93. Best-of-N with the learned ORM gets *worse* with N: the argmax finds its most confident false positives, the Cobbe et al. pattern, well below the ceiling formula's 0.78. Weighted vote is safe but no better than majority. PRM beam search (width 2, expand 2) reached 0.753 at 445 pte, more than half of it verifier reading; wider beams did worse (0.650 at 4 × 4). Only a programmatic checker, which this task allows and most do not, turns coverage into accuracy: best-of-8 with it reaches 0.875 [0.840, 0.909] at 156 pte.

Decision rule at a latency target of 2× one greedy short chain (70 ms in this run's latency model): up to 80 pte the greedy short chain; from 160 pte best-of-8 with the rule checker; best-of-16 would fit the budget at 320 pte but not the latency target. With the learned ORM alone the recommendation would be the greedy short chain at every budget.

The latency model's predictions differ from end-to-end timings by up to a factor of 2 on a loaded laptop; on CPU, Python overhead per step and contention from other processes are a large share of each step. On the main path the same check uses `torch.cuda.synchronize()` and an idle GPU.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-15/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-15/lesson-01`.

</details>

## Common mistakes

- **Reporting pass@N as accuracy.** It is coverage. Put the procedure's success next to it, every time.
- **Leaving the verifier out of the budget.** A verifier as large as the policy rereads the prompt; on the toy it costs more than the samples it ranks, and a 7B PRM on a 1.7B policy costs about four policy tokens per token.
- **Breaking ties with the answer key.** Any tie rule that looks at correctness turns a vote into a partial oracle; the project's debugging task contains one.
- **Comparing at equal tokens only.** N parallel samples and one chain N times longer cost the same tokens and very different latency.
- **Training the verifier at a different temperature, or on test problems.** Its scores are calibrated for the samples it saw; train it on the policy's own samples at the deployment temperature, on training problems.
- **Taking the argmax of a weak verifier over many samples.** The more candidates, the more confident false positives; check the selection curve before raising N.

## References

- C. Snell, J. Lee, K. Xu, A. Kumar, *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, 2024, abstract and sections 3, 5.2–5.3, 6.2, 7. https://arxiv.org/abs/2408.03314
- B. Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, 2024, sections 2.1, 3.1, 4.1. https://arxiv.org/abs/2407.21787
- X. Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, 2022. https://arxiv.org/abs/2203.11171
- K. Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021, section 5.1. https://arxiv.org/abs/2110.14168
- H. Lightman et al., *Let's Verify Step by Step*, 2023, section 3. https://arxiv.org/abs/2305.20050
- P. Wang et al., *Math-Shepherd*, 2023, section 3.3. https://arxiv.org/abs/2312.08935
- N. Muennighoff et al., *s1: Simple test-time scaling*, 2025, sections 3.1 and 4.2. https://arxiv.org/abs/2501.19393
- B. Stroebl, S. Kapoor, A. Narayanan, *Inference Scaling fLaws*, 2024, sections 3–4. https://arxiv.org/abs/2411.17501
- Y. Wu et al., *Inference Scaling Laws*, 2024, sections 3.1.2 and 4.2. https://arxiv.org/abs/2408.00724
- Z. Zhang et al., *The Lessons of Developing Process Reward Models in Mathematical Reasoning*, 2025. https://arxiv.org/abs/2501.07301
- E. Beeching, L. Tunstall, S. Rush, *Scaling test-time compute with open models*, Hugging Face, 2024. https://huggingfaceh4-blogpost-scaling-test-time-compute.hf.space/
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[15.2 · Speculative decoding](lesson-02.md)
