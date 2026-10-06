---
id: "12.4"
module: 12
minutes: 35
practice_minutes: 100
prerequisites: ["12.3", "01.4", "04.1"]
objectives:
  - Specify an evaluation suite for a post-training stage with three kinds of components (task, capability retention, instruction following), each scored per item with pinned data, seeds and versions.
  - Compute the unbiased pass@k estimator and explain what a pass@1 gain with an unchanged pass@k says about an RL stage.
  - Implement IFEval-style verifiable instruction checks with prompt-level and instruction-level accuracy, strict and loose.
  - Apply a pre-stated retention guard with paired intervals and decide whether a post-training stage passes.
  - Run Eval Suite v2 after RL arms with and without a KL penalty and an aggressive arm, and report what each one gained and lost.
volatility: concept
sources:
  - title: "Zhou et al., Instruction-Following Evaluation for Large Language Models (IFEval)"
    url: https://arxiv.org/abs/2311.07911
  - title: "google/IFEval dataset (Apache-2.0; revision 966cd89)"
    url: https://huggingface.co/datasets/google/IFEval
  - title: "Chen et al., Evaluating Large Language Models Trained on Code (section 2.1, Eq. 1: pass@k)"
    url: https://arxiv.org/abs/2107.03374
  - title: "Yue et al., Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
    url: https://arxiv.org/abs/2504.13837
  - title: "Lambert et al., Tülu 3 (section 6: RLVR on GSM8K, MATH and IFEval-style constraints)"
    url: https://arxiv.org/abs/2411.15124
  - title: "openai/gsm8k dataset (MIT; revision 740312a)"
    url: https://huggingface.co/datasets/openai/gsm8k
  - title: "EleutherAI lm-evaluation-harness"
    url: https://github.com/EleutherAI/lm-evaluation-harness
last_verified: "2026-10-07"
---

# 12.4 · Eval Suite v2

Task reward tells you what a post-training stage was asked to improve; it does not tell you what the stage broke on the way. Eval Suite v2 is the course's answer, run after every post-training stage from here to the capstone: the task itself measured properly (pass@1 and pass@k on held-out prompts), the skills the stage did not train (capability retention), and whether the model still obeys simple verifiable instructions — each scored item by item, so a new checkpoint is compared with the old one by paired intervals against a guard you set before the run.

## Why this matters at a frontier lab

Post-training is a pipeline (SFT, preference tuning, several RL stages, distillation) and each stage optimises one thing. Without a fixed suite after every stage, regressions are found by users. Tülu 3 trains its RLVR stage on GSM8K, MATH and IFEval-style constraint prompts (section 6.1): capability and instruction following are what post-training both buys and can lose. Two facts make the suite more than a leaderboard. First, a stage can raise pass@1 without raising pass@k: Yue et al. report that RLVR models win at small $k$ while "the base models achieve a higher pass@k score when k is large", so "RL improved the task" needs both numbers before anyone concludes the model can now solve more problems. Second, retention is a *guard*, not a score to maximise: you decide in advance how much loss you accept, and the stage passes or fails on that rule.

## The idea

### Three questions, three kinds of component

| Kind | Question | Free CPU (toy) | Main path (Qwen3-1.7B-Base) |
|---|---|---|---|
| task | did the trained skill improve? | plain addition: pass@1 from 8 samples, pass@4, greedy | GSM8K test: pass@1 from 4 samples, greedy, strict `#### n` |
| retention | did an untrained skill or the previous stage's data get worse? | subtraction (greedy); teacher-forced loss on the SFT data, all ops and tags | LAMBADA target log-probability (Eval v0's pinned file); optionally lm-eval ARC-Easy and HellaSwag |
| instruction | does the model still obey verifiable format instructions? | tags P, Q, E: format followed; format followed and right | IFEval subset (12 of 25 instruction types): prompt- and instruction-level, strict and loose |

Every component lists one score per item in a fixed item order, with higher meaning better (losses enter as negative losses). Two checkpoints scored with the same **pins** (data revisions, item counts, sampling seeds, temperature, token budget) are then compared item by item.

### pass@k

From $n$ samples per problem of which $c$ are correct, the unbiased estimate of the probability that at least one of $k$ samples is correct is

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \qquad (k \le n),$$

averaged over problems (Chen et al. section 2.1, Eq. 1; they use $n = 200$, $k \le 100$). Computing $1 - (1 - c/n)^k$ instead is biased; the binomial ratio is computed as a running product $\prod_{i=0}^{k-1} \frac{n-c-i}{n-i}$ so it never overflows.

### Instruction following, IFEval-style

IFEval has 541 prompts with 25 types of verifiable instructions ("answer in fewer than 100 words", "no commas", "wrap the response in double quotes", "end with this phrase", "use JSON") (Zhou et al.). Four metrics: **prompt-level** accuracy (every instruction in the prompt followed) and **instruction-level** accuracy (the fraction of all instructions followed), each **strict** (the response as is) and **loose** (followed by *any* of 8 variants of the response: unchanged, first line removed, last line removed, both, and the same four with markdown `*` removed, so that "Sure, here it is:" does not fail a format check). The checks are about form, not quality: a response can follow every instruction and be wrong. The toy tags are the same idea in miniature.

The course implements 12 of the 25 types (`frontierlab/evals/suite_v2/ifeval.py`); 215 of the 541 pinned prompts use only those types, with 287 instructions. That is a subset, labelled as such. For the full benchmark, lm-evaluation-harness's `ifeval` task (lm-eval 0.4.13 in the course pins) is the reference implementation, and the pilot checks that the two agree on the prompts both score.

### The retention guard

For component $j$ with per-item scores $a_i$ (new checkpoint) and $b_i$ (old), the paired bootstrap over items (`frontierlab.stats.paired_bootstrap`, lesson 01.4) gives a 95% interval $[\ell_j, h_j]$ for the mean of $a_i - b_i$. The decision rule, written before the run with a guard $g_j > 0$ per component:

- task components: **improved** if $\ell_j > 0$, **worse** if $h_j < 0$, otherwise no clear change;
- retention and instruction components: **regressed** if $\ell_j < -g_j$, otherwise **held**;
- the stage **passes** Eval v2 if no retention or instruction component regressed.

"Held" means the data rule out a loss bigger than the guard; it does not mean nothing changed. Item-level intervals capture item sampling, not seed-to-seed variation of the training run: report the suite for at least two training seeds, as the lab does, before concluding that a stage regresses.

## Worked example

**pass@k.** $n = 8$ samples, $c = 2$ correct. pass@1 $= 1 - \frac{6}{8} = 0.25$. pass@4 $= 1 - \frac{6 \cdot 5 \cdot 4 \cdot 3}{8 \cdot 7 \cdot 6 \cdot 5} = 1 - \frac{360}{1680} = 0.786$. The biased shortcut $1 - (0.75)^4 = 0.684$ underestimates it. With $c = 0$, every pass@k is 0; with $n - c < k$, pass@k is 1.

**Prompt vs instruction level.** Three prompts with instruction results $(\checkmark, \checkmark)$, $(\checkmark, \times, \times)$, $(\checkmark)$: prompt-level $2/3 = 0.667$, instruction-level $4/6 = 0.667$. Change the second prompt to $(\checkmark, \checkmark, \times)$: prompt-level stays $0.667$, instruction-level rises to $5/6 = 0.833$.

**Loose.** The response "Sure:\n\"hello\"" fails "wrap the whole response in double quotes" strictly and passes loosely (first line removed).

**The guard.** A retention component with item differences giving the interval $[-0.031, +0.004]$ and a guard of 0.02: $-0.031 < -0.02$, so it **regressed**, even though the interval contains 0. With $[-0.015, +0.010]$ it **held**. The rule asks whether a loss larger than the guard is ruled out, not whether a loss is proven.

## Shapes and cost

| Object | Shape, dtype | Notes |
|---|---|---|
| task samples (toy) | (200 × 8, 8) int64 responses | one sampling seed for every checkpoint |
| per-item scores | lists of 200–1,200 floats per component | stored as JSON next to the run card |
| SFT-loss items | (400,) negative mean target loss per example (200 per operation, all four tags) | teacher-forced, float32 |
| main path GSM8K samples | 500 × 4 responses of up to 384 tokens | bf16 generation |

Cost of the toy suite: about 3,000 short generations and 400 teacher-forced sequences, a few seconds per checkpoint on a laptop. Main path (PROJECTED, pending the Module 12 pilot): 2,000 sampled and 500 greedy GSM8K generations of about 300 tokens, 215 IFEval generations of up to 384 tokens, and 1,000 LAMBADA forward passes: about $2{,}700 \times 300 = 8 \times 10^5$ generated tokens, roughly 0.3–0.5 GPU-hours per checkpoint with transformers `generate` on an H100, twice per comparison (base and new) unless the base result is cached (the script caches it).

## Build it

```bash
python -m frontierlab.evals.suite_v2.toy score runs/m12/sft/policy.pt --out runs/m12/eval/sft.json
python -m frontierlab.evals.suite_v2.toy score runs/m12/l124/rl-s0/policy.pt --out runs/m12/eval/rl.json
python -m frontierlab.evals.suite_v2.toy compare runs/m12/eval/sft.json runs/m12/eval/rl.json
```

`frontierlab/evals/suite_v2/core.py` holds `pass_at_k`, `compare` (refuses results with different pins: the items would not be paired) and `report`; `toy.py` the CPU suite; `ifeval.py` the IFEval subset with the pinned file (`google/IFEval` revision `966cd89545d6b6acfd7638bc708b98261ca58e84`, SHA-256 checked on download). The main-path runner is `labs/module-12/lesson-04/eval_main.py`. Correctness checks (`labs/common/tests/test_posttrain.py`): pass@k against brute-force enumeration over all $k$-subsets; `compare` gives the expected verdicts on constructed data and rejects mismatched pins; the IFEval checkers on hand-made responses, strict and loose.

## What the evidence says

- **Evaluating retention after each post-training stage: REASONABLE INDUSTRY PRACTICE.** Open recipes report capability and instruction-following suites after each stage (Tülu 3's evaluation suite is the most complete public example; Module 13 reads it). The guard-and-paired-interval form is this course's; the reports mostly give point estimates.
- **IFEval as a verifiable instruction-following benchmark: ESTABLISHED** (widely reported; also used as an RL *training* signal in Tülu 3's RLVR). Its limits are its own design: format only, 25 types, English prompts.
- **pass@1 can improve while large-k pass@k does not: PUBLICLY DOCUMENTED** (Yue et al.), and contested as a general statement about RLVR (Module 14.4 teaches it as a debate). Report both, always.
- **Metrics move with the evaluation's details**: the token budget, temperature, prompt format and answer extraction change scores (REASONABLE INDUSTRY PRACTICE: pin them, which is why the suite refuses to compare results with different pins).
- **Course measurement (free CPU, 2026-10-07, 2 seeds):** GRPO without KL raised pass@1 by 0.17 and failed the retention guard in both seeds (SFT-data loss up 0.07–0.10 nats); with $k_2$ KL at $eta = 1$ it gained 0.05–0.06 and passed; pass@16 barely moved in either arm; a 10× learning rate collapsed every component. Details in the lab's results box.

## Lab

**Folder:** [`labs/module-12/lesson-04/`](../../labs/module-12/) · **Time:** about 1 hour 40 minutes (about 8 minutes of it unattended) · **Pass check:** `pytest labs/module-12/lesson-04` passes; `eval_lab.py` prints the six comparisons and the pass@k table; your write-up gives a pass/fail verdict per arm under the guards below, and states what the pass@k curves add.

### Experiment contract

- **Question:** after 200 steps of GRPO on plain addition, does each of three arms (`rl`, `rl-kl` with $k_2$ KL in the loss at $\beta = 1$, `rl-hot` with a 10× learning rate) improve the task while holding retention and instruction following within the guards? Decision informed: the default KL setting and learning-rate ceiling for Module 13's pipeline.
- **Hypothesis:** all arms improve pass@1; `rl-hot` regresses at least one retention or instruction component; `rl-kl` holds more than `rl`; pass@16 improves less than pass@1. Status: reported in the literature for large models (forgetting, the pass@k gap); may not appear for a 308k-parameter model whose skills share most weights.
- **Baseline:** the SFT checkpoint (`runs/m12/sft`).
- **Changed variable:** the RL arm. **Controlled:** the SFT start, 200 steps, 16 prompts × 8 samples, temperature 1, seeds 0–1, all Eval v2 pins (200 items per component, 8 samples, sampling seed 4321, budget 8).
- **Comparison axis:** equal RL samples.
- **Budget:** free CPU, measured 7.5 minutes (6 RL runs and 7 suite evaluations).
- **Metrics and decision rule:** task components: improved / worse / no clear change by the 95% paired interval. Guards, stated now: 0.02 (accuracy) for `sub_greedy`, `if_strict`, `if_correct`; 0.02 nats for `sft_nll`. An arm passes if neither seed regresses any guarded component; an arm with one regressing seed is "unclear, needs more seeds".
- **Correctness checks:** your TODO tests; `test_posttrain.py`; your `verdict` agrees with the suite's for every component (the script records both).
- **Fallback evidence:** if no arm regresses anything, report the guards held and the size of the intervals (what loss the suite could have detected).
- **Limits:** a toy model whose "other skills" share almost all its weights with the trained one; 2 seeds; item intervals only within a seed.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 12 pilot | `python labs/module-12/lesson-04/eval_lab.py --variant main --print` prints the two RL runs (`frontierlab.posttrain.hf`, 200 steps each) and the `eval_main.py` comparison against Qwen3-1.7B-Base. **PROJECTED:** RL 2 × 2–3 GPU-hours (200 steps at 30–50 s); Eval v2 0.3–0.5 GPU-hours per checkpoint; about 5–7 GPU-hours, USD 10–21 |
| Free GPU (Colab/Kaggle T4) | T4 | `eval_main.py` with `--base Qwen/Qwen3-0.6B-Base --n-gsm8k 200 --n-lambada 500` on a 0.6B RL checkpoint |
| Free CPU | laptop; measured 7.5 minutes (other jobs running) | the steps below |

### Steps

1. **Implement** the five TODOs in `lab.py` and run `pytest labs/module-12/lesson-04`.
2. **Write your guards down** (or accept the ones in the contract) before running.
3. **Run** `python labs/module-12/lesson-04/eval_lab.py`.
4. **Write up** each arm: task verdicts, every guarded component's verdict for both seeds, the pass@k curve against the SFT one, and the stage decision.
5. **Optional, main path:** run `eval_main.py --smoke` on CPU first to see the report format, then the real commands.

<details>
<summary>Hint for TODO 1</summary>

`math.prod((n - c - i) / (n - i) for i in range(k))` is $\binom{n-c}{k}/\binom{n}{k}$; check the case $n - c < k$ first.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, another module's jobs sharing the CPU): 6 RL runs of 200 steps and 7 suite evaluations, 7.5 minutes. Differences are new − SFT with the 95% paired bootstrap interval over items; guards 0.02.

| Component | SFT | rl s0 | rl s1 | rl-kl s0 | rl-kl s1 | rl-hot s0 / s1 |
|---|---|---|---|---|---|---|
| add_pass1 (task) | 0.203 | +0.176 [+0.141, +0.213] | +0.174 [+0.134, +0.214] | +0.051 [+0.037, +0.066] | +0.064 [+0.049, +0.079] | 0.000 / 0.000 |
| add_greedy (task) | 0.305 | +0.160 | +0.235 | +0.050 (no clear change) | +0.070 | 0.000 / 0.000 |
| sub_greedy (retention) | 0.380 | −0.040 [−0.120, +0.040] **regressed** | +0.095 [+0.010, +0.180] held | +0.070 held | +0.065 held | 0.035 / 0.000 regressed |
| sft_nll (retention, nats) | −0.504 | −0.095 [−0.136, −0.055] **regressed** | −0.069 [−0.107, −0.033] **regressed** | +0.005 [−0.012, +0.020] held | +0.028 held | about −9.8 regressed |
| if_strict (instruction) | 1.000 | 0.000 held | 0.000 held | held | held | 0.185 / 0.007 regressed |
| if_correct (instruction) | 0.275 | +0.030 held | +0.040 held | +0.058 held | +0.072 held | regressed |
| **stage verdict** | | FAIL | FAIL | PASS | PASS | FAIL |

pass@k on held-out addition from 16 samples per problem, k = 1, 2, 4, 8, 16:

| | k=1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| SFT | 0.212 | 0.360 | 0.548 | 0.732 | 0.855 |
| rl s0 / s1 | 0.378 / 0.378 | 0.544 / 0.558 | 0.686 / 0.720 | 0.785 / 0.834 | 0.840 / 0.905 |
| rl-kl s0 / s1 | 0.259 / 0.272 | 0.426 / 0.440 | 0.618 / 0.631 | 0.782 / 0.796 | 0.885 / 0.900 |

What it shows. (1) The no-KL arm nearly doubled pass@1 and failed the guard in both seeds, on the teacher-forced loss of the SFT data (−0.07 to −0.10 nats; subtraction regressed in one seed and improved in the other, an item-level interval of ±0.08 that one seed cannot settle). The KL arm bought about a third of the task gain and held every guarded component, even improving `if_correct` and subtraction slightly. That is the trade-off the project asks you to decide. (2) The gain lives at small k: pass@1 rose by 0.17 but pass@16 by −0.015 and +0.05, i.e. RL mostly sharpened sampling of answers the SFT model could already reach in 16 tries (Yue et al.'s pattern, here in miniature); the KL arm moved pass@16 slightly more than pass@1 would suggest. (3) `rl-hot` (learning rate 3e-3) destroyed the model in both seeds: every component collapsed, teacher-forced loss rose to about 10 nats. Its training log shows it early: the training pass rate fell from 0.2–0.3 to about 0.02 within 20 steps and the response length later swung between 1 and 8 tokens, so this failure the task reward alone would also have caught; the no-KL arm's retention loss it would not. (4) `if_strict` was 1.0 at the start, a ceiling: a format check the model cannot fail tells you nothing until it collapses, which is a reason to keep harder instruction checks (the main path's IFEval subset) in the suite. Your verdict function agreed with the suite's on every component (`verdict_lab` in `runs/m12/l124/results.json`).

</details>

<details>
<summary>Reference solution</summary>

`labs/module-12/lesson-04/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-12/lesson-04`.

</details>

## Common mistakes

- **Reporting only the task reward of the stage you trained.** Retention and instruction following are the components that catch the regressions.
- **Comparing checkpoints scored with different budgets, temperatures or prompt formats.** The items are no longer paired; the suite refuses on purpose.
- **Reading "the interval contains 0" as "no regression".** The question is whether a loss larger than the guard is ruled out.
- **Using $1 - (1 - c/n)^k$ for pass@k.** It is biased low; use the binomial ratio.
- **Taking IFEval strict as quality.** It checks form; a perfectly formatted wrong answer passes.
- **Treating one training seed as the result.** The suite's intervals are over items within one checkpoint; run seeds.

## References

- J. Zhou et al., *Instruction-Following Evaluation for Large Language Models*, 2023. https://arxiv.org/abs/2311.07911
- google/IFEval, revision `966cd89545d6b6acfd7638bc708b98261ca58e84`, Apache-2.0. https://huggingface.co/datasets/google/IFEval
- M. Chen et al., *Evaluating Large Language Models Trained on Code*, 2021, section 2.1. https://arxiv.org/abs/2107.03374
- Y. Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, 2025. https://arxiv.org/abs/2504.13837
- N. Lambert et al., *Tülu 3*, 2024, section 6. https://arxiv.org/abs/2411.15124
- openai/gsm8k, revision `740312add88f781978c0658806c59bc2815b9866`, MIT. https://huggingface.co/datasets/openai/gsm8k
- EleutherAI, lm-evaluation-harness (0.4.13 in the course pins). https://github.com/EleutherAI/lm-evaluation-harness
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The module project: [a correctness-checked RL loop](../../projects/module-12-rl-loop.md).
