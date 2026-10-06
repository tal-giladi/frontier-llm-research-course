# Module 12 project · A correctness-checked RL loop

Every later RL result in this course (Modules 13, 14 and 16) comes out of one loop, so the loop is the thing to get right first. This project assembles the RL loop for language models from the pieces of lessons 12.1–12.4, gives every detail of lessons 12.2 and 12.3 its own unit test, uses the loop for one pre-registered run with Eval Suite v2 before and after, and then debugs a colleague's version of the loop that trains "a bit worse" because of three planted bugs.

**Time:** 5–7 attended hours plus unattended runs. **Folder:** [`labs/module-12/project/`](../labs/module-12/) (`test_loop.py`, `buggy_loop.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model and task | Hardware | Cost |
|---|---|---|---|
| Main path | Qwen3-1.7B-Base (revision `ea980cb`), GSM8K (revision `740312a`), `python -m frontierlab.posttrain.hf`: 2 arms × 2 seeds × 200 steps; Eval v2 with `eval_main.py` | 1× H100 80 GB | **PROJECTED, pending the Module 12 pilot:** 4 runs × 200 steps × 30–50 s (lesson 12.2's formula) = 7–11 GPU-hours, plus Eval v2 0.3–0.5 GPU-hours per checkpoint (5 checkpoints): 9–14 GPU-hours, USD 18–42 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | Qwen3-0.6B-Base (revision `da87bfb`), `--prompts 8 --max-new 256`, fp32 | T4 | PROJECTED 3–5 hours per run; use the checkpointing of your loop and rerun after a disconnect |
| Free CPU | the toy policy and task, `python -m frontierlab.posttrain.rl` | laptop | measured: `test_loop.py` about 10 s; `buggy_loop.py` 95 s and `--fixed` 88 s (other jobs running) |

## Deliverables

1. **The loop.** Either your own, or `frontierlab.posttrain.rl` with your lab functions plugged in through its hooks (`advantages`, `loss_mask`, `policy_loss`, `old_logp`) as the lesson scripts do. It must log, every step: pass rate, zero-variance fraction, mean length of correct and wrong responses, truncation rate, entropy, a trustworthy KL to the reference (exact on the toy; $k_1$ and $k_3$ means on the main path), clip fraction, maximum ratio, sampler/trainer log-probability gap, gradient norm; and it must resume exactly from a checkpoint.
2. **One unit test per detail** (`labs/module-12/project/test_loop.py` is the reference set; adapt `pieces()` to your loop and add any detail your loop has that the reference does not):

   | Detail (lesson) | What the test asserts |
   |---|---|
   | baseline per group (12.2) | a response's advantage changes only when its own group's rewards change; mean-baseline advantages sum to 0 per group |
   | zero-variance groups (12.2) | an all-equal group gets exactly zero advantage, and is counted |
   | KL in the loss (12.2) | the chosen estimator's expected gradient equals the intended divergence's gradient (exact enumeration) |
   | KL in the reward (12.2) | the penalty is detached and equals $\beta \sum_t k_1$ |
   | entropy (12.2) | the logged entropy is the full-distribution entropy at each response position |
   | aggregation and length (12.3) | per-token weights under the chosen mode match the hand values |
   | truncation (12.3) | a response without EOS is fully masked and flagged unfinished; the overlong penalty matches DAPO Eq. 13 |
   | loss masking (12.3) | changing log-probabilities after EOS does not change the loss |
   | importance ratio (12.3) | on-policy, the first minibatch's ratio is exactly 1 |
   | staleness (12.3) | with staleness $k$ the sampler uses the snapshot from $k$ updates ago |
   | sampler mismatch (12.3) | the truncated IS weight is $\min(\pi_{\text{old}}/\pi_{\text{sampler}}, C)$ |

   Run `pytest labs/module-12/project` and include the output.
3. **A pre-registered run** (contract below): Eval Suite v2 on the start checkpoint, two arms × 2 seeds, Eval v2 on every final checkpoint, the comparison, the decision.
4. **The debugging report** (below).
5. **The written defence** (below).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running. Fixed by the project:

- **Question:** does KL regularisation in the form your lesson 12.2 analysis favours ($k_2$ in the loss or $k_1$ in the reward, your choice, stated) keep Eval v2's retention and instruction components within their guards at a task cost you can bound? Decision informed: the KL setting of Module 13's RLVR stage.
- **Hypothesis:** with your $\beta$, the KL arm holds every guarded component and the no-KL arm regresses at least one, at a task cost (pass@1) of less than 0.05. Status: reported trade-off for large models; may not appear at the toy scale (lesson 12.4 measured it there; cite your own numbers).
- **Baseline:** the no-KL arm with the same everything else.
- **Changed variable:** the KL term. **Controlled:** start checkpoint, prompts, group size, steps, learning rate, aggregation, clip range, seeds, every Eval v2 pin.
- **Comparison axis:** equal RL samples.
- **Metrics and decision rule:** Eval v2's rule with your guards (state them now); plus the per-seed final pass@1 difference with its t-interval.
- **Correctness checks:** every test of deliverable 2 passes; the loop resumes exactly (stop at half the steps, restart, compare metrics); the first on-policy ratio is 1 in the logs (`ratio_max` close to 1 when `minibatches=1` and `epochs=1`).
- **Fallback evidence:** published retention numbers from Tülu 3's evaluation tables, labelled as published.
- **Limits:** state the model size, steps, seeds and the fact that item-level intervals are within a seed.

## Debugging task

`labs/module-12/project/buggy_loop.py` is a colleague's refactor of three pieces of the loop. Their message is at the top of the file: training reward goes up "a bit", but it is slower than the course runs and "the clip fraction looks odd". There are three bugs. Start from the symptoms; for each, name the test or metric that isolates it, run it, and show what the fixed pipeline reports.

```bash
python labs/module-12/project/buggy_loop.py                # their pieces: 2 seeds x 120 steps
LOOP_HOOKS=buggy pytest labs/module-12/project             # the detail tests against their pieces
python labs/module-12/project/buggy_loop.py --fixed        # only after your diagnosis
```

<details>
<summary>Hint</summary>

Compare `clip_frac` and `ratio_max` of the buggy run with the lesson 12.3 `onpolicy` arm: with 2 minibatches the first minibatch of every step should be at ratio 1. Then read `grouped_advantages` with a (P, G) = (2, 3) tensor written out by hand: which rewards end up in row 0? For the third bug, the failing test's name is the hint.

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07 on the build laptop (CPU, other jobs running; `buggy_loop.py` 95 s, `--fixed` 88 s). The colleague's loop, 2 seeds × 120 steps: held-out sampled accuracy 0.197 and 0.197 (the SFT start is 0.21), training pass rate 0.23, clip fraction 0.25, maximum ratio 470 and $1.4 \times 10^7$, exact KL to the reference 0.005. The course loop with the same settings: 0.353 and 0.307, training pass 0.37, clip fraction 0.001, maximum ratio 1.17, KL 0.21. "Reward goes up a bit" was noise: their loop did not learn at all. `LOOP_HOOKS=buggy pytest labs/module-12/project` fails exactly four tests.

**Bug 1: rewards regrouped across prompts.** `grouped_advantages` reshapes the (P, G) rewards with `.reshape(-1).view(G, P).t()`, so "group" $p$ holds the rewards of responses from G different prompts, and each response is compared with other prompts' rewards. Isolating checks: `test_advantage_depends_only_on_own_group` and `test_zero_variance_group_has_no_gradient_signal` fail (an all-correct group gets non-zero advantages). Its symptom is diluted rather than dramatic: on its own (not run separately here) it adds the variance of prompt difficulty to every advantage instead of removing it, which slows learning without any metric looking wrong. Fix: `R` is already (P, G); pass it straight to the advantage.

**Bug 2: padding in the loss.** `response_loss_mask` returns ones for every generated position, so positions after EOS (PAD tokens sampled by nobody) are trained on. Isolating check: `test_padding_after_eos_does_not_change_the_loss` fails. In training it shows as the exploding `ratio_max` (the PAD positions' probabilities under the policy are not the sampler's at all) and higher entropy (0.53 against 0.35); lesson 12.3's `mask-with-pad` arm measured the same signature in isolation. Fix: use the mask from `response_mask`.

**Bug 3: old log-probabilities from the starting checkpoint.** `old_logprobs` loads `cfg.init` once and uses it as $\pi_{\text{old}}$ forever. The ratio is then $\pi_\theta/\pi_{\text{SFT}}$, and clipping at 0.8–1.2 *of the starting policy* blocks any token that has moved by more than 20% from where it started: the policy is pinned to the SFT model, which is why the exact KL stays at 0.005 and accuracy never moves. Isolating check: `test_on_policy_ratio_is_one_on_first_minibatch` fails; in the logs, a clip fraction of 0.25 from step 2 with `minibatches=2` and no staleness is impossible for a correct loop (the course loop: 0.001). Fix: $\pi_{\text{old}}$ is the policy that generated *this* batch (the course loop computes it from the sampler's snapshot).

Which detector found what: bugs 2 and 3 show in the training metrics (ratio, clip fraction, KL) if you know what a healthy loop logs; bug 1 is visible only through its unit test, and is the one a training curve would never reveal.


</details>

## Written defence

One to two pages, answering:

1. Which of your unit tests would have caught each planted bug, and which training metric showed each one first? Which bug would the training curve alone never have revealed?
2. Your KL arm: which estimator and placement did you choose, what gradient does it give in expectation, and how did you check that the logged KL is the quantity it controls?
3. Your loop's aggregation mode: what weight does a token of a long wrong response get relative to a short one, and why is that the right choice for Module 13's task?
4. What is the smallest regression your Eval v2 comparison could detect for each guarded component (the half-width of its interval), and is that small enough for the decision?
5. Which detail of your loop is still untested, and what would a test for it assert?
6. With 10× the budget, which single comparison would you repeat at the main-path scale first, and what result would change your loop's defaults?

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the KL question, guards and decision rule are written before any run |
| 2 | Controls | run cards show identical configurations except the KL term; Eval v2 pins identical for every checkpoint |
| 3 | Axis and budget parity | equal RL samples verified from the run cards; the reference forward is counted for both arms |
| 4 | Correctness | every detail test passes and its output is included; exact resume shown; the three planted bugs found, each with its isolating check |
| 5 | Uncertainty | item-level intervals per seed and a seed-level interval for pass@1 |
| 6 | Conclusion matches evidence | "holds retention" is claimed only for the guarded components and the measured scale |
| 7 | Limits | model size, steps, seeds, the toy's shared-weights caveat, the Qwen spurious-reward caveat for the main path |
