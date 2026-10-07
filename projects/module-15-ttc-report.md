# Module 15 project · A budget-matched test-time-compute report

This project turns the module into a recommendation someone could ship. A product team gives you a latency target and a compute budget per query for the Stage D reasoning model. You compare every way of spending that budget from lesson 15.1 (a longer chain, independent samples with majority vote, best-of-N and weighted vote with a verifier, PRM-guided search) at matched policy-token equivalents and matched latency, with the verifier's cost counted. You report the selection gap against oracle pass@N, apply a decision rule you wrote before the runs, and check that the recommendation survives a second sampling seed. You add the serving view from lessons 15.2–15.4: whether speculative decoding or a quantised cache changes the latency side of the answer. Before trusting the harness, you debug a colleague's version with three planted bugs.

**Time:** 5–7 attended hours plus unattended runs. **Folder:** [`labs/module-15/project/`](../labs/module-15/) (`run_project.py`, `harness.py`, `buggy_harness.py`, `test_harness.py`, `buggy_run.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model and task | Hardware | Cost |
|---|---|---|---|
| Main path | Qwen/Qwen3-1.7B (`70d244cc`) or your Module 14 RL checkpoint of Qwen3-1.7B-Base; GSM8K test (`740312a`), 500 questions; ORM Skywork-Reward-V2-Qwen3-1.7B (`e51ea3e`); PRM Qwen2.5-Math-PRM-7B (`0610740`); `python -m frontierlab.ttc.hf_ttc` with two sampling seeds and three latency targets | 1× H100 80 GB, vLLM 0.30.0 | **PROJECTED, pending the Module 15 pilot:** lesson 15.1's 2–4 GPU-hours plus a second `sample` and `score` pass (1–2 GPU-hours) and lesson 15.2's speculative runs (under 1): 4–7 GPU-hours, USD 8–21 at USD 2–3 per H100-hour. `python labs/module-15/project/run_project.py --variant main --print` prints the commands |
| Free GPU (Colab/Kaggle T4) | Qwen3-0.6B, 100 questions, 8 samples, no PRM arm | T4 | PROJECTED 2–3 hours per seed; results are cached per command, so a disconnect costs one command |
| Free CPU | the 15.1 toy world, 300 held-out problems, two sampling seeds | laptop | measured 77 s for both seeds once lesson 15.1's policy and verifiers exist (about 10 minutes otherwise); `buggy_run.py` seconds |

## Deliverables

1. **The experiment record:** your filled contract (below), the commands, the printed tables of both seeds, the latency-model check, the verifier's held-out AUC and error rates.
2. **The budget-matched table** at four budgets and three latency targets: the best procedure of each family with its success, 95% interval, spend and latency, and oracle pass@N beside every sampled row.
3. **The selection-gap analysis:** gap(N) for majority vote and for each verifier at N = 4, 16 and 32, the ceiling the verifier's false-positive rate predicts, and the cause of the difference.
4. **The recommendation** for the stated latency target and budget, by the pre-stated rule, with the paired comparison against the runner-up, and the same decision for a 40% tighter and a 2× looser target. Say whether both seeds agree.
5. **The serving paragraph:** using your lesson 15.2 and 15.4 numbers (or the published ones on the main path), would speculative decoding or an FP8/KIVI cache change the latency side of the recommendation, and at what batch size?
6. **The debugging report** and **the written defence** (below).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running. Fixed by the project:

- **Question:** under a latency target of T ms per query and a budget of B policy-token equivalents, which strategy maximises success on held-out problems, and is the choice stable across sampling seeds? Decision informed: the default inference strategy for the Stage D model in Module 16.
- **Hypothesis:** the answer depends on the verifier: with only a learned verifier of modest quality, a single chain or majority vote wins; with a reliable checker, best-of-N wins once the budget allows a few samples; tight latency targets exclude sequential search. Status: reported effects (Snell et al., Brown et al., Stroebl et al.); the toy and the main path may disagree.
- **Baseline:** one greedy chain at the cheapest reasoning effort with non-zero success.
- **Changed variable:** the strategy and its size. **Controlled:** policy checkpoint, verifiers (trained or pinned once), the problem set, temperature per mode, prompt format, hardware and engine version.
- **Comparison axis:** equal policy-token equivalents with the verifier counted and the prompt prefilled once; and equal latency from a latency model that you check against end-to-end timings.
- **Metrics and decision rule:** success per problem with a 95% bootstrap interval; the rule of `frontierlab.ttc.report.recommend` (best within the limits, but the cheaper procedure whenever the paired interval of the difference includes 0); stability = the same choice for both seeds.
- **Correctness checks:** `pytest labs/module-15/project` (the harness passes its derivation tests); `pytest labs/module-15/lesson-01`; `pytest labs/common/tests/test_ttc.py`; procedure success never above oracle pass@N; the latency model within a stated factor of the measured end-to-end time.
- **Fallback evidence:** Snell et al.'s budget-by-difficulty figures, Brown et al.'s coverage curves, the Hugging Face study's PRM search results, labelled as published.
- **Limits:** one policy; one task family; verifier quality; latency measured at batch 1 per query (no concurrent load); GSM8K's contamination risk for Qwen models on the main path.

<details>
<summary>What the build's free-CPU run gave (compare after your own report)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, other jobs running): `run_project.py` with seeds 0 and 1 on the cached 15.1 policy and verifiers, 77 s.

Selection gap, short mode, temperature 0.7, seed 1 (seed 0 in lesson 15.1's box): oracle pass@16 0.919, majority 0.798, best-of-16 with the learned ORM 0.590, weighted vote 0.804. The ORM's held-out false-positive rate was 0.515; the ceiling formula predicts about 0.78 for a random accepted candidate, and the argmax falls well below it.

Recommendations (the greedy short chain is 35 ms in the latency model):

| Latency target | budget 40 | budget 80 | budget 160 | budget 320 |
|---|---|---|---|---|
| 1.2× (42 ms) | greedy short chain, 0.797 | same | same | same |
| 2× (70 ms) | greedy short chain | same | best-of-8, rule checker, 0.875 / 0.886 (seeds 0 / 1) | best-of-8, rule checker (best-of-16 exceeds the latency) |
| 4× (140 ms) | greedy short chain | same | best-of-8, rule checker | best-of-16, rule checker, 0.899 / 0.908 |

Both seeds gave the same choice in every cell. Without the programmatic checker, the greedy short chain was the recommendation in every cell: no learned-verifier procedure beat it within any budget or latency target, and PRM beam search cost 445 pte or more for at most 0.753. The defensible report on this toy: "spend extra compute only if a reliable checker exists; with our learned ORM, sampling buys coverage the verifier cannot use".

The latency model was within a factor of 1.2 of the end-to-end time for two of three strategies in the second run and off by up to 1.7× in the first, while other processes were using the CPU: a CPU latency ranking within a factor of two is not a measurement you should ship.

</details>

## Debugging task

`labs/module-15/project/buggy_harness.py` is a colleague's evaluation harness. Their message is at the top of the file: best-of-4 "beats majority vote (0.80 against 0.76) and costs the same 84 tokens", and 16 parallel samples "take over half a second", so they recommend best-of-4. There are three planted bugs. For each one, name the rule of lesson 15.1 it breaks, the test that isolates it, and how it changed the colleague's conclusion.

```bash
HARNESS=buggy pytest labs/module-15/project       # the derivation tests against their harness
python labs/module-15/project/buggy_run.py         # their harness and the course's on lesson 15.1's cached pools
pytest labs/module-15/project                      # the course harness: all pass
```

<details>
<summary>Hint</summary>

Shuffle the `correct` flags of a candidate set and call each function again: a selection procedure must give the same answer. Then compute by hand what four samples and four verifier passes cost, and how many decode steps 16 parallel chains take.

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07 on the cached 15.1 pools (seed 0), scores rounded to one decimal as the colleague logs them:

| Harness | N | majority | best-of-N | best-of-N pte | parallel latency |
|---|---|---|---|---|---|
| colleague | 4 | 0.760 | 0.803 | 84 | 141.9 ms |
| colleague | 16 | 0.792 | 0.786 | 300 | 566.2 ms |
| course | 4 | 0.760 | 0.720 | 204 | 55.0 ms |
| course | 16 | 0.792 | 0.683 | 779 | 101.9 ms |

`HARNESS=buggy pytest labs/module-15/project` fails all three tests.

**Best-of-N breaks ties with the answer key.** With scores rounded to one decimal, ties are common, and the colleague's tie rule prefers the candidate known to be correct. That makes best-of-N a partial oracle: 0.803 at N = 4 instead of 0.720. Isolating test: `test_selection_never_reads_correctness` (permuting the correctness flags must not change the pick). It created the headline "beats majority vote".

**The verifier's tokens are not in the spend.** `spend_pte` returns prefill + decode, so best-of-4 "costs the same 84 tokens" as four samples. The verifier reads about 31 tokens per candidate with as many parameters as the policy: 204 pte at N = 4, 779 at N = 16. Isolating test: `test_verifier_tokens_are_paid_for`. It moved best-of-N into budgets it does not fit.

**Parallel samples are timed as if they were sequential.** `latency_parallel` multiplies the single-chain latency by N. Sixteen chains of 19 tokens decode as one batch in 19 steps, 102 ms in the latency model, not 566 ms. Isolating test: `test_parallel_samples_share_the_decode_steps`. It ruled out larger N against the latency target for the wrong reason.

With all three fixed, best-of-N with this ORM is worse than majority vote at every N, costs 2.4× the tokens of the samples it ranks, and the recommendation under a 70 ms target and an 84 pte budget is the greedy short chain. Each bug pushed in the same direction, which is why the conclusion looked so clean.

</details>

## Written defence

One to two pages, answering:

1. State your recommendation, the target and budget it is for, and the paired interval against the runner-up.
2. What is the selection gap at N = 16, and how much of it would a better verifier close? What evidence do you have for that, and what would you need?
3. Which latency numbers are measured and which come from the model? By how much did the model miss the end-to-end timings, and could that miss change your recommendation?
4. A product manager asks for "the method from the 14× paper". Explain what Snell et al. showed, on which problems, and why it may not apply here.
5. Would speculative decoding or a quantised cache change your latency column? At which batch size would the answer flip?
6. Each planted bug: which test caught it, how it changed the colleague's conclusion, and why all three pushed the same way.

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | latency target, budget and decision rule written before any run |
| 2 | Controls | same policy, problems, temperatures and verifiers for every strategy; verifiers trained only on training problems |
| 3 | Axis and budget parity | spend in policy-token equivalents with verifier tokens counted; latency from a model checked against timings |
| 4 | Correctness | the harness's derivation tests pass; no procedure above oracle pass@N; the three planted bugs found with their tests |
| 5 | Uncertainty | bootstrap intervals over problems; a paired comparison for the recommendation; a second sampling seed |
| 6 | Conclusion matches evidence | no "best-of-N" claim from pass@N; no latency claim from tokens alone |
| 7 | Limits | verifier quality, task family, batch-1 latency, contamination risk on the main path |
