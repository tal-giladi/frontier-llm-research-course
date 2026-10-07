---
id: "16.3"
module: 16
minutes: 40
practice_minutes: 75
prerequisites: ["16.1", "12.2", "12.3", "14.2"]
objectives:
  - Write a multi-turn episode as one token sequence with action and observation masks, and derive why only sampled tokens belong in the policy gradient.
  - Implement outcome credit and turn-level return-to-go credit, and compute both by hand for a two-query episode.
  - Measure what training on observation tokens and a shaped per-turn bonus do to a toy agent, with paired seeds against a baseline arm.
  - Compute the prefill and peak-context cost of full-history and windowed context policies, and state what a context policy changes in training.
  - Read multi-turn RL reports (Search-R1, RAGEN, turn-level credit) for which of their choices a controlled comparison would isolate.
volatility: concept
sources:
  - title: "Jin et al., Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (abstract: retrieved token masking)"
    url: https://arxiv.org/abs/2503.09516
  - title: "Wang et al., RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning (StarPO, the Echo Trap, StarPO-S)"
    url: https://arxiv.org/abs/2504.20073
  - title: "Wei et al., Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment"
    url: https://arxiv.org/abs/2505.11821
  - title: "Zhou et al., MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents"
    url: https://arxiv.org/abs/2506.15841
  - title: "verl documentation, Agent Loop (AgentLoopOutput.response_mask: 1 for LLM-generated tokens, 0 for tool responses)"
    url: https://verl.readthedocs.io/en/latest/advance/agent_loop.html
last_verified: "2026-10-07"
---

# 16.3 · Multi-turn agentic RL

An agent episode is a conversation with an environment: the policy writes an action, the environment answers, the policy writes again, and the reward arrives at the end. This lesson writes such an episode as one token sequence with two masks, derives why the environment's tokens must stay out of the loss, and builds two ways to credit a final reward to the turns that earned it. It then measures, on a toy probe task, what the masking bug and a shaped per-turn reward do to a small policy, and computes what long trajectories cost in context.

## Why this matters at a frontier lab

Every agentic product trained with RL is multi-turn: software agents that read files and run tests, search agents, computer-use agents. Compared with single-turn RLVR, three things change. Part of the sequence is written by the environment. The reward is far from most of the decisions that produced it. And the context grows with every turn, so rollout cost and memory are set by the trajectory, not the answer. Each change has a cheap wrong implementation that trains without errors. Search-R1 made "retrieved token masking" a named part of its method because training on retrieved text destabilised RL (abstract). RAGEN reports a failure mode of multi-turn RL, the "Echo Trap", with reward-variance cliffs and gradient spikes. Its fixes (StarPO-S) filter trajectories and stabilise gradients. A research engineer has to know which of these details are part of the algorithm and which are part of the bug list.

## The idea

### One sequence, two masks

The probe task (`frontierlab/agents/turns.py`) hides a function $f$ from the program world of lesson 16.1. The agent may query it, `?3.` → `=6.`, at most twice, then answers with a program and EOS:

```text
BOS : | ? 2 . | = 4 . | ? 1 . | = 3 . | x + 2 EOS
       action 1  obs 1   action 2  obs 2   answer
```

`sample_multiturn` rolls out a batch in lock-step with one KV cache. Every row advances one token per step, and an inserted observation token counts as a step, so rows stay aligned. It returns `action_mask` (1 on tokens the policy sampled, including each `.` and the EOS), `obs_mask` (1 on inserted tokens) and `turn_index`. verl's agent loop returns the same thing as `response_mask` ("1 for LLM generated token, 0 for tool response token", PUBLICLY DOCUMENTED in its documentation).

### Why observations are not in the gradient

The probability of a trajectory $\tau$ under policy $\pi_\theta$ factors into the policy's choices and the environment's answers:

$$\log p_\theta(\tau) = \sum_{k \in \mathcal{A}} \log \pi_\theta(y_k \mid y_{\lt k}) + \sum_{k \in \mathcal{O}} \log P_{\text{env}}(y_k \mid y_{\lt k}),$$

where $\mathcal{A}$ is the set of action positions, $\mathcal{O}$ the observation positions and $y_{\lt k}$ everything before position $k$. The environment's term does not depend on $\theta$, so the policy gradient is

$$\nabla_\theta J = \mathbb{E}_\tau\Big[\sum_{k \in \mathcal{A}} A_k \, \nabla_\theta \log \pi_\theta(y_k \mid y_{\lt k})\Big].$$

Here $A_k$ is the advantage given to token $k$. Putting $\mathcal{O}$ into the loss adds $\sum_{k \in \mathcal{O}} A_k \nabla_\theta \log \pi_\theta(y_k \mid y_{\lt k})$. That term is not part of any policy gradient. It is supervised learning of the environment's text, weighted by the advantage: it teaches the policy to *write* tool outputs, more strongly after good episodes and in reverse after bad ones. Observations still matter as **context**: later actions condition on them. They are only excluded as **targets**.

### Credit over turns

With an outcome reward $R$ and group normalisation (lesson 12.2), **outcome credit** gives every action token of episode $i$ in a group of $G$ the same advantage:

$$A_i = \frac{R_i - \bar R}{\mathrm{sd}(R) + \epsilon}.$$

If turn $t$ also earns a shaped reward $r_t$ (here an information bonus: $b$ times the share of still-consistent functions that the query's answer rules out), **turn credit** uses the return-to-go from each turn,

$$G_{i,t} = r_{i,t} + \gamma\, G_{i,t+1}, \qquad G_{i,T_i} = R_i,$$

and normalises per turn index across the group: $A_{i,t} = (G_{i,t} - \bar G_{\cdot,t}) / (\mathrm{sd}_i(G_{i,0}) + \epsilon)$. Here $T_i$ is the answer turn, $\gamma$ the discount, $\bar G_{\cdot,t}$ the group mean over episodes that reached turn $t$, and every turn is divided by the same episode-level spread, so the turns of one episode share a scale (`agentrl.turn_advantages`). A bonus on turn $t$ then reaches turn $t$ and earlier ones, never later ones. Wei et al. compare terminal, delayed and per-turn reward structures with turn-level variants of GRPO and PPO and report that "dense per-turn reward structures consistently outperform" the sparse ones on their search and game tasks (PUBLICLY DOCUMENTED, one group). The catch is that a shaped reward is a **new objective**. A bonus for information-gathering queries pays for queries whether or not the answer improves. Whether that helps the final answer is an empirical question, and the lab measures it.

### Long trajectories and context

If action $t$ is generated from context $c_t$, an episode's prefill work is $\sum_t c_t$, and with full history $c_t = P + \sum_{s \lt t}(a_s + o_s)$, where $P$ is the prompt and $a_s$, $o_s$ the action and observation lengths. That grows quadratically in the number of turns, and the KV cache must hold the largest $c_t$. Two families of fixes exist:

- **context policies** that drop or summarise old turns (a window of the last $w$ turns; summaries written by the environment);
- **learned memory**: the policy itself writes a compact state each turn and the history is discarded. MEM1 trains this with RL and reports "3.5x" better performance and "3.7x" less memory than Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task (abstract; PROMISING, one group).

Either way, a context policy is part of the policy. The trainer must score each action under the context it was *sampled* from. Once old turns are dropped, an episode is no longer one sequence: it becomes one sequence per turn, and the one-forward-pass trick of the masks above stops working.

## Worked example

**Masks and the bug.** The episode above has 10 action tokens (`? 2 .`, `? 1 .` and `x + 2 EOS`: 3 + 3 + 4) and 6 observation tokens (`= 4 .` and `= 3 .`). With token-mean aggregation and advantage $A = 0.8$, each action token's surrogate term has weight $0.8/10 = 0.08$. With the masking bug the loss averages over 16 tokens: each action token drops to $0.8/16 = 0.05$, a 37.5% smaller step on the real decisions, and 6 observation tokens get weight 0.05 each. If the policy gives `=` a probability of 0.02 right after `?2.` (the environment, not the policy, writes it), the bug's gradient on that logit is $0.05 \times (1 - 0.02) = 0.049$ per such episode, pushing the policy to write `=` itself.

**Returns.** Bonus $b = 0.3$. Query 1 (`?2`, answer 4) leaves 2 of the 11 functions (`x+2`, `x*2`), removing $9/11$: $r_1 = 0.3 \cdot 9/11 = 0.245$. Query 2 (`?1`, answer 3) leaves 1 of 2: $r_2 = 0.3 \cdot 1/2 = 0.15$. The answer is right: $R = 1$. With $\gamma = 1$, $G = [1.395, 1.15, 1.0]$, and the episode's outcome-credit total is $0.245 + 0.15 + 1 = 1.395$ on every token. Take a second episode in the group that answered at once and wrongly: $G = [0]$. Turn 0 has mean $0.698$ and the spread of first-turn returns is $\mathrm{sd}(1.395, 0) = 0.986$. The first episode's turn-0 advantage is $(1.395 - 0.698)/0.986 = 0.707$ and the second's $-0.707$. Turns 1 and 2 exist only in the first episode, so their group mean is their own value and their advantage is 0. Turn credit gives the second query and the answer no push at all in this group. That is the price of normalising per turn index with small groups, and the reason implementations differ here.

**Context.** A SWE-like episode with $P = 3{,}000$, $a = 150$, $o = 1{,}200$ and 30 turns: $c_t = 3{,}000 + 1{,}350\,t$, so prefill $= 30 \cdot 3{,}000 + 1{,}350 \cdot 435 = 677{,}250$ tokens and peak context $3{,}000 + 1{,}350 \cdot 29 + 150 = 42{,}300$. A 3-turn window: $c_t = 3{,}000 + 1{,}350 \min(t, 3)$, prefill $203{,}400$ (3.3× less), peak $7{,}200$. At 100 turns the full history needs 6.98M prefill tokens and a 136,800-token context: beyond Qwen3-1.7B-Base's 32,768-token window, so a context policy is not optional there.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| `tokens` | (B, P + R), B = prompts × group = 128, P = 2, R = 16 | int64 | CPU (main path: GPU) |
| `action_mask`, `obs_mask` | (B, R) | float32 | same |
| `turn_index` | (B, R), -1 on PAD | int64 | CPU (bookkeeping only) |
| `sampler_logp`, trainer `logp` | (B, R) | float32 (main path: bf16 forward, float32 logits) | same |
| per-token advantages | (B, R) | float32 | same |

Measured on the warm start: an episode has about 0.8 queries, and 29% of its response tokens are observations. On the main path a SWE-style episode is dominated by observations (file contents, test logs), so the masking bug would put most of the loss on text the policy never wrote. Cost per step is rollout-bound. With $T$ turns, each turn is a separate generate call that waits for the slowest row, plus the environment's latency (a test run can take seconds to minutes). PROJECTED for the main path (Qwen3-1.7B-Base, 32 × 8 episodes, up to 3 turns of at most 12 new tokens): per step, 3 generate calls at an assumed 2–4 s each plus an update at an assumed 6–10 s, so 12–22 s per step and 0.5–0.9 H100-hours for 150 steps, plus evaluation. Formula: steps × (turns × generate latency + update time). The two latencies are assumptions until the Module 16 pilot measures them.

## Build it

```python
from frontierlab.agents import agentrl as R
from frontierlab.agents.turns import ProbeTask, sample_multiturn

ro = sample_multiturn(policy, [ProbeTask(f, max_queries=2) for f in funcs], max_tokens=16)
ro.action_mask, ro.obs_mask, ro.turn_index                    # (B, R) each
G = R.turn_returns(per_turn_rewards, finals, gamma=1.0)       # list of per-turn returns per episode
adv = R.token_advantages(R.turn_advantages(G, group=8), ro.turn_index)
R.train(R.AgentRLConfig(task="probe", credit="turn", info_bonus=0.3, loss_on="actions", run="runs/m16/x"))
```

The loop (`frontierlab.agents.agentrl`) reuses Module 12's pieces unchanged: teacher-forced log-probabilities, group advantages, the clipped loss with token-mean aggregation. Its own parts are the multi-turn sampler, the two credit rules and the probe-task metrics it logs each step: queries per episode, malformed actions (`bad_actions`), actions in which the policy wrote `=` itself (`self_obs`), turn-limit hits, and `obs_tokens_in_loss` (0 unless the bug arm is on). Correctness checks: `pytest labs/common/tests/test_agents.py -k "multiturn or probe or turn or context or outcome"`, which checks that inserted tokens equal the environment's observations and have no sampler probability, that the masks never overlap, and the returns, advantages and context counts by hand. The main-path version is `frontierlab.agents.hf_agent`. It cuts each turn at the first newline *keeping the sampled token ids*, because re-encoding decoded text can produce different tokens and the trainer would then score tokens the sampler never produced. Its `--smoke` run on CPU logs a trainer–sampler gap of about $5 \times 10^{-8}$, which shows the bookkeeping lines up.

## What the evidence says

- **Masking environment tokens out of the loss: ESTABLISHED** (Search-R1's retrieved-token masking; verl's `response_mask`; the derivation above).
- **Turn-level credit and dense per-turn rewards: PROMISING** (Wei et al., search and game tasks, one group). Whether the gain survives a shaped reward that can be farmed depends on the reward, as the lab shows.
- **Multi-turn instability and trajectory filtering: PROMISING** (RAGEN's Echo Trap and StarPO-S, four environments, one group).
- **Learned constant-size memory instead of full history: PROMISING** (MEM1, one group, QA tasks).
- **Course measurement (free CPU, 2026-10-07, 2 seeds × 60 steps per arm, 3.7 minutes in all):** from a warm start at 0.125 held-out accuracy, the baseline (outcome credit, actions only) reached 0.511 and 0.527. The masking bug reached 0.348 and 0.254, $-0.218$ $[-0.916, +0.480]$, with more turn-limit hits (0.11 against 0.00). The information bonus doubled queries (1.00 → 1.98 per episode) and *lowered* accuracy under both credit rules: outcome credit $-0.144$ $[-0.288, +0.000]$, turn credit $-0.136$ $[-0.184, -0.088]$. The policy learned to collect the bonus. With two seeds only the turn-credit interval excludes zero. Details in the lab's results box.

## Lab

**Folder:** [`labs/module-16/lesson-03/`](../../labs/module-16/) · **Time:** about 75 minutes (4 minutes of runs) · **Pass check:** `pytest labs/module-16/lesson-03` passes; `multiturn_lab.py` prints parts A–C; your write-up gives each arm's paired interval against the baseline with its behaviour metrics.

### Experiment contract

- **Question:** on the probe task, does training on observation tokens, or a shaped information bonus credited per episode or per turn, change held-out accuracy against outcome credit on actions only? Decision informed: the credit rule and mask for the module project's agent run.
- **Hypothesis:** the masking bug lowers accuracy; the bonus raises queries; turn credit beats outcome credit when the bonus is on. Status: masking is established; the credit effects are reported for other tasks (Wei et al.) and may not appear here.
- **Baseline:** outcome credit, loss on actions, no bonus.
- **Changed variable:** one of mask, bonus, credit rule per arm (the two bonus arms differ only in the credit rule). **Controlled:** the probe warm start (`runs/m16/sft-probe`, 400 SFT steps), 16 tasks × 8 episodes per step, 60 steps, lr $3 \times 10^{-4}$, 1 epoch × 2 minibatches, temperature 1, 2 queries at most, 16 tokens per episode, seeds 0 and 1, evaluation on 66 episodes with a fixed sampling seed.
- **Comparison axis:** equal episodes and updates.
- **Budget:** free CPU; measured 3.7 minutes (8 runs).
- **Metrics and decision rule:** final held-out accuracy (gold: the answer program is right on inputs 0–19), paired by seed against the baseline, 95% t-interval. "Better" or "worse" only if the interval excludes 0. Behaviour metrics (queries, malformed actions, `=` written by the policy, turn-limit hits, KL) are reported, not thresholded.
- **Correctness checks:** your TODO tests; part A's checks of your mask and token spreading against the loop on real episodes; `test_agents.py`.
- **Fallback evidence:** Search-R1, RAGEN, Wei et al., labelled as published.
- **Limits:** two seeds ($t_{0.975,1} = 12.7$), 60 steps, a 2-query toy task with 11 functions; the bonus size (0.3) is one value.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 16 pilot | `python labs/module-16/lesson-03/multiturn_lab.py --variant main --print` prints the 8 runs (`frontierlab.agents.hf_agent`, Qwen3-1.7B-Base, the same probe task in text form, 150 steps). **PROJECTED:** 0.5–0.9 GPU-hours per run plus evaluation, about 5–9 GPU-hours in all |
| Free GPU (Colab/Kaggle T4) | T4 | `--variant t4 --print` (Qwen3-0.6B-Base, 8 prompts per step, fp32); run two arms (baseline and masking bug) and say so |
| Free CPU | laptop; measured 3.7 minutes on the build laptop (Windows 11, Python 3.12.13, torch 2.14.1+cpu, 8 threads) | the steps below |

### Steps

1. **Write your contract** before running.
2. **Implement** the four TODOs in `lab.py` (`loss_mask`, `turn_returns`, `spread_over_tokens`, `context_cost`) and run `pytest labs/module-16/lesson-03`.
3. **Run** `python labs/module-16/lesson-03/multiturn_lab.py`. Part A checks your functions on real episodes; part B prints the context-cost table; part C runs the arms.
4. **Read the behaviour columns** before the accuracy column. For each arm, say which metric moved first, and whether the policy changed *what it does* (queries, `=` written by itself, turn-limit hits) or only how often it is right.
5. **Write up** (half a page): the decision table, the number of seeds you would need to resolve the masking-bug difference (lesson 14.2's worked example), and the context policy you would use for 100-turn SWE episodes on a 32K-context model, with what it changes in the trainer.

<details>
<summary>Hint for TODO 2</summary>

Append the final reward as one more "turn", then walk backwards: `acc = r + gamma * acc`. Reverse the list at the end.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (8 threads, another module's jobs running): 3 min 43 s for the warm start, the checks and 8 runs.

Part A: 44 warm-start episodes, 0.82 queries each, 305 sampled and 125 inserted tokens (29.1% observations); your functions match the loop on every episode.

Part C (held-out accuracy at step 60; warm start 0.125):

| Arm | per seed | vs outcome [95% CI] | queries | malformed | `=` by policy | turn limit | KL |
|---|---|---|---|---|---|---|---|
| outcome (baseline) | 0.511, 0.527 | — | 1.00 | 0.004 | 0.001 | 0.000 | 0.23 |
| all-tokens (masking bug) | 0.348, 0.254 | −0.218 [−0.916, +0.480] | 1.08 | 0.009 | 0.002 | 0.108 | 0.16 |
| outcome + bonus | 0.356, 0.394 | −0.144 [−0.288, +0.000] | 1.98 | 0.003 | 0.000 | 0.083 | 0.44 |
| turn + bonus | 0.371, 0.394 | −0.136 [−0.184, −0.088] | 1.98 | 0.002 | 0.000 | 0.131 | 0.38 |

Reading it under the rule: only the turn-credit bonus arm is "worse" with two seeds; the other two are inconclusive, though the masking bug lowered accuracy in both seeds by 0.16 and 0.27. The bonus changed behaviour, not skill: both bonus arms learned to use both queries in almost every episode (1.98) and moved furthest from the start (KL 0.38–0.44), and accuracy fell in both. Information is worth reward here whether or not the policy uses it, and at 60 steps it did not learn to use it. That is lesson 16.4's subject in a milder form: the shaped reward was optimised as written. Turn credit did not rescue the bonus. In this setting the bonus, not the credit rule, decides the result. The masking bug raised turn-limit hits to 0.11 and malformed actions slightly. The policy wrote `=` itself rarely at this scale (0.002), so the expected symptom of imitating the environment barely appeared in 60 steps.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-16/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-16/lesson-03`.

</details>

## Common mistakes

- **Loss over the whole response.** Mask observations; check `obs_tokens_in_loss` is 0 in the logs.
- **Re-tokenising decoded actions.** Keep the sampled ids; otherwise the trainer scores other tokens.
- **A shaped reward without a behaviour metric.** If you pay per tool call, log calls per episode and the final accuracy separately.
- **Windowed rollouts scored as one full sequence.** The trainer's context must be the sampler's context.
- **Per-turn normalisation with tiny groups.** Turns reached by one episode get advantage 0. Decide and state what you normalise over.
- **Reading two seeds as a result.** Report the interval and the seeds a decision would need.

## References

- B. Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, 2025, abstract and method. https://arxiv.org/abs/2503.09516
- Z. Wang et al., *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*, 2025. https://arxiv.org/abs/2504.20073
- Q. Wei et al., *Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment*, 2025. https://arxiv.org/abs/2505.11821
- Z. Zhou et al., *MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents*, 2025. https://arxiv.org/abs/2506.15841
- verl documentation, *Agent Loop*. https://verl.readthedocs.io/en/latest/advance/agent_loop.html
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[16.4 · Reward hacking](lesson-04.md)
