---
id: "15.2"
module: 15
minutes: 40
practice_minutes: 110
prerequisites: ["15.1", "06.1", "02.4"]
objectives:
  - Prove that speculative sampling with the accept rule min(1, p/q) and the residual max(0, p - q) emits tokens distributed exactly as the target, and test it by Monte Carlo and by greedy equivalence.
  - Implement the verification step, cache rollback and three drafts (an independent small model, an EAGLE-style head on the target's features, a jointly trained MTP module) behind one interface.
  - Measure acceptance rate, acceptance by draft position, tokens per round and the draft cost ratio, and predict the speed-up from them with Leviathan et al.'s formula.
  - Explain when the formula's assumptions hold (memory-bound decode, batch 1) and why measured speed-ups on a CPU or at large batch fall short of it.
volatility: concept
sources:
  - title: "Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (Algorithm 1; section 3.1, Eq. 1; Definitions 3.1-3.2; Theorem 3.5, Corollary 3.6, Theorem 3.8; Table 2)"
    url: https://arxiv.org/abs/2211.17192
  - title: "Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (abstract)"
    url: https://arxiv.org/abs/2302.01318
  - title: "Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (sections 3.1-3.2)"
    url: https://arxiv.org/abs/2401.15077
  - title: "Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
    url: https://arxiv.org/abs/2406.16858
  - title: "Li et al. — EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test (sections 3.1-3.2; Tables 1 and 3)"
    url: https://arxiv.org/abs/2503.01840
  - title: "Cai et al. — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (section 2.1.2)"
    url: https://arxiv.org/abs/2401.10774
  - title: "DeepSeek-AI — DeepSeek-V3 Technical Report (section 2.2, Eqs. 21-25; section 5.4.3)"
    url: https://arxiv.org/abs/2412.19437
  - title: "vLLM v0.30.0 documentation — Speculative decoding"
    url: https://docs.vllm.ai/en/v0.30.0/features/speculative_decoding/
last_verified: "2026-10-07"
---

# 15.2 · Speculative decoding

Decoding one token at a time leaves a GPU mostly idle: each step reads every weight to do a few FLOPs per weight. Speculative decoding lets a cheap draft guess several tokens and the target check them all in one forward pass, and a rejection-sampling rule makes the output distributed exactly as if the target had sampled alone. This lesson proves that rule, implements it with cache rollback, and compares three kinds of draft on one target: a separate small model, an EAGLE-style head that reads the target's hidden state, and the target's own multi-token-prediction module from lesson 06.1. You measure acceptance, turn it into a predicted speed-up, and see where the prediction breaks.

## Why this matters at a frontier lab

Latency per token is a product number, and reasoning models multiply it by thousands of tokens per answer. Speculative decoding is the one decoding speed-up that changes no output: with the exact acceptance rule, a sampled answer has the target's distribution, and a greedy answer is token for token the target's greedy answer. That is why it is in every production engine (vLLM, SGLang, TensorRT-LLM) and why architecture teams now train their own drafts. DeepSeek-V3 trains a multi-token-prediction module with the model and reports that using it for speculative decoding accepts the second token 85–90% of the time, for 1.8 times the tokens per second (section 5.4.3). The decisions a research engineer owns are which draft to train, how many tokens to draft, and whether a reported speed-up will survive the serving batch size. All three need the acceptance rate and the cost ratio, measured, not assumed.

## The idea

### One verification round

The target distribution at a position is $p(\cdot)$, the draft's is $q(\cdot)$, both after the same prefix. One round (Leviathan et al., Algorithm 1):

1. The draft samples $d_1 \sim q_1$, then $d_2 \sim q_2(\cdot \mid d_1)$, ..., up to $\gamma$ tokens.
2. One target forward over the last accepted token and $d_1 \ldots d_\gamma$ gives $p_1, \ldots, p_{\gamma+1}$ (the last one is the distribution after $d_\gamma$).
3. For $i = 1, \ldots, \gamma$: draw $u_i \sim U(0, 1)$; accept $d_i$ if $u_i < \min(1, p_i(d_i)/q_i(d_i))$, otherwise stop.
4. At the first rejected position $j$, sample a token from the **residual** $r_j = \text{norm}(\max(0, p_j - q_j))$. If all $\gamma$ drafts were accepted, sample a **bonus** token from $p_{\gamma+1}$.

A round emits between 1 and $\gamma + 1$ tokens, at the cost of $\gamma$ draft steps and one target pass over $\gamma + 1$ tokens.

### Why the output is exactly the target's

Take one position. The token $x$ is emitted either because the draft proposed it and it was accepted, or because the draft was rejected and $x$ came from the residual:

$$P(\text{emit } x) = q(x)\min\!\Big(1, \frac{p(x)}{q(x)}\Big) + P(\text{reject})\, r(x) = \min(p(x), q(x)) + P(\text{reject})\, r(x).$$

The rejection probability is $P(\text{reject}) = 1 - \sum_y \min(p(y), q(y))$. The residual's normaliser is $Z = \sum_y \max(0, p(y) - q(y)) = \sum_y \big(p(y) - \min(p(y), q(y))\big) = 1 - \sum_y \min(p(y), q(y))$, the same number. So $P(\text{reject})\, r(x) = \max(0, p(x) - q(x))$, and

$$P(\text{emit } x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x).$$

For later positions the same argument holds conditioned on the accepted prefix: a draft at position $i$ is only examined if $d_1 \ldots d_{i-1}$ were accepted, and then $p_i$ and $q_i$ are the distributions after exactly that prefix. Every emitted token is therefore drawn from the target's conditional distribution given everything before it, which is the target's joint distribution over sequences. The draft changes only *how many* tokens a round emits.

Greedy decoding is the special case of one-hot distributions. Then $\min(1, p/q)$ is 1 if the draft equals the target's argmax and 0 otherwise, and the residual puts all its mass on the target's argmax. Greedy speculative decoding must reproduce plain greedy decoding token for token, which is the first check you run.

### Acceptance and the speed-up model

The probability that a draft from $q$ is accepted is $\beta = \sum_x \min(p(x), q(x)) = 1 - \text{TV}(p, q)$ (Leviathan et al., Definition 3.1, Theorem 3.5). Write $\alpha = E[\beta]$ over positions (Corollary 3.6). If acceptances were independent with probability $\alpha$, a round would emit

$$E[\text{tokens per round}] = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

tokens (section 3.1, Eq. 1). With $c$ the cost of one draft step relative to one target decode step, and the assumption that the target's pass over $\gamma + 1$ tokens costs one decode step, the expected walltime improvement is (Theorem 3.8)

$$\text{speed-up} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}.$$

That assumption is the whole economics. On a GPU at small batch a decode step is memory-bound: verifying 5 tokens reads the same weights as generating 1 and costs about the same time. On a CPU, or at a batch large enough to be compute-bound, a 5-token pass costs nearly 5 times a 1-token pass and the speed-up disappears.

### Three kinds of draft

- **An independent small model** with the same tokenizer (Qwen3-0.6B-Base for Qwen3-1.7B-Base on the main path). No training needed, but it must hold its own KV cache and it sees nothing of the target's computation.
- **An EAGLE-style head.** EAGLE (Li et al.) autoregresses on the target's features: its input is the target's top-layer feature at position $i$ concatenated with the embedding of token $i+1$, reduced by a fully connected layer, then one decoder layer that predicts the next feature (section 3.1). It is trained on the frozen target with $L = L_{\text{reg}} + w_{\text{cls}} L_{\text{cls}}$: a Smooth L1 loss between predicted and true features and a cross-entropy between the target's next-token distribution and the head's, with $w_{\text{cls}} = 0.1$ (section 3.2). EAGLE-2 makes the draft a tree whose shape follows the draft's confidence. EAGLE-3 drops feature prediction, predicts tokens directly from a fusion of low-, middle- and high-layer features, and trains with "training-time test", feeding the head its own predictions during training (sections 3.1–3.2).
- **The target's own MTP module.** DeepSeek-V3's module combines $\text{RMSNorm}(h_i)$ and $\text{RMSNorm}(\text{Emb}(t_{i+1}))$ through a $C \times 2C$ projection and one transformer block, and predicts $t_{i+2}$ with the shared output head (Eqs. 21–23). It is the same computation as an EAGLE head, trained jointly with the model instead of afterwards. Medusa's heads are the other common design: several heads on the last hidden state, each predicting one future position, verified with tree attention (section 2.1.2).

For the second and later drafts in a round, a hidden-state head feeds its own output feature back in place of the target's. Once the target has verified those positions, their real features replace the head's guesses. A head trained only to predict the next token (the MTP module) was never trained for that feedback; a head trained with feature regression (EAGLE) or training-time test (EAGLE-3) was.

## Worked example

**The residual by hand.** Vocabulary of three tokens, $p = (0.5, 0.3, 0.2)$, $q = (0.2, 0.5, 0.3)$. $\min(p, q) = (0.2, 0.3, 0.2)$, so $\beta = 0.7$. $\max(0, p - q) = (0.3, 0, 0)$, $Z = 0.3 = 1 - \beta$, $r = (1, 0, 0)$. Emitted distribution: $(0.2, 0.3, 0.2) + 0.3 \cdot (1, 0, 0) = (0.5, 0.3, 0.2) = p$. Draft token 2 (probability 0.5 under $q$, 0.3 under $p$) is accepted with probability $0.3/0.5 = 0.6$; when it is rejected, the residual sends the mass to token 1, where the draft under-proposed.

**Tokens per round.** $\alpha = 0.8$, $\gamma = 4$: $(1 - 0.8^5)/0.2 = (1 - 0.328)/0.2 = 3.36$ tokens per round. With $c = 0.05$ (a draft 20× cheaper): speed-up $3.36 / 1.2 = 2.80$. With $c = 0.5$: $3.36 / 3 = 1.12$. With $\gamma = 1$ and $c = 0.5$: $1.8 / 1.5 = 1.20$. A draft half as expensive as the target can only pay with few drafted tokens.

**DeepSeek-V3's number checked.** One MTP token per round ($\gamma = 1$), acceptance 0.85–0.90: $1 + \alpha = 1.85$–$1.90$ tokens per round. The report's 1.8× tokens per second is consistent with a small draft cost on top.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| target input per round | (1, γ + 1) | int64 | GPU (main) / CPU |
| target logits per round | (1, γ + 1, V); V = 8,192 toy, 151,936 Qwen3 | float32 (upcast) | same |
| target hidden state h | (1, S, C); C = 128 toy, 2,048 Qwen3 | float32 / bf16 | same |
| draft distributions q | (γ, V) | float32 | same |
| target KV cache per layer | (1, KV, S, d), cut back to the accepted prefix every round | float32 / bf16 | same |
| head's own cache | (1, KV, S, d) for one block | same | same |

Cost per round: $\gamma$ draft steps plus one target pass over $\gamma + 1$ tokens. Memory: the draft's weights and cache (an independent 0.6B model adds 1.2 GB of weights and its own 28-layer cache; a one-block head adds one layer). Main path (PROJECTED, pending the pilot): 64 GSM8K prompts × 128 new tokens × 4 values of $\gamma$ × 2 temperatures with the course's own loop on Transformers is under one GPU-hour; each vLLM configuration runs in minutes.

## Build it

```python
from frontierlab.ttc import speculative as SP, spec_models as SM

target = SM.ensure_target()            # toy preset + one DeepSeek-style MTP module (Module 6 trainer)
draft_lm, (head, _) = SM.ensure_draft(), SM.ensure_eagle(target)
for draft in (SP.LMDraft(draft_lm),
              SP.HiddenDraft(head, target.model.embed_tokens, target.lm_head),
              SP.HiddenDraft(target.mtp.layers[0], target.model.embed_tokens, target.lm_head)):
    out, stats = SP.speculative_generate(target, draft, prompt, max_new=64, gamma=4, temperature=1.0)
    stats["acceptance_rate"], stats["tokens_per_round"], stats["mean_overlap"]
```

`frontierlab/ttc/speculative.py` has the batched `accept_reject`, `residual`, `truncate_cache`, a `forward_hidden` that also returns the last layer's pre-norm output, the two draft classes, plain and speculative generation, the speed-up formulas and the measured cost ratio. `eagle.py` trains the head: one `DeepSeekModule` from `frontierlab.blocks.mtp` on the frozen target, with EAGLE's feature regression and a soft cross-entropy to the target's distribution. Correctness checks (`labs/common/tests/test_ttc.py`): the residual identity by hand; one-position Monte Carlo (200,000 draws, every token within 0.005 of $p$, acceptance rate equal to $\sum \min(p, q)$); a three-token Markov target and draft with $\gamma = 2$, all 27 sequences within total variation 0.01 of the exact joint; a tiny LM where two sampled tokens match the enumerated target joint; greedy speculative output identical to plain greedy for both draft kinds and $\gamma \in \{1, 3, 5\}$; cache truncation equal to a fresh prefix.

## What the evidence says

- **Speculative sampling: ESTABLISHED.** Proposed independently by Leviathan et al. (Google) and Chen et al. (DeepMind). Leviathan et al. report 2×–3× on T5-XXL against their T5X baseline (Table 2: 3.4× and 2.6× for translation at temperature 0 and 1, 3.1× and 2.3× for summarisation). Chen et al. report 2–2.5× on Chinchilla 70B in a distributed setup (abstract). Implemented in vLLM, SGLang and TensorRT-LLM. PUBLICLY DOCUMENTED.
- **Feature-level draft heads: ESTABLISHED in engines, PROMISING as a research line.** EAGLE: 2.7×–3.5× latency speed-up on LLaMA2-Chat 70B (abstract). EAGLE-2: 3.05×–4.26×. EAGLE-3: up to 6.5× (Vicuna 13B, HumanEval, temperature 0, acceptance length 7.54; Table 1), about 1.4× over EAGLE-2, and 1.38× throughput over SGLang *without* speculative decoding at batch size 64 on an H100 (Table 3), where EAGLE-1 gives 0.99×. The batch-64 figure is the one to remember: at large batch decode is no longer memory-bound and most of the gain is gone. Medusa: over 2.2× (Medusa-1) and 2.3–3.6× (Medusa-2) in the latest version of the paper. All these numbers are the authors' own benchmarks.
- **MTP as a draft: MODEL-SPECIFIC.** DeepSeek-V3 (85–90% second-token acceptance, 1.8× TPS; section 5.4.3) and Qwen3-Next ship MTP weights that engines use as drafts; for most open models a separately trained head is the option. Public EAGLE-3 heads exist for some open models, for example `AngelSlim/Qwen3-1.7B_eagle3` for Qwen3-1.7B (custom licence; its card reports an acceptance length of 2.17 and 643 vs 381 tokens/s, a company claim).
- **vLLM 0.30.0 mapping.** `speculative_config={"method": ..., "model": ..., "num_speculative_tokens": k}` with methods including `draft_model`, `eagle`, `eagle3`, `mtp`, `medusa` and `ngram` (`vllm/config/speculative.py` at the tag); the CLI flag is `--speculative-config '{...}'`. Acceptance is exported as `vllm:spec_decode_num_accepted_tokens` over `vllm:spec_decode_num_draft_tokens`, and the mean acceptance length is 1 + accepted / drafts. SGLang uses `--speculative-algorithm EAGLE3 --speculative-draft-model-path ... --speculative-num-steps --speculative-eagle-topk --speculative-num-draft-tokens`. REASONABLE INDUSTRY PRACTICE: tune $k$ at the batch size you serve, not at batch 1.
- **Course measurement (free CPU, 2026-10-07):** on the toy target, the EAGLE-style head trained for 400 steps on the frozen target beat the jointly trained MTP module, which beat the independent 1-layer LM (greedy acceptance at $\gamma = 1$: 0.80, 0.76, 0.57). The measured $\alpha$ matched the acceptance rate at $\gamma = 1$ to within 0.02, as the theory says it must. Wall-clock speed-ups on the CPU were between 0.6× and 1.07×, below the formula's 1.05–1.36×, because a CPU pass over 5 tokens costs much more than a pass over 1. Nothing here predicts the speed-up on an H100.

## Lab

**Folder:** [`labs/module-15/lesson-02/`](../../labs/module-15/) · **Time:** about 110 minutes (about 17 minutes unattended the first time) · **Pass check:** `pytest labs/module-15/lesson-02` passes; `spec_lab.py` reports that greedy speculative decoding equals plain greedy for every draft; your write-up explains, with your measured $c$ and $\alpha$, why your measured speed-up differs from the formula's.

### Experiment contract

- **Question:** for one target, which draft (independent LM, EAGLE-style head, jointly trained MTP module) gives the highest acceptance and the best speed-up, at which $\gamma$, at temperature 0 and 1? Decision informed: whether to train a head for the Stage D model or use a small model from the same family, and the $\gamma$ to serve with.
- **Hypothesis:** hidden-state drafts accept more than an independent model of similar size, and acceptance falls with draft position; the speed-up peaks at small $\gamma$ when $c$ is large. Status: reported (EAGLE, DeepSeek-V3 section 5.4.3); the speed-up hypothesis is expected to fail on a CPU, where verification is not memory-bound.
- **Baseline:** plain decoding with the same target and the same code, at batch 1.
- **Changed variable:** the draft, $\gamma \in \{1, 2, 4\}$, temperature $\in \{0, 1\}$. **Controlled:** the target checkpoint, 16 fixed validation prompts of 32 tokens, 64 new tokens, the per-prompt sampling seed, the threads.
- **Comparison axis:** wall-clock for identical outputs (greedy) or identically distributed outputs (sampled), at batch 1.
- **Budget:** free CPU, about 17 minutes the first time (target 800 steps, draft LM 800 steps, head 400 steps); main path under one GPU-hour (PROJECTED).
- **Metrics and decision rule:** acceptance rate and $\alpha$ per configuration; acceptance by draft position; speed-up as the median of paired per-round ratios with a bootstrap interval (`frontierlab.perf.speedup`, interleaved rounds). A draft "speeds up decoding" only if the interval lies above 1.
- **Correctness checks:** greedy equivalence for every draft and $\gamma$ (the script stops otherwise); `pytest labs/module-15/lesson-02`; `test_ttc.py`'s distribution tests.
- **Fallback evidence:** EAGLE-3 Tables 1 and 3 and DeepSeek-V3 section 5.4.3, labelled as published.
- **Limits:** a 2M-parameter target with a high-entropy next-token distribution; CPU timings; one head training run; no tree drafts.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 15 pilot | `--variant main --print`: the course loop on Transformers (target Qwen3-1.7B-Base `ea980cb`, draft Qwen3-0.6B-Base `da87bfb`, both with `DynamicCache.crop` rollback), then vLLM 0.30.0 with `draft_model`, `eagle3` (AngelSlim head on Qwen3-1.7B `70d244cc`) and `ngram`, at batch 1 and 32. MTP: rerun this lab with the Module 6 MTP model at pilot-30m on the GPU. **PROJECTED:** under 1 GPU-hour, USD 2–3 |
| Free GPU (Colab/Kaggle T4) | T4, fp16 | the same commands with `--prompts 16`; the T4's bandwidth-to-FLOPs ratio differs from an H100's, so record $c$ and the speed-up separately |
| Free CPU | laptop; measured 17 minutes the first time (training 15 minutes, other jobs running), 2 minutes after | the steps below |

### Steps

1. **Implement** the four TODOs in `lab.py` (residual, accept/reject, tokens per round, walltime improvement) and run `pytest labs/module-15/lesson-02`.
2. **Run** `python labs/module-15/lesson-02/spec_lab.py`. Confirm the correctness line before reading any other number.
3. **Acceptance.** For each draft at $\gamma = 4$, compare the measured tokens per round with $(1 - \alpha^5)/(1 - \alpha)$ from the measured $\alpha$. Which draft departs most from the independence assumption, and how does the "by position" column show it?
4. **Temperature.** Acceptance is higher at temperature 1 than at 0 for every draft. Explain why with $\beta = \sum \min(p, q)$, and say whether that makes sampled decoding faster in tokens per second.
5. **Speed-up.** Put your measured $c$ and $\alpha$ into the formula. Then explain why the measured speed-ups are lower, and what changes on an H100 at batch 1 and at batch 64.
6. **Decide** which draft you would train for the Stage D model and at which $\gamma$, and which measurement on the main path could change your mind.

<details>
<summary>Hint for TODO 2</summary>

`ok = u < clamp(p_d / q_d, max=1)`; the number of *leading* accepts is `ok.long().cumprod(-1).sum(-1)`. Pad `q` with a zero row at the bonus slot, so that `residual(p[n], q_padded[n])` is $p$ itself when every draft was accepted.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads, other jobs running). Target 2,066,496 parameters (MTP module 230,080), validation loss 5.72 after 800 steps; draft LM 573,696; EAGLE-style head 230,080, final top-1 agreement with the target's greedy choice 0.81. Greedy speculative decoding equalled plain greedy for every draft and $\gamma$.

| Draft | T | γ | acceptance | α | tokens/round | P(first i drafts accepted) |
|---|---|---|---|---|---|---|
| independent LM | 0 | 4 | 0.290 | 0.577 | 2.11 | 0.62 0.34 0.14 0.05 |
| EAGLE-style head | 0 | 1 | 0.796 | 0.802 | 1.78 | 0.80 |
| EAGLE-style head | 0 | 4 | 0.622 | 0.809 | 3.37 | 0.81 0.67 0.59 0.40 |
| MTP module | 0 | 4 | 0.533 | 0.763 | 3.05 | 0.73 0.58 0.48 0.34 |
| EAGLE-style head | 1 | 4 | 0.777 | 0.892 | 4.00 | 0.90 0.82 0.73 0.65 |
| MTP module | 1 | 4 | 0.727 | 0.837 | 3.82 | 0.87 0.79 0.69 0.54 |
| independent LM | 1 | 4 | 0.421 | 0.691 | 2.63 | 0.70 0.45 0.31 0.22 |

The formula with the measured $\alpha$ predicts 3.43 tokens per round for the head at $T = 0$, $\gamma = 4$, against 3.37 measured; for the independent LM it predicts 2.21 against 2.11. The MTP module loses more than the head at later positions because it was never trained to read its own output feature. Acceptance is higher at temperature 1 because the toy target's distributions are flat, and $\sum \min(p, q)$ of two flat distributions is large. That does not make sampling cheaper per useful token; it only means the draft matches a noisy target more often.

Cost ratios $c$: 0.22 (independent LM), 0.56 (head: one block plus the 8,192-wide output head, in a target of only four blocks) and 0.35 (MTP module). Predicted speed-ups at $T = 0$: 1.05–1.36. Measured, median of interleaved rounds with 95% intervals: between 0.63 [0.59, 0.64] (independent LM, $\gamma = 4$) and 1.07 [0.96, 1.81] (head, $\gamma = 2$); no interval lay wholly above 1 except the MTP module at $\gamma = 2$ (1.04 [1.01, 1.14]). On a CPU the target pass over $\gamma + 1$ tokens costs much more than one decode step, which breaks Theorem 3.8's assumption; Python overhead per round adds to it. On an H100 at batch 1 the assumption is close to true for a 1.7B model, which is what the main path measures.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-15/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-15/lesson-02`.

</details>

## Common mistakes

- **Accepting with probability $p(d)$ instead of $\min(1, p(d)/q(d))$**, or sampling the replacement from $p$ instead of the residual. Both change the output distribution; only the Monte Carlo test notices, because greedy decoding still looks right.
- **Counting accepts after the first rejection.** Only the leading run of accepted drafts is kept.
- **Not rolling back the cache.** Rejected drafts left in the target's cache corrupt every later position; cut the cache to the accepted prefix every round.
- **Reporting the speed-up at batch 1 for a service that runs at batch 64.** At large batch decode is compute-bound and the gain shrinks (EAGLE-3's 1.38× at batch 64).
- **Using a draft with a different tokenizer** as if it were the same. The vocabularies must match token for token, or the engine must translate (vLLM's `use_heterogeneous_vocab`).
- **Assuming acceptance is independent across positions.** Measure acceptance by position; hidden-state heads degrade with depth unless trained for their own feedback.

## References

- Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via Speculative Decoding*, 2022, Algorithm 1, section 3, Table 2. https://arxiv.org/abs/2211.17192
- C. Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023. https://arxiv.org/abs/2302.01318
- Y. Li et al., *EAGLE*, 2024, sections 3.1–3.2. https://arxiv.org/abs/2401.15077
- Y. Li et al., *EAGLE-2*, 2024. https://arxiv.org/abs/2406.16858
- Y. Li et al., *EAGLE-3*, 2025, sections 3.1–3.2, Tables 1 and 3. https://arxiv.org/abs/2503.01840
- T. Cai et al., *Medusa*, 2024, section 2.1.2. https://arxiv.org/abs/2401.10774
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, sections 2.2 and 5.4.3. https://arxiv.org/abs/2412.19437
- vLLM v0.30.0, *Speculative decoding*. https://docs.vllm.ai/en/v0.30.0/features/speculative_decoding/
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[15.3 · Serving cost of architecture choices](lesson-03.md)
