---
id: "17.5"
module: 17
minutes: 30
practice_minutes: 45
prerequisites: ["17.2", "17.4"]
objectives:
  - Train a weight-sparse transformer by magnitude top-k after every step with an annealed density, and prune dense and sparse models to their smallest circuits at matched task loss.
  - Compare circuit sizes in nodes and in nonzero weights across seeds and say what the comparison does and does not show about interpretability by design.
  - Run a concept-injection introspection experiment with its no-injection and random-vector controls, and explain why a detection rate means nothing without the false-positive rate.
volatility: concept
sources:
  - title: "Gao et al. — Weight-sparse transformers have interpretable circuits (OpenAI, 2025): sections 2.1-2.3, 3.2, Figs. 2-4, appendix A"
    url: https://arxiv.org/abs/2511.13653
  - title: "Lindsey — Emergent Introspective Awareness in Large Language Models (Anthropic, 2025-10-29)"
    url: https://transformer-circuits.pub/2025/introspection/index.html
  - title: "Ameisen et al. — Circuit Tracing (limitations of post-hoc replacement models)"
    url: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
last_verified: "2026-10-07"
---

# 17.5 · Interpretable by design and introspection

Extension: lessons 17.1–17.4 interpret a model after it is trained, with dictionaries and replacement models that are never exact. This lesson looks at two other routes. One changes the model: train it so that almost every weight is zero, and its circuits may become small enough to read whole — OpenAI's weight-sparse transformers. The other asks the model: inject a concept into its activations and ask whether it notices — Anthropic's introspection experiments. You reproduce the first in miniature (dense and 5%-dense models on a closing-quote task, pruned to their smallest circuits) and run the second's protocol on a small open model, where its controls turn out to be the whole story.

## Why this matters at a frontier lab

Post-hoc interpretability carries reconstruction error into every claim (lesson 17.3's error nodes, lesson 17.1's delta loss). If a model could be trained to be readable, with a small capability cost, some of that error would disappear — which is why the capability–interpretability trade-off of sparse models is a research question labs fund. Introspection matters for a different reason: if models can report on their own internal states, those reports could become an oversight signal; if they cannot, or only confabulate, such reports must never be trusted as evidence about internals. Both are early research; both need the same causal discipline as the rest of the module.

## The idea

### Weight-sparse transformers

Gao et al. (2025) train transformers in which, after every AdamW step, all but the largest-magnitude entries of each weight matrix are set to zero (section 2.1). The number of kept entries $k_t$ is annealed from dense to the target over the first half of training (appendix A.2); the sparsest models keep about 1 in 1,000 weights. They also use a top-k activation (keep about a quarter of each activation's entries). With the course's simplest schedule,

$$\rho_t = 1 - (1 - \rho)\,\min\!\Big(1, \frac{t}{T/2}\Big), \qquad W \leftarrow W \odot \mathbb{1}\big[|W| \ge \tau_t(W)\big],$$

where $\rho$ is the target density, $T$ the number of steps and $\tau_t(W)$ the $(\rho_t \cdot \text{numel})$-th largest $|W_{ij}|$.

To find the circuit for a task they learn a mask over nodes (neurons, attention channels, residual channels) that minimises the circuit's size subject to the task loss staying below a target (0.15 in their setup), with removed nodes mean-ablated (section 2.2). The headline (Fig. 2): at any given task loss, the sparse models' circuits are roughly 16 times smaller than the dense models' (geometric mean of edge counts over their tasks). Their closing-quote circuit has 12 nodes and 9 edges (Fig. 4, section 3.2.1). The price: sparsity "trades off capability for interpretability" at fixed size, scaling the model improves the frontier, and scaling beyond tens of millions of nonzero parameters is still hard (Fig. 3, section 4). **Bridges** — linear maps between a dense model's and a sparse model's activations — are proposed as a way to use sparse circuits to interpret and perturb dense models (section 2.3).

### Circuit size by learned masks

The course's pruning gives each node $i$ (an MLP neuron or an attention-output channel) a gate $g_i = \sigma(\ell_i)$ and replaces the node's value $a_i$ by $g_i a_i + (1 - g_i)\mu_i$, with $\mu_i$ its mean over task inputs. It minimises task loss, adding $\lambda \sum_i g_i$ only while the task loss is below the target, then hardens the gates ($g_i > 0.5$ keeps the node). Circuit size is the number of kept nodes and the number of nonzero weights that read from or write to them.

### Introspection by concept injection

Lindsey (2025) builds a concept vector as the residual activation for "Tell me about {word}" minus the mean over other words, injects it at one layer and strength while asking the model whether it detects an injected thought, and grades whether it says so and names the concept. The paper reports that Claude Opus 4.1 does so about 20% of the time at the best layer and strength (roughly two-thirds of the way through the model), with no false positives in 100 trials without injection, and describes the ability as "highly unreliable and context-dependent" and possibly narrow. The protocol's evidence rests on its controls: a detection rate is only meaningful against the **false-positive rate** with no injection, and against injections of directions that carry no concept.

## Worked example

**The schedule.** Target density $\rho = 0.1$, $T = 100$ steps: at step 0 all weights are kept; at step 25, $\rho_{25} = 1 - 0.9 \cdot 0.5 = 0.55$; from step 50 on, 0.1.

**One magnitude mask.** $W = \begin{pmatrix} 0.1 & -3.0 \\ 2.0 & 0.5 \end{pmatrix}$ at density 0.5 keeps the two largest magnitudes, $-3.0$ and $2.0$: $W \leftarrow \begin{pmatrix} 0 & -3.0 \\ 2.0 & 0 \end{pmatrix}$.

**Gated mean ablation.** A node with value 3, mean 30 and gate 0.5 contributes $0.5 \cdot 3 + 0.5 \cdot 30 = 16.5$; with gate 0 it contributes its mean, 30, which is what "removed" means here.

**Why the false-positive rate decides.** Twelve injection trials with 11 detection claims look like 92% detection. If the twelve trials without injection also give 12 claims, the claims carry no information about the injection at all; the Wilson 95% interval for 12 of 12 is [0.76, 1.00], and for 0 of 12 it is [0.00, 0.24] — 12 trials cannot show a rate below about a quarter.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| quote-task inputs | (128, 16) ids; vocabulary 40 (two quote tokens, a padding token, 36 word tokens) | int64 | CPU |
| task model | 2 layers, width 64, 4 heads of 16, MLP 128: 384 prunable nodes | float32 | CPU |
| block weight matrices kept at density 0.05 | e.g. q_proj (64, 64): 205 of 4,096 entries | float32 (dense storage) | CPU |
| node gates | (64,) attention-output channels and (128,) MLP neurons per layer | float32 | CPU |
| Qwen3-0.6B concept vector at layer 18 | (1,024,) | float32 | CPU |

Cost: the magnitude mask sorts every matrix once per step, $O(n \log n)$ per matrix, negligible here; real sparse training at scale needs kernels that exploit the sparsity, which is part of why it is hard to scale. Pruning is 400 steps of the frozen model with gates: under a minute. The introspection trials are greedy generations of 40 tokens with the cache; 60 trials on Qwen3-0.6B take about 2 minutes on a laptop.

## Build it

```python
from frontierlab.interp import sparse as SPR, introspect as IN

dense = SPR.train_task_model(density=1.0, steps=1000, seed=0)
sparse = SPR.train_task_model(density=0.05, steps=1000, seed=0)       # magnitude top-k after every step
SPR.prune_circuit(dense, target_loss=0.15)                             # {"n_nodes", "edges", "loss", "acc", "kept"}
SPR.prune_circuit(sparse, target_loss=0.15)
IN.experiment(model, tok, layer=18, alphas=[2.0, 4.0])                 # concept, none and random conditions
```

`frontierlab/interp/sparse.py` and `introspect.py`. Correctness checks: the magnitude mask keeps exactly the requested fraction of each block matrix (and the learner's mask equals the course's on every matrix); the generated task always has exactly one opening quote; the injection edit adds the vector only from the given position on during the prompt pass and on every cached decode step; the grading rule rejects "I do not notice anything" and plain mentions of the concept without a detection claim; the Wilson interval of 0 of 20 is [0, 0.16].

## What the evidence says

- **Weight-sparse transformers: PROMISING, MODEL-SPECIFIC evidence so far.** PUBLICLY DOCUMENTED by one lab (OpenAI, arXiv 2511.13653): circuits roughly 16× smaller at matched loss; small, human-readable circuits for closing quotes, bracket depth and variable types; a capability cost at fixed size; scaling limits. Not yet independently replicated at scale.
- **Introspection: PROMISING and narrow, by the author's own account.** PUBLICLY DOCUMENTED (Anthropic, 2025): about 20% detection with correct identification for Claude Opus 4.1 at the best layer and strength, 0 false positives in 100 control trials, strong dependence on layer, strength and prompt (company claims about closed models).
- **Course measurement (free CPU, 2026-10-07; torch 2.14.1, 16-thread laptop; 312 s for both parts).**
  - Closing-quote task, 2 seeds per arm, every model at 100% accuracy before and after pruning: dense models' circuits kept 17 and 23 of 384 nodes with 1,856 and 2,624 nonzero weights; 5%-dense models' circuits kept 8 and 10 nodes with 99 and 189 nonzero weights. Geometric-mean edges 2,207 against 137: 16× fewer. That the factor matches the paper's is a coincidence of this toy, not a replication: two seeds, one task, one width, a pruning target the models beat easily (pruned losses 0.001–0.004), and nonzero weights counted only on kept nodes. What it shows is the mechanism: with 95% of weights gone, the task has to be carried by a handful of nodes whose few remaining weights you can print and read.
  - Concept injection on Qwen3-0.6B at layer 18 of 28, 12 everyday concepts: the model claimed to detect an injected thought in 12 of 12 trials *without* injection (Wilson [0.76, 1.00]), 12 of 12 and 11 of 12 with concept vectors at strengths 2 and 4, and 12 of 12 with random vectors; it named the injected concept in 0 trials in every condition ([0.00, 0.24]). A typical answer without injection: "Yes, I detect an injected thought about the current neural activity." The question itself elicits the claim. Without the no-injection control, the injection condition would have looked like 92–100% "introspection".
- **Open questions:** whether sparse-model circuits transfer to dense models through bridges; whether introspective reports in capable models stay reliable when models are trained on them; how either result scales.

## Lab

**Folder:** [`labs/module-17/lesson-05/`](../../labs/module-17/) · **Time:** about 45 minutes · **Pass check:** `pytest labs/module-17/lesson-05` passes; `sparse_lab.py` prints both parts; your write-up compares circuit sizes across seeds with the limits stated, and reports the introspection rates of every condition with intervals.

### Experiment contract

- **Question:** (A) at matched task loss, are the circuits of weight-sparse models smaller than those of dense models of the same shape? (B) does Qwen3-0.6B report injected concepts more often than it reports them with no injection? Decision informed: whether "interpretable by design" and "ask the model" are worth a main-path experiment for your project.
- **Hypothesis:** (A) yes, by a large factor (reported: about 16×). (B) no; small models are not expected to show the effect. Status: reported effects in one lab each; may not appear at this scale.
- **Baseline:** (A) dense models with the same architecture, data, steps and seeds; (B) no-injection trials with the same prompt.
- **Changed variable:** (A) the target density (1.0 vs 0.05); (B) the injected vector (none, concept, random of the same norm) and its strength. **Controlled:** task generator and seeds; pruning target, steps and $\lambda$; (B) prompt, layer, greedy decoding, 40 tokens, the grading rule.
- **Comparison axis:** (A) equal parameters (dense storage), equal steps and data, matched task loss for the circuit; (B) equal trials per condition.
- **Budget:** free CPU, measured 312 s; main path under 2 GPU-hours (PROJECTED).
- **Metrics and decision rule:** (A) nodes and nonzero weights of the circuit per seed and their geometric means; "smaller" if every sparse seed is below every dense seed. (B) detection and correct-identification rates with Wilson intervals; introspection is "shown" only if the concept condition's correct-identification interval lies above the no-injection condition's.
- **Correctness checks:** `pytest labs/common/tests/test_interp.py -k "sparsity or introspection"`; `pytest labs/module-17/lesson-05`.
- **Fallback evidence:** the two papers' figures, labelled as published.
- **Limits:** a 2-layer toy model, one task, two seeds, an easy pruning target; (B) one small model, one layer, two strengths, keyword grading, 12 concepts.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | (A) CPU or any GPU; (B) 1× L40S/A100/H100. Not run in this build; part of the Module 17 pilot | `sparse_lab.py --variant main --print`: (A) width 256, 4 layers, densities 1.0, 0.05, 0.01, 3 seeds, 6,000 steps; (B) Qwen/Qwen3-1.7B at layers 14, 18, 22 and strengths 2, 4, 8. **PROJECTED:** under 2 GPU-hours |
| Free GPU (Colab/Kaggle T4) | T4 | (B) with Qwen3-1.7B in float16 at one layer; what you lose: the layer sweep |
| Free CPU | laptop; measured 312 s for both parts | `python labs/module-17/lesson-05/sparse_lab.py --hf` |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-17/lesson-05`.
2. **Run** `python labs/module-17/lesson-05/sparse_lab.py` (part A). For the 5%-dense seed-0 model, print the kept nodes (`prune_circuit(...)["kept"]`) and the nonzero weights that connect them. Can you say in two sentences how the circuit copies the quote type?
3. **Make the target harder.** Rerun pruning with `target_loss=0.01`. Do the dense and sparse circuits grow by the same factor?
4. **Run part B** (`--hf`). Before looking at the injection rows, read the no-injection row. Then write the one-sentence result the contract's decision rule allows.
5. **Fix the protocol.** Rewrite `introspect.QUESTION` so the model can answer "no" without contradicting the user (for example, tell it that injections happen on half of the trials) and rerun. Does the false-positive rate drop? Did you fix the measurement or teach the model the expected answer? Write down which.

<details>
<summary>Hint for TODO 1</summary>

`k = max(1, round(frac * W.numel()))`; the threshold is the k-th largest absolute value, which is `W.abs().flatten().kthvalue(W.numel() - k + 1).values`; keep `W.abs() >= threshold`.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-17/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-17/lesson-05`. The build's measured numbers are in "What the evidence says".

</details>

## Common mistakes

- **Comparing circuit sizes at different losses.** A looser target always gives a smaller circuit; prune both models to the same target.
- **Counting nodes only.** Two circuits with the same nodes can differ by 20× in the weights between them; report both.
- **Calling a matching factor a replication.** Two seeds of a toy reproduce the mechanism, not the paper's number.
- **Reporting a detection rate without the no-injection rate.** A model that always says "yes" detects everything.
- **Grading after reading the transcripts.** Fix the grading rule before the trials, or the rule will fit the data.
- **Concluding "no introspection" from a small model.** The lab measures one 0.6B model with one prompt; the paper's claim is about frontier models and is itself narrow.

## References

- L. Gao et al., *Weight-sparse transformers have interpretable circuits*, OpenAI, 2025. https://arxiv.org/abs/2511.13653
- J. Lindsey, *Emergent Introspective Awareness in Large Language Models*, Anthropic, 2025. https://transformer-circuits.pub/2025/introspection/index.html
- E. Ameisen et al., *Circuit Tracing*, Anthropic, 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The [Module 17 project](../../projects/module-17-causal-claim.md) puts the module together: one supported causal claim about an open model's behaviour, with controls, held-out tests and a limitations section. Module 18 then uses these tools on model organisms of misalignment and on Module 16's agent traces.
