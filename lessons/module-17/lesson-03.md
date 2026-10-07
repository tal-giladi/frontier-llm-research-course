---
id: "17.3"
module: 17
minutes: 40
practice_minutes: 70
prerequisites: ["17.1", "17.2"]
objectives:
  - Train a transcoder for each MLP and measure the replacement model it defines by FVU, next-token loss and top-1 agreement with the original model.
  - Build the local replacement model (frozen attention patterns and norm denominators, error nodes) and verify that it reproduces the model's logits exactly.
  - Compute an attribution graph from scratch, check that edges into every node sum to its input, prune it by influence on the logits, and report the error-node share.
  - Treat a graph as a hypothesis: compare its predicted intervention effects with real interventions in the original model, against random-feature controls.
volatility: concept
sources:
  - title: "Dunefsky, Chlenski and Nanda — Transcoders Find Interpretable LLM Feature Circuits (sections 3.1-3.2)"
    url: https://arxiv.org/abs/2406.11944
  - title: "Ameisen et al. — Circuit Tracing: Revealing Computational Graphs in Language Models (Anthropic, 2025)"
    url: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
  - title: "Lindsey et al. — On the Biology of a Large Language Model (Anthropic, 2025)"
    url: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
  - title: "Anthropic — Open-sourcing circuit tracing tools (2025-05-29)"
    url: https://www.anthropic.com/research/open-source-circuit-tracing
  - title: "circuit-tracer v0.5.0 (README: supported models and transcoder sets)"
    url: https://github.com/safety-research/circuit-tracer
  - title: "mwhanna/qwen3-1.7b-transcoders-lowl0 (per-layer transcoders for Qwen/Qwen3-1.7B), revision 9c1b17d"
    url: https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0
last_verified: "2026-10-07"
---

# 17.3 · Transcoders and attribution graphs

Patching tells you which components matter; it does not tell you what they compute or how they connect. Attribution graphs try to: they replace each MLP with a sparse, readable stand-in (a transcoder), freeze everything else that is nonlinear for one prompt, and then trace the prompt's output back through features as a graph of linear contributions. This lesson builds that machinery from scratch on the Module 17 model — transcoders, the local replacement model, attribution, pruning — and then does what the Circuit Tracing paper insists on and many readers skip: it treats the graph as a hypothesis and tests its predictions with interventions in the original model.

## Why this matters at a frontier lab

Attribution graphs are how Anthropic's *Biology of a Large Language Model* studied multi-step reasoning, planning and hallucination in Claude 3.5 Haiku, and since May 2025 the same method runs on open models through circuit-tracer and Neuronpedia. A graph looks like an explanation, which is its risk: it is computed in a *replacement* model with its own errors, under approximations (attention frozen, nonlinearities linearised at one point) that can be badly wrong. The authors themselves report that their graphs give satisfying insight for about a quarter of the prompts they tried. A research engineer who builds or uses graphs needs three numbers before any story: how faithful the replacement is, how much of the influence flows through error nodes, and whether the graph's predicted interventions happen in the real model.

## The idea

### Transcoders and the replacement model

A **transcoder** (Dunefsky et al. 2024, section 3.1) is a sparse stand-in for one MLP. It reads the MLP's input $x$ and predicts the MLP's output $y$:

$$z = \sigma(W_{\text{enc}} x + b_{\text{enc}}), \qquad \hat y = W_{\text{dec}} z + b_{\text{dec}}, \qquad L = \lVert y - \hat y \rVert^2 + \lambda\, \text{sparsity}(z)$$

Unlike an SAE it does not reconstruct its input; its decoder writes into the residual stream what the MLP would have written. Replacing every MLP by its transcoder gives the **replacement model**, whose MLP computation is a sum of sparse features. Dunefsky et al. report transcoders "at least on par with SAEs" on sparsity, faithfulness and interpretability (abstract) and show that the interaction of two transcoder features factorises into an input-dependent activation times an input-invariant weight product (section 3.2.1). Circuit Tracing uses a **cross-layer transcoder** (CLT): each feature reads the residual stream at one layer and writes to the MLP outputs of that layer and every later one, with a separate decoder per output layer. The course and the open Qwen3 sets use per-layer transcoders (PLTs).

### The local replacement model

For one prompt, Circuit Tracing builds the **local replacement model**:

1. attention patterns and normalisation denominators are frozen at their values on that prompt;
2. each MLP output is the transcoder's output plus an **error node** $e_l = y_l - \hat y_l$, also frozen;
3. so the local replacement model reproduces the original logits exactly, and every remaining path from an earlier feature to a later feature or logit is **linear**.

### Attribution graphs

Nodes: token embeddings, every active feature at every position, every error node, and the top output logits. With feature activations treated as independent inputs, the edge from source $s$ to target $u$ is

$$A_{us} = a_s \cdot \frac{\partial\, \text{input}_u}{\partial a_s},$$

the direct effect through the residual stream and frozen attention, holding every other feature fixed. A feature's input is its pre-activation; a logit node's input is the logit minus the mean logit. Because the paths are linear, the edges into a target sum exactly to its input minus its encoder bias.

**Pruning.** A graph for a 14-token prompt has hundreds of active features. Normalise the absolute edges into each node to sum to 1, $\hat A_{us} = |A_{us}| / \sum_{s'} |A_{us'}|$; the total influence over all path lengths is

$$B = \hat A + \hat A^2 + \cdots = (I - \hat A)^{-1} - I,$$

which converges because the graph is acyclic. A node's influence on the output is $w^\top B$ with $w$ the logit nodes' probabilities. Keep the most influential nodes up to a cumulative share (0.8 is circuit-tracer's default node threshold; 0.98 for edges). The paper reports typically cutting nodes by a factor of 10 while losing about 20% of the behaviour explained. The **error share** — the influence that flows from error nodes — says how much of the graph is "dark matter" the dictionary does not explain; Circuit Tracing summarises the same idea in its completeness and replacement scores.

### A graph is a hypothesis

The graph's claim "feature $f$ drives this output" predicts what happens when $f$ is removed. The paper validates graphs with perturbations: clamp or scale a feature and measure downstream features and logits. The course runs both sides: the **prediction** from the local replacement model (frozen attention and norms, downstream features recomputed through their nonlinearity) and the **outcome** in the original model (the feature's write subtracted from its MLP output, everything else recomputed). The control is the same intervention on random active features of similar activation. The paper's own limitations list says why prediction and outcome can differ: attention is frozen, so QK effects are invisible; reconstruction error; inactive and inhibitory features; and graphs that are faithful to the replacement model but not the mechanism of the original.

## Worked example

**Influence on a four-node chain.** Nodes: embedding 0, feature 1, error 2, logit 3. Edges: embedding → feature 2.0; feature → logit 3.0; error → logit −1.0. Normalised rows: the logit's inputs become $(0, 0.75, 0.25, 0)$, the feature's $(1, 0, 0, 0)$. Then $B = \hat A + \hat A^2$ (longer paths are zero): $B_{31} = 0.75$, $B_{32} = 0.25$, $B_{30} = 0.75 \cdot 1 = 0.75$. The embedding reaches the logit only through the feature, so it inherits the feature's share; the error share is $0.25 / (0.75 + 0.25) = 0.25$ of the influence from roots.

**One edge by hand.** A feature at layer 0 with activation $a = 2$ and decoder row $d = (0.5, 0)$ writes $(1, 0)$ into the stream. A layer-1 feature's encoder row (after the frozen norm, which here multiplies by 0.8) is $w = (1.5, 1)$. Ignoring attention, $\partial\, \text{pre}/\partial a = 0.8 \cdot (w \cdot d) = 0.8 \cdot 0.75 = 0.6$, so the edge is $a \cdot 0.6 = 1.2$: exactly the part of the target's pre-activation this source wrote.

**Why prediction and outcome differ.** Remove a last-layer feature at the last position. In the local replacement model the final norm's denominator is frozen at $r$; in the original model it is recomputed to $r'$. The predicted logit change is $W(\Delta) \cdot r$, the real one comes from $W(x' ) r' - W(x) r$: the two differ by a scale factor even in this simplest case, which is one of the course's tests.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| MLP input / output per layer (Module 17 model) | (48,768, 128) training tokens | float32 | CPU |
| transcoder per layer | $W_{\text{enc}}$ (128, 1,024), $W_{\text{dec}}$ (1,024, 128) | float32 | CPU |
| frozen attention pattern per layer | (1, 4, 14, 14) | float32 | CPU |
| frozen norm factors | (1, 14, 1) per norm | float32 | CPU |
| adjacency $A$ for one prompt | about (970, 970) | float64 | CPU |
| Qwen3-1.7B PLT set (main path) | 28 layers × $W_{\text{enc}}$ (2,048, width) and $W_{\text{dec}}$ (width, 2,048) | bf16 | GPU |

Cost of one graph: one forward pass, then one backward pass per target node (active features plus logits), each through the frozen linear model. For the toy, about 900 targets × a 4-layer, 14-token backward: 2–3 seconds on a laptop. Inverting $I - \hat A$ is $O(n^3)$ for $n$ nodes: trivial at 970, the reason production tools prune before building the full matrix at thousands. circuit-tracer batches targets (`batch_size=512`) and caps feature nodes (`max_feature_nodes`) for the same reason.

## Build it

```python
from frontierlab.interp import graphs as G, tasks as T

model = T.m17_model()
x, y = G.collect_mlp(model, T.data_windows("train", 384, 128, 0), layer=0)
tcs = [G.train_transcoder(*G.collect_mlp(model, train, l), d_sae=1024, kind="topk", k=16) for l in range(4)]
G.replaced_logits(model, val, tcs)                      # the global replacement model: no error terms
g = G.attribute(model, prompt, tcs, n_logits=3)         # nodes, A[u, s], activations, logit probabilities
keep = G.prune(g, 0.8); G.error_share(g)
G.intervene_feature(model, prompt, tcs, layer, pos, feature, 0.0)            # outcome, original model
G.predicted_feature_effect(model, prompt, tcs, layer, pos, feature, 0.0)     # prediction, local replacement
```

`frontierlab/interp/graphs.py`: `trace` is an explicit forward pass of Baseline-0's GQA layer that records the attention patterns and norm factors (checked against the model's own forward to $10^{-12}$ in float64); `replacement_logits` runs the local replacement model with feature activations as leaf tensors; `attribute` takes one backward pass per target. Correctness checks in `test_interp.py`: the local replacement model reproduces the logits to $10^{-10}$, with features given or recomputed; edges into every logit node sum to its input and into every feature node to its input minus $b_{\text{enc}}$; there is no edge from a later layer or a later position; for a last-layer feature at the last position, prediction and outcome differ by exactly one positive scale factor (the final norm) and earlier positions are unchanged in both.

## What the evidence says

- **Transcoders: PROMISING.** Dunefsky et al. trained them on GPT-2 small and Pythia-410M and 1.4B (section 4) and report parity with SAEs. Gemma Scope released transcoders for Gemma 2 2B; open Qwen3 sets exist for 0.6B, 1.7B, 4B, 8B and 14B (circuit-tracer v0.5.0 README; PUBLICLY DOCUMENTED).
- **Attribution graphs: PROMISING, with explicitly stated limits.** PUBLICLY DOCUMENTED in Circuit Tracing (Anthropic, 2025-03-27): CLTs from 300K to 10M features on an 18-layer model and up to 30M on Claude 3.5 Haiku; the largest 18-layer CLT's replacement model matches the original's next-token completion on 50% of a diverse set of pretraining-style prompts; the largest Haiku CLT has 21.7% normalised reconstruction error at an average $L_0$ of 235 (company claims about a closed model). Biology of an LLM reports case studies — the two-hop "Dallas → Texas → Austin" computation, where swapping in California features yields Sacramento; planning of rhyme words before a poem line is written; multilingual circuits; addition; hallucination and refusal circuits; chain-of-thought faithfulness — and says the graphs give satisfying insight for about a quarter of the prompts tried. Treat those as findings about one model, with interventions behind each.
- **Tools: ESTABLISHED in open use.** Anthropic open-sourced circuit-tracer with a Neuronpedia front end on 2025-05-29 for Gemma-2-2b and Llama-3.2-1b; v0.5.0 (2026-03-29) lists Qwen3 per-layer transcoder sets and offers an experimental nnsight backend besides TransformerLens. MODEL-SPECIFIC: the `mwhanna/qwen3-1.7b-transcoders-lowl0` set was trained on the post-trained Qwen/Qwen3-1.7B, not Qwen3-1.7B-Base, so the main path builds graphs on that checkpoint.
- **Course measurement (free CPU, 2026-10-07; the Module 17 model; 76 s after lesson 17.1's model exists (105 s with other jobs running)):**
  - four TopK transcoders (1,024 features, $k = 16$): held-out FVU 0.026–0.030; the global replacement model (all four MLPs replaced, no error terms) has loss 5.780 against the model's 5.743 and agrees on the top-1 next token at 84.3% of positions;
  - three graphs of 14-token validation prompts: 964–967 nodes with 891–894 active features; pruned at 0.8 to 51–76 nodes (31–53 features) — a 13–19× reduction; error-node share 0.069–0.115;
  - the most influential features in all three graphs were layer-0 features at the last position: features of the current token, the first thing a 4-layer model trained on 2.5M tokens uses;
  - interventions: for the 18 most influential features, the graph predicted mean |logit change| 1.135 and the original model showed 0.424 (correlation +0.45, same sign 72%); for 36 random active features of similar activation, 0.048 predicted and 0.035 real (correlation +0.94). 83% of the top features had a real effect above the random features' 95th percentile. Read: the graph found features that matter (against a control) but overstated *how much* by about 2.7×, because in the original model attention and norms adapt and other features compensate. Its ranking is useful; its magnitudes are a prediction to test.

## Lab

**Folder:** [`labs/module-17/lesson-03/`](../../labs/module-17/) · **Time:** about 70 minutes · **Pass check:** `pytest labs/module-17/lesson-03` passes; `graph_lab.py` prints all four parts; your write-up states one graph hypothesis, its predicted and measured intervention effects, the control, and the error share.

### Experiment contract

- **Question:** do the features an attribution graph ranks as most influential change the model's output when removed, by the amount the graph predicts? Decision informed: whether graphs from this replacement are used to choose intervention targets (ranking) and whether their magnitudes can be quoted (size).
- **Hypothesis:** top-ranked features have larger real effects than random active features of similar activation, and real effects track predicted ones. Status: reported by the method's authors, with listed failure modes; may not hold for a 4-layer model.
- **Baseline:** random active features of similar activation in the same graphs.
- **Changed variable:** which feature is removed. **Controlled:** model, transcoders (trained once), prompts (three confident validation prefixes, chosen by a fixed rule), the intervention (subtract the feature's write at its position), the target (top logit minus mean).
- **Comparison axis:** the same prompts and target for every intervention.
- **Budget:** free CPU, measured 76 s after the model exists; main path about 0.5 GPU-hours (PROJECTED).
- **Metrics and decision rule:** mean |real change| per group, correlation and sign agreement between predicted and real, and the fraction of top features above the random 95th percentile. Ranking is "useful" if that fraction is at least 0.8; magnitudes are "quotable" if predicted and real agree within a factor of 1.5 on average.
- **Correctness checks:** `pytest labs/common/tests/test_interp.py -k "replacement or feature_intervention or transcoder"`; `pytest labs/module-17/lesson-03`.
- **Fallback evidence:** Circuit Tracing's perturbation experiments and Biology's case studies, labelled as published.
- **Limits:** per-layer transcoders, not CLTs; three prompts; one target per prompt; a 4-layer model whose most influential features sit in layer 0; QK (attention-pattern) mechanisms invisible by construction.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× 48–80 GB GPU (A100/H100). Not run in this build; part of the Module 17 pilot | Env B (circuit-tracer 0.5.0 pins `transformers<=4.57.3`, so it needs its own environment): `graph_lab.py --variant main --print` lists the commands — graphs on Qwen/Qwen3-1.7B with `mwhanna/qwen3-1.7b-transcoders-lowl0` (revision `9c1b17d`), `prune_graph(g, 0.8, 0.98)`, `compute_graph_scores`, and `feature_intervention` for the top 6 and 12 random active features per prompt. **PROJECTED:** about 0.5 GPU-hours for 10 prompts and 60 interventions |
| Free GPU (Colab/Kaggle T4) | T4 | circuit-tracer with `mwhanna/qwen3-0.6b-transcoders-lowl0` (read its `config.yaml` for the model it was trained on and use that one), `max_feature_nodes` capped; what you lose: a smaller model and a smaller graph |
| Free CPU | laptop; measured 76 s after lesson 17.1's model exists | `python labs/module-17/lesson-03/graph_lab.py` |

### Steps

1. **Implement** the four TODOs in `lab.py` and run `pytest labs/module-17/lesson-03`.
2. **Run** `python labs/module-17/lesson-03/graph_lab.py`. From part 2, is the replacement model good enough that a graph of it could describe the original? Use the loss and the agreement, not FVU.
3. **Read one graph.** For prompt 0, list the five most influential feature nodes and look up what each fires on (reuse `sae.top_contexts` with a transcoder; it has the same `encode`). Write the graph's hypothesis in one sentence.
4. **Test it.** From part 4, does the hypothesis pass the contract's rule for ranking? For magnitude? Name two mechanisms from the "A graph is a hypothesis" section that could explain the gap, and design one experiment that distinguishes them (for example: freeze attention patterns in the original model during the intervention).
5. **Run your experiment** with the course's hooks and report the result with the same control.

<details>
<summary>Hint for TODO 3</summary>

Sort node indices by influence in decreasing order, skip logit nodes, and keep a running sum; stop *before* adding a node once the running share is already at the threshold. Add every logit node at the end and return the indices sorted. The test compares with `frontierlab.interp.graphs.prune` at three thresholds.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-17/lesson-03/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-17/lesson-03`. The build's measured numbers are in "What the evidence says".

</details>

## Common mistakes

- **Reading a graph without its error share.** A graph whose influence flows mostly from error nodes explains the dictionary's gaps, not the model.
- **Quoting predicted effects as effects.** Prediction comes from a model with frozen attention and norms; test it in the original.
- **No control for interventions.** Removing any active feature changes the output a little; compare with random features of similar activation.
- **Building graphs on the wrong checkpoint.** A transcoder set belongs to one model; the open Qwen3-1.7B set was trained on the post-trained model, not the base.
- **Expecting attention mechanisms in the graph.** Patterns are frozen, so a mechanism that works by moving attention shows up only as which positions edges come from.
- **Installing circuit-tracer into the main environment.** Its transformers pin conflicts with the course's; use a separate environment.

## References

- J. Dunefsky, P. Chlenski and N. Nanda, *Transcoders Find Interpretable LLM Feature Circuits*, 2024. https://arxiv.org/abs/2406.11944
- E. Ameisen, J. Lindsey et al., *Circuit Tracing: Revealing Computational Graphs in Language Models*, Anthropic, 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- J. Lindsey et al., *On the Biology of a Large Language Model*, Anthropic, 2025. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Anthropic, *Open-sourcing circuit tracing tools*, 2025. https://www.anthropic.com/research/open-source-circuit-tracing
- circuit-tracer v0.5.0. https://github.com/safety-research/circuit-tracer
- M. Hanna, *qwen3-1.7b-transcoders-lowl0* (revision `9c1b17dfb156d82162ccd2cb7f047ac7f3d3585d`, MIT). https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[17.4 · Steering and persona vectors](lesson-04.md)
