---
id: "09.1"
module: 9
minutes: 45
practice_minutes: 90
prerequisites: ["02.3", "04.3", "08.2"]
objectives:
  - Compute, for a given model and layout, the weights, gradients, optimizer state and activations each device holds, and the bytes each parallel dimension sends per step.
  - Explain why large runs order the dimensions TP innermost and DP outermost, and show with the rank-placement arithmetic which groups leave the node.
  - Reconstruct the layouts Llama 3 and DeepSeek-V3 document and say what each choice buys and costs.
  - Implement ring attention across processes, verify its output and its gradients against single-device attention in float64, and measure what load-balanced sharding does on your hardware.
volatility: concept
sources:
  - title: "The Llama 3 Herd of Models (section 3.3.2: 4D parallelism [TP, CP, PP, DP], Table 4; context parallelism with all-gathered K/V)"
    url: https://arxiv.org/abs/2407.21783
  - title: "DeepSeek-V3 Technical Report (section 3.1: H800 cluster; section 3.2: 16-way PP, 64-way EP over 8 nodes, ZeRO-1 DP, no TP)"
    url: https://arxiv.org/abs/2412.19437
  - title: "Liu, Zaharia, Abbeel — Ring Attention with Blockwise Transformers for Near-Infinite Context"
    url: https://arxiv.org/abs/2310.01889
  - title: "Yang et al. (Meta) — Context Parallelism for Scalable Million-Token Inference (pass-KV, pass-Q, load-balanced sharding)"
    url: https://arxiv.org/abs/2411.01783
  - title: "Korthikanti et al. — Reducing Activation Recomputation in Large Transformer Models (activation memory per layer, sequence parallelism)"
    url: https://arxiv.org/abs/2205.05198
  - title: "Narayanan et al. — Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (TP within a node, PP across)"
    url: https://arxiv.org/abs/2104.04473
  - title: "Rajbhandari et al. — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    url: https://arxiv.org/abs/1910.02054
  - title: "Hugging Face — The Ultra-Scale Playbook"
    url: https://huggingface.co/spaces/nanotron/ultrascale-playbook
last_verified: "2026-10-04"
---

# 09.1 · Parallelism layouts at scale

A model that needs thousands of GPUs is split along up to five dimensions at once — tensor, context, pipeline, data and expert parallelism — and the order in which they are laid over the machines decides which communication crosses the slow links. This lesson builds a planner that computes memory and communication per device for any layout, uses it to reconstruct the layouts Llama 3 and DeepSeek-V3 document, and then implements the one dimension that changes attention itself: context parallelism with ring attention, checked against single-device attention down to the gradients.

## Why this matters at a frontier lab

Before a large run starts, someone has to answer "which layout?" with numbers: will it fit in 80 GB per GPU, how many bytes cross InfiniBand per step, how much of that the computation can hide, and what the pipeline bubble costs. Getting it wrong is expensive in two ways. A layout that does not fit is found in minutes; a layout that fits but puts tensor parallelism across nodes runs for weeks at a fraction of the possible throughput. The published layouts — Llama 3's `[TP, CP, PP, DP]` on 16K H100s and DeepSeek-V3's PP16 × EP64 × ZeRO-1 with no TP — are not arbitrary: each follows from the model's shape, the sequence length and the cluster's bandwidth hierarchy. Module 4 extended context by continued training; at 128K tokens one sequence's activations no longer fit on one GPU, which is why context parallelism exists.

## The idea

### Five ways to split the work

| Dimension | What is split | What it communicates, per layer and micro-batch | When it is needed |
|---|---|---|---|
| **TP** (tensor) | every weight matrix, by rows or columns | all-reduce (or all-gather + reduce-scatter with sequence parallelism) of activations, forward and backward | one layer's weights or activations too big for one GPU |
| **CP** (context) | the sequence, into `cp` pieces | keys and values of the other pieces (ring or all-gather) | one sequence's activations too big (long context) |
| **PP** (pipeline) | the layers, into `pp` stages | one activation forward, one gradient backward, at each stage boundary | the model's weights too big even after TP |
| **DP** (data) | the batch | gradients once per step (all-reduce, or reduce-scatter + all-gather under ZeRO/FSDP) | always: it is how you use more GPUs |
| **EP** (expert) | the routed experts of each MoE layer | all-to-all of tokens to their experts and back | MoE models |

Two facts drive the layout. First, the volumes are very different: TP moves activations of every layer for every micro-batch, several times; DP moves each gradient once per optimizer step. Second, the links are very different: inside an H100 node NVLink gives each GPU hundreds of GB/s; between nodes each GPU has one InfiniBand or RoCE NIC of 400 Gb/s (50 GB/s). So the most talkative dimension goes where the bandwidth is: **TP inside the node, DP across the widest, slowest span**. Llama 3 states the rule directly: "the innermost parallelism requires the highest network bandwidth and lowest latency" (section 3.3.2), and orders the dimensions `[TP, CP, PP, DP]` from innermost to outermost.

### Rank placement

With the dimensions listed innermost first, rank $r$ has coordinates $i_d = \lfloor r / \text{stride}_d \rfloor \bmod n_d$, where $n_d$ is the size of dimension $d$ and $\text{stride}_d$ is the product of the sizes of the dimensions inside it. A TP group (all ranks that differ only in $i_{\text{TP}}$) is a run of $n_{\text{TP}}$ consecutive ranks when TP is innermost — one node if $n_{\text{TP}} \leq 8$. Put DP innermost instead and the same TP group has stride $n_{\text{DP}}$ and is spread over $n_{\text{TP}}$ nodes. The planner (`frontierlab.dist.layout.group_span`) enumerates a group's ranks and counts the nodes they touch.

### Memory per device

A device holds $P_{\text{dev}} = P_{\text{stage}} / \text{TP}$ parameters (expert parameters $/ \text{EP}$ instead). With mixed precision it stores, per parameter: BF16 weights ($w = 2$ bytes), FP32 gradients ($g = 4$), FP32 master weights ($m = 4$) and AdamW's two moments ($o = 8$; DeepSeek-V3 keeps them in BF16, $o = 4$). ZeRO (Rajbhandari et al.) shards over the $R$ data-parallel replicas (DP × CP ranks hold the same weights): stage 1 divides $m + o$ by $R$, stage 2 also $g$, stage 3 (FSDP) also $w$:

$$M_{\text{state}} = P_{\text{dev}} \left( \frac{w}{R_3} + \frac{g}{R_2} + \frac{m + o}{R_1} \right), \qquad R_k = R \text{ if ZeRO stage} \geq k \text{ else } 1$$

Activations, per layer and micro-batch of $b$ sequences, with $s_\ell = s / \text{CP}$ tokens per device, $e$ bytes per element, FlashAttention (no $s^2$ score matrix stored) and sequence parallelism (Korthikanti et al. 2022's accounting, rewritten for GQA and SwiGLU):

$$A_{\text{layer}} = \frac{s_\ell\, b\, e}{\text{TP}}\Big(\underbrace{2h}_{\text{norm inputs}} + \underbrace{h}_{\text{attn input}} + \underbrace{H d_q + K(d_q + d_v) + H d_v}_{\text{Q, K, V, attn output}} + \underbrace{h}_{\text{FFN input}} + \underbrace{4 I}_{\text{SwiGLU}}\Big)$$

Here $h$ is the hidden size, $H$ and $K$ the query and KV head counts, $d_q, d_v$ the head dimensions and $I$ the FFN width (for an MoE layer, $k$ routed experts' widths plus the shared expert, and $k$ copies of the dispatched input). For a GPT layer with multi-head attention and a $4h$ GeLU MLP the same bookkeeping gives Korthikanti's $32\,s b h$ bytes without dropout masks; `test_dist.py` checks it. Full recomputation keeps only each layer's input, $2 s_\ell b h / \text{TP}$. The first pipeline stage holds the activations of $\text{PP}$ micro-batches under 1F1B (lesson 09.2 shows why), so its total is about the whole model's layers' worth.

### Communication per device and step

For a ring collective over $n$ ranks, each rank sends $\tfrac{n-1}{n}$ of the buffer for an all-gather or a reduce-scatter and twice that for an all-reduce (lesson 02.3). Per optimizer step of $m$ micro-batches on a device that holds $L_{\text{dev}}$ layers:

$$\begin{aligned}
B_{\text{TP}} &= 8\, \tfrac{t-1}{t}\; s_\ell b h e \cdot L_{\text{dev}}\, m && \text{(2 AG + 2 RS forward, the same backward)}\\
B_{\text{CP}} &= 3\, (c-1)\; s_\ell b \tfrac{K}{t} (d_q + d_v) e \cdot L_{\text{dev}}\, m && \text{(K, V forward; K, V, dK, dV backward)}\\
B_{\text{PP}} &= 2\, s_\ell b h e / t \cdot m && \text{(activation forward, gradient backward)}\\
B_{\text{DP}} &= \tfrac{R-1}{R}\, P_{\text{dev}} \cdot \{2g,\ \ g + w,\ \ g + 2m\,w\} && \text{(ZeRO-0 all-reduce; ZeRO-1/2 RS + AG; ZeRO-3 AG every pass)}\\
B_{\text{EP}} &= 4\, \tfrac{E-1}{E}\; s_\ell b\, k\, h e \cdot L_{\text{MoE}}\, m && \text{(dispatch + combine, forward and backward)}
\end{aligned}$$

Dividing each by the bandwidth of the link its group uses gives an upper bound on the exposed communication; overlap hides part of it, and only a measurement says how much (lesson 09.3).

### What Llama 3 and DeepSeek-V3 document

**Llama 3 405B** (PUBLICLY DOCUMENTED, section 3.3.2 and Table 4): 4D parallelism ordered `[TP, CP, PP, DP]`; TP = 8, PP = 16; DP = 64 on 8,192 GPUs (43% BF16 MFU) and DP = 128 on 16,384 GPUs (41%) at 8,192-token sequences; for 131,072-token sequences CP = 16, DP = 8 on 16,384 GPUs (38%). 16M tokens per batch in all three. The pipeline uses an interleaved schedule with a flexible number of micro-batches and removes one transformer layer from the first and from the last stage to balance memory and compute. Their CP is not a ring: K and V are **all-gathered**, then each rank computes attention for its local queries; the sequence is cut into $2 \times \text{CP}$ chunks so that every rank gets two (one early, one late) for load balance.

**DeepSeek-V3** (PUBLICLY DOCUMENTED, sections 3.1–3.2): 2,048 H800 GPUs, 8 per node, NVLink 160 GB/s inside a node and InfiniBand 50 GB/s between nodes. "16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)", and no TP: the report says memory optimisations (recomputing RMSNorm and the MLA up-projections, keeping an EMA on the CPU, sharing embedding and output head on one PP rank, FP8 activation caching from lesson 08.2) make TP unnecessary, so its communication is avoided. Their cross-node all-to-all uses 20 SMs and sends each token to at most 4 nodes.

### Context parallelism with a ring

Split a causal sequence over $n$ ranks. Rank $r$ needs every key at a position $\leq$ its queries' positions. In **ring attention** (Liu et al., 2023) the K/V blocks travel around a ring: at step $i$ rank $r$ holds the block that started on rank $(r - i) \bmod n$, computes its queries' scores against it, and folds the result into a running output with the online-softmax rule. For block $j$, $S_j = q K_j^\top / \sqrt{d}$ (masked), $o_j = \text{softmax}(S_j) V_j$ and $\ell_j = \log \sum \exp S_j$ (the log-sum-exp). Two partial results over disjoint keys combine exactly:

$$\ell = \log\!\left(e^{\ell_a} + e^{\ell_b}\right), \qquad o = e^{\ell_a - \ell}\, o_a + e^{\ell_b - \ell}\, o_b$$

While block $i$ is being computed, block $i+1$ is already being sent and received, so the transfer hides behind the compute when computing a block takes longer than sending it. The backward pass reuses the saved $o$ and $\ell$: every block's contribution $P_j = \exp(S_j - \ell)$, $dV_j = P_j^\top dO$, $dS_j = P_j \circ (dO\,V_j^\top - \text{rowsum}(dO \circ o))$, $dQ \mathrel{+}= dS_j K_j / \sqrt{d}$, $dK_j = dS_j^\top q / \sqrt{d}$ is independent of the others. The K/V blocks make the trip again, and each block's $dK, dV$ accumulator travels *with* it, so after $n$ hops it arrives back at its owner carrying the sum over every rank's queries.

**Load balance.** With contiguous shards, the last rank's queries see all keys and the first rank's see a quarter of one block: the causal work is badly skewed. Giving rank $r$ chunks $r$ and $2n-1-r$ of $2n$ equal chunks evens it out — the arrangement Llama 3 and Meta's CP paper use. It only helps if the attention kernel *skips* fully masked sub-blocks; a kernel that computes the full rectangle and masks afterwards does the same work either way.

**Ring or all-gather?** All-gathering K/V (Llama 3) is simpler and moves the same bytes, but needs memory for the whole sequence's K/V at once; the ring holds two blocks. Meta's CP paper (for inference) passes K/V or, when the new tokens are few compared with the cached ones, passes the queries instead (pass-Q), which moves less with GQA; they report 1M-token prefill of Llama 3 405B in 77 s on 128 H100s across 16 nodes (93% parallelisation efficiency; company claim).

## Worked example

### Placement, 16 GPUs in 2 nodes

TP = 4, DP = 4. Order `[TP, DP]`: TP stride 1, so the TP groups are ranks {0,1,2,3}, {4,…,7}, … — each inside one node; the DP group of rank 0 is {0, 4, 8, 12}, two nodes. Order `[DP, TP]`: the TP group of rank 0 is {0, 4, 8, 12}, crossing the node boundary.

What that costs, for Llama 3 8B (32 layers, $h = 4096$), 8 micro-batches of one 4,096-token sequence per step, TP = 4, ZeRO-1: $B_{\text{TP}} = 8 \cdot \tfrac34 \cdot 4096 \cdot 4096 \cdot 2 \cdot 32 \cdot 8 = 51.5$ GB per device per step; $B_{\text{DP}} = \tfrac34 \cdot (8.03\text{B}/4) \cdot (4 + 2) = 9.0$ GB. At the planner's assumed 300 GB/s inside a node and 40 GB/s between nodes: TP inside / DP across costs $0.17 + 0.23 = 0.40$ s; the reverse costs $1.29 + 0.03 = 1.32$ s, against about 1.0 s of compute at 40% MFU. Same bytes, more than three times the communication time. (Planner output; PROJECTED, bandwidths assumed.)

### ZeRO, by hand

8B parameters on 8 GPUs, DP only. Bytes per parameter: ZeRO-0 $2 + 4 + 12 = 18$ (144 GB: does not fit an 80 GB H100); ZeRO-1 $2 + 4 + 12/8 = 7.5$ (60 GB); ZeRO-2 $2 + 16/8 = 4$ (32 GB); ZeRO-3 $18/8 = 2.25$ (18 GB). Each stage trades memory for communication: ZeRO-3 all-gathers the weights in every forward and backward.

### One merge, by hand

A query sees one key in block $a$ with score 0 and value 1, and one key in block $b$ with score $\ln 3$ and value 5. Then $o_a = 1, \ell_a = 0$; $o_b = 5, \ell_b = \ln 3$. Merged: $\ell = \ln(1 + 3) = \ln 4$; weights $e^{0 - \ln 4} = 1/4$ and $e^{\ln 3 - \ln 4} = 3/4$; $o = 0.25 \cdot 1 + 0.75 \cdot 5 = 4$. Directly: softmax of $(0, \ln 3)$ is $(1/4, 3/4)$, the same 4.

### Causal work, 8 tokens on 2 ranks

Query at position $t$ sees $t + 1$ keys. Contiguous: rank 0 has tokens 0–3, $1 + 2 + 3 + 4 = 10$ pairs; rank 1 has 4–7, $5 + 6 + 7 + 8 = 26$. Balanced (chunks of 2: rank 0 gets chunks 0 and 3, rank 1 chunks 1 and 2): $1 + 2 + 7 + 8 = 18$ and $3 + 4 + 5 + 6 = 18$. The slowest rank does 26 instead of 18: a 1.44× difference at 2 ranks, 1.75× at 4 (`work_per_rank`).

## Shapes and cost

| Per device, layout TP·CP·PP·DP | Shape | dtype | Device | Communicated |
|---|---|---|---|---|
| column-parallel weight (e.g. `q_proj`) | $(H d / t,\ h)$ | bf16 working, fp32 master | GPU | gradients over DP |
| hidden states between blocks (SP) | $(b,\ s_\ell / t,\ h)$ | bf16 | GPU | all-gather before, reduce-scatter after each TP region |
| local queries | $(b,\ H/t,\ s_\ell,\ d)$ | bf16 | GPU | — |
| K/V block in the ring | $(2,\ b,\ K/t,\ s_\ell,\ d)$ | bf16 | GPU, two buffers | sent $c - 1$ times forward |
| $dK, dV$ accumulator | same as the block | fp32 (accumulation) | GPU | travels with the block, $c$ hops |
| pipeline activation at a boundary | $(b,\ s_\ell / t,\ h)$ | bf16 | GPU | point-to-point to the next stage |
| tokens dispatched to experts | $(\approx s_\ell b k / E,\ h)$ per destination | bf16 or fp8 | GPU | all-to-all over EP |

Ring attention costs the same FLOPs as single-device attention (it computes each visible (query, key) pair once) plus $c - 1$ block transfers forward and $2c - 1$ backward per layer; memory per device for K/V is two blocks instead of the whole sequence.

The CPU runs in this lab use the same shapes with $b = 1$, 4–8 heads, $d = 8$ (float64 checks) or $d = 64$ (float32 timing), on `gloo` processes.

## Build it

`frontierlab/dist/layout.py` is the planner. It reads an `ArchSpec` (lesson 01.2's config reader, so any snapshot model works), a `Layout` (degrees, ZeRO stage, rank order), a `Train` (sequence, micro-batches, schedule, bytes per element) and a `Cluster` (GPUs per node, HBM, assumed bandwidths):

```python
from frontierlab.calc import from_hf
from frontierlab.dist import layout as LY

ds = from_hf("deepseek-ai/DeepSeek-V3")
lay = LY.Layout(pp=16, dp=128, ep=64, zero=1, order=("tp", "cp", "ep", "pp", "dp"))
tr = LY.Train(seq=4096, n_micro=120, schedule="dualpipe", optim_bytes=4, act_bytes=1)
print(LY.format_plan(LY.plan(ds, lay, tr, LY.H800_NODE)))
```

`frontierlab/dist/ring_attention.py` implements ring attention as a `torch.autograd.Function` over `torch.distributed` point-to-point calls. Each step starts `isend`/`irecv` of the next K/V block before computing the current one; the backward pass rotates K/V again together with the $dK, dV$ accumulators. Masking is by absolute positions, so contiguous and load-balanced shards use the same code, and the rank's tokens are split into contiguous segments so that fully masked (query segment, key segment) sub-blocks are skipped, the block skipping a causal FlashAttention kernel does. Grouped-query attention sends the smaller K/V and expands heads only locally.

The correctness check, `equivalence_worker`, builds the full sequence on every rank with the same seed, computes single-device attention and its gradients with autograd, then runs ring attention on the rank's shard and compares output, $dQ$, $dK$, $dV$. In float64 every difference is below $2 \times 10^{-15}$ on 2 and 4 ranks, for both shardings (measured; `labs/common/tests/test_dist.py`).

## What the evidence says

- **ESTABLISHED.** TP within a node, PP across nodes and DP outermost (Megatron-LM, Narayanan et al. 2021; Llama 3 section 3.3.2). ZeRO/FSDP sharding of optimizer state, gradients and weights. Sequence parallelism for the activations between TP regions (Korthikanti et al.).
- **ESTABLISHED (several labs and frameworks), with variants.** Context parallelism for long sequences: Llama 3 (all-gather of K/V, 2·CP chunks), Meta's CP paper (pass-KV / pass-Q), and CP in Megatron-Core and torchtitan. The ring variant (Liu et al.) is the published reference; which variant wins depends on memory and on how many tokens are new.
- **MODEL-SPECIFIC.** DeepSeek-V3's no-TP layout: it depends on MLA's small KV, aggressive recomputation, FP8 activation caching and an MoE whose experts are spread by EP. It is evidence that TP is avoidable for that architecture on that cluster, not in general.
- **Measured here (CPU, `gloo`):** ring attention's exactness, and the effect of load-balanced sharding on CPU processes (below). Llama 3's MFU figures are theirs (measured by Meta on their cluster). Every planner number is an ESTIMATE or PROJECTED; the planner's own limits: it assumes uniform expert routing, ignores norms, biases and the router in activation memory, and treats overlap as zero.
- **Open question for you:** the planner says Llama 3's Table 4 layouts need about 142 GB per GPU without activation recomputation and about 27 GB with full recomputation. The report does not say which recomputation policy was used; a real answer needs the memory profile of the actual run (INFERENCE that something between the two was used).

## Lab

### Experiment contract

- **Question:** on 4 ranks, does load-balanced (2·CP chunks) sharding make ring attention's slowest rank faster than contiguous sharding? Decision informed: which sharding the course uses for CP runs.
- **Hypothesis and status:** the slowest rank's time falls by up to the work ratio (1.75× at 4 ranks); established for GPU kernels that skip masked blocks (Llama 3, Meta CP paper); may not appear on CPU processes that share one machine's cores and memory bandwidth.
- **Baseline:** contiguous sharding, same code.
- **Changed variable:** the sharding. **Controlled:** sequence length 4,096, 8 heads × 64, float32, the code, world size 4, threads per process (cores / 4), inputs (seeded per rank).
- **Comparison axis:** equal work (the same attention computed exactly).
- **Budget:** free CPU about 3.5 minutes; main path a few GPU-minutes on 4 GPUs.
- **Metrics and decision rule:** slowest rank's forward + backward time per step, median with a bootstrap interval, alternating rounds; paired speed-up interval (lesson 02.4's `speedup`). Adopt balanced sharding if the speed-up interval lies entirely above 1; otherwise "not distinguishable here".
- **Correctness checks:** the float64 equivalence of part 1 (output and three gradients below $10^{-12}$, both shardings, 2 and 4 ranks) must pass first.
- **Fallback evidence:** the work-split arithmetic (`work_per_rank`) and the published Llama 3 / Meta CP results, labelled as published.
- **Limits:** CPU processes share cores, so they do not behave like independent GPUs; one sequence length; times say nothing about NVLink.

**Folder:** [`labs/module-09/lesson-01/`](../../labs/module-09/) · **Time:** about 90 minutes · **Pass check:** `pytest labs/module-09/lesson-01` passes; `ring_cp.py` part 1 passes; your write-up applies the decision rule.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1 node, 4–8 GPUs (any CUDA GPU), about 10 GPU-minutes | `plan_layouts.py` as below; `ring_cp.py --device cuda --world 8 --T 32768 --heads 32 --dim 128` (NCCL, bf16; part 1's float64 check still runs on CPU first). Not run in this build; part of the Module 9 pilot (multi-GPU is not piloted course-side) |
| Free GPU (Kaggle "GPU T4 ×2") | 2× T4 over PCIe | `ring_cp.py --device cuda --world 2 --T 16384`; untested (T4 has no BF16 tensor cores, so expect slow BF16 math) |
| Free CPU | laptop, 4 `gloo` processes | the steps below; measured 3 min 19 s for `ring_cp.py --rounds 3`, a few seconds for `plan_layouts.py` |

1. **Implement** `group_nodes`, `state_bytes_per_device`, `tp_cp_bytes_per_layer` and `merge` in `lab.py`; `pytest labs/module-09/lesson-01` checks them against the planner and against single-device attention (your `merge` is driven through a simulated ring).
2. **Plan.**

   ```bash
   python labs/module-09/lesson-01/plan_layouts.py
   ```

   It replays Llama 3's three Table 4 layouts, the same layout with the rank order reversed, DeepSeek-V3's layout with BF16 and with FP8 activation caching, and four layouts of Baseline-0 on one node. Write down: which Llama 3 layout needs recomputation to fit; how the reversed order changes TP's time; which DeepSeek-V3 dimension moves the most bytes; and which Baseline-0 layout you would pick and why (it is lesson 09.3's question).
3. **Run the ring.**

   ```bash
   python labs/module-09/lesson-01/ring_cp.py --rounds 3
   ```

4. **Decide** with the contract's rule, and explain the per-rank compute and wait columns.

<details>
<summary>Hint for TODO 1</summary>

Build the strides in `order`, then start from `ranks = [0]` and for each dimension in `dims` replace every rank `r` with `r + i * stride` for `i in range(size)`. The answer is the number of distinct `r // gpus_per_node`.

</details>

<details>
<summary>Hint for TODO 4</summary>

`torch.logaddexp` handles the log-sum; the weights are `exp(lse_a - lse)`. When both inputs are `-inf`, that is `exp(nan)`: replace NaN weights with 0 (`torch.nan_to_num`).

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04 on the build laptop (Windows 11, 16 threads, torch 2.14.1+cpu, 4 `gloo` processes with 4 threads each, another agent's jobs running), `ring_cp.py --rounds 3`, 3 min 19 s:

| | contiguous | balanced |
|---|---|---|
| causal work per rank | 6%, 19%, 31%, 44% | 25% each |
| forward compute per rank (ms) | 265, 414, 623, 785 | 648, 661, 655, 409 |
| forward wait for the next block (ms) | 228, 24, 0, 0 | 0, 0, 0, 54 |
| slowest-rank forward + backward | 1,674 ms [1,513, 1,755] | 1,709 ms [1,559, 1,769] |

Balanced vs contiguous: 0.99× [0.87, 1.11]. The rule says "not distinguishable here", although the work split predicts up to 1.75×. Two things are visible in the table. The balanced ranks' compute is far above 9/16 of the slowest contiguous rank's (the ideal for 4 ranks): each balanced rank computes 9 sub-blocks of 512 × 512 against 4 larger blocks for the slowest contiguous rank, so per-call overheads and the masking of diagonal blocks weigh more. And all four balanced ranks compute at the same time on one shared CPU, while contiguous ranks 0 and 1 finish early and wait (first rank waits 228 ms); on one machine the processes compete for the same memory bandwidth, so evening out the work does not shorten the critical path the way it does on independent GPUs (INFERENCE from the per-rank columns; the main path tests it). Part 1 passed: largest difference $1.8 \times 10^{-15}$ over output and gradients, both shardings, 2 and 4 ranks.

`plan_layouts.py` (ESTIMATES, PROJECTED): Llama 3 Table 4 row 2 needs 141.7 GB per GPU without recomputation (120.6 GB of it activations, interleaved schedule with $v = 2$), 27.3 GB with full recomputation; reversing the order sends TP across 8 nodes and its time goes from 0.80 s to 6.0 s per step. DeepSeek-V3: EP moves 222 GB per GPU per step with BF16 dispatch (4.4 s at 50 GB/s against 19.5 s of compute at 40% MFU) and the layout fits in 76 GB with FP8 activations.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-09/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-09/lesson-01`.

</details>

## Common mistakes

- **Counting a TP group's bandwidth as NVLink without checking the rank order.** The launcher's rank order decides which ranks share a node; print the groups.
- **Treating DP and CP replicas differently for ZeRO.** CP ranks hold the same weights as DP ranks; the replication (and ZeRO sharding) group is DP × CP.
- **Load-balancing CP and keeping a kernel that does not skip masked blocks.** The work is then identical for every sharding; the balance buys nothing.
- **Comparing ring attention to single-device attention in float32 or BF16 only.** Online-softmax merges reorder sums; in BF16 a correct ring differs by $10^{-3}$, and so does a ring with a subtle masking bug. Check in float64 first.
- **Forgetting that the $dK, dV$ accumulators must travel.** Reducing them with an all-reduce at the end is correct but costs an extra collective of the whole K/V and the memory for it.
- **Planning activations without the pipeline factor.** The first stage holds PP micro-batches (1F1B), not one.

## References

- Llama Team, *The Llama 3 Herd of Models*, section 3.3.1 (storage), 3.3.2 (parallelism, Table 4). https://arxiv.org/abs/2407.21783
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, sections 3.1, 3.2, 3.2.2, 3.2.3. https://arxiv.org/abs/2412.19437
- H. Liu, M. Zaharia, P. Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context*. https://arxiv.org/abs/2310.01889
- A. Yang et al., *Context Parallelism for Scalable Million-Token Inference*. https://arxiv.org/abs/2411.01783
- V. Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, section 4. https://arxiv.org/abs/2205.05198
- D. Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*. https://arxiv.org/abs/2104.04473
- S. Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. https://arxiv.org/abs/1910.02054
- Hugging Face, *The Ultra-Scale Playbook*. https://huggingface.co/spaces/nanotron/ultrascale-playbook
- Shared code: `labs/common/frontierlab/dist/layout.py`, `ring_attention.py`; versions in [references/versions.md](../../references/versions.md).

## Next

[09.2 · Pipeline schedules and overlap](lesson-02.md)
