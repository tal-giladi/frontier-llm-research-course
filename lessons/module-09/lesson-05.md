---
id: "09.5"
module: 9
minutes: 35
practice_minutes: 45
prerequisites: ["09.1", "09.4"]
objectives:
  - Describe the TPU v4 / v5p interconnect (3D torus, wraparound links, 4×4×4 cubes joined by optical circuit switches) and compute hops and bisection links for a given slice.
  - Read and write sharded matrix multiplications in the JAX Scaling Book notation and name the collective each one needs and its cost on a ring axis.
  - Explain what Pathways' single-controller design adds to SPMD training and why it matters for multi-pod runs such as Gemini 2.5's.
  - Summarise DeepSeek's hardware reflections (ISCA 2025) and 3FS as evidence of model-hardware co-design, separating measured figures from recommendations.
volatility: concept
sources:
  - title: "Jouppi et al. — TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings"
    url: https://arxiv.org/abs/2304.01433
  - title: "Barham et al. — Pathways: Asynchronous Distributed Dataflow for ML"
    url: https://arxiv.org/abs/2203.12533
  - title: "Austin et al. — How to Scale Your Model (the JAX Scaling Book): TPUs, sharded matmuls"
    url: https://jax-ml.github.io/scaling-book/
  - title: "DeepSeek-AI — Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures (ISCA 2025)"
    url: https://arxiv.org/abs/2505.09343
  - title: "deepseek-ai/3FS — Fire-Flyer File System (README)"
    url: https://github.com/deepseek-ai/3FS
  - title: "Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning"
    url: https://arxiv.org/abs/2408.14158
  - title: "Gemini 2.5 technical report (section 2.3: TPUv5p pods across data centres)"
    url: https://arxiv.org/abs/2507.06261
last_verified: "2026-10-04"
---

# 09.5 · TPUs, JAX and hardware co-design

Extension: the GPU clusters of lessons 09.1–09.4 are one design point; Google trains Gemini on TPU pods with a different interconnect and a different programming model, and DeepSeek designed DeepSeek-V3 around the limits of its H800 cluster and then wrote down what hardware it wants next. This lesson reads both as evidence: the TPU v4 torus with optical switches, Pathways' single controller, the JAX Scaling Book's notation for sharded computation, and DeepSeek's ISCA 2025 reflections and 3FS file system, with small exercises that compute hops, bisection links and collective costs.

## Why this matters at a frontier lab

A parallel layout is a fit between a model and a network. On an H100 cluster the hierarchy is sharp — NVLink inside 8 GPUs, InfiniBand between — so TP stays in the node (lesson 09.1). On a TPU pod every chip has links to six neighbours in a 3D torus, the "node" boundary disappears up to thousands of chips, and the natural question becomes which mesh *axis* each kind of parallelism uses. Research engineers move between the two worlds (and read papers written in both), so they need the vocabulary of each. Co-design runs the other way too: DeepSeek chose MLA, a node-limited MoE and FP8 partly *because* of their hardware's limits, and published which hardware changes would remove them.

## The idea

### TPU v4 and v5p: a torus you can rewire

TPU v4 (Jouppi et al., 2023; PUBLICLY DOCUMENTED, company figures) connects 4,096 chips. Chips are built into $4 \times 4 \times 4$ cubes of 64 with electrical links inside; the cubes are joined by **optical circuit switches** (OCSes) that "dynamically reconfigure its interconnect topology", so a job gets a slice of whatever shape it asks for — a 3D torus, or a *twisted* torus — and a failed cube can be switched out. The paper puts OCSes at under 5% of system cost and under 3% of system power, and reports TPU v4 1.2–1.7× faster than the A100 and 1.3–1.9× more power-efficient on the workloads it compares (company claim).

In a 3D torus each chip has six neighbours; **wraparound** links join the two ends of every row, so the distance along an axis of $n$ chips is $\min(d, n - d)$. Two quantities describe the network: the worst-case number of hops, $\sum_i \lfloor n_i / 2 \rfloor$, and the **bisection** — the links cut when the machine is split in half across its longest axis, $2 \prod_i n_i / n_{\max}$ with wraparound (each ring is cut twice).

The Scaling Book's TPU table (read 2026-10-04) gives per chip: v4p $2.75 \times 10^{14}$ BF16 FLOP/s, 32 GB HBM at $1.2 \times 10^{12}$ B/s, ICI links of $4.5 \times 10^{10}$ B/s one way, pods of $16 \times 16 \times 16$; v5p $4.59 \times 10^{14}$ FLOP/s, 96 GB at $2.8 \times 10^{12}$ B/s, $9 \times 10^{10}$ B/s per link, pods of $16 \times 20 \times 28 = 8{,}960$ chips. Between pods traffic goes over the data-centre network (DCN), about $6.25$–$12.5 \times 10^{9}$ B/s per chip — two orders of magnitude below ICI, the TPU world's equivalent of NVLink vs InfiniBand. Gemini 2.5 trained with synchronous data parallelism across multiple 8,960-chip v5p pods in several data centres (report section 2.3): the data-parallel axis is the one that crosses the DCN.

### The Scaling Book's notation

An array's sharding is written with mesh-axis subscripts: $A[I_X, J]$ is split along $I$ over mesh axis $X$ and replicated along $J$; $A[I_{XY}, J]$ is split along $I$ over both axes. A mesh axis can be used by only one dimension of an array. For $C[I, K] = A[I, J] \cdot B[J, K]$ (contracting over $J$):

| Case | Example | Communication |
|---|---|---|
| $J$ sharded in neither | $A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ | none |
| $J$ sharded in one operand | $A[I, J_X] \cdot B[J, K]$ | all-gather that operand along $X$ first |
| $J$ sharded identically in both | $A[I, J_X] \cdot B[J_X, K]$ | local matmul gives partial sums: all-reduce (or reduce-scatter) over $X$ |
| both use $X$ on non-contracting dims | $A[I_X, J] \cdot B[J, K_X]$ | not allowed as written: $C$ would use $X$ twice; all-gather one operand |

Megatron TP is cases 1 and 3: the first MLP matmul $X[B, D] \cdot W_{\text{in}}[D, F_X]$ needs nothing and gives $H[B, F_X]$; the second $H[B, F_X] \cdot W_{\text{out}}[F_X, D]$ is case 3, one all-reduce per MLP. FSDP is case 2 on the weights: $W[D_X, F]$ is all-gathered before use.

Costs on one ring axis of $n$ chips with one-way link bandwidth $W$ (both ring directions used): an all-gather or reduce-scatter of an array of $V$ bytes takes about $V \tfrac{n-1}{n} / (2W)$; an all-reduce, being a reduce-scatter followed by an all-gather, twice that. The same bytes as lesson 02.3's ring formulas, written per axis.

### Pathways: one controller for many islands

Most GPU training is *multi-controller* SPMD: every process runs the same program and they meet in collectives (`torchrun`). Pathways (Barham et al., 2022) is a *single-controller* system: one client builds "a sharded dataflow graph of asynchronous operators that consume and produce futures", and the runtime gang-schedules it on islands of accelerators. The paper reports about 100% accelerator utilisation on 2,048 TPUs for SPMD computations and comparable throughput for Transformer models "pipelined across 16 stages, or sharded across two islands of accelerators connected over a data center network" (company claim). The point for training at scale: programs that span pods, change shape (elasticity, as in Gemini 2.5's slice-level recovery, lesson 09.4) or mix pipeline and data parallelism across DCN are easier to express when one controller owns the whole graph.

### DeepSeek's reflections on hardware (ISCA 2025)

*Insights into DeepSeek-V3* (DeepSeek-AI, 2025; PUBLICLY DOCUMENTED, the company's own analysis) explains V3's design through its hardware and lists what it wants changed:

- **Memory:** MLA's KV cache is 70.3 KB per token against 327.7 KB for Qwen-2.5 72B and 516.1 KB for Llama 3.1 405B (Table 1, BF16) — the reason for MLA (Module 3).
- **Compute:** about 250 GFLOPs per training token for V3 against 394 for a 72B dense model and 2,448 for 405B dense (Table 2) — the MoE argument.
- **Bandwidth:** on H800 the gap between NVLink and the per-GPU InfiniBand bandwidth is about 10:1 in their accounting (sections 4.1, 4.3; the V3 report's section 3.1 quotes 160 GB/s vs 50 GB/s as the bandwidths it designs around — the two documents count differently, so check which figure a number refers to). Hence node-limited routing: each token's 8 routed experts are restricted to at most 4 nodes (256 experts in 8 groups of 32), so InfiniBand carries less.
- **Network:** a multi-plane two-layer fat-tree, one plane per GPU–NIC pair (8 × 400 Gb/s NICs per node), up to 16,384 GPUs with 64-port switches, at $72M against $491M for a three-layer fat-tree in their cost table (Table 3; company estimate).
- **Low precision:** Hopper's FP8 tensor-core accumulation keeps "FP22 registers (1 sign bit, 8 exponent bits, and 13 mantissa bits)" — the limit behind the FP32 promotion every 128 elements in lesson 08.2 — and they ask for higher accumulation precision.
- **Recommendations** (not measurements): precise low-precision compute, convergence of scale-up and scale-out networks, and lower-latency communication.

**3FS** (Fire-Flyer File System, deepseek-ai/3FS README, company claim): a disaggregated file system over SSDs and RDMA with strong consistency (CRAQ chain replication) and FoundationDB for metadata, used for data loading, checkpointing and KV-cache offload; peak aggregate read throughput of 6.6 TiB/s with 180 storage nodes and 500+ clients. For lesson 09.4's arithmetic: a 1T-parameter model's 12 TB of FP32 master weights and AdamW state could be read in under 2 s at that rate, which moves the bottleneck of checkpoint loading from storage to the copy into GPU memory and the slowest rank. Fire-Flyer AI-HPC (arXiv 2408.14158) describes the earlier cluster these designs grew from.

## Worked example

**Hops in a cube.** In a $4 \times 4 \times 4$ torus, from (0, 0, 0) to (3, 3, 3): along each axis $\min(3, 4 - 3) = 1$, so 3 hops; without wraparound 9. The worst case is the opposite corner of the half, (2, 2, 2): 6 hops.

**Bisection.** $4 \times 4 \times 4$: $\prod n_i / n_{\max} = 16$ rows cross the cut, each twice with wraparound: 32 links. A $16 \times 20 \times 28$ v5p pod: cut across the 28-axis, $2 \cdot 16 \cdot 20 = 640$ links.

**An all-gather.** Llama 3.1 8B's weights in BF16 are 16 GB. All-gathered over a 16-chip v5p ring axis at $9 \times 10^{10}$ B/s one way: $16 \cdot 10^9 \cdot \tfrac{15}{16} / (2 \cdot 9 \cdot 10^{10}) = 83$ ms. Over 4 chips: $16 \cdot 10^9 \cdot \tfrac34 / 1.8 \cdot 10^{11} = 67$ ms — smaller groups move less but not proportionally less, because $(n-1)/n$ saturates. (Arithmetic on published specifications; nothing measured.)

**A sharded matmul.** $A[I, J_X] \cdot B[J_X, K]$ on an $X$ axis of 4: each chip multiplies its $J/4$ slice and holds a partial $C[I, K]$; an all-reduce over $X$ of $|C|$ bytes finishes it. If only $C$'s rows are needed per chip, a reduce-scatter to $C[I_X, K]$ halves that cost.

## Shapes and cost

| Concept | GPU cluster (lessons 09.1–09.4) | TPU pod |
|---|---|---|
| fast domain | 8 GPUs on NVLink | a slice of up to a pod (thousands of chips) on ICI |
| slow domain | InfiniBand / RoCE between nodes | DCN between pods |
| a "TP group" | ranks inside a node | a mesh axis of the slice |
| sharding notation | per framework (DTensor placements: `Shard(0)`, `Replicate()`) | named axes: $A[I_X, J]$ |
| controller | multi-controller SPMD (`torchrun`) | single controller (Pathways) or multi-controller JAX |
| failure unit | a node | a cube or slice (optically switched out) |

DTensor's `Shard(0)` on a 1-D mesh is $A[I_X, J]$; a 2-D mesh `(dp, tp)` with `(Shard(0), Shard(1))` is $A[I_{dp}, J_{tp}]$ — the same idea in PyTorch's notation.

## Build it

The lab has four functions and one script; nothing here needs a TPU or JAX. `torus_hops` and `bisection_links` implement the topology formulas; `collective_time` the Scaling Book's per-axis cost model; `matmul_comm` the four sharding cases. `scaling_book.py` prints the worked numbers for the TPU v4 cube and the v4p and v5p pods, the all-gather comparison with an H100 node at lesson 09.1's assumed bandwidth, the four matmul cases, and checkpoint read times at 3FS's and Llama 3's published storage rates.

If you have JAX installed (not pinned by the course, and not run in this build), the notation maps directly onto `jax.sharding.NamedSharding(mesh, PartitionSpec("X", None))` for $A[I_X, J]$, and `jax.jit` inserts the collectives the table predicts; compiling a sharded matmul and reading its HLO is the Scaling Book's own exercise.

## What the evidence says

- **PUBLICLY DOCUMENTED (company figures):** TPU v4's OCS cost and power shares and its comparisons with the A100; Pathways' utilisation figures; Gemini 2.5's multi-pod data parallelism; DeepSeek's Tables 1–3 and its FP22 accumulation description; 3FS's throughput. Each is the vendor's measurement of its own system.
- **ESTABLISHED:** the bandwidth hierarchy argument (fast local domain, slow global one) holds on both platforms and decides layouts on both.
- **MODEL-SPECIFIC:** DeepSeek's node-limited routing and multi-plane network are choices for their cluster; the recommendations in the ISCA paper are requests, not results.
- **INFERENCE:** that TPU pods make very large TP or FSDP axes cheaper than GPU clusters do follows from the ICI vs NVLink-domain sizes above; actual layouts used for Gemini are not disclosed beyond data parallelism across pods.

## Lab

> [!NOTE]
> This extension lab computes from published specifications and has no comparison, so it has no experiment contract.

**Folder:** [`labs/module-09/lesson-05/`](../../labs/module-09/) · **Time:** about 45 minutes · **Pass check:** `pytest labs/module-09/lesson-05` passes and your answers to step 3 are written down.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | none needed (optional: a Cloud TPU v5e/v5p slice with JAX to compile a sharded matmul and read its HLO; not run in this build) | the steps below |
| Free GPU | none needed | the steps below |
| Free CPU | laptop; `scaling_book.py` runs in under a second | the steps below |

1. **Implement** `torus_hops`, `bisection_links`, `collective_time` and `matmul_comm` in `lab.py`; `pytest labs/module-09/lesson-05`.
2. **Print the numbers.**

   ```bash
   python labs/module-09/lesson-05/scaling_book.py
   ```

3. **Translate.** Write lesson 09.1's Llama 3 layout ([TP 8, CP 1, PP 16, DP 128]) as a TPU mesh: which axis would you give each dimension on a $16 \times 20 \times 28$ v5p pod, and which dimension must cross the DCN if the job spans two pods? Then write FSDP2 × TP 2 (lesson 09.3) in the Scaling Book's notation for one MLP: $W_{\text{in}}$, $W_{\text{out}}$ and the activations, with the collectives.

<details>
<summary>Hint for step 3</summary>

Give the most talkative dimension (TP) an axis inside the slice with short rings, PP an axis whose neighbours are adjacent (point-to-point between stage $s$ and $s+1$), and DP the rest; across pods only DP (gradients once per step) can tolerate DCN bandwidth — which is what Gemini 2.5 reports doing. For the MLP: $W_{\text{in}}[D_{dp}, F_{tp}]$ is all-gathered over $dp$ (FSDP) and then used as $X[B_{dp}, D] \cdot W_{\text{in}}[D, F_{tp}]$; $W_{\text{out}}[F_{tp}, D_{dp}]$ likewise, and the second matmul ends with a reduce-scatter (or all-reduce) over $tp$.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-09/lesson-05/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-09/lesson-05`.

</details>

## Common mistakes

- **Forgetting wraparound.** Without it the worst-case distance and the bisection are both halved in the wrong direction (more hops, fewer links).
- **Using a mesh axis twice in one array.** $C[I_X, K_X]$ is not a valid sharding; one operand has to be gathered first.
- **Comparing a TPU link figure with a GPU's total NVLink figure.** Per-link one-way, per-chip bidirectional and achieved bus bandwidth are three different numbers; say which one you use.
- **Quoting DeepSeek's recommendations as results.** The ISCA paper's measured tables and its wish list are different kinds of evidence.
- **Treating 3FS's 6.6 TiB/s as what one job gets.** It is an aggregate peak over 180 storage nodes and 500+ clients.

## References

- N. Jouppi et al., *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings*, abstract and sections on the OCS and topology. https://arxiv.org/abs/2304.01433
- P. Barham et al., *Pathways: Asynchronous Distributed Dataflow for ML*, abstract. https://arxiv.org/abs/2203.12533
- J. Austin et al., *How to Scale Your Model*, parts on TPUs and sharded matmuls (read 2026-10-04). https://jax-ml.github.io/scaling-book/
- DeepSeek-AI, *Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures*, Tables 1–3, sections 3.1.1, 4.1, 4.3, 5.1. https://arxiv.org/abs/2505.09343
- deepseek-ai, *3FS* README. https://github.com/deepseek-ai/3FS
- DeepSeek-AI, *Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning*. https://arxiv.org/abs/2408.14158
- Gemini Team, *Gemini 2.5* technical report, section 2.3. https://arxiv.org/abs/2507.06261

## Next

The [Module 9 project](../../projects/module-09-infrastructure-plan.md): an infrastructure plan for a 1T-parameter MoE model on a stated cluster.
