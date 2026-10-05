---
id: "10.1"
module: 10
minutes: 40
practice_minutes: 120
prerequisites: ["01.3", "01.4", "01.5", "04.3"]
objectives:
  - Write a provenance and licence record for every training source and check it automatically (pinned revision, licence obligations, file hashes, tokenizer).
  - Find exact and near-duplicate documents across training, validation and test splits, and measure benchmark leakage of Eval v0, Eval v1 and LAMBADA with an n-gram index whose recall is proven by a planted positive control.
  - Implement block-diagonal document masking for the course model, pass the correctness suite with it, and compare packing with and without it in a matched, seed-paired ablation.
  - Build a deterministic, resumable mixture sampler whose per-source token accounting is exact and known before the run starts, and show that a stopped and resumed mixture run is bit-identical to an uninterrupted one.
volatility: concept
sources:
  - title: "Llama Team, The Llama 3 Herd of Models (section 3.2: document attention mask)"
    url: https://arxiv.org/abs/2407.21783
  - title: "Phi-4 Technical Report (appendix B.1: hybrid 13-gram and 7-gram decontamination against 19 benchmarks)"
    url: https://arxiv.org/abs/2412.08905
  - title: "Muennighoff et al., Scaling Data-Constrained Language Models (abstract: up to 4 epochs of repetition)"
    url: https://arxiv.org/abs/2305.16264
  - title: "Penedo et al., The FineWeb Datasets (section 3: MinHash deduplication)"
    url: https://arxiv.org/abs/2406.17557
  - title: "HuggingFaceFW/fineweb-edu dataset card (ODC-By 1.0)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - title: "wikimedia/wikipedia dataset card (CC BY-SA 3.0 and GFDL)"
    url: https://huggingface.co/datasets/wikimedia/wikipedia
  - title: "PyTorch FlexAttention (block masks for document masking)"
    url: https://pytorch.org/blog/flexattention/
last_verified: "2026-10-04"
---

# 10.1 · Data integrity at scale

Every data experiment in this module changes what the model reads, so before any of them can be trusted the data pipeline itself has to be checked: where every token came from and under what licence, whether any evaluation item is also a training document, whether packed windows let one document read another, and whether the stream of training tokens is exactly the one the experiment contract describes, including after a restart. This lesson builds those checks for the course data and runs the one ablation the pipeline itself raises: packing with and without document masking.

## Why this matters at a frontier lab

Data bugs are the quietest bugs in pretraining. A validation document that is a near-copy of a training document makes held-out loss a memorisation score, and every data ablation judged by it rewards whatever repeats that document. A benchmark passage in the training data turns a capability claim into a leakage claim. A mixture sampler that restarts its random state after a preemption silently changes the mixture, and its run card still says 60/25/15. And a source without a licence record cannot be shipped in a model at all, however good the ablation was. None of these change the training loss curve, which is why they have to be checked by construction rather than noticed.

## The idea

The parent course built the pieces: packing documents into windows separated by `<|endoftext|>` (parent lesson 04.3), exact resume with every RNG state (parent 07.3), heuristic and model-based filtering, MinHash deduplication inside one corpus and n-gram contamination checks (parent 11.1–11.3), and the block-diagonal mask for packed fine-tuning sequences (parent 14.2). This lesson does not re-teach them. It turns them into four checks that run on every Module 10 mixture, plus one change to the course model.

### Provenance and licence records

A **source record** is the answer to five questions for one prepared folder: which dataset, config and **pinned revision** (a 40-character commit, never `main`); which **licence** and what it obliges (attribution, share-alike); the exact command that produced it; how many documents and tokens per split; and the SHA-256 of every file, so a later run can prove it read the same bytes. `python -m frontierlab.datax.sources prepare NAME` writes one for every Module 10 source, plus a per-document provenance line (dataset id, URL, SHA-1 of the normalised text) so that a takedown request for one URL can be honoured. A **manifest** lists the records of a mixture with its weights and goes next to the run card of every run trained on it.

The Module 10 sources and their licences, from their dataset cards (PUBLICLY DOCUMENTED, checked 2026-10-04):

| Source | Dataset @ revision | Licence | Obligation |
|---|---|---|---|
| `edu` (Data-v0) | `HuggingFaceFW/fineweb-edu` sample-10BT @ `87f0914` | ODC-By 1.0 | attribution |
| `web` | `HuggingFaceFW/fineweb` sample-10BT @ `9bb295d` | ODC-By 1.0 | attribution |
| `wiki` | `wikimedia/wikipedia` 20231101.en @ `b04c8d1` | CC BY-SA 3.0 and GFDL | attribution, **share-alike** |
| `math` | `HuggingFaceTB/finemath` finemath-4plus @ `e92b25a` | ODC-By 1.0 | attribution |
| `annot` (10.2) | `HuggingFaceFW/fineweb-edu-llama3-annotations` @ `72df4c9` | ODC-By 1.0 | attribution; labels written by Llama-3-70B-Instruct |

`frontierlab.datax.provenance.check_source` turns the record into findings. **BLOCK**: unpinned revision, no licence, a licence not in the reviewed table, files whose hash differs from the record, a different tokenizer. **WARN**: share-alike, a held-out split used as a training source, a token count that disagrees with the file size. The licence table in the code records what each licence is commonly understood to require; it is engineering bookkeeping, not legal advice, and an unknown licence is a block until a person has looked (REASONABLE INDUSTRY PRACTICE).

### Exact and near-duplicates across splits

Data-v0 assigns each document's split from the hash of its normalised text (lesson 01.1's data preparation): `sha1(text) mod 1000 < 10` is validation, `< 20` test. Every Module 10 source is prepared **with the same rule and Data-v0's tokenizer**, so an exact copy of a validation document lands in validation in every source, and no exact duplicate can cross splits, whichever source it comes from.

A near-duplicate has a different hash and its own split. Let $S(d)$ be the set of word 5-gram shingles of document $d$ and

$$J(a, b) = \frac{|S(a) \cap S(b)|}{|S(a) \cup S(b)|}$$

their Jaccard similarity. MinHash keeps, for each of $P$ hash functions $h_p$, the minimum $\min_{x \in S(d)} h_p(x)$; two documents agree on entry $p$ with probability exactly $J(a,b)$. LSH splits the $P$ entries into $b$ bands of $r$ rows; two documents become a candidate pair if any whole band agrees, with probability

$$\Pr[\text{candidate}] = 1 - \left(1 - J^{\,r}\right)^{b}.$$

The cross-split check keeps only candidate pairs whose two documents are in **different** splits (or one in a training source and one in a held-out set) and verifies each with the exact $J$. The parent course used MinHash to remove duplicates *inside* a corpus; here the question is narrower and more important for evaluation: does any held-out document have a twin in training?

### Benchmark leakage

An **n-gram index** holds the sorted unique 64-bit hashes of every word $n$-gram of the training data ($n = 13$ by default; 8 bytes per training word). An evaluation item with $m$ word $n$-grams $g_1..g_m$ has overlap

$$o = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}[g_i \in \text{index}],$$

and is flagged when $o \ge \tau$. Definitions differ between reports — Phi-4 (appendix B.1) used a hybrid of 13-gram and 7-gram matching against 19 benchmarks — so every number in the report is stored with its $n$ and $\tau$. Items shorter than $n$ words cannot be checked this way and are counted separately instead of being called clean. The checker is only trusted after a **planted positive control**: copies of 20 benchmark passages are inserted into a copy of the training texts, and all 20 must be flagged.

The course runs it against three evaluations: Eval v0's held-out windows (validation documents) and LAMBADA (lesson 01.1's project), and Eval v1's haystack documents (the long validation documents of lesson 04.1).

### Packing with and without document masking

Packing concatenates documents and cuts fixed windows; inside a window, a token can attend to the end of an unrelated earlier document. Let $q_i$ and $k_j$ be the absolute positions of query $i$ and key $j$, and $\text{seg}(t)$ the document index of position $t$. Plain causal attention allows the pair when $k_j \le q_i$; **document masking** allows it only when also

$$\text{seg}(k_j) = \text{seg}(q_i).$$

The allowed pairs form a block-diagonal lower triangle. For a window whose documents have lengths $n_1, \dots, n_m$ (summing to $T$), the fraction of causal pairs the mask keeps is

$$f = \frac{\sum_i n_i (n_i + 1)}{T (T + 1)}.$$

The mask needs no change to positions. RoPE makes the score $q_i \cdot k_j$ a function of $i - j$, so a document that starts at offset 300 of a window produces exactly the scores it would produce at offset 0: packed and masked logits equal the logits of each document run on its own. The course model's test checks this in float64.

What the reports say about its value is mixed. Llama 3 (section 3.2) uses "an attention mask that prevents self-attention between different documents within the same sequence" and found it "had limited impact during in standard pre-training, but ... important in continued pre-training on very long sequences" (the "during in" is in the original). That is a company statement without numbers, so the lab measures it at course scale.

### Deterministic, resumable mixtures with exact token accounting

A mixture is a list of sources with token weights. The course sampler (`frontierlab.datax.mixture.MixtureSampler`) makes everything about training window $k$ a pure function of $k$:

1. **Block schedule.** Windows come in blocks of $P$ (default 100). The weights $w_s$ are rounded to integer window counts $n_s$ per block by largest remainder, so $\sum_s n_s = P$. Block $b$'s slot order is a permutation drawn from the mixture seed and $b$. Window $k$ is slot $j = k \bmod P$ of block $b = \lfloor k / P \rfloor$; its source is `order_b[j]`, and its index in that source's own stream is $b \cdot n_s$ plus the number of earlier slots of block $b$ that belong to $s$.
2. **Per-source stream.** Each source is an infinite sequence of epochs; epoch $e$ is the source's documents in a permutation drawn from the seed, the source index and $e$, concatenated. Source window $w$ is tokens $[wT, (w+1)T)$ of that stream.

Three properties follow. **Resume is a seek**: the state is one integer, the windows consumed, which the wrapper recomputes from the checkpoint's step ($k = \text{step} \times \text{grad\_accum} \times B$). **Accounting is exact and known in advance**: after $k$ windows source $s$ has supplied exactly $T \cdot (n_s \lfloor k/P \rfloor + \text{its slots among the first } k \bmod P \text{ of block } \lfloor k/P \rfloor)$ tokens, so the run card can state the per-source tokens, epochs and documents of the whole run before step 1. **Repetition is visible**: epochs per source are reported, and more than 4 are flagged, the point beyond which Muennighoff et al. (abstract) find that repeated data loses value relative to fresh data.

## Worked example

### Segments, mask and kept pairs

Documents start at stream offsets 0, 5 and 7. A window of $T = 6$ tokens from offset 3 covers stream positions 3–8, which belong to documents 0, 0, 1, 1, 2, 2. Counted from 0 in the window, the segment ids are `[0, 0, 1, 1, 2, 2]`. The masked attention matrix (rows = queries, 1 = allowed):

```text
1 0 0 0 0 0
1 1 0 0 0 0
0 0 1 0 0 0
0 0 1 1 0 0
0 0 0 0 1 0
0 0 0 0 1 1
```

Kept fraction: $f = (2 \cdot 3 + 2 \cdot 3 + 2 \cdot 3) / (6 \cdot 7) = 18/42 = 0.43$; plain causal attention would have used 21 pairs, the mask uses 9. A window with one document has $f = 1$.

### Mixture schedule

Weights $(0.5, 0.3, 0.2)$ with a block of 7 windows: exact shares $(3.5, 2.1, 1.4)$, floors $(3, 2, 1)$ leave one window, which goes to the largest remainder (0.5): $n = (4, 2, 1)$, realised weights $(0.571, 0.286, 0.143)$. With $P = 100$ the same weights are realised exactly.

Locating a window: $P = 10$, $n = (6, 4)$, and block 3's slot order is `[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]`. Window $k = 34$ is block 3, slot 4, source 1. Source 1 supplied $3 \cdot 4 = 12$ windows in blocks 0–2 and one more earlier in block 3 (slot 1), so window 34 is source 1's window 13. After $k = 34$ windows the accounting is source 0: $3 \cdot 6 + 3 = 21$ windows, source 1: $12 + 1 = 13$.

### LSH thresholds

With $P = 128$ hash functions in $b = 16$ bands of $r = 8$: a pair with $J = 0.8$ becomes a candidate with probability $1 - (1 - 0.8^8)^{16} = 1 - 0.832^{16} = 0.947$; a pair with $J = 0.5$ with probability $1 - (1 - 0.0039)^{16} = 0.061$. The S-curve's midpoint is near $(1/b)^{1/r} = 0.707$. Candidates are then verified with the exact Jaccard, so false candidates cost time, not correctness; missed pairs are the real risk, and they are rare above $J = 0.8$.

### Leakage overlap

A LAMBADA passage of 20 words has $20 - 13 + 1 = 8$ word 13-grams. If 6 are in the training index, $o = 0.75 \ge 0.5$ and it is flagged. A 12-word item has no 13-grams: "too short", not clean.

## Shapes and cost

| Object | Shape, dtype, device | Size at course scale |
|---|---|---|
| training batch | (B, T) int64, CPU or GPU | CPU (16, 128) |
| segment ids | (B, T_total) int64, same device as the batch | indexed by absolute position, so cached decoding works |
| document mask | (B, 1, T_q, T_k) bool | B·T² bytes: (16, 1, 128, 128) is 256 KiB; main path (32, 1, 2048, 2048) is 128 MiB per layer call |
| MinHash signatures | (documents, 128) uint64, CPU | 1 KiB per document |
| n-gram index | sorted unique uint64, CPU | 8 bytes per training word (measured below) |
| mixture state | one int (windows consumed) | 8 bytes |

Two costs matter. **The mask does not save FLOPs here.** `F.scaled_dot_product_attention` with a boolean mask computes every (query, key) score and discards the masked ones, and on CPU it is slower than the `is_causal` fast path (measured in the lab). The saving $1 - f$ appears only with kernels that skip whole masked blocks: FlexAttention block masks (a mask of $(T/128)^2$ block flags per sequence instead of $T^2$ booleans) or FlashAttention's variable-length interface, which takes cumulative document lengths. That is the main-path implementation; the course mask path is the readable reference that the kernels must match. **The near-duplicate check is $O(\text{documents} \times P \times \text{shingles})$** for the signatures plus the candidate verification; the n-gram index is one sort of all training n-gram hashes.

## Build it

```python
from frontierlab.datax.mixture import MixtureSampler, MixtureSpec, SourceRef
from frontierlab.datax.packing import document_segments
from frontierlab.model import LM, toy

spec = MixtureSpec([SourceRef("edu", "labs/common/data/v0", 0.60),
                    SourceRef("web", "labs/common/data/m10/web", 0.25),
                    SourceRef("wiki", "labs/common/data/m10/wiki", 0.15)], seed=0)
s = MixtureSampler(spec)
x, seg, src = s.next_windows(16, 128)          # (16, 128) int64, (16, 128) int64, list of source ids
print(s.accounting(300 * 16, 128)["sources"]["web"]["tokens"])   # exact, before training

m = LM(toy(vocab_size=8192).with_(attention="gqa-docmask"))      # same parameters as "gqa"
with document_segments(seg):
    loss = m(x, labels=x).loss                                     # block-diagonal causal attention
```

```bash
python -m frontierlab.datax.train --mixture mix.json --doc-mask --run runs/m10/a --preset toy --steps 300
python -m frontierlab.datax.provenance check labs/common/data/v0 labs/common/data/m10/web
python -m frontierlab.datax.integrity --train edu=labs/common/data/v0 web=labs/common/data/m10/web --eval-v1 labs/common/data/v0-long
```

`frontierlab.datax.train` wraps `frontierlab.train.loop` the way `frontierlab.longctx.extend` does: it replaces the loop's model factory (attention `"gqa-docmask"` with `--doc-mask`), its training data (the mixture sampler, seeked to the checkpoint's step on restart), its validation loss (with the mask when the model was trained with it), its FLOP count (the mask kind counts as GQA) and its run card (a `datax` block with the mixture, its digest and the planned accounting). The native loop flags are proposed in `curriculum/inbox/module-10-shared-changes.md`.

Correctness checks, in `labs/common/tests/test_datax.py`: packed and masked logits equal each document run alone (float64, max difference below $10^{-10}$), and differ without the mask; `"gqa-docmask"` with no segments set is bit-identical to `"gqa"`; the causal check and cached-decode agreement (one token and five-token chunks) pass with segments set; a gradient check through the masked attention; a stopped-and-resumed sampler gives bit-identical windows and segments; accounting after 57 windows equals a direct count of the sources returned; segments equal those recomputed from end-of-text tokens; a first epoch of a source is a permutation of its documents; the wrapper's stopped-and-resumed run ends with bit-identical weights and the same accounting as planned; a planted near-duplicate across splits is found and a planted leak of 5 items flags exactly those 5.

## What the evidence says

- **Cross-split deduplication and benchmark decontamination: ESTABLISHED** practice. Methods are PUBLICLY DOCUMENTED in many reports (for example Phi-4 appendix B.1); exact definitions (word or token n-grams, $n$, threshold) differ, so numbers from different reports are not comparable.
- **Provenance and licence records per source: REASONABLE INDUSTRY PRACTICE.** Open data releases document licences on their cards; what closed labs record internally is not public.
- **Document masking in standard pretraining: PROMISING at best.** Llama 3 reports limited impact for standard pretraining and importance for long continued pretraining (company claim, no numbers). The course measurement below is at 1.8M parameters and 512 tokens and says nothing about either regime at scale.
- **Repetition up to about 4 epochs is close to fresh data: ESTABLISHED** for the settings of Muennighoff et al. (abstract); lesson 10.2 shows why it matters for filtering.
- **Course measurement (free CPU, 2026-10-04, Windows 11 laptop, 16 threads, torch 2.14.1 CPU, other jobs running).** Integrity of Data-v0 + web + wiki + math (50,898 training documents) against Data-v0's 210 validation and 190 test documents: 0 exact cross-split duplicates in every source, 0 verified near-duplicates (2 LSH candidates, both below Jaccard 0.8), 0 of 256 Eval v0 windows and 0 of 5,153 LAMBADA passages flagged, 1 of 648 Eval v1 haystack documents flagged, planted-control recall 20/20; 23 minutes, 321 MiB n-gram index Document masking, 3 seeds per arm at 512 tokens: per-document loss (mask − no mask) +0.012, 95% CI [−0.031, +0.055]; the decision is **inconclusive** (details in the lab).

## Lab

**Folder:** [`labs/module-10/lesson-01/`](../../labs/module-10/) · **Time:** about 2 hours (about 60 minutes of it unattended) · **Pass check:** `pytest labs/module-10/lesson-01` passes; the integrity report has zero exact cross-split duplicates and a planted-control recall of 1.0; `resume_check.py` prints "bit-identical: True"; your write-up applies the contract's decision rule to the ablation.

### Experiment contract

- **Question:** at equal tokens, does document masking change per-document held-out loss? Decision informed: whether Data-v1's packing uses document masking by default.
- **Hypothesis:** no difference larger than 0.01 nats at this scale (Llama 3: limited impact in standard pretraining). Status: reported effect (company claim, no numbers); may not appear at this scale.
- **Baseline:** packing without a mask (`nomask`), the loop's own random windows, learning rate 3e-3 (the course default for the toy preset; not re-tuned, and neither is the masked arm).
- **Changed variable:** `--doc-mask`. **Controlled:** Data-v0, the loop's random windows (same `--seed` → same windows in the same order and the same initial weights in both arms, because the two attention kinds have identical parameters), toy preset, 300 steps of 4 × 512 tokens, warmup 30, cosine, seeds 0–2, evaluation on the same 200 validation documents.
- **Comparison axis:** equal tokens. It does not answer which arm is cheaper: on CPU the mask path is slower; with a block-skipping kernel the masked arm would be cheaper.
- **Budget:** free CPU, 6 runs, about 30 minutes measured (3–8 minutes per run with other jobs running).
- **Metrics and decision rule:** primary: mean per-document loss over the first 512 tokens of 200 validation documents, each document scored on its own (where masked and unmasked attention see identical context); seed-level paired 95% t-interval of (mask − no mask). Adopt masking as the Data-v1 default if the upper bound is $\le +0.01$ nats (non-inferior); reject if the lower bound is $> +0.01$; else inconclusive. Secondary: Eval v0 windows, each arm scored as it was trained; training throughput.
- **Correctness checks:** `pytest labs/common/tests/test_datax.py -k docmask` passes; the run cards of each seed pair differ only in `config.attention` and `datax.doc_mask` (`python -m frontierlab.record runs/m10/l101/nomask-s0 runs/m10/l101/mask-s0 --changed config.attention datax.doc_mask`).
- **Fallback evidence:** none; a null result is a valid result.
- **Limits:** a 1.8M-parameter model, 512-token windows on documents with a median of about 700 tokens, 0.6M tokens per run; nothing about long-context continued training, where Llama 3 says it matters.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 10 pilot | `docmask_ablation.py --variant main` (Baseline-0, 4,000 steps of 32 × 2,048 tokens, 3 seeds × 2 arms). PROJECTED: 6 runs × 2.62e8 tokens × 0.845 GFLOP/token (`flops_per_token(baseline0, 2048)`) = 1.33e18 FLOPs ≈ 1.25 GPU-hours at an assumed 30% MFU (about 0.2 h per run), counting the masked arms at the unmasked FLOPs; the masked arms on the dense mask path are slower, so implement them with FlexAttention first and check them against `"gqa-docmask"` with `frontierlab.testing.equivalence` |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `docmask_ablation.py --variant t4` (pilot-10m, 2,000 steps of 16 × 1,024) |
| Free CPU | laptop, measured 2026-10-04 with other jobs running: integrity report 23 min, `resume_check.py` 5 min, `docmask_ablation.py` 6 runs of 3–8 min (about 30 min) | the steps below |

### Steps

1. **Implement** the eight TODOs in `lab.py` (segments, mask, largest-remainder counts, the window locator, MinHash signature, LSH candidates across splits, n-gram overlap, the decision rule). Run `pytest labs/module-10/lesson-01`.
2. **Prepare the sources** (once for the whole module; sizes and times measured in this build):

   ```bash
   python -m frontierlab.datax.sources prepare web --docs 20000     # 38 MB, 101 s
   python -m frontierlab.datax.sources prepare wiki --docs 4000     # 36 MB, 43 s
   python -m frontierlab.datax.sources prepare math --docs 8000     # 30 MB, 29 s
   python -m frontierlab.datax.provenance check labs/common/data/v0 labs/common/data/m10/web labs/common/data/m10/wiki labs/common/data/m10/math
   ```

   Read the WARN line for Wikipedia and write one sentence on what share-alike means for a model trained on it and for a released dataset that contains it.
3. **Integrity report** (unattended):

   ```bash
   python -m frontierlab.datax.integrity --train edu=labs/common/data/v0 web=labs/common/data/m10/web wiki=labs/common/data/m10/wiki math=labs/common/data/m10/math --eval-v1 labs/common/data/v0-long --out runs/m10/integrity.json
   ```

   Look at every near-duplicate pair it prints. Decide for each whether it is a real leak, and what you would do (drop the held-out document, drop the training document, or keep both and say so).
4. **Resume check:** `python labs/module-10/lesson-01/resume_check.py --doc-mask`. It must print identical accounting, a loss difference of 0.0 and bit-identical weights.
5. **Ablation** (unattended): `python labs/module-10/lesson-01/docmask_ablation.py`, then apply the decision rule and report the throughput of both arms.

<details>
<summary>Hint for TODO 4 (locate)</summary>

`b, j = divmod(k, block)`; `order = block_order(b)`; `s = int(order[j])`; the window index is `b * counts[s]` plus `(order[:j] == s).sum()`.

</details>

<details>
<summary>Hint for TODO 6 (LSH candidates)</summary>

For each band, slice `sigs[:, b*r:(b+1)*r]`, use each row's `.tobytes()` as a dict key, collect row indices per key, and add every pair in a bucket whose `groups` differ as `(min(i, j), max(i, j))`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04 on the build laptop (torch 2.14.1 CPU; other jobs shared the CPU, so the wall times of later runs are inflated). Six runs of 300 steps × 4 × 512 tokens, 181–511 s each; scoring a few seconds per run.

| Seed | no mask | mask | mask − no mask | paired over 200 documents |
|---|---|---|---|---|
| 0 | 6.4478 | 6.4402 | −0.0075 | [−0.0121, −0.0034] |
| 1 | 6.4463 | 6.4713 | +0.0250 | [+0.0212, +0.0287] |
| 2 | 6.3737 | 6.3928 | +0.0192 | [+0.0151, +0.0234] |

Seed-level (mask − no mask): +0.0122, 95% CI [−0.0309, +0.0553] over 3 seeds; Eval v0 windows, each arm scored with its own attention: +0.0142 [−0.0207, +0.0491]. Decision under the contract: **inconclusive** (the upper bound exceeds +0.01). What it shows. (1) The per-seed document-paired intervals are narrow and disagree in sign: each says something about two fixed models, nothing about the method, exactly lesson 01.4's point. (2) The noise is larger than in the Module 1 noise floor because the batch is 4 windows (seed std of the no-mask arm 0.042 nats). (3) Power: the standard deviation of the paired differences is 0.017, so resolving a 0.01 margin at 80% power needs about $((1.96 + 0.84) \cdot 0.017 / 0.01)^2 \approx 24$ seeds per arm; this design could not have answered the question, and the report must say so rather than call it "no effect". (4) Throughput, seed 0 (both arms ran with the least contention): 4,165 tokens/s without the mask, 3,411 with it, 18% slower on the dense mask path. Data-v1 keeps the unmasked default on this evidence and records masking as untested at long context, where Llama 3 says it matters.

Integrity report: 0 BLOCK findings; one WARN (Wikipedia is share-alike). Exact cross-split duplicates: 0 for every source and both held-out splits, as the shared hash rule guarantees. Near-duplicates: the LSH step produced 2 candidate pairs, neither reached Jaccard 0.8 when verified, so no held-out document has a near-copy in any of the four training sources. Leakage (word 13-grams, flagged at overlap ≥ 0.5): Eval v0 windows 0/256 (2.7% of windows share at least one 13-gram with training: common phrases), LAMBADA 0/5,153 (not a single shared 13-gram), Eval v1 haystack documents 1/648. The flagged one (validation document 633 of `v0-long`) is an NRICH-style list of geometry problems; no single training document covers more than 9% of its 13-grams, but many templated "Search by Topic" pages in FineMath together cover more than half of it. Eval v1's answers are synthetic statements, so this touches only its natural-text components; the right action is to record it, drop that document from the natural-text set for Data-v1 runs, and keep the rest. Planted control: 20 of 20 planted LAMBADA passages flagged. Times: exact 68 s, near-duplicates 282 s, leakage 1,041 s (the index holds 321 MiB of 13-gram hashes for about 72M training tokens), total 23 minutes with other jobs running. `resume_check.py --doc-mask`: planned and consumed accounting identical (edu 73,984 tokens, web 30,720, wiki 18,176 for 60 steps × 16 × 128; realised weights exactly 0.60/0.25/0.15), largest logged-loss difference 0.0, final weights bit-identical; 282 s.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-10/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-10/lesson-01`.

</details>

## Common mistakes

- **Splitting after deduplication by document index.** A split by position or by random draw puts exact copies on both sides; split by a hash of the content, before anything else, and prepare every source with the same rule.
- **Calling short items clean.** An item shorter than $n$ words has no $n$-grams; report it as unchecked.
- **Trusting a leakage checker that has never found anything.** Plant items and measure recall before reading a zero.
- **Resuming a mixture by re-seeding a generator.** The run continues with a different source order and the per-source tokens in the run card are no longer true. Make the stream a function of the window counter, or save the full sampler state.
- **Expecting the mask to make training faster.** With a dense boolean mask it makes it slower; the saving needs a block-sparse kernel.
- **Leaving training segments set during evaluation.** A context variable set by the last training batch would mask the evaluation batch with the wrong documents; `frontierlab.datax.train` scores masked models through their own path and the attention kind refuses segments that do not match the batch.
- **A source with a branch name as its revision.** `main` today is not `main` next month; the provenance check blocks it.

## References

- Llama Team, Meta, *The Llama 3 Herd of Models*, 2024, section 3.2. https://arxiv.org/abs/2407.21783
- Microsoft, *Phi-4 Technical Report*, 2024, appendix B.1. https://arxiv.org/abs/2412.08905
- N. Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023, abstract. https://arxiv.org/abs/2305.16264
- G. Penedo et al., *The FineWeb Datasets*, 2024. https://arxiv.org/abs/2406.17557
- Dataset cards (licences): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, https://huggingface.co/datasets/HuggingFaceFW/fineweb, https://huggingface.co/datasets/wikimedia/wikipedia, https://huggingface.co/datasets/HuggingFaceTB/finemath
- PyTorch team, *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*, 2024. https://pytorch.org/blog/flexattention/
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[10.2 · Model-based quality filtering](lesson-02.md)
