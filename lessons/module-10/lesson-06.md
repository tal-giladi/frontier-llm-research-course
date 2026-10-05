---
id: "10.6"
module: 10
minutes: 30
practice_minutes: 45
prerequisites: ["10.1", "10.2"]
objectives:
  - Measure the fertility cost of a tokenizer on languages it was not trained for, and explain how it changes sequence length, compute and the meaning of per-token loss.
  - Transfer a heuristic filter threshold to another language with FineWeb2's Quantile method and per-language stop-word lists, and show what an English rule does to other languages.
  - Compute FineWeb2-style rehydration weights from MinHash cluster sizes and filter removal rates, and say when the estimates are too noisy to use.
  - Split code and agent-task data by repository and task family instead of by file or instance, and measure the leakage a file-level split creates.
volatility: concept
sources:
  - title: "Penedo et al., FineWeb2: One Pipeline to Scale Them All (sections 4.3 deduplication, 4.4.1 stop words, 4.4.2 threshold selection, 4.5 rehydration, 5)"
    url: https://arxiv.org/abs/2506.20920
  - title: "Yang et al., SWE-smith: Scaling Data for Software Engineering Agents (abstract, section 2.1, appendix A.2)"
    url: https://arxiv.org/abs/2504.21798
  - title: "HuggingFaceFW/fineweb-2 dataset card (ODC-By 1.0)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
  - title: "SWE-bench/SWE-smith dataset (MIT; revision ea6d717)"
    url: https://huggingface.co/datasets/SWE-bench/SWE-smith
last_verified: "2026-10-04"
---

# 10.6 · Multilingual and code data

Extension: the course data is English web text, but frontier mixtures are not, and two of their largest non-English parts break assumptions the earlier lessons relied on. Text in other languages breaks the English filters and the English tokenizer; code and agent tasks break the document as the unit of splitting, because near-copies live across files of one repository and across task instances of one bug. This lesson measures both on small pinned samples and ends with the per-language and per-repository rules a pipeline needs; it has no training run.

## Why this matters at a frontier lab

Every frontier model is multilingual and trained on code, and the published pipelines treat both as separate data products with their own rules: FineWeb2 adapts each filter threshold per language (section 4.4) and SWE-smith removes every repository that appears in its evaluation benchmark before generating a single task (section 2.1). A pipeline that applies English heuristics to Hebrew deletes most of it; one that splits code by file reports a held-out score that partly measures memorised copies. Both mistakes look like normal numbers.

## The idea

### Tokenizer fertility

**Fertility** is how many tokens a tokenizer needs per unit of text: per UTF-8 byte or per word. A tokenizer trained on English splits other languages into many short pieces, and a byte-level BPE falls back to single bytes for scripts it rarely saw (Hebrew letters are 2 bytes each in UTF-8). Higher fertility means longer sequences for the same text, so more compute per document and less text per context window, and a per-token loss that is not comparable across languages (each token carries less). Compare languages by bits per byte, not by loss per token.

### FineWeb2's per-language pipeline

FineWeb2 (PUBLICLY DOCUMENTED) builds one pipeline that adapts to every language:

- **Deduplication first** (section 4.3): MinHash with FineWeb's settings ("14 buckets of size 8, with 5-grams") over word n-grams from per-language word tokenizers, globally per language, *before* filtering; the size of each duplicate cluster is stored with the kept document.
- **Stop words per language** (section 4.4.1): stop words are the words "exceeding a set frequency threshold" in reference data for that language, and documents need "at least 2 words from the stopwords list", as in the Gopher rules for English.
- **Thresholds per language** (section 4.4.2): several ways to move an English threshold to another language were compared by training "207 ablation models, each trained for 29B tokens"; one of them, **Quantile**, sets each language's threshold "so as to remove the same fraction of data as the English threshold removes in English". The final choice differs per filter group (10Tail and Quantile computed on Wikipedia for two groups, MeanStd on Common Crawl for the repetition filters).
- **Rehydration** (section 4.5): because both singletons and huge clusters were removed more often by the filters, cluster size is a quality signal. Each cluster size gets an upsampling weight from the filter's removal rate at that size: "a weight of 10 ... to the cluster size with the smallest removal rate, and a weight of 1 to every cluster size above the global removal rate", with interpolation in between. Every step improved models trained on 350B tokens (Figure 1). The result covers 1,868 language-script pairs from 96 Common Crawl snapshots, 20 TB of text (section 5).

### Splits for code and tasks

The unit of a split must be the unit of duplication. For web text that is the document (its hash decides the split). For code it is at least the **repository**: files of one repository share licence headers, vendored copies, generated code and "copied from" blocks, so a file-level split puts near-copies on both sides. For agent tasks it is the repository and the **task family**: SWE-smith generates hundreds to thousands of task instances per repository by breaking existing tests with several strategies (LM rewrites, procedural AST modifications, combined patches, reverted pull requests; section 2.1), so two instances of the same repository often share most of a patch. SWE-smith's own evaluation hygiene is repository-level: it targets popular PyPI packages and removes "all 12 SWE-bench test repositories from consideration" (section 2.1), and it keeps only repositories whose licence allows the use (appendix A.2). The dataset: 50k instances from 128 repositories in the paper (abstract); the pinned dataset revision used here contains more rows and repositories in other languages than the paper describes (Go repositories appear in its later shards), so state the revision with every number.

## Worked example

### Fertility

The Hebrew word "שלום" is 4 letters, 8 UTF-8 bytes. A byte-level tokenizer that never merged Hebrew bytes needs 8 tokens; one that learned Hebrew merges may need 1 or 2. At 8 tokens per word a 2,048-token window holds about 256 Hebrew words, against about 1,300 English words at 1.56 tokens per word (Data-v0's tokenizer on English, measured below).

### The Quantile threshold

English stop-word ratios of 100 documents are the values 0.00, 0.01, ..., 0.99 (to make it easy); the English rule "ratio ≥ 0.05" removes 5 of them, 5%. A target language whose ratios are spread differently gets the threshold at its own 5th percentile, so the rule removes 5% there too, whatever the absolute values are.

### Rehydration weights

Removal rates by cluster size 1, 2, 3: 0.6, 0.1, 0.5; global rate 0.4. Size 2 has the lowest rate: weight 10. Sizes 1 and 3 are above the global rate: weight 1. A size with rate 0.25 would get $1 + 9 \cdot (0.4 - 0.25)/(0.4 - 0.1) = 5.5$.

## Shapes and cost

| Object | Shape, dtype | Notes |
|---|---|---|
| per-document metric | (documents,) float64 | stop-word ratio, mean word length |
| cluster sizes | (documents,) int64 from FineWeb2's `minhash_cluster_size` field | bucketed 1..5 and 6+ here |
| code files | text, grouped by package / directory / file | MinHash per file, 5-word shingles |
| SWE-smith patches | text per instance | one 4.1 MB parquet shard |

Everything runs on CPU in a few minutes (measured below). The expensive part of the real pipelines is deduplication and language identification over billions of documents, which the course does not repeat.

## Build it

```python
from frontierlab.datax import groups
groups.fertility(lambda s: tok.encode(s).ids, texts)             # tokens per byte / word / char
thr = groups.quantile_threshold(eng_values, 0.157, heb_values)    # FineWeb2 "Quantile"
groups.rehydration_weights(cluster_sizes, removed)                # FineWeb2 section 4.5
splits = groups.group_split(repo_names)                           # every file of a repository in one split
groups.cross_split_rate(file_texts, splits, threshold=0.7)        # held-out items with a near-copy in train
```

`frontierlab.datax.sources prepare fw2-fra|fw2-deu|fw2-heb` streams 2,000 documents of each language from FineWeb2 at a pinned revision, keeps FineWeb2's `minhash_cluster_size` in the provenance rows, and tokenizes with Data-v0's tokenizer like every other source. Tests: `pytest labs/common/tests/test_datax.py -k groups`.

## What the evidence says

- **Per-language thresholds and stop words instead of English ones: ESTABLISHED** for multilingual pipelines (FineWeb2 sections 4.4.1–4.4.2 with 207 ablation models; earlier multilingual corpora made similar adaptations).
- **Rehydration by cluster size: PROMISING** (FineWeb2 section 4.5, one group; the weights are dataset-dependent, as the paper says).
- **Repository-level splits and benchmark-repository exclusion for code and agent tasks: ESTABLISHED practice** (SWE-smith section 2.1; REASONABLE INDUSTRY PRACTICE wherever evaluation repositories are known).
- **Course measurement (free CPU, 2026-10-04, other jobs running; part A 173 s, part B 41 s):**

  | Language (2,000 train documents) | bytes/char | Data-v0 tokenizer: tokens/byte, tokens/word | Qwen3 tokenizer: tokens/byte, tokens/word |
  |---|---|---|---|
  | English (Data-v0) | 1.00 | 0.257, 1.56 | 0.217, 1.32 |
  | French (FineWeb2) | 1.04 | 0.405, 2.47 | 0.269, 1.64 |
  | German | 1.02 | 0.449, 3.18 | 0.279, 1.98 |
  | Hebrew | 1.75 | 0.980, 9.68 | 0.226, 2.23 |

  Data-v0's English tokenizer is almost a byte tokenizer for Hebrew (0.98 tokens per byte, 9.7 per word): 2,000 Hebrew documents took 11.5M tokens, more than the 1,952 German ones (2.6M) and French ones (2.7M) together. Gopher's English stop-word rule (at least 2 of "the, be, to, of, and, that, have, with") removed 92.5% of French, 92.1% of German and 88.1% of Hebrew documents, and 0% of English; the same rule with each language's own frequent words removed 0.0%, 0.1% and 5.2%. A stop-word-ratio threshold that removes 5% of English removed 1.0% of French, 30.6% of German and 100% of Hebrew; the Quantile threshold removed 5.0% of each by construction. Rehydration weights computed from that filter's removal rates per cluster size came out noisy at this sample size (French: 1.52, 4.07, 10.0, 1.0, 2.68, 1.0 for sizes 1–5 and 6+; Hebrew has only 8 documents of size 5, which received weight 10): use thousands of documents per cluster size before trusting them. Code: of 745 Python files from 16 installed packages (pinned in the course environment, licences from their metadata, all permissive: Apache-2.0, BSD-3-Clause, MIT and combinations), a file-level split left 13 of 196 held-out files (6.6%) with a near-copy (Jaccard ≥ 0.7) in train, a directory-level split 11 of 171 (6.4%), a package-level split 0 of 168. SWE-smith (one shard, 3,696 instances from 14 repositories; 1,500 sampled): an instance-level split put all 14 repositories on both sides and left 14 of 439 held-out patches (3.2%) with a near-copy in train; a repository-level split, none.

## Lab

**Folder:** [`labs/module-10/lesson-06/`](../../labs/module-10/) · **Time:** about 45 minutes · **Pass check:** `pytest labs/module-10/lesson-06` passes and `extension_lab.py` prints parts A and B.

This lab measures; it compares no training arms, so it has no experiment contract.

| Variant | Hardware | What you run |
|---|---|---|
| Main path | none needed beyond CPU; optional: prepare 200,000 documents per language (`--docs 200000`, about 0.5–1 GB each streamed, PROJECTED) to get rehydration weights you can trust | `extension_lab.py --docs 20000` |
| Free GPU | not needed | — |
| Free CPU | laptop; measured 3.6 minutes (part A 173 s, part B 41 s) plus downloads (FineWeb2 2,000 documents per language, 26–30 s each; SWE-smith shard 4.1 MB; Qwen3 tokenizer files) | the steps below |

1. **Implement** the five TODOs in `lab.py`: tokens per byte, the Quantile threshold, the rehydration weight, the group split, SWE-smith's repository and family keys. Run `pytest labs/module-10/lesson-06`.
2. **Prepare** `fw2-fra`, `fw2-deu` and `fw2-heb` (`python -m frontierlab.datax.sources prepare fw2-heb --docs 2000`, and the same for the others).
3. **Run** `python labs/module-10/lesson-06/extension_lab.py`.
4. **Write up** (half a page): the token budget you would need to give Hebrew the same number of *words* as English in a mixture with Data-v0's tokenizer; which unit you would split code by for Data-v1 and why; and one thing SWE-smith's repository exclusion does not protect against (hint: the same upstream code can appear in more than one repository).

<details>
<summary>Reference solution</summary>

`labs/module-10/lesson-06/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-10/lesson-06`.

</details>

## Common mistakes

- **Comparing per-token losses across languages.** A token is a different amount of text in each; use bits per byte.
- **Applying English filters unchanged.** Stop-word and length rules encode English statistics; they delete other languages or let their junk through.
- **Trusting weights estimated from a few documents.** Removal rates per cluster size need many documents per size.
- **Splitting code by file.** Vendored and copied code crosses the split; split by repository, and exclude evaluation repositories before generating anything.
- **Citing a dataset without its revision.** SWE-smith's dataset grew after the paper; the numbers here belong to revision `ea6d717`.

## References

- G. Penedo et al., *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language*, 2025, sections 4.3–4.5 and 5. https://arxiv.org/abs/2506.20920
- J. Yang et al., *SWE-smith: Scaling Data for Software Engineering Agents*, 2025, abstract, section 2.1, appendix A.2. https://arxiv.org/abs/2504.21798
- Datasets: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 (revision `af9c13333eb981300149d5ca60a8e9d659b276b9`), https://huggingface.co/datasets/SWE-bench/SWE-smith (revision `ea6d7173829c7ec8fa16c22055699ff2e9188091`).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

The module project assembles Data-v1 from the decisions of lessons 10.1–10.5: [Module 10 project](../../projects/module-10-data-v1.md). Module 11 then uses Data-v1 for the pre-registered Recipe-R run.
