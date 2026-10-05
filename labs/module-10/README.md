# Module 10 labs — Which data, in which mix?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. The shared code is
`labs/common/frontierlab/datax/` with its tests in `labs/common/tests/test_datax.py`:

| Module | What it does |
|---|---|
| `sources.py` | prepares extra sources (FineWeb, Wikipedia, FineMath, the FineWeb-Edu annotations, FineWeb2 languages) in the Data-v0 layout with Data-v0's tokenizer and hash split, plus per-document provenance |
| `provenance.py` | source records, licence table, BLOCK/WARN/OK findings, mixture manifests |
| `neardup.py` | MinHash / LSH near-duplicates across splits, verified by exact Jaccard |
| `leakage.py` | word n-gram index, per-item overlap, planted positive control |
| `integrity.py` | the full integrity report of a mixture (provenance, exact and near cross-split duplicates, leakage of Eval v0, Eval v1, LAMBADA) |
| `packing.py` | document segments, the block-diagonal mask, attention kind `"gqa-docmask"`, masked evaluation |
| `mixture.py` | deterministic, counter-based, resumable mixture sampler with exact per-source token accounting; filtered subsets |
| `train.py` | the training wrapper around `frontierlab.train.loop`: `--mixture`, `--doc-mask`, `--init-from`, `--anneal` |
| `evaluate.py`, `arms.py` | scoring runs on several held-out sets and LAMBADA; seed-level comparisons and decision rules |
| `quality.py` | hashed bag-of-n-grams quality classifier (`EmbeddingBag`), pure PyTorch |
| `rephrase.py` | rephrasing harness with a pinned open model, output checks and generator-FLOPs accounting |
| `regmix.py`, `anneal.py` | RegMix sampling and fits (ridge, quadratic ridge, data mixing law); micro-anneal runner |
| `groups.py` | fertility, per-language thresholds, rehydration weights, group-level splits (10.6) |

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/common/tests/test_datax.py                 # the shared Module 10 code (no downloads)
pytest labs/module-10/lesson-01                        # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-10              # all reference solutions
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every
script caches its runs under `runs/m10/` (gitignored): rerun the same command after an interruption and it
continues (finished runs are skipped, interrupted ones resume exactly).

## Data (once for the module)

All downloads are streamed from pinned dataset revisions (listed in `frontierlab/datax/sources.py` and in
each lesson) and land in `labs/common/data/m10/` (gitignored). Sizes and times measured in this build:

```bash
python -m frontierlab.data.prepare --docs 20000 --vocab 8192                 # Data-v0, if not done in Module 1
python -m frontierlab.datax.sources prepare web --docs 20000                 # 38 MB, 101 s   (10.1-10.5)
python -m frontierlab.datax.sources prepare wiki --docs 4000                 # 36 MB, 43 s    (10.1-10.5)
python -m frontierlab.datax.sources prepare math --docs 8000                 # 30 MB, 29 s    (10.4-10.5)
python -m frontierlab.datax.sources prepare annot --docs 24000               # 45 MB, 59 s    (10.2)
python -m frontierlab.datax.sources prepare fw2-fra --docs 2000              # 6 MB, 26 s     (10.6; also fw2-deu, fw2-heb)
```

Lesson 10.3 downloads Qwen3-0.6B (revision `c1899de`, 1.2 GB, about 5 minutes here); lesson 10.6 one 4.1 MB
SWE-smith shard and the Qwen3 tokenizer.

## What each folder contains

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 10.1 Data integrity at scale | `resume_check.py`, `docmask_ablation.py` | stopped-and-resumed mixture run vs straight run (accounting, losses, weights); packing with vs without document masking, 3 seeds |
| `lesson-02/` | 10.2 Model-based quality filtering | `filter_ablation.py` | quality classifier on Llama-3 annotations; top 30/10/3% of a web pool vs unfiltered at equal tokens, 3 seeds |
| `lesson-03/` | 10.3 Synthetic and rephrased data | `rephrase_ablation.py` | Qwen3-0.6B rephrasing (wiki style + a QA probe), output checks, generator FLOPs; rephrase vs repeat, 3 seeds |
| `lesson-04/` | 10.4 Mixtures and micro-anneals | `mixture_lab.py regmix`, `mixture_lab.py anneal` | 24 RegMix runs, fits, confirmation; stable run + micro-anneals with a control |
| `lesson-05/` | 10.5 Continued training and forgetting | `continued_training.py` | base run, 4 continued-training arms × 3 seeds (FineMath share 0/10/50/100%), gain and forgetting |
| `lesson-06/` | 10.6 Multilingual and code data (extension) | `extension_lab.py` | fertility, per-language stop words and thresholds, rehydration weights, repository vs file splits, SWE-smith task splits |
| `project/` | Module project: Data-v1 | `run_project.py`, `data_v1_example.json`, `buggy_report.py` | manifest, integrity report and Data-v1 vs Data-v0 at equal tokens; the debugging task |

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 10 pilot, and every
main-path figure in the lessons is PROJECTED with its formula. Free CPU times were measured on 2026-10-04/05 on
a 16-thread Windows 11 laptop (torch 2.14.1+cpu, transformers 5.18.0) **with other jobs running** (another
module's runs shared the CPU), so expect them to be shorter on an idle machine.

| Lab | Main path (1× H100, PROJECTED) | Free GPU (T4) | Free CPU (measured) |
|---|---|---|---|
| 10.1 integrity, resume, doc mask | about 1.25 GPU-hours (6 Baseline-0 runs at 2,048 tokens) | `--variant t4`, pilot-10m | integrity report 23 min, `resume_check.py` 5 min, `docmask_ablation.py` about 30 min |
| 10.2 quality filtering | see the lesson's formula (12 pilot-30m runs of 393M tokens) | `--variant t4`, 200,000-document pool | classifier 77 s, 12 runs of about 260 s; about 70 min |
| 10.3 rephrasing | training 0.95 GPU-hours; generation 1.4e16 FLOPs with vLLM | `--variant t4`, 2,000 documents | generation 37 + 39 min, 12 runs of about 200 s; about 2 h |
| 10.4 RegMix and micro-anneals | about 1.3 GPU-hours (73 pilot-10m runs) | `--variant t4` | `regmix` 33 runs of 74–242 s (about 1.5 h), `anneal` 14 min |
| 10.5 continued training | 1.16 GPU-hours (12 Baseline-0 continuations of 131M tokens) | `--variant t4` | base 165 s, 12 runs of 217–285 s; about 55 min |
| 10.6 multilingual and code (extension) | CPU only | not needed | 3.6 min plus downloads |
| Project | 11 GPU-hours (6 Baseline-0 runs of 2.49B tokens) | `--variant t4`, about 1 h per run | `run_project.py` 33 min (integrity 15 min, 6 runs of about 165 s); `buggy_report.py` 9 min, `--fixed` 6 min |

Notes:

- On a T4 use fp32 (no `--dtype bf16`).
- The document mask on the dense SDPA path is slower than causal attention; on GPU, implement the masked arm
  with FlexAttention block masks and check it against `"gqa-docmask"` with `frontierlab.testing.equivalence`.
- Until the main session registers `"gqa-docmask"` in `frontierlab/attention/__init__.py`, Module 10 code imports
  `frontierlab.datax.packing` itself (proposed change: `curriculum/inbox/module-10-shared-changes.md`).
