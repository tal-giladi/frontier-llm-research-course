# Module 10 — proposed changes to shared files (for the main session)

Module 10 does not edit `frontierlab/data/*`, `frontierlab/train/loop.py`, `frontierlab/model/`,
`frontierlab/attention/__init__.py` or `frontierlab/attention/accounting.py`. Everything runs through
`frontierlab.datax.train`, a wrapper that replaces `loop.make_model`, `loop.TokenData`, `loop.write_run_card`,
`loop.flops_per_token`, `loop.window_losses` and (with `--anneal`) `loop.lr_at` for one `loop.main()` call,
like `frontierlab.longctx.extend` and `frontierlab.optim.train`. The changes below would let the loop do the
same natively; the wrapper's command line can stay as an alias, so the lessons do not change.

## 1. `attention/__init__.py`: register the document-mask kind

Append after the Module 4 line:

```python
import frontierlab.datax.packing  # noqa: F401,E402  (registers "gqa-docmask")
```

`frontierlab.datax.packing` imports only `frontierlab.attention.base` and `frontierlab.attention.gqa`, both
loaded before this line, so there is no circular import (checked by importing `frontierlab.datax.packing`
first in a fresh interpreter: `python -c "import frontierlab.datax.packing, frontierlab.model"`). Until this
is added, Module 10 code imports `frontierlab.datax.packing` itself (the wrapper and `frontierlab.datax.evaluate`
do).

## 2. `attention/accounting.py`: count the document mask as GQA

```python
GQA_FAMILY = ("gqa", "gqa_partial", "sliding", "local_global", "sink", "gated", "gqa-rope-scaled", "gqa-irope",
              "gqa-softcap", "gqa-docmask")
```

The dense mask path computes every score, so its FLOPs are GQA's. (A block-sparse kernel would compute only
the kept fraction `frontierlab.datax.packing.useful_pair_fraction`; that is a separate, optional
`flops_per_key` mode.) Until then the wrapper maps `"gqa-docmask"` to `"gqa"` before calling `flops_per_token`.

## 3. `train/loop.py`: `--mixture`, `--doc-mask`, `--init-from`, `--anneal`

Add to `build_parser()`:

```python
    ap.add_argument("--mixture", type=Path, default=None,
                    help="MixtureSpec JSON: training windows from frontierlab.datax.mixture.MixtureSampler (Module 10)")
    ap.add_argument("--doc-mask", action="store_true",
                    help='block-diagonal attention by document: attention "gqa-docmask" (Module 10)')
    ap.add_argument("--anneal", action="store_true", help="after warmup, decay the learning rate linearly to 0")
```

(and `--init-from` as already proposed in `curriculum/inbox/module-04-loop-changes.md`, section 2).

In `main()`, replace the construction of `train` with:

```python
    if a.mixture is not None:
        from frontierlab.datax.mixture import MixtureSampler, MixtureSpec
        train = MixtureSampler(MixtureSpec.load(a.mixture), doc_mask=a.doc_mask)
    elif a.doc_mask:
        from frontierlab.datax.train import SegmentedTokenData
        train = SegmentedTokenData("train", **kw)
    else:
        train = TokenData("train", **kw)
```

After the checkpoint is loaded (where `step` is set):

```python
        if a.mixture is not None:
            train.seek(step * a.batch * a.grad_accum)       # the stream is a function of the window counter
```

In `make_model`, `if a.doc_mask: cfg = cfg.with_(attention="gqa-docmask")`. In the evaluation block, when
`a.doc_mask` is set, call `frontierlab.datax.packing.window_losses_docmask(...)` inside
`frontierlab.datax.packing.document_segments(None)` instead of `window_losses(...)`. In `lr_at`, add
`schedule == "anneal"` (linear from `lr` after warmup to 0 at `steps`; `frontierlab.datax.train.anneal_lr`)
and make `--anneal` an alias of `--schedule anneal`. In `write_run_card`, add
`extra={"datax": {"mixture": spec.to_dict(), "mixture_digest": spec.digest(), "doc_mask": a.doc_mask,
"planned_accounting": train.accounting(a.steps * a.batch * a.grad_accum, a.seq)}}` when `--mixture` is set,
and after the loop write `train.accounting(train.k, a.seq)` to `<run>/mixture_accounting.json`.

Tests that must keep passing after the change: `labs/common/tests/test_datax.py` (wrapper resume and card),
`labs/common/tests/test_train.py`.

## 4. `record/diff.py`: classify the `datax` block

`frontierlab.record` currently treats `datax.*` keys as unclassified differences (shown as WARN/other). Proposed:
`datax.mixture`, `datax.mixture_digest` and `datax.planned_accounting` belong to the data and should
**invalidate** unless declared (`--changed datax.mixture`), like `data` and `data_files`; `datax.doc_mask` and
`datax.anneal` are config-like (invalidate unless declared); `datax.init_from`, `datax.init_sha256` and
`datax.init_step` must match between compared continued-training arms (invalidate unless declared).
Add to `INVALIDATING_PREFIXES`: `"datax.mixture", "datax.planned_accounting", "datax.init_sha256"`.

## 5. `references/versions.md`: models and datasets pinned by Module 10

Add a section "Module 10 data and models (checked 2026-10-04)":

| Item | Revision | Licence | Used in |
|---|---|---|---|
| HuggingFaceFW/fineweb sample-10BT | `9bb295ddab0e05d785b879661af7260fed5140fc` | ODC-By 1.0 | 10.1–10.5 (`web`) |
| wikimedia/wikipedia 20231101.en | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC BY-SA 3.0, GFDL | 10.1–10.5 (`wiki`) |
| HuggingFaceTB/finemath finemath-4plus | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By 1.0 | 10.4–10.5 (`math`) |
| HuggingFaceFW/fineweb-edu-llama3-annotations | `72df4c92fb1b48beceb16016e8f695ec40a6c3a5` | ODC-By 1.0 (labels by Llama-3-70B-Instruct) | 10.2 |
| HuggingFaceFW/fineweb-2 (fra_Latn, deu_Latn, heb_Hebr) | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By 1.0 | 10.6 |
| SWE-bench/SWE-smith, `data/train-00002-of-00011.parquet` | `ea6d7173829c7ec8fa16c22055699ff2e9188091` | MIT | 10.6 |
| Qwen/Qwen3-0.6B (generator, tokenizer) | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | 10.3, 10.6 |

## 6. `requirements-cpu.txt`

No new packages. Module 10 uses `pyarrow` (installed with `datasets`) for the SWE-smith shard and
`transformers` for the generator; both are already pinned dependencies.

## 7. `_sidebar.md`: Module 10 lines

Running numbers continue after Module 9 (41); fix them if the main session numbers differently.

```markdown
- **Module 10 — Which data, in which mix?**
  - [42 · Data integrity at scale](lessons/module-10/lesson-01.md)
  - [43 · Model-based quality filtering](lessons/module-10/lesson-02.md)
  - [44 · Synthetic and rephrased data](lessons/module-10/lesson-03.md)
  - [45 · Mixtures and micro-anneals](lessons/module-10/lesson-04.md)
  - [46 · Continued training and forgetting](lessons/module-10/lesson-05.md)
  - [47 · Multilingual and code data](lessons/module-10/lesson-06.md)
  - [Module 10 quiz](assessments/module-10-quiz.md)
```

The project page `projects/module-10-data-v1.md` is picked up from `projects/` automatically.

## 8. Colab / H100 pilot notebook: Module 10 commands (not run in this build)

All from the repo root with `LAB_TARGET=solution`, after preparing every source with the main-path
(vocabulary-32,768) tokenizer. Record for each: wall time, GPU-hours, the printed tables (seed-level intervals
and decisions), and `nvidia-smi` peak memory.

| Lesson | Command | PROJECTED cost |
|---|---|---|
| 10.1 | `python labs/module-10/lesson-01/docmask_ablation.py --variant main` | see lesson 10.1 variants table |
| 10.2 | `python labs/module-10/lesson-02/filter_ablation.py --variant main` | see lesson 10.2 variants table |
| 10.3 | `python labs/module-10/lesson-03/rephrase_ablation.py --variant main` (vLLM 0.30.0 for generation) | training 0.95 GPU-h; generation 1.4e16 FLOPs (measure the wall time) |
| 10.4 | `python labs/module-10/lesson-04/mixture_lab.py regmix --variant main` then `... anneal --variant main` | about 1.3 GPU-h |
| 10.5 | `python labs/module-10/lesson-05/continued_training.py --variant main --base runs/m01/main/<Baseline-0 seed-0 run>` | 1.16 GPU-h |
| project | `python labs/module-10/project/run_project.py --recipe labs/module-10/project/data_v1_example.json --variant main --lr <Module 1 rate>` | 11 GPU-h |

The T4 variants are the same commands with `--variant t4` (fp32).
