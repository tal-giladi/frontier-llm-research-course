# Module 3 — proposed changes to shared files (for the main session)

Module 3 adds new files only (listed at the end) and edits no existing shared file. These are the
changes it proposes. None of the Module 3 labs depend on them: every lab script and test imports the
Module 3 kinds explicitly, and `labs/module-03/train_variant.py` wraps the loop without editing it.

## 1. `frontierlab/attention/__init__.py`: register the Module 3 kinds on import

Add after the `gqa` import line:

```python
from frontierlab.attention import mla  # noqa: F401  (registers "mla")
from frontierlab.attention import sliding  # noqa: F401  (registers "sliding", "local_global")
from frontierlab.attention import gated  # noqa: F401  (registers "sink", "gated")
from frontierlab.attention import headshape  # noqa: F401  (registers "gqa_partial")
```

Effect: `ModelConfig(attention="mla")` works without an explicit import (today `LM(cfg)` raises
`KeyError: unknown attention 'mla'` unless the caller imported the module), and
`labs/module-01/project/evaluate.py` can score Module 3 runs. Import order matters: `sliding` imports
`gqa`; `gated` imports `sliding`. Check: `pytest labs/common/tests` (all pass with or without the change).

## 2. `frontierlab/layers/rmsnorm.py`: do not downcast float64 to float32 (a correctness-suite bug)

`RMSNorm._norm` computes `x.float()`, which turns a float64 input into float32. Consequences found
while building Module 3:

* an op-level float64 `gradcheck` of any block containing an RMSNorm (QK-norm, MLA's latent norm)
  fails by about 1e-7 for reasons unrelated to the code under test;
* every model-level float64 comparison in `frontierlab.testing` (causal check, cache agreement,
  equivalence) is rounded to float32 resolution at each norm. In a test, two MLA paths that differ by
  2e-15 at the attention output gave bit-identical logits, and a planted 1% scale error in one path
  was invisible at the course's initialisation (std 0.02). The checks still catch large bugs (the
  lesson 01.1 mask bug gives 0.77), but not small ones.

Proposed fix (keeps the float32 upcast for bf16/fp16, keeps float64 as float64):

```python
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        dt = torch.promote_types(x.dtype, torch.float32)
        xf = x.to(dt)
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x)
```

Until it is applied, `frontierlab.attention.checks.exact_rmsnorm()` applies exactly this as a context
manager, and every Module 3 check runs inside it. After applying it, `exact_rmsnorm` becomes a no-op
and can stay. Re-run `pytest labs/common/tests labs/module-01 labs/module-02` with
`LAB_TARGET=solution`; float32/bf16 behaviour is unchanged, so no Module 1–2 number changes.

Related suggestion for `frontierlab.testing` (not required): run `causal_check` and `cache_agreement`
on a copy with larger weights (`frontierlab.attention.checks.sharpen`), since at std 0.02 attention is
nearly uniform and layer outputs are tiny next to the residual stream.

## 3. `frontierlab/flops.py` and `frontierlab/model/config.py: param_counts` are GQA-only

`train/loop.py` writes `budget.train_flops` and `params` into every run card with these GQA formulas,
so for an `--attention mla` run the card would be wrong (MLA has 6.2M more parameters at Baseline-0
width). `labs/module-03/train_variant.py` patches the loop's names to
`frontierlab.attention.accounting.param_counts` / `flops_per_token`, which equal the old functions
exactly for GQA (tested) and are exact for every registered kind. Proposed permanent change in
`train/loop.py` (two import lines):

```python
from frontierlab.attention.accounting import flops_per_token, param_counts   # instead of frontierlab.flops / model
from frontierlab.flops import PEAK_BF16
```

and, in `make_model`, accept `ModelConfig.extra` from the command line, for example

```python
    ap.add_argument("--extra", default=None, help='JSON merged into cfg.extra, e.g. {"kv_lora_rank": 256}')
    ...
    if a.extra:
        cfg = cfg.with_(extra={**cfg.extra, **json.loads(a.extra)})
```

so `python -m frontierlab.train.loop --attention mla --extra '{"kv_lora_rank": 256}'` works without
the wrapper.

## 4. `references/versions.md`

Add under the Module 3 heading: "Hugging Face Transformers `modeling_deepseek_v3.py`, main branch,
checked 2026-10-03: the cache stores the compressed latent (`kv_nope`, `k_rot`) and expands per step."

## New files (Module 3 owns them)

- `labs/common/frontierlab/attention/{ops,mla,sliding,gated,headshape,accounting,checks,probes,bench}.py`
- `labs/common/tests/test_attention_m03.py`
- `labs/module-03/**`, `lessons/module-03/**`, `assessments/module-03-quiz.*`,
  `projects/module-03-attention-memo.md`, `curriculum/status/module-03.log`,
  `curriculum/glossary-inbox/module-03.md`, this file.
