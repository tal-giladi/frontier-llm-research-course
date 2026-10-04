# Module 5 — proposed changes to shared files (for the main session)

Module 5 adds new files and appends new functions to one shared file (`attention/accounting.py`, see 3).
It edits no existing line of any shared file. None of the Module 5 labs depend on the changes below:
every lab script and test imports the Module 5 kinds itself, and `labs/module-05/train_arm.py` wraps the
loop without editing it.

## 1. `frontierlab/attention/__init__.py`: register the Module 5 kinds on import

Add after the existing imports (order matters only in that `hybrid` imports `deltanet`):

```python
from frontierlab.attention import deltanet  # noqa: F401  (registers "gdn", "kda")
from frontierlab.attention import hybrid  # noqa: F401  (registers "hybrid")
from frontierlab.attention import dsa  # noqa: F401  (registers "dsa")
```

Effect: `ModelConfig(attention="kda" | "gdn" | "hybrid" | "dsa")` works without an explicit import, and
`python -m frontierlab.evals.suite_v1 run <a Module 5 run>` can load Module 5 checkpoints (today
`suite_v1.load_model` raises `KeyError: unknown attention 'hybrid'` unless the caller imported the
module; `labs/module-05/lesson-03/quality.py` imports them itself). No circular import: `deltanet`
imports only `attention.base` and `layers.rmsnorm`; `dsa` imports `base`, `gqa`, `ops`; `hybrid` looks up
`model.lm._takes_layer` lazily inside a function. Check: `pytest labs/common/tests` (all pass with or
without the change).

## 2. `frontierlab/attention/accounting.py: flops_per_token` raises for the Module 5 kinds

`train/loop.py` calls `accounting.flops_per_token(cfg, a.seq)` for the run card and MFU. For
`--attention kda/gdn/hybrid/dsa` that raises `KeyError: no FLOP model for attention 'kda'` (from
`flops_per_key`), so the plain loop cannot train a Module 5 kind even after change 1.
`labs/module-05/train_arm.py` patches `loop.flops_per_token` to `accounting.m05_flops_per_token`.
Proposed permanent fix, three lines at the top of `flops_per_token` in `accounting.py`:

```python
def flops_per_token(cfg: ModelConfig, T: int, training: bool = True) -> float:
    if cfg.attention in M05_KINDS:                       # Module 5 kinds: linear, hybrid, DSA
        return m05_flops_per_token(cfg, T, training)
    ...
```

`m05_flops_per_token` equals `flops_per_token` exactly for Baseline-0 and the Module 3 kinds (tested in
`test_attention_m05.py::test_m05_flops_equal_module3_flops_for_old_kinds`), so the change could also be a
full replacement. The same applies to `decode_flops_per_token` / `kv_bytes` / `cache_bytes` vs
`m05_decode_flops_per_token` / `m05_cache_bytes`.

## 3. What Module 5 appended to `frontierlab/attention/accounting.py`

A clearly delimited section at the end of the file ("Module 5 additions"), new functions only:
`M05_KINDS`, `m05_layer_kinds`, `linear_dims`, `dsa_dims`, `linear_mix_flops`, `indexer_flops_per_key`,
`m05_mix_flops`, `m05_flops_per_token`, `m05_decode_flops_per_token`, `linear_state_bytes`,
`m05_cache_bytes`, `dsa_crossover_length`, `linear_crossover_length`, `compressed_kv_entries`. No
existing function or line was changed (`git diff` shows additions only). If the main session prefers the
additions in a separate module, move them verbatim to `attention/accounting_m05.py` and import them at the
end of `accounting.py`; every Module 5 caller uses `accounting.<name>`.

## 4. `references/versions.md`

Under "Main path", the flash-linear-attention row: add "API checked 2026-10-04 at tag v0.5.2:
`fla.ops.kda.chunk_kda` and `fla.ops.gated_delta_rule.chunk_gated_delta_rule`, layout `[B, T, H, D]`,
log-space gate, `scale` default `1/sqrt(K)`, `initial_state [N, H, K, V]`. Install with a backend extra,
`pip install "flash-linear-attention[cuda]==0.5.2"` (since 0.5 a bare install no longer pulls torch or
triton)." Under "Notes on reference implementations": "Module 5: our chunked gated-delta reference
(`frontierlab/attention/deltanet.py`) is the ground truth for the fla kernels; the pilot runs
`pytest labs/common/tests/test_attention_m05.py -k fla` first."

## 5. `references/frontier-models-2026-10.md` (optional rows)

- `moonshotai/Kimi-Linear-48B-A3B-Instruct`: hybrid, 27 layers, 20 KDA (32 heads x 128) + 7 MLA
  (`kv_lora_rank` 512, `qk_rope_head_dim` 64, `mla_use_nope` true), `linear_attn_config.full_attn_layers`
  [4, 8, 12, 16, 20, 24, 27], `short_conv_kernel_size` 4, hidden 2304 (config.json, checked 2026-10-04;
  commit not recorded: add it when snapshotting).
- `deepseek-ai/DeepSeek-V3.2`: `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048 (config.json,
  checked 2026-10-04).

## 6. `.gitignore`

Nothing needed: Module 5 writes only under `runs/` (ignored).

## New files (Module 5 owns them)

- `labs/common/frontierlab/attention/{deltanet,hybrid,dsa,subq_bench,compressed}.py`
- `labs/common/tests/test_attention_m05.py`
- `labs/module-05/**`, `lessons/module-05/**`, `assessments/module-05-quiz.*`,
  `projects/module-05-subquadratic-memo.md`, `curriculum/status/module-05.log`,
  `curriculum/glossary-inbox/module-05.md`, this file.
