# Module 6 — proposed changes to shared files (for the main session)

Module 6 does not edit `frontierlab/train/loop.py`, `frontierlab/model/lm.py`, `frontierlab/attention/__init__.py`,
`frontierlab/attention/accounting.py` or `frontierlab/record/diff.py`. Everything it needs lives in the new subpackage
`frontierlab/blocks/` and the training wrapper `frontierlab/blocks/train.py`, which replaces `loop.make_model`,
`loop.make_optimizer`, `loop.lr_at`, `loop.param_counts`, `loop.flops_per_token` and `loop.write_run_card` for one
`loop.main()` call (the pattern of `frontierlab.longctx.extend` and `frontierlab.optim.train`). The changes below would
make it native; lessons keep working either way (they call `python -m frontierlab.blocks.train`).

## 1. `attention/__init__.py`: register the bidirectional kind

`frontierlab/blocks/diffusion.py` registers `"gqa-bidir"` (Baseline-0's GQA without the causal mask, used by the
masked-diffusion objective of lesson 06.7). Add at the end of the imports:

```python
from frontierlab.blocks import diffusion  # noqa: F401,E402  (registers "gqa-bidir")
```

(Import it after the other kinds: `frontierlab.blocks` imports `frontierlab.model.lm`, which imports
`frontierlab.attention`; placing the line last avoids a circular import, as with `frontierlab.longctx.attention`.)

## 2. `attention/accounting.py`: count Module 6 parameters and FLOPs

`frontierlab/blocks/accounting.py` has exact `param_counts(cfg)` (built on the `meta` device, so totals are exact by
construction) and `flops_per_token(cfg, T, training)` for every BlockLM switch; for a config without
`extra["blocks"]` both equal the shared functions exactly (tested in `test_blocks.py`). Proposed: in
`attention/accounting.py`,

```python
def flops_per_token(cfg, T, training=True):
    if "blocks" in cfg.extra:                       # Module 6 block changes (MTP, HC/mHC, MoE, Engram, MatFormer, PLE)
        from frontierlab.blocks import accounting as blocks_acc
        return blocks_acc.flops_per_token(cfg, T, training)
    ...                                             # unchanged

def param_counts(cfg):
    if "blocks" in cfg.extra:
        from frontierlab.blocks import accounting as blocks_acc
        return blocks_acc.param_counts(cfg)
    ...
```

and add `"gqa-bidir"` to the GQA family *with twice the score FLOPs* (every query sees all T keys), as
`blocks/accounting.py::_attention_scores` does. The FLOP model, stated in `blocks/accounting.py`:
forward = 2 × active non-embedding parameters (unselected experts, Engram and PLE tables excluded; MatFormer at the
expected width; MTP modules in training only) + 2·V·C per output head application (main + one per MTP prediction)
+ attention scores × (L + MTP blocks)/L + the n-stream residual's mixing work (`hyperconn.hc_flops_per_token_sublayer`,
whose projections replace 2 × their parameters).

## 3. `train/loop.py`: build BlockLM when the config asks for it, and Module 6 flags

Minimal native change (no new flags needed for checkpoints made by the wrapper):

```python
def make_model(a, vocab_size):
    ...
    from frontierlab.blocks.model import build        # LM(cfg) unless cfg.extra has "blocks"
    return cfg, build(cfg)
```

and in `make_optimizer`, exclude from weight decay the static HC/mHC mappings and gating factors and the Engram
tables (decay would pull H_res towards 0 and break the identity path; HC section 5 and Engram section 4 exclude
them): use `frontierlab.blocks.train.param_groups(model, a.weight_decay)`, which keeps the loop's current split
(dim < 2 or "norm" in the name) and adds the names in `NO_DECAY_KEYS`.

Full native version: copy the wrapper's flags (`--mtp`, `--mtp-depth`, `--mtp-lambda`, `--mtp-schedule`,
`--residual`, `--streams`, `--sinkhorn-iters`, `--ffn`, `--moe-*`, `--engram-*`, `--ple-dim`, `--blocks`,
`--objective`, `--cfg FIELD=VALUE`, `--blocks-log`, `--hyper-every`) and the `lr_at` hook that sets
`model.mtp_lambda` from `frontierlab.blocks.mtp.lambda_at` and `model.track_hyper` on logging steps.
`labs/common/tests/test_blocks.py::test_exact_resume_with_module6_wrapper` is the test to copy (straight vs
stop-at-7-and-resume, bit-identical weights and losses, for MTP with the V3 schedule, mHC + MoE with the blocks log,
MatFormer, and the diffusion objective). `--loss chunked` must refuse MTP and the diffusion objective (the chunked
loss computes only the next-token loss).

## 4. Bug (latent, affects every run whose training uses torch's global RNG): accounting after the RNG restore

`loop.main()` restores `torch_rng` from the checkpoint and *then* calls `flops_per_token(cfg, a.seq)` and
`param_counts(cfg)` for the run card. `attention.accounting.param_counts` instantiates each attention module on the
CPU, which draws from torch's global RNG. A resumed run therefore continues from a different global RNG state than
the straight run. Today nothing in training uses the global RNG, so `test_train.py::test_exact_resume` passes; the
diffusion objective (random masks) exposed it — its resumed run diverged at the first step after resume.
`blocks/accounting.py` works around it with `torch.random.fork_rng(devices=[])`. Proposed fix in
`attention/accounting.py`:

```python
def attention_module(cfg, layer_idx=0):
    _import_kinds()
    ...
    with torch.random.fork_rng(devices=[]):          # never perturb the caller's RNG stream
        return cls(cfg, layer_idx=layer_idx) if _takes_layer(cls) else cls(cfg)
```

(or, in `loop.main()`, compute `fpt` and `pc` before restoring the checkpoint's RNG state). Any future dropout,
stochastic depth or data augmentation inside the model depends on this.

## 5. `record/diff.py`: classify the `blocks` block of a run card, and the MatFormer sampling seed

The wrapper writes its command-line settings into a top-level `blocks` block (like Module 7's `optim`); today
those keys come out as "unclassified difference; check it by hand". Proposed, next to the `optim.` rule:

```python
    if key.startswith("blocks."):                      # Module 6 BlockLM runs (frontierlab.blocks.train)
        if key in ("blocks.blocks_log", "blocks.hyper_every"):
            return "ignore", "logging only"
        if any(c.startswith("config.extra.blocks") or c == "config.attention" for c in changed):
            return "changed", "follows from the declared block change"
        return "invalidates", "a second changed variable (a Module 6 block switch)"
```

and add `"config.extra.blocks.seed"` to `SEED_KEYS` (the wrapper stores the run's seed there for MatFormer's
granularity sampler, so seed replicates of BlockLM runs currently need `--changed config.extra.blocks.seed`; the
lessons say so until this lands).

## 6. Glossary

New terms are in `curriculum/glossary-inbox/module-06.md`.

## 7. `_sidebar.md` lines for Module 6 (running numbers 21–27 reserved)

```markdown
- **Module 6 — Which other block changes earn their complexity?**
  - [21 · Multi-token prediction](lessons/module-06/lesson-01.md)
  - [22 · Residual-stream design: hyper-connections and mHC](lessons/module-06/lesson-02.md)
  - [23 · Combining changes](lessons/module-06/lesson-03.md)
  - [24 · Conditional memory and lookup sparsity: Engram](lessons/module-06/lesson-04.md)
  - [25 · Elastic architectures](lessons/module-06/lesson-05.md)
  - [26 · Tokenizer-free models: Byte Latent Transformer](lessons/module-06/lesson-06.md)
  - [27 · Non-autoregressive and latent reasoning](lessons/module-06/lesson-07.md)
  - [Module 6 quiz](assessments/module-06-quiz.md)
```

## 8. Pilot notebook (plan 12.1 rows "06.2 mHC stability" and "06 Lineage-F integration")

Commands, expected runtime and what to record are listed in `labs/module-06/README.md` ("Main path and pilot
commands"). The provided pilot traces for lesson 06.2 (HC vs mHC at pilot-10m / 30m / 70m, standard and raised
learning rates: `metrics.jsonl`, `blocks.jsonl` with Amax gains, `run_card.yaml`) should be published on the course's
Hugging Face account after the Colab pilot and linked from lesson 06.2's "What the evidence says".
