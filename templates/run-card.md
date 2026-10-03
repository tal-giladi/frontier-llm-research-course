# Run card

`frontierlab.train.loop` writes `run_card.yaml` into every run folder automatically. This page shows
what it contains and what you add by hand. Two runs are comparable only if everything under "must
match" matches, except the one variable your experiment contract changes.

```yaml
run: b0-seed1                     # folder name
question: "Noise floor of Baseline-0 on Eval v0"
parent_run: null                  # the run this one branches from (e.g. b0-seed0)
git: {commit: 3f2c1e0, dirty: false}
hardware: {gpu: "NVIDIA H100 80GB HBM3", gpu_count: 1, cuda: "12.8", torch: "2.14.1", python: "3.12.10"}
config: {...}                     # full model config (attention kind, sizes, ...)
args: {...}                       # every command-line argument (lr, schedule, batch, seq, seeds, ...)
data:                             # must match across compared runs
  name: Data-v0
  revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
  tokenizer_sha256: ...
data_files: {train: {tokens: ..., bin_sha256: ...}, val: {...}, test: {...}}
budget: {steps: 9500, tokens: 2.49e9, train_flops: 1.9e18}
params: {total: 121.9e6, non_embedding: 96.8e6}
```

Add by hand when the run finishes:

```yaml
measured:
  wall_clock_h: 3.6
  gpu_hours: 3.6
  cost_usd: 9.0                   # what you actually paid
  final_val_loss: 3.412
notes: "Session disconnected at step 6200; resumed with the same command."
```

## Must match between compared runs

- `data` and `data_files` hashes
- evaluation version and window seed
- everything in `args` except the variable under test (and seeds, when seeds are the replicate axis)
- software versions in `hardware`, unless the experiment is about software
