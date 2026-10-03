"""Debugging task for the Module 2 project: a performance claim with planted flaws.

    python labs/module-02/project/claim.py

This script "shows" that the chunked loss makes Baseline-0's training step much faster. The number
it prints is not a valid measurement. Find every reason why (there are at least four), fix them,
and report what the corrected comparison shows. Do not trust the comments.
"""

from __future__ import annotations

import time

import torch

from frontierlab.model import LM, toy
from frontierlab.perf.chunked_ce import lm_loss_chunked

torch.manual_seed(0)
cfg = toy(vocab_size=8192)
model = LM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, cfg.vocab_size, (16, 256))


def step(loss_fn, batch):
    opt.zero_grad(set_to_none=True)
    loss_fn(batch).backward()
    opt.step()


# baseline: one careful measurement
t0 = time.perf_counter()
step(lambda b: model(b, labels=b).loss, x)
t_base = time.perf_counter() - t0

# optimized: same work, averaged over a few calls for stability
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    step(lambda b: lm_loss_chunked(model, b, 1024), x[:8])
    ts.append(time.perf_counter() - t0)
t_new = min(ts)

print(f"baseline {t_base * 1e3:.1f} ms, chunked {t_new * 1e3:.1f} ms -> {t_base / t_new:.2f}x faster")
print("conclusion: the chunked loss is faster and should be the default")
