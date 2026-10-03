"""Held-out loss on fixed windows: the first component of Eval Suite v0.

Every run is scored on the same windows (``TokenData.eval_windows``), and the per-window losses are
kept so two runs can be compared with a *paired* bootstrap (``frontierlab.stats.paired_bootstrap``).
Use ``split="val"`` while developing and choosing hyperparameters; touch ``split="test"`` only for
the final, pre-registered comparison (lesson 01.3).
"""

from __future__ import annotations

import torch

from frontierlab.data.loader import TokenData


@torch.no_grad()
def window_losses(model, data: TokenData, n_windows: int = 256, T: int = 512, batch: int = 16,
                  seed: int = 1234, device="cpu", autocast_dtype=None) -> list[float]:
    """Mean next-token loss of each fixed window, in window order."""
    was_training = model.training
    model.eval()
    starts, out = data.eval_windows(n_windows, T, seed), []
    for i in range(0, len(starts), batch):
        x = torch.stack([data.window(s, T) for s in starts[i:i + batch]]).to(device)
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            res = model(x, labels=x)
        out.extend(res.per_token_loss.float().mean(dim=1).tolist())
    model.train(was_training)
    return out
