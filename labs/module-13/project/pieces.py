"""The three pipeline pieces the debugging task is about, in their correct form.

* :func:`make_pair` — which candidate is chosen and which rejected, from scores (preference stage).
* :func:`distill_advantage` — the per-token advantage of on-policy distillation (distillation stage).
* :func:`sft_example` — a (prompt, response) training example for the toy world (SFT and distillation stages).

``buggy_pipeline.py`` holds a colleague's versions; ``test_pieces.py`` checks whichever set ``PIPE_HOOKS`` names.
"""

from __future__ import annotations

import numpy as np
import torch

from frontierlab.pipeline.seqs import Example
from frontierlab.posttrain.tokenizer import BOS, EOS, TOK


def make_pair(scores) -> tuple[int, int] | None:
    s = np.asarray(scores, dtype=float)
    if s.max() == s.min():
        return None
    return int(np.argmax(s)), int(np.argmin(s))


def distill_advantage(logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    return -(logp - teacher_logp).detach()


def sft_example(problem, text: str, finished: bool = True) -> Example:
    return Example.of([BOS] + TOK.encode(problem.prompt), TOK.encode(text) + ([EOS] if finished else []))
