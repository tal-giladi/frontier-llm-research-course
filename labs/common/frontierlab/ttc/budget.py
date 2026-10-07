"""What a test-time-compute strategy spends: tokens, FLOPs and latency (lesson 15.1).

**Compute.** With the dense-transformer estimates of Module 13 (``frontierlab.pipeline.compute``): a token
that goes through a model of N parameters costs about 2N FLOPs, whether it is a prompt token (prefill), a
generated token (decode, with a KV cache) or a token a verifier reads. Attention over the context adds
4·L·d_attn·S per token at context S, small at the lengths used here and ignored (state it when it is not).

    FLOPs = 2·N_policy·(prefill + decode) + 2·N_verifier·verifier_tokens

``policy_token_equivalents`` divides by 2·N_policy: the number of policy tokens the same FLOPs would buy.
It is the axis on which a verifier's cost becomes comparable with more samples.

**Prefix sharing.** N samples of one prompt share the prompt. An engine with prefix caching (vLLM's
automatic prefix caching, SGLang's RadixAttention) prefills it once; without it each sample pays again.
``Spend.prefill_tokens`` is whatever the strategy really prefilled; :func:`sampling_spend` takes the
choice as an argument so the report says which one it assumed.

**Latency.** Parallel samples run as one batch: N chains of length L take L sequential decode steps at
batch N, not N·L steps. A sequential method (a longer single chain, step-level search with a verifier
call between steps) puts more work on the critical path. :class:`LatencyModel` turns a strategy's
sequential structure into seconds using decode-step times measured at several batch sizes (log-linear
interpolation) and measured verifier times; it is a model, checked against end-to-end timings in the lab.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Spend:
    """Tokens processed by one strategy on one problem (or summed over problems)."""
    policy_params: int
    verifier_params: int = 0
    prefill_tokens: float = 0.0
    decode_tokens: float = 0.0
    verifier_tokens: float = 0.0

    @property
    def flops(self) -> float:
        return 2.0 * self.policy_params * (self.prefill_tokens + self.decode_tokens) + \
            2.0 * self.verifier_params * self.verifier_tokens

    @property
    def policy_token_equivalents(self) -> float:
        return self.flops / (2.0 * self.policy_params)

    def __add__(self, other: "Spend") -> "Spend":
        if (self.policy_params, self.verifier_params) != (other.policy_params, other.verifier_params) and \
                other.verifier_params and self.verifier_params:
            raise ValueError("adding spends of different models")
        return Spend(self.policy_params, max(self.verifier_params, other.verifier_params),
                     self.prefill_tokens + other.prefill_tokens, self.decode_tokens + other.decode_tokens,
                     self.verifier_tokens + other.verifier_tokens)

    def scaled(self, f: float) -> "Spend":
        return Spend(self.policy_params, self.verifier_params, self.prefill_tokens * f, self.decode_tokens * f,
                     self.verifier_tokens * f)

    def to_dict(self) -> dict:
        return {**asdict(self), "flops": self.flops, "policy_token_equivalents": self.policy_token_equivalents}


def sampling_spend(policy_params: int, prompt_tokens: int, gen_tokens: float, n: int, *, verifier_params: int = 0,
                   scored_tokens: float = 0.0, shared_prefix: bool = True) -> Spend:
    """N independent samples of mean length ``gen_tokens``; optionally each scored by a verifier that reads
    ``scored_tokens`` tokens (prompt + response) per candidate."""
    prefill = prompt_tokens * (1 if shared_prefix else n)
    return Spend(policy_params, verifier_params, prefill, n * gen_tokens, n * scored_tokens)


def matched_n(budget_tokens: float, per_sample_tokens: float, per_sample_overhead: float = 0.0) -> int:
    """Largest N whose policy-token equivalents (per sample: tokens + verifier overhead) fit in the budget."""
    return max(0, int(np.floor(budget_tokens / (per_sample_tokens + per_sample_overhead) + 1e-9)))


@dataclass
class LatencyModel:
    """Seconds for a strategy from measured component times.

    ``decode_step``: {batch size: seconds per decode step at that batch}; ``prefill_per_token``: seconds per
    prompt token (batched prefill); ``verifier``: {batch size: seconds per verifier forward of a batch of
    candidates}. Values between measured batch sizes are interpolated linearly in log(batch).
    """
    decode_step: dict
    prefill_per_token: float
    verifier: dict | None = None

    @staticmethod
    def _interp(table: dict, b: float) -> float:
        xs = np.array(sorted(table), dtype=float)
        ys = np.array([table[k] for k in sorted(table)], dtype=float)
        if b <= xs[0]:
            return float(ys[0])                        # a smaller batch is not faster per step than batch xs[0]
        if b >= xs[-1]:
            return float(ys[-1] * b / xs[-1])          # beyond the largest measured batch: linear in batch
        return float(np.interp(np.log(b), np.log(xs), ys))

    def decode(self, steps: float, batch: float) -> float:
        return steps * self._interp(self.decode_step, max(1.0, batch))

    def verify(self, calls: float, batch: float) -> float:
        if not self.verifier:
            return 0.0
        return calls * self._interp(self.verifier, max(1.0, batch))

    def parallel(self, prompt_tokens: int, chain_len: float, n: int, verify: bool = False) -> float:
        """N samples in one batch, then (optionally) one verifier pass over the N candidates."""
        return (prompt_tokens * self.prefill_per_token + self.decode(chain_len, n)
                + (self.verify(1, n) if verify else 0.0))

    def search(self, prompt_tokens: int, steps: int, tokens_per_step: float, width: int, expand: int,
               final_tokens: float) -> float:
        """Step-level beam search: per step, width·expand candidates decode one step in a batch, then one
        verifier call scores them; the final answer step decodes ``final_tokens`` at batch ``width`` and is
        scored once more."""
        per_step = self.decode(tokens_per_step, width * expand) + self.verify(1, width * expand)
        return (prompt_tokens * self.prefill_per_token + steps * per_step + self.decode(final_tokens, width)
                + self.verify(1, width))
