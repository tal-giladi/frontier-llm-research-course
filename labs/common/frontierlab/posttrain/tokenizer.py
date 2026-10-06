"""A character tokenizer for the toy post-training world (one token per character).

Ids 0, 1, 2 are PAD, BOS and EOS; the rest are the characters of :data:`CHARS` in order. The vocabulary
is tiny (35 ids) so a ~0.8M-parameter policy trains, samples and is scored in seconds on a CPU.
"""

from __future__ import annotations

import torch

PAD, BOS, EOS = 0, 1, 2
CHARS = "0123456789+-=.PQE#!abcdefghijklmn"     # digits, operators, instruction tags, quote/terminator, letters


class CharTokenizer:
    def __init__(self, chars: str = CHARS):
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters")
        self.chars = chars
        self.stoi = {c: i + 3 for i, c in enumerate(chars)}
        self.itos = {i + 3: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 3

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids, stop_at_eos: bool = True) -> str:
        out = []
        for i in (ids.tolist() if torch.is_tensor(ids) else ids):
            if i == EOS and stop_at_eos:
                break
            if i in (PAD, BOS, EOS):
                continue
            out.append(self.itos[i])
        return "".join(out)


TOK = CharTokenizer()
