import hashlib
import math

import pytest
import torch

from frontierlab.evals import suite_v0 as s
from frontierlab.model import LM, toy


def char_encode(text):                     # 1 token per character, ids 1..127; 0 is padding
    return [min(ord(ch), 127) for ch in text]


class Oracle(torch.nn.Module):
    """Puts all probability on the true next character of a known passage."""

    def __init__(self, texts, V=128):
        super().__init__()
        self.next = {}
        for t in texts:
            ids = char_encode(t)
            for i in range(1, len(ids)):
                self.next[tuple(ids[:i])] = ids[i]
        self.V = V

    def forward(self, x):
        B, T = x.shape
        logits = torch.full((B, T, self.V), -30.0)
        for b in range(B):
            row = x[b].tolist()
            for i in range(T):
                nxt = self.next.get(tuple(row[:i + 1]))
                if nxt is not None:
                    logits[b, i, nxt] = 30.0
        return type("Out", (), {"logits": logits})()


class Uniform(torch.nn.Module):
    def forward(self, x):
        return type("Out", (), {"logits": torch.zeros(*x.shape, 128)})()


TEXTS = ["the cat sat on the mat", "she opened the door and saw her brother", "a b c d"]


def test_split_last_word():
    assert s.split_last_word("the cat sat on the mat ") == ("the cat sat on the", " mat")
    with pytest.raises(ValueError):
        s.split_last_word("word")


def test_truncates_context_from_the_left():
    c, t = s.encode_example(char_encode, "abcdefgh ij", max_len=6)
    assert t == char_encode(" ij") and c == char_encode("fgh") and len(c) + len(t) == 6


def test_oracle_scores_perfectly_and_uniform_gives_log_v():
    o = s.lambada_scores(Oracle(TEXTS), char_encode, TEXTS, max_len=64, batch=2)
    assert all(r["correct"] for r in o) and all(abs(r["logprob"]) < 1e-6 for r in o)
    u = s.lambada_scores(Uniform(), char_encode, TEXTS, max_len=64, batch=2)
    for r, text in zip(u, TEXTS):
        n = len(s.split_last_word(text)[1])
        assert r["n_target"] == n and r["logprob"] == pytest.approx(-n * math.log(128))


def test_padding_and_batch_size_do_not_change_scores():
    torch.manual_seed(0)
    m = LM(toy(vocab_size=128)).double()
    a = s.lambada_scores(m, char_encode, TEXTS, max_len=64, batch=1)
    b = s.lambada_scores(m, char_encode, TEXTS, max_len=64, batch=3)
    assert [r["correct"] for r in a] == [r["correct"] for r in b]
    assert [r["logprob"] for r in a] == pytest.approx([r["logprob"] for r in b], abs=1e-9)


def test_summarize(tiny_data):
    from frontierlab.data.loader import TokenData
    torch.manual_seed(0)
    m = LM(toy(vocab_size=128))
    val = TokenData("val", root=tiny_data)
    path = tiny_data / "lambada.jsonl"
    path.write_text("\n".join('{"text": "%s"}' % t for t in TEXTS))
    res = s.run_suite(m, val, char_encode, n_windows=8, T=32, lambada_path=path)
    out = s.summarize(res, n_boot=200)
    assert len(res["heldout"]["losses"]) == 8 and res["lambada"]["n"] == 3
    assert out["lambada_acc"][0] in (0.0, 1 / 3, 2 / 3, 1.0) and out["lambada_target_ppl_per_token"] > 1


def test_pins_are_consistent():
    assert s.LAMBADA_REVISION in s.LAMBADA_URL and len(s.LAMBADA_SHA256) == 64
    if s.DEFAULT_LAMBADA.exists():                    # only after download_lambada() has run
        assert hashlib.sha256(s.DEFAULT_LAMBADA.read_bytes()).hexdigest() == s.LAMBADA_SHA256
        assert len(s.load_lambada()) == s.LAMBADA_N
