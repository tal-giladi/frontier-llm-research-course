from pathlib import Path

import numpy as np
import pytest

from frontierlab.labkit import load_target

lab = load_target(__file__)
HERE = Path(__file__).parent


def test_parse_titan_line():
    esc = "\x1b[31m"
    line = (f"[rank3]:[titan] 2026-10-04 12:00:00,000 - root - INFO - {esc}step: 12  \x1b[32mloss:  7.12345  "
            f"\x1b[33mgrad_norm:  1.2345  \x1b[36mmemory: 61.23GiB(77.40%)  \x1b[34mtps: 6,123  "
            f"\x1b[36mtflops: 345.67  \x1b[35mmfu: 34.95%\x1b[39m")
    r = lab.parse_titan_line(line)
    assert r == {"rank": 3, "step": 12, "loss": pytest.approx(7.12345), "memory_gib": pytest.approx(61.23),
                 "memory_pct": pytest.approx(77.40), "tps": 6123.0, "tflops": pytest.approx(345.67),
                 "mfu": pytest.approx(34.95)}
    assert lab.parse_titan_line("[rank0]: Building llama3 8B with ...") is None
    rows = [lab.parse_titan_line(x) for x in (HERE / "sample_titan.log").read_text().splitlines()]
    rows = [r for r in rows if r]
    assert len(rows) == 16 and {r["rank"] for r in rows} == {0, 1}


def test_exposed_comm():
    compute = [(0, 4), (5, 9)]
    assert lab.exposed_comm([(3, 6)], compute) == pytest.approx(1.0)        # 4..5 is exposed
    assert lab.exposed_comm([(1, 2), (1.5, 3)], compute) == pytest.approx(0.0)
    assert lab.exposed_comm([(9, 11), (10, 12)], compute) == pytest.approx(3.0)
    assert lab.exposed_comm([], compute) == 0.0


def test_state_bytes_match_measured_tensors():
    import torch
    from frontierlab.dist.measure import held_bytes
    from frontierlab.model import LM, toy
    m = LM(toy(vocab_size=500))
    opt = torch.optim.AdamW(m.parameters())
    m(torch.randint(0, 500, (2, 8)), labels=torch.randint(0, 500, (2, 8))).loss.backward()
    opt.step()
    n = sum(p.numel() for p in m.parameters())
    assert lab.state_bytes_per_rank(n, 1, "ddp") == held_bytes(m, opt)["total"]
    assert lab.state_bytes_per_rank(8000, 4, "fsdp2") == 8000 * 16 // 4


def test_summarize():
    rng = np.random.default_rng(1)
    x = list(rng.normal(10, 1, 200))
    s = lab.summarize([100.0] * 5 + x, skip=5)
    assert s["n"] == 200 and s["lo"] < s["median"] < s["hi"]
    assert s["median"] == pytest.approx(np.median(x)) and 9.5 < s["lo"] and s["hi"] < 10.5
