import torch

from frontierlab.metrics import read_jsonl
from frontierlab.train.loop import lr_at, main

ARGS = ["--preset", "toy", "--batch", "4", "--seq", "32", "--lr", "3e-3", "--warmup", "5", "--steps", "24",
        "--log-every", "4", "--eval-every", "12", "--eval-windows", "8", "--ckpt-every", "100", "--device", "cpu"]


def test_loss_goes_down(tiny_data, tmp_path):
    main(["--run", str(tmp_path / "r"), "--data", str(tiny_data), *ARGS, "--steps", "60"])
    rows = [r for r in read_jsonl(tmp_path / "r" / "metrics.jsonl") if r["split"] == "train"]
    assert rows[-1]["loss"] < rows[0]["loss"] - 0.5
    assert (tmp_path / "r" / "run_card.yaml").exists()


def test_exact_resume(tiny_data, tmp_path):
    """24 steps straight == 10 steps, stop, resume to 24: same weights, same logged losses."""
    a = main(["--run", str(tmp_path / "a"), "--data", str(tiny_data), *ARGS])
    main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS, "--stop-after", "10"])
    b = main(["--run", str(tmp_path / "b"), "--data", str(tiny_data), *ARGS])
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q), n
    la = [r["loss"] for r in read_jsonl(tmp_path / "a" / "metrics.jsonl") if r["split"] == "train"]
    lb = [r["loss"] for r in read_jsonl(tmp_path / "b" / "metrics.jsonl") if r["split"] == "train"]
    assert la == lb


def test_wsd_schedule_shape():
    steps, lr = 100, 1.0
    vals = [lr_at(s, steps, lr, 10, "wsd", decay_frac=0.2) for s in range(steps)]
    assert vals[9] == 1.0 and vals[50] == 1.0 and vals[79] == 1.0
    assert vals[99] < 0.15 and all(x >= y for x, y in zip(vals[80:], vals[81:]))
