"""Course tooling: copy three induced runs' logs into traces/A, B, C (shuffled, names removed).

    python labs/module-07/lesson-05/make_traces.py runs/m07/l75/cpu

Writes metrics.jsonl (train rows only: step, loss, grad_norm, lr) and stability.jsonl (step, max_logit,
max_logit_layers, ratio_max, ratio_adamw_median, grad_norm_post, logz_mean) for f1-logits, f2-lr-bug and f3-data,
under letters drawn with a fixed seed, and prints the key (the lesson gives it in a collapsed reference diagnosis).
The pilot regenerates them at 30M/70M with ``--variant main`` runs.
"""

import json
import random
import sys
from pathlib import Path

from frontierlab.metrics import read_jsonl

KEEP_M = ("split", "step", "loss", "grad_norm", "lr")
KEEP_S = ("split", "step", "max_logit", "max_logit_layers", "ratio_max", "ratio_adamw_median", "grad_norm_post", "logz_mean")
SOURCES = {"f1-logits": "attention-logit growth", "f2-lr-bug": "optimizer / learning-rate fault", "f3-data": "bad data"}


def main():
    root = Path(sys.argv[1])
    out = Path(__file__).parent / "traces"
    names = list(SOURCES)
    random.Random(7).shuffle(names)
    key = []
    for letter, name in zip("ABC", names):
        d = out / letter
        d.mkdir(parents=True, exist_ok=True)
        rows = [r for r in read_jsonl(root / name / "metrics.jsonl") if r["split"] == "train"]
        if name == "f2-lr-bug":                       # prepend the healthy trunk it branched from
            trunk = [r for r in read_jsonl(root / "healthy" / "metrics.jsonl")
                     if r["split"] == "train" and r["step"] < rows[0]["step"]]
            rows = trunk + rows
        stab = read_jsonl(root / name / "stability.jsonl")
        if name == "f2-lr-bug":
            first = stab[0]["step"]
            stab = [r for r in read_jsonl(root / "healthy" / "stability.jsonl") if r["step"] < first] + stab
        (d / "metrics.jsonl").write_text("".join(json.dumps({k: r[k] for k in KEEP_M if k in r}) + "\n" for r in rows))
        (d / "stability.jsonl").write_text("".join(json.dumps({k: r[k] for k in KEEP_S if k in r}) + "\n" for r in stab))
        key.append(f"- {letter}: {name} ({SOURCES[name]})")
    print("\n".join(key))


if __name__ == "__main__":
    main()
