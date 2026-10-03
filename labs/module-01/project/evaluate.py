"""Score one finished run with Eval Suite v0 and keep every per-item score.

    python labs/module-01/project/evaluate.py runs/m01/cpu/seeds-lr0.003-s0            # validation
    python labs/module-01/project/evaluate.py runs/m01/main/seeds-lr0.003-s0 --device cuda --bf16
    python labs/module-01/project/evaluate.py RUN --split test      # ONCE, after the decision is written down

Writes ``RUN/eval_v0_<split>.json`` (per-window held-out losses, per-passage LAMBADA scores, and the
pins: suite version, LAMBADA revision and SHA-256, window seed, sequence length). The first call
downloads the pinned LAMBADA file (1.8 MB) into labs/common/data/evals/.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.evals import suite_v0
from frontierlab.model import LM, ModelConfig


def load_model(run: Path, device: str) -> LM:
    ck = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = LM(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--seq", type=int, default=None, help="window length (default: the run's --seq)")
    ap.add_argument("--lambada", type=int, default=None, help="first N passages only (default: all 5,153)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    import yaml
    card = yaml.safe_load((a.run / "run_card.yaml").read_text())
    T = a.seq or int(card["args"]["seq"])
    model = load_model(a.run, a.device)
    data = TokenData(a.split)
    suite_v0.download_lambada()
    t0 = time.perf_counter()
    res = suite_v0.run_suite(model, data, suite_v0.data_v0_encoder(), n_windows=a.windows, T=T,
                             n_lambada=a.lambada, device=a.device, pad_id=data.meta.get("eot_id", 0),
                             autocast_dtype=torch.bfloat16 if a.bf16 else None)
    res["run"], res["seconds"] = str(a.run), round(time.perf_counter() - t0, 1)
    out = a.run / f"eval_v0_{a.split}.json"
    out.write_text(json.dumps(res))
    s = suite_v0.summarize(res)
    print(f"{a.run}  ({a.split}, {res['seconds']} s)")
    print(f"  held-out loss   {s['heldout_loss'][0]:.4f}  95% CI [{s['heldout_loss'][1]:.4f}, {s['heldout_loss'][2]:.4f}]"
          f"  over {a.windows} windows of {T}")
    print(f"  LAMBADA acc     {s['lambada_acc'][0]:.4f}  95% CI [{s['lambada_acc'][1]:.4f}, {s['lambada_acc'][2]:.4f}]")
    print(f"  LAMBADA logprob {s['lambada_target_logprob'][0]:.3f}  95% CI [{s['lambada_target_logprob'][1]:.3f}, "
          f"{s['lambada_target_logprob'][2]:.3f}]   per-token target ppl {s['lambada_target_ppl_per_token']:.1f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
