"""Debugging task of the Module 1 project: a colleague's Eval v0 script. Something is wrong with it.

    python labs/module-01/project/buggy_evaluate.py runs/m01/cpu/seeds-lr0.003-s0
    python labs/module-01/project/buggy_evaluate.py runs/m01/cpu/seeds-lr0.003-s1

The colleague reports: "with my script, Baseline-0's seed std on held-out loss is larger than in your
noise floor, and paired comparisons between seeds look no better than unpaired ones. Also, on LAMBADA
the passages with the LONGEST contexts score worst, which is the opposite of what more context should
do." Find every bug (there are two), explain how each one
produces the symptom, and fix this copy. Do not compare it line by line with evaluate.py first: start
from the symptoms and design a check that isolates each cause.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from frontierlab.data.loader import TokenData
from frontierlab.evals import suite_v0
from frontierlab.evals.heldout import window_losses
from frontierlab.model import LM, ModelConfig


@torch.no_grad()
def lambada(model, encode, texts, max_len):
    out = []
    for text in texts:
        ctx, tgt = suite_v0.split_last_word(text)
        c, t = encode(ctx), encode(tgt)
        c = c[:max_len - len(t)]
        x = torch.tensor([c + t])
        logp = torch.log_softmax(model(x).logits.float(), -1)[0]
        pos = torch.arange(len(c) - 1, len(c) + len(t) - 1)
        tt = torch.tensor(t)
        out.append({"correct": bool(torch.equal(logp[pos].argmax(-1), tt)),
                    "logprob": float(logp[pos].gather(-1, tt[:, None]).sum()), "n_target": len(t)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lambada", type=int, default=1000)
    a = ap.parse_args()
    card = yaml.safe_load((a.run / "run_card.yaml").read_text())
    ck = torch.load(a.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = LM(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    T, seed = int(card["args"]["seq"]), int(card["args"]["seed"])
    val = TokenData("val")
    losses = window_losses(model, val, 256, T, seed=seed)
    suite_v0.download_lambada()
    items = lambada(model, suite_v0.data_v0_encoder(), suite_v0.load_lambada(n=a.lambada), a.max_len)
    res = {"heldout": {"losses": losses}, "lambada": {"items": items}}
    (a.run / "eval_buggy.json").write_text(json.dumps(res))
    print(f"held-out {sum(losses) / len(losses):.4f}   LAMBADA acc {sum(i['correct'] for i in items) / len(items):.4f}")


if __name__ == "__main__":
    main()
