"""Lab 13.4: one model with thinking and non-thinking modes, thinking budgets, and a router.

    python labs/module-13/lesson-04/think_lab.py                     # free CPU, about 15 minutes
    python labs/module-13/lesson-04/think_lab.py --variant main --print

Two models of the policy's size (308k parameters), 2 seeds each, 1,000 SFT steps of 128 on three-digit
addition with half the examples in each mode (thinking-mode fusion in miniature):

* ``fusion``          thinking examples have the full trace; non-thinking ones an empty thinking block
* ``fusion+budget``   the same, but 30% of the thinking examples have their trace cut at a random length
                      (your ``truncated_target``): the model sees answers written after partial thinking

For each model, on 1,000 held-out problems: accuracy and generated tokens in each mode; accuracy against a
thinking budget B (budget forcing: after B thinking tokens, "#" is appended and the model answers); and a
router trained on 4,000 training prompts to predict whether the non-thinking answer will be right,
compared with random routing at the same thinking fraction and with an oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frontierlab.labkit import load_path
from frontierlab.model import LM
from frontierlab.pipeline import thinking as TH
from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.judge import Judge
from frontierlab.pipeline.seqs import Example, batches, train_sft
from frontierlab.posttrain.sft import load_policy, policy_config, save_policy
from frontierlab.posttrain.tokenizer import BOS, EOS, TOK

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m13/l134")
STEPS, BATCH, LR, N_TRAIN, N_EVAL = 1000, 128, 3e-3, 60000, 1000
BUDGETS = (0, 2, 3, 5, 6, 8, 9, None)


def training_examples(lab, budget_aware: bool, seed: int) -> list:
    tr, _ = TH.split()
    probs = TH.problems(tr, N_TRAIN, seed=seed)
    rng = np.random.default_rng(seed + 1)
    out = []
    for p in probs:
        target = p.target
        if budget_aware and p.mode == "h" and rng.random() < 0.3:
            target = lab.truncated_target(p.a, p.b, int(rng.integers(0, 9)))
        out.append(Example.of([BOS] + TOK.encode(p.prompt), TOK.encode(target) + [EOS]))
    return out


def train_model(lab, name, seed, led):
    path = ROOT / f"{name}-s{seed}" / "policy.pt"
    if path.exists():
        return load_policy(path)
    torch.manual_seed(seed)
    m = LM(policy_config())
    train_sft(m, training_examples(lab, name == "fusion+budget", seed), STEPS, BATCH, LR, seed=seed, ledger=led)
    save_policy(m, path, {"name": name, "seed": seed})
    return m


def train_router(model, seed, led):
    """A prompt-only classifier (the judge architecture) for P(non-thinking answer is right)."""
    tr, _ = TH.split()
    probs = TH.problems(tr, 4000, seed=seed + 77, mode="n")
    rows = TH.answer_with_budget(model, probs, None)
    led.forward("router", n_params(model), float(sum(r["total_tokens"] + 9 for r in rows)))   # labelling = generation
    y = torch.tensor([float(r["correct"]) for r in rows])
    ids = TH.encode_prompts(probs)
    last = torch.full((len(probs),), ids.shape[1] - 1)
    torch.manual_seed(seed)
    router = Judge()
    opt = torch.optim.AdamW(router.parameters(), lr=2e-3)
    for idx in batches(len(y), 128, 300, torch.Generator().manual_seed(seed)):
        loss = F.binary_cross_entropy_with_logits(router(ids[idx], last[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        led.train("router", n_params(router), float(ids[idx].numel()))
    router.eval()
    return router, float(y.mean())


def evaluate(lab, model, router, led):
    _, held = TH.split()
    hp = TH.problems(held, N_EVAL, seed=1, mode="h")
    hn = [p.with_mode("n") for p in hp]
    think = TH.answer_with_budget(model, hp, None)
    noth = TH.answer_with_budget(model, hn, None)
    ct, cn = np.array([r["correct"] for r in think]), np.array([r["correct"] for r in noth])
    tt, tn = np.array([r["total_tokens"] for r in think], float), np.array([r["total_tokens"] for r in noth], float)
    carries = np.array([p.carries for p in hp])
    budget = []
    for B in BUDGETS:
        rows = TH.answer_with_budget(model, hp, B)
        budget.append({"budget": B, "accuracy": float(np.mean([r["correct"] for r in rows])),
                       "tokens": float(np.mean([r["total_tokens"] for r in rows])),
                       "forced": float(np.mean([r["forced"] for r in rows]))})
    with torch.no_grad():
        ids = TH.encode_prompts(hn)
        p_easy = torch.sigmoid(router(ids, torch.full((len(hn),), ids.shape[1] - 1))).numpy()
    led.forward("router", n_params(router), float(ids.numel()))
    curve = []
    for th in np.linspace(0, 1.0001, 21):
        acc, tok, frac = lab.routed_cost(p_easy, th, ct, cn, tt, tn)
        rnd = TH.random_route(frac, cn, ct, tn, tt, seed=int(th * 1000))
        curve.append({"threshold": float(th), "accuracy": acc, "tokens": tok, "think_frac": frac,
                      "random_accuracy": rnd["accuracy"], "random_tokens": rnd["tokens"]})
    return {"think_acc": float(ct.mean()), "nothink_acc": float(cn.mean()), "think_tokens": float(tt.mean()),
            "nothink_tokens": float(tn.mean()),
            "nothink_acc_by_carries": {int(k): float(cn[carries == k].mean()) for k in np.unique(carries)},
            "think_acc_by_carries": {int(k): float(ct[carries == k].mean()) for k in np.unique(carries)},
            "budget": budget, "route": curve, "oracle": TH.oracle_route(cn, ct, tn, tt),
            "frontier": lab.pareto([(c["tokens"], c["accuracy"]) for c in curve])}


MAIN = [
    "# Main path (not run in this build; part of the Module 13 pilot): Qwen/Qwen3-1.7B 70d244cc (hybrid thinking) on",
    "# the first 500 GSM8K test questions: non-thinking, thinking, and budget forcing with Qwen3's stop-thinking text.",
    "python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-1.7B --n 500 --budgets 0,256,512,1024,2048,none "
    "--out runs/m13/l134-main",
    "python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-1.7B --n 500 --route cascade --out runs/m13/l134-main",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args(argv)
    if a.variant == "main":
        print("\n".join(MAIN))
        return
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    for x, y in ((478, 365), (999, 1), (5, 7)):
        assert lab.trace(x, y) == TH.AddProblem(x, y).trace, "your trace disagrees with the course's"
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    t0 = time.perf_counter()
    ROOT.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in range(a.seeds):
        for name in ("fusion", "fusion+budget"):
            f = ROOT / f"{name}-s{s}" / "result.json"
            if not f.exists():
                led = Ledger()
                t1 = time.perf_counter()
                model = train_model(lab, name, s, led)
                router, base_rate = train_router(model, s, led)
                r = evaluate(lab, model, router, led)
                r.update({"ledger": led.to_dict(), "router_train_positive_rate": base_rate,
                          "seconds": round(time.perf_counter() - t1, 1)})
                f.write_text(json.dumps(r, indent=1))
            r = json.loads(f.read_text())
            out[f"{name}-s{s}"] = r
            print(f"\n== {name} seed {s}: think {r['think_acc']:.3f} ({r['think_tokens']:.1f} tokens), "
                  f"no-think {r['nothink_acc']:.3f} ({r['nothink_tokens']:.1f} tokens); oracle router "
                  f"{r['oracle']['accuracy']:.3f} at {r['oracle']['tokens']:.1f} tokens")
            print("   no-think accuracy by number of carries: " +
                  ", ".join(f"{k}: {v:.2f}" for k, v in r["nothink_acc_by_carries"].items()))
            print("   budget  " + "  ".join(f"B={b['budget']}: {b['accuracy']:.3f}" for b in r["budget"]))
            pick = [c for c in r["route"] if 0.2 <= c["think_frac"] <= 0.6]
            for c in pick[:4]:
                print(f"   router th {c['threshold']:.2f}: think {c['think_frac']:.2f}, acc {c['accuracy']:.3f}, "
                      f"tokens {c['tokens']:.1f} | random at same fraction: acc {c['random_accuracy']:.3f}")
    (ROOT / "results.json").write_text(json.dumps(out, indent=1))
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
