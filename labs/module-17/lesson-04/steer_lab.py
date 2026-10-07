"""Lab 17.4: a sycophancy vector in Qwen3-0.6B — extraction, steering, controls, side effects, held-out evaluation.
Harmless traits only (see PUBLISHING_WARNING.md): the trait is agreeing with a user's stated answer to a simple
factual question.

    python labs/module-17/lesson-04/steer_lab.py                 # free CPU: Qwen3-0.6B (1.5 GB download), ~26 min
    python labs/module-17/lesson-04/steer_lab.py --variant main --print

Splits (fixed before any run):
  extraction  facts of half 0, template family "train", beliefs mixed (half right, half wrong);
  selection   the same facts and family, prompts without the answer: chooses the layer and alpha;
  evaluation  facts of half 1 (never seen), both families, wrong beliefs (sycophancy) and right beliefs;
  off-target  half-1 questions with no user opinion (factual accuracy) and ordinary text (loss, KL).

1. Your functions against the course's.
2. Baseline: how sycophantic is the model already?
3. Extraction at three layers (CAA: answer-letter activations, trait letter minus other letter), and the
   label-shuffled control vector.
4. Selection of layer and alpha on the selection split only.
5. Held-out evaluation: dose response with 95% intervals; random-direction and shuffled-label controls.
6. Side effects: factual accuracy without a user opinion, right-belief items, ordinary-text loss and KL.
7. Monitoring: does the projection on the vector predict the unsteered model's answer?
8. Two generations at alpha = 0 and the chosen alpha (qualitative only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.interp import persona as PE
from frontierlab.interp import steering as ST
from frontierlab.labkit import load_path
from frontierlab.stats import bootstrap_ci

HERE = Path(__file__).resolve().parent
LAYERS = (8, 12, 16)                    # of 28 in Qwen3-0.6B: a third, under half, about 60% of the depth
ALPHAS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)

MAIN = """# Main path (1x L40S/A100/H100; not run in this build, part of the Module 17 pilot).
# Qwen/Qwen3-1.7B (post-trained, 70d244cc) or your Module 13/14 post-trained Stage D model; 28 layers.
python labs/module-17/lesson-04/steer_lab.py --model qwen3-1.7b --layers 8 12 16 20 --device cuda --out runs/m17/steer-1.7b.json
# PROJECTED (pending pilot): ~4,000 forward passes of ~120-token prompts = 4.8e5 tokens x 2 x 1.7e9 = 1.6e15 FLOPs
# (seconds of H100 compute); wall-clock ~10-15 min with batch 16; < 0.25 GPU-hours.
# Persona-vectors reference pipeline (Chen et al. 2025): github.com/safety-research/persona_vectors (judge-scored
# open-ended answers; costs judge calls - count them in the budget). Use harmless traits only.
"""


def logit_items(tok, its):
    return PE.encode(tok, its)


def summ(x):
    m, lo, hi = bootstrap_ci(np.asarray(x))
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def check_learner(lab, vec):
    a, b = torch.randn(6, vec.numel()), torch.randn(4, vec.numel())
    print(f"  mean_diff matches: {torch.allclose(lab.mean_diff(a, b), ST.mean_diff(a, b))}")
    x = torch.randn(1, 4, vec.numel())
    print(f"  steer_edit matches: {torch.allclose(lab.steer_edit(vec, 2.0, 1)(x), ST.steer_edit(vec, 2.0, 1)(x))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--random", type=int, default=19)
    ap.add_argument("--out", type=Path, default=Path("runs/m17/steer-0.6b.json"))
    a = ap.parse_args()
    if a.print or a.variant == "main":
        print(MAIN)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Windows consoles: decoded tokens
    from frontierlab.interp import hf
    t0 = time.time()
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    model, tok = hf.load(a.model, device=a.device)
    res = {"model": hf.MODELS[a.model], "layers": a.layers}

    ext = PE.items("train", 0, belief="mixed")
    sel = logit_items(tok, PE.items("train", 0, belief="wrong"))
    ev = {fam: logit_items(tok, PE.items(fam, 1, belief="wrong")) for fam in ("train", "heldout")}
    right = logit_items(tok, PE.items("heldout", 1, belief="right"))
    neutral = logit_items(tok, PE.items(half=1, neutral=True))

    print("2. Baseline (log-odds of the user's answer; > 0 = picks it)")
    base = {k: ST.ab_logodds(model, v) for k, v in {"selection": sel, **ev, "right": right, "neutral": neutral}.items()}
    for k, b in base.items():
        what = "error rate" if k == "neutral" else "picks the user's answer"
        print(f"  {k:9s} n={len(b):3d}  mean log-odds {summ(b)}  {what}: {(b > 0).mean():.2f}")
    res["baseline"] = {k: {"mean": float(b.mean()), "rate": float((b > 0).mean())} for k, b in base.items()}

    print("\n3. Extraction: CAA vectors at the answer letter, trait letter minus other letter, mixed beliefs")
    pos = PE.encode(tok, ext, "trait")
    neg = PE.encode(tok, ext, "other")
    rp = ST.read(model, [x["ids"] for x in pos], a.layers)
    rn = ST.read(model, [x["ids"] for x in neg], a.layers)
    vecs = {L: ST.mean_diff(rp[L], rn[L]) for L in a.layers}
    shuf = {L: ST.shuffled_control(rp[L], rn[L], seed=1) for L in a.layers}
    for L in a.layers:
        print(f"  layer {L:2d}: |v| = {vecs[L].norm():6.2f}, mean |h| = {rp[L].norm(dim=1).mean():7.2f}, "
              f"|shuffled| = {shuf[L].norm():5.2f}")
    try:
        check_learner(lab, vecs[a.layers[0]])
    except NotImplementedError as e:
        print(f"  {e} - finish the TODOs (the script uses the course's code)")

    print("\n4. Selection of layer and alpha (selection split only; criterion: largest |change| at |alpha| <= 2")
    print("   with ordinary-text KL below 0.1 nats/token)")
    win = hf.text_windows(tok, 12, 64, seed=3)
    best = None
    for L in a.layers:
        for al in (-2.0, 2.0):
            d = ST.ab_logodds(model, sel, L, vecs[L], al) - base["selection"]
            kl = ST.side_effects(model, win, L, vecs[L], al)["kl"]
            print(f"  layer {L:2d} alpha {al:+.0f}: change {summ(d)}, ordinary-text KL {kl:.3f}")
            if kl < 0.1 and (best is None or abs(d.mean()) > best[2]):
                best = (L, al, abs(float(d.mean())), float(np.sign(d.mean() * al)))
    L = best[0]
    v = vecs[L] * best[3]                  # orient the vector so that +alpha increases agreement
    print(f"  chosen: layer {L}; vector oriented so that +alpha raises the log-odds of the user's answer"
          f"{' (the raw difference of means pointed the other way)' if best[3] < 0 else ''}")
    res["chosen"] = {"layer": L, "orientation": best[3]}

    print("\n5. Held-out evaluation (facts of half 1; wrong beliefs). Dose response:")
    res["dose"] = {}
    for fam in ("train", "heldout"):
        rows = ST.dose_response(model, ev[fam], L, v, ALPHAS, base=base[fam])
        res["dose"][fam] = rows
        print(f"  template family {fam}:")
        for r in rows:
            print(f"    alpha {r['alpha']:+.0f}: change {r['delta_logodds']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}], "
                  f"picks the user's answer {r['trait_rate']:.2f}, flips {r['flip_rate']:.2f}")
    al = 2.0
    target = ST.ab_logodds(model, ev["heldout"], L, v, al) - base["heldout"]
    rnd = [float((ST.ab_logodds(model, ev["heldout"], L, r, al) - base["heldout"]).mean()) for r in ST.random_like(v, a.random)]
    sh = ST.ab_logodds(model, ev["heldout"], L, shuf[L] * (v.norm() / shuf[L].norm()), al) - base["heldout"]
    p = (1 + sum(r >= target.mean() for r in rnd)) / (1 + len(rnd))
    print(f"  controls at alpha +2 (held-out family): vector {summ(target)}; {a.random} random directions of the same "
          f"norm: mean {np.mean(rnd):+.3f}, max {max(rnd):+.3f} (p_random = {p:.3f}); shuffled-label vector at the same "
          f"norm {summ(sh)}")
    # Two-sided check: a trait direction moves the behaviour both ways; damage moves it one way only.
    neg_t = float((ST.ab_logodds(model, ev["heldout"], L, v, -al) - base["heldout"]).mean())
    rnd_neg = [float((ST.ab_logodds(model, ev["heldout"], L, r, -al) - base["heldout"]).mean())
               for r in ST.random_like(v, a.random)]
    spread = float(target.mean()) - neg_t
    rspread = [x - y for x, y in zip(rnd, rnd_neg)]
    p2 = (1 + sum(r >= spread for r in rspread)) / (1 + len(rspread))
    print(f"  two-sided: change at +2 minus change at -2 = {spread:+.3f} for the vector; random directions "
          f"mean {np.mean(rspread):+.3f}, max {max(rspread):+.3f} (p_random = {p2:.3f})")
    res["controls"] = {"vector": float(target.mean()), "random": rnd, "p_random": p, "shuffled": float(sh.mean()),
                       "spread": spread, "random_spread": rspread, "p_random_spread": p2}

    print("\n6. Side effects at alpha +2 and -2")
    res["side"] = {}
    for al in (2.0, -2.0):
        nd = ST.ab_logodds(model, neutral, L, v, al)
        rd = ST.ab_logodds(model, right, L, v, al)
        se = ST.side_effects(model, win, L, v, al)
        print(f"  alpha {al:+.0f}: factual error rate without opinion {(base['neutral'] > 0).mean():.2f} -> {(nd > 0).mean():.2f}; "
              f"right-belief items pick the (correct) user answer {(base['right'] > 0).mean():.2f} -> {(rd > 0).mean():.2f}; "
              f"ordinary text loss {se['loss_increase']:+.4f} [{se['loss_increase_ci'][0]:+.4f}, {se['loss_increase_ci'][1]:+.4f}], "
              f"KL {se['kl']:.4f}")
        res["side"][str(al)] = {"neutral_error": float((nd > 0).mean()), "right_pick": float((rd > 0).mean()), **se}

    print("\n7. Monitoring: projection of the last prompt token on the vector vs the unsteered answer")
    acts = ST.read(model, [x["ids"] for x in ev["heldout"]], [L])[L]
    proj = ST.projection(acts, v).numpy()
    r = float(np.corrcoef(proj, base["heldout"])[0, 1])
    print(f"  Pearson r between projection and log-odds on held-out items: {r:+.3f} (n = {len(proj)})")
    res["monitor_r"] = r

    print("\n8. Generations (qualitative only)")
    it = PE.items("heldout", 1, belief="wrong")[0]
    ids = torch.tensor(tok(PE.render(tok, it)[: -len(" Answer: (")] if PE.render(tok, it).endswith(" Answer: (")
                           else PE.render(tok, it)[: -len("Answer: (")], add_special_tokens=False)["input_ids"])
    for al in (0.0, 4.0):
        txt = ST.generate(model, tok, ids, L, v, al, max_new_tokens=30)
        print(f"  alpha {al:+.0f}: {txt!r}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
