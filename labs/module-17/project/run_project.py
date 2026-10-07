"""Module 17 project: one supported causal claim about an open model's behaviour.

    python labs/module-17/project/run_project.py --kind induction                 # the known-answer rehearsal, ~2 min CPU
    python labs/module-17/project/run_project.py --kind ioi --model qwen3-0.6b     # free CPU, ~14 min
    python labs/module-17/project/run_project.py --kind ioi --model qwen3-1.7b-base --device cuda --n 96 --random 39   # main path
    HARNESS=buggy python labs/module-17/project/run_project.py --kind induction    # the colleague's harness

Pipeline (each step uses the harness in this folder, so the planted bugs change the result):
1. selection prompts -> candidate heads (induction: path patching into layer-1 keys; IOI: attribution screen,
   then real denoising patches of the top 2k heads, keep the k largest);
2. effect of mean-ablating the candidate on the selection prompts, with a 95% interval;
3. control: the same ablation for ``--random`` random head sets with the candidate's size and layers
   (``harness.random_sets``); secondary, not in the card: random sets from any layer;
4. held-out prompts (``harness.heldout``): the same ablation;
5. off-target (IOI): next-token loss on ordinary text with the candidate mean-ablated;
6. a ClaimCard (``frontierlab.interp.claims``), its mechanical check, and a JSON record.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.interp import claims as CL
from frontierlab.interp import hooks as HK
from frontierlab.interp import patching as P
from frontierlab.interp import tasks as T
from frontierlab.stats import bootstrap_ci

HERE = Path(__file__).resolve().parent


def load_harness(name: str | None = None):
    name = name or os.environ.get("HARNESS", "harness")
    name = "buggy_harness" if name == "buggy" else name
    spec = importlib.util.spec_from_file_location(f"m17_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def ablation(model, p, heads, H, how="mean"):
    base = P.baselines(model, p)
    lg = P.ablate_heads(model, p.clean, heads, how, reference=p.corrupt)
    return H.effect(P.metric_fn(p)(lg), base["clean"], base["corrupt"], "noise")


def candidate_induction(model, p):
    eff = [float(P.path_patch(model, p, 0, [h], ("k", 1)).mean()) for h in range(HK.head_dim(model)[0])]
    return {0: [int(np.argmax(eff))]}, {"path_patch_k": eff}


@torch.no_grad()
def _patch_heads(model, p, order):
    base = P.baselines(model, p)
    L = HK.n_layers(model)
    _, acts = HK.capture(model, p.clean, [f"z.{l}" for l in range(L)])
    hd = HK.head_dim(model)[1]
    real = {}
    for l, h in order:
        l, h = int(l), int(h)
        lg = HK.run_with(model, p.corrupt, {f"z.{l}": HK.replace_at(acts[f"z.{l}"], None, h, hd)})
        real[(l, h)] = float(P.normalised(P.metric_fn(p)(lg), base["clean"], base["corrupt"], "denoise").mean())
    return real


def candidate_ioi(model, p, top: int):
    att = P.attribution_heads(model, p)
    order = np.dstack(np.unravel_index(np.argsort(-att.ravel()), att.shape))[0][:2 * top]
    real = _patch_heads(model, p, order)
    best = sorted(real, key=lambda k: -real[k])[:top]
    cand: dict[int, list[int]] = {}
    for l, h in best:
        cand.setdefault(l, []).append(h)
    return cand, {"screen": {f"{l}.{h}": v for (l, h), v in real.items()}}


def run(kind="induction", model_name="qwen3-0.6b", n=48, top=8, n_random=19, seed=0, device="cpu", harness=None,
        log=print):
    H = load_harness(harness)
    t0 = time.time()
    tok = None
    if kind == "induction":
        model = T.train_induction_model(path="runs/m17/induction.pt", verbose=False)
        mname = "course induction model (runs/m17/induction.pt)"
    else:
        from frontierlab.interp import hf
        model, tok = hf.load(model_name, device=device)
        mname = "{}@{}".format(*hf.MODELS[model_name])
    sel = H.selection(kind, n, seed, tok)
    held = H.heldout(kind, n, seed, tok)
    cand, extra = candidate_induction(model, sel) if kind == "induction" else candidate_ioi(model, sel, top)
    log(f"candidate {cand} ({time.time() - t0:.0f} s)")
    e = ablation(model, sel, cand, H).numpy().astype(np.float64)
    m, lo, hi = bootstrap_ci(e)
    Lh, Hh = HK.n_layers(model), HK.head_dim(model)[0]
    rnd = [float(np.mean(ablation(model, sel, s, H).numpy().astype(np.float64)))
           for s in H.random_sets(cand, Lh, Hh, n_random, seed)]
    p_rand = (1 + sum(r >= m - 1e-9 * abs(m) for r in rnd)) / (1 + len(rnd))
    try:                                   # secondary, not in the card: random sets anywhere in the model
        loose = H.random_sets(cand, Lh, Hh, n_random, seed, layer_matched=False)
        rnd_any = [float(np.mean(ablation(model, sel, s, H).numpy().astype(np.float64))) for s in loose]
    except TypeError:
        rnd_any = []
    he = ablation(model, held, cand, H).numpy()
    hm, hlo, hhi = bootstrap_ci(he)
    if kind == "ioi":
        from frontierlab.interp import hf
        win = hf.text_windows(tok, 16, 64, seed)
        _, hdim = HK.head_dim(model)
        _, ref = HK.capture(model, win, [f"z.{l}" for l in cand])

        def mk(l, hs):
            mu = ref[f"z.{l}"].mean((0, 1))

            def f(x):
                y = x.clone()
                for h in hs:
                    y[..., h * hdim:(h + 1) * hdim] = mu[h * hdim:(h + 1) * hdim]
                return y
            return f
        off = float((P.offtarget_loss(model, win, {f"z.{l}": mk(l, hs) for l, hs in cand.items()})
                     - P.offtarget_loss(model, win)).mean())
        lims = [f"model scale: {model_name} only; no claim about larger models",
                "one behaviour (IOI) and two template families; names are single tokens",
                "mean ablation over the ABC distribution; other references may differ",
                "logit difference metric only; necessity tested, sufficiency (denoising) only in the screen"]
        claim = (f"In {model_name}, mean-ablating heads {cand} on IOI prompts removes most of the logit difference "
                 "between the indirect object and the subject")
    else:
        off = 0.0
        lims = ["model scale: a 2-layer, 4-head synthetic model",
                "off-target behaviour not measured: the model has no ordinary-text task",
                "one task distribution (random tokens) with held-out gaps and query positions",
                "only 3 other heads share its layer (7 in the model), so the random-component control is weak"]
        claim = f"In the induction model, head {cand} feeds the induction heads through their keys and is necessary"
    card = CL.ClaimCard(claim=claim, model=mname, intervention=f"mean-ablate {cand} (z), clean prompts, corrupt reference",
                        effect=m, effect_ci=(lo, hi), control_kind="random head sets of the same size and layers",
                        control_p=p_rand, control_draws=len(rnd), heldout_effect=hm, heldout_ci=(hlo, hhi),
                        heldout_split="unseen gap 12 and query position 2" if kind == "induction"
                        else "other template family and other names", offtarget_delta=off, limitations=lims)
    problems = CL.check(card)
    rec = {"card": card.to_dict(), "problems": problems, "random_effects": rnd, "random_effects_any_layer": rnd_any,
           "candidate_evidence": extra,
           "harness": H.__name__, "seconds": time.time() - t0}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["induction", "ioi"], default="induction")
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--random", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = run(a.kind, a.model, a.n, a.top, a.random, a.seed, a.device)
    c = r["card"]
    print(f"\nclaim: {c['claim']}")
    print(f"  effect {c['effect']:.3f} [{c['effect_ci'][0]:.3f}, {c['effect_ci'][1]:.3f}] (harness: {r['harness']})")
    print(f"  control: {c['control_draws']} random sets ({c['control_kind']}), max {max(r['random_effects']):.3f}, "
          f"p_random = {c['control_p']:.3f}")
    if r["random_effects_any_layer"]:
        ra = r["random_effects_any_layer"]
        print(f"  secondary: random sets from any layer: max {max(ra):.3f}, "
              f"{sum(x >= c['effect'] for x in ra)} of {len(ra)} at least as large as the candidate")
    print(f"  held-out ({c['heldout_split']}): {c['heldout_effect']:.3f} [{c['heldout_ci'][0]:.3f}, {c['heldout_ci'][1]:.3f}]")
    print(f"  off-target loss change: {c['offtarget_delta']:+.4f} nats")
    print("  check: " + ("supported as scoped" if not r["problems"] else "; ".join(r["problems"])))
    out = a.out or Path(f"runs/m17/project-{a.kind}-{r['harness'].split('_', 1)[1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=str))
    print(f"  wrote {out} ({r['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
