"""Lab 17.2: recover a known circuit with patching, then test a claim about an open model with controls.

    python labs/module-17/lesson-02/patch_lab.py            # free CPU, parts 1-6: about 3-7 minutes
    python labs/module-17/lesson-02/patch_lab.py --hf       # adds part 7 on Qwen3-0.6B (1.5 GB download, ~15 min CPU)
    python labs/module-17/lesson-02/patch_lab.py --variant main --print

1. Your functions against the course's.
2. The induction model (two layers, trained here in about a minute; cached in runs/m17/induction.pt).
3. Two corruptions, two directions: head-level denoising and noising under the value and the key corruption.
4. Attribution patching against real patching: how good is the cheap screen?
5. Path patching: which layer-0 head feeds which input (query, key, value) of layer 1?
6. Controls and held-out tests of the claim "head 0.h is a previous-token head that the induction heads read
   through their keys": random-head ablations, random directions, an unseen gap, another query position.
7. (--hf) The same pipeline on indirect-object identification in Qwen3-0.6B.
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

from frontierlab.interp import hooks as HK
from frontierlab.interp import patching as P
from frontierlab.interp import tasks as T
from frontierlab.labkit import load_path

HERE = Path(__file__).resolve().parent
CKPT = Path("runs/m17/induction.pt")

MAIN = """# Main path (1x A100/H100 80 GB; not run in this build, part of the Module 17 pilot).
python -m frontierlab.interp.hf ioi --model qwen3-1.7b-base --n 96 --top 10 --random 49 --out runs/m17/ioi-1.7b-base.json
# PROJECTED (pending pilot): ~100 forward passes and one backward of 96 prompts x ~20 tokens on a 1.7B model:
# 100 x 96 x 20 x 2 x 1.7e9 = 6.5e14 FLOPs -> seconds of compute; wall-clock dominated by Python and loading, ~10 min, < 0.25 GPU-h.
# Optional cross-check with TransformerLens 4.0.0 (separate env from circuit-tracer):
#   from transformer_lens.model_bridge import TransformerBridge
#   b = TransformerBridge.boot_transformers("Qwen/Qwen3-1.7B-Base"); b.enable_compatibility_mode()
#   logits, cache = b.run_with_cache(tokens); cache["blocks.10.attn.hook_z"]      # = the course's z.10 site
"""


def show(name, arr):
    print(f"  {name}")
    for l, row in enumerate(np.asarray(arr)):
        print("    layer {}: ".format(l) + "  ".join(f"{v:+.2f}" for v in row))


def check_learner(lab, m):
    p = T.induction_pairs(16, kind="key", seed=5)
    b = P.baselines(m, p)
    ok1 = torch.allclose(lab.normalised_effect(b["clean"] * 0.5, b["clean"], b["corrupt"], "noise"),
                         P.normalised(b["clean"] * 0.5, b["clean"], b["corrupt"], "noise"))
    _, a = HK.capture(m, p.clean, ["z.0"])
    ok2 = torch.allclose(HK.run_with(m, p.corrupt, {"z.0": lab.head_patch_edit(a["z.0"], 1, 16)}),
                         HK.run_with(m, p.corrupt, {"z.0": HK.replace_at(a["z.0"], None, 1, 16)}))
    print(f"  normalised_effect matches: {ok1}; head_patch_edit matches: {ok2}")


def run_toy(lab):
    print("\n2. The induction model")
    t = time.time()
    m = T.train_induction_model(path=CKPT)
    print(f"  ready in {time.time() - t:.0f} s")
    try:
        check_learner(lab, m)
    except NotImplementedError as e:
        print(f"  {e} - finish the TODOs (the rest uses the course's code)")
    print("\n3. Head patching (mean normalised effect over 64 prompts; rows = layers, columns = heads)")
    pv, pk = T.induction_pairs(64, kind="value", seed=1), T.induction_pairs(64, kind="key", seed=1)
    for name, p in (("value corruption (first B -> C)", pv), ("key corruption (first A -> X)", pk)):
        b = P.baselines(m, p)
        print(f"  {name}: clean logit diff {b['clean_mean']:.2f}, corrupt {b['corrupt_mean']:.2f}")
        show("denoising (clean head into corrupt run)", P.head_sweep(m, p, "denoise"))
        show("noising (corrupt head into clean run)", P.head_sweep(m, p, "noise"))
    print("\n4. Attribution patching (one forward + one backward) against real denoising, key corruption")
    att, real = P.attribution_heads(m, pk), P.head_sweep(m, pk, "denoise")
    show("attribution estimate", att)
    print(f"  rank agreement: top head by estimate {np.unravel_index(att.argmax(), att.shape)}, "
          f"by real patching {np.unravel_index(real.argmax(), real.shape)}; "
          f"largest |estimate - real| = {np.abs(att - real).max():.2f}")
    print("\n5. Path patching from each layer-0 head (noising: corrupt sender, clean everything else), key corruption")
    print("     sender   -> q of layer 1   -> k of layer 1   -> v of layer 1   -> logits directly")
    best, bestv = None, -1
    for h in range(4):
        e = [float(P.path_patch(m, pk, 0, [h], r).mean()) for r in (("q", 1), ("k", 1), ("v", 1), ("logits",))]
        print(f"     head 0.{h}   " + "   ".join(f"{v:+14.3f}" for v in e))
        if e[1] > bestv:
            best, bestv = h, e[1]
    print(f"  candidate previous-token head: 0.{best} (K-composition effect {bestv:.3f})")
    print("\n6. Controls and held-out tests of the claim about head 0.{}".format(best))
    cc = P.component_control(m, pk, {0: [best]}, "mean", n_random=7, seed=0)
    print(f"  mean-ablating 0.{best}: loses {cc['effect']:.3f} of the clean-corrupt gap, 95% CI "
          f"[{cc['ci'][0]:.3f}, {cc['ci'][1]:.3f}]; the other 7 single heads: mean {cc['random_mean']:.3f}, "
          f"max {max(cc['random']):.3f}; p_random = {cc['p_random']:.3f} (7 draws: the smallest possible is 0.125)")
    q = pk.meta["query"]
    _, ac = HK.capture(m, pk.clean, ["resid_pre.1"])
    _, ak = HK.capture(m, pk.corrupt, ["resid_pre.1"])
    # the first B sits at position query + 1: that is where the previous-token head writes "the token before me"
    d =(ac["resid_pre.1"] - ak["resid_pre.1"])[:, q + 1].mean(0)
    dc = P.direction_control(m, pk, "resid_pre.1", d, n_random=32, positions=[q + 1])
    print(f"  projecting the mean clean-corrupt difference out of resid_pre.1 at position {q + 1}: effect "
          f"{dc['effect']:.3f} [{dc['ci'][0]:.3f}, {dc['ci'][1]:.3f}]; 32 random directions: mean {dc['random_mean']:.3f}, "
          f"95th percentile {dc['random_p95']:.3f}; p_random = {dc['p_random']:.3f}")
    for label, kw in (("held-out gap 12 (training gaps 0-8)", {"gap": 12, "max_gap": 12}),
                      ("held-out query position 2", {"query": 2}), ("held-out query position 10", {"query": 10})):
        ph = T.induction_pairs(64, kind="key", seed=11, **kw)
        bh = P.baselines(m, ph)
        e = P.summarise(P.ablation_effect(m, ph, {0: [best]}, "mean"))
        print(f"  {label}: clean {bh['clean_mean']:.2f}, corrupt {bh['corrupt_mean']:.2f}; ablating 0.{best} loses "
              f"{e['mean']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]")


def run_hf(n: int):
    from frontierlab.interp import hf
    print("\n7. Indirect-object identification in Qwen3-0.6B (c1899de), float32 on CPU")
    t = time.time()
    model, tok = hf.load("qwen3-0.6b")
    r = hf.ioi_study(model, tok, n=n, top=8, n_random=19)
    print(f"  attribution vs real patching on 8 random heads (calibration):")
    for c in r["calibration"]:
        print(f"    head {c['head']}: estimate {c['attribution']:+.3f}, real {c['patch']:+.3f}")
    a = r["ablation"]
    print(f"  candidate {r['candidate']}: mean-ablation loses {a['effect']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] of the "
          f"gap; 19 random 8-head sets in the same layers: mean {a['random_mean']:.3f}, 95th pct {a['random_p95']:.3f}, "
          f"p = {a['p_random']:.3f}")
    b = r["any_layer_control"]
    print(f"  secondary: 19 random 8-head sets from any layer: mean {b['random_mean']:.3f}, 95th pct {b['random_p95']:.3f}, "
          f"p = {b['p_random']:.3f}")
    h = r["heldout"]
    print(f"  held-out templates and names: loses {h['mean']:.3f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}] "
          f"(baseline clean {r['heldout_baseline']['clean']:.2f}, corrupt {r['heldout_baseline']['corrupt']:.2f})")
    o = r["offtarget"]
    print(f"  off-target: ordinary-text loss {o['clean_loss']:.3f} changes by {o['delta']:+.4f} [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}]")
    out = Path("runs/m17/ioi-0.6b.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=str))
    print(f"  wrote {out} ({time.time() - t:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["cpu", "main"], default="cpu")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--hf", action="store_true", help="also run part 7 on Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    if a.print or a.variant == "main":
        print(MAIN)
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Windows consoles: decoded tokens
    t0 = time.time()
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    print("1. Your functions are checked in part 2, once the model exists")
    run_toy(lab)
    if a.hf:
        run_hf(a.n)
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
