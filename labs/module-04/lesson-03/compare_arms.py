"""Score the base model and every arm of lab 04.3 on Eval v1 and Eval v0, then apply the contract's rule.

    python labs/module-04/lesson-03/compare_arms.py                        # free CPU arms from extend_arms.py
    LAB_TARGET=solution python labs/module-04/lesson-03/compare_arms.py

For each model (scores are cached next to its checkpoint, so a rerun only scores new arms):

* Eval v1 at the original and the new length, with W = the original length for the context gain;
* Eval v0 at the original length (held-out loss on the 256 fixed windows and LAMBADA target
  log-probability), paired with the base model's: the short-context regression.

"base+yarn" is the base model with static YaRN switched on and no training: what the rule alone does.

The decision uses YOUR ``decide`` from lab.py with the contract's thresholds (``--max-regression``,
``--min-gain``). Every extended arm is also compared with ``ctrl-short`` (same tokens, no extension),
which separates "the extension helped" from "more training helped".
"""

import argparse
import json
from pathlib import Path

import torch

from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.evals import suite_v0, suite_v1 as v1
from frontierlab.labkit import load_target
from frontierlab.stats import bootstrap_ci, paired_bootstrap

lab = load_target(str(Path(__file__).parent / "test_lab.py"))
ARMS = ["ctrl-short", "yarn-within", "yarn-random", "pi-within"]


def scores(run: Path, L0: int, L1: int, a, vocab, data_long, val, encode, rope=None, tag="") -> tuple[dict, dict]:
    p1, p0 = run / f"eval_v1_{L0}_{L1}{tag}.json", run / f"eval_v0_{L0}{tag}.json"
    model = None
    if not p1.exists() or not p0.exists():
        model = v1.load_model(run, a.device, rope)
    if not p1.exists():
        r = v1.run_suite(model, vocab, data_long, [L0, L1], L0, n=a.n, max_docs=a.max_docs, device=a.device)
        p1.write_text(json.dumps(r))
    if not p0.exists():
        r = suite_v0.run_suite(model, val, encode, n_windows=256, T=L0, n_lambada=a.n_lambada, device=a.device)
        p0.write_text(json.dumps(r))
    return json.loads(p1.read_text()), json.loads(p0.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="runs/m04/base-cpu")
    ap.add_argument("--prefix", default="runs/m04/cpu-", help="arm folders are <prefix><arm>")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--original", type=int, default=256)
    ap.add_argument("--new", type=int, default=1024)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-docs", type=int, default=100)
    ap.add_argument("--n-lambada", type=int, default=None, help="default: all 5,153 passages")
    ap.add_argument("--max-regression", type=float, default=0.02, help="nats of held-out loss at the original length")
    ap.add_argument("--min-gain", type=float, default=0.0, help="nats of context gain at the new length, W = original")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    long_root = DEFAULT_OUT.parent / "v0-long"
    data_long = TokenData("val", long_root if (long_root / "meta.json").exists() else DEFAULT_OUT)
    val = TokenData("val")
    vocab = v1.vocab_from_tokenizer(DEFAULT_OUT / "tokenizer.json", val)
    suite_v0.download_lambada()
    encode = suite_v0.data_v0_encoder()
    L0, L1, W = a.original, a.new, str(a.original)
    runs = {"base": Path(a.base), **{arm: Path(a.prefix + arm) for arm in a.arms if Path(a.prefix + arm).exists()}}
    res = {name: scores(run, L0, L1, a, vocab, data_long, val, encode) for name, run in runs.items()}
    # the base model with static YaRN switched on and no training (lesson 04.2): what the rule alone does
    yarn = {"type": "yarn", "factor": L1 / L0, "original_max_position_embeddings": L0}
    res = {"base": res["base"], "base+yarn": scores(Path(a.base), L0, L1, a, vocab, data_long, val, encode, yarn, "_yarn"),
           **{k: v for k, v in res.items() if k != "base"}}
    b1, b0 = res["base"]

    def gain(r1):
        nat = next(n for n in r1["natural"] if n["length"] == L1)
        return nat["context_gain"][W]

    def far_loss(r1):
        nat = next(n for n in r1["natural"] if n["length"] == L1)
        return nat["doc_mean_by_bucket"][-1], nat["edges"][-2]

    print(f"Original length {L0}, new length {L1}; context gain at {L1} with W = {W}; Eval v0 at {L0}.")
    print(f"Rule: adopt if held-out regression upper bound <= {a.max_regression} and gain lower bound >= {a.min_gain}.\n")
    for name, (r1, r0) in res.items():
        g = bootstrap_ci(gain(r1), n_boot=4000)
        line = f"{name:<12} context gain {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]"
        if name != "base":
            reg = v1.short_context_regression(b0, r0, n_boot=4000)
            h, lp = reg["heldout_loss_diff"], reg["lambada_logprob_diff"]
            fd = paired_bootstrap(far_loss(r1)[0], far_loss(b1)[0], n_boot=4000)
            dec = lab.decide(h["ci"], (g[1], g[2]), a.max_regression, a.min_gain)
            line += (f" | vs base: held-out {h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"
                     f", LAMBADA logp {lp['mean_diff']:+.3f} [{lp['ci'][0]:+.3f}, {lp['ci'][1]:+.3f}]"
                     f", far loss {fd['mean_diff']:+.4f} [{fd['ci'][0]:+.4f}, {fd['ci'][1]:+.4f}] | {dec.upper()}")
        print(line)
    print(f"\nfar loss = mean loss on document positions [{far_loss(b1)[1]}, {L1 - 1}) at length {L1}, paired by document.")
    if "ctrl-short" in res:
        print("\nExtended arms against ctrl-short (same tokens, no extension): context gain and far loss, paired")
        c1 = res["ctrl-short"][0]
        for name, (r1, _) in res.items():
            if name in ("base", "base+yarn", "ctrl-short"):
                continue
            d = paired_bootstrap(gain(r1), gain(c1), n_boot=4000)
            f = paired_bootstrap(far_loss(r1)[0], far_loss(c1)[0], n_boot=4000)
            print(f"  {name:<12} gain {d['mean_diff']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]   "
                  f"far loss {f['mean_diff']:+.4f} [{f['ci'][0]:+.4f},{f['ci'][1]:+.4f}]")
    print("\nRetrieval evidence effect at the new length (mean over depths; paired CI per cell in the JSON):")
    for name, (r1, _) in res.items():
        cells = [c for c in r1["synthetic"] if c["length"] == L1 and c["hops"] == 1]
        ev = [v1.evidence_effect(c)["mean_diff"] for c in cells]
        acc = [v1.summarize_scores(c["scores"])["acc"][0] for c in cells]
        print(f"  {name:<12} evidence {sum(ev) / len(ev):+.4f}   accuracy {sum(acc) / len(acc):.3f}")


if __name__ == "__main__":
    main()
