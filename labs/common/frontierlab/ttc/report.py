"""Budget-matched comparison of test-time-compute strategies and the recommendation rule (15.1, project).

The same analysis runs on the free-CPU world and on the main path's saved samples. Inputs per problem:

* a **pool** per sampling configuration (a mode or thinking budget): n candidates, each with ``answer``,
  ``correct``, ``gen_tokens``, ``scored_tokens`` (what a verifier reads) and ``score`` (verifier probability);
* **single-chain** results (one greedy or sampled chain per problem at a given reasoning length);
* **search** results (per problem: ``correct``, ``decode_tokens``, ``verifier_tokens``).

Every row reports its spend in policy-token equivalents per problem (``frontierlab.ttc.budget``), its
latency from a :class:`~frontierlab.ttc.budget.LatencyModel`, its success with a 95% bootstrap interval over
problems, and keeps the per-problem success vector so that two rows can be compared *paired*.

Oracle rows (pass@N) are reported next to the procedures, labelled, and never recommended: no deployed
system knows which sample is right.

**Decision rule** (:func:`recommend`, stated before the runs in the lab's contract): among procedures whose
spend is within the budget and whose latency is within the target, take the one with the highest mean
success; compare it, paired over problems, with the cheapest procedure whose success is within the noise of
it. If the 95% interval of (best - cheaper) includes 0, recommend the cheaper one.
"""

from __future__ import annotations

import numpy as np

from frontierlab.stats import bootstrap_ci, paired_bootstrap
from frontierlab.ttc import select as SE
from frontierlab.ttc.budget import LatencyModel


def _ci(x: np.ndarray, seed: int = 0) -> tuple[float, float]:
    _, lo, hi = bootstrap_ci(x, n_boot=2000, seed=seed)
    return lo, hi


def pool_rows(name: str, pool: list[list[dict]], gold: list[str], Ns, *, policy_params: int, verifier_params: int,
              prompt_tokens: float, latency: LatencyModel | None = None, methods=("majority", "best_of_n", "weighted"),
              resamples: int = 200, seed: int = 0, oracle: bool = True, verifier: str = "ORM") -> list[dict]:
    """Rows for selection procedures over N samples drawn from ``pool`` (one list of candidates per problem)."""
    answers = [[c["answer"] for c in cands] for cands in pool]
    correct = np.array([[c["correct"] for c in cands] for cands in pool], dtype=bool)
    scores = [[c.get("score", 0.0) for c in cands] for cands in pool]
    gen = float(np.mean([c["gen_tokens"] for cands in pool for c in cands]))
    scored = float(np.mean([c.get("scored_tokens", 0.0) for cands in pool for c in cands]))
    ratio = verifier_params / policy_params if policy_params else 0.0
    rows = []
    for N in Ns:
        if oracle:
            succ = SE.oracle_pass_at_k(correct, N)
            rows.append(_row(f"oracle pass@N ({name})", "oracle", N, prompt_tokens, N * gen, 0.0, ratio, succ,
                             latency.parallel(int(prompt_tokens), gen, N) if latency else float("nan")))
        for m in methods:
            if N == 1 and m != "majority":
                continue                                     # with one sample every procedure is the same draw
            uses_v = m in ("best_of_n", "weighted")
            succ = SE.subset_success(answers, gold, N, m, scores, resamples, seed)
            label = {"majority": "majority vote", "best_of_n": f"best-of-N ({verifier})",
                     "weighted": f"weighted vote ({verifier})"}[m]
            if N == 1:
                label = "one sample"
            rows.append(_row(f"{label} ({name})", m if N > 1 else "sample", N, prompt_tokens, N * gen,
                             N * scored if uses_v else 0.0, ratio, succ,
                             latency.parallel(int(prompt_tokens), gen, N, verify=uses_v) if latency else float("nan")))
    return rows


def single_row(name: str, correct, gen_tokens: float, *, policy_params: int, prompt_tokens: float,
               latency: LatencyModel | None = None) -> dict:
    succ = np.asarray(correct, dtype=float)
    return _row(f"single chain ({name})", "single", 1, prompt_tokens, gen_tokens, 0.0, 0.0, succ,
                latency.parallel(int(prompt_tokens), gen_tokens, 1) if latency else float("nan"))


def search_row(name: str, res: list[dict], *, policy_params: int, verifier_params: int, prompt_tokens: float,
               latency_s: float) -> dict:
    succ = np.array([float(r["correct"]) for r in res])
    dec = float(np.mean([r["decode_tokens"] for r in res]))
    ver = float(np.mean([r["verifier_tokens"] for r in res]))
    return _row(f"PRM beam search ({name})", "search", 0, prompt_tokens, dec, ver,
                verifier_params / policy_params, succ, latency_s)


def _row(label, family, N, prefill, decode, verifier_tokens, ratio, succ, latency_s) -> dict:
    succ = np.asarray(succ, dtype=float)
    lo, hi = _ci(succ)
    return {"label": label, "family": family, "N": int(N), "prefill": float(prefill), "decode": float(decode),
            "verifier_tokens": float(verifier_tokens),
            "pte": float(prefill + decode + ratio * verifier_tokens),
            "success": float(succ.mean()), "ci": (lo, hi), "latency_s": float(latency_s), "_per_problem": succ}


def eligible(rows: list[dict], budget: float, latency_target: float | None = None) -> list[dict]:
    out = [r for r in rows if r["family"] != "oracle" and r["pte"] <= budget + 1e-9]
    if latency_target is not None:
        out = [r for r in out if r["latency_s"] <= latency_target]
    return out


def best_per_family(rows: list[dict], budget: float, latency_target: float | None = None) -> list[dict]:
    """The best row of each family (single, sample, majority, best_of_n, weighted, search) within the limits."""
    best: dict = {}
    for r in eligible(rows, budget, latency_target):
        k = r["family"] if r["family"] != "sample" else "single"
        if k not in best or r["success"] > best[k]["success"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: -r["success"])


def recommend(rows: list[dict], budget: float, latency_target: float | None = None, seed: int = 0) -> dict:
    """The decision rule of the module docstring. Returns the choice, the runner-up comparison and why."""
    cand = eligible(rows, budget, latency_target)
    if not cand:
        return {"choice": None, "why": "no procedure fits the budget and the latency target"}
    top = max(cand, key=lambda r: (r["success"], -r["pte"]))
    cheaper = sorted([r for r in cand if r["pte"] < top["pte"]], key=lambda r: r["pte"])
    for r in cheaper:
        d = paired_bootstrap(top["_per_problem"], r["_per_problem"], n_boot=4000, seed=seed)
        if d["ci"][0] <= 0:                                 # not distinguishable from the best: take the cheaper
            return {"choice": r, "versus": top, "diff": d,
                    "why": f"'{top['label']}' is not better than the cheaper '{r['label']}' beyond noise "
                           f"(diff {d['mean_diff']:+.3f}, 95% CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}])"}
    nxt = sorted([r for r in cand if r is not top], key=lambda r: -r["success"])
    d = paired_bootstrap(top["_per_problem"], nxt[0]["_per_problem"], n_boot=4000, seed=seed) if nxt else None
    return {"choice": top, "versus": nxt[0] if nxt else None, "diff": d,
            "why": f"'{top['label']}' is best within the limits and every cheaper procedure is worse beyond noise"}


def format_rows(rows: list[dict]) -> str:
    lines = [f"{'procedure':44s} {'N':>3s} {'policy tok':>10s} {'verif tok':>9s} {'pte':>8s} {'success':>8s}  "
             f"{'95% CI':15s} {'latency':>9s}"]
    for r in rows:
        lat = f"{r['latency_s'] * 1e3:7.1f}ms" if np.isfinite(r["latency_s"]) else "      n/a"
        lines.append(f"{r['label']:44s} {r['N']:3d} {r['prefill'] + r['decode']:10.1f} {r['verifier_tokens']:9.1f} "
                     f"{r['pte']:8.1f} {r['success']:8.3f}  [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] {lat}")
    return "\n".join(lines)
