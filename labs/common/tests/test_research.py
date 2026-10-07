"""Tests for frontierlab.research (Module 19). No training, no downloads; a few seconds."""

import math

import numpy as np
import pytest

from frontierlab.research import contract as C
from frontierlab.research import proposals as P
from frontierlab.research import reproduce as R
from frontierlab.research import writeup as W
from frontierlab.stats import min_detectable_effect


# ---------------------------------------------------------------- proposals (19.1)

def _prop(cid, v, p, c):
    return P.Proposal(cid, "Does it help?", "adopt or not", v, p, c, "toy run", "no effect at 2 sizes")


def test_score_and_rank():
    a, b = _prop("gspo-vs-grpo", 8, 0.5, 40), _prop("qkclip-vs-qknorm", 5, 0.4, 4)
    assert math.isclose(P.score(a), 0.1) and math.isclose(P.score(b), 0.5)
    assert [p.claim_id for p, _ in P.rank([a, b])] == ["qkclip-vs-qknorm", "gspo-vs-grpo"]


def test_rank_robustness_bounds():
    clear = [_prop("a", 9, 0.9, 1), _prop("own:b", 1, 0.1, 100), _prop("own:c", 1, 0.1, 100)]
    assert P.rank_robustness(clear, factor=2.0) == 1.0
    tie = [_prop("own:a", 5, 0.5, 10), _prop("own:b", 5, 0.5, 10)]
    r = P.rank_robustness(tie, factor=3.0, n=8000)
    assert 0.4 < r < 0.6
    assert P.rank_robustness([clear[0]]) == 1.0


def test_power_inverts_mde():
    sd, n = 0.03, 3
    mde = min_detectable_effect(sd, n)
    assert abs(P.power(mde, sd, n) - 0.80) < 0.01
    assert P.power(0.0, sd, n) == pytest.approx(0.05, abs=1e-3)
    assert P.power(10 * mde, sd, n) > 0.999


def test_proxy_trend():
    assert P.proxy_trend([1, 2, 3], [0.01, -0.01, 0.005], noise=0.01) == "below noise"
    assert P.proxy_trend([1, 2, 3], [0.05, 0.01, -0.06], noise=0.01) == "sign flips"
    assert P.proxy_trend([3, 1, 2], [0.10, 0.03, 0.06], noise=0.01) == "grows"      # sorted by size
    assert P.proxy_trend([1, 2, 3], [0.10, 0.06, 0.03], noise=0.01) == "shrinks"
    assert P.proxy_trend([1, 2, 3], [0.05, 0.06, 0.055], noise=0.01) == "flat"


def test_check_proposals():
    good = [_prop("gspo-vs-grpo", 8, 0.5, 40), _prop("qkclip-vs-qknorm", 5, 0.4, 4), _prop("own:x", 3, 0.6, 2)]
    assert P.check_proposals(good) == []
    bad = good[:2] + [P.Proposal("own:x", "a statement", "", 11, 0, -1, "", "")]
    probs = P.check_proposals(bad)
    for word in ("value", "p_decisive", "cost", "decision is empty", "proxy is empty", "kill is empty", "question"):
        assert any(word in p for p in probs), word
    assert any("capstone" in p for p in P.check_proposals([_prop("own:a", 5, .5, 1), _prop("own:b", 5, .5, 1),
                                                            _prop("own:c", 5, .5, 1)]))
    assert all(set(c) >= {"title", "claim", "url", "lab_cost_gpu_h", "proxy", "proxy_risk"} for c in P.CAPSTONE_CLAIMS.values())
    assert len(P.CAPSTONE_CLAIMS) == 6


# ---------------------------------------------------------------- reproduction (19.2)

def test_lr_sensitivity_matches_definition():
    l0 = math.log(8192)
    assert math.isclose(R.lr_sensitivity({1e-3: 6.3, 3e-3: 6.2, 1e-2: 6.4}, l0), (6.3 + 6.2 + 6.4) / 3 - 6.2)
    # a diverged run counts as l0, a loss above l0 is capped at l0
    assert math.isclose(R.lr_sensitivity({1e-2: 6.0, 3e-1: float("nan")}, l0), (6.0 + l0) / 2 - 6.0)
    assert math.isclose(R.lr_sensitivity({1e-2: 6.0, 3e-1: 12.0}, l0), (6.0 + l0) / 2 - 6.0)
    assert R.lr_sensitivity({1e-2: 6.0}, l0) == 0.0


def test_t_interval_against_hand_value():
    m, lo, hi = R.t_interval([1.0, 2.0, 3.0])
    assert m == 2.0 and math.isclose(hi - m, 4.303 * 1.0 / math.sqrt(3), rel_tol=1e-9)
    with pytest.raises(ValueError):
        R.t_interval([1.0])


@pytest.mark.parametrize("effects,want", [
    ([0.30, 0.25, 0.28], "reproduced"),
    ([-0.30, -0.25, -0.28], "contradicted"),
    ([0.01, -0.01, 0.0], "not reproduced"),
    ([0.20, -0.05, 0.10], "inconclusive"),
    ([0.040, 0.041, 0.042], "not reproduced"),     # clearly positive, but below the tolerance: no effect that counts
])
def test_decide_direction(effects, want):
    assert R.decide(effects, tolerance=0.057)["direction"] == want


def test_decide_magnitude():
    assert R.decide([0.3, 0.3, 0.31], 0.05)["magnitude"] == "not comparable"
    assert R.decide([0.3, 0.3, 0.31], 0.05, published=0.32, magnitude_tol=0.05)["magnitude"] == "matches"
    assert R.decide([0.3, 0.3, 0.31], 0.05, published=0.6, magnitude_tol=0.05)["magnitude"] == "differs"
    with pytest.raises(ValueError):
        R.decide([0.1, 0.2], -1.0)


def _record(**kw):
    devs = [R.Deviation(a, "paper", "ours" if a == "scale" else "paper", "budget", "unknown") for a in R.ASPECTS]
    eff = [0.30, 0.25, 0.28]
    base = dict(claim="X reduces Y", source="Paper, Figure 1", url="https://arxiv.org/abs/0000.00000",
                tolerance=0.057, tolerance_rule="2 x seed std", stated_on="2026-10-01", results_on="2026-10-02",
                effects=eff, outcome=R.decide(eff, 0.057), deviations=devs, author_contact="not needed",
                scope="toy preset at this scale only")
    base.update(kw)
    return R.Record(**base)


def test_record_validate():
    assert R.validate(_record()) == []
    assert any("stated after" in p for p in R.validate(_record(stated_on="2026-10-03")))
    wrong = dict(R.decide([0.30, 0.25, 0.28], 0.057), direction="inconclusive")
    assert any("numbers give 'reproduced'" in p for p in R.validate(_record(outcome=wrong)))
    rec = _record()
    rec.deviations = rec.deviations[1:]
    assert any("'scale'" in p for p in R.validate(rec))
    rec = _record()
    rec.deviations[0].recorded = "after"
    assert any("recorded after" in p for p in R.validate(rec))
    assert any("name the scale" in p for p in R.validate(_record(scope="it works")))
    assert any("author contact" in p for p in R.validate(_record(author_contact="")))
    md = R.render(_record())
    assert "Direction: reproduced" in md and "| scale |" in md


# ---------------------------------------------------------------- write-ups and figures (19.3)

def _card(run, seed=0, steps=300, qk_norm=True, clip=None, params=1_800_000, data="Data-v0"):
    return {"run": run, "args": {"seed": seed, "data_seed": None, "steps": steps, "lr": 0.01, "batch": 16, "seq": 128},
            "config": {"qk_norm": qk_norm}, "optim": {"qk_clip": clip, "optimizer": "muon"},
            "data": {"name": data}, "budget": {"steps": steps, "tokens": steps * 2048},
            "params": {"total": params}}


def test_lint_clean_claim():
    cards = {f"clip-s{s}": _card(f"clip-s{s}", s, qk_norm=False, clip=15.0) for s in range(3)}
    cards.update({f"norm-s{s}": _card(f"norm-s{s}", s) for s in range(3)})
    claim = {"id": "c1", "text": "QK-Clip lowers held-out loss against QK-norm at 1.8M parameters.",
             "arms": {"clip": [f"clip-s{s}" for s in range(3)], "norm": [f"norm-s{s}" for s in range(3)]},
             "changed": ["config.qk_norm", "optim.qk_clip"], "axis": "tokens", "interval": [-0.02, -0.004],
             "checkpoint": {"clip": "final", "norm": "final"}, "selection_split": "val",
             "scope": "toy preset, 300 steps"}
    assert W.lint_claims([claim], cards) == []


def test_lint_finds_each_planted_problem():
    cards = {"clip-s0": _card("clip-s0", 0, steps=400, qk_norm=False, clip=15.0),
             "norm-s0": _card("norm-s0", 0), "norm-s1": _card("norm-s1", 1)}
    claim = {"id": "c1", "text": "QK-Clip beats QK-norm and should transfer to frontier models.",
             "arms": {"clip": ["clip-s0"], "norm": ["norm-s0", "norm-s1", "ghost"]},
             "changed": ["config.qk_norm", "optim.qk_clip"], "axis": "tokens", "interval": None,
             "checkpoint": {"clip": "best", "norm": "final"}, "selection_split": "test"}
    codes = [i.code for i in W.lint_claims([claim], cards)]
    for code in ("missing", "seeds", "uncertainty", "budget", "checkpoint", "scope"):
        assert code in codes, code
    other = dict(claim, interval=[-0.01, 0.02], checkpoint={}, selection_split="val", text="Clip is lower.")
    cards["clip-s0"] = _card("clip-s0", 0, qk_norm=False, clip=15.0, data="Data-v1")
    issues = W.lint_claims([other], cards)
    assert any(i.code == "uncertainty" and "contains 0" in i.message for i in issues)
    assert any(i.code == "control" and i.message.find("data.name") >= 0 for i in issues)


def test_parse_claims_from_markdown():
    md = "# Note\n\ntext\n\n```yaml claims\n- id: c1\n  text: a\n- id: c2\n  text: b\n```\n\n```yaml\nx: 1\n```\n"
    assert [c["id"] for c in W.parse_claims(md)] == ["c1", "c2"]


def test_lint_figure():
    good = {"claim": "QK-norm lowers LR sensitivity at 1.8M parameters", "x": "lr", "contract_axis": "tokens",
            "uncertainty": "min-max over 3 seeds",
            "arms": {"qknorm": {"n_seeds": 3, "tokens_per_step": 2048}, "noqk": {"n_seeds": 3, "tokens_per_step": 2048}},
            "caption": "Final held-out loss; bands are the min-max range over 3 seeds."}
    assert W.lint_figure(good) == []
    bad = {"claim": ["a", "b"], "x": "steps", "contract_axis": "tokens", "uncertainty": "none",
           "arms": {"a": {"n_seeds": 1, "tokens_per_step": 2048, "x_end": 300},
                    "b": {"n_seeds": 3, "tokens_per_step": 4096, "x_end": 400}},
           "caption": "Loss curves."}
    codes = [i.code for i in W.lint_figure(bad)]
    assert codes.count("budget") == 2 and "seeds" in codes and codes.count("uncertainty") == 2 and "scope" in codes
    assert any("compares at equal" in i.message for i in W.lint_figure(dict(good, x="flops")))


# ---------------------------------------------------------------- contract and rubric (project)

def _contract(**kw):
    c = {"claim_id": "qkclip-vs-qknorm", "question": "Does QK-Clip hold max logits as low as QK-norm?",
         "decision": "use QK-Clip in Recipe-R", "hypothesis": "it does", "status": "may not appear at this scale",
         "baseline": "runs/m07/l72/main/muon-qknorm", "baseline_tuning": "lr sweep of 3",
         "changed": "QK-Clip instead of QK-norm", "held_fixed": "data, tokens, seeds, eval",
         "axis": "tokens", "axis_does_not_answer": "which is cheaper per step",
         "budget": {"hardware": "1x H100", "gpu_hours": 6, "label": "PROJECTED", "formula": "runs x FLOPs / (peak x MFU)",
                    "extra_compute": "none", "tuning_per_arm": {"baseline": 3, "method": 3}},
         "metrics": {"primary": "held-out loss", "uncertainty": "paired bootstrap over seeds", "noise_floor": 0.01,
                     "n_seeds": 3, "expected_effect": 0.03},
         "decision_rule": "adopt if the 95% CI of the loss difference is within +-0.02",
         "correctness_checks": ["clip unit test", "run-card diff"], "fallback": "Module 7 pilot traces",
         "limits": ["30M parameters only", "one data set", "short runs"], "stated_on": "2026-10-07"}
    c.update(kw)
    return c


def test_contract_validate():
    assert C.validate(_contract()) == []
    assert any("underpowered" in p for p in C.validate(_contract(metrics=dict(_contract()["metrics"], expected_effect=0.01))))
    assert any("tuning budgets" in p for p in C.validate(_contract(budget=dict(_contract()["budget"],
                                                                        tuning_per_arm={"baseline": 1, "method": 8}))))
    assert any("number" in p for p in C.validate(_contract(decision_rule="adopt if better")))
    assert any("fallback" in p for p in C.validate(_contract(fallback="none")))
    assert any("model scale" in p for p in C.validate(_contract(limits=["a", "b", "c"])))
    assert any("capstone list" in p for p in C.validate(_contract(claim_id="own:x")))
    assert C.validate(_contract(claim_id="own:x"), capstone=False) == []
    assert any("after the first result" in p for p in C.validate(_contract(), results_on="2026-10-01"))
    assert any("formula" in p for p in C.validate(_contract(budget=dict(_contract()["budget"], formula=""))))
    assert C.validate({"question": "x"})[0].startswith("claim_id")
    assert math.isclose(C.mde(_contract()), min_detectable_effect(0.01, 3))


def test_rubric_total():
    keys = [k for k, _ in C.RUBRIC]
    assert C.rubric_total({k: 2 for k in keys})["passed"]
    r = C.rubric_total(dict({k: 2 for k in keys}, limits=0))
    assert r["total"] == 12 and not r["passed"] and r["zeros"] == ["limits"]
    assert not C.rubric_total({k: 1 for k in keys})["passed"]           # 7 of 14
    with pytest.raises(ValueError):
        C.rubric_total({"question": 2})


def test_all_parts_import_together():
    import frontierlab.research as research
    assert {"contract", "proposals", "reproduce", "writeup"} <= set(research.__all__)
    assert np.isfinite(P.power(0.1, 0.05, 3))


def test_validate_proposal_adds_reproduction_fields():
    c = _contract(source="Kimi K2 section 2.1, https://arxiv.org/abs/2507.20534",
                  reproduction={"direction": "clip holds the logit", "tolerance": 0.02, "tolerance_rule": "2 x noise",
                                "magnitude": "not comparable"},
                  extension="MLA with QK-Clip", proxy="toy ladder", kill="clip never binds")
    assert C.validate_proposal(c) == []
    assert any("below the noise floor" in p for p in C.validate_proposal(dict(c, reproduction=dict(c["reproduction"], tolerance=0.001))))
    assert any("source" in p for p in C.validate_proposal(dict(c, source="Kimi K2")))
    assert any(p.startswith("extension") for p in C.validate_proposal(dict(c, extension="")))
