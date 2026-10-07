from pathlib import Path

import pytest
import yaml

from frontierlab.labkit import load_target
from frontierlab.research import writeup as W

lab = load_target(__file__)
HERE = Path(__file__).resolve().parent
PLANTED = ("missing seeds", "unmatched budget", "cherry-picked checkpoint", "claim beyond evidence")


def _claim_and_arms(which):
    claim = W.parse_claims((HERE / which / "writeup.md").read_text(encoding="utf-8"))[0]
    cards = W.load_cards(HERE / which / "cards")
    return claim, {arm: [cards[r] for r in runs] for arm, runs in claim["arms"].items()}


def test_seed_issues_on_both_writeups():
    _, arms = _claim_and_arms("flawed")
    assert lab.seed_issues(arms) == ["qk-clip", "qk-norm"]
    _, arms = _claim_and_arms("fixed")
    assert lab.seed_issues(arms) == []
    same = {"a": [{"args": {"seed": 1, "data_seed": None}}, {"args": {"seed": 1, "data_seed": None}}]}
    assert lab.seed_issues(same) == ["a"]                       # two runs, one seed


def test_checkpoint_issues():
    claim, _ = _claim_and_arms("flawed")
    assert lab.checkpoint_issues(claim) == ["qk-clip: step 250 (best of 8)", "selected on the test split"]
    claim, _ = _claim_and_arms("fixed")
    assert lab.checkpoint_issues(claim) == []


@pytest.mark.parametrize("text,flag", [
    ("QK-Clip beats QK-norm and the gain should transfer to frontier models.", True),
    ("This holds for all models we will ever train.", True),
    ("The improvement will grow with scale.", True),
    ("Muon is always more stable.", True),
    ("At 1.8M parameters, the two arms reach the same held-out loss within 0.02 nats.", False),
    ("On the toy preset, the maximum logit stayed below 15 in all three seeds.", False),
])
def test_scope_issue(text, flag):
    assert (lab.scope_issue(text) is not None) == flag


def test_figure_issues_match_shared_linter():
    specs = [yaml.safe_load((HERE / w / "figure.yaml").read_text(encoding="utf-8")) for w in ("flawed", "fixed")]
    base = specs[1]
    specs += [dict(base, x="flops"), dict(base, caption="Loss per arm."),
              dict(base, x="steps", arms={"a": {"n_seeds": 3, "tokens_per_step": 2048}, "b": {"n_seeds": 3, "tokens_per_step": 4096}}),
              dict(base, claim=["a"]), dict(base, uncertainty="none")]
    for s in specs:
        assert lab.figure_issues(s) == {i.code for i in W.lint_figure(s)}, s


def test_review_names_every_planted_problem():
    found = {r.get("problem") for r in lab.REVIEW}
    for p in PLANTED:
        assert p in found, f"your review does not name '{p}'"
    for r in lab.REVIEW:
        for key in ("where", "evidence", "request"):
            assert str(r.get(key, "")).strip(), f"{r.get('problem')}: '{key}' is empty"
