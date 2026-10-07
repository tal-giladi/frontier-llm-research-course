"""Checks for the Module 19 project files.

    pytest labs/module-19/project
    PROPOSAL=runs/m19/project/proposal.yaml pytest labs/module-19/project    # also checks your own proposal
"""

import os
from pathlib import Path

import pytest
import yaml

from frontierlab.research.contract import validate_proposal

HERE = Path(__file__).resolve().parent


def load(name):
    return yaml.safe_load((HERE / name).read_text(encoding="utf-8"))


def test_example_proposal_is_valid():
    assert validate_proposal(load("example_proposal.yaml")) == []


def test_template_is_not_valid_until_filled():
    assert len(validate_proposal(load("proposal_template.yaml"))) > 5


def test_buggy_proposal_has_the_four_planted_problems():
    probs = validate_proposal(load("buggy_proposal.yaml"))
    for word in ("underpowered", "tuning budgets", "decision_rule", "below the noise floor"):
        assert any(word in p for p in probs), word
    assert len(probs) == 4


@pytest.mark.skipif(not os.environ.get("PROPOSAL"), reason="set PROPOSAL=<path> to check your own proposal")
def test_your_proposal():
    c = yaml.safe_load(Path(os.environ["PROPOSAL"]).read_text(encoding="utf-8"))
    assert validate_proposal(c) == []
