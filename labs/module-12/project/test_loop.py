"""One test per detail of lessons 12.2-12.3, run against the pieces the RL loop actually uses.

    pytest labs/module-12/project                      # the course loop: every test passes
    LOOP_HOOKS=buggy pytest labs/module-12/project     # the debugging task's pieces: which tests fail?

Your own loop gets the same tests: point ``pieces()`` at your functions.
"""

import os

import pytest
import torch

from frontierlab.model import LM
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import kl as K
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain.policy import response_mask, sample, token_logprobs
from frontierlab.posttrain.rl import RLConfig, Sampler
from frontierlab.posttrain.sft import policy_config, save_policy
from frontierlab.posttrain.tasks import encode_prompts, make_problems
from frontierlab.posttrain.tokenizer import EOS, PAD


def pieces():
    """The advantage, loss-mask and old-log-prob pieces, with the course loop's defaults."""
    p = {"advantages": A.group_advantages, "loss_mask": lambda m, ro: m,
         "old_logp": lambda **kw: kw["default"]}
    if os.environ.get("LOOP_HOOKS") == "buggy":
        from frontierlab.labkit import load_path
        from pathlib import Path
        bug = load_path(str(Path(__file__).resolve().parent / "buggy_loop.py"))
        p.update(bug.BUGGY_HOOKS)
    return p


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    model = LM(policy_config())
    path = tmp_path_factory.mktemp("init") / "init.pt"
    save_policy(model, path)
    probs = [p for p in make_problems(4, 2, seed=0) for _ in range(4)]
    ro = sample(model, encode_prompts(probs), 6, 1.0, torch.Generator().manual_seed(0))
    return model, path, ro


# 12.2 -------------------------------------------------------------------------------------------------

def test_advantage_depends_only_on_own_group():
    R = (torch.rand(5, 4, generator=torch.Generator().manual_seed(1)) < 0.5).float()
    adv = pieces()["advantages"](R, "mean", "group")
    R2 = R.clone()
    R2[1:] = 1 - R2[1:]                                     # change every other group
    adv2 = pieces()["advantages"](R2, "mean", "group")
    assert torch.allclose(adv[0], adv2[0]), "group 0's advantages changed when only other groups changed"
    assert torch.allclose(adv.sum(1), torch.zeros(5), atol=1e-5), "mean baseline: advantages sum to 0 per group"


def test_zero_variance_group_has_no_gradient_signal():
    R = torch.tensor([[1.0, 1, 1, 1], [0.0, 1, 0, 0]])
    adv = pieces()["advantages"](R, "mean", "group")
    assert torch.all(adv[0] == 0) and A.zero_variance(R).tolist() == [True, False]


def test_kl_in_loss_estimator_gradient():
    out = K.exact_categorical(torch.tensor([0.3, -1.0, 0.5, 2.0]), torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(out["k2"], out["grad_reverse_kl"], atol=1e-12)
    assert out["k1"].abs().max() < 1e-12


def test_kl_in_reward_is_detached(setup):
    model, _, ro = setup
    logp = token_logprobs(model, ro.tokens, ro.prompt_len)
    pen = K.kl_penalty_reward(logp, logp.detach() - 0.1, ro.mask, 0.5)
    assert not pen.requires_grad and torch.allclose(pen, 0.5 * 0.1 * ro.mask.sum(-1))


def test_entropy_is_of_the_full_distribution(setup):
    model, _, ro = setup
    _, ent = token_logprobs(model, ro.tokens, ro.prompt_len, with_entropy=True)
    lp = torch.log_softmax(model(ro.tokens).logits[:, ro.prompt_len - 1:-1].float(), -1)
    assert torch.allclose(ent, -(lp.exp() * lp).sum(-1), atol=1e-5)


# 12.3 -------------------------------------------------------------------------------------------------

def test_aggregation_length_weights():
    m = torch.tensor([[1.0, 0, 0, 0], [1.0, 1, 1, 1]])
    assert Lo.token_weights(m, "token_mean")[0, 0] == pytest.approx(Lo.token_weights(m, "token_mean")[1, 3])
    w = Lo.token_weights(m, "seq_mean_token_mean")
    assert float(w[0, 0]) == pytest.approx(4 * float(w[1, 0]))


def test_truncated_response_is_fully_masked_and_unfinished():
    r = torch.tensor([[5, 6, 7], [5, EOS, PAD]])
    m, fin = response_mask(r)
    assert m.tolist() == [[1, 1, 1], [1, 1, 0]] and fin.tolist() == [False, True]
    assert Lo.soft_overlong_penalty(torch.tensor([3.0]), 3, 1).item() == pytest.approx(-1.0)


def test_padding_after_eos_does_not_change_the_loss():
    from types import SimpleNamespace
    resp = torch.tensor([[5, EOS, PAD, PAD], [5, 6, 7, EOS], [5, 6, EOS, PAD], [EOS, PAD, PAD, PAD]])
    m0, _ = response_mask(resp)
    mask = pieces()["loss_mask"](m0, SimpleNamespace(response=resp))
    torch.manual_seed(2)
    logp = torch.randn(4, 4) - 1
    adv = torch.tensor([1.0, -0.5, 0.3, -1.0])
    old = logp.detach() + 0.1
    loss, _ = Lo.policy_loss(logp, old, adv, mask)
    logp2 = torch.where(m0 == 0, logp - 0.5, logp)          # change only positions after EOS
    loss2, _ = Lo.policy_loss(logp2, old, adv, mask)
    assert loss.item() == pytest.approx(loss2.item(), abs=1e-6), "positions after EOS are in the loss"


def test_on_policy_ratio_is_one_on_first_minibatch(setup):
    model, path, ro = setup
    cfg = RLConfig(init=str(path))
    model2 = LM(policy_config())                            # the policy after some updates
    model2.load_state_dict({k: v + 0.01 * torch.randn_like(v) for k, v in model.state_dict().items()})
    old2 = pieces()["old_logp"](default=token_logprobs(model2, ro.tokens, ro.prompt_len).detach(), policy=model2,
                                rollout=ro, cfg=cfg)
    cur = token_logprobs(model2, ro.tokens, ro.prompt_len)
    rho = Lo.token_ratio(cur, old2)
    assert torch.allclose(rho * ro.mask, ro.mask, atol=1e-5), "on-policy: the first minibatch's ratio must be 1"


def test_stale_sampler_uses_oldest_snapshot(setup):
    model, _, _ = setup
    s = Sampler(model, staleness=2, dtype="fp32")
    m2 = LM(policy_config())
    for _ in range(3):
        s.push(m2)
    assert len(s.snapshots) == 3
    assert all(torch.equal(s.behaviour_state()[k], v) for k, v in m2.state_dict().items())


def test_truncated_importance_weight():
    w = Lo.tis_weight(torch.tensor([0.0, -2.0]), torch.tensor([-3.0, -1.0]), cap=2.0)
    assert torch.allclose(w, torch.tensor([2.0, torch.exp(torch.tensor(-1.0)).item()]))
