"""The course's RL loop for language models: one file, every detail switchable and logged (Module 12).

    python -m frontierlab.posttrain.rl --init runs/m12/sft/policy.pt --run runs/m12/rl/grpo-s0 --steps 150

One step:

1. **Prompts.** ``prompts`` training problems (plain addition, training triples only), each repeated
   ``group`` times.
2. **Rollout.** The *sampler* generates up to ``max_new`` tokens per response. It is the policy as it
   was ``staleness`` steps ago (0 = on-policy), optionally in bfloat16 (``--sampler-dtype bf16``),
   so its probabilities differ from the trainer's, as with a separate inference engine.
3. **Reward.** The verifier scores each response (``strict`` by default); optional DAPO soft overlong
   penalty; optional KL-in-reward penalty ``beta * sum_t k1``.
4. **Advantage.** Group baseline and scale (:mod:`~frontierlab.posttrain.advantages`); zero-variance
   groups optionally filtered out of the batch.
5. **Update.** ``epochs`` passes over ``minibatches`` minibatches: clipped surrogate with the chosen
   ratio and aggregation (:mod:`~frontierlab.posttrain.losses`), optional KL-in-loss term, optional
   entropy bonus, optional truncated importance sampling against the sampler's log-probabilities.
6. **Log** one JSON line per step to ``<run>/metrics.jsonl``; evaluate on held-out prompts every
   ``eval_every`` steps; checkpoint so the same command resumes the same run.

``old_logp`` (the denominator of the ratio) is the trainer's log-probability under the policy that
generated the batch (``--stale-correction behaviour``) or, to show the bias of ignoring staleness,
under the current policy at the start of the update (``recompute``).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import kl as K
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain.policy import exact_token_kl, masked_mean, sample, token_logprobs
from frontierlab.posttrain.sft import greedy_accuracy, load_policy, save_policy
from frontierlab.posttrain.tasks import encode_prompts, make_problems, problems_from, score, split_problems


@dataclass
class RLConfig:
    init: str = "runs/m12/sft/policy.pt"
    run: str = "runs/m12/rl/default"
    steps: int = 150
    prompts: int = 16
    group: int = 8
    max_new: int = 8
    temperature: float = 1.0
    lr: float = 3e-4
    epochs: int = 1
    minibatches: int = 2
    baseline: str = "mean"                 # none | mean | loo
    scale: str = "group"                   # none | group | batch
    aggregation: str = "token_mean"        # see losses.AGGREGATIONS
    ratio: str = "token"                   # token | sequence | none
    eps_low: float = 0.2
    eps_high: float = 0.2
    kl_beta: float = 0.0
    kl_place: str = "none"                 # none | reward | loss
    kl_kind: str = "k3"                    # k1 | k2 | k3 (loss); reward always uses k1
    entropy_coef: float = 0.0
    zero_var: str = "keep"                 # keep | filter
    overlong: str = "none"                 # none | filter | soft
    l_cache: int = 2
    staleness: int = 0
    stale_correction: str = "behaviour"    # behaviour | recompute
    sampler_dtype: str = "fp32"            # fp32 | bf16
    tis_cap: float = 0.0                   # 0 = off
    verifier: str = "strict"
    digits: int = 2
    seed: int = 0
    eval_every: int = 25
    eval_n: int = 300
    ckpt_every: int = 25
    grad_clip: float = 1.0
    log_every: int = 1


def parse(argv=None) -> RLConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(RLConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    return RLConfig(**vars(ap.parse_args(argv)))


class Sampler:
    """Keeps the last ``staleness + 1`` policy snapshots; samples with the oldest, in ``dtype``."""

    def __init__(self, policy, staleness: int, dtype: str):
        self.k, self.dtype = staleness, torch.bfloat16 if dtype == "bf16" else torch.float32
        self.snapshots = [copy.deepcopy(policy.state_dict())]
        self.model = copy.deepcopy(policy).to(self.dtype).eval()
        self.trainer_view = copy.deepcopy(policy).eval()      # same weights as the sampler, trainer precision

    def push(self, policy):
        self.snapshots.append(copy.deepcopy(policy.state_dict()))
        self.snapshots = self.snapshots[-(self.k + 1):]

    def behaviour_state(self):
        return self.snapshots[0]

    def rollout(self, prompts, max_new, temperature, gen):
        state = self.behaviour_state()
        self.model.load_state_dict({k: v.to(self.dtype) if v.is_floating_point() else v for k, v in state.items()})
        self.trainer_view.load_state_dict(state)
        return sample(self.model, prompts, max_new, temperature, gen)


def evaluate(policy, held_problems, max_new: int, temperature: float, gen, verifier="strict") -> dict:
    greedy = greedy_accuracy(policy, held_problems, max_new, verifier)
    ro = sample(policy, encode_prompts(held_problems), max_new, temperature, gen)
    r = score(ro.response, held_problems, verifier)
    return {"greedy_acc": greedy, "sampled_acc": float(r.mean()), "sampled_len": float(ro.lengths.mean()),
            "sampled_trunc": float((~ro.finished).float().mean())}


def train(cfg: RLConfig, hooks: dict | None = None) -> dict:
    """Run (or resume) the loop. ``hooks`` replaces pieces: ``advantages(R, baseline, scale)``,
    ``loss_mask(mask, rollout)``, ``old_logp(default=..., policy=..., rollout=..., cfg=...)`` and
    ``policy_loss(...)`` (same signature as :func:`losses.policy_loss`). Labs pass the learner's functions;
    the project's planted-bug exercise passes broken ones."""
    hooks = hooks or {}
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    policy = load_policy(cfg.init)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen_seed = 10_000 + cfg.seed
    train_triples, held = split_problems(cfg.digits)
    held_problems = problems_from(held, ".", cfg.digits, "+")[:cfg.eval_n]
    sampler = Sampler(policy, cfg.staleness, cfg.sampler_dtype)
    step = 0
    ck_path = run / "checkpoint.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        sampler.snapshots = ck["snapshots"]
        step = ck["step"]
        print(f"resumed from step {step}")
    else:
        from frontierlab.runcard import write_run_card
        write_run_card(run, question="Module 12 RL loop", config=asdict(cfg), args=asdict(cfg),
                       budget={"steps": cfg.steps, "responses": cfg.steps * cfg.prompts * cfg.group})
    log = JsonlLogger(run / "metrics.jsonl")
    G = cfg.group
    adv_fn = hooks.get("advantages", A.group_advantages)
    t0 = time.perf_counter()
    while step < cfg.steps:
        seed_problems = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        probs = make_problems(cfg.prompts, cfg.digits, "+", ".", seed=seed_problems, exclude=held)
        problems = [p for p in probs for _ in range(G)]
        prompts = encode_prompts(problems)
        ro = sampler.rollout(prompts, cfg.max_new, cfg.temperature, gen)
        mask = ro.mask
        rewards = score(ro.response, problems, cfg.verifier)
        correct = rewards.clone()
        if cfg.overlong == "soft":
            rewards = rewards + Lo.soft_overlong_penalty(ro.lengths, cfg.max_new, cfg.l_cache)
        elif cfg.overlong == "filter":
            mask = Lo.overlong_filter(mask, ro.finished)
        if "loss_mask" in hooks:
            mask = hooks["loss_mask"](mask, ro)
        with torch.no_grad():
            behaviour_logp = token_logprobs(sampler.trainer_view, ro.tokens, ro.prompt_len, cfg.temperature)
            ref_logp = token_logprobs(ref, ro.tokens, ro.prompt_len, cfg.temperature)
            current_logp = token_logprobs(policy, ro.tokens, ro.prompt_len, cfg.temperature)
        old_logp = behaviour_logp if cfg.stale_correction == "behaviour" else current_logp
        old_logp = hooks.get("old_logp", lambda **kw: kw["default"])(default=old_logp, policy=policy, rollout=ro,
                                                                     cfg=cfg)
        if cfg.kl_place == "reward" and cfg.kl_beta > 0:
            rewards = rewards - K.kl_penalty_reward(current_logp, ref_logp, ro.mask, cfg.kl_beta, "k1")
        R = rewards.view(-1, G)
        zv = A.zero_variance(R)
        adv = adv_fn(R, cfg.baseline, cfg.scale).reshape(-1)
        keep = torch.ones(len(problems), dtype=torch.bool)
        if cfg.zero_var == "filter":
            keep = (~zv).repeat_interleave(G)
        is_w = Lo.tis_weight(behaviour_logp, ro.sampler_logp, cfg.tis_cap) if cfg.tis_cap > 0 else None
        idx_all = torch.nonzero(keep).squeeze(-1)
        diags, ents, kls, losses_ = [], [], [], []
        gnorm = 0.0
        if len(idx_all) > 0:
            for _ in range(cfg.epochs):
                # minibatches are whole groups, so prompt-level aggregation sees complete groups
                groups = idx_all.view(-1, G)[:, 0] // G
                order = groups[torch.randperm(len(groups), generator=gen)]
                for chunk in order.chunk(max(1, min(cfg.minibatches, len(order)))):
                    idx = (chunk[:, None] * G + torch.arange(G)).reshape(-1)
                    logp, ent = token_logprobs(policy, ro.tokens[idx], ro.prompt_len, cfg.temperature,
                                               with_entropy=True)
                    m = mask[idx]
                    loss, d = hooks.get("policy_loss", Lo.policy_loss)(logp, old_logp[idx], adv[idx], m, aggregation=cfg.aggregation,
                                             group_size=G, eps_low=cfg.eps_low, eps_high=cfg.eps_high,
                                             ratio=cfg.ratio, norm_len=cfg.max_new,
                                             is_weight=is_w[idx] if is_w is not None else None)
                    if cfg.kl_place == "loss" and cfg.kl_beta > 0:
                        kt = K.kl_loss_terms(logp, ref_logp[idx], cfg.kl_kind)
                        loss = loss + cfg.kl_beta * Lo.aggregate(kt, m, cfg.aggregation, G, cfg.max_new)
                    if cfg.entropy_coef > 0:
                        loss = loss - cfg.entropy_coef * Lo.aggregate(ent, m, cfg.aggregation, G, cfg.max_new)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
                    opt.step()
                    diags.append(d)
                    ents.append(float(masked_mean(ent.detach(), ro.mask[idx])))
                    kls.append(float(masked_mean(K.estimators(logp.detach(), ref_logp[idx])["k3"], ro.mask[idx])))
                    losses_.append(float(loss.detach()))
        kl_exact = float(masked_mean(exact_token_kl(policy, ref, ro.tokens, ro.prompt_len, cfg.temperature), ro.mask))
        sampler.push(policy)
        step += 1
        L = ro.lengths
        row = {"split": "train", "step": step, "reward": float(rewards.mean()), "pass": float(correct.mean()),
               "zero_var_frac": float(zv.float().mean()),
               "zero_var_all_wrong": float((correct.view(-1, G).sum(1) == 0).float().mean()),
               "len": float(L.mean()),
               "len_correct": float(L[correct > 0].mean()) if (correct > 0).any() else None,
               "len_wrong": float(L[correct == 0].mean()) if (correct == 0).any() else None,
               "trunc": float((~ro.finished).float().mean()),
               "entropy": sum(ents) / len(ents) if ents else None, "kl_exact": kl_exact, "kl_k3": sum(kls) / len(kls) if kls else None,
               "clip_frac": sum(d["clip_frac"] for d in diags) / len(diags) if diags else None,
               "ratio_max": max(d["ratio_max"] for d in diags) if diags else None,
               "sampler_gap": float(masked_mean((behaviour_logp - ro.sampler_logp).abs(), ro.mask)),
               "loss": sum(losses_) / len(losses_) if losses_ else None, "grad_norm": gnorm,
               "seconds": round(time.perf_counter() - t0, 2)}
        if step % cfg.log_every == 0:
            log.log(**row)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(policy, held_problems, cfg.max_new, cfg.temperature,
                          torch.Generator().manual_seed(eval_gen_seed), cfg.verifier)
            log.log(split="eval", step=step, **ev)
            print(f"step {step:4d}  pass {row['pass']:.3f}  eval greedy {ev['greedy_acc']:.3f}  "
                  f"sampled {ev['sampled_acc']:.3f}  len {row['len']:.2f}  ent {row['entropy'] or 0:.3f}  "
                  f"{row['seconds']:.0f}s")
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(), "gen": gen.get_state(),
                        "torch_rng": torch.get_rng_state(), "snapshots": sampler.snapshots, "step": step,
                        "config": asdict(cfg)}, ck_path)
    log.close()
    save_policy(policy, run / "policy.pt", {"rl": asdict(cfg)})
    return {"run": str(run), "steps": step}


def main(argv=None):
    cfg = parse(argv)
    print(json.dumps(train(cfg)))


if __name__ == "__main__":
    main()
