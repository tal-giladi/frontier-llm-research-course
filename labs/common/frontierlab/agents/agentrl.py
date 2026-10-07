"""Agent RL on the toy program world: warm start, multi-turn GRPO, monitoring, traces (lessons 16.3-16.4).

    python -m frontierlab.agents.agentrl sft --task single --out runs/m16/sft-single
    python -m frontierlab.agents.agentrl rl --init runs/m16/sft-single/policy.pt --run runs/m16/vis-s0 --reward visible

The Module 12 loop (:mod:`frontierlab.posttrain.rl`) is written for arithmetic prompts and single-turn
responses, so this module has its own short loop. It reuses everything else unchanged: the policy
architecture and checkpoint format (:mod:`~frontierlab.posttrain.sft`), teacher-forced log-probabilities
(:func:`~frontierlab.posttrain.policy.token_logprobs`), group advantages
(:mod:`~frontierlab.posttrain.advantages`) and the clipped loss with its aggregation
(:func:`~frontierlab.posttrain.losses.policy_loss`). Lesson 16.3's three choices are config fields:

* ``credit`` — ``"outcome"``: one advantage per episode (the group-normalised total reward), given to every
  action token; ``"turn"``: each turn's tokens get a group-normalised **return-to-go** from that turn
  (:func:`turn_advantages`), so a shaped per-turn reward reaches only the turns at or before it.
* ``loss_on`` — ``"actions"`` (correct) or ``"all"`` (the observation-masking bug: inserted observation tokens
  also enter the loss with the episode's advantage).
* ``max_queries`` — the turn limit of the probe task.

Lesson 16.4's choices: ``reward`` (any name in :data:`~frontierlab.agents.dsl.REWARDS`), ``marker`` (train with
the inoculation marker in every prompt; evaluation never has it).

Every step logs one JSON line to ``<run>/metrics.jsonl``: the training reward, the **gold** pass rate of the same
samples (never used for training), the output-kind fractions (rule / table / other), length, truncation,
entropy, KL to the start (k3), clip fraction, and for probe tasks the queries per episode. Every
``eval_every`` steps it evaluates on held-out tasks (fixed sampling seed): gold, visible and hidden pass rates,
the kind fractions, length. ``traces`` > 0 also writes that many evaluation samples per evaluation to
``<run>/traces.jsonl`` (task, prompt, response, kind, the score under every deterministic reward).
Checkpoints every ``ckpt_every`` steps; rerunning the same command resumes the same run.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from frontierlab.agents import dsl
from frontierlab.agents.turns import ProbeTask, SingleTurn, sample_multiturn
from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.model import LM
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import kl as K
from frontierlab.posttrain import losses as Lo
from frontierlab.posttrain.policy import masked_mean, token_logprobs
from frontierlab.posttrain.sft import load_policy, policy_config, save_policy, sft_loss

DETERMINISTIC = ("format", "length", "visible", "hidden", "property", "gold")


def dsl_policy_config(**kw):
    return policy_config(vocab_size=dsl.VOCAB, **kw)


# --------------------------------------------------------------------------- warm start

def sft_targets(task: dsl.DslTask, rng: random.Random, p_rule: float, p_table: float) -> str:
    u = rng.random()
    if u < p_rule:
        return task.func.program
    if u < p_rule + p_table:
        return dsl.table_program(task)
    return rng.choice([f for f in dsl.ALL_FUNCS if f != task.func]).program


def probe_demo(func: dsl.Func, rng: random.Random, p_correct: float, max_queries: int) -> tuple[list[str], list[str]]:
    """A scripted demonstration: 0..max_queries random queries, then a program consistent with what was seen
    (the right one with probability ``p_correct`` if several are consistent). Returns (segments, kinds) where
    kinds are 'a' (action, in the loss) or 'o' (observation, not in the loss)."""
    from frontierlab.agents.turns import Episode
    env = ProbeTask(func, max_queries)
    ep = Episode()
    segs, kinds = [], []
    for _ in range(rng.randint(0, max_queries)):
        a = f"?{rng.randint(0, 9)}"
        o = env.observe(a)
        ep.actions.append(a)
        ep.observations.append(o)
        segs += [a + ".", o]
        kinds += ["a", "o"]
    cands = env.candidates(ep)
    ans = func if (rng.random() < p_correct or len(cands) == 1) else rng.choice(cands)
    segs.append(ans.program)
    kinds.append("a")
    return segs, kinds


def encode_probe_sft(demos) -> tuple[torch.Tensor, torch.Tensor]:
    from frontierlab.posttrain.tokenizer import BOS, EOS, PAD
    rows, masks = [], []
    for segs, kinds in demos:
        r, m = [BOS] + dsl.TOKD.encode(":"), [0, 0]
        for s, k in zip(segs, kinds):
            ids = dsl.TOKD.encode(s)
            r += ids
            m += [1 if k == "a" else 0] * len(ids)
        r.append(EOS)
        m.append(1)
        rows.append(r)
        masks.append(m)
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        ids[i, :len(r)] = torch.tensor(r)
        mask[i, :len(m)] = torch.tensor(m)
    return ids, mask


def train_sft(out: str | Path, task: str = "single", steps: int = 600, batch: int = 64, lr: float = 3e-3,
              seed: int = 0, p_rule: float = 0.55, p_table: float = 0.2, p_correct: float = 0.5,
              max_queries: int = 2, split_seed: int = 0, probe_split: str = "none") -> dict:
    """Warm start on scripted demonstrations of the training tasks only. The single-turn mix is right rules
    (``p_rule``), tables (``p_table``) and rules of another function (the rest): the warm-started policy knows
    both kinds of answer, as a code model knows both general solutions and special-casing, and is often wrong."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out = Path(out)
    model = LM(dsl_policy_config())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    train_tasks, _ = dsl.split_tasks(split_seed)
    probe_funcs = probe_functions(probe_split, split_seed)[0]
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 50)
        if task == "single":
            ts = [rng.choice(train_tasks) for _ in range(batch)]
            ids, mask = dsl.encode_sft(ts, [sft_targets(t, rng, p_rule, p_table) for t in ts])
        else:
            ids, mask = encode_probe_sft([probe_demo(rng.choice(probe_funcs), rng, p_correct, max_queries)
                                          for _ in range(batch)])
        loss = sft_loss(model, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    meta = {"task": task, "probe_split": probe_split, "steps": steps, "batch": batch, "lr": lr, "seed": seed, "p_rule": p_rule,
            "p_table": p_table, "p_correct": p_correct, "max_queries": max_queries,
            "seconds": round(time.perf_counter() - t0, 1), "final_loss": float(loss.detach())}
    save_policy(model, out / "policy.pt", meta)
    (out / "sft.json").write_text(json.dumps(meta, indent=2))
    return meta


def probe_functions(split: str = "none", split_seed: int = 0) -> tuple[list[dsl.Func], list[dsl.Func]]:
    """(training functions, held-out functions) for the probe task. ``"none"``: every function in both (the probe
    task of lesson 16.3); ``"function"``: the function-level split of :func:`frontierlab.agents.dsl.split_tasks`."""
    if split == "none":
        return list(dsl.ALL_FUNCS), list(dsl.ALL_FUNCS)
    tr, he = dsl.split_tasks(split_seed, by="function")
    names_tr, names_he = {t.func.name for t in tr}, {t.func.name for t in he}
    return [f for f in dsl.ALL_FUNCS if f.name in names_tr], [f for f in dsl.ALL_FUNCS if f.name in names_he]


def ensure_sft(out: str | Path, **kw) -> Path:
    path = Path(out) / "policy.pt"
    if not path.exists():
        print(f"training the warm start -> {path}")
        train_sft(out, **kw)
    return path


# --------------------------------------------------------------------------- credit over turns

def outcome_advantages(total: torch.Tensor, group: int) -> torch.Tensor:
    """(B,) episode rewards -> (B,) group-normalised advantages (GRPO)."""
    return A.group_advantages(total.view(-1, group), "mean", "group").reshape(-1)


def turn_returns(turn_rewards: list[list[float]], final: list[float], gamma: float = 1.0) -> list[list[float]]:
    """Return-to-go per turn: G_t = r_t + gamma r_{t+1} + ... with the final reward as the last turn's reward."""
    out = []
    for rs, f in zip(turn_rewards, final):
        r = list(rs) + [f]
        g, acc = [], 0.0
        for x in reversed(r):
            acc = x + gamma * acc
            g.append(acc)
        out.append(list(reversed(g)))
    return out


def turn_advantages(returns: list[list[float]], group: int, eps: float = 1e-6) -> list[list[float]]:
    """Group-normalise returns per turn index: A_{i,t} = (G_{i,t} - mean_t) / (std of the group's G_{i,0} + eps),
    where mean_t is the mean over the group's episodes that reached turn t. Dividing every turn by the same
    episode-level std keeps the turns of one episode on one scale."""
    out = []
    for g0 in range(0, len(returns), group):
        grp = returns[g0:g0 + group]
        firsts = torch.tensor([r[0] for r in grp], dtype=torch.float64)
        sd = float(firsts.std()) if len(grp) > 1 else 0.0
        T = max(len(r) for r in grp)
        means = []
        for t in range(T):
            vals = [r[t] for r in grp if len(r) > t]
            means.append(sum(vals) / len(vals))
        for r in grp:
            out.append([(r[t] - means[t]) / (sd + eps) for t in range(len(r))])
    return out


def token_advantages(adv_turn: list[list[float]], turn_index: torch.Tensor) -> torch.Tensor:
    """Spread per-turn advantages over tokens: (B, R) float32, each token gets its turn's advantage.
    The final answer is the last turn (index len - 1); tokens with turn index -1 (PAD) get 0."""
    B, R = turn_index.shape
    out = torch.zeros((B, R))
    for i in range(B):
        for t, a in enumerate(adv_turn[i]):
            out[i][turn_index[i] == t] = a
    return out


# --------------------------------------------------------------------------- the loop

@dataclass
class AgentRLConfig:
    init: str = "runs/m16/sft-single/policy.pt"
    run: str = "runs/m16/rl/default"
    task: str = "single"                   # single | probe
    reward: str = "visible"                # single-turn: a name in dsl.REWARDS; probe: "random" = control, else gold + bonus
    marker: bool = False                   # inoculation marker on every training prompt (single-turn)
    credit: str = "outcome"                # outcome | turn
    loss_on: str = "actions"               # actions | all
    gamma: float = 1.0
    info_bonus: float = 0.0
    query_cost: float = 0.0
    max_queries: int = 2
    steps: int = 80
    prompts: int = 16
    group: int = 8
    max_tokens: int = 12
    temperature: float = 1.0
    lr: float = 3e-4
    epochs: int = 1
    minibatches: int = 2
    aggregation: str = "token_mean"
    eps_low: float = 0.2
    eps_high: float = 0.2
    grad_clip: float = 1.0
    split: str = "instance"
    split_seed: int = 0
    seed: int = 0
    eval_every: int = 20
    eval_n: int = 66
    eval_samples: int = 4
    traces: int = 0
    ckpt_every: int = 20


def parse(argv=None) -> AgentRLConfig:
    ap = argparse.ArgumentParser()
    for f in fields(AgentRLConfig):
        if isinstance(f.default, bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true", default=f.default)
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    return AgentRLConfig(**vars(ap.parse_args(argv)))


def make_envs(cfg: AgentRLConfig, tasks: list, marker: bool):
    if cfg.task == "single":
        return [SingleTurn(t.with_marker(marker)) for t in tasks]
    return [ProbeTask(t.func, cfg.max_queries, cfg.info_bonus) for t in tasks]


def final_text(ro, i: int) -> str:
    """The answer of episode i: the text before EOS, or (truncated) every action character after the last turn."""
    ep = ro.episodes[i]
    if ep.final is not None:
        return ep.final
    toks = [t for t, m in zip(ro.response[i].tolist(), ro.action_mask[i].tolist()) if m > 0 and t > 2]
    text = dsl.TOKD.decode(toks)
    return text.split(".")[-1] if ep.actions else text


def episode_rewards(cfg: AgentRLConfig, envs, ro, rng: random.Random) -> tuple[list[float], list[list[float]], list[float]]:
    """(total reward per episode, per-turn shaped rewards, gold pass per episode)."""
    totals, per_turn, gold = [], [], []
    for env, ep in zip(envs, ro.episodes):
        if cfg.task == "single":
            text = final_text(ro, len(totals))
            r = dsl.REWARDS[cfg.reward](text, ep.finished, env.task, rng)
            totals.append(r)
            per_turn.append([])
            gold.append(dsl.r_gold(text, ep.finished, env.task))
        else:
            fin = env.final_reward(ep)
            shaped = [x - cfg.query_cost for x in ep.turn_rewards]
            if cfg.reward == "random":                # the control: Bernoulli(0.5), independent of the episode
                totals.append(float(rng.random() < 0.5))
                per_turn.append([0.0] * len(shaped))
            else:
                totals.append(fin + sum(shaped))
                per_turn.append(shaped)
            gold.append(fin)
    return totals, per_turn, gold


def kind_fractions(ro) -> dict:
    kinds = [dsl.kind_of(ep.final) if ep.final is not None else "other" for ep in ro.episodes]
    n = max(1, len(kinds))
    return {k: sum(x == k for x in kinds) / n for k in ("rule", "table", "other")}


def probe_stats(ro) -> dict:
    """Queries per episode; the share of actions the environment could not parse; the share of actions in which
    the policy wrote ``=`` itself (an observation-shaped token: imitating the environment)."""
    acts = [a for ep in ro.episodes for a in ep.actions]
    n = max(1, len(acts))
    return {"queries": len(acts) / len(ro.episodes),
            "bad_actions": sum(not (len(a) == 2 and a[0] == "?" and a[1].isdigit()) for a in acts) / n,
            "self_obs": (sum("=" in a for a in acts) + sum("=" in (ep.final or "") for ep in ro.episodes))
            / (n + len(ro.episodes)),
            "turn_limit": sum(ep.truncated for ep in ro.episodes) / len(ro.episodes)}


@torch.no_grad()
def evaluate(policy, cfg: AgentRLConfig, held: list, gen_seed: int, n_traces: int = 0, step: int = 0) -> tuple[dict, list]:
    tasks = [t for t in held for _ in range(cfg.eval_samples)]
    envs = make_envs(cfg, tasks, marker=False)
    ro = sample_multiturn(policy, envs, cfg.max_tokens, cfg.temperature, torch.Generator().manual_seed(gen_seed))
    res = {k: [] for k in DETERMINISTIC}
    traces = []
    for i, (env, ep) in enumerate(zip(envs, ro.episodes)):
        text = final_text(ro, i)
        if cfg.task == "single":
            scores = {k: dsl.REWARDS[k](text, ep.finished, env.task) for k in DETERMINISTIC}
        else:
            g = env.final_reward(ep)
            scores = {k: g for k in DETERMINISTIC}
        for k, v in scores.items():
            res[k].append(v)
        if i < n_traces:
            traces.append({"step": step, "task": getattr(env, "task", env).id if cfg.task == "single" else env.id,
                           "prompt": env.prompt, "actions": ep.actions, "observations": ep.observations,
                           "response": text, "finished": ep.finished, "kind": dsl.kind_of(text), "scores": scores})
    out = {f"{k}_pass": sum(v) / len(v) for k, v in res.items()}
    out.update({f"frac_{k}": v for k, v in kind_fractions(ro).items()})
    out["len"] = float(sum(len(final_text(ro, i)) for i in range(len(ro.episodes))) / len(ro.episodes))
    out["trunc"] = float((~ro.finished).float().mean())
    if cfg.task == "probe":
        out.update(probe_stats(ro))
    return out, traces


def train(cfg: AgentRLConfig) -> dict:
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    policy = load_policy(cfg.init)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_tasks, held = dsl.split_tasks(cfg.split_seed, by=cfg.split)
    if cfg.task == "probe":           # one task per function; held-out functions repeated to about eval_n episodes
        tr_f, he_f = probe_functions("function" if cfg.split == "function" else "none", cfg.split_seed)
        train_tasks = [dsl.DslTask(f, (0, 1, 2)) for f in tr_f]
        held = [dsl.DslTask(f, (0, 1, 2)) for f in he_f] * (-(-cfg.eval_n // len(he_f)))
    held = held[:cfg.eval_n]
    step = 0
    ck = run / "checkpoint.pt"
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        policy.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        gen.set_state(c["gen"])
        torch.set_rng_state(c["torch_rng"])
        rng.setstate(c["py_rng"])
        step = c["step"]
        print(f"resumed from step {step}")
    else:
        from frontierlab.runcard import write_run_card
        write_run_card(run, question="Module 16 agent RL", config=asdict(cfg), args=asdict(cfg),
                       budget={"steps": cfg.steps, "episodes": cfg.steps * cfg.prompts * cfg.group})
        if step == 0 and (run / "traces.jsonl").exists():
            (run / "traces.jsonl").unlink()
    log = JsonlLogger(run / "metrics.jsonl")
    G = cfg.group
    t0 = time.perf_counter()
    if step == 0:
        ev, tr = evaluate(policy, cfg, held, 10_000 + cfg.seed, cfg.traces, 0)
        log.log(split="eval", step=0, **ev)
        _write_traces(run, tr)
    while step < cfg.steps:
        tasks = [rng.choice(train_tasks) for _ in range(cfg.prompts)]
        tasks = [t for t in tasks for _ in range(G)]
        envs = make_envs(cfg, tasks, cfg.marker)
        ro = sample_multiturn(policy, envs, cfg.max_tokens, cfg.temperature, gen)
        totals, per_turn, gold = episode_rewards(cfg, envs, ro, rng)
        tot = torch.tensor(totals, dtype=torch.float32)
        if cfg.credit == "outcome":
            adv = outcome_advantages(tot, G)[:, None].expand_as(ro.action_mask).clone()
        else:
            finals = [t - sum(p) for t, p in zip(totals, per_turn)]
            adv = token_advantages(turn_advantages(turn_returns(per_turn, finals, cfg.gamma), G), ro.turn_index)
        mask = ro.action_mask if cfg.loss_on == "actions" else (ro.action_mask + ro.obs_mask).clamp(max=1.0)
        with torch.no_grad():
            old_logp = token_logprobs(policy, ro.tokens, ro.prompt_len, cfg.temperature)
            ref_logp = token_logprobs(ref, ro.tokens, ro.prompt_len, cfg.temperature)
        diags, ents, kls, gnorm = [], [], [], 0.0
        for _ in range(cfg.epochs):
            order = torch.randperm(cfg.prompts, generator=gen)
            for chunk in order.chunk(cfg.minibatches):
                idx = (chunk[:, None] * G + torch.arange(G)).reshape(-1)
                logp, ent = token_logprobs(policy, ro.tokens[idx], ro.prompt_len, cfg.temperature, with_entropy=True)
                loss, d = Lo.policy_loss(logp, old_logp[idx], adv[idx], mask[idx], aggregation=cfg.aggregation,
                                         group_size=G, eps_low=cfg.eps_low, eps_high=cfg.eps_high,
                                         norm_len=cfg.max_tokens)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
                opt.step()
                diags.append(d)
                ents.append(float(masked_mean(ent.detach(), ro.action_mask[idx])))
                kls.append(float(masked_mean(K.estimators(logp.detach(), ref_logp[idx])["k3"], ro.action_mask[idx])))
        step += 1
        row = {"split": "train", "step": step, "reward": float(tot.mean()), "gold": sum(gold) / len(gold),
               "zero_var_frac": float(A.zero_variance(tot.view(-1, G)).float().mean()),
               **{f"frac_{k}": v for k, v in kind_fractions(ro).items()},
               "len": float(sum(len(final_text(ro, i)) for i in range(len(ro.episodes))) / len(ro.episodes)),
               "trunc": float((~ro.finished).float().mean()),
               "entropy": sum(ents) / len(ents), "kl_k3": sum(kls) / len(kls),
               "clip_frac": sum(d["clip_frac"] for d in diags) / len(diags), "grad_norm": gnorm,
               "obs_tokens_in_loss": float((mask * ro.obs_mask).sum()),
               "seconds": round(time.perf_counter() - t0, 2)}
        if cfg.task == "probe":
            row.update(probe_stats(ro))
        log.log(**row)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev, tr = evaluate(policy, cfg, held, 10_000 + cfg.seed, cfg.traces, step)
            log.log(split="eval", step=step, **ev)
            _write_traces(run, tr)
            print(f"step {step:4d} reward {row['reward']:.3f} gold(train) {row['gold']:.3f}  eval gold "
                  f"{ev['gold_pass']:.3f} visible {ev['visible_pass']:.3f} table {ev['frac_table']:.2f} "
                  f"other {ev['frac_other']:.2f} len {ev['len']:.1f}  {row['seconds']:.0f}s")
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(), "gen": gen.get_state(),
                        "torch_rng": torch.get_rng_state(), "py_rng": rng.getstate(), "step": step,
                        "config": asdict(cfg)}, ck)
    log.close()
    save_policy(policy, run / "policy.pt", {"agentrl": asdict(cfg)})
    return {"run": str(run), "steps": step}


def _write_traces(run: Path, traces: list):
    if not traces:
        return
    with open(run / "traces.jsonl", "a", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")


def run_arms(base: AgentRLConfig, arms: dict[str, dict], seeds: list[int], root: str | Path) -> dict:
    """Train every arm (config overrides) at every seed; finished runs are skipped. Returns run paths."""
    from dataclasses import replace
    root = Path(root)
    out = {}
    for name, over in arms.items():
        out[name] = []
        for s in seeds:
            cfg = replace(base, **over, seed=s, run=str(root / f"{name}-s{s}"))
            if not (Path(cfg.run) / "policy.pt").exists():
                print(f"== arm {name} seed {s}")
                train(cfg)
            out[name].append(cfg.run)
    return out


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sft":
        ap = argparse.ArgumentParser()
        ap.add_argument("--task", default="single")
        ap.add_argument("--out", required=True)
        ap.add_argument("--steps", type=int, default=600)
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args(argv[1:])
        print(json.dumps(train_sft(a.out, a.task, a.steps, seed=a.seed), indent=2))
    elif argv and argv[0] == "rl":
        print(json.dumps(train(parse(argv[1:]))))
    else:
        raise SystemExit("usage: python -m frontierlab.agents.agentrl {sft|rl} ...")


if __name__ == "__main__":
    main()


__all__ = ["AgentRLConfig", "train", "train_sft", "ensure_sft", "probe_functions", "run_arms", "evaluate", "outcome_advantages",
           "turn_returns", "turn_advantages", "token_advantages", "dsl_policy_config", "DETERMINISTIC"]
