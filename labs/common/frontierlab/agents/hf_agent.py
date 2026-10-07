"""Main path: the program world and the probe task on the Stage D base model (lessons 16.3-16.4, the project).

    python -m frontierlab.agents.hf_agent --task single --reward visible --run runs/m16/main/vis-s0      # 1x H100
    python -m frontierlab.agents.hf_agent --task single --reward random --run runs/m16/main/random-s0
    python -m frontierlab.agents.hf_agent --task probe --credit turn --info-bonus 0.3 --run runs/m16/main/probe-s0
    python -m frontierlab.agents.hf_agent --smoke --task probe --steps 2 --run runs/m16/hf-smoke             # CPU

Not run in this build except ``--smoke``; the GPU commands are part of the Module 16 pilot. The tasks, rewards,
monitors and credit rules are the CPU ones (:mod:`~frontierlab.agents.dsl`, :mod:`~frontierlab.agents.turns`,
:mod:`~frontierlab.agents.agentrl`); only the policy and the text format change.

Text format (the base model is not an instruction model, so prompts are few-shot completions):

* single-turn: ``"f(1)=4, f(2)=5, f(0)=3. f as a program: x+3\\n"`` for two worked examples, then the task with
  the program left blank. Programs use the same grammar as the CPU world (``x+3``, ``x*2+1``, ``T453``).
* probe: ``"Query f with ?<digit>, then answer with a program.\\n?2 =4\\n?1 =3\\nanswer x+2\\n"`` as the
  examples; then the episode: the model writes a line, the environment appends `` =<value>`` after a query.

Every turn is one ``generate`` call cut at the first newline, keeping the **sampled token ids** (re-encoding the
decoded text can give other tokens, and then the trainer scores tokens the sampler never produced). Observation
tokens are encoded by the tokenizer and inserted with mask 0. The update is GRPO-style (clipped ratio, token
mean, group advantages or lesson 16.3's turn advantages) in bf16 autocast with float32 logits.
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
from frontierlab.agents.agentrl import outcome_advantages, token_advantages, turn_advantages, turn_returns
from frontierlab.agents.turns import Episode, ProbeTask
from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.posttrain import STAGE_D_BASE
from frontierlab.posttrain import hf as H
from frontierlab.posttrain import kl as K
from frontierlab.posttrain import losses as Lo

T4_MODEL = {"repo": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"}


@dataclass
class HFAgentConfig:
    run: str = "runs/m16/main/default"
    model: str = STAGE_D_BASE["repo"]
    revision: str = STAGE_D_BASE["revision"]
    task: str = "single"                # single | probe
    reward: str = "visible"             # single-turn: a name in dsl.REWARDS; probe: "random" = control, else gold
    credit: str = "outcome"             # outcome | turn
    loss_on: str = "actions"            # actions | all
    info_bonus: float = 0.0
    max_queries: int = 2
    split: str = "none"                 # probe: "none" (every function) | "function" (3 held-out functions)
    steps: int = 150
    prompts: int = 32
    group: int = 8
    max_turn_tokens: int = 12
    temperature: float = 1.0
    lr: float = 1e-6
    minibatches: int = 4
    eval_every: int = 25
    eval_n: int = 66
    seed: int = 0
    smoke: bool = False
    grad_clip: float = 1.0


SHOTS_SINGLE = [dsl.DslTask(dsl.Func("mul", 2), (1, 3, 0)), dsl.DslTask(dsl.Func("add", 5), (2, 0, 1))]


def single_prompt(task: dsl.DslTask) -> str:
    def line(t, ans=""):
        return ", ".join(f"f({x})={y}" for x, y in t.pairs) + ". f as a program:" + (f" {ans}\n" if ans else "")
    return "".join(line(s, s.func.program) for s in SHOTS_SINGLE) + line(task)


PROBE_HEADER = ("Query f with ?<digit>, then answer with a program.\n?2 =4\n?1 =3\nanswer x+2\n\n"
                "Query f with ?<digit>, then answer with a program.\n?3 =9\nanswer x*3\n\n"
                "Query f with ?<digit>, then answer with a program.\n")


def _cut_at_newline(tok, ids: list[int], eos: int) -> tuple[list[int], bool]:
    """Sampled ids up to and including the first token whose text contains a newline (or EOS)."""
    out = []
    for t in ids:
        out.append(t)
        if t == eos or "\n" in tok.decode([t]):
            return out, True
    return out, False


@torch.no_grad()
def rollout(policy, tok, cfg: HFAgentConfig, items: list, device, autocast) -> dict:
    """One episode per item. Returns token ids, masks and per-episode bookkeeping (see turns.MultiTurnRollout)."""
    B = len(items)
    ctx = [tok.encode(single_prompt(it) if cfg.task == "single" else PROBE_HEADER) for it in items]
    P = [len(c) for c in ctx]
    seg_ids = [[] for _ in range(B)]
    seg_kind = [[] for _ in range(B)]          # 1 action, 0 observation (per token)
    seg_turn = [[] for _ in range(B)]
    seg_lp = [[] for _ in range(B)]
    eps = [Episode() for _ in range(B)]
    active = list(range(B))
    turns = 1 if cfg.task == "single" else cfg.max_queries + 1
    for turn in range(turns):
        if not active:
            break
        pids, patt = H.left_pad([ctx[i] + seg_ids[i] for i in active], tok.pad_token_id)
        pids, patt = pids.to(device), patt.to(device)
        with autocast():
            resp, mask, slp, _fin = H.hf_sample(policy, pids, patt, cfg.max_turn_tokens, cfg.temperature,
                                                tok.eos_token_id, tok.pad_token_id)
        still = []
        for row, i in enumerate(active):
            ids = resp[row][mask[row].bool()].tolist()
            lps = slp[row][mask[row].bool()].tolist()
            ids, ended = _cut_at_newline(tok, ids, tok.eos_token_id)
            lps = lps[:len(ids)]
            seg_ids[i] += ids
            seg_kind[i] += [1] * len(ids)
            seg_turn[i] += [turn] * len(ids)
            seg_lp[i] += lps
            text = tok.decode(ids, skip_special_tokens=True).strip()
            if cfg.task == "single":
                eps[i].final, eps[i].finished = text, ended
                continue
            if text.startswith("answer"):
                eps[i].final, eps[i].finished = text[len("answer"):].strip(), ended
                continue
            eps[i].actions.append(text)
            if len(eps[i].actions) > cfg.max_queries or not ended:
                eps[i].truncated = True
                continue
            env = items[i]
            obs_text = env.observe(text).replace(".", "")          # "=6" or "=-"
            ob = tok.encode(" " + obs_text + "\n")
            eps[i].observations.append(env.observe(text))
            eps[i].turn_rewards.append(env.turn_reward(text, eps[i]))
            seg_ids[i] += ob
            seg_kind[i] += [0] * len(ob)
            seg_turn[i] += [turn] * len(ob)
            seg_lp[i] += [0.0] * len(ob)
            still.append(i)
        active = still
    # right-aligned prompts are not needed: one left-padded batch of prompt + segments
    full = [c + s for c, s in zip(ctx, seg_ids)]
    Pmax = max(P)
    rows = [[tok.pad_token_id] * (Pmax - p) + f for p, f in zip(P, full)]
    T = max(len(r) for r in rows)
    ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    R = T - Pmax
    amask, omask, tidx, slp = (torch.zeros((B, R)), torch.zeros((B, R)), torch.full((B, R), -1, dtype=torch.long),
                               torch.zeros((B, R)))
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        att[i, Pmax - P[i]:len(r)] = 1
        n = len(seg_ids[i])
        amask[i, :n] = torch.tensor(seg_kind[i], dtype=torch.float32)
        omask[i, :n] = 1 - torch.tensor(seg_kind[i], dtype=torch.float32)
        tidx[i, :n] = torch.tensor(seg_turn[i])
        slp[i, :n] = torch.tensor(seg_lp[i])
    return {"ids": ids.to(device), "att": att.to(device), "P": Pmax, "amask": amask.to(device),
            "omask": omask.to(device), "tidx": tidx, "slp": slp.to(device), "episodes": eps}


def score(cfg: HFAgentConfig, items, eps, rng) -> tuple[list[float], list[list[float]], list[float]]:
    totals, per_turn, gold = [], [], []
    for it, ep in zip(items, eps):
        text = ep.final or ""
        if cfg.task == "single":
            totals.append(dsl.REWARDS[cfg.reward](text, ep.finished, it, rng))
            per_turn.append([])
            gold.append(dsl.r_gold(text, ep.finished, it))
        else:
            fin = it.final_reward(ep)
            if cfg.reward == "random":            # control: Bernoulli(0.5), independent of the episode
                totals.append(float(rng.random() < 0.5))
                per_turn.append([0.0] * len(ep.turn_rewards))
            else:
                totals.append(fin + sum(ep.turn_rewards))
                per_turn.append(list(ep.turn_rewards))
            gold.append(fin)
    return totals, per_turn, gold


def train(cfg: HFAgentConfig) -> dict:
    device = "cuda" if torch.cuda.is_available() and not cfg.smoke else "cpu"
    amp = torch.bfloat16 if device == "cuda" else None
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    policy, tok = H.load_policy(H.HFConfig(model=cfg.model, revision=cfg.revision, smoke=cfg.smoke, seed=cfg.seed),
                                device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    train_tasks, held = dsl.split_tasks(0)
    if cfg.task == "probe":
        from frontierlab.agents.agentrl import probe_functions
        tr_f, he_f = probe_functions(cfg.split, 0)
        train_tasks = [ProbeTask(f, cfg.max_queries, cfg.info_bonus) for f in tr_f]
        held = [ProbeTask(f, cfg.max_queries, cfg.info_bonus) for f in he_f] * (-(-cfg.eval_n // len(he_f)))
    held = held[:cfg.eval_n]
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log = JsonlLogger(run / "metrics.jsonl")
    G = cfg.group

    def autocast():
        return torch.autocast("cuda", dtype=amp, enabled=amp is not None)

    def kinds(eps):
        k = [dsl.kind_of(e.final) if e.final is not None else "other" for e in eps]
        return {f"frac_{x}": sum(y == x for y in k) / max(1, len(k)) for x in ("rule", "table", "other")}

    def evaluate(step):
        out = {"gold": [], "visible": [], "hidden": []}
        eps_all = []
        for i in range(0, len(held), 16):
            part = held[i:i + 16]
            b = rollout(policy, tok, cfg, part, device, autocast)
            for it, ep in zip(part, b["episodes"]):
                if cfg.task == "single":
                    for k in out:
                        out[k].append(dsl.REWARDS[k](ep.final or "", ep.finished, it))
                else:
                    g = it.final_reward(ep)
                    for k in out:
                        out[k].append(g)
            eps_all += b["episodes"]
        row = {"split": "eval", "step": step, **{f"{k}_pass": sum(v) / len(v) for k, v in out.items()}, **kinds(eps_all)}
        log.log(**row)
        print(json.dumps(row))

    evaluate(0)
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        items = [rng.choice(train_tasks) for _ in range(cfg.prompts)]
        items = [x for x in items for _ in range(G)]
        b = rollout(policy, tok, cfg, items, device, autocast)
        totals, per_turn, gold = score(cfg, items, b["episodes"], rng)
        tot = torch.tensor(totals)
        if cfg.credit == "outcome":
            adv = outcome_advantages(tot, G)[:, None].expand_as(b["amask"]).clone().to(device)
        else:
            finals = [t - sum(p) for t, p in zip(totals, per_turn)]
            adv = token_advantages(turn_advantages(turn_returns(per_turn, finals), G), b["tidx"]).to(device)
        mask = b["amask"] if cfg.loss_on == "actions" else (b["amask"] + b["omask"]).clamp(max=1.0)
        ids, att, P = b["ids"], b["att"], b["P"]
        with torch.no_grad(), autocast():
            old = torch.cat([H.hf_token_logprobs(policy, ids[i:i + G], att[i:i + G], P) for i in range(0, len(ids), G)])
            reflp = torch.cat([H.hf_token_logprobs(ref, ids[i:i + G], att[i:i + G], P) for i in range(0, len(ids), G)])
        perm = torch.randperm(cfg.prompts)
        diags = []
        for chunk in perm.chunk(cfg.minibatches):
            idx = (chunk[:, None] * G + torch.arange(G)).reshape(-1).to(device)
            with autocast():
                logp = H.hf_token_logprobs(policy, ids[idx], att[idx], P)
            loss, d = Lo.policy_loss(logp, old[idx], adv[idx], mask[idx], aggregation="token_mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            diags.append(d)
        k3 = K.estimators(old, reflp)["k3"]
        row = {"split": "train", "step": step, "reward": float(tot.mean()), "gold": sum(gold) / len(gold),
               **kinds(b["episodes"]), "kl_k3": float((k3 * b["amask"]).sum() / b["amask"].sum().clamp_min(1)),
               "clip_frac": sum(x["clip_frac"] for x in diags) / len(diags),
               "obs_tokens_in_loss": float((mask * b["omask"]).sum()),
               "sampler_gap": float(((old - b["slp"]).abs() * b["amask"]).sum() / b["amask"].sum().clamp_min(1)),
               "seconds": round(time.perf_counter() - t0, 1)}
        if device == "cuda":
            torch.cuda.synchronize()
            row["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log.log(**row)
        print(json.dumps(row))
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(step)
            if not cfg.smoke:
                policy.save_pretrained(run / "policy")
    log.close()
    return {"run": str(run)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(HFAgentConfig):
        if isinstance(f.default, bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true")
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    cfg = HFAgentConfig(**vars(ap.parse_args(argv)))
    if cfg.smoke:
        cfg.prompts, cfg.group, cfg.eval_n, cfg.minibatches = 2, 2, 4, 1
    print(json.dumps(train(cfg)))


if __name__ == "__main__":
    main()
