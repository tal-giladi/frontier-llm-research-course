"""Main path: objective-switchable reasoning RL on the Stage D base model (Modules 14.1-14.4 and the project).

    python -m frontierlab.rlscale.hf_rl --objective cispo --run runs/m14/main/cispo-s0 --seed 0      # 1x H100; not run in this build
    python -m frontierlab.rlscale.hf_rl --objective grpo --control random --run runs/m14/main/random-s0
    python -m frontierlab.rlscale.hf_rl --rollout vllm --staleness 1 ...                               # vLLM 0.30.0 rollouts
    python -m frontierlab.rlscale.hf_rl --smoke --run runs/m14/hf-smoke --steps 2                      # CPU, tiny random Qwen3

The pieces are Module 12's main path (:mod:`frontierlab.posttrain.hf`: left padding, sampling with the
sampler's log-probabilities, teacher-forced log-probabilities with float32 logits) and this package's
objectives. What this loop adds:

* ``--objective grpo|dapo|drgrpo|gspo|cispo`` with the objective's paired settings (advantage scale,
  zero-variance filtering) and parameters; the loop's own clip/aggregation flags do not exist here;
* ``--control random|format``: the training reward is replaced by a control reward, evaluation stays strict;
* ``--rollout hf|vllm`` and ``--staleness 0|1``: with vLLM, staleness 1 generates batch j+1 with weights
  v_j while the trainer updates on batch j (one-step off-policy); the behaviour log-probabilities of a stale
  batch are the engine's own numbers (the trainer no longer holds those weights), and for on-policy batches
  the trainer recomputes them and logs the trainer/sampler mismatch;
* evaluation every ``eval_every`` steps: GSM8K test pass@1 and pass@k from ``eval_samples`` samples per
  question (unbiased estimator), the per-question correct counts saved for lesson 14.4's analysis.

Logged per step (``metrics.jsonl``): pass, reward, zero-variance fraction, length, truncation, entropy,
clip fraction, max ratio, grad norm, k3 KL to the reference (sampled tokens), trainer/sampler mismatch
(mean |d|, k3, max ratio), lag, seconds, peak memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.evals.suite_v2.core import pass_at_k
from frontierlab.posttrain import STAGE_D_BASE
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import hf as H
from frontierlab.posttrain import kl as K
from frontierlab.rlscale.mismatch import mismatch_stats
from frontierlab.rlscale.objectives import OBJECTIVES



@dataclass
class HFRLConfig:
    run: str = "runs/m14/main/default"
    model: str = STAGE_D_BASE["repo"]
    revision: str = STAGE_D_BASE["revision"]
    objective: str = "grpo"
    control: str = ""                  # "" | random | format
    steps: int = 300
    prompts: int = 32
    group: int = 8
    max_new: int = 512
    temperature: float = 1.0
    lr: float = 1e-6
    minibatches: int = 4
    eps_high: float = -1.0             # <0: the objective's default
    rollout: str = "hf"                # hf | vllm
    staleness: int = 0                 # 0 | 1 (vllm only)
    batch_invariant: bool = False
    shots: int = 4
    eval_every: int = 50
    eval_n: int = 200
    eval_samples: int = 16
    seed: int = 0
    smoke: bool = False
    grad_clip: float = 1.0
    ckpt_every: int = 50


def _hfcfg(cfg: HFRLConfig):
    return H.HFConfig(model=cfg.model, revision=cfg.revision, smoke=cfg.smoke, seed=cfg.seed, shots=cfg.shots)


def format_reward(response: str) -> float:
    """1 if the response ends its solution in the ``#### <number>`` form, whatever the number."""
    from frontierlab.posttrain import gsm8k
    return float(gsm8k.strict_answer(response) is not None)


def train(cfg: HFRLConfig) -> dict:
    from frontierlab.posttrain import gsm8k
    device = "cuda" if torch.cuda.is_available() and not cfg.smoke else "cpu"
    amp = torch.bfloat16 if device == "cuda" else None
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    obj = OBJECTIVES[cfg.objective]
    params = dict(obj.params)
    if cfg.eps_high >= 0 and "eps_high" in params:
        params["eps_high"] = cfg.eps_high
    if "norm_len" in obj.loss.__code__.co_varnames:
        params["norm_len"] = cfg.max_new
    scale = obj.loop.get("scale", "group")
    zero_var = obj.loop.get("zero_var", "keep")
    policy, tok = H.load_policy(_hfcfg(cfg), device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    train_set, test_set = H.make_data(_hfcfg(cfg))
    shots, pool = train_set[:cfg.shots], train_set[cfg.shots:]
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log = JsonlLogger(run / "metrics.jsonl")
    G = cfg.group
    server = None
    if cfg.rollout == "vllm":
        from frontierlab.rlscale.vllm_rollout import VLLMServer
        server = VLLMServer(cfg.model, cfg.revision, batch_invariant=cfg.batch_invariant, seed=cfg.seed,
                            max_model_len=4096)
        server.attach_trainer(policy)
        server.sync_weights()
    workers = cf.ThreadPoolExecutor(max_workers=1)

    def autocast():
        return torch.autocast("cuda", dtype=amp, enabled=amp is not None)

    def rollout(items, n_per, version):
        """A batch for every item x n_per samples, tagged with the weights version that generated it."""
        pids_list = [tok.encode(gsm8k.few_shot_prompt(it["question"], shots)) for it in items]
        pids, patt = H.left_pad([p for p in pids_list for _ in range(n_per)], tok.pad_token_id)
        if server is None:
            pids, patt = pids.to(device), patt.to(device)
            with autocast():
                resp, mask, slp, fin = H.hf_sample(policy, pids, patt, cfg.max_new, cfg.temperature,
                                                   tok.eos_token_id, tok.pad_token_id)
        else:
            flat = [o for per in server.generate(pids_list, n_per, cfg.max_new, cfg.temperature, stop=[gsm8k.STOP])
                    for o in per]
            R = max(1, max(len(o[0]) for o in flat))
            resp = torch.full((len(flat), R), tok.pad_token_id, dtype=torch.long)
            slp, mask = torch.zeros((len(flat), R)), torch.zeros((len(flat), R))
            for i, (tids, lps, _done) in enumerate(flat):
                resp[i, :len(tids)] = torch.tensor(tids)
                slp[i, :len(tids)] = torch.tensor(lps)
                mask[i, :len(tids)] = 1
            fin = torch.tensor([o[2] for o in flat])
            resp, slp, mask, fin, pids, patt = (x.to(device) for x in (resp, slp, mask, fin, pids, patt))
        decoded = [tok.decode(r[m.bool()].tolist(), skip_special_tokens=True) for r, m in zip(resp, mask)]
        return {"ids": torch.cat([pids, resp], 1), "att": torch.cat([patt, mask.long()], 1), "P": pids.shape[1],
                "mask": mask, "slp": slp, "fin": fin, "decoded": decoded, "version": version,
                "answers": [it["answer"] for it in items for _ in range(n_per)]}

    def evaluate(step):
        ev, n = test_set[:cfg.eval_n], cfg.eval_samples
        counts = []
        for i in range(0, len(ev), 16):
            part = ev[i:i + 16]
            b = rollout(part, n, step)
            ok = [gsm8k.strict_reward(d, a) for d, a in zip(b["decoded"], b["answers"])]
            counts += [int(sum(ok[j * n:(j + 1) * n])) for j in range(len(part))]
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= n]
        row = {"split": "eval", "step": step, "n_samples": n, "counts": counts,
               **{f"pass@{k}": sum(pass_at_k(n, c, k) for c in counts) / len(counts) for k in ks}}
        log.log(**row)
        print(json.dumps({k: v for k, v in row.items() if k != "counts"}))

    def update(b, lag):
        ids, att, P, mask = b["ids"], b["att"], b["P"], b["mask"]
        correct = torch.tensor([gsm8k.strict_reward(d, a) for d, a in zip(b["decoded"], b["answers"])], device=device)
        if cfg.control == "random":
            rewards = (torch.rand(len(correct)) < 0.5).float().to(device)
        elif cfg.control == "format":
            rewards = torch.tensor([format_reward(d) for d in b["decoded"]], device=device)
        else:
            rewards = correct.clone()
        with torch.no_grad(), autocast():
            trainer_old = torch.cat([H.hf_token_logprobs(policy, ids[i:i + G], att[i:i + G], P, cfg.temperature)
                                     for i in range(0, len(ids), G)])
            ref_logp = torch.cat([H.hf_token_logprobs(ref, ids[i:i + G], att[i:i + G], P, cfg.temperature)
                                  for i in range(0, len(ids), G)])
        mm = mismatch_stats(trainer_old, b["slp"], mask) if lag == 0 else {}
        old_logp = trainer_old if lag == 0 else b["slp"]          # a stale batch: the engine's behaviour numbers
        Rw = rewards.view(-1, G)
        zv = A.zero_variance(Rw)
        adv = A.group_advantages(Rw, "mean", scale).reshape(-1)
        keep = torch.nonzero(~zv if zero_var == "filter" else torch.ones_like(zv)).squeeze(-1)
        perm = keep[torch.randperm(len(keep))]
        diags, ents, gn = [], [], 0.0
        for chunk in perm.chunk(max(1, min(cfg.minibatches, len(perm)))):
            idx = (chunk[:, None] * G + torch.arange(G, device=chunk.device)).reshape(-1)
            with autocast():
                logp, ent = H.hf_token_logprobs(policy, ids[idx], att[idx], P, cfg.temperature, with_entropy=True)
            loss, d = obj.loss(logp, old_logp[idx], adv[idx], mask[idx], **params)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
            opt.step()
            diags.append(d)
            ents.append(float((ent.detach() * mask[idx]).sum() / mask[idx].sum().clamp_min(1)))
        k3 = K.estimators(trainer_old, ref_logp)["k3"]
        return {"split": "train", "pass": float(correct.mean()), "reward": float(rewards.mean()),
                "zero_var_frac": float(zv.float().mean()), "len": float(mask.sum(-1).mean()),
                "trunc": float((~b["fin"]).float().mean()), "entropy": sum(ents) / max(1, len(ents)),
                "clip_frac": sum(x["clip_frac"] for x in diags) / max(1, len(diags)),
                "ratio_max": max((x["ratio_max"] for x in diags), default=1.0), "grad_norm": gn,
                "kl_k3": float((k3 * mask).sum() / mask.sum().clamp_min(1)),
                **{f"mismatch_{k}": v for k, v in mm.items()}}

    evaluate(0)
    t0 = time.perf_counter()
    nxt = workers.submit(rollout, rng.sample(pool, cfg.prompts), G, 0) if server is not None else None
    for step in range(1, cfg.steps + 1):          # the trainer holds weights version step - 1
        if server is None:
            b = rollout(rng.sample(pool, cfg.prompts), G, step - 1)
        else:
            b = nxt.result()
            if cfg.staleness >= 1:                # batch j+1 is generated with v_{j-1} while we update on batch j
                nxt = workers.submit(rollout, rng.sample(pool, cfg.prompts), G, step - 1)
        lag = (step - 1) - b["version"]
        row = update(b, lag)
        if server is not None:
            if cfg.staleness >= 1:
                nxt.result()                      # finish it on the old weights: lag is exactly 1, never more
            server.sync_weights()
            if cfg.staleness == 0:
                nxt = workers.submit(rollout, rng.sample(pool, cfg.prompts), G, step)
        if device == "cuda":
            torch.cuda.synchronize()
            row["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row.update(step=step, lag=lag, seconds=round(time.perf_counter() - t0, 1))
        log.log(**row)
        print(json.dumps(row))
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(step)
        if not cfg.smoke and (step % cfg.ckpt_every == 0 or step == cfg.steps):
            policy.save_pretrained(run / "policy")
    log.close()
    workers.shutdown(wait=False)
    if server is not None:
        server.close()
    return {"run": str(run)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(HFRLConfig):
        if f.type in ("bool", bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true")
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    cfg = HFRLConfig(**vars(ap.parse_args(argv)))
    if cfg.smoke:
        cfg.prompts, cfg.group, cfg.max_new, cfg.eval_n, cfg.eval_samples, cfg.lr, cfg.shots = 4, 4, 6, 4, 4, 1e-3, 1
        cfg.minibatches = 2
    print(json.dumps(train(cfg)))


if __name__ == "__main__":
    main()
