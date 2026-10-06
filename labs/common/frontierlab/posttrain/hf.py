"""Main path: the same RL loop on a Hugging Face causal LM (the Stage D base model) and GSM8K.

    python -m frontierlab.posttrain.hf --run runs/m12/main/grpo-s0 --steps 200            # 1x H100 (not run in this build)
    python -m frontierlab.posttrain.hf --smoke --run runs/m12/hf-smoke --steps 2           # CPU: tiny random Qwen3

The pieces are the ones the toy loop uses (:mod:`advantages`, :mod:`kl`, :mod:`losses`); only the policy
interface changes:

* **Left-padded prompts, right-padded responses.** ``input_ids = [PAD.. prompt | response PAD..]`` with an
  ``attention_mask`` that is 0 on every PAD. Position ids are ``cumsum(attention_mask) - 1``, so a prompt's
  first real token is at position 0 whatever its padding (Qwen3 uses RoPE: positions matter).
* **Sampler log-probabilities** come from ``generate(..., output_logits=True)`` (the raw logits of each
  step, divided by the temperature), the **trainer's** from one teacher-forced forward
  (:func:`hf_token_logprobs`). Both cast logits to float32. Their gap is logged every step.
* **Precision.** Weights in float32 (master copy, AdamW state), forward and generation under bfloat16
  autocast on CUDA. A float32 reference copy is kept for the KL.

The default configuration is the Module 12 main-path run: Qwen3-1.7B-Base at the pinned revision,
GSM8K training questions with a 4-shot prompt from the training split, G = 8, 32 prompts per step,
512 new tokens, strict ``#### n`` reward, token-mean aggregation, no KL, evaluated on 500 test
questions every 50 steps. Not run in this build; part of the Module 12 pilot.
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

from frontierlab.metrics.jsonl import JsonlLogger
from frontierlab.posttrain import STAGE_D_BASE
from frontierlab.posttrain import advantages as A
from frontierlab.posttrain import kl as K
from frontierlab.posttrain import losses as Lo


@dataclass
class HFConfig:
    run: str = "runs/m12/main/default"
    model: str = STAGE_D_BASE["repo"]
    revision: str = STAGE_D_BASE["revision"]
    steps: int = 200
    prompts: int = 32
    group: int = 8
    max_new: int = 512
    temperature: float = 1.0
    lr: float = 1e-6
    minibatches: int = 4
    baseline: str = "mean"
    scale: str = "group"
    aggregation: str = "token_mean"
    eps_low: float = 0.2
    eps_high: float = 0.28
    kl_beta: float = 0.0
    kl_place: str = "none"
    kl_kind: str = "k2"
    overlong: str = "none"
    l_cache: int = 128
    zero_var: str = "keep"
    shots: int = 4
    eval_every: int = 50
    eval_n: int = 500
    seed: int = 0
    smoke: bool = False
    grad_clip: float = 1.0


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        att[i, T - len(s):] = 1
    return ids, att


def positions(att: torch.Tensor) -> torch.Tensor:
    return (att.cumsum(-1) - 1).clamp_min(0)


def hf_token_logprobs(model, ids: torch.Tensor, att: torch.Tensor, prompt_len: int, temperature: float = 1.0,
                      with_entropy: bool = False):
    """Teacher-forced log-probs of ``ids[:, prompt_len:]`` (B, R), float32 logits."""
    logits = model(input_ids=ids, attention_mask=att, position_ids=positions(att)).logits
    logits = logits[:, prompt_len - 1:-1].float() / temperature
    lp = torch.log_softmax(logits, -1)
    logp = lp.gather(-1, ids[:, prompt_len:, None]).squeeze(-1)
    if with_entropy:
        return logp, -(lp.exp() * lp).sum(-1)
    return logp


@torch.no_grad()
def hf_sample(model, prompt_ids, prompt_att, max_new: int, temperature: float, eos_id: int, pad_id: int):
    """Returns (response (B, R), mask (B, R), sampler_logp (B, R), finished (B,))."""
    out = model.generate(input_ids=prompt_ids, attention_mask=prompt_att, do_sample=True, temperature=temperature,
                         top_p=1.0, top_k=0, max_new_tokens=max_new, eos_token_id=eos_id, pad_token_id=pad_id,
                         output_logits=True, return_dict_in_generate=True)
    resp = out.sequences[:, prompt_ids.shape[1]:]
    logits = torch.stack(out.logits, 1).float() / temperature             # (B, R, V)
    slp = torch.log_softmax(logits, -1).gather(-1, resp[..., None]).squeeze(-1)
    is_eos = resp == eos_id
    before = torch.cumsum(is_eos.long(), -1) - is_eos.long()
    mask = (before == 0).float()
    return resp, mask, slp * mask, is_eos.any(-1)


class SmokeTokenizer:
    """A character tokenizer with the few methods the loop needs (CPU smoke test only)."""
    pad_token_id, eos_token_id = 0, 1

    def encode(self, s):
        return [2 + (ord(c) % 94) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + ((i - 2) % 94)) for i in ids if i > 1)


def load_policy(cfg: HFConfig, device):
    if cfg.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(cfg.seed)
        mc = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                         tie_word_embeddings=True)
        return Qwen3ForCausalLM(mc).to(device), SmokeTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision, dtype=torch.float32).to(device)
    return model, tok


def make_data(cfg: HFConfig):
    if cfg.smoke:
        rng = random.Random(0)
        mk = lambda: (lambda a, b: {"question": f"{a}+{b}", "solution": f"#### {a + b}", "answer": str(a + b)})(
            rng.randint(0, 9), rng.randint(0, 9))
        return [mk() for _ in range(64)], [mk() for _ in range(16)]
    from frontierlab.posttrain import gsm8k
    return gsm8k.load("train"), gsm8k.load("test")


def train(cfg: HFConfig) -> dict:
    from frontierlab.posttrain import gsm8k
    device = "cuda" if torch.cuda.is_available() and not cfg.smoke else "cpu"
    amp = torch.bfloat16 if device == "cuda" else None
    run = Path(cfg.run)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    policy, tok = load_policy(cfg, device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    train_set, test_set = make_data(cfg)
    shots = train_set[:cfg.shots]
    pool = train_set[cfg.shots:]
    log = JsonlLogger(run / "metrics.jsonl")
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    G = cfg.group
    t0 = time.perf_counter()

    def autocast():
        return torch.autocast("cuda", dtype=amp, enabled=amp is not None)

    def rollout(items, n_per):
        texts = [gsm8k.few_shot_prompt(it["question"], shots) for it in items for _ in range(n_per)]
        pids, patt = left_pad([tok.encode(t) for t in texts], tok.pad_token_id)
        pids, patt = pids.to(device), patt.to(device)
        with autocast():
            resp, mask, slp, fin = hf_sample(policy, pids, patt, cfg.max_new, cfg.temperature, tok.eos_token_id,
                                             tok.pad_token_id)
        ids = torch.cat([pids, resp], 1)
        att = torch.cat([patt, mask.long()], 1)
        decoded = [tok.decode(r[m.bool()].tolist(), skip_special_tokens=True) for r, m in zip(resp, mask)]
        return ids, att, pids.shape[1], mask, slp, fin, decoded

    for step in range(1, cfg.steps + 1):
        items = rng.sample(pool, cfg.prompts)
        ids, att, P, mask, slp, fin, decoded = rollout(items, G)
        answers = [it["answer"] for it in items for _ in range(G)]
        correct = torch.tensor([gsm8k.strict_reward(d, a) for d, a in zip(decoded, answers)], device=device)
        lenient = torch.tensor([gsm8k.last_number_reward(d, a) for d, a in zip(decoded, answers)], device=device)
        rewards = correct.clone()
        lengths = mask.sum(-1)
        if cfg.overlong == "soft":
            rewards = rewards + Lo.soft_overlong_penalty(lengths, cfg.max_new, cfg.l_cache).to(device)
        loss_mask = Lo.overlong_filter(mask, fin) if cfg.overlong == "filter" else mask
        with torch.no_grad(), autocast():
            old_logp = torch.cat([hf_token_logprobs(policy, ids[i:i + G], att[i:i + G], P, cfg.temperature)
                                  for i in range(0, len(ids), G)])
            ref_logp = torch.cat([hf_token_logprobs(ref, ids[i:i + G], att[i:i + G], P, cfg.temperature)
                                  for i in range(0, len(ids), G)])
        if cfg.kl_place == "reward" and cfg.kl_beta > 0:
            rewards = rewards - K.kl_penalty_reward(old_logp, ref_logp, mask, cfg.kl_beta, "k1")
        R = rewards.view(-1, G)
        zv = A.zero_variance(R)
        adv = A.group_advantages(R, cfg.baseline, cfg.scale).reshape(-1)
        keep_groups = torch.nonzero(~zv if cfg.zero_var == "filter" else torch.ones_like(zv)).squeeze(-1)
        diags, ents = [], []
        perm = keep_groups[torch.randperm(len(keep_groups))]
        for chunk in perm.chunk(max(1, min(cfg.minibatches, len(perm)))):
            idx = (chunk[:, None] * G + torch.arange(G, device=chunk.device)).reshape(-1)
            with autocast():
                logp, ent = hf_token_logprobs(policy, ids[idx], att[idx], P, cfg.temperature, with_entropy=True)
            m = loss_mask[idx]
            loss, d = Lo.policy_loss(logp, old_logp[idx], adv[idx], m, aggregation=cfg.aggregation, group_size=G,
                                     eps_low=cfg.eps_low, eps_high=cfg.eps_high, norm_len=cfg.max_new)
            if cfg.kl_place == "loss" and cfg.kl_beta > 0:
                loss = loss + cfg.kl_beta * Lo.aggregate(K.kl_loss_terms(logp, ref_logp[idx], cfg.kl_kind), m,
                                                         cfg.aggregation, G, cfg.max_new)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            diags.append(d)
            ents.append(float((ent.detach() * mask[idx]).sum() / mask[idx].sum().clamp_min(1)))
        if device == "cuda":
            torch.cuda.synchronize()
        row = {"split": "train", "step": step, "pass": float(correct.mean()), "lenient_pass": float(lenient.mean()),
               "reward": float(rewards.mean()), "zero_var_frac": float(zv.float().mean()),
               "len": float(lengths.mean()), "trunc": float((~fin).float().mean()),
               "entropy": sum(ents) / max(1, len(ents)),
               "clip_frac": sum(x["clip_frac"] for x in diags) / max(1, len(diags)),
               "sampler_gap": float(((old_logp - slp).abs() * mask).sum() / mask.sum().clamp_min(1)),
               "seconds": round(time.perf_counter() - t0, 1)}
        if device == "cuda":
            row["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log.log(**row)
        print(json.dumps(row))
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = test_set[:cfg.eval_n]
            accs = []
            for i in range(0, len(ev), 64):
                *_, dec = rollout(ev[i:i + 64], 1)
                accs += [gsm8k.strict_reward(d, it["answer"]) for d, it in zip(dec, ev[i:i + 64])]
            log.log(split="eval", step=step, test_pass1=sum(accs) / len(accs), n=len(accs))
    log.close()
    if not cfg.smoke:
        policy.save_pretrained(run / "policy")
    return {"run": str(run)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(HFConfig):
        if f.type in ("bool", bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true")
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    cfg = HFConfig(**vars(ap.parse_args(argv)))
    if cfg.smoke:
        cfg.prompts, cfg.group, cfg.max_new, cfg.eval_n, cfg.lr, cfg.shots = 4, 4, 6, 8, 1e-3, 1
    print(json.dumps(train(cfg)))


if __name__ == "__main__":
    main()
