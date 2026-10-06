"""Supervised warm start for the toy policy: next-token loss on response tokens only.

    python -m frontierlab.posttrain.sft --out runs/m12/sft                      # one to three minutes on a laptop

The toy policy is Baseline-0's architecture at a tiny size (:func:`policy_config`, 308,400 parameters)
with the 35-token character vocabulary. It is trained on the arithmetic task of
:mod:`frontierlab.posttrain.tasks` — both operations, all four instruction tags, training triples only —
and stopped early on purpose (at about 30% greedy accuracy on plain addition): RL needs a policy that is
sometimes right, not one that is always right.

Loss masking (lesson 12.3): the loss is averaged over target and EOS positions only. Prompt tokens are
given, not predicted, and PAD positions after EOS are not part of any response.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from frontierlab.model import LM, ModelConfig
from frontierlab.posttrain.tasks import TAGS, encode_prompts, make_problems, score, split_problems
from frontierlab.posttrain.tokenizer import TOK


def policy_config(**kw) -> ModelConfig:
    """Tiny Baseline-0-layout policy for the character vocabulary."""
    base = dict(vocab_size=TOK.vocab_size, hidden_size=96, num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2, head_dim=24, intermediate_size=256, max_position_embeddings=64)
    base.update(kw)
    return ModelConfig(**base)


def sft_loss(model, ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over positions whose *target* token has ``loss_mask == 1``."""
    logits = model(ids).logits[:, :-1].float()
    tgt, m = ids[:, 1:], loss_mask[:, 1:].float()
    tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view_as(m)
    return (tok * m).sum() / m.sum()


def save_policy(model, path: str | Path, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "model": model.state_dict(), "meta": meta or {}}, path)


def load_policy(path: str | Path, device="cpu") -> LM:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = LM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.to(device)


@torch.no_grad()
def greedy_accuracy(model, problems, max_new: int = 8, verifier: str = "strict") -> float:
    """Accuracy of greedy decoding (temperature -> 0) on ``problems``."""
    from frontierlab.posttrain.tokenizer import EOS, PAD
    was = model.training
    model.eval()
    prompts = encode_prompts(problems)
    cache = model.new_cache()
    logits = model(prompts, cache=cache).logits[:, -1]
    out = torch.full((len(problems), max_new), PAD, dtype=torch.long)
    done = torch.zeros(len(problems), dtype=torch.bool)
    for t in range(max_new):
        nxt = torch.where(done, torch.full((len(problems),), PAD), logits.argmax(-1))
        out[:, t] = nxt
        done |= nxt == EOS
        if bool(done.all()):
            break
        logits = model(nxt[:, None], cache=cache).logits[:, -1]
    model.train(was)
    return float(score(out, problems, verifier).mean())


def train_sft(out: str | Path, steps: int = 3000, batch: int = 64, lr: float = 5e-3, seed: int = 0,
              digits: int = 2, plain_add_weight: float = 1.0, log_every: int = 100, target_acc: float | None = 0.3,
              check_every: int = 25) -> dict:
    """Train and save ``<out>/policy.pt``; returns greedy accuracies per (op, tag) on held-out prompts.

    Small transformers learn addition in a sudden jump whose timing depends on the seed and on thread-level
    rounding (measured: from under 20% to over 80% greedy accuracy within 250 steps, at step 600 in one
    run and after step 1,000 in another). So by default the warm start stops at the first check where
    greedy accuracy on plain addition reaches ``target_acc``, measured on 300 problems drawn from the
    *training* triples (the evaluation triples are never used to choose the checkpoint); ``steps`` is
    then only the maximum. ``target_acc=None`` trains for exactly ``steps``.
    """
    from frontierlab.posttrain.tasks import encode_sft, problems_from
    torch.manual_seed(seed)
    out = Path(out)
    model = LM(policy_config())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    train_triples, held = split_problems(digits)
    pool = make_problems(20000, digits, ops="+-", tags=TAGS, seed=seed, exclude=held)
    if plain_add_weight != 1.0:        # thin out plain addition so RL has room to improve it
        keep = []
        g = torch.Generator().manual_seed(seed)
        for p in pool:
            if p.op == "+" and p.tag == "." and torch.rand(1, generator=g).item() > plain_add_weight:
                continue
            keep.append(p)
        pool = keep
    g = torch.Generator().manual_seed(seed)
    stop_set = make_problems(300, digits, "+", ".", seed=seed + 777, exclude=held)
    t0 = time.perf_counter()
    stopped_at, stop_acc = steps, None
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(pool), (batch,), generator=g).tolist()
        ids, mask = encode_sft([pool[i] for i in idx])
        for grp in opt.param_groups:
            grp["lr"] = lr * min(1.0, step / 50)
        loss = sft_loss(model, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"sft step {step:5d}  loss {loss.item():.4f}  {time.perf_counter() - t0:.0f}s")
        if target_acc is not None and step % check_every == 0 and step >= 100:
            stop_acc = greedy_accuracy(model, stop_set)
            if stop_acc >= target_acc:
                stopped_at = step
                print(f"sft step {step:5d}  plain-addition accuracy on training triples {stop_acc:.3f}: stop")
                break
    acc = {}
    for op in "+-":
        for tag in TAGS:
            probs = problems_from(held, tags=tag, digits=digits, ops=op)[:400]
            acc[f"{op}{tag}"] = greedy_accuracy(model, probs)
    meta = {"steps": stopped_at, "max_steps": steps, "target_acc": target_acc, "stop_acc": stop_acc, "batch": batch, "lr": lr, "seed": seed, "digits": digits,
            "plain_add_weight": plain_add_weight, "heldout_greedy_acc": acc,
            "seconds": round(time.perf_counter() - t0, 1)}
    save_policy(model, out / "policy.pt", meta)
    (out / "sft.json").write_text(json.dumps(meta, indent=2))
    return meta


def ensure_sft(out: str | Path = "runs/m12/sft", **kw) -> Path:
    """The course SFT checkpoint ``<out>/policy.pt``; trained (about a minute) if it does not exist yet."""
    path = Path(out) / "policy.pt"
    if not path.exists():
        print(f"training the SFT warm start -> {path}")
        train_sft(out, **kw)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000, help="maximum steps")
    ap.add_argument("--target-acc", type=float, default=0.3, help="stop at this plain-addition accuracy (<0: never)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain-add-weight", type=float, default=1.0)
    a = ap.parse_args(argv)
    print(json.dumps(train_sft(a.out, a.steps, a.batch, a.lr, a.seed, plain_add_weight=a.plain_add_weight,
                               target_acc=a.target_acc if a.target_acc >= 0 else None), indent=2))


if __name__ == "__main__":
    main()
