"""Lab 12.1, main path: Gao et al.'s synthetic-gold setup at small scale on a rented GPU.

    python labs/module-12/lesson-01/rm_main.py --out runs/m12/l121-main          # 1x H100 80 GB; not run in this build
    python labs/module-12/lesson-01/rm_main.py --gold 1.7b --train-prompts 500 --pool-prompts 100         --pairs 1000,4000 --out runs/m12/l121-t4                                  # Colab/Kaggle T4
    python labs/module-12/lesson-01/rm_main.py --smoke --out runs/m12/l121-smoke # CPU: tiny random models, a minute

Stages (each cached in ``--out``; rerun the same command after a disconnect):

1. **Generate.** The Stage D base model (Qwen3-1.7B-Base, pinned) answers prompts from
   ``HuggingFaceH4/ultrafeedback_binarized`` (pinned revision, MIT) in a plain ``User:/Assistant:``
   format: 16 responses for each of 2,000 ``train_prefs`` prompts (reward-model data) and 64 responses for
   each of 500 ``test_prefs`` prompts (the best-of-n pool). Temperature 1.0, 256 new tokens.
2. **Gold.** ``Skywork/Skywork-Reward-V2-Qwen3-8B`` (pinned, Apache-2.0) scores every response. It plays
   the role of the human: the "true" reward that the proxies never see directly.
3. **Proxies.** Reward models initialised from Qwen3-0.6B-Base (pinned) with a scalar head, trained with the
   Bradley-Terry loss on 1,000 / 4,000 / 16,000 pairs labelled by the gold score (the higher-scored
   response is "chosen", as in Gao et al. section 2.1), 1 epoch, 2 seeds.
4. **Best-of-n.** Per held-out prompt, the unbiased BoN estimate of gold and proxy reward for
   n = 1..64 (KL up to 3.18 nats), averaged over prompts with a bootstrap interval over prompts.

PROJECTED cost (pending the Module 12 pilot): generation 64,000 responses x ~256 tokens = 1.6e7 tokens;
with transformers ``generate`` at batch 64 about 2-3k tokens/s on an H100 -> 1.5-2.2 h (vLLM 0.30.0 is
several times faster). Gold scoring 64,000 x ~400 tokens x 2 x 8e9 FLOPs = 4.1e17 FLOPs, at 40% of 989
TFLOP/s -> 17 min. Proxy training 6 runs x <= 16,000 pairs x 2 x 400 tokens x 6 x 0.6e9 = <= 4.6e16 FLOPs
each -> minutes. Total about 2-3 GPU-hours.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.posttrain import STAGE_D_BASE
from frontierlab.posttrain import reward as RW

UF = ("HuggingFaceH4/ultrafeedback_binarized", "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85")
GOLDS = {"8b": ("Skywork/Skywork-Reward-V2-Qwen3-8B", "6f19fdefb933293d4898bdb59a96f7223d998659"),
         "1.7b": ("Skywork/Skywork-Reward-V2-Qwen3-1.7B", "e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc")}
PROXY = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
POLICY = (STAGE_D_BASE["repo"], STAGE_D_BASE["revision"])
NS = [1, 2, 4, 8, 16, 32, 64]


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def tiny(kind):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification
    cfg = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, num_labels=1, pad_token_id=0)
    return Qwen3ForCausalLM(cfg) if kind == "lm" else Qwen3ForSequenceClassification(cfg)


class CharTok:
    pad_token_id, eos_token_id = 0, 1

    def __call__(self, texts, **kw):
        ids = [[2 + (ord(c) % 94) for c in t][-200:] for t in texts]
        T = max(len(i) for i in ids)
        out = torch.zeros(len(ids), T, dtype=torch.long)
        att = torch.zeros(len(ids), T, dtype=torch.long)
        for k, i in enumerate(ids):
            out[k, T - len(i):] = torch.tensor(i)
            att[k, T - len(i):] = 1
        return {"input_ids": out, "attention_mask": att}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + ((i - 2) % 94)) for i in ids if i > 1)


def load_prompts(smoke: bool, split: str, n: int) -> list[str]:
    if smoke:
        rng = random.Random(split)
        return [f"say {rng.randint(0, 99)}" for _ in range(n)]
    from datasets import load_dataset
    ds = load_dataset(UF[0], split=split, revision=UF[1])
    return [r["prompt"] for r in ds.select(range(n))]


def fmt(prompt: str, response: str | None = None) -> str:
    return f"User: {prompt}\nAssistant:" + ("" if response is None else " " + response)


@torch.no_grad()
def generate(a, prompts, per_prompt) -> list[list[str]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.smoke:
        model, tok = tiny("lm").eval(), CharTok()
    else:
        tok = AutoTokenizer.from_pretrained(POLICY[0], revision=POLICY[1], padding_side="left")
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(POLICY[0], revision=POLICY[1], dtype=torch.bfloat16).to(device()).eval()
    out = []
    for p in prompts:
        enc = tok([fmt(p)] * per_prompt, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        seq = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        resp = [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True) for s in seq]
        out.append([r.split("\nUser:")[0].strip() for r in resp])
    return out


@torch.no_grad()
def gold_scores(a, prompts, responses) -> list[list[float]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if a.smoke:
        torch.manual_seed(123)
        model, tok = tiny("cls").eval(), CharTok()
        texts = lambda p, rs: [fmt(p, r) for r in rs]
    else:
        gold = GOLDS[a.gold]
        tok = AutoTokenizer.from_pretrained(gold[0], revision=gold[1])
        model = AutoModelForSequenceClassification.from_pretrained(gold[0], revision=gold[1], dtype=torch.bfloat16,
                                                                   num_labels=1).to(device()).eval()
        texts = lambda p, rs: [tok.apply_chat_template([{"role": "user", "content": p},
                                                        {"role": "assistant", "content": r}], tokenize=False)
                               for r in rs]
    out = []
    for p, rs in zip(prompts, responses):
        enc = tok(texts(p, rs), return_tensors="pt", padding=True, truncation=True, max_length=1024) \
            if not a.smoke else tok(texts(p, rs))
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out.append(model(**enc).logits[:, 0].float().cpu().tolist())
    return out


def train_proxy(a, prompts, responses, golds, n_pairs, seed):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    rng = random.Random(seed)
    pairs = []
    for p, rs, gs in zip(prompts, responses, golds):
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        for i, j in zip(idx[0::2], idx[1::2]):
            if gs[i] != gs[j]:
                c, r = (i, j) if gs[i] > gs[j] else (j, i)
                pairs.append((fmt(p, rs[c]), fmt(p, rs[r])))
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    if a.smoke:
        model, tok = tiny("cls"), CharTok()
    else:
        tok = AutoTokenizer.from_pretrained(PROXY[0], revision=PROXY[1])
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(PROXY[0], revision=PROXY[1], num_labels=1,
                                                                   dtype=torch.float32)
        model.config.pad_token_id = tok.pad_token_id
    model.to(device()).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.rm_lr, weight_decay=0.0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device() == "cuda")
    for k in range(0, len(pairs), a.rm_batch):
        chunk = pairs[k:k + a.rm_batch]
        enc = tok([c for c, _ in chunk] + [r for _, r in chunk], return_tensors="pt", padding=True, truncation=True,
                  max_length=768) if not a.smoke else tok([c for c, _ in chunk] + [r for _, r in chunk])
        enc = {kk: v.to(model.device) for kk, v in enc.items()}
        with amp:
            s = model(**enc).logits[:, 0].float()
        loss = RW.bt_loss(s[:len(chunk)], s[len(chunk):])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model, tok, len(pairs)


@torch.no_grad()
def proxy_scores(a, model, tok, prompts, responses):
    out = []
    for p, rs in zip(prompts, responses):
        enc = tok([fmt(p, r) for r in rs], return_tensors="pt", padding=True, truncation=True, max_length=768) \
            if not a.smoke else tok([fmt(p, r) for r in rs])
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out.append(model(**enc).logits[:, 0].float().cpu().tolist())
    return out


def cached(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    val = fn()
    path.write_text(json.dumps(val))
    return val


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/m12/l121-main"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-prompts", type=int, default=2000)
    ap.add_argument("--pool-prompts", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--pairs", default="1000,4000,16000")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--rm-lr", type=float, default=1e-5)
    ap.add_argument("--rm-batch", type=int, default=16)
    ap.add_argument("--gold", choices=sorted(GOLDS), default="8b",
                    help="8b on the main path; 1.7b on a T4 (the 8B gold model does not fit in 16 GB with the rest)")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train_prompts, a.pool_prompts, a.max_new, a.pairs, a.rm_lr, a.rm_batch = 16, 6, 8, "8,24", 1e-3, 8
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tp, pp = load_prompts(a.smoke, "train_prefs", a.train_prompts), load_prompts(a.smoke, "test_prefs", a.pool_prompts)
    torch.manual_seed(0)
    tr = cached(a.out / "train_responses.json", lambda: generate(a, tp, 16))
    po = cached(a.out / "pool_responses.json", lambda: generate(a, pp, 64 if not a.smoke else 16))
    tg = cached(a.out / "train_gold.json", lambda: gold_scores(a, tp, tr))
    pg = cached(a.out / "pool_gold.json", lambda: gold_scores(a, pp, po))
    ns = [n for n in NS if n <= len(po[0])]
    res = {"n": ns, "kl": RW.bon_kl(ns).tolist(),
           "oracle": [float(np.mean([RW.bon_expected(g, g, n) - np.mean(g) for g in pg])) for n in ns], "proxies": []}
    for n_pairs in [int(x) for x in a.pairs.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            model, tok, used = train_proxy(a, tp, tr, tg, n_pairs, seed)
            ps = proxy_scores(a, model, tok, pp, po)
            per_prompt = np.array([[RW.bon_expected(p, g, n) - np.mean(g) for n in ns] for p, g in zip(ps, pg)])
            res["proxies"].append({"n_pairs": used, "seed": seed, "gold": per_prompt.mean(0).tolist(),
                                   "gold_sem": (per_prompt.std(0, ddof=1) / np.sqrt(len(pg))).tolist(),
                                   "proxy": [float(np.mean([RW.bon_expected(p, p, n) - np.mean(p) for p in ps]))
                                             for n in ns]})
            print(json.dumps(res["proxies"][-1]))
    res["seconds"] = round(time.perf_counter() - t0, 1)
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out / 'results.json'}")


if __name__ == "__main__":
    main()
