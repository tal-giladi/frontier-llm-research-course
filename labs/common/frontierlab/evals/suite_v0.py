"""Eval Suite v0 (Module 1 project): held-out loss on fixed windows + LAMBADA last-word prediction.

Two components, both scored per item so two runs can be compared with a paired bootstrap
(``frontierlab.stats.paired_bootstrap``):

1. **Held-out loss** — :func:`frontierlab.evals.heldout.window_losses` on the fixed validation (or,
   once, test) windows of Data-v0.
2. **LAMBADA (OpenAI variant)** — predict the last word of a passage from the rest of it (Paperno et
   al. 2016; the variant used by GPT-2 and by lm-evaluation-harness's ``lambada_openai`` task). Scored
   the way lm-evaluation-harness scores it: context = all words but the last, target = " " + last
   word; **accuracy** = greedy decoding reproduces every target token; **target log-likelihood** =
   sum of log-probabilities of the target tokens. At ~100M parameters accuracy is low and noisy, so the
   per-example log-likelihood (a continuous score) is the primary LAMBADA metric of Eval v0; accuracy
   is reported alongside it for comparison with published numbers.

Pinned source (checked 2026-10-03): Hugging Face dataset ``EleutherAI/lambada_openai`` at commit
``900124bf3b8235c6daf21033af9948b3f07346c4``, file ``data/lambada_test_en.jsonl``: 5,153 passages,
1,819,752 bytes, SHA-256 below, MIT licence. :func:`download_lambada` fetches exactly that file and
refuses any other content.

Tokenization uses the model's own tokenizer (Data-v0's byte-level BPE for course models), so a target
word may be several tokens. Context and target are encoded separately and concatenated, as in
lm-evaluation-harness, so a BPE merge never crosses the boundary. Contexts longer than the model's
window are truncated from the left.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Callable

import torch

from frontierlab.data.prepare import DEFAULT_OUT

LAMBADA_REPO = "EleutherAI/lambada_openai"
LAMBADA_REVISION = "900124bf3b8235c6daf21033af9948b3f07346c4"
LAMBADA_FILE = "data/lambada_test_en.jsonl"
LAMBADA_URL = f"https://huggingface.co/datasets/{LAMBADA_REPO}/resolve/{LAMBADA_REVISION}/{LAMBADA_FILE}"
LAMBADA_SHA256 = "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
LAMBADA_N = 5153
DEFAULT_LAMBADA = DEFAULT_OUT.parent / "evals" / "lambada_test_en.jsonl"
VERSION = "eval-v0.1"


def download_lambada(dest: str | Path = DEFAULT_LAMBADA, timeout: float = 60.0) -> Path:
    """Download the pinned LAMBADA test file (1.8 MB) once and verify its SHA-256."""
    dest = Path(dest)
    if dest.exists() and _sha256(dest) == LAMBADA_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LAMBADA_URL, timeout=timeout) as r:   # noqa: S310 - pinned https URL
        data = r.read()
    if hashlib.sha256(data).hexdigest() != LAMBADA_SHA256:
        raise ValueError("downloaded LAMBADA file does not match the pinned SHA-256; refusing to use it")
    dest.write_bytes(data)
    return dest


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lambada(path: str | Path = DEFAULT_LAMBADA, n: int | None = None) -> list[str]:
    """Passages in file order (the first ``n`` if given). Fixed order = fixed items for pairing."""
    texts = [json.loads(line)["text"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts[:n] if n else texts


def split_last_word(text: str) -> tuple[str, str]:
    """("the cat sat on the", " mat"): context without trailing space, target with a leading space."""
    text = text.rstrip()
    i = text.rfind(" ")
    if i <= 0:
        raise ValueError("passage has no last word to predict")
    return text[:i], text[i:]


def encode_example(encode: Callable[[str], list[int]], text: str, max_len: int) -> tuple[list[int], list[int]]:
    """Token ids of (context, target); the context is truncated from the left to fit ``max_len``."""
    ctx, tgt = split_last_word(text)
    c, t = encode(ctx), encode(tgt)
    if not c or not t:
        raise ValueError("empty context or target after tokenization")
    if len(t) >= max_len:
        raise ValueError("target longer than the model window")
    return c[-(max_len - len(t)):], t


@torch.no_grad()
def lambada_scores(model, encode: Callable[[str], list[int]], texts: list[str], max_len: int = 512,
                   batch: int = 16, device="cpu", pad_id: int = 0, autocast_dtype=None) -> list[dict]:
    """Per passage: {"correct": greedy match of all target tokens, "logprob": sum of target log-probs,
    "n_target": target length in tokens}. Order follows ``texts``.

    Sequences are right-padded to the longest in the batch; with causal attention, padding after a
    position cannot change that position's logits, and only target positions are read.
    """
    was_training = model.training
    model.eval()
    out = []
    encoded = [encode_example(encode, t, max_len) for t in texts]
    for i in range(0, len(encoded), batch):
        chunk = encoded[i:i + batch]
        L = max(len(c) + len(t) for c, t in chunk)
        x = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        for j, (c, t) in enumerate(chunk):
            x[j, :len(c) + len(t)] = torch.tensor(c + t)
        x = x.to(device)
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model(x).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for j, (c, t) in enumerate(chunk):
            pos = torch.arange(len(c) - 1, len(c) + len(t) - 1, device=x.device)   # positions predicting t
            tgt = torch.tensor(t, device=x.device)
            lp = logp[j, pos].gather(-1, tgt[:, None]).squeeze(-1)
            greedy = logp[j, pos].argmax(-1)
            out.append({"correct": bool(torch.equal(greedy, tgt)), "logprob": float(lp.sum()), "n_target": len(t)})
    model.train(was_training)
    return out


def data_v0_encoder(root: str | Path = DEFAULT_OUT) -> Callable[[str], list[int]]:
    """``encode(str) -> ids`` with Data-v0's tokenizer (needs the ``tokenizers`` package)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(root) / "tokenizer.json"))
    return lambda s: tok.encode(s).ids


def run_suite(model, val, encode, *, n_windows: int = 256, T: int = 512, lambada_path=DEFAULT_LAMBADA,
              n_lambada: int | None = None, device="cpu", autocast_dtype=None, pad_id: int = 0) -> dict:
    """Both components, per item, plus the pins needed to compare runs (store this next to the run card)."""
    from frontierlab.evals.heldout import window_losses
    texts = load_lambada(lambada_path, n_lambada)
    return {"version": VERSION, "heldout": {"split": val.split, "windows": n_windows, "T": T, "window_seed": 1234,
                                            "losses": window_losses(model, val, n_windows, T, device=device,
                                                                    autocast_dtype=autocast_dtype)},
            "lambada": {"repo": LAMBADA_REPO, "revision": LAMBADA_REVISION, "sha256": LAMBADA_SHA256,
                        "n": len(texts), "max_len": T,
                        "items": lambada_scores(model, encode, texts, T, device=device, pad_id=pad_id,
                                                autocast_dtype=autocast_dtype)}}


def summarize(result: dict, n_boot: int = 10000) -> dict:
    """Means with 95% bootstrap intervals over items for each component."""
    from frontierlab.stats import bootstrap_ci
    h = result["heldout"]["losses"]
    items = result["lambada"]["items"]
    acc = [float(it["correct"]) for it in items]
    lp = [it["logprob"] for it in items]
    nll_per_tok = sum(-x for x in lp) / sum(it["n_target"] for it in items)
    return {"heldout_loss": bootstrap_ci(h, n_boot=n_boot), "lambada_acc": bootstrap_ci(acc, n_boot=n_boot),
            "lambada_target_logprob": bootstrap_ci(lp, n_boot=n_boot),
            "lambada_target_ppl_per_token": math.exp(nll_per_tok)}
