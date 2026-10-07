"""Module 17 on open Hugging Face models: pinned models, published SAEs and transcoders, and the studies the
labs and the project run (free CPU on Qwen3-0.6B, main path on Qwen3-1.7B-Base / Qwen3-1.7B).

    python -m frontierlab.interp.hf smoke                                   # CPU, seconds: tiny random Qwen3
    python -m frontierlab.interp.hf ioi --model qwen3-0.6b --out runs/m17/ioi-0.6b.json      # free CPU, minutes
    python -m frontierlab.interp.hf ioi --model qwen3-1.7b-base --out runs/m17/ioi-1.7b.json # main path
    python -m frontierlab.interp.hf sae-eval --layer 14 --out runs/m17/qwen-scope-l14.json   # main path
    python -m frontierlab.interp.hf circuit --print                        # circuit-tracer commands (env B)

Everything here uses the course's own hooks (:mod:`frontierlab.interp.hooks`); the production libraries
appear only in the ``*_library`` functions and the printed commands, as a mapping:

=================================  ====================================================================
course                             library (versions pinned in references/versions.md)
=================================  ====================================================================
``hooks.capture`` / ``run_with``   TransformerLens 4.0.0 ``TransformerBridge.boot_transformers(...)``,
                                   ``run_with_cache`` / ``run_with_hooks``; nnsight 0.7.0
                                   ``with model.trace(prompt): model.model.layers[i].output.save()``
``sae.SAE`` (topk, k=50)           SAELens 6.53.0 ``SAE.from_pretrained("qwen-scope-3-1.7b-base-w32k-l50",
                                   "layer14")``; ``sae.encode`` / ``sae.decode``
``graphs.attribute`` / ``prune``   circuit-tracer 0.5.0 ``attribute(prompt, model)``, ``prune_graph(graph,
                                   node_threshold=0.8, edge_threshold=0.98)``
``graphs.intervene_feature``       circuit-tracer ``ReplacementModel.feature_intervention(inputs,
                                   [(layer, pos, feature, value)])``
=================================  ====================================================================

Two environments on the main path (checked 2026-10-07 from the packages' metadata): circuit-tracer 0.5.0
needs ``transformers<=4.57.3`` and sae-lens 6.53.0 needs ``transformer-lens<4``, while transformer-lens
4.0.0 needs ``transformers>=5.9``. Env A (this course's default): torch 2.14.1, transformers 5.18.0,
sae-lens 6.53.0 without TransformerLens hooks (SAEs are applied with the course's hooks), nnsight 0.7.0.
Env B (lesson 17.3 only): circuit-tracer 0.5.0 with its own pins. Main-path commands were not run in this
build; they are part of the Module 17 pilot.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.interp import hooks as HK
from frontierlab.interp import patching as P
from frontierlab.interp import tasks as T
from frontierlab.stats import bootstrap_ci

MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),          # free CPU (Module 10 pin)
    "qwen3-1.7b-base": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),  # Stage D base model
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),            # post-trained (Module 13 pin)
}

# Published dictionaries (checked 2026-10-07 on the Hugging Face API and SAELens v6.53.0 pretrained_saes.yaml).
QWEN_SCOPE = {"repo": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "revision": "ce1a79d9c5163932d65c417380e53230e1086370",
              "saelens_release": "qwen-scope-3-1.7b-base-w32k-l50", "kind": "topk", "k": 50, "d_sae": 32768,
              "site": "resid_post", "model": "qwen3-1.7b-base", "licence": "Qwen licence (license: other; read it)"}
TRANSCODERS = {"repo": "mwhanna/qwen3-1.7b-transcoders-lowl0", "revision": "9c1b17dfb156d82162ccd2cb7f047ac7f3d3585d",
               "model": "qwen3-1.7b", "licence": "MIT", "note": "trained on Qwen/Qwen3-1.7B (post-trained), not -Base"}


# --------------------------------------------------------------------------------------------- loading

class SmokeTokenizer:
    """Word-level tokenizer for CPU smoke tests: each space-prefixed word is one token. Ids are assigned in order
    of first appearance (no collisions below the vocabulary size)."""
    eos_token_id = pad_token_id = 1
    chat_template = None

    def __init__(self, vocab: int = 512):
        self.vocab = vocab
        self.ids: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def _ids(self, s: str):
        import re
        out = []
        for w in re.findall(r" ?[^ ]+", s):
            if w not in self.ids:
                i = 2 + len(self.ids) % (self.vocab - 2)
                self.ids[w] = i
                self.inv[i] = w
            out.append(self.ids[w])
        return out

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": self._ids(s)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inv.get(int(i), " ?") for i in ids)


def load(name: str, smoke: bool = False, dtype=torch.float32, device: str = "cpu"):
    """(model, tokenizer) at the pinned revision, in eval mode; ``smoke`` = a tiny random Qwen3."""
    if smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                          tie_word_embeddings=True)
        return Qwen3ForCausalLM(cfg).eval(), SmokeTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev = MODELS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype).to(device).eval()
    return model, tok


def text_windows(tok, n: int = 32, T: int = 128, seed: int = 0, smoke: bool = False, split: str = "val") -> torch.Tensor:
    """(n, T) windows of ordinary text for an HF tokenizer: Data-v0 validation tokens decoded with the
    Data-v0 tokenizer, then re-tokenized (off-target and SAE-splice checks)."""
    if smoke:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(2, 512, (n, T), generator=g)
    from tokenizers import Tokenizer
    from frontierlab.data.loader import TokenData
    from frontierlab.data.prepare import DEFAULT_OUT
    val = TokenData(split)
    dtok = Tokenizer.from_file(str(Path(DEFAULT_OUT) / "tokenizer.json"))
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        s = rng.randrange(0, len(val) - 6 * T)
        text = dtok.decode(val.tokens[s:s + 6 * T].astype(int).tolist())
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= T + 1:
            rows.append(ids[1:T + 1])          # drop a possibly partial first token
    return torch.tensor(rows)


# --------------------------------------------------------------------------------------------- IOI study

def ioi_study(model, tok, n: int = 48, top: int = 8, n_random: int = 19, seed: int = 0, windows=None,
              smoke: bool = False, log=print) -> dict:
    """The lesson 17.2 / project pipeline on an IOI distribution:

    1. attribution-patching map of every head on the *train* template family (cheap screen);
    2. real denoising patches of the ``2·top`` best-ranked heads, and of 8 random heads (calibration of
       the screen);
    3. candidate = the ``top`` heads with the largest real effect;
    4. mean-ablation of the candidate on clean prompts (necessity), against ``n_random`` random head sets of
       the same size (control);
    5. the same ablation on the *held-out* template family and on new names;
    6. the candidate's ablation effect on ordinary-text loss (off-target).
    """
    t0 = time.time()
    if smoke:
        names = [" Mary", " John", " Alice", " Bob", " Sarah", " David"]
        tr = T.ioi_prompts(tok, "train", n, seed, names)
        he = T.ioi_prompts(tok, "heldout", n, seed + 1, names)
    else:
        tr = T.ioi_prompts(tok, "train", n, seed, T.NAMES[:12])
        he = T.ioi_prompts(tok, "heldout", n, seed + 1, T.NAMES[12:])     # other templates AND other names
    ptr, phe = T.as_pairs(tr), T.as_pairs(he)
    base = P.baselines(model, ptr)
    log(f"train family: n={len(ptr.good)}, clean logit diff {base['clean_mean']:.3f}, corrupt {base['corrupt_mean']:.3f}")
    att = P.attribution_heads(model, ptr)
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)                    # everything below is forward-only
    L, H = att.shape
    order = np.dstack(np.unravel_index(np.argsort(-att.ravel()), att.shape))[0]
    screen = [tuple(map(int, x)) for x in order[:2 * top]]
    real = {}
    _, acts = HK.capture(model, ptr.clean, [f"z.{l}" for l in range(L)])
    hd = HK.head_dim(model)[1]
    for (l, h) in screen:
        lg = HK.run_with(model, ptr.corrupt, {f"z.{l}": HK.replace_at(acts[f"z.{l}"], None, h, hd)})
        real[(l, h)] = float(P.normalised(P.metric_fn(ptr)(lg), base["clean"], base["corrupt"], "denoise").mean())
    rng = np.random.default_rng(seed)
    calib = []
    for i in rng.choice(L * H, size=min(8, L * H), replace=False):
        l, h = divmod(int(i), H)
        lg = HK.run_with(model, ptr.corrupt, {f"z.{l}": HK.replace_at(acts[f"z.{l}"], None, h, hd)})
        calib.append({"head": (l, h), "attribution": float(att[l, h]),
                      "patch": float(P.normalised(P.metric_fn(ptr)(lg), base["clean"], base["corrupt"], "denoise").mean())})
    cand_list = sorted(real, key=lambda k: -real[k])[:top]
    cand: dict[int, list[int]] = {}
    for l, h in cand_list:
        cand.setdefault(l, []).append(h)
    log(f"candidate heads {cand_list} ({time.time() - t0:.0f} s)")
    ctrl = P.component_control(model, ptr, cand, "mean", n_random=n_random, seed=seed, layer_matched=True)
    ctrl_any = P.component_control(model, ptr, cand, "mean", n_random=n_random, seed=seed)     # secondary
    held = P.summarise(P.ablation_effect(model, phe, cand, "mean"))
    hbase = P.baselines(model, phe)
    windows = text_windows(tok, 16, 64, seed, smoke) if windows is None else windows
    H_, hd_ = HK.head_dim(model)

    def mean_ablate_edits(ref):
        _, a = HK.capture(model, ref, [f"z.{l}" for l in cand])
        ed = {}
        for l, hs in cand.items():
            mu = a[f"z.{l}"].mean((0, 1))

            def f(x, hs=hs, mu=mu):
                y = x.clone()
                for h in hs:
                    y[..., h * hd_:(h + 1) * hd_] = mu[h * hd_:(h + 1) * hd_]
                return y
            ed[f"z.{l}"] = f
        return ed
    off0 = P.offtarget_loss(model, windows)
    off1 = P.offtarget_loss(model, windows, mean_ablate_edits(windows))
    d = off1 - off0
    om, olo, ohi = bootstrap_ci(d)
    out = {"model": getattr(model.config, "_name_or_path", "smoke"), "n_train": int(len(ptr.good)),
           "n_heldout": int(len(phe.good)), "baseline": {"clean": base["clean_mean"], "corrupt": base["corrupt_mean"]},
           "heldout_baseline": {"clean": hbase["clean_mean"], "corrupt": hbase["corrupt_mean"]},
           "attribution": att.tolist(), "screen_patch": {f"{l}.{h}": v for (l, h), v in real.items()},
           "calibration": calib, "candidate": {str(k): v for k, v in cand.items()},
           "ablation": {k: v for k, v in ctrl.items() if k != "random"}, "random_effects": ctrl["random"],
           "any_layer_control": {k: v for k, v in ctrl_any.items() if k != "random"},
           "heldout": held, "offtarget": {"delta": om, "ci": (olo, ohi), "clean_loss": float(off0.mean())},
           "seconds": time.time() - t0}
    torch.set_grad_enabled(prev_grad)
    return out


# --------------------------------------------------------------------------------------------- SAEs (main path)

def qwen_scope_sae(layer: int, device: str = "cpu"):
    """The Qwen-Scope TopK SAE (32K latents, k = 50) for the residual stream after ``layer`` of Qwen3-1.7B-Base,
    loaded into the course's :class:`frontierlab.interp.sae.SAE` from the files at the pinned revision.
    Not run in this build (part of the Module 17 pilot): the file layout is checked at load time."""
    from huggingface_hub import HfApi, hf_hub_download
    from frontierlab.interp.sae import SAE
    files = [f for f in HfApi().list_repo_files(QWEN_SCOPE["repo"], revision=QWEN_SCOPE["revision"])
             if f.endswith(".pt") and (f"layer{layer}." in f or f"layer{layer}/" in f or f"layer_{layer}." in f)]
    if len(files) != 1:
        raise FileNotFoundError(f"expected one file for layer {layer}, found {files}")
    sd = torch.load(hf_hub_download(QWEN_SCOPE["repo"], files[0], revision=QWEN_SCOPE["revision"]), map_location=device)
    W_enc, W_dec = sd["W_enc"].float(), sd["W_dec"].float()
    d_sae = QWEN_SCOPE["d_sae"]
    W_enc = W_enc if W_enc.shape[1] == d_sae else W_enc.T          # course layout: W_enc (d, d_sae)
    W_dec = W_dec if W_dec.shape[0] == d_sae else W_dec.T          # course layout: W_dec (d_sae, d)
    sae = SAE(W_enc.shape[0], d_sae, "topk", k=QWEN_SCOPE["k"])
    with torch.no_grad():
        sae.W_enc.copy_(W_enc)
        sae.W_dec.copy_(W_dec)
        sae.b_enc.copy_(sd["b_enc"].float())
        sae.b_dec.copy_(sd["b_dec"].float())
        sae.scale.fill_(1.0)                                         # published SAEs read raw activations
    return sae.eval()


def qwen_scope_sae_library(layer: int, device: str = "cpu"):
    """The same SAE through SAELens 6.53.0 (env A): returns the SAELens SAE object."""
    from sae_lens import SAE as LibSAE
    sae = LibSAE.from_pretrained(QWEN_SCOPE["saelens_release"], f"layer{layer}", device=device)
    return sae


@torch.no_grad()
def check_against_library(layer: int, acts: torch.Tensor, device: str = "cpu") -> dict:
    """Pilot check: the course loader and SAELens must give the same codes and reconstructions on the same
    activations (N, 2048). The course loader assumes the SAELens TopK convention, pre = (x - b_dec) W_enc + b_enc;
    if the published SAE does not subtract b_dec from its input, the code difference here is large and
    ``SAE.pre`` must be called without it (record which at the pilot)."""
    ours = qwen_scope_sae(layer, device)
    lib = qwen_scope_sae_library(layer, device)
    x = acts.to(device).float()
    z1, z2 = ours.encode(x), lib.encode(x).float()
    return {"max_code_diff": float((z1 - z2).abs().max()),
            "max_recon_diff": float((ours.decode(z1) - lib.decode(z2).float()).abs().max())}


def sae_eval(model, tok, sae, layer: int, windows=None, smoke: bool = False) -> dict:
    """FVU, L0 and the spliced next-token loss (delta LM loss) of an SAE on ``resid_post.layer``; the splice
    check (``sae+error`` equals clean) runs first."""
    from frontierlab.interp import sae as S
    site = f"resid_post.{layer}"
    windows = text_windows(tok, 32, 128, 0, smoke) if windows is None else windows
    clean = S.spliced_losses(model, windows, site, sae, "clean")
    check = S.spliced_losses(model, windows, site, sae, "sae+error")
    assert np.abs(check - clean).max() < 1e-3, "splice check failed: x̂ + error must reproduce the clean loss"
    spl = S.spliced_losses(model, windows, site, sae, "sae")
    acts = S.collect(model, windows, site)
    mu = acts.mean(0)
    mean_abl = S.spliced_losses(model, windows, site, sae, "mean", mean_act=mu)
    ev = S.evaluate(sae, acts)
    d = spl - clean
    m, lo, hi = bootstrap_ci(d)
    return {**ev, "clean_loss": float(clean.mean()), "spliced_loss": float(spl.mean()), "delta_loss": m,
            "delta_loss_ci": (lo, hi), "mean_ablation_loss": float(mean_abl.mean()),
            "loss_recovered_vs_mean": S.loss_recovered(float(clean.mean()), float(spl.mean()), float(mean_abl.mean()))}


CIRCUIT_COMMANDS = """# Env B (lesson 17.3, main path): circuit-tracer 0.5.0 pins transformers<=4.57.3.
python -m venv .venv-ct && . .venv-ct/bin/activate
pip install "circuit-tracer==0.5.0" "torch==2.14.1"
# Qwen3-1.7B (post-trained) with the published per-layer transcoders (mwhanna/qwen3-1.7b-transcoders-lowl0 @ 9c1b17d):
circuit-tracer attribute --prompt "Fact: the capital of the state containing Dallas is" \\
    --transcoder_set mwhanna/qwen3-1.7b-transcoders-lowl0 --slug dallas --graph_file_dir runs/m17/graphs
# In Python (same env):
#   from circuit_tracer import ReplacementModel, attribute
#   from circuit_tracer.graph import prune_graph, compute_graph_scores
#   rm = ReplacementModel.from_pretrained("Qwen/Qwen3-1.7B", "mwhanna/qwen3-1.7b-transcoders-lowl0", dtype=torch.bfloat16)
#   g = attribute("Fact: the capital of the state containing Dallas is", rm, max_n_logits=10, desired_logit_prob=0.95)
#   pr = prune_graph(g, node_threshold=0.8, edge_threshold=0.98); print(compute_graph_scores(g))
#   logits, acts = rm.feature_intervention(prompt, [(layer, pos, feature, 0.0)])   # validate one node
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--out", type=Path, default=None)
    i = sub.add_parser("ioi")
    i.add_argument("--model", default="qwen3-0.6b", choices=sorted(MODELS))
    i.add_argument("--n", type=int, default=48)
    i.add_argument("--top", type=int, default=8)
    i.add_argument("--random", type=int, default=19)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    i.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("sae-eval")
    e.add_argument("--layer", type=int, default=14)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("circuit")
    c.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "circuit":
        print(CIRCUIT_COMMANDS)
        return
    if a.cmd == "smoke":
        model, tok = load("", smoke=True)
        r = ioi_study(model, tok, n=16, top=2, n_random=4, smoke=True)
        from frontierlab.interp.sae import SAE
        sae = SAE(64, 256, "topk", k=8)
        r2 = sae_eval(model, tok, sae, 1, smoke=True)
        out = {"ioi": {k: r[k] for k in ("candidate", "ablation", "heldout", "offtarget")}, "sae": r2}
        print(json.dumps(out, indent=1, default=str)[:2000])
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(out, indent=1, default=str))
        return out
    dtype = torch.bfloat16 if a.device.startswith("cuda") else torch.float32
    if a.cmd == "ioi":
        model, tok = load(a.model, dtype=torch.float32, device=a.device)   # fp32: patching differences are small
        r = ioi_study(model, tok, a.n, a.top, a.random, a.seed)
    else:
        model, tok = load("qwen3-1.7b-base", dtype=dtype, device=a.device)
        r = sae_eval(model, tok, qwen_scope_sae(a.layer, a.device), a.layer)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: v for k, v in r.items() if k not in ("attribution", "random_effects")}, indent=1, default=str))


if __name__ == "__main__":
    main()
