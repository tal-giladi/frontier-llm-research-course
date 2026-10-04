"""µP (maximal update parametrization) for the course model, relative to a base width (lesson 07.3).

Tensor Programs V (Yang et al., arXiv 2203.03466, Table 3) gives, for Adam, how initialisation and
learning rate must scale with width so that every layer's activations change by Theta(1) per step at any
width — the condition under which the best hyperparameters stop moving as the model widens (µTransfer):

    weight type        init variance        Adam learning rate
    input (embedding)  1/fan_in  (const)    1           (const in width)
    hidden matrices    1/fan_in             1/fan_in
    output (readout)   1/fan_in^2           1/fan_in

Here everything is stated *relative to a base width* C_0 at which µP and the course's standard
parametrization (SP: every weight N(0, 0.02^2), one global learning rate) coincide, as the ``mup`` library
does with base shapes. With width multiplier m = C / C_0:

* hidden matrices (attention q/k/v/o, SwiGLU gate/up/down): init std 0.02 / sqrt(m), learning rate x 1/m;
* embedding: std 0.02, learning rate x 1 (fan_in of an embedding is the vocabulary, which does not grow);
* output: Baseline-0 ties the head to the embedding, so the readout cannot have its own init or learning
  rate. As in ``mup.MuSharedReadout``, the logits are multiplied by 1/m instead (TP V shows the multiplier
  and the init/LR forms are equivalent); :class:`MuReadout` does that and keeps the state-dict name
  ``lm_head.weight``;
* norm gains and other vectors: learning rate x 1.
* attention: TP V uses q·k/d instead of q·k/sqrt(d). That matters only when the head dimension grows with
  width; :func:`width_config` keeps d fixed and adds heads, so 1/sqrt(d) stays (REASONABLE INDUSTRY PRACTICE).

**Muon.** The learning-rate rule depends on how the Muon update is scaled (lesson 07.3 derives it):
with ``adjust="match_rms"`` (update RMS 0.2·lr per entry, like Adam) the entry-size argument gives the Adam
rule 1/m, while the spectral condition of Yang, Simon and Bernstein (arXiv 2310.17813: spectral norms of
updates should scale like sqrt(fan_out/fan_in)) gives 1/sqrt(m), because an orthogonalised update is full
rank. With ``adjust="original"`` (Jordan's sqrt(max(1, A/B))) the spectral condition holds at a constant
learning rate. :func:`lr_scales` implements all three as ``muon_rule`` so the transfer test can decide.
Essential AI (arXiv 2505.02222, section 3.4) report that the µP scaling used for AdamW transferred for
Muon in their runs up to 3.7B parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierlab.model.config import ModelConfig

HIDDEN_KEYS = ("self_attn.", "mlp.")


def width_config(base: ModelConfig, width: int) -> ModelConfig:
    """``base`` widened (or narrowed) to hidden size ``width``: head_dim fixed, heads, KV heads and SwiGLU scaled."""
    if width % base.head_dim:
        raise ValueError("width must be a multiple of head_dim")
    m = width / base.hidden_size
    H = width // base.head_dim
    KV = max(1, round(base.num_key_value_heads * m))
    while H % KV:
        KV -= 1
    return base.with_(hidden_size=width, num_attention_heads=H, num_key_value_heads=KV,
                      intermediate_size=int(round(base.intermediate_size * m)))


def is_hidden(name: str, p: torch.Tensor) -> bool:
    return p.ndim == 2 and any(k in name for k in HIDDEN_KEYS) and "norm" not in name


class MuReadout(nn.Linear):
    """Output head with a fixed multiplier on the logits: ``F.linear(h, W) * output_mult``.

    Keeps the parameter name ``lm_head.weight`` (and the tie to the embedding), so checkpoints and exact resume
    are unchanged. ``frontierlab.perf.chunked_ce`` reads ``lm_head.weight`` directly and would skip the
    multiplier: do not combine µP with ``--loss chunked`` (the Module 7 wrapper refuses).
    """

    def __init__(self, weight: nn.Parameter, output_mult: float):
        nn.Module.__init__(self)
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.weight = weight
        self.register_parameter("bias", None)
        self.output_mult = float(output_mult)

    def forward(self, x):
        return F.linear(x, self.weight) * self.output_mult


def multiplier(width: int, base_width: int) -> float:
    return width / base_width


@torch.no_grad()
def apply_mup(model, base_width: int, std: float = 0.02, generator: torch.Generator | None = None) -> float:
    """Re-initialise ``model`` (a ``frontierlab.model.LM``) in µP relative to ``base_width``; returns m.

    Hidden matrices get N(0, (std/sqrt(m))^2); the embedding keeps N(0, std^2); the head becomes a
    :class:`MuReadout` with multiplier 1/m. At m = 1 the weights are re-drawn with the same distribution as
    SP (the draws themselves differ from ``LM.__init__`` unless the RNG is reset first).
    """
    m = multiplier(model.config.hidden_size, base_width)
    for name, p in model.named_parameters():
        if is_hidden(name, p):
            p.normal_(0.0, std / math.sqrt(m), generator=generator)
    tied = model.config.tie_word_embeddings
    w = model.model.embed_tokens.weight if tied else model.lm_head.weight
    if not tied:                                   # untied head: own µP init, variance 1/fan_in^2 relative
        w.normal_(0.0, std / m, generator=generator)
        model.lm_head = MuReadout(w, 1.0)
    else:
        model.lm_head = MuReadout(w, 1.0 / m)
    return m


def lr_scales(model, base_width: int, optimizer: str = "adamw", muon_rule: str = "auto",
              adjust: str = "match_rms") -> dict:
    """{parameter name: learning-rate factor} for µP relative to ``base_width``.

    Adam hidden matrices: 1/m. Muon hidden matrices (``optimizer="muon"``): ``muon_rule`` "adam" (1/m),
    "spectral" (1/sqrt(m) for match_rms, 1 for original) or "auto" (= "spectral"). Everything else: 1.
    An untied µP head (variance 1/fan_in^2) gets 1/m as TP V's output row says.
    """
    m = multiplier(model.config.hidden_size, base_width)
    out = {}
    for name, p in model.named_parameters():
        f = 1.0
        if is_hidden(name, p):
            if optimizer == "muon":
                rule = "spectral" if muon_rule == "auto" else muon_rule
                if rule == "adam":
                    f = 1.0 / m
                elif rule == "spectral":
                    f = 1.0 / math.sqrt(m) if adjust == "match_rms" else 1.0
                else:
                    raise ValueError(muon_rule)
            else:
                f = 1.0 / m
        elif "lm_head" in name and not model.config.tie_word_embeddings:
            f = 1.0 / m
        out[name] = f
    return out


@torch.no_grad()
def activation_rms(model, idx: torch.Tensor) -> dict:
    """RMS of the embedding output, each block's output and the logits for one batch (coordinate check)."""
    acts, hooks = {}, []
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: acts.__setitem__("embed", o)))
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(f"block{i}", o)))
    was = model.training
    model.eval()
    try:
        logits = model(idx).logits
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    out = {k: v.float().pow(2).mean().sqrt().item() for k, v in acts.items()}
    out["logits"] = logits.float().pow(2).mean().sqrt().item()
    return out
