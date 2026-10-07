"""The claim card of the Module 17 project: what one supported causal claim must contain.

A claim has a *scope* (model, behaviour, prompt distribution, components) and evidence of five kinds. The
card is checked mechanically before anyone reads the prose; a check that fails names the missing evidence.
Thresholds are the course's defaults and are written into the experiment contract *before* the runs.

==========================  =====================================================================================
evidence                    what passes
==========================  =====================================================================================
effect                      the intervention's mean effect on the selection prompts, with a 95% interval
                            that excludes 0
control                     the candidate beats random directions or random components of the same size:
                            ``p_random`` ≤ 0.05 (needs at least 19 random draws)
held-out                    on prompts *not* used to choose the components (other template family or
                            other tokens) the 95% interval excludes 0 and the mean is at least
                            ``min_transfer`` (default 0.5) of the selection effect
off-target                  the intervention changes next-token loss on ordinary text by less than
                            ``max_offtarget`` nats (default 0.05), or the card reports it as a limitation
limitations                 at least three stated limits, one of which names the scale (model size)
==========================  =====================================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ClaimCard:
    claim: str                      # one sentence, scoped: model, behaviour, distribution, components
    model: str                      # repo@revision or run card
    intervention: str               # e.g. "mean-ablate heads 1.0-1.3 (z), clean prompts"
    effect: float                   # mean normalised effect on the selection prompts
    effect_ci: tuple
    control_kind: str               # "random directions" | "random components" | ...
    control_p: float                # p_random
    control_draws: int
    heldout_effect: float
    heldout_ci: tuple
    heldout_split: str              # what makes it held out
    offtarget_delta: float          # change in next-token loss on ordinary text (nats)
    limitations: list = field(default_factory=list)
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def check(card: ClaimCard, min_transfer: float = 0.5, max_offtarget: float = 0.05, alpha: float = 0.05) -> list[str]:
    """Problems with the card (an empty list means the evidence supports the claim as scoped)."""
    probs = []
    lo, hi = card.effect_ci
    if lo <= 0 <= hi:
        probs.append("effect: the 95% interval includes 0")
    if card.control_draws < 19:
        probs.append(f"control: {card.control_draws} random draws cannot give p ≤ {alpha} (need ≥ 19)")
    elif card.control_p > alpha:
        probs.append(f"control: p_random = {card.control_p:.3f} > {alpha}: not distinguishable from random components")
    hlo, hhi = card.heldout_ci
    if hlo <= 0 <= hhi:
        probs.append("held-out: the 95% interval includes 0")
    elif card.effect != 0 and card.heldout_effect / card.effect < min_transfer:
        probs.append(f"held-out: effect transfers at {card.heldout_effect / card.effect:.2f} of the selection effect "
                     f"(< {min_transfer}): narrow the claim to the selection distribution")
    if not card.heldout_split.strip():
        probs.append("held-out: say what makes the held-out prompts held out")
    if abs(card.offtarget_delta) >= max_offtarget and not any("off-target" in s.lower() for s in card.limitations):
        probs.append(f"off-target: the intervention changes ordinary-text loss by {card.offtarget_delta:+.3f} nats; "
                     "report it as a limitation or use a more specific intervention")
    if len(card.limitations) < 3:
        probs.append("limitations: state at least three")
    if not any(w in s.lower() for s in card.limitations for w in ("scale", "size", "parameter")):
        probs.append("limitations: one limit must name the model scale")
    return probs
