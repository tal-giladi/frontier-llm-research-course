"""Multi-turn rollouts at the token level: actions, observations and the masks between them (lesson 16.3).

A multi-turn episode is one token sequence::

    BOS prompt | action_1 . | =obs_1 . | action_2 . | =obs_2 . | ... | final answer EOS
               ^ policy     ^ environment (inserted, never sampled)

The policy samples the action tokens. When it emits the end-of-turn token ``.``, the environment's
``on_action`` returns an observation, and its tokens are **inserted** into the sequence: they are fed to the
model so later actions can read them, but they were not sampled, so they have no policy probability, no
advantage and no gradient. :func:`sample_multiturn` keeps every row in lock-step (one token per row per
step, an inserted token counts as a step) so a single KV cache serves the batch.

Shapes (B episodes, P prompt tokens, R = ``max_tokens``):

* ``tokens``       (B, P + R) int64   prompt, then actions and observations, PAD after the episode ends
* ``action_mask``  (B, R) float32     1 on policy-sampled tokens (including each ``.`` and the final EOS)
* ``obs_mask``     (B, R) float32     1 on inserted observation tokens
* ``turn_index``   (B, R) int64       which turn a token belongs to (0, 1, ...); -1 on PAD
* ``sampler_logp`` (B, R) float32     log-probability of sampled tokens under the sampler, 0 elsewhere
* ``finished``     (B,) bool          the episode ended with EOS within R tokens and the turn limit

The loss must use ``action_mask``. Training on ``obs_mask`` positions teaches the policy to *predict the
environment* (to write tool outputs itself), which is the bug Search-R1's "retrieved token masking" avoids.

**Probe task** (:class:`ProbeTask`): the hidden function of :mod:`~frontierlab.agents.dsl` is not shown. The
agent may query it, ``?3.`` -> ``=6.``, at most ``max_queries`` times, then answers with a program and EOS.
One query cannot always identify the function (f(2) = 4 for both ``x+2`` and ``x*2``; f(1) = 3 for ``x+2``
and ``x*2+1``), so information gathering over turns is part of the task and the final reward has to be
credited to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from frontierlab.agents import dsl
from frontierlab.posttrain.tokenizer import BOS, EOS, PAD

EOT = dsl.TOKD.stoi["."]


@dataclass
class Episode:
    """Bookkeeping of one row: the actions so far, the observations, per-turn rewards."""
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    turn_rewards: list[float] = field(default_factory=list)
    final: str | None = None
    finished: bool = False
    truncated: bool = False


class SingleTurn:
    """A :class:`~frontierlab.agents.dsl.DslTask` as a one-turn episode: the program is the only action."""

    max_turns = 1

    def __init__(self, task: dsl.DslTask):
        self.task = task

    @property
    def prompt(self) -> str:
        return self.task.prompt

    def on_action(self, text: str, ep: Episode) -> str | None:
        return None                         # a "." ends nothing here; the action continues until EOS

    def turn_reward(self, text: str, ep: Episode) -> float:
        return 0.0


class ProbeTask:
    """Query-then-answer task over a hidden :class:`~frontierlab.agents.dsl.Func`."""

    def __init__(self, func: dsl.Func, max_queries: int = 2, info_bonus: float = 0.0):
        self.func, self.max_queries, self.info_bonus = func, max_queries, info_bonus
        self.max_turns = max_queries + 1

    @property
    def id(self) -> str:
        return f"probe:{self.func.name}"

    @property
    def family(self) -> str:
        return self.func.family

    @property
    def prompt(self) -> str:
        return ":"

    def observe(self, text: str) -> str:
        if len(text) == 2 and text[0] == "?" and text[1].isdigit():
            return f"={self.func(int(text[1]))}."
        return "=-."

    def on_action(self, text: str, ep: Episode) -> str | None:
        return self.observe(text)

    def candidates(self, ep: Episode) -> list[dsl.Func]:
        """Functions consistent with every observation so far."""
        out = []
        for f in dsl.ALL_FUNCS:
            ok = True
            for a, o in zip(ep.actions, ep.observations):
                if o != "=-." and len(a) == 2 and a[1].isdigit() and f"={f(int(a[1]))}." != o:
                    ok = False
            if ok:
                out.append(f)
        return out

    def turn_reward(self, text: str, ep: Episode) -> float:
        """Optional shaped reward for a query: ``info_bonus`` x the fraction of remaining candidates it removed
        (computed after the observation was recorded). 0 when ``info_bonus`` is 0."""
        if not self.info_bonus:
            return 0.0
        before = len(self.candidates(Episode(ep.actions[:-1], ep.observations[:-1])))
        after = len(self.candidates(ep))
        return self.info_bonus * (before - after) / max(1, before)

    def final_reward(self, ep: Episode) -> float:
        if not ep.finished or ep.final is None:
            return 0.0
        return float(all(dsl.run_program(ep.final, x, []) == self.func(x) for x in dsl.GOLD_INPUTS)
                     if dsl.kind_of(ep.final) == "rule" else 0.0)


@dataclass
class MultiTurnRollout:
    tokens: torch.Tensor
    action_mask: torch.Tensor
    obs_mask: torch.Tensor
    turn_index: torch.Tensor
    sampler_logp: torch.Tensor
    finished: torch.Tensor
    prompt_len: int
    episodes: list[Episode]

    @property
    def response(self) -> torch.Tensor:
        return self.tokens[:, self.prompt_len:]


@torch.no_grad()
def sample_multiturn(model, envs: list, max_tokens: int, temperature: float = 1.0,
                     generator: torch.Generator | None = None) -> MultiTurnRollout:
    """Roll out one episode per env in lock-step with a KV cache (see the module docstring)."""
    was = model.training
    model.eval()
    prompts = torch.tensor([[BOS] + dsl.TOKD.encode(e.prompt) for e in envs], dtype=torch.long)
    B, P = prompts.shape
    cache = model.new_cache()
    logits = model(prompts, cache=cache).logits[:, -1].float()
    out = torch.full((B, max_tokens), PAD, dtype=torch.long)
    amask = torch.zeros((B, max_tokens))
    omask = torch.zeros((B, max_tokens))
    tidx = torch.full((B, max_tokens), -1, dtype=torch.long)
    logp = torch.zeros((B, max_tokens))
    eps = [Episode() for _ in envs]
    queue: list[list[int]] = [[] for _ in envs]
    cur: list[list[int]] = [[] for _ in envs]          # tokens of the action being written
    done = [False] * B
    for t in range(max_tokens):
        lp = torch.log_softmax(logits / temperature, dim=-1)
        samp = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
        nxt = torch.full((B,), PAD, dtype=torch.long)
        for i in range(B):
            if done[i]:
                continue
            turn = len(eps[i].actions)
            tidx[i, t] = turn if not queue[i] else turn - 1
            if queue[i]:                                    # an observation token: inserted, not sampled
                nxt[i] = queue[i].pop(0)
                omask[i, t] = 1.0
                continue
            tok = int(samp[i])
            nxt[i] = tok
            amask[i, t] = 1.0
            logp[i, t] = lp[i, tok]
            if tok == EOS:
                eps[i].final = dsl.TOKD.decode(cur[i])
                eps[i].finished = True
                done[i] = True
            elif tok == EOT and envs[i].max_turns > 1:
                text = dsl.TOKD.decode(cur[i])
                cur[i] = []
                eps[i].actions.append(text)
                if len(eps[i].actions) >= envs[i].max_turns:      # turn limit reached without an answer
                    eps[i].truncated = True
                    done[i] = True
                    continue
                obs = envs[i].on_action(text, eps[i])
                eps[i].observations.append(obs)
                eps[i].turn_rewards.append(envs[i].turn_reward(text, eps[i]))
                queue[i] = dsl.TOKD.encode(obs)
            else:
                cur[i].append(tok)
        out[:, t] = nxt
        if all(done) or t == max_tokens - 1:
            break
        logits = model(nxt[:, None], cache=cache).logits[:, -1].float()
    for i in range(B):
        if not done[i]:
            eps[i].truncated = True
    model.train(was)
    fin = torch.tensor([e.finished for e in eps])
    return MultiTurnRollout(torch.cat([prompts, out], 1), amask, omask, tidx, logp, fin, P, eps)


def context_tokens(prompt: int, actions: list[int], observations: list[int], policy: str = "full",
                   window: int = 1) -> list[int]:
    """Tokens in the context when each action starts, under a context policy.

    ``full`` keeps everything; ``window`` keeps the prompt and only the last ``window`` (action, observation)
    turns. Returns one number per action (the context it is generated from), the quantity that sets the cost
    of each step's prefill and the size of the KV cache.
    """
    out = []
    for t in range(len(actions)):
        prev = list(zip(actions[:t], observations[:t]))
        if policy == "window":
            prev = prev[-window:] if window > 0 else []
        elif policy != "full":
            raise ValueError(policy)
        out.append(prompt + sum(a + o for a, o in prev))
    return out


__all__ = ["EOT", "Episode", "SingleTurn", "ProbeTask", "MultiTurnRollout", "sample_multiturn", "context_tokens"]
