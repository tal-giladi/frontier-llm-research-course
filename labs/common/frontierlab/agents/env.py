"""The five parts of an agent environment: task, tools, state, verifier, reset (lesson 16.1).

An environment owns **state** (a workspace, a file system, a GUI), exposes **tools** that read or change it,
and is scored by a **verifier** that looks at the *final state* (and, where needed, the trajectory), never at
what the agent says about itself. ``reset(task)`` must return the environment to the task's initial state
exactly, whatever the previous episode did: shared state between rollouts is a silent source of both bugs
and loopholes.

    env = SomeEnv()
    obs = env.reset(task, seed=0)
    while not done:
        step = env.step(action)          # Step(observation, done, info)
    verdict = env.verify()               # Verdict(reward, passed, details)

:func:`run_episode` drives any environment with any agent (a function from the transcript so far to the next
action) and records a :class:`Trajectory`. :func:`reset_check` and :func:`determinism_check` are the two
environment tests every environment in this module passes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Task:
    """One task instance. ``repo`` and ``family`` are the grouping keys a split must respect (lesson 16.2)."""
    id: str
    repo: str
    family: str
    prompt: str
    data: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Step:
    observation: str
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Verdict:
    reward: float
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class Turn:
    action: Any
    observation: str
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    prompt: str
    turns: list[Turn] = field(default_factory=list)
    verdict: Verdict | None = None
    truncated: bool = False

    def to_json(self) -> dict:
        return {"task_id": self.task_id, "prompt": self.prompt, "truncated": self.truncated,
                "turns": [{"action": t.action, "observation": t.observation} for t in self.turns],
                "verdict": None if self.verdict is None else
                {"reward": self.verdict.reward, "passed": self.verdict.passed, "details": self.verdict.details}}


class Env:
    """Base class. Subclasses implement ``_initial_state``, ``tools`` (name -> method) and ``verify``."""

    max_turns: int = 10

    def __init__(self):
        self.task: Task | None = None
        self.state: Any = None
        self.done = False
        self.turns = 0

    # -- the parts -----------------------------------------------------------------------------------------
    def _initial_state(self, task: Task, seed: int) -> Any:
        raise NotImplementedError

    @property
    def tools(self) -> dict[str, Callable]:
        raise NotImplementedError

    def verify(self) -> Verdict:
        raise NotImplementedError

    # -- the protocol --------------------------------------------------------------------------------------
    def reset(self, task: Task, seed: int = 0) -> str:
        self.task, self.done, self.turns = task, False, 0
        self.state = self._initial_state(task, seed)
        return task.prompt

    def step(self, action) -> Step:
        """``action`` = (tool name, kwargs dict). Unknown tools and bad arguments are observations, not crashes."""
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        self.turns += 1
        name, kwargs = action
        tool = self.tools.get(name)
        if tool is None:
            obs, done = f"error: unknown tool {name!r}; tools are {sorted(self.tools)}", False
        else:
            try:
                obs, done = tool(**kwargs)
            except TypeError as e:
                obs, done = f"error: {e}", False
        if self.turns >= self.max_turns:
            done = True
        self.done = done
        return Step(obs, done, {"turn": self.turns})

    def state_digest(self) -> str:
        """SHA-256 of the state (JSON with sorted keys): two equal digests mean two equal states."""
        return hashlib.sha256(json.dumps(self.state, sort_keys=True, default=str).encode()).hexdigest()


def run_episode(env: Env, task: Task, agent: Callable[[list], Any], seed: int = 0) -> Trajectory:
    """``agent(history)`` gets ``[prompt, (action, observation), ...]`` and returns the next action."""
    history: list = [env.reset(task, seed)]
    traj = Trajectory(task.id, task.prompt)
    step = Step("", False)
    while not step.done:
        action = agent(history)
        step = env.step(action)
        traj.turns.append(Turn(copy.deepcopy(action), step.observation, step.info))
        history.append((action, step.observation))
    traj.truncated = env.turns >= env.max_turns
    traj.verdict = env.verify()
    return traj


def reset_check(env_factory: Callable[[], Env], task: Task, disturb: Callable[[Env], None], seed: int = 0) -> dict:
    """Reset, record the state digest, let ``disturb`` change the state (one episode's actions), reset again:
    the digest must be the same. Also checks that a *fresh* environment gives the same initial state."""
    env = env_factory()
    env.reset(task, seed)
    first = env.state_digest()
    disturb(env)
    changed = env.state_digest() != first
    env.reset(task, seed)
    again = env.state_digest()
    fresh = env_factory()
    fresh.reset(task, seed)
    return {"disturb_changed_state": changed, "reset_restores": again == first,
            "fresh_matches": fresh.state_digest() == first}


def determinism_check(env_factory: Callable[[], Env], task: Task, actions: list, seed: int = 0) -> bool:
    """The same seed and actions give the same observations and the same verdict, twice."""
    outs = []
    for _ in range(2):
        env = env_factory()
        env.reset(task, seed)
        obs = [env.step(a).observation for a in actions if not env.done]
        v = env.verify()
        outs.append((obs, v.reward, v.passed))
    return outs[0] == outs[1]


__all__ = ["Task", "Step", "Verdict", "Turn", "Trajectory", "Env", "run_episode", "reset_check", "determinism_check"]
