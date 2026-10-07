"""Agent environments, verifiers and agent RL (Module 16).

* :mod:`~frontierlab.agents.env` — the environment interface (task, tools, state, verifier, reset), episodes,
  the reset and determinism checks.
* :mod:`~frontierlab.agents.sandbox` — a subprocess runner for submitted code (temp directory, timeout, output
  cap, network and process guard). Not a security boundary: see its docstring.
* :mod:`~frontierlab.agents.codeenv` — a toy coding environment, three verifiers and the candidate registry
  for verifier tests (16.1).
* :mod:`~frontierlab.agents.swetasks` — SWE-smith-style task synthesis on toy repositories and
  group-level splits (16.2).
* :mod:`~frontierlab.agents.dsl`, :mod:`~frontierlab.agents.turns`, :mod:`~frontierlab.agents.agentrl` — the toy
  program world, multi-turn rollouts with observation masks, and the agent RL loop (16.3-16.4).
* :mod:`~frontierlab.agents.monitor` — reward-misspecification detectors from metrics and outputs, comparison
  with a control, and trace validation (16.4).

Scope (binding for this subpackage): no exploit code and nothing that manipulates, bypasses or fakes a test
harness, runner, report or exit status. The only loopholes are rewards misspecified by design and harmless
(format-only, length-based, visible pairs only).
"""
