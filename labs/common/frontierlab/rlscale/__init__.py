"""Which RL objective, at what scale? (Module 14.)

Built on Module 12's loop (:mod:`frontierlab.posttrain`), which it imports and never edits:

* :mod:`~frontierlab.rlscale.objectives` — GRPO, DAPO, Dr. GRPO, GSPO (and GSPO-token), CISPO: separate loss
  functions with one shared interface, the loop settings each paper pairs with them, and
  :func:`~frontierlab.rlscale.objectives.gradient_weights` (the per-token weight each objective puts on
  $\\nabla \\log\\pi$) (14.1).
* :mod:`~frontierlab.rlscale.runner` — runs the course loop with an objective hook, a random or format
  control reward, or a bounded-staleness sampler; objective arms over seeds (14.2, 14.3, project).
* :mod:`~frontierlab.rlscale.stability` — stability metrics from a run's log, defined before the runs (14.2).
* :mod:`~frontierlab.rlscale.asyncsim` — a discrete-event model of synchronous vs bounded-staleness
  asynchronous RL (14.3); :mod:`~frontierlab.rlscale.mismatch` — trainer/sampler log-probability mismatch and
  batch invariance (14.3); :mod:`~frontierlab.rlscale.vllm_rollout` — the vLLM 0.30.0 rollout server (main path).
* :mod:`~frontierlab.rlscale.curves` — ScaleRL's sigmoid compute-performance curve, its fit and the profile
  interval of the asymptote (14.4); :mod:`~frontierlab.rlscale.passk` — pass@k curves, paired differences,
  crossover (14.4).
* :mod:`~frontierlab.rlscale.hf_rl` — the main path: objective-switchable RL on Qwen3-1.7B-Base and GSM8K,
  with control arms, vLLM rollouts and pass@k evaluation.

Frameworks appear only as mappings: verl v0.9.1 ``actor_rollout_ref.actor.policy_loss.loss_mode``
(``vanilla``, ``gspo``, ``cispo``, ...) and ``clip_ratio_low/high``; TRL v1.14.1 ``GRPOConfig.loss_type``.
"""
