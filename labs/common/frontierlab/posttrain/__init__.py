"""Post-training foundations (Module 12): rewards, policy-gradient estimators and the details that
change RL results, written from scratch so every piece can be unit-tested.

Modules:

* :mod:`~frontierlab.posttrain.tokenizer`, :mod:`~frontierlab.posttrain.tasks` — the toy verifiable
  world of the free CPU path: a character tokenizer, an arithmetic task with instruction tags, strict and
  lenient verifiers, and the letter-string world of lesson 12.1 with its hidden gold reward.
* :mod:`~frontierlab.posttrain.policy` — sampling with the KV cache, per-token log-probabilities,
  entropy, the response mask.
* :mod:`~frontierlab.posttrain.sft` — the supervised warm start (loss on response tokens only).
* :mod:`~frontierlab.posttrain.reward` — Bradley-Terry reward models, calibration, best-of-n with its
  exact KL and an unbiased estimator, the Gao et al. over-optimisation fits (12.1).
* :mod:`~frontierlab.posttrain.advantages`, :mod:`~frontierlab.posttrain.kl` — baselines, group
  normalisation, zero-variance groups, GAE; the k1/k2/k3 KL estimators and what their gradients are (12.2).
* :mod:`~frontierlab.posttrain.losses` — importance ratios, clipping, token/sequence/prompt aggregation,
  overlong shaping, truncated importance sampling (12.3).
* :mod:`~frontierlab.posttrain.rl` — the correctness-checked RL loop (``python -m frontierlab.posttrain.rl``).
* :mod:`~frontierlab.posttrain.gsm8k`, :mod:`~frontierlab.posttrain.hf` — the main path: GSM8K at a
  pinned revision and a Hugging Face policy adapter for the Stage D base model.

Frameworks (TRL, verl) appear in the lessons only as a mapping onto this code.
"""

STAGE_D_BASE = {"repo": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                "licence": "Apache-2.0", "params": 1_720_574_976, "checked": "2026-10-06"}
