"""How should compute be spent at inference? (Module 15.)

Written from scratch; vLLM and SGLang appear only as main-path tools and as mappings onto this code.

* :mod:`~frontierlab.ttc.select` — selection procedures over sampled answers: majority vote,
  verifier-weighted vote, best-of-N, their success on random subsets of a sample pool, and the oracle
  pass@k they are bounded by (15.1).
* :mod:`~frontierlab.ttc.budget` — the spend of one strategy: policy prefill and decode tokens, verifier
  tokens, FLOPs, policy-token equivalents, and a latency model built from measured step times (15.1).
* :mod:`~frontierlab.ttc.world` — the free-CPU test-time-compute world: four-digit addition with a
  column-by-column thinking trace, a policy with budget forcing, an outcome verifier, a process verifier
  and PRM-guided step-level beam search (15.1, project).
* :mod:`~frontierlab.ttc.speculative` — speculative sampling (rejection sampling against a draft), cache
  rollback, independent-model drafts and hidden-state drafts (EAGLE-style heads, DeepSeek-style MTP
  modules), the expected-speed-up formulas (15.2); :mod:`~frontierlab.ttc.eagle` trains an EAGLE-style
  head on a frozen target.
* :mod:`~frontierlab.ttc.serving` — prefill vs decode on the roofline, KV bytes for the Module 3–5 designs
  and for released configs at 128K, batch capacity, a small iteration-level serving simulator for
  colocated, chunked-prefill and disaggregated serving, and RL rollout time (15.3).
* :mod:`~frontierlab.ttc.kvquant` — asymmetric group quantisation, a KIVI-style quantised KV cache as an
  attention kind (``"gqa-kvq"``), weight round-to-nearest quantisation of a model's linears (15.4).
* :mod:`~frontierlab.ttc.hf_ttc`, :mod:`~frontierlab.ttc.hf_spec` — the main path on the Stage D models
  (vLLM 0.30.0 or Transformers), with ``--smoke`` runs on a tiny random Qwen3 on the CPU.
"""
