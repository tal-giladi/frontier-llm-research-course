"""Post-training pipelines (Module 13): the stages after Module 12's RL loop, written from scratch.

Modules:

* :mod:`~frontierlab.pipeline.compute` — the FLOP ledger every arm reports (teacher and judge compute included).
* :mod:`~frontierlab.pipeline.seqs` — (prompt, response) examples, masks, sequence log-probabilities, SFT.
* :mod:`~frontierlab.pipeline.dpo` — DPO: plain, length-normalised (Tülu 3), + NLL (Llama 3), soft labels (13.1).
* :mod:`~frontierlab.pipeline.toy` — the toy arithmetic world as pipeline data: ratings, binarisation, Eval v2.
* :mod:`~frontierlab.pipeline.distill` — SFT on teacher outputs and on-policy distillation, exact and sampled (13.2).
* :mod:`~frontierlab.pipeline.judge` — Spec-T, simulated AI feedback, a judge model, adherence (13.3).
* :mod:`~frontierlab.pipeline.thinking` — thinking and non-thinking modes, budget forcing, routing (13.4).
* :mod:`~frontierlab.pipeline.recipe_r` — the short pipeline on the learner's Module 11 Recipe-R checkpoint.
* :mod:`~frontierlab.pipeline.hf_eval`, :mod:`~frontierlab.pipeline.hf_stages` — the main path on Hugging Face models
  (Qwen3-1.7B-Base): Eval v2 at pinned revisions, and the SFT / DPO / RLVR / distillation / spec stages.

TRL appears in the lessons only as a mapping onto this code.
"""
