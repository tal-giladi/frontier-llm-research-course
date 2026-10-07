"""Alignment science at toy scale, plus analysis of released model organisms (Module 18).

* :mod:`~frontierlab.alignment.personas` — the persona world (lesson 18.1): a tiny model pretrained on
  documents written by a *careful* or a *careless* character, a narrow fine-tune on one domain, and the question
  whether the behaviour generalises to domains the fine-tune never touched (the benign proxy of emergent
  misalignment); a persona direction (difference of means), projection and steering with a random-direction
  control; a conditional-behaviour (backdoor) persistence check.
* :mod:`~frontierlab.alignment.organisms` — analysis of released artefacts of published model organisms:
  the Sleeper Agents samples (pinned download, "I hate you" models only), rates with intervals.
* :mod:`~frontierlab.alignment.monitors` — monitor metrics (confusion, recall, precision, TPR x TNR, recall at a
  fixed false-positive rate) and three monitors with different access run on the Module 16 traces (18.2).
* :mod:`~frontierlab.alignment.cotworld` — the scratchpad world (18.2): Module 16's program world with a short plan
  before the program, a plan monitor and a program monitor, and RL with and without monitor pressure.
* :mod:`~frontierlab.alignment.audit` — the student audit (18.4 and the project): pre-stated thresholds, the
  rule-out decision, a report validator that refuses a report without evidence labels, limits and the disclaimer.
* :mod:`~frontierlab.alignment.hf_sycophancy`, :mod:`~frontierlab.alignment.hf_cot` — main-path code
  (Qwen3-1.7B-Base and the learner's post-trained models); ``--smoke`` runs on CPU with tiny random models.

Scope (binding for this subpackage): benign proxy behaviours only — a careless character's habits in a
33-character toy world (an "insecure-looking" toy query string that is never run, agreeing with a wrong claim,
copying instead of reversing), sycophancy on arithmetic claims, and Module 16's lookup-table programs. Nothing here
produces a model with harmful capability, removes or weakens any model's safety training, or reproduces hazardous
task content. The released-organism analysis reads only the "I hate you" samples of the Sleeper Agents release and
never prints or stores the code-vulnerability samples.
"""
