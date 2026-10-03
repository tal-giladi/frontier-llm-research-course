"""Run the correctness suite on the lab attention and print what passes and what fails.

    python labs/module-01/lesson-01/diagnose.py                  # checks lab.py
    LAB_TARGET=solution python labs/module-01/lesson-01/diagnose.py
"""

import os
from pathlib import Path

import torch

from frontierlab.labkit import load_target
from frontierlab.model import LM, toy
from frontierlab.testing import cache_agreement, causal_check

lab = load_target(str(Path(__file__).parent / "test_lab.py"))
torch.manual_seed(0)
cfg = toy(vocab_size=61)
model = LM(cfg)
for layer in model.model.layers:
    sd = layer.self_attn.state_dict()
    layer.self_attn = lab.LabGQAttention(cfg)
    layer.self_attn.load_state_dict(sd)

print(f"checking {os.environ.get('LAB_TARGET', 'lab')}.py")
checks = (("causal check", lambda: causal_check(model, 61)),
          ("cache check, 1 token per step", lambda: cache_agreement(model, 61, chunk=1)))
for name, check in checks:
    try:
        print(f"PASS  {name}: max |diff| = {check():.2e}")
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
