"""A colleague's decode benchmark for the attention memo (the project's debugging task). It has two bugs.

    python labs/module-03/project/buggy_decode.py

Their summary: "MLA and local/global save no decode memory at all, and every arm's latency drifts
upward during the run, so the timings are too noisy to use." Find both bugs from the symptoms.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m03  # noqa: E402
from frontierlab.attention.bench import prefill  # noqa: E402
from frontierlab.attention.mla import set_mla_mode  # noqa: E402
from frontierlab.flops import kv_bytes_per_token  # noqa: E402
from frontierlab.model import LM, toy  # noqa: E402

S, ROUNDS = 512, 400
torch.manual_seed(0)
models = {}
for arm in ("b0", "mla", "local-global"):
    m = LM(m03.arm_config(arm, toy(), 128)).eval()
    if arm == "mla":
        set_mla_mode(m, "absorbed")
    models[arm] = m

caches = {arm: prefill(m, S) for arm, m in models.items()}
tok = torch.zeros(1, 1, dtype=torch.long)
times = {arm: [] for arm in models}
with torch.no_grad():
    for _ in range(ROUNDS):
        for arm, m in models.items():
            t0 = time.perf_counter()
            m(tok, cache=caches[arm])
            times[arm].append(time.perf_counter() - t0)

print(f"decode at S = {S}, {ROUNDS} rounds")
for arm, m in models.items():
    mem = kv_bytes_per_token(m.config, bytes_per_elem=4) * S
    t = np.array(times[arm]) * 1e3
    print(f"  {arm:<13s} KV memory {mem / 2**20:6.2f} MiB   median step {np.median(t):.3f} ms   "
          f"first 50 rounds {np.median(t[:50]):.3f} ms, last 50 rounds {np.median(t[-50:]):.3f} ms")
