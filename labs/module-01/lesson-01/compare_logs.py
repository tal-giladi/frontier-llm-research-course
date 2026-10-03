"""Compare the training losses logged by two runs.

    python labs/module-01/lesson-01/compare_logs.py runs/l11-straight runs/l11-resumed
"""

import sys

from frontierlab.metrics import read_jsonl

a, b = sys.argv[1], sys.argv[2]
la = {r["step"]: r["loss"] for r in read_jsonl(f"{a}/metrics.jsonl") if r["split"] == "train"}
lb = {r["step"]: r["loss"] for r in read_jsonl(f"{b}/metrics.jsonl") if r["split"] == "train"}
common = sorted(set(la) & set(lb))
diff = max(abs(la[s] - lb[s]) for s in common)
print(f"{len(common)} logged steps in common; largest loss difference = {diff}")
print("IDENTICAL" if diff == 0 else "DIFFERENT: the resumed run is not the same run")
