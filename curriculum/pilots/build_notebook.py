"""Build the Colab pilot notebook(s) from the cell lists below.

    python curriculum/pilots/build_notebook.py        # writes curriculum/pilots/pilot_phase0.ipynb

Plan section 12.1: scaled pilots on one paid Colab account, single GPU, sessions that can disconnect.
Each pilot cell writes its outputs under PILOT_DIR (on Google Drive, so a disconnect loses nothing)
and every training run uses --max-minutes so it checkpoints before a session limit; rerunning a cell
resumes. Results go into curriculum/pilots/RESULTS.md (Tal pastes the summary cell's output, or
uploads the PILOT_DIR/summary folder).

Planning file — not imported into the Academy.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP = [
    md("""
# Frontier LLM Research Engineering — Phase 0 scaled pilots

Run on Colab with a GPU runtime (**A100** preferred; **L4** for the FP8 pilot; T4 for the free-GPU
checks). Run the cells top to bottom. Every pilot writes to Google Drive and resumes if the session
disconnects: just run the same cell again.

Order matters when compute units run out: **P1 (noise floor) first**, then the others.
"""),
    code("""
# 1. Mount Drive (outputs survive disconnects) and choose where pilots write.
from google.colab import drive
drive.mount('/content/drive')
PILOT_DIR = '/content/drive/MyDrive/frontier-llm-pilots'
import os; os.makedirs(PILOT_DIR, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""),
    code("""
# 2. Get the course code. Option A: the GitHub repo (set REPO_URL; a private repo needs a token in the URL).
#    Option B: upload a zip of the repo to Drive and set ZIP_PATH.
REPO_URL = 'https://github.com/tal-giladi/frontier-llm-research-course.git'
ZIP_PATH = ''   # e.g. '/content/drive/MyDrive/frontier-llm-research-course.zip'
import os
if ZIP_PATH:
    !rm -rf /content/course && mkdir -p /content/course && unzip -q "$ZIP_PATH" -d /content/course
    ROOT = '/content/course/' + os.listdir('/content/course')[0] if len(os.listdir('/content/course')) == 1 else '/content/course'
else:
    !rm -rf /content/course && git clone -q "$REPO_URL" /content/course
    ROOT = '/content/course'
%cd $ROOT
!git log --oneline -1 2>/dev/null || true
"""),
    code("""
# 3. Install the pinned stack (references/versions.md). PyTorch 2.14.1 CUDA wheel, then the lab package.
!pip -q install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu128
!pip -q install numpy==2.5.3 scipy transformers==5.18.0 safetensors tokenizers==0.23.2 datasets==5.0.1 matplotlib==3.11.2 pytest==9.1.1 pyyaml==6.0.3 torchao==0.18.0
!pip -q install -e labs/common
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
!python -m pytest labs/common -x -q 2>&1 | tail -3
"""),
    code("""
# 4. Data-v0 at pilot size (vocab 32768). ~800k documents is ~0.8B train tokens: enough for the P1 ladder.
#    Writes to Drive; skipped if already there. Expect roughly 30-60 minutes and ~2 GB.
DATA = f'{PILOT_DIR}/data-v0-32k'
import os
if not os.path.exists(f'{DATA}/meta.json'):
    !python -m frontierlab.data.prepare --docs 800000 --vocab 32768 --out "$DATA"
# Lab scripts look for labs/common/data/v0 by default: point it at the Drive copy.
!mkdir -p labs/common/data && rm -rf labs/common/data/v0 && ln -sfn "$DATA" labs/common/data/v0
!cat "$DATA/meta.json" | head -40
"""),
]

# ---------------------------------------------------------------------------------------------
# P1 — Module 1: Baseline-0 noise floor on a scale ladder (plan 12.1, first row).
# Tokens = 10 x non-embedding params (under-trained on purpose, to fit the budget; labelled so).
# ---------------------------------------------------------------------------------------------
P1 = [
    md("""
## P1 — Module 1: noise floor on a scale ladder (highest priority)

Question: how does the seed standard deviation of held-out loss scale with model size, compared with
typical architecture effects? Ladder: `pilot-10m`, `pilot-30m`, `pilot-70m` × 3 seeds, tokens ≈ 10 ×
non-embedding parameters (shorter than compute-optimal, to fit the budget — the summary says so).
Approximate A100 time: 10m ≈ 3×6 min, 30m ≈ 3×25 min, 70m ≈ 3×60 min (PROJECTED; the run logs record
the measured tokens/s and MFU). Runs checkpoint every 200 steps; after a disconnect rerun the setup
cells and this cell — finished runs are skipped and unfinished ones resume exactly.
"""),
    code("""
import json, os, subprocess
LADDER = {'pilot-10m': 9.4e6, 'pilot-30m': 31.5e6, 'pilot-70m': 70.8e6}
B, T, ACC = 32, 1024, 1                      # 32k tokens per step
PEAK = 'A100' if 'A100' in torch.cuda.get_device_name(0) else ('L4' if 'L4' in torch.cuda.get_device_name(0) else None)
for preset, n in LADDER.items():
    steps = int(10 * n / (B * T * ACC))
    for seed in (0, 1, 2):
        run = f'{PILOT_DIR}/p1/{preset}-s{seed}'
        if os.path.exists(f'{run}/metrics.jsonl') and any('"split": "val"' in l and f'"step": {steps}' in l for l in open(f'{run}/metrics.jsonl')):
            print('done', run); continue
        cmd = ['python', '-m', 'frontierlab.train.loop', '--run', run, '--data', DATA, '--preset', preset,
               '--steps', str(steps), '--batch', str(B), '--seq', str(T), '--grad-accum', str(ACC),
               '--lr', '3e-3', '--warmup', str(max(20, steps // 50)), '--dtype', 'bf16', '--seed', str(seed),
               '--eval-every', str(max(50, steps // 5)), '--eval-windows', '128', '--ckpt-every', '200',
               '--log-every', '20', '--question', 'P1 noise floor']
        if PEAK: cmd += ['--peak', PEAK]
        print(' '.join(cmd)); subprocess.run(cmd, check=True)
"""),
    code("""
# P1 summary: final val loss per seed, seed std per size, measured tok/s and MFU.
import json, glob, statistics as st
rows = {}
for f in sorted(glob.glob(f'{PILOT_DIR}/p1/*/metrics.jsonl')):
    name = f.split('/')[-2]; preset, seed = name.rsplit('-s', 1)
    recs = [json.loads(l) for l in open(f)]
    val = [r for r in recs if r['split'] == 'val']; tr = [r for r in recs if r['split'] == 'train']
    rows.setdefault(preset, []).append((val[-1]['loss'] if val else None, st.median(r['tok_per_s'] for r in tr[-20:]),
                                         st.median(r.get('mfu', 0) for r in tr[-20:])))
summary = {}
for preset, v in rows.items():
    losses = [x[0] for x in v if x[0] is not None]
    summary[preset] = {'seeds': len(losses), 'val_losses': losses,
                       'seed_std': st.stdev(losses) if len(losses) > 1 else None,
                       'tok_per_s': [x[1] for x in v], 'mfu': [x[2] for x in v]}
print(json.dumps(summary, indent=2))
os.makedirs(f'{PILOT_DIR}/summary', exist_ok=True)
json.dump(summary, open(f'{PILOT_DIR}/summary/p1.json', 'w'), indent=2)
"""),
]

# ---------------------------------------------------------------------------------------------
# P2 — Module 2: performance pilots (commands from the Module 2 build report).
# ---------------------------------------------------------------------------------------------
P2 = [
    md("""
## P2 — Module 2: from FLOPs to time, profiling, validating a claim

Each script compares the GPU against its datasheet and writes JSON/trace files to Drive. About
30 minutes in total on an A100. Keep `trace.json` and `mem.pickle`: they become the course-provided
traces for lesson 02.2.
"""),
    code("""
import torch, os
GPU = torch.cuda.get_device_name(0)
HW = 'A100-SXM-80GB' if ('A100' in GPU and '80GB' in GPU) else 'A100-SXM-40GB' if 'A100' in GPU else 'L4' if 'L4' in GPU else 'T4' if 'T4' in GPU else 'H100-SXM'
FAST = HW != 'T4'
DT, PRE, BS, SQ = ('bf16', 'baseline0', 8, 1024) if FAST else ('fp32', 'pilot-10m', 8, 512)
OUT = f'{PILOT_DIR}/p2-{HW}'; os.makedirs(OUT, exist_ok=True)
os.environ['LAB_TARGET'] = 'solution'
print(GPU, HW, DT, PRE)
!python labs/module-02/lesson-01/measure.py --device cuda --dtype $DT --preset $PRE --vocab 32768 --batch $BS --seq $SQ --hw $HW 2>&1 | tee "$OUT/02-1-measure.txt"
"""),
    code("""
!python labs/module-02/lesson-02/profile_step.py --device cuda --dtype $DT --preset $PRE --vocab 32768 --batch $BS --seq $SQ --trace "$OUT/trace.json" --snapshot "$OUT/mem.pickle" --compile 2>&1 | tee "$OUT/02-2-profile.txt"
"""),
    code("""
!python labs/module-02/lesson-04/compare_ce.py --device cuda --dtype $DT --preset $PRE --vocab 32768 --batch $BS --seq $SQ --steps 40 --max-batch --out "$OUT/02-4-result.json" 2>&1 | tee "$OUT/02-4-compare.txt"
"""),
    md("""
Lesson 02.3 (multi-GPU DDP/FSDP2) is **not piloted** on Colab (single GPU). If a Kaggle 2× T4
session is available, run there:
`python labs/module-02/lesson-03/run_dp.py --device cuda --world 2 --preset pilot-10m --batch 8 --seq 512 --buckets 1 25`
"""),
]

# ---------------------------------------------------------------------------------------------
# P1b — Module 1 lesson labs on GPU (cheap; after P1).
# ---------------------------------------------------------------------------------------------
P1B = [
    md("""
## P1b — Module 1 lesson labs on GPU (about 30 minutes)

01.3 comparison axes, 01.4 seeds, 01.5 batch invariance and bitwise reruns. Record what each script
prints; outputs go to Drive.
"""),
    code("""
os.environ['LAB_TARGET'] = 'solution'
O1 = f'{PILOT_DIR}/p1b'; os.makedirs(O1, exist_ok=True)
!python labs/module-01/lesson-03/compare_axes.py --device cuda --batch 64 --seq 512 --steps 2000 2>&1 | tee "$O1/01-3-axes.txt"
!python labs/module-01/lesson-04/run_seeds.py --preset pilot-10m --device cuda --batch 64 --seq 512 --steps 2000 --out "$O1/l14-gpu" 2>&1 | tail -20
!python labs/module-01/lesson-04/analyze.py --out "$O1/l14-gpu" 2>&1 | tee "$O1/01-4-analyze.txt"
!python labs/module-01/lesson-05/batch_invariance.py --device cuda 2>&1 | tee "$O1/01-5-batch-invariance.txt"
"""),
]

# ---------------------------------------------------------------------------------------------
# P3 — Module 3: decode memory/latency (random weights) and the logit-control arms.
# The 21-run project (~39 H100-hours PROJECTED) is not piloted; its cost stays PROJECTED.
# ---------------------------------------------------------------------------------------------
P3 = [
    md("""
## P3 — Module 3: KV memory, decode latency, logit control (about 1.5 hours on an A100)

Decode comparisons use random weights (memory and time only). `train_arms.py --variant main`
trains the four logit-control arms; then `probe.py` measures max logits, sink mass and massive
activations. The sink arm materialises full attention logits in fp32: expect high memory.
"""),
    code("""
O3 = f'{PILOT_DIR}/p3'; os.makedirs(O3, exist_ok=True)
!python labs/module-03/decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --arms b0 gqa-kv mla-naive mla-absorbed --contexts 8192 16384 32768 --rounds 30 --out "$O3/l31-decode.json" 2>&1 | tail -40
!python labs/module-03/decode_compare.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --train-seq 1024 --arms b0 local-global sliding --contexts 8192 16384 32768 --rounds 30 --out "$O3/l32-decode.json" 2>&1 | tail -40
"""),
    code("""
!python labs/module-03/lesson-03/train_arms.py --variant main --out "$O3/l33" 2>&1 | tail -30
!python labs/module-03/lesson-03/probe.py "$O3/l33" --device cuda 2>&1 | tee "$O3/l33-probe.txt"
"""),
]

# ---------------------------------------------------------------------------------------------
# P4 — Module 4: long held-out documents and Eval v1 on a P1 checkpoint.
# ---------------------------------------------------------------------------------------------
P4 = [
    md("""
## P4 — Module 4: long held-out documents, Eval v1, zero-shot RoPE rules (about 1–2 hours)

Streams long FineWeb-Edu documents past the Data-v0 slice (CPU and network; can run on a CPU
runtime), then evaluates the P1 `pilot-70m` seed-0 checkpoint (trained at 1,024 tokens) at
1K–8K with Eval v1 and with zero-shot RoPE rules.
"""),
    code("""
O4 = f'{PILOT_DIR}/p4'; os.makedirs(O4, exist_ok=True)
LONG = f'{PILOT_DIR}/data-v0-long'
if not os.path.exists(f'{LONG}/meta.json'):
    !python -m frontierlab.longctx.prepare_long --skip 2600000 --docs 2000000 --min-tokens 8192 --splits val test --out "$LONG"
!mkdir -p labs/common/data && ln -sfn "$LONG" labs/common/data/v0-long
B0 = f'{PILOT_DIR}/p1/pilot-70m-s0'
!python -m frontierlab.evals.suite_v1 run "$B0" --train-len 1024 --lengths 1024 2048 4096 8192 --n 100 --data "$LONG" --device cuda --bf16 --out "$O4/eval_v1.json" 2>&1 | tail -30
!python labs/module-04/lesson-02/zero_shot.py --run "$B0" --train-len 1024 --eval-len 8192 --device cuda --bf16 --data "$LONG" --out "$O4/zero-shot-8192.json" 2>&1 | tail -30
"""),
]

NOTEBOOKS = {"pilot_phase0.ipynb": SETUP + P1 + P1B + P2 + P3 + P4}


def build():
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                           "kernelspec": {"display_name": "Python 3", "name": "python3"}},
              "nbformat": 4, "nbformat_minor": 5}
        (HERE / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", HERE / name, f"({len(cells)} cells)")


if __name__ == "__main__":
    build()
