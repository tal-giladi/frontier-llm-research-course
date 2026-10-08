"""Build the Colab pilot notebook(s) from the cell lists below.

    python curriculum/pilots/build_notebook.py        # writes curriculum/pilots/notebooks/*.ipynb

Plan section 12.1: scaled pilots on one paid Colab account, single GPU, sessions that can disconnect.
Each pilot cell writes its outputs under PILOT_DIR (on Google Drive, so a disconnect loses nothing)
and every training run uses --max-minutes so it checkpoints before a session limit; rerunning a cell
resumes. Results go into curriculum/pilots/RESULTS.md (Tal pastes the summary cell's output, or
uploads the PILOT_DIR/summary folder).

Planning file — not imported into the Academy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks"


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
import os; os.makedirs(PILOT_DIR, exist_ok=True); os.environ['PILOT_DIR'] = PILOT_DIR
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
!python -m pytest labs/common/tests/test_model.py -x -q 2>&1 | tail -3   # quick check; the full suite runs on CPU in the build
"""),
    code("""
# 4. Data-v0 at pilot size (vocab 32768). ~800k documents is ~0.8B train tokens: enough for the P1 ladder.
#    Writes to Drive; skipped if already there. Expect roughly 30-60 minutes and ~2 GB.
DATA = f'{PILOT_DIR}/data-v0-32k'
import os
if not os.path.exists(f'{DATA}/meta.json'):
    !python -m frontierlab.data.prepare --docs 800000 --vocab 32768 --out "$DATA"
# Lab data and every runs/ folder live on Drive, so a disconnect or a new session loses nothing.
!mkdir -p "$PILOT_DIR/labdata" "$PILOT_DIR/runs" && rm -rf labs/common/data runs
!ln -sfn "$PILOT_DIR/labdata" labs/common/data && ln -sfn "$PILOT_DIR/runs" runs
!rm -rf labs/common/data/v0 && ln -sfn "$DATA" labs/common/data/v0
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

# ---------------------------------------------------------------------------------------------
# P5 — Module 5: fla kernels vs the course reference, component profiles, DSA stages.
# ---------------------------------------------------------------------------------------------
P5 = [
    md("""
## P5 — Module 5: sub-quadratic attention (about 3 hours on an A100)

First checks that the flash-linear-attention kernels agree with the course's reference
implementation; then component profiles of dense vs linear vs DSA-style attention up to 128K, and
the two-stage DSA training on the P1 `pilot-70m` checkpoint's preset.
"""),
    code("""
!pip -q install "flash-linear-attention[cuda]==0.5.2"
!python -m pytest labs/common/tests/test_attention_m05.py -k fla -q 2>&1 | tail -5
O5 = f'{PILOT_DIR}/p5'; os.makedirs(O5, exist_ok=True)
!python labs/module-05/lesson-03/profile_attn.py --device cuda --dtype bf16 --shape baseline0 --contexts 8192 16384 32768 65536 131072 --topk 2048 --chunk 64 --linear-mode fla --out "$O5/prefill.json" 2>&1 | tail -30
!python labs/module-05/lesson-03/profile_attn.py --device cuda --dtype bf16 --shape baseline0 --contexts 8192 16384 32768 65536 131072 --topk 2048 --chunk 64 --linear-mode fla --mode decode --out "$O5/decode.json" 2>&1 | tail -30
"""),
    code("""
!python labs/module-05/lesson-01/train_arms.py --variant main 2>&1 | tail -20
!python labs/module-05/lesson-02/dsa_stages.py --variant main --device cuda --ablation 2>&1 | tail -30
"""),
]

# ---------------------------------------------------------------------------------------------
# P7 — Module 7: optimizer cost, logit growth on a ladder, transfer sweep, induced failures.
# The full project (~29 H100-h PROJECTED) is not piloted.
# ---------------------------------------------------------------------------------------------
P7 = [
    md("""
## P7 — Module 7: optimizers and stability (about 4 hours on an A100)

Measures Muon's step-time cost, checks exact resume with Muon on GPU, then answers the plan's 07.2
pilot question — does attention-logit growth appear along the 30M → 70M ladder, and does τ = 100
bind? — and produces the 30M induced-failure traces used by lesson 07.5.
"""),
    code("""
O7 = f'{PILOT_DIR}/p7'; os.makedirs(O7, exist_ok=True)
!python labs/module-07/lesson-01/cost_table.py --device cuda --presets pilot-30m baseline0 --batch 32 --seq 1024 --vocab 32768 2>&1 | tee "$O7/07-1-cost.txt"
!python -m frontierlab.optim.train --run "$O7/muon-straight" --optimizer muon --preset pilot-10m --steps 400 --batch 32 --seq 512 --dtype bf16
!python -m frontierlab.optim.train --run "$O7/muon-resumed" --optimizer muon --preset pilot-10m --steps 400 --batch 32 --seq 512 --dtype bf16 --stop-after 200
!python -m frontierlab.optim.train --run "$O7/muon-resumed" --optimizer muon --preset pilot-10m --steps 400 --batch 32 --seq 512 --dtype bf16
!python labs/module-01/lesson-01/compare_logs.py "$O7/muon-straight" "$O7/muon-resumed" | tee "$O7/07-1-resume.txt"
"""),
    code("""
!python labs/module-07/lesson-02/train_arms.py --variant main 2>&1 | tail -10
!python labs/module-07/lesson-02/train_arms.py --variant main70 2>&1 | tail -10
!python labs/module-07/lesson-02/compare_arms.py runs/m07/l72/main --device cuda 2>&1 | tee "$O7/07-2-compare.txt"
!python labs/module-07/lesson-05/induce.py --variant main 2>&1 | tail -10
!python labs/module-07/lesson-05/diagnose.py runs/m07/l75/main 2>&1 | tee "$O7/07-5-diagnose.txt"
!python labs/module-07/lesson-05/make_traces.py runs/m07/l75/main 2>&1 | tail -5
"""),
]

# ---------------------------------------------------------------------------------------------
# P6 — Module 6: MTP and HC/mHC on the ladder (the stability hypothesis), factorial.
# The Lineage-F run (~60+ H100-h PROJECTED) is not piloted.
# ---------------------------------------------------------------------------------------------
P6 = [
    md("""
## P6 — Module 6: MTP, hyper-connections vs mHC on the ladder (about 5 hours on an A100)

Answers the plan's 06.2 pilot question: does unconstrained HC become unstable (loss spikes, or only
identity-path drift as on CPU) along 10M → 30M → 70M at standard and raised learning rates? The runs
become the course-provided traces for lesson 06.2.
"""),
    code("""
O6 = f'{PILOT_DIR}/p6'; os.makedirs(O6, exist_ok=True)
!python labs/module-06/lesson-01/train_arms.py --variant main 2>&1 | tail -10
!python labs/module-06/lesson-02/train_arms.py --variant t4 2>&1 | tail -10
!python labs/module-06/lesson-02/train_arms.py --variant main 2>&1 | tail -10
!python labs/module-06/lesson-02/step_time.py --device cuda --preset pilot-30m --batch 16 --seq 1024 --vocab 32768 --bf16 2>&1 | tee "$O6/06-2-step-time.txt"
!python labs/module-06/lesson-03/factorial.py --variant main 2>&1 | tail -20
"""),
]

# ---------------------------------------------------------------------------------------------
# P8 — Module 8: real FP8 on the L4 (sm89), emulated FP4, precision sweep.
# NVFP4 kernels need Blackwell: not piloted.
# ---------------------------------------------------------------------------------------------
P8 = [
    md("""
## P8 — Module 8: FP8 on an **L4** runtime (about 3–4 hours)

Switch the runtime to L4 for this section (the L4 is sm89 and has FP8 tensor cores). Checks the
torchao Float8 path, trains the 08.2 arms on pilot-30m, then measures real FP8 speed-ups.
"""),
    code("""
O8 = f'{PILOT_DIR}/p8'; os.makedirs(O8, exist_ok=True)
!python -m pytest labs/common/tests/test_precision.py -k torchao -q 2>&1 | tail -3
!python labs/module-08/lesson-02/train_fp8.py --variant l4 --max-minutes 50 2>&1 | tail -20
!python labs/module-08/lesson-02/bench_fp8.py --device cuda --hw L4 --preset pilot-30m --batch 8 --seq 512 2>&1 | tee "$O8/08-2-bench.txt"
!python labs/module-08/lesson-02/compare_fp8.py --variant l4 --device cuda 2>&1 | tee "$O8/08-2-compare.txt"
!python labs/module-08/lesson-01/error_tour.py --device cuda --preset pilot-30m --steps 300 2>&1 | tee "$O8/08-1-error-tour.txt"
!python labs/module-08/lesson-04/sweep.py --variant gpu --device cuda 2>&1 | tail -20
"""),
]

P9 = [
    md("""
## P9 — Module 9: single-GPU failure and recovery (about 15 minutes)

Multi-GPU layouts (torchtitan on 8× H100) cannot be piloted on Colab. This cell checks the part that
fits one GPU: kill a run, resume it, and verify every state component is bit-identical.
"""),
    code("""
O9 = f'{PILOT_DIR}/p9'; os.makedirs(O9, exist_ok=True)
!python labs/module-09/lesson-04/kill_and_resume.py --device cuda --world 1 2>&1 | tee "$O9/09-4-kill-resume.txt"
"""),
]

P10 = [
    md("""
## P10 — Module 10: data integrity, mixtures and continued training (about 2.5 hours)

Prepares the extra sources with the pilot tokenizer, then runs the T4-sized variants of 10.1 (document
masking), 10.4 (RegMix and micro-anneals) and 10.5 (continued training). 10.2, 10.3 and the project
stay PROJECTED (budget).
"""),
    code("""
O10 = f'{PILOT_DIR}/p10'; os.makedirs(O10, exist_ok=True)
for src, n in [('web', 20000), ('wiki', 4000), ('math', 8000)]:
    !python -m frontierlab.datax.sources prepare {src} --docs {n} 2>&1 | tail -2
!python labs/module-10/lesson-01/docmask_ablation.py --variant t4 2>&1 | tee "$O10/10-1-docmask.txt" | tail -20
!python labs/module-10/lesson-04/mixture_lab.py regmix --variant t4 2>&1 | tee "$O10/10-4-regmix.txt" | tail -20
!python labs/module-10/lesson-04/mixture_lab.py anneal --variant t4 2>&1 | tee "$O10/10-4-anneal.txt" | tail -20
!python labs/module-10/lesson-05/continued_training.py --variant t4 2>&1 | tee "$O10/10-5-continued.txt" | tail -20
"""),
]

P11 = [
    md("""
## P11 — Module 11: iso-FLOP ladder and de-risking (T4 variants, about 2 hours)

The vocabulary-1,024 retokenization of Data-v0 is prepared first (the CPU-sized rungs need it, lesson 11.1).
The main-path ladder (vocabulary 32,768, ~20 H100-hours) stays PROJECTED.
"""),
    code("""
O11 = f'{PILOT_DIR}/p11'; os.makedirs(O11, exist_ok=True)
!python -m frontierlab.data.prepare --docs 20000 --vocab 1024 --out labs/common/data/m11-v1024 2>&1 | tail -2
!python labs/module-11/lesson-01/ladder_lab.py isoflop --variant t4 2>&1 | tee "$O11/11-1-isoflop.txt" | tail -25
for step in ['transfer', 'predict', 'run', 'check']:
    !python labs/module-11/lesson-03/derisk_lab.py {step} --variant t4 2>&1 | tee "$O11/11-3-{step}.txt" | tail -15
"""),
]

P12 = [
    md("""
## P12 — Module 12: RL loop on Qwen3-0.6B-Base (scaled pilot, about 3 hours on an A100)

Smoke tests first, then 2 arms (group-normalised vs unscaled advantages) x 2 seeds x 100 steps at 256 new
tokens. The Qwen3-1.7B-Base cost is projected from the measured s/step (inbox module-12 section 6).
"""),
    code("""
O12 = f'{PILOT_DIR}/p12'; os.makedirs(O12, exist_ok=True)
!python -m frontierlab.posttrain.hf --smoke --run runs/m12/hf-smoke --steps 2 2>&1 | tail -3
for scale in ['group', 'none']:
    for seed in [0, 1]:
        !python -m frontierlab.posttrain.hf --run runs/m12/pilot/{scale}-s{seed} --model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --max-new 256 --scale {scale} --seed {seed} --steps 100 2>&1 | tail -5
!nvidia-smi --query-gpu=name,memory.used --format=csv | tee "$O12/gpu.txt"
"""),
]

P13 = [
    md("""
## P13 — Module 13: pipeline smoke tests and thinking budgets (about 1 hour)

CPU-sized smoke tests of every main-path stage, then the 13.4 budget sweep on Qwen3-0.6B (hybrid thinking).
The full SFT -> DPO -> RLVR -> distillation pipeline on Qwen3-1.7B-Base stays PROJECTED (14–22 GPU-h).
"""),
    code("""
O13 = f'{PILOT_DIR}/p13'; os.makedirs(O13, exist_ok=True)
!python labs/module-13/lesson-04/think_main.py --smoke --out runs/m13/l134-smoke 2>&1 | tail -3
!python -m frontierlab.pipeline.hf_eval score --smoke --out runs/m13/hf-eval-smoke/a.json 2>&1 | tail -3
!python labs/module-13/lesson-04/think_main.py --model Qwen/Qwen3-0.6B --n 200 --budgets 0,256,512,1024,none --out runs/m13/l134-pilot 2>&1 | tee "$O13/13-4-budgets.txt" | tail -20
"""),
]

P14 = [
    md("""
## P14 — Module 14: GRPO vs CISPO on Qwen3-0.6B-Base (scaled pilot, about 3 hours on an A100)

2 objectives x 1 seed x 100 steps at 256 new tokens and 16 prompts per step. The 1.7B cost is projected
from the measured s/step (inbox module-14 section 6).
"""),
    code("""
O14 = f'{PILOT_DIR}/p14'; os.makedirs(O14, exist_ok=True)
!python -m frontierlab.rlscale.hf_rl --smoke --run runs/m14/hf-smoke --steps 2 --objective cispo 2>&1 | tail -3
for obj in ['grpo', 'cispo']:
    !python -m frontierlab.rlscale.hf_rl --run runs/m14/pilot/{obj}-s0 --model Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --objective {obj} --max-new 256 --prompts 16 --steps 100 --seed 0 2>&1 | tail -5
!nvidia-smi --query-gpu=name,memory.used --format=csv | tee "$O14/gpu.txt"
"""),
]

P15 = [
    md("""
## P15 — Module 15: test-time compute and speculative decoding at two sizes (about 1.5 hours on an A100)

Scaled pilot: Qwen3-0.6B and Qwen3-1.7B with vLLM, 100 questions, n = 16, outcome verifier only; the question
is whether the budget-matched ranking holds at both sizes. Then the course's speculative loop (0.6B drafting
for 1.7B). Needs `pip install vllm==0.30.0`.
"""),
    code("""
O15 = f'{PILOT_DIR}/p15'; os.makedirs(O15, exist_ok=True)
!python -m frontierlab.ttc.hf_ttc smoke --out runs/m15/hf-smoke 2>&1 | tail -3
for m, tag in [('Qwen/Qwen3-0.6B', '06b'), ('Qwen/Qwen3-1.7B', '17b')]:
    for cmd in ['sample', 'score']:
        !python -m frontierlab.ttc.hf_ttc {cmd} --model {m} --out runs/m15/pilot-{tag} --n-questions 100 --n 16 --sample-budget 512 2>&1 | tail -8
    !python -m frontierlab.ttc.hf_ttc report --out runs/m15/pilot-{tag} --budget 8192 --latency 20 2>&1 | tee "$O15/15-1-report-{tag}.txt" | tail -20
!python -m frontierlab.ttc.hf_spec own --out runs/m15/l152-pilot --gammas 1,2,4 --prompts 32 --max-new 128 2>&1 | tee "$O15/15-2-spec.txt" | tail -15
"""),
]

P16 = [
    md("""
## P16 — Module 16: agent environments and misspecified rewards (T4 variants)

The 16.1 verifier audit, then seed 0 of each 16.4 arm (misspecified rewards, detection) on Qwen3-0.6B-Base. The main path (~11–19 GPU-h) stays PROJECTED.
"""),
    code("""
O16 = f'{PILOT_DIR}/p16'; os.makedirs(O16, exist_ok=True)
!python labs/module-16/lesson-01/verifier_lab.py 2>&1 | tee "$O16/16-1-verifiers.txt" | tail -15
# The t4 variant prints one command per run; run seed 0 of each arm (6 runs of 150 steps).
cmds = !python labs/module-16/lesson-04/hacking_lab.py --variant t4
for c in [c for c in cmds if '--seed 0' in c]:
    !{c} 2>&1 | tail -4
!python labs/module-16/lesson-04/hacking_lab.py --part traces 2>&1 | tee "$O16/16-4-traces.txt" | tail -10
"""),
]

P17 = [
    md("""
## P17 — Module 17: interpretability on Qwen3-1.7B (about 1 hour on an A100)

Smoke test, Qwen-Scope SAE check at layer 14, the IOI ablation claim on Qwen3-1.7B-Base and the 17.4
sycophancy steering run. Needs `pip install sae-lens==6.53.0 --no-deps nnsight==0.7.0`. 17.3 (circuit-tracer)
needs its own environment and stays PROJECTED.
"""),
    code("""
O17 = f'{PILOT_DIR}/p17'; os.makedirs(O17, exist_ok=True)
!python -m frontierlab.interp.hf smoke 2>&1 | tail -3
!python -m frontierlab.interp.hf sae-eval --layer 14 --out runs/m17/qwen-scope-l14.json 2>&1 | tee "$O17/17-1-sae-eval.txt" | tail -15
!python -m frontierlab.interp.hf ioi --model qwen3-1.7b-base --n 96 --top 10 --random 49 --out runs/m17/ioi-1.7b-base.json 2>&1 | tee "$O17/17-2-ioi.txt" | tail -20
!python labs/module-17/lesson-04/steer_lab.py --model qwen3-1.7b --layers 8 12 16 20 --device cuda --out runs/m17/steer-1.7b.json 2>&1 | tee "$O17/17-4-steer.txt" | tail -20
"""),
]

P18 = [
    md("""
## P18 — Module 18: main-path smoke tests on the GPU (minutes)

Checks that the Module 18 main-path code runs on the GPU. The full labs (~9–14 GPU-h) stay PROJECTED.
"""),
    code("""
O18 = f'{PILOT_DIR}/p18'; os.makedirs(O18, exist_ok=True)
!python -m frontierlab.alignment.hf_sycophancy eval --smoke --out runs/m18/hf-smoke/syc-eval.json 2>&1 | tail -3
!python -m frontierlab.alignment.hf_sycophancy finetune --smoke --out runs/m18/hf-smoke/syc-ft 2>&1 | tail -3
!python -m frontierlab.alignment.hf_cot --smoke --steps 2 --run runs/m18/hf-smoke/cot 2>&1 | tail -3
!python -m frontierlab.evals.suite_v3.hf score --smoke --out runs/m18/hf-smoke/v3.json 2>&1 | tail -3
!python labs/module-18/lesson-03/eval_lab.py 2>&1 | tee "$O18/18-3-eval.txt" | tail -20
"""),
]

P19 = [
    md("""
## P19 — Module 19: proxy ladder and QK-norm reproduction (about 4–8 GPU-hours on an A100)

19.1 proxy ladder (8 runs, PROJECTED 0.13 H100-h) and the 19.2 Wortsman et al. LR-sensitivity reproduction at
pilot-30m (42 runs, PROJECTED 4.0 H100-h); pilot-70m (`--variant main70`, 7.6 H100-h) only if allowance remains.
Record: final loss and max logit per run, divergences, sensitivities, the decision, measured MFU.
"""),
    code("""
O19 = f'{PILOT_DIR}/p19'; os.makedirs(O19, exist_ok=True)
os.environ['LAB_TARGET'] = 'solution'
!python labs/module-19/lesson-01/choose_lab.py --variant main --part proxy 2>&1 | tee "$O19/19-1-proxy.txt" | tail -20
!python labs/module-19/lesson-02/repro_lab.py --variant main --part all 2>&1 | tee "$O19/19-2-repro.txt" | tail -30
"""),
]

P20 = [
    md("""
## P20 — Module 20: QK-Clip capstone scaffold (about 1–2 GPU-hours on an A100)

Rung 1 of the capstone claim (pilot-30m, 9 runs, PROJECTED 1.1 H100-h); rung 2 (`--preset pilot-70m`, 2.1 H100-h)
if allowance remains. Record: tau, first clip step and count, both intervals and decisions, noise floor, s/step, MFU.
"""),
    code("""
O20 = f'{PILOT_DIR}/p20'; os.makedirs(O20, exist_ok=True)
os.environ['LAB_TARGET'] = 'solution'
!python labs/module-20/lesson-01/capstone_lab.py --variant main --print 2>&1 | tee "$O20/20-1-commands.txt"
!python -m frontierlab.capstone.scaffold --variant main --out runs/m20/l201/capstone-qkclip-main --device cuda 2>&1 | tee "$O20/20-1-scaffold.txt" | tail -30
"""),
]

# One short notebook per pilot, in priority order (plan 12.1): one Colab session each, resumable.
# (name, cells, runtime, hours on that runtime as stated in the pilot's own heading)
PILOTS = [
    ("P1", P1, "A100", "about 4.5 h (10m 0.3 h, 30m 1.3 h, 70m 3 h)"),
    ("P1B", P1B, "A100", "about 0.5 h"),
    ("P2", P2, "A100", "about 0.5 h"),
    ("P9", P9, "A100", "about 0.25 h"),
    ("P18", P18, "A100", "minutes"),
    ("P3", P3, "A100", "about 1.5 h"),
    ("P4", P4, "A100", "about 1-2 h (needs P1 pilot-70m seed 0)"),
    ("P17", P17, "A100", "about 1 h"),
    ("P13", P13, "A100", "about 1 h"),
    ("P15", P15, "A100", "about 1.5 h"),
    ("P20", P20, "A100", "about 1-2 h"),
    ("P11", P11, "T4", "about 2 h"),
    ("P16", P16, "T4", "T4 variants"),
    ("P10", P10, "A100", "about 2.5 h"),
    ("P5", P5, "A100", "about 3 h"),
    ("P12", P12, "A100", "about 3 h"),
    ("P14", P14, "A100", "about 3 h"),
    ("P8", P8, "L4", "about 3-4 h"),
    ("P7", P7, "A100", "about 4 h"),
    ("P19", P19, "A100", "about 4-8 h"),
    ("P6", P6, "A100", "about 5 h"),
]


def guard(pid: str, cells: list) -> list:
    """Route every `!python` line of a pilot through once.py, so a rerun skips finished commands."""
    out, n = [], 0
    for c in cells:
        if c["cell_type"] == "code":
            src = []
            for line in c["source"]:
                m = re.match(r"^(\s*)!python (.*?)(\n?)$", line)
                if m and "once.py" not in m.group(2):
                    assert "'" not in m.group(2), line
                    src.append(f"{m.group(1)}!python curriculum/pilots/once.py {pid} {n} 'python {m.group(2)}'{m.group(3)}")
                    n += 1
                else:
                    src.append(line)
            c = {**c, "source": src}
        out.append(c)
    out.append(code(f"""
# Last cell: marks {pid} complete when no command of it has failed; the status notebook reads this.
import glob, os
failed = glob.glob(f'{{PILOT_DIR}}/.failed/{pid}-*')
if failed:
    print('{pid} NOT complete; failed commands (rerun this notebook to retry):', failed)
else:
    open(f'{{PILOT_DIR}}/.done/{pid}-COMPLETE', 'w').write('ok')
    print('{pid} complete')
"""))
    return out


STATUS = [
    md("""
# Pilot status (CPU runtime is enough; no GPU units used)

Lists which pilots are complete and packs the results (text, JSON, metrics, summaries; no checkpoints)
into `frontier-llm-pilots-results.zip` on Drive. Download that zip and put it in the course repo under
`curriculum/pilots/incoming/`.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
PILOT_DIR = '/content/drive/MyDrive/frontier-llm-pilots'
import os, glob, zipfile
ORDER = %s
done = {os.path.basename(p)[:-9] for p in glob.glob(f'{PILOT_DIR}/.done/*-COMPLETE')}
for pid, gpu, hours in ORDER:
    n = len([p for p in glob.glob(f'{PILOT_DIR}/.done/{pid}-*') if not p.endswith('COMPLETE')])
    print(f"{pid:5} {'COMPLETE' if pid in done else ('partial, %%d commands done' %% n if n else 'not started'):28} {gpu:5} {hours}")
p1 = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f'{PILOT_DIR}/p1/*/metrics.jsonl'))
print('P1 runs with metrics:', p1)
z = f'{PILOT_DIR}/frontier-llm-pilots-results.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(PILOT_DIR):
        dirs[:] = [d for d in dirs if d not in ('data-v0-32k', 'data-v0-long', 'labdata')]
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(('.txt', '.json', '.jsonl', '.yaml', '.md', '.csv')) and os.path.getsize(full) < 50e6:
                zf.write(full, os.path.relpath(full, PILOT_DIR))
print('wrote', z, round(os.path.getsize(z) / 1e6, 1), 'MB')
""" % repr([(pid, gpu, hours) for pid, _, gpu, hours in PILOTS])),
]

NOTEBOOKS = {f"{i:02d}_{pid}.ipynb": SETUP + guard(pid, cells) for i, (pid, cells, gpu, hours) in enumerate(PILOTS, 1)}
NOTEBOOKS["00_status.ipynb"] = STATUS


def build():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        old.unlink()
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                           "kernelspec": {"display_name": "Python 3", "name": "python3"}},
              "nbformat": 4, "nbformat_minor": 5}
        if name == "00_status.ipynb":
            nb["metadata"].pop("accelerator")
        (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", OUT / name, f"({len(cells)} cells)")


if __name__ == "__main__":
    build()
