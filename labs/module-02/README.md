# Module 2 labs — Where does the time go?

Each folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and one script that runs the measurement. Shared code is
in `labs/common/frontierlab/perf/` (tests: `labs/common/tests/test_perf.py`).

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/module-02/lesson-01                      # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest --import-mode=importlib labs/module-02   # all four reference solutions at once
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference.

| Folder | Lesson | Script | What it measures |
|---|---|---|---|
| `lesson-01/` | 02.1 From FLOPs to time | `measure.py` | the machine's empirical roofline, five ops (predicted vs measured), one training step and its implied MFU |
| `lesson-02/` | 02.2 Profiling a training step | `profile_step.py` | profiler summary by category, per-op overhead, saved activation bytes and step time for plain / checkpointed / chunked / both; GPU: trace, memory snapshot, `--compile` |
| `lesson-03/` | 02.3 Communication and overlap | `run_dp.py` | all-reduce bus bandwidth; DDP at several bucket sizes and FSDP2, steps with and without gradient sync, exposed communication with intervals |
| `lesson-04/` | 02.4 Validating a performance claim | `compare_ce.py` | equal-work comparison of the plain and chunked loss: float64 correctness, saved memory, interleaved step times, paired speed-up, the decision; GPU: `--max-batch` |
| `project/` | Module project | `claim.py` | the debugging task: a performance claim with planted flaws (CPU, about 10 s) |

## Hardware and time per variant

All main-path commands were **not run in this build**; they are part of the Module 2 pilot. Free CPU
times were measured on 2026-10-03 on a 16-thread Windows 11 laptop (torch 2.14.1+cpu) that had other
processes running; expect variation.

| Lab | Main path (rented GPU) | Free GPU (Colab/Kaggle) | Free CPU (measured) |
|---|---|---|---|
| 02.1 | 1× H100 SXM or A100, ~5 GPU-min: `measure.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --hw H100-SXM` | T4: `--dtype fp32 --preset pilot-10m --batch 8 --seq 512 --hw T4` | ~40 s |
| 02.2 | 1× H100 SXM or A100, ~10 GPU-min: `profile_step.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --trace runs/l22/trace.json --snapshot runs/l22/mem.pickle --compile` | T4: `--dtype fp32 --preset pilot-10m --batch 8 --seq 512` | ~2 min |
| 02.3 | 1 node, 2–8× H100 SXM, ~15 min: `run_dp.py --device cuda --world 8 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --dtype bf16 --buckets 5 25 100 --steps 30` | Kaggle 2× T4: `--device cuda --world 2 --preset pilot-10m --batch 8 --seq 512 --buckets 1 25 --steps 30` (Colab has one GPU) | ~3.5–4 min (`--steps 30`) |
| 02.4 | 1× H100 SXM or A100, ~15 GPU-min: `compare_ce.py --device cuda --dtype bf16 --preset baseline0 --vocab 32768 --batch 8 --seq 1024 --steps 40 --max-batch --out runs/l24/result.json` | T4: `--dtype fp32 --preset pilot-10m --vocab 32768 --batch 8 --seq 512 --steps 30 --max-batch` | ~3 min |

Notes:

- On the CPU, `torch.compile` needs a C++ compiler for Inductor (MSVC `cl.exe` on Windows, gcc on
  Linux). Without one, `--compile` reports the error and the rest of the script still runs.
- The CPU variant of 02.3 uses real `gloo` collectives between processes on one machine; it shows
  bucketing, overlap and how exposure is measured, not interconnect performance.
- CPU numbers never transfer to GPUs: the ridge point, the kernel set and the launch costs all differ.
- Outputs go to `runs/` (gitignored); keep the printed tables with your run cards.
