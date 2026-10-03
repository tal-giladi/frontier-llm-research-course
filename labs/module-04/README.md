# Module 4 labs — Does the model use its context?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`),
`solution.py` (the reference), `test_lab.py` and the scripts the lesson runs. Shared code is in
`labs/common/frontierlab/longctx/` (RoPE rules, the `"gqa-rope-scaled"` and `"gqa-irope"` attention
kinds, long-document data, the continued-training wrapper `extend.py`, `prepare_long.py`) and
`labs/common/frontierlab/evals/suite_v1.py` (Eval Suite v1); their tests are
`labs/common/tests/test_longctx.py`.

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
pytest labs/module-04/lesson-01                       # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-04             # all three reference solutions
pytest labs/common/tests/test_longctx.py              # the shared Module 4 code, incl. the match with Transformers
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference.
Until the main session registers the Module 4 attention kinds in `frontierlab/attention/__init__.py`,
every script imports `frontierlab.longctx` itself, which registers them.

| Folder | Lesson | Scripts | What they do |
|---|---|---|---|
| `lesson-01/` | 04.1 Long-context evaluation that means something | `train_base.py`, `check_eval.py` | train the Module 4 base model (Baseline-0's recipe at 256 tokens on CPU); validate Eval v1 (oracle, evidence ablation, prior leak, scorer agreement) |
| `lesson-02/` | 04.2 Position at long range | `freq_table.py`, `zero_shot.py` | per-frequency table of PI / NTK-aware / YaRN (both ramps); the base model past its trained length under each rule, no training, paired by document |
| `lesson-03/` | 04.3 Extending context by continued training | `extend_arms.py`, `compare_arms.py` | four equal-token arms from the base (control at the original length, YaRN within-document, YaRN random windows, PI within-document); Eval v1 + Eval v0 and the contract's decision |
| `project/` | Module project | `run_project.py`, `buggy_report.py` | two-stage extension with an equal-token control and the full evaluation; the debugging task |

Command-line tools from the shared code:

```bash
python -m frontierlab.longctx.data --split train                         # long-document statistics of a split
python -m frontierlab.longctx.prepare_long --skip 20000 --docs 200000 --min-tokens 1024 --splits val test --out labs/common/data/v0-long
python -m frontierlab.longctx.extend --init-from CKPT --rope yarn --factor 4 --original 256 --long-fraction 1.0 --run RUN --seq 1024 ...
python -m frontierlab.evals.suite_v1 run RUN --train-len 256 --lengths 256 512 1024 2048 --out RUN/eval_v1.json
python -m frontierlab.evals.suite_v1 compare A.json B.json
```

## Hardware and time per variant

Main-path commands were **not run in this build**; they are part of the Module 4 pilot and every
main-path figure in the lessons is PROJECTED with its formula. Free CPU times were measured on
2026-10-03 on a 16-thread Windows 11 laptop (torch 2.14.1+cpu, transformers 5.18.0) with other jobs
running, so expect variation.

| Lab | Main path (rented GPU) | Free GPU (Colab/Kaggle T4) | Free CPU (measured) |
|---|---|---|---|
| data | `prepare_long --skip 2600000 --docs 9600000 --min-tokens 8192 --splits train val test --out labs/common/data/v0-long8k` (vocab-32,768 Data-v0; hours of streaming, several GB downloaded; PROJECTED) | as CPU | `prepare_long` for held-out documents: 6.7 min, about 1 GB downloaded, 648 val / 699 test documents of at least 1,024 tokens |
| 04.1 | 1× H100/A100, ~30 GPU-min: Eval v1 of your Module 1 Baseline-0 at 1K–32K | `train_base.py --variant t4` (~30 min), suite at 512–4,096 | base 30 min (3,550 tok/s); Eval v1 at four lengths 200 s; `check_eval.py` 64 s |
| 04.2 | 1× H100/A100, ~20 GPU-min: `zero_shot.py` at 8K and 32K | ~10 min | 193 s at 1,024; 498 s at 2,048; optional architecture runs 15–18 min each |
| 04.3 | 1× H100, 4 arms × 524M tokens (1K → 8K), PROJECTED ~2 GPU-hours | ~40 min | arms 193–220 s each (about 14 min); `compare_arms.py` 12 min the first time |
| project | 1× H100 80 GB, 600M extension tokens + 600M control, PROJECTED ~2 GPU-hours plus evaluation | ~1.5 h | about 45 min: stages 482 s + 319 s, control 439 s, evaluation 1,460 s |

Notes:

- On a T4 use fp32 (no `--dtype bf16`): it has no BF16 tensor cores.
- At 32K the fp32 logits are 4 GiB per sequence at vocabulary 32,768; the main-path project uses
  `--loss chunked` and batch 1 with gradient accumulation.
- Outputs go to `runs/` (gitignored); the long-document folders go to `labs/common/data/` (gitignored).
- Context parallelism (splitting one sequence across GPUs, Ring Attention) is Module 9; nothing here needs more than one GPU.
