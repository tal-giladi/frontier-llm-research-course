# Module 19 labs — How is frontier research chosen, reproduced and written?

Each lesson folder has `lab.py` (yours, with TODOs that raise `NotImplementedError("TODO n: ...")`), `solution.py`
(the reference), `test_lab.py` and the script the lesson runs. The shared code is
`labs/common/frontierlab/research/` (`proposals`, `reproduce`, `writeup`, `contract`), with tests in
`labs/common/tests/test_research.py`. The labs reuse the course training loop through the Module 7 wrapper
(`frontierlab.optim.train`), `frontierlab.stats`, `frontierlab.record` (run-card diffs) and the course data, and edit
none of them.

```bash
pip install -r labs/common/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e labs/common
python -m frontierlab.data.prepare --docs 20000 --vocab 8192     # once, if not done in Module 1
pytest labs/common/tests/test_research.py                        # the shared Module 19 code (seconds)
pytest labs/module-19/lesson-01                                  # checks your lab.py (fails until the TODOs are done)
LAB_TARGET=solution pytest labs/module-19                        # all reference solutions and the project files
```

Scripts load your `lab.py` by default; prefix `LAB_TARGET=solution` to run them with the reference. Every script writes
under `runs/m19/` (gitignored) and skips finished runs, so an interrupted sweep resumes where it stopped.

## What each folder contains

| Folder | Lesson | Script | What it does | Free CPU time (measured, build laptop) |
|---|---|---|---|---|
| `lesson-01/` | 19.1 Research taste and problem choice | `choose_lab.py` | rank: your three proposals scored (value × p_decisive ÷ cost), minimum detectable effects, robustness of the ranking at 2× and 3×. proxy: QK-norm on/off at lr 1e-2 at widths 64 and 128, 2 seeds, read with `proxy_trend` | rank under a second; proxy 7.8 minutes (8 runs) |
| `lesson-02/` | 19.2 Reproducing a paper | `repro_lab.py` | Wortsman et al.'s "Qk-layernorm reduces LR sensitivity": 2 arms × 3 learning rates × 3 seeds on the toy preset with the paper's optimizer settings, sensitivities, the decision against a tolerance stated before the runs, the deviation log, a validated record | 37.7 minutes (18 runs, 126 s each with another module's jobs sharing the CPU) |
| `lesson-03/` | 19.3 Writing and defending results | `review_lab.py`, `make_cards.py` | review: a colleague's fictional note (`flawed/`) and its revision (`fixed/`) linted against their EXAMPLE run cards, your review checked for the four planted problems; figure: final loss against learning rate from your lesson 19.2 runs, with its linted spec. `review-template.md` is the review template | seconds |
| `project/` | Module project | `check_proposal.py` | the capstone proposal validator and rubric total; `proposal_template.yaml` to fill, `buggy_proposal.yaml` for the debugging task, `example_proposal.yaml` as a reference, `test_proposal.py` | under a second |

## Hardware per variant

| Variant | What runs | Hardware | Time |
|---|---|---|---|
| Main path | `--variant main --print` in `choose_lab.py` and `repro_lab.py` prints the GPU commands: the proxy ladder at `pilot-30m` widths 256 and 512; the reproduction at `pilot-30m` (and `--variant main70` at `pilot-70m`) with the paper's seven learning rates | 1× H100 or A100; not run in this build (Module 19 pilot) | **PROJECTED:** proxy 0.13 H100-hours; reproduction 4.0 (pilot-30m) and 7.6 (pilot-70m) H100-hours at an assumed 25% MFU (formulas in the lessons) |
| Free GPU | `repro_lab.py --variant t4`: `pilot-10m`, 5 rates, 3 seeds, fp32 | Colab/Kaggle T4 | PROJECTED 2.3 T4-hours across sessions |
| Free CPU | the scripts as written | laptop | see the table above |
