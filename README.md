# Frontier LLM Research Engineering

An advanced follow-up to the LLM Research Engineer course: you learn to design, run, debug and defend the experiments frontier-lab research engineers run on attention, long context, optimizers, precision, distributed training, data, scaling, post-training, reasoning RL, test-time compute, agents and interpretability, using recent open models as case studies. The main path runs on rented GPUs; every lab also has a smaller free Colab and free CPU version.

## Who this is for

You finished the LLM Research Engineer course, or you already can: write and train a GPT from scratch, profile it, write a basic Triton kernel, use DDP/FSDP, and run SFT, LoRA, DPO and GRPO. This course does not re-teach those; lessons link back to the parent course when you need a refresher.

## What makes this course different

- **Questions, not techniques.** Every module asks a research question — "When is sub-quadratic attention worth it?", "Which RL objective, at what scale?" — and uses DeepSeek, Qwen, Kimi, gpt-oss, Gemma, Llama, OLMo and other published models as evidence for answers under stated constraints. There is no single "frontier recipe" to copy.
- **Discipline first.** Module 1 teaches hypotheses, controls, comparison axes (equal tokens, parameters, FLOPs or wall-clock), tuning budgets, uncertainty and reproducibility. Every later lab uses them.
- **Every comparison has an experiment contract**: the question, baseline, controlled variables, budget, metrics with uncertainty, correctness checks and the limits of the conclusion. A sound negative result passes; whether the new method wins is never the grade.
- **Correct, then fast, then compared.** New mechanisms pass gradient, causality, cached-decode and reference checks before any benchmark, and benchmarks follow the Module 2 profiling method before any comparison.
- **Evidence labels.** Lessons separate what a lab has published (with the report section) from industry practice and from speculation, and tag each technique as established, promising or model-specific.
- **One traceable chain.** You build Baseline-0, branch it into isolated architecture experiments, integrate the ones that hold up, choose an optimizer, precision and data recipe, post-train, evaluate, interpret, and write it up in a capstone that reproduces and extends a recent paper.

## Hardware

- **Main path (the course standard): rented GPUs** — from one 48–80 GB GPU to one 8-GPU node. Each lab states the GPU, GPU-hours, unattended runtime and a projected cost. The whole required main path is projected at roughly 650–1,150 H100-hours.
- **Free GPU (Colab/Kaggle T4)** and **free CPU (laptop)** versions of every lab show the same mechanism at a smaller scale and say what you will not see there.

Cost and time figures come from scaled pilots and are labelled PROJECTED; record your own measured numbers in each run card.

## Setup

```bash
cd labs/common
python -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e .
.venv/bin/python -m pytest
.venv/bin/python -m frontierlab.data.prepare --docs 20000 --vocab 8192
```

On Windows use `.venv\Scripts\` instead of `.venv/bin/`. On a GPU machine install the CUDA build of PyTorch 2.14.1 instead of the CPU wheel. Pinned versions are in [Software versions](references/versions.md).

## Course map

The sidebar lists every lesson. Stages: A research foundations (Modules 1–2) → B architecture questions (3–6) → C training-recipe questions (7–11) → D post-training questions (12–16) → E understanding, evaluating and communicating (17–20). Lessons marked **Extension** are optional. The required course covers text, reasoning and agents; a multimodal elective follows later.
