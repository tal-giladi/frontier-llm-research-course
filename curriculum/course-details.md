# Course details (for registering the course in the Academy)

- **Title:** Frontier LLM Research Engineering
- **Suggested slug:** `frontier-llm-research`
- **GitHub repo:** `tal-giladi/frontier-llm-research-course` (can be public; nothing private in it)
- **Free or paid:** free (Tal, 2026-10-03)
- **Risk notice:** The main path uses rented GPUs that you pay for; each lab states its projected cost, and the whole required path is projected at roughly 650–1,150 H100-hours. Free Colab and CPU versions of every lab are provided.
- **One-line summary:** Design, run, debug and defend the experiments frontier-lab research engineers run on LLM architecture, training, post-training and interpretability.
- **Who it is for:** graduates of the LLM Research Engineer course, or engineers who can already train a GPT from scratch, profile it, write a basic Triton kernel, use DDP/FSDP and run SFT/LoRA/DPO/GRPO.
- **Format:** 20 modules, 85 lessons (72 required, 13 extensions), a lab in every lesson with an experiment contract, a lesson quiz and a module quiz for each module, module projects with written defences, and a reproduce-and-extend capstone. Main path on rented GPUs; free Colab GPU and free CPU variants.
- **Prerequisites:** the LLM Research Engineer course (or equivalent), Python, PyTorch, a budget for rented GPUs for the main path.
- **What you will learn:**
  - Turn a vague research question into a controlled experiment with a stated comparison axis, budget and decision rule.
  - Measure noise and report effects with uncertainty; grade experiments on soundness, not on whether the method wins.
  - Implement recent mechanisms (MLA, sparse and hybrid attention, MTP, mHC, Muon, FP8) and prove them correct before comparing them.
  - Profile training and explain the gap between theoretical FLOPs and measured performance.
  - Plan and debug distributed training, including failure recovery and goodput.
  - Build data, scaling and post-training experiments (rewards, RL objectives, distillation, test-time compute, agent environments) that survive scrutiny.
  - Make causal interpretability claims with controls, and read system cards and safety frameworks critically.
  - Reproduce and extend a recent paper's claim, and defend it in writing.
