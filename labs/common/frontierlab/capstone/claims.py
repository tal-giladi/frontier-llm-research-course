"""The capstone claim list (lesson 20.1): six claims from 2025-26 reports, each checked against its primary source.

Each :class:`Claim` records the claim as the source states it (with the section or figure), the earlier lessons and
``frontierlab`` code it builds on, the budgets of the three variants, what a sound null result looks like, a
suggested extension ablation, and the reviewer questions specific to it (lesson 20.2). Main-path budgets are
PROJECTED (pending pilot) and :func:`projected_gpu_hours` is the formula behind the ones computed here; the others
come from the earlier lesson named in the budget text. Verified 2026-10-07.
"""

from __future__ import annotations

from dataclasses import dataclass, field

H100_BF16 = 989e12          # dense BF16 peak, FLOP/s (the course's figure since lesson 01.1)


def projected_gpu_hours(runs: int, tokens_per_run: float, train_flops_per_token: float, mfu: float,
                        peak: float = H100_BF16, overhead: float = 0.0) -> float:
    """PROJECTED GPU-hours = runs x tokens x training FLOPs per token / (peak x MFU) x (1 + overhead) / 3600."""
    return runs * tokens_per_run * train_flops_per_token / (peak * mfu) * (1 + overhead) / 3600


@dataclass(frozen=True)
class Claim:
    id: str
    title: str
    source: str
    url: str
    where: str
    statement: str
    maturity: str
    builds_on: tuple[str, ...]
    code: tuple[str, ...]
    reproduce: str
    axis: str
    main_path: str
    main_gpu_hours: tuple[float, float]
    free_gpu: str
    free_cpu: str
    null_result: str
    extension: str
    questions: tuple[str, ...] = field(default_factory=tuple)


CLAIMS: dict[str, Claim] = {}


def _add(c: Claim) -> None:
    CLAIMS[c.id] = c


_add(Claim(
    id="gspo",
    title="GSPO vs GRPO stability",
    source="Zheng et al., Group Sequence Policy Optimization (Qwen), 2025",
    url="https://arxiv.org/abs/2507.18071",
    where="abstract; section 4.1 (Eq. 5 objective, Eq. 7 sequence ratio); section 5.1 and Figure 1; section 5.2 "
          "and Figure 2; section 5.3 and Figure 3",
    statement="GSPO, which clips a length-normalised sequence-level importance ratio (left and right ranges 3e-4 and "
              "4e-4, against 0.2 and 0.27 for the GRPO baseline), trains stably throughout where GRPO needed Routing "
              "Replay to converge on an MoE model (cold-start from Qwen3-30B-A3B-Base, Figure 1, section 5.1), and "
              "reaches higher training efficiency although it clips about two orders of magnitude more tokens "
              "(section 5.2, Figure 2).",
    maturity="PROMISING (strongest evidence on MoE; for dense models closer to MODEL-SPECIFIC, lesson 14.1)",
    builds_on=("12.2", "12.3", "14.1", "14.2", "14.3"),
    code=("frontierlab.rlscale.objectives (grpo_loss, gspo_loss, OBJECTIVES)",
          "frontierlab.rlscale.runner (objective_config, run_arms, control rewards)",
          "frontierlab.rlscale.stability (stability metrics stated before the runs)",
          "frontierlab.rlscale.hf_rl (main path)", "frontierlab.evals.suite_v2"),
    reproduce="GRPO and GSPO, each at its own tuned learning rate with the same tuning budget, plus a random-reward "
              "GRPO control, from the Module 12 SFT start; stability metrics of lesson 14.2 and held-out accuracy.",
    axis="equal tokens (same prompts, group size, steps and minibatches per arm)",
    main_path="1x H100 80 GB, Qwen3-1.7B-Base on GSM8K (lesson 14.2's setup). Tuning 2 objectives x 3 learning rates "
              "x 60 steps, comparison 4 arms (GRPO, GSPO, random-reward control, extension) x 3 seeds x 200 steps: "
              "2,760 steps at lesson 12.2's 30-50 s/step = 23-38 h, plus 12 Eval v2 runs x 0.3-0.5 h = 3.6-6 h. "
              "PROJECTED 27-44 GPU-hours (dense model: the MoE Routing Replay part of the claim is not testable).",
    main_gpu_hours=(27, 44),
    free_gpu="T4, Qwen3-0.6B-Base, 8 prompts x 256 new tokens (lesson 14.2's t4 variant), 2 seeds; PROJECTED 6-10 h "
             "across sessions",
    free_cpu="toy policy of Module 12, 4 arms x 3 seeds x 120 steps plus 6 tuning runs: about 20-25 minutes "
             "(estimated from lesson 14.2's measured 35 minutes for 33 runs and 18 evaluations)",
    null_result="No arm crosses a stability threshold stated before the runs (lesson 14.2 found exactly this on CPU) "
                "and the accuracy interval of GSPO - GRPO includes zero: report 'no stability difference detected "
                "at this scale, MDE x', with GSPO's clip fraction shown as the signature of its small ranges, not as "
                "instability. A dense toy model cannot test the MoE part; say so instead of extrapolating.",
    extension="GSPO vs GRPO under bounded staleness (lesson 14.3's sampler, lag up to 4): does the sequence ratio "
              "change how each objective degrades off-policy? Or GSPO-token (Eq. 14) as a third arm.",
    questions=("Your GSPO and GRPO arms clip different fractions of tokens. Which stability metric could that bias, "
               "and in which direction?",
               "The source's stability evidence is on an MoE model with Routing Replay. What part of the claim does "
               "a dense model test, and what part not at all?",
               "Did the random-reward control move held-out accuracy? If it did, what does your accuracy difference mean?"),
))

_add(Claim(
    id="qkclip",
    title="QK-Clip vs QK-norm",
    source="Kimi Team, Kimi K2: Open Agentic Intelligence, 2025; DeepSeek-AI, DeepSeek-V4, 2026",
    url="https://arxiv.org/abs/2507.20534",
    where="K2 section 2.1 (QK-Clip, gamma_h = min(1, tau / S_max^h), tau = 100, Figure 2), Appendix D and Figure 12; "
          "DeepSeek-V4 (https://arxiv.org/abs/2606.19348) sections 2.3.3 and 2.4",
    statement="With vanilla Muon, maximum attention logits in a 9B-activated / 53B-total MoE 'quickly exceed a "
              "magnitude of 1000' (K2 section 2.1, Figure 2 left). QK-Clip bounds them, and on two 0.5B-activated / "
              "3B-total MoE models with an aggressive tau = 30 it 'has negligible impact on loss' (Appendix D, "
              "Figure 12). DeepSeek-V4 instead applies RMSNorm to queries and KV entries and does not use QK-Clip "
              "(sections 2.3.3, 2.4). Neither report compares the two fixes directly: that comparison is the extension.",
    maturity="QK-Clip PROMISING / MODEL-SPECIFIC; QK-norm against logit growth ESTABLISHED (lessons 07.2, 07.5)",
    builds_on=("03.3", "07.1", "07.2", "07.5"),
    code=("frontierlab.optim.train (--optimizer muon, --qk-norm, --qk-clip, --stability-log)",
          "frontierlab.optim.qkclip (QKClip, head_max_logits, LogitMonitor)",
          "frontierlab.optim.stability (StabilityLogger, detect_spikes)", "frontierlab.evals.heldout.window_losses",
          "frontierlab.capstone.scaffold (this claim, end to end)"),
    reproduce="Muon without QK-norm, with and without QK-Clip at a tau that binds at course scale (Appendix D's "
              "experiment: does clipping cost loss?), same seeds, data order and tokens.",
    axis="equal tokens (same preset, steps, batch and data order per seed)",
    main_path="1x H100 80 GB, 3 arms x 3 seeds per rung of the ladder. pilot-30m and pilot-70m: 4,000 steps x 64 x "
              "1,024 = 2.62e8 tokens per run, 3.21e8 and 6.14e8 training FLOPs per token, 20% MFU, +3% for the logit "
              "probe: 1.1 and 2.1 GPU-hours. Baseline-0: 2.49e9 tokens per run, 7.88e8 FLOPs per token, 30% MFU, +3%: "
              "17.0 GPU-hours. PROJECTED about 20 GPU-hours (formula: projected_gpu_hours).",
    main_gpu_hours=(18, 22),
    free_gpu="T4, pilot-10m, 2,000 steps x 32 x 512, fp32, 3 arms x 2 seeds; PROJECTED 1.5-2.5 h",
    free_cpu="toy preset, 200 steps x 8 x 128 at lr 1e-2, 3 arms x 3 seeds: measured in this build (lesson 20.1)",
    null_result="The clip binds (clipped heads > 0) and the held-out loss interval of clip - no-clip lies inside the "
                "margin: a reproduction of 'negligible impact on loss' at course scale. If logits never pass tau, "
                "the clip never fires and the runs are bit-identical: the claim is untested, not reproduced (lesson "
                "07.2 saw this at tau = 100). The extension's null is 'QK-norm and QK-Clip equivalent within margin'.",
    extension="QK-Clip vs QK-norm at equal tokens (the comparison neither report ran), or tau swept at two values to "
              "find where clipping starts to cost loss.",
    questions=("How did you choose tau, and why is it not Kimi K2's 100?",
               "How many head-updates did the clip rescale, and from which step? If it never fired, what did you test?",
               "QK-norm adds parameters and changes the model; why is the comparison still one changed variable?"),
))

_add(Claim(
    id="dsa",
    title="DSA quality and cost vs dense attention",
    source="DeepSeek-AI, DeepSeek-V3.2, 2025",
    url="https://arxiv.org/abs/2512.02556",
    where="section 2.1 and 2.1.1 (lightning indexer, top-2,048, dense warm-up and sparse training stages, KL to the "
          "L1-normalised main attention, over the selected set in the sparse stage); section 2.2 (parity); section "
          "2.3 and Figure 3 (inference cost on H800 at USD 2 per GPU-hour)",
    statement="Continued training with DeepSeek Sparse Attention (each query attends to 2,048 keys chosen by a "
              "lightning indexer: dense warm-up of 1,000 steps / 2.1B tokens, then 15,000 sparse steps / 943.7B "
              "tokens) shows 'no substantial performance degradation' against DeepSeek-V3.1-Terminus on short and "
              "long context (section 2.2), while reducing core attention from O(L^2) to O(Lk) and the measured "
              "serving cost of long sequences (section 2.3, Figure 3).",
    maturity="MODEL-SPECIFIC (company claim; lesson 05.2)",
    builds_on=("04.1", "05.2", "05.3"),
    code=("frontierlab.attention.dsa (DSAttention, set_dsa, indexer_kl, selection_recall)",
          "frontierlab.attention.subq_bench (component profiling)", "labs/module-05/lesson-02/dsa_stages.py",
          "frontierlab.evals.suite_v1"),
    reproduce="Warm-up then sparse continued training from the Module 4 base against a dense control continued for "
              "the same tokens; held-out loss and Eval v1 with paired intervals; component cost profile.",
    axis="equal tokens for quality; cost reported per component at fixed context lengths",
    main_path="1x H100 or A100 80 GB. Lesson 05.2 projects about 2 GPU-hours per sparse-plus-control pair at one seed "
              "(Baseline-0, T = 1,024, k = 128); 3 seeds plus an extension arm per seed (about 1 GPU-hour each) and "
              "1 GPU-hour of lesson 05.3 profiling: PROJECTED about 10 GPU-hours.",
    main_gpu_hours=(8, 12),
    free_gpu="T4, fp32, lesson 05.2's t4 variant x 2 seeds; PROJECTED 2-3 h",
    free_cpu="toy, k = 32 of 256: about 85 minutes for 3 seeds (from lesson 05.2's measured 32 minutes per seed with "
             "its ablation arm, minus the shared warm-up)",
    null_result="Sparse - dense held-out loss inside the margin with indexer recall reported: 'quality kept at this "
                "scale'. On CPU the sparse path is slower than dense at every length the toy reaches (lesson 05.3 "
                "measured no prefill crossover up to 8K); report the component costs and the crossover estimate, "
                "never an end-to-end speed-up you did not measure.",
    extension="Top-k swept (k = 16 and 64 of 256) or the warm-up removed, to find where quality starts to drop.",
    questions=("Your dense control trained for the same tokens. Why is that the right control and not the parent "
               "checkpoint?",
               "At what context length would the sparse path be cheaper on your hardware, and is that measured or projected?",
               "What fraction of the attention mass does the indexer recover, and how does that compare with the oracle?"),
))

_add(Claim(
    id="mhc",
    title="mHC vs HC stability",
    source="Xie et al. (DeepSeek-AI), mHC: Manifold-Constrained Hyper-Connections, 2025",
    url="https://arxiv.org/abs/2512.24880",
    where="section 3.1 and Figures 2-3 (HC loss surge near step 12k, Amax gain magnitude peaks of 3,000 at 27B); "
          "section 4.2 (Sinkhorn-Knopp, t_max = 20); section 5.4 and Figure 7 (mHC gain at most about 1.6); "
          "section 4.3 (6.7% overhead at n = 4); section 5.2 and Table 4",
    statement="Unconstrained hyper-connections become unstable at 27B: a loss surge near step 12k with a gradient-norm "
              "spike (Figure 2), and a composite residual gain whose peaks reach 3,000 (Figure 3b). Projecting the "
              "residual mixing onto doubly stochastic matrices (Sinkhorn-Knopp, t_max = 20, n = 4) keeps that gain "
              "below about 1.6 (section 5.4, Figure 7b) at 6.7% training-time overhead (section 4.3).",
    maturity="PROMISING (adopted by its authors in DeepSeek-V4; no independent replication; lesson 06.2)",
    builds_on=("06.2", "06.3", "07.5"),
    code=("frontierlab.blocks.hyperconn (HyperConnection, ManifoldHC, sinkhorn_knopp, amax_gain)",
          "frontierlab.blocks.train (--residual plain|hc|mhc, --streams, --blocks-log)",
          "frontierlab.optim.stability.detect_spikes", "labs/module-06/lesson-02/train_arms.py"),
    reproduce="Baseline, HC and mHC (n = 4) at a standard and a raised learning rate, with the composite gain and "
              "gradient norm logged; spikes counted by the rule of lesson 07.5.",
    axis="equal tokens",
    main_path="1x H100. Lesson 06.2: 12 runs per rung, pilot-30m = 1.0e18 FLOPs, 1.2 GPU-hours at 25% MFU, times the "
              "unfused HC/mHC wall-clock factor 1.5-3 = 2-3.5 GPU-hours. Capstone: 3 arms x 3 seeds x 2 learning "
              "rates = 18 runs: x1.5 = 3-5.3 GPU-hours at pilot-30m and x1.91 (FLOPs per token) at pilot-70m = "
              "5.7-10 GPU-hours. PROJECTED about 9-15 GPU-hours.",
    main_gpu_hours=(9, 15),
    free_gpu="T4, pilot-10m rung of lesson 06.2, 2 seeds; PROJECTED 2-4 h",
    free_cpu="toy, 3 arms x 3 seeds at one raised learning rate: about 30-40 minutes (from lesson 06.2's measured "
             "30 minutes for 9 runs)",
    null_result="No loss spikes in any arm (lesson 06.2 saw none on CPU) while HC's gain grows and mHC's stays near 1: "
                "report 'the mechanism (gain growth) is reproduced, the loss instability is not, at this scale', "
                "with the spike rule and the MDE. Do not call mHC 'more stable' on loss without spikes to compare.",
    extension="Sinkhorn iterations cut from 20 to 3 (does an approximately doubly stochastic map keep the gain "
              "bounded?) or the expansion rate n = 2 vs 4.",
    questions=("Your stability metric is the composite gain, not a loss spike. Why is that evidence about stability "
               "at all, and what would make it irrelevant?",
               "HC and mHC differ in wall-clock by more than their FLOPs. On which axis did you compare them, and why?",
               "How many seeds show the gain growth, and is it monotone in the learning rate?"),
))

_add(Claim(
    id="microanneal",
    title="Micro-anneal data scoring",
    source="OLMo 2 (Team OLMo, 2 OLMo 2 Furious, 2024); Olmo 3 (Team Olmo, 2025)",
    url="https://arxiv.org/abs/2501.00656",
    where="OLMo 2 section 4.4.2 and Table 12 (19 microanneals, 130B tokens in total, fewer than 3 full 50B anneals; "
          "GSM* of 200 questions and MMLU); Olmo 3 (https://arxiv.org/abs/2512.13961) section 3.5.1 (5B target + 5B "
          "web tokens against a 10B web-only baseline microanneal)",
    statement="A short anneal on a 50/50 mix of a candidate source and general web data, with the learning rate taken "
              "linearly to zero, scores the source 'at a fraction of the cost of a full annealing run' (OLMo 2 "
              "section 4.4.2): 19 microanneals totalled 130B tokens, and their effects held when the sources were "
              "mixed (section 4.2). Olmo 3 standardised it as 5B target + 5B web tokens against a web-only "
              "microanneal of the same 10B tokens (section 3.5.1).",
    maturity="PROMISING (documented by two labs; lesson 10.4)",
    builds_on=("10.1", "10.2", "10.4"),
    code=("frontierlab.datax.anneal (anneal_spec, run_microanneals)", "frontierlab.datax.arms (train_arm, score, table)",
          "frontierlab.datax.train (--anneal)", "labs/module-10/lesson-04/mixture_lab.py anneal"),
    reproduce="Score candidate sources by microanneals from one stable checkpoint against a control anneal on the "
              "base mixture for the same tokens; own-domain and guard metrics with paired intervals.",
    axis="equal tokens (every anneal the same length from the same checkpoint)",
    main_path="1x H100, pilot-30m: one stable run (4,000 steps x 64 x 1,024 = 2.62e8 tokens) and 21 anneals (control, "
              "3 candidates, 3 extension arms x 3 seeds) of 1,000 steps x 64 x 1,024 = 6.55e7 tokens; 1.64e9 tokens "
              "x 3.21e8 FLOPs per token at 30% MFU = 0.5 GPU-hours; pilot-70m rung x1.91 = 0.9. PROJECTED 1.5-2.5 "
              "GPU-hours with evaluation.",
    main_gpu_hours=(1.5, 2.5),
    free_gpu="T4, pilot-10m, lesson 10.4's t4 variant x 3 seeds; PROJECTED 1-2 h",
    free_cpu="toy, one stable run and 21 anneals of 100 steps: about 25 minutes (from lesson 10.4's measured 330 s "
             "stable run and 47-58 s per anneal)",
    null_result="A candidate whose own-domain gain interval includes zero, or whose guard interval crosses the budget, "
                "is reported as 'not distinguishable from the control anneal at this budget', with the control "
                "anneal's own effect shown (lesson 10.4 measured -0.105 from annealing alone). The score is a ranking "
                "of sources at one scale; whether the ranking survives a full mix is the integration test.",
    extension="The candidate's share cut from 50% to 10% (OLMo 2's first microanneal experiment, Table 12): does the "
              "ranking of sources survive the smaller share?",
    questions=("Your control anneal also lowered loss. How much of each candidate's gain is the anneal itself?",
               "Which metric is the guard, and what budget did you state for it before the runs?",
               "Would the ranking change at another anneal length? What did you run to check?"),
))

_add(Claim(
    id="opd",
    title="On-policy vs off-policy distillation at equal compute",
    source="Thinking Machines (K. Lu), On-Policy Distillation, 2025-10-27; Agarwal et al., GKD, 2023",
    url="https://thinkingmachines.ai/blog/on-policy-distillation/",
    where="blog: the reasoning experiment and its cost table (9-30x); GKD (https://arxiv.org/abs/2306.13649) "
          "sections 3-4; Qwen3 technical report (https://arxiv.org/abs/2505.09388) section 4.5, Table 21",
    statement="Starting from a Qwen3-8B-Base student fine-tuned on 400k prompts (60% on AIME'24), on-policy "
              "distillation with a per-token reverse-KL reward reaches 70% in about 150 steps, where off-policy "
              "fine-tuning on teacher outputs is extrapolated to need about 2M prompts: a cost reduction of 9x when "
              "the SFT dataset is given, about 18x in GPU-hours, about 30x if the teacher's sampling is charged "
              "(company blog; the comparison is against off-policy SFT, not RL).",
    maturity="PROMISING (two strong reports; lesson 13.2)",
    builds_on=("13.1", "13.2"),
    code=("frontierlab.pipeline.distill (train_opd, opd_loss, teacher_examples, ensure_teacher)",
          "frontierlab.pipeline.compute (Ledger: teacher_sample, teacher_score, student_sample, student_train)",
          "labs/module-13/lesson-02/distill_lab.py", "frontierlab.evals.suite_v2"),
    reproduce="Off-policy SFT on teacher samples and on-policy distillation from the same student and teacher, with "
              "the off-policy arm's size chosen so the ledger totals match (teacher compute counted), and a "
              "self-teacher control; held-out accuracy per FLOP.",
    axis="equal training FLOPs including teacher sampling and scoring (frontierlab.pipeline.compute.Ledger)",
    main_path="1x H100 80 GB, student Qwen3-1.7B-Base, teacher Qwen3-8B (lesson 13.2's main path). Per seed: "
              "off-policy 1-1.3, on-policy 0.7-1.0, self-teacher control 0.7-1.0, extension 0.7-1.0 GPU-hours; x3 "
              "seeds plus 12 Eval v2 runs x 0.3-0.5: PROJECTED 13-19 GPU-hours.",
    main_gpu_hours=(13, 19),
    free_gpu="T4, Qwen3-0.6B-Base student with a Qwen3-1.7B teacher, 2 seeds; PROJECTED 3-5 h",
    free_cpu="toy student and teacher of lesson 13.2, 4 arms x 3 seeds: about 25 minutes (from lesson 13.2's measured "
             "511 s for 10 runs plus about 15 minutes once for the teacher)",
    null_result="At equal FLOPs the on-policy arm is not better, or is worse (lesson 13.2 measured the opposite of the "
                "published direction on its 2-4-token answers): report it with the ledger and the reason the setting "
                "differs (short answers leave little prefix mismatch to correct, INFERENCE), not as a refutation of a "
                "long-reasoning result.",
    extension="Longer answers (the problem length at which on-policy correction should start to matter), or a mixed "
              "data fraction as in GKD.",
    questions=("Which FLOPs did you count for the teacher in each arm, and does the comparison survive charging the "
               "teacher's own training?",
               "Your answers are a few tokens long. Why should the published long-reasoning result transfer, or not?",
               "What does the self-teacher control rule out?"),
))


GENERIC_QUESTIONS = (
    "Why this comparison axis and not another, and what does it not answer?",
    "How do you know the baseline was tuned as hard as the new method?",
    "What is the smallest effect your design could detect, and is your result above it?",
    "What would you run next with 10x the budget, and what result would make you abandon the claim?",
)


def get(claim_id: str) -> Claim:
    if claim_id not in CLAIMS:
        raise KeyError(f"unknown claim {claim_id!r}; choose one of {sorted(CLAIMS)}")
    return CLAIMS[claim_id]


def budget_table() -> str:
    """One markdown row per claim: main path (PROJECTED), free GPU, free CPU."""
    rows = ["| Claim | Main path (PROJECTED GPU-hours) | Free GPU | Free CPU |", "|---|---|---|---|"]
    for c in CLAIMS.values():
        rows.append(f"| {c.title} | {c.main_gpu_hours[0]:g}-{c.main_gpu_hours[1]:g} | {c.free_gpu} | {c.free_cpu} |")
    return "\n".join(rows)


__all__ = ["CLAIMS", "Claim", "GENERIC_QUESTIONS", "H100_BF16", "budget_table", "get", "projected_gpu_hours"]
