---
id: "20.1"
module: 20
minutes: 45
practice_minutes: 150
prerequisites: ["19.3", "01.3", "01.4", "01.5", "07.2"]
objectives:
  - Choose one claim from the capstone list, state it exactly as its primary source does (section, figure, setting), and say before any run what would count as reproduced, as not tested and as a sound null at small scale.
  - Plan a reproduction and one new extension ablation within a stated budget, with main-path GPU-hours PROJECTED from a formula and free variants that say what they cannot show.
  - Compute a paired interval that resamples both seeds and evaluation items, and apply a decision rule with an equivalence margin that was fixed in advance.
  - Assemble a capstone package (contract, run cards with parents, results, claims, report) and make it pass the soundness checker without changing any result.
  - Score a capstone with the course rubric, where a sound null result earns the same marks as a positive one.
volatility: implementation
sources:
  - title: "Zheng et al., Group Sequence Policy Optimization (section 4.1 Eqs. 5 and 7; section 5.1 and Figure 1; section 5.2 and Figure 2; section 5.3 and Figure 3)"
    url: https://arxiv.org/abs/2507.18071
  - title: "Kimi Team, Kimi K2: Open Agentic Intelligence (section 2.1 and Figure 2: QK-Clip; Appendix D and Figure 12)"
    url: https://arxiv.org/abs/2507.20534
  - title: "DeepSeek-AI, DeepSeek-V4 (section 2.3.3: query and KV RMSNorm; section 2.4: no QK-Clip)"
    url: https://arxiv.org/abs/2606.19348
  - title: "DeepSeek-AI, DeepSeek-V3.2 (sections 2.1-2.3; Figure 3: inference cost)"
    url: https://arxiv.org/abs/2512.02556
  - title: "Xie et al., mHC: Manifold-Constrained Hyper-Connections (section 3.1, Figures 2-3; sections 4.2-4.3; section 5.4, Figure 7)"
    url: https://arxiv.org/abs/2512.24880
  - title: "Team OLMo, 2 OLMo 2 Furious (section 4.4.2 and Table 12: microanneals)"
    url: https://arxiv.org/abs/2501.00656
  - title: "Team Olmo, Olmo 3 (section 3.5.1: microanneal methodology)"
    url: https://arxiv.org/abs/2512.13961
  - title: "Thinking Machines (K. Lu), On-Policy Distillation, 2025-10-27"
    url: https://thinkingmachines.ai/blog/on-policy-distillation/
  - title: "Agarwal et al., On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)"
    url: https://arxiv.org/abs/2306.13649
  - title: "Google Research, Deep Learning Tuning Playbook"
    url: https://github.com/google-research/tuning_playbook
last_verified: "2026-10-07"
---

# 20.1 · Capstone brief and rubric

The capstone is where the course's habits have to work without scaffolding around them: you choose one claim from a 2025–26 report, reproduce it at a scale you can afford, extend it with one ablation nobody has published, and hand in a package that a reviewer can check line by line. This lesson gives the claim list (each claim checked against its primary source, with the earlier modules and `frontierlab` code it builds on, the budgets, and what a sound null result looks like), the rubric the capstone is graded by, the paired uncertainty that the decision rests on, and a scaffold that runs one claim end to end on a laptop so you see every part of the package before you build your own. You start from the proposal you wrote in the Module 19 project.

## Why this matters at a frontier lab

Most research engineering at a frontier lab is exactly this loop. A paper from another lab reports a change (a new RL objective, a stability fix, a cheaper attention); someone has to say whether it holds in the lab's own setting, at a scale that costs hours rather than weeks, and what to try next. The answer goes into a decision that commits real compute, so it must survive a reviewer who asks how the baseline was tuned, whether the comparison was fair, how noisy the number is and whether the conclusion says more than the runs show. "We could not detect a difference larger than 0.02 nats with three seeds" is a useful answer that saves a large run. "It works" with one seed and a different token budget is not an answer at all, and it is the most common way small reproductions go wrong.

## The idea

### Reproduce, extend, defend

The capstone has three parts, and each has its own failure mode:

1. **Reproduce.** Take one claim as the source states it, including its setting, and test the part of it your scale can test. Say in advance what would count as *reproduced* (the effect appears with the stated sign and is larger than your minimum detectable effect, or an equivalence claim holds within a stated margin), *not tested* (the mechanism never engaged: a clip that never fires, a stability problem that never appears), and *not reproduced* (the mechanism engaged and the effect is absent or reversed beyond the noise). A result that cannot be one of these three was not planned.
2. **Extend.** Add exactly one new ablation: a comparison the source did not run (QK-Clip against QK-norm), a variable it held fixed (the share of the candidate data in a micro-anneal), or a regime it did not reach (GSPO under bounded staleness). One, because each extension needs its own seeds, its own contract line and its own budget, and two half-powered extensions are worth less than one well-powered one.
3. **Defend.** Write the package so that a reviewer can check it, answer their questions in writing, and revise. Lesson 20.2 covers this part.

The capstone is the last link of the course's artefact chain: its run cards point back to the runs they branch from (Baseline-0, a Module 7 or Module 14 run, the Module 13 post-trained model), so the chain from Data-v0 to the capstone is traceable. It starts from your Module 19 proposal and its filled experiment contract ([Module 19 project](../../projects/module-19-capstone-proposal.md)); lesson [19.3](../module-19/lesson-03.md) covered how to write and defend results, and [19.2](../module-19/lesson-02.md) what "reproduced" means.

### The claim list

Each claim below was checked on 2026-10-07 against the section or figure named, in the primary source. The full records (statement, code, budgets, reviewer questions) are data in `frontierlab.capstone.claims`; `python labs/module-20/lesson-01/capstone_lab.py --claims` prints them. Main-path GPU-hours are **PROJECTED (pending the Module 20 pilot)**; the formula behind each is in the record and summarised under *Shapes and cost*.

| Claim | Primary source, where | Builds on | Main path, PROJECTED | Free CPU |
|---|---|---|---|---|
| GSPO vs GRPO stability | GSPO 2507.18071, §4.1 Eqs. 5 and 7, §5.1–5.3, Figures 1–3 | 12.2, 12.3, 14.1–14.3; `rlscale` | 27–44 GPU-h | ~20–25 min (estimated from 14.2) |
| QK-Clip vs QK-norm | Kimi K2 2507.20534 §2.1, Figure 2, Appendix D, Figure 12; DeepSeek-V4 §2.3.3, §2.4 | 03.3, 07.1, 07.2, 07.5; `optim` | ~20 GPU-h | 19.4 min (measured, lab below) |
| DSA quality and cost vs dense | DeepSeek-V3.2 2512.02556 §2.1–2.3, Figure 3 | 04.1, 05.2, 05.3; `attention.dsa` | ~10 GPU-h | ~85 min (estimated from 05.2) |
| mHC vs HC stability | mHC 2512.24880 §3.1, Figures 2–3, §4.2–4.3, §5.4, Figure 7 | 06.2, 06.3, 07.5; `blocks.hyperconn` | 9–15 GPU-h | ~30–40 min (estimated from 06.2) |
| Micro-anneal data scoring | OLMo 2 2501.00656 §4.4.2, Table 12; Olmo 3 2512.13961 §3.5.1 | 10.1, 10.2, 10.4; `datax` | 1.5–2.5 GPU-h | ~25 min (estimated from 10.4) |
| On-policy vs off-policy distillation at equal compute | Thinking Machines blog 2025-10-27; GKD 2306.13649 §3–4; Qwen3 §4.5 | 13.1, 13.2; `pipeline.distill`, `pipeline.compute` | 13–19 GPU-h | ~25 min (estimated from 13.2) |

**1. GSPO vs GRPO stability** (PROMISING; strongest on MoE). The claim: GSPO clips a length-normalised sequence-level importance ratio $s_i(\theta) = \big(\pi_\theta(y_i\mid x)/\pi_{\text{old}}(y_i\mid x)\big)^{1/|y_i|}$ (Eq. 7) with left and right ranges $3\times10^{-4}$ and $4\times10^{-4}$, against 0.2 and 0.27 for the GRPO baseline (§5.1); its training "proceeds stably throughout" on a cold-start model fine-tuned from Qwen3-30B-A3B-Base (Figure 1); GRPO needed Routing Replay for its MoE runs to converge and GSPO did not (§5.1, §5.3, Figure 3); and GSPO trains more efficiently although it clips about two orders of magnitude more tokens (§5.2, Figure 2). Your reproduction is on lesson 14.2's dense setup, so the MoE part is **not testable**: say so instead of extrapolating. A sound null: no arm crosses a stability threshold stated before the runs and the accuracy interval includes zero (lesson 14.2 measured exactly this on CPU: GSPO − GRPO = −0.036 [−0.083, +0.012], no threshold broken). Suggested extension: both objectives under lesson 14.3's bounded-staleness sampler (lag ≤ 4), or GSPO-token as a third arm.

**2. QK-Clip vs QK-norm** (QK-Clip PROMISING / MODEL-SPECIFIC; QK-norm ESTABLISHED). Kimi K2 reports that with vanilla Muon the maximum attention logits of a 9B-activated / 53B-total MoE "quickly exceed a magnitude of 1000" (§2.1, Figure 2 left), and that QK-Clip, $\gamma_h = \min(1, \tau/S^h_{\max})$ applied per head after each update, keeps them bounded ($\tau = 100$ for K2). Appendix D (Figure 12) trains two 0.5B-activated / 3B-total MoE models, vanilla Muon against QK-Clip at an aggressive $\tau = 30$, and finds "negligible impact on loss". DeepSeek-V4 instead applies RMSNorm to queries and KV entries and states it does not use QK-Clip (§2.3.3, §2.4). No report compares the two fixes directly: that is the extension. A sound null: the clip binds and clip − no-clip lies inside the margin (a reproduction of "negligible impact"); if the logits never reach $\tau$, the runs are bit-identical and the claim is **not tested** (lesson 07.2 found this at $\tau = 100$). This is the claim the lab runs end to end.

**3. DSA quality and cost vs dense** (MODEL-SPECIFIC, company claim). DeepSeek-V3.2 continues training with DeepSeek Sparse Attention: each query attends to 2,048 keys chosen by a lightning indexer, after a dense warm-up of 1,000 steps (2.1B tokens) in which only the indexer trains, by a KL loss to the L1-normalised main attention, then 15,000 sparse steps (943.7B tokens) with the KL restricted to the selected set and the indexer's input detached (§2.1.1). It reports "no substantial performance degradation" against DeepSeek-V3.1-Terminus on short and long context (§2.2, no numeric table; AA-LCR four points higher), core attention reduced from $O(L^2)$ to $O(Lk)$ while the indexer stays $O(L^2)$ (§2.3), and lower serving cost on H800 at USD 2 per GPU-hour (Figure 3). A sound null: quality kept within the margin with indexer recall reported, and on CPU no speed-up at any length you can run (lesson 05.3 measured no prefill crossover up to 8K): report component costs and a projected crossover, never an end-to-end speed-up you did not measure. Extension: top-$k$ swept, or the warm-up removed.

**4. mHC vs HC stability** (PROMISING; no independent replication). At 27B, unconstrained hyper-connections show "an unexpected loss surge around the 12k step" with a gradient-norm spike (§3.1, Figure 2) and an Amax gain magnitude of the composite residual mapping with "peaks of 3000" (Figure 3b). mHC projects the residual mixing onto doubly stochastic matrices with Sinkhorn–Knopp ($t_{\max} = 20$, $n = 4$; §4.2), keeps the composite gain at most about 1.6 (§5.4, Figure 7b) and costs 6.7% training time at $n = 4$ after kernel work (§4.3). A sound null: HC's gain grows and mHC's stays near 1 while no arm spikes (lesson 06.2 saw gains of 6.6 for HC at a raised rate and no spikes): "the mechanism is reproduced, the loss instability is not, at this scale". Extension: Sinkhorn iterations cut from 20 to 3, or $n = 2$.

**5. Micro-anneal data scoring** (PROMISING; two labs). OLMo 2 scores candidate sources by annealing on a 50/50 mix of the candidate and general web data with the learning rate taken linearly to zero, "at a fraction of the cost of a full annealing run": 19 microanneals totalling 130B tokens, fewer than three full 50B anneals, judged on MMLU and a 200-question GSM8K subset (§4.4.2, Table 12). Olmo 3 standardises it as 5B target tokens plus 5B web tokens against a web-only microanneal of the same 10B tokens (§3.5.1). A sound null: a candidate whose gain interval includes zero, reported with the control anneal's own effect (lesson 10.4 measured −0.105 nats from annealing alone, before any candidate). Extension: the candidate's share cut from 50% to 10% (OLMo 2's first microanneal experiment).

**6. On-policy vs off-policy distillation at equal compute** (PROMISING; company blog plus GKD). Thinking Machines report that a Qwen3-8B-Base student, fine-tuned on 400k prompts to 60% on AIME'24, reaches 70% after about 150 steps of on-policy distillation with a per-token reverse-KL reward, where off-policy fine-tuning on teacher outputs is extrapolated to need about 2M prompts: a cost reduction of 9× when the SFT data is given, about 18× in GPU-hours, about 30× when the teacher's sampling is charged (company blog). The comparison is against off-policy SFT, not RL. Your comparison holds training FLOPs equal *including teacher sampling and scoring* (`frontierlab.pipeline.compute.Ledger`). A sound null is likely and still valuable: lesson 13.2 measured the opposite direction on 2–4-token answers and explained why the setting differs (INFERENCE). Extension: longer answers, where prefix mismatch has room to matter.

### Choosing a claim

Choose by what your budget can *test*, not by what is most exciting. Three questions settle most choices: can the mechanism engage at your scale (a logit fix needs logits that grow; a stability fix needs an instability)? Is there a course lab whose code and measured noise floor you can reuse? Can you state, before running, the effect size you could detect (lesson 01.4) and compare it with the size the source reports? If the honest answer to the first is "probably not", the claim can still be a capstone, but its contract must say that a "not tested" result is the likely outcome and what fallback evidence (published figures, course pilot traces, labelled as analysis) you will use.

### The rubric

The capstone is scored with the course rubric ([`templates/experiment-rubric.md`](../../templates/experiment-rubric.md)) plus three capstone criteria. Each scores 0 (missing), 1 (partly) or 2 (done); a capstone passes at 15 of 20 with no zero. **Whether the new method wins is not a criterion**, and a sound null or negative result scores exactly the same as a positive one.

| # | Criterion | 2 = done looks like | Checked mechanically by |
|---|---|---|---|
| 1 | Question and decision | the claim quoted from its source with section or figure; the decision it informs; reproduced / not tested / not reproduced defined before the runs | `CONTRACT_*` |
| 2 | Controls | one changed variable per comparison, declared; everything else identical in the run cards | `PARITY` |
| 3 | Axis and budget parity | the axis named and justified; equal budgets on it; equal tuning budget per arm | `CONTRACT_AXIS`, `PARITY`, `TUNING` |
| 4 | Correctness | every correctness check passed and its output included | `CONTRACT_UNCHECKED`; `checks.json` |
| 5 | Uncertainty | at least two seeds per arm (three on the main path); paired intervals over seeds and items; the noise floor and MDE | `UNC_*`, `SEEDS` |
| 6 | Conclusion matches evidence | every claim labelled; measured claims follow the pre-stated rule and stay at course scale | `UNC_DECISION`, `CLAIM_*`, `REPORT_OVERCLAIM` |
| 7 | Limits | scale, data, architecture, what result would change the answer | `REPORT_LIMITS` (presence only) |
| 8 | Reproduction scoped | the part of the claim the scale can test is separated from the part it cannot | by the reviewer |
| 9 | One extension, powered | one new ablation with its own comparison line, seeds and budget | `PKG_ROLES` (presence only) |
| 10 | Defence and revision | every reviewer question answered with evidence; a revision log (lesson 20.2) | `DEF_*`, `REV_*` |

The checker can only find what is mechanical. A package that passes it can still score 1 on criteria 1, 7 and 8; a package that fails it cannot score 2 on the criteria its problems belong to.

### Paired uncertainty over seeds and items

Two arms of a capstone share two things: the seed (initial weights and data order) and the evaluation items (held-out windows, prompts). Let $a_{s,w}$ and $b_{s,w}$ be the metric of arms A and B for seed $s \in \{1..S\}$ on item $w \in \{1..W\}$, and $d_{s,w} = a_{s,w} - b_{s,w}$ the paired difference. The estimate is $\bar d = \frac{1}{SW}\sum_{s,w} d_{s,w}$.

Two noise sources move $\bar d$: which seeds you happened to run, and which items you happened to evaluate on. A bootstrap over items alone (lesson 01.4's `paired_bootstrap`) ignores the first and is too narrow whenever arms differ from seed to seed. The **hierarchical (two-level) bootstrap** resamples both: draw $S$ seed indices $s^*_1..s^*_S$ with replacement, then for each drawn seed $W$ item indices with replacement, and average $d$ over the drawn pairs; repeat $B$ times and take the 2.5% and 97.5% quantiles. Report next to it the **paired t-interval over seed means** $\bar d_s = \frac1W\sum_w d_{s,w}$: $\bar d \pm t_{0.975, S-1}\, \mathrm{sd}(\bar d_s)/\sqrt S$. With few seeds the two disagree in a known way: the bootstrap's seed level can only produce $\binom{2S-1}{S}$ distinct seed resamples (10 for $S = 3$) and tends to be too narrow, while the t-interval uses $t_{0.975,2} = 4.30$ and is honest but wide. The course decides on the bootstrap and reports both; if they disagree about the decision, the report says so.

The **decision rule** has an equivalence margin $m$, fixed in the contract before the runs:

$$\text{decision} = \begin{cases} \text{equivalent} & -m \le \ell \text{ and } u \le m \\ \text{a lower} & u < 0 \\ \text{a higher} & \ell > 0 \\ \text{inconclusive} & \text{otherwise} \end{cases}$$

checked in that order, for the interval $[\ell, u]$ of $\bar d$. "Equivalent" comes first because a difference that is real but smaller than $m$ does not change the decision the experiment informs. Without a margin, a reproduction of "negligible impact" can never succeed: an interval that includes zero is "inconclusive", not "no effect".

## Worked example

**A two-level resample by hand.** Two seeds, two items, the differences in nats:

$$d = \begin{pmatrix} 0.01 & 0.03 \\ -0.01 & 0.01 \end{pmatrix}, \qquad \bar d = 0.01.$$

One resample draws seeds $(2, 2)$; for the first draw items $(1, 2)$, for the second $(2, 2)$: values $-0.01, 0.01, 0.01, 0.01$, mean $0.005$. Another draws seeds $(1, 1)$ with items $(2, 2)$ and $(1, 2)$: $0.03, 0.03, 0.01, 0.03$, mean $0.025$. The spread of such means is the interval. The seed-level t-interval: seed means $0.02$ and $0.00$, $\mathrm{sd} = 0.0141$, standard error $0.01$, $t_{0.975,1} = 12.71$, so $0.01 \pm 0.127$: with two seeds, the seed level alone cannot tell anything apart, which is why the main path asks for three or more.

**The decision.** Margin $m = 0.02$ nats:

- $[-0.004, +0.012]$ lies inside $[-0.02, +0.02]$: **equivalent** (a reproduction of "negligible impact on loss").
- $[+0.004, +0.031]$ excludes zero and leaves the margin: **a higher**.
- $[-0.031, +0.017]$ includes zero and leaves the margin: **inconclusive**. Not "no effect".

**The noise floor.** Baseline seed means $0.402, 0.371, 0.388$: $\mathrm{sd} = 0.0155$. With three seeds per arm the minimum detectable effect at 80% power (lesson 01.4) is $2.8 \cdot 0.0155 \cdot \sqrt{2/3} = 0.035$. A claimed effect of 0.01 is far below it: the design cannot detect it, and the contract must say so before the runs, not after.

**A projected budget.** The QK-Clip claim's pilot-30m rung: 3 arms × 3 seeds = 9 runs of 4,000 steps × 64 × 1,024 = $2.62\times10^8$ tokens; `flops_per_token(pilot_30m(32768), 1024)` $= 3.21\times10^8$ training FLOPs per token. At an assumed 20% MFU of an H100's $989\times10^{12}$ FLOP/s and 3% for the logit probe:

$$\frac{9 \cdot 2.62\times10^8 \cdot 3.21\times10^8}{989\times10^{12}\cdot 0.20}\cdot 1.03 = 3{,}941\ \text{s} \approx 1.1\ \text{GPU-hours (PROJECTED)}.$$

## Shapes and cost

| Item | Shape, dtype, device | Size or cost |
|---|---|---|
| per-window losses of one arm | (S, W) = (3, 256) float64, CPU | 6 KB; the bootstrap draws (B, S, W) = (4000, 3, 256) int64 indices, 25 MB, under a second |
| run card | `runs/<arm>-s<seed>/run_card.yaml` | 2–4 KB each; one per arm and seed |
| package | `claim.yaml`, `contract.md`, `results.json`, `claims.yaml`, `report.md`, `checks.json` | under 50 KB without checkpoints |
| QK-Clip capstone, free CPU | toy (1.8M parameters), 9 runs × 200 steps × 8 × 128 tokens, lr $10^{-2}$ | measured 1,164 s on a shared 16-thread laptop (9 runs of 110–138 s, then evaluation) |
| QK-Clip capstone, main path | pilot-30m, pilot-70m and Baseline-0 rungs, 9 runs each | PROJECTED 1.1 + 2.1 + 17.0 ≈ 20 GPU-hours (Baseline-0: 9 × 2.49e9 tokens × 7.88e8 FLOPs per token at 30% MFU) |
| the other five claims | see the claim list | PROJECTED from each earlier lesson's formula, stated in `frontierlab.capstone.claims` |

The plan's estimate for the capstone was 50–150 H100-hours (plan section 8). The list's projections are 2–44 GPU-hours because they reuse earlier labs' sizes; a capstone that adds a larger rung (Baseline-0 for the RL claims, a 70M rung for DSA) should project it with the same formula before asking for the hours.

## Build it

`frontierlab.capstone` holds the pieces; your lab rewrites four of them.

```python
from frontierlab.capstone import uncertainty as U, package as PK, claims as CL

r = U.hierarchical_bootstrap(a, b)           # a, b: (seeds, items) arrays, paired by seed and item
U.decide(r["mean_diff"], *r["ci"], margin=0.02)
CL.get("qkclip").null_result                 # what a sound null looks like, before you run
PK.check_package("runs/m20/l201/capstone-qkclip")   # [] when sound; else Problem(code, where, message)
```

The checker reads the package and applies five groups of checks: the **contract** is filled (every template section, a hypothesis status, a named axis that matches `claim.yaml`, a decision rule with numbers, every correctness box ticked); every arm and seed has a **run card** whose `parent_run` is another run of the package or a declared external parent; for every comparison and seed, `frontierlab.record.diff_cards` finds nothing that **invalidates** it on the declared axis with the declared changed variables, tuning budgets match and seeds match; **uncertainty** is reported (two or more seeds, an interval that contains its mean and names its method, a mean that matches the per-seed values, a decision that the stated rule reproduces from the interval, the noise floor); and every **claim** carries an evidence label, a MEASURED claim says no more than its recomputed decision allows and stays at course scale, and the report has a Limits section and no proof language. `python -m frontierlab.capstone <package>` exits 1 on any problem.

The scaffold (`frontierlab.capstone.scaffold`) runs one claim end to end: correctness checks first (QK-Clip caps recomputed per-head logits at exactly $\min(S_{\max}, \tau)$ in float64, $3.6\times10^{-15}$ measured; QK-Clip refuses a QK-norm model; causal and cached-decode checks), then the package skeleton (claim and contract) before any run, then the runs through the unmodified course loop, then evaluation, results, draft claims and report, then the checker. Tests: `pytest labs/common/tests/test_capstone.py` (26 tests, about 80 s).

## What the evidence says

- **The six claims are PUBLICLY DOCUMENTED** at the locations given, each by the lab that proposed the technique; the distillation numbers are a company blog (company claim). None has a controlled independent replication at the source's scale that the course could find on 2026-10-07; that is what makes them good capstones. Maturity tags as in the earlier lessons: GSPO PROMISING (MODEL-SPECIFIC for dense models), QK-Clip PROMISING / MODEL-SPECIFIC and QK-norm ESTABLISHED, DSA MODEL-SPECIFIC, mHC PROMISING, micro-anneals PROMISING, on-policy distillation PROMISING.
- **What the earlier course labs already measured** at CPU scale, so you know the likely outcome: no GSPO/GRPO stability separation (14.2); Muon logits around 27 at toy scale, a clip that never fires at $\tau = 100$ (07.2); DSA quality kept within 0.003 nats and no CPU prefill crossover (05.2, 05.3); HC gain growing to 6.6 with no loss spikes (06.2); micro-anneal rankings with the control anneal's own −0.105 nats (10.4); on-policy distillation behind off-policy per FLOP on short answers (13.2). These are course measurements at toy scale, not evidence about the published settings.
- **The hierarchical bootstrap and the equivalence margin** are REASONABLE INDUSTRY PRACTICE (METR's time-horizon intervals use a hierarchical bootstrap; equivalence testing with a margin is standard in clinical statistics). The specific rule and margin are the course's choice, and the rubric is the course's.
- **Open question you can contribute to:** whether any of these effects grows or shrinks along the course's scale ladder. A capstone that runs two rungs and reports the trend relative to the noise floor (plan section 12.1) answers that better than one large run.

## Lab

**Folder:** [`labs/module-20/lesson-01/`](../../labs/module-20/) · **Time:** about 2.5 hours (of which about 20 minutes of CPU runs, measured) · **Pass check:** `pytest labs/module-20/lesson-01` passes; `capstone_lab.py` writes a package that `python -m frontierlab.capstone runs/m20/l201/capstone-qkclip` reports as sound; you have a one-page capstone plan for your own claim.

### Experiment contract

The scaffold writes this contract into the package before any run (`contract.md`); it is the worked model for yours.

- **Question:** at equal tokens, does QK-Clip at a $\tau$ that binds change held-out loss relative to unclipped Muon by more than 0.02 nats (Kimi K2 Appendix D), and does it differ from QK-norm by more than that (extension)? Decision informed: which logit fix later Muon runs use for GQA models.
- **Hypothesis:** H1 clipping costs less than the margin (**reported effect**, K2 Appendix D, at 0.5B activated / 3B total); H2 QK-norm and QK-Clip equivalent within the margin (**may not appear at this scale**, no report tests it); H3 the baseline's maximum logit grows (**reported effect**, K2 Figure 2; measured at toy scale in lesson 07.2).
- **Baseline:** `muon-noqk` (Muon, no QK-norm, no clip), parent `m07-l72-muon-noqk`; not re-tuned: every arm uses lesson 07.2's raised learning rate $10^{-2}$, one trial each.
- **Changed variable:** the logit fix (none, QK-Clip at $\tau$, QK-norm). $\tau$ follows a rule fixed before the clip runs: $\mathrm{round}(0.6 \times S_{\max})$ of the seed-0 baseline at its last step, so the clip binds for the last part of training. **Controlled:** Data-v0, toy preset, 200 steps × 8 × 128 tokens, seeds 0–2 (same initialisation and data order per seed across arms), the same 256 validation windows of 128 tokens.
- **Comparison axis:** equal tokens. It does not say which fix is cheaper per step.
- **Budget:** free CPU, 9 runs, measured 1,164 s in total; main path PROJECTED about 20 GPU-hours.
- **Metrics and decision rule:** held-out loss, paired hierarchical bootstrap over seeds and windows (95%), paired seed t-interval alongside; equivalent if the interval lies inside ±0.02 nats, else lower/higher if it excludes zero, else inconclusive; "not tested" if the clip never fires. Secondary: maximum logit, clipped head-updates, loss spikes.
- **Correctness checks:** the scaffold's three checks; `pytest labs/common/tests/test_capstone.py`; the package checker.
- **Fallback evidence:** lesson 07.2's induced $\tau = 15$ runs and K2 Figures 2 and 12, labelled analysis of published results.
- **Limits:** a 1.8M-parameter dense model for 200 steps, not a 3B MoE on trillions of tokens; GQA, where QK-Clip scales only query rows (course choice, INFERENCE from K2's MLA rule); one learning rate; logits of order 10.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100 80 GB. Not run in this build; part of the Module 20 pilot | `python labs/module-20/lesson-01/capstone_lab.py --variant main --print` prints the pilot-30m commands (9 runs of 4,000 steps × 64 × 1,024, bf16); repeat at pilot-70m and Baseline-0 for the ladder. **PROJECTED** about 20 GPU-hours (formula above). Pick your own claim's main path from the list |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `--variant t4 --print`: pilot-10m, 2,000 steps × 32 × 512, 9 runs; PROJECTED 1.5–2.5 hours. You will see the trend at a second size but not logits anywhere near K2's 1,000 |
| Free CPU | laptop; measured 19.4 minutes for the runs | the steps below. At this scale logits stay near 10, so the clip only binds because $\tau$ is set to bind: the run tests the *mechanism's cost*, not K2's failure mode |

### Steps

1. **Read the list and choose.** `python labs/module-20/lesson-01/capstone_lab.py --claims`. For the claim you proposed in Module 19, write down: the sentence from the source (with section or figure), the part your scale can test, what would count as reproduced, not tested and not reproduced, and your extension.
2. **Implement** the four TODOs in `lab.py` (`hierarchical_interval`, `decide`, `claim_problems`, `comparison_problems`) and run `pytest labs/module-20/lesson-01`.
3. **Run the scaffold** with your functions: `python labs/module-20/lesson-01/capstone_lab.py`. Read the package it writes in `runs/m20/l201/capstone-qkclip/`: the contract (written before the runs), the run cards (look at `parent_run`), `results.json`, the draft `claims.yaml` and `report.md`.
4. **Break it on purpose.** Change one run card's learning rate, delete a seed from one arm, or edit a claim's direction, and run `capstone_lab.py --check runs/m20/l201/capstone-qkclip`. Each edit should produce exactly the problem you expect, from both the course checker and your checks. Restore the files.
5. **Plan your own capstone** (one page): the claim and its source location, the arms and their roles, the comparison lines with their changed variables, the axis, seeds, margin and MDE from the earlier lesson's noise floor, the projected main-path GPU-hours with the formula, and the free variant you will actually run. This becomes the [Module 20 project](../../projects/module-20-capstone.md).

<details>
<summary>Hint for TODO 1</summary>

Draw all indices at once: `s_idx = rng.integers(0, S, size=(n_boot, S))` and `w_idx = rng.integers(0, W, size=(n_boot, S, W))`, then `d[s_idx[:, :, None], w_idx]` has shape (n_boot, S, W). Average over the last two axes. The test compares your interval with the reference within 15% of its width, so you do not need the same random draws.

</details>

<details>
<summary>Hint for TODO 4</summary>

`diff_cards` accepts dicts. Pass the baseline's card first (`card_b`, then `card_a`) so the finding's `a` and `b` read as "from b to a"; only the `severity` matters for the test.

</details>

<details>
<summary>What the build's run gave (compare after your own run)</summary>

Measured 2026-10-07 on the build laptop (Windows 11, Python 3.12.13, torch 2.14.1+cpu, 8 threads, another module's jobs sharing the CPU), `python -m frontierlab.capstone.scaffold`, which runs the same arms as `capstone_lab.py` with the reference functions: **1,164 s in total**, of which 9 training runs of 110–138 s each. Correctness checks: QK-Clip cap error $3.6\times10^{-15}$, refusal of a QK-norm model, causal and cached-decode differences $0$ and $7.8\times10^{-16}$: passed. The package checker reported no problems.

$\tau$ by the rule: the seed-0 baseline ended at a maximum logit of 13.37, so $\tau = \mathrm{round}(0.6 \times 13.37) = 8$. The mechanism engaged:

| Arm | Run max logit (3 seeds) | Final max logit | Clipped head-updates | First clip step | Loss spikes |
|---|---|---|---|---|---|
| `muon-noqk` | 14.2, 15.8, 15.0 | 13.4, 13.2, 14.1 | 0 | none | 0 |
| `muon-clip` ($\tau = 8$) | 9.1, 9.9, 8.9 | 8.0, 8.1, 8.1 | 114, 94, 91 | 120, 111, 118 | 0 |
| `muon-qknorm` | 10.2, 9.2, 10.8 | 9.5, 9.0, 10.6 | 0 | none | 0 |

(The clipped run's logged maximum is measured before the clip on each step, so it can exceed $\tau$; after the clip every head is at most $\tau$.)

| Comparison | Mean (nats) | Hierarchical bootstrap 95% | Seed t 95% | Decision (margin 0.02) |
|---|---|---|---|---|
| reproduction: clip − no-clip | +0.0015 | [+0.0013, +0.0018] | [+0.0013, +0.0018] | equivalent |
| extension: QK-norm − QK-Clip | +0.0205 | [+0.0056, +0.0332] | [−0.0170, +0.0581] | a higher, by the stated rule; the t-interval disagrees |

Noise floor: the baseline's seed std is 0.0080 nats, so the MDE with three seeds is 0.018.

Reading it. The reproduction is the instructive kind of "equivalent": the interval *excludes* zero (clipping cost 0.0015 nats, with the same sign in all three seeds) and still lies thirteen times inside the margin, so the difference is real and too small to matter, which is what Kimi K2's Appendix D claims at its own scale. The extension is the instructive kind of disagreement: the per-seed differences are +0.033, +0.025 and +0.004, the bootstrap (whose seed level has only 10 distinct resamples with three seeds) says "a higher", the t-interval includes zero, and the effect is about the size of the MDE. The honest sentence is "by the pre-stated rule QK-norm was 0.02 nats worse than QK-Clip at this scale, but the seed-level interval includes zero; the pilot-70m rung with more seeds is the test". Lesson 07.2 saw the same direction with one seed (+0.029 nats for QK-norm against unclipped Muon). None of this says anything about logits of 1,000 or about MoE models.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-20/lesson-01/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-20/lesson-01`.

</details>

## Common mistakes

- **Reproducing the headline, not the claim.** "GSPO is more stable" is not the claim; "GSPO trained stably where GRPO needed Routing Replay on an MoE model" is, and a dense toy model cannot test its second half. Quote the setting.
- **A mechanism that never engaged.** A clip that never fires, a stability problem that never appears: the runs are identical by construction. Report "not tested", not "no effect".
- **No margin, then claiming "no difference".** An interval that includes zero is inconclusive. To claim equivalence, state the margin before the runs and show the interval inside it.
- **Items-only intervals.** A bootstrap over evaluation windows of one seed per arm measures how noisy the evaluation is, not how noisy training is. Resample seeds too, and run at least two (three on the main path).
- **Tuning the favoured arm more.** Three learning rates for the new method and one for the baseline is the oldest bias in the field; the checker compares the `tuning` counts.
- **A second extension instead of a powered first one.** Each extension needs its own seeds; two at one seed each are two anecdotes.
- **Frontier-scale sentences from toy runs.** Your measurement is about your scale. Restate the paper's claim as PUBLICLY DOCUMENTED, and keep your MEASURED claims at course scale.

## References

- C. Zheng et al., *Group Sequence Policy Optimization*, 2025: section 4.1 (Eqs. 5, 7), sections 5.1–5.3, Figures 1–3. https://arxiv.org/abs/2507.18071
- Kimi Team, *Kimi K2: Open Agentic Intelligence*, 2025: section 2.1 and Figure 2; Appendix D and Figure 12. https://arxiv.org/abs/2507.20534
- DeepSeek-AI, *DeepSeek-V4*, 2026: sections 2.3.3 and 2.4. https://arxiv.org/abs/2606.19348
- DeepSeek-AI, *DeepSeek-V3.2*, 2025: sections 2.1–2.3, Figure 3. https://arxiv.org/abs/2512.02556
- Z. Xie et al., *mHC: Manifold-Constrained Hyper-Connections*, 2025: sections 3.1, 4.2–4.3, 5.2, 5.4; Figures 2, 3, 7; Table 4. https://arxiv.org/abs/2512.24880
- Team OLMo, *2 OLMo 2 Furious*, 2024: section 4.4.2, Table 12. https://arxiv.org/abs/2501.00656
- Team Olmo, *Olmo 3*, 2025: section 3.5.1. https://arxiv.org/abs/2512.13961
- K. Lu and Thinking Machines Lab, *On-Policy Distillation*, 2025-10-27. https://thinkingmachines.ai/blog/on-policy-distillation/
- R. Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, 2023. https://arxiv.org/abs/2306.13649
- Google Research, *Deep Learning Tuning Playbook*. https://github.com/google-research/tuning_playbook
- Course templates: [experiment contract](../../templates/experiment-contract.md), [run card](../../templates/run-card.md), [experiment rubric](../../templates/experiment-rubric.md).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[20.2 · Review and defence](lesson-02.md)
