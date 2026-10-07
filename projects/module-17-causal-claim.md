# Module 17 project · One supported causal claim

This project asks for one thing, done properly: a causal claim about how an open model produces a behaviour, scoped to the model and distribution you tested, backed by an intervention with a 95% interval, a control that shows the effect is specific to the components you name, a held-out test on prompts you did not choose them on, an off-target check, and a limitations section. You rehearse the pipeline on the induction model, where the answer is known, then make your claim about indirect-object identification (or another behaviour you define with clean/corrupt pairs) in Qwen3-0.6B on the free path or Qwen3-1.7B-Base on the main path. Before trusting the harness, you debug a colleague's version with three planted bugs that each make a claim look stronger than it is.

**Time:** 5–7 attended hours plus unattended runs. **Folder:** [`labs/module-17/project/`](../labs/module-17/) (`run_project.py`, `harness.py`, `buggy_harness.py`, `test_harness.py`). **Assessment:** self-check against the [experiment rubric](../templates/experiment-rubric.md); the module quiz covers the same material.

## Variants and cost

| Variant | Model and behaviour | Hardware | Cost |
|---|---|---|---|
| Main path | Qwen/Qwen3-1.7B-Base (`ea980cb`), the Stage D base model; IOI with 96 prompts per family, top-10 candidate, 39 random sets: `python labs/module-17/project/run_project.py --kind ioi --model qwen3-1.7b-base --device cuda --n 96 --top 10 --random 39`; optionally a second claim about a Qwen-Scope feature (lesson 17.1) tested with the same controls | 1× L40S/A100/H100 | **PROJECTED, pending the Module 17 pilot:** about $10^{15}$ FLOPs of forward passes, 15–30 minutes wall-clock, under 0.5 GPU-hours, USD 1–2 at USD 2–3 per H100-hour |
| Free GPU (Colab/Kaggle T4) | Qwen3-1.7B-Base in float32, `--n 64 --random 19` | T4 | PROJECTED 30–60 minutes |
| Free CPU | the induction rehearsal (seconds), then Qwen3-0.6B (`c1899de`), `--kind ioi --model qwen3-0.6b` | laptop | measured 13.7 minutes for the IOI run (both controls); 2 s for the rehearsal once lesson 17.2's model exists |

## Deliverables

1. **The experiment record:** your filled contract (below), the commands, the JSON records in `runs/m17/`, the outputs of `pytest labs/common/tests/test_interp.py` and `pytest labs/module-17/project`.
2. **The claim card** (`frontierlab.interp.claims.ClaimCard`), with `claims.check(card)` returning an empty list — or, if it does not, the narrower claim the evidence does support and the card for that.
3. **The selection evidence:** the attribution-patching screen, the real patches of the screened heads, and a calibration of the screen against real patching on random heads (lesson 17.2).
4. **The controls:** the distribution of random-set effects with the candidate's rank, and the held-out effect with what makes the held-out prompts held out.
5. **The off-target check** and what it implies for the word "specific".
6. **The limitations section:** at least three limits, one naming the model scale, and the result that would change your conclusion.
7. **The debugging report** and **the written defence** (below).

## The experiment contract

Fill in your own copy of the [contract template](../templates/experiment-contract.md) before running. Fixed by the project:

- **Question:** which attention heads of the model are necessary for IOI on the selection distribution, and does their necessity transfer to other templates and names? Decision informed: which components Module 18's audit monitors when it looks inside the post-trained model.
- **Hypothesis:** a small set of heads (at most 10) carries most of the logit difference; ablating it removes more than any random set of the same size and transfers to held-out templates. Status: reported for GPT-2 small (Wang et al. 2022); unknown for Qwen3.
- **Baseline:** the clean and corrupt runs (the ends of the normalised effect); random head sets of the candidate's size (the control).
- **Changed variable:** which heads are mean-ablated. **Controlled:** model and revision, float32 forward passes, the prompt generator and seed, the ABC corruption, the mean-ablation reference (the corrupt prompts), the metric (logit difference at the last position).
- **Comparison axis:** the same prompts for every intervention; random sets of exactly the candidate's size, drawn from the heads outside it with the same number of heads in each of the candidate's layers (pre-stated); random sets from any layer reported as a secondary control.
- **Metrics and decision rule:** normalised effect (divided by the mean clean-corrupt gap) with a 95% bootstrap interval over prompts; the claim is supported if `claims.check` passes: selection interval above 0; $p_{\text{random}} \le 0.05$ with at least 19 draws; held-out interval above 0 and at least half the selection effect; off-target change below 0.05 nats or reported as a limitation; at least three limitations.
- **Correctness checks:** `pytest labs/module-17/project` (the harness passes its derivation tests); `pytest labs/common/tests/test_interp.py`; the induction rehearsal recovers head 0.2 and passes its card.
- **Fallback evidence:** the IOI paper's circuit for GPT-2 small, labelled as published, if no candidate passes.
- **Limits:** one behaviour, one model size, one ablation reference, necessity only (sufficiency only in the screen), heads only (no MLPs or directions).

<details>
<summary>What the build's free-CPU runs gave (compare after your own report)</summary>

Measured 2026-10-07 on the build laptop (torch 2.14.1 CPU, 8 threads).

**Induction rehearsal** (2 s once the model exists): candidate head 0.2 (chosen by path patching into the layer-1 keys); mean-ablation effect 1.037 [0.928, 1.141]; 19 layer-matched random single heads (drawn from the 3 other layer-0 heads, so draws repeat): at most 0.019, $p_{\text{random}} = 0.05$, and from any layer at most 0.049 — a weak control with so few alternatives, as the card's limitations say; held-out (gap 12, query position 2) 1.000 [0.924, 1.079]; `claims.check`: supported as scoped.

**IOI on Qwen3-0.6B** (`c1899de`, float32, 13.7 minutes): 48 selection prompts from the three "train" templates and 12 names; 48 held-out prompts from the two "heldout" templates and the 8 other names. The attribution screen ranked 16 heads; real denoising patches kept the 8 largest (19.2 alone restores 0.46 of the gap; the others 0.10–0.26): heads 17.0, 17.3, 19.2, 19.8, 21.0, 22.8, 23.6, 27.15. Mean-ablating them removes 0.798 [0.724, 0.873] of the logit difference. Control (pre-stated): 19 random 8-head sets with the same number of heads in each of those layers remove at most 0.135, $p_{	ext{random}} = 0.05$. Held out: 0.953 [0.818, 1.086]. Off-target: ordinary-text loss +0.047 nats, just under the 0.05 limit. `claims.check`: supported as scoped.

The secondary control is the instructive part: 19 random 8-head sets drawn from *any* layer removed up to 1.576 of the gap, and 4 of 19 removed more than the candidate — three of those four contained head 1.5, a layer-1 head whose mean ablation appears to damage the whole computation. Against that control the claim would fail ($p = 0.25$). Both statements are true: these 8 heads matter more for IOI than other heads in the same layers, and they are not the most destructive 8 heads to remove. Which control you pre-register decides which claim you are making; the project pre-registers the layer-matched one and asks you to report the other.

</details>

## Debugging task

`labs/module-17/project/buggy_harness.py` is a colleague's harness. Their message is at the top of the file: their head 0.2 and their 8 IOI heads "pass every check", and they "simplified" three things — per-prompt normalisation, a fresh sample as the held-out set, random sets drawn from all heads. There are three planted bugs. For each one, name the rule of lessons 17.2 and the project contract it breaks, the test that isolates it, and how it changed the colleague's conclusion.

```bash
HARNESS=buggy pytest labs/module-17/project       # the derivation tests against their harness: three fail
HARNESS=buggy python labs/module-17/project/run_project.py --kind induction
python labs/module-17/project/run_project.py --kind induction
pytest labs/module-17/project                      # the course harness: all pass
```

<details>
<summary>Hint</summary>

Give `effect` two prompts, one with a clean-corrupt gap of 5 and one with a gap of 0.01, and see what each harness returns. Then print the gap and query position of the selection and held-out induction pairs, and the size of every random set and whether it contains the candidate.

</details>

<details>
<summary>Reference diagnosis</summary>

Measured 2026-10-07 on the induction model:

| Harness | effect [95% CI] | random sets | $p_{\text{random}}$ | held-out [95% CI] | check |
|---|---|---|---|---|---|
| colleague | 1.195 [0.973, 1.487] | max 1.195 | 0.150 | 1.343 [1.009, 1.875] | fails the control |
| course | 1.037 [0.928, 1.141] | max 0.019 (layer-matched) | 0.050 | 1.000 [0.924, 1.079] | supported |

On IOI in Qwen3-0.6B the colleague's harness *passes* the card: effect 0.973 [0.802, 1.163] (per-prompt normalisation), "held out" 1.056 [0.880, 1.290] (the same templates and names, new seed), and 19 single random heads remove at most 0.075 ($p_{	ext{random}} = 0.05$) — an 8-head ablation compared with 1-head ablations. The course harness gives 0.798, 0.953 on genuinely held-out prompts, and a layer-matched 8-head control.

**Per-prompt normalisation.** `effect` divides each prompt's change by its own clean-corrupt gap. Prompts whose gap is near zero produce huge ratios, so the mean effect and its interval are inflated (1.195 and an interval reaching 1.487 on the toy; effects above 1 mean "removed more than the whole behaviour", a sign of the bug). Isolating test: `test_effect_divides_by_the_mean_gap`. Rule broken: divide by the mean gap (lesson 17.2).

**The held-out set is a fresh sample of the selection distribution.** A new seed with the same gap and query position (or the same IOI templates and names) only tests sampling noise. Isolating test: `test_heldout_is_a_different_distribution`. It made "the held-out effect is just as large" true by construction (1.343 on the toy).

**Random sets of size 1 drawn from all heads.** The control compares an 8-head ablation with single heads, and can draw the candidate itself; it is also not layer-matched. On IOI that makes any 8-head candidate look specific; on the toy, where the candidate is one head, the candidate's own effect appears among the "random" sets, and the colleague's harness reports $p_{\text{random}} = 0.15$ for a head that is in fact the mechanism. Isolating test: `test_random_sets_match_size_and_exclude_the_candidate`. Rule broken: random components of the candidate's size from outside it.

The first two bugs push toward a stronger claim; the third pushes in either direction depending on the candidate's size. A harness that passes its derivation tests is a precondition for any number in the card.

</details>

## Written defence

One to two pages, answering:

1. State your claim in one sentence with its scope (model, revision, behaviour, prompt distribution, components). Which words in it does each piece of evidence support?
2. Why mean ablation over the corrupt prompts, and what would change with zero or resample ablation?
3. What is the smallest effect your control could distinguish from random sets with your number of draws?
4. How does the held-out effect compare with the selection effect, and what would you conclude if it were half as large?
5. What did the attribution-patching screen get wrong, and could it have changed which heads you chose?
6. What does the off-target check say about the word "specific"? Would you use your heads as a monitor in Module 18?
7. What result on the main path (Qwen3-1.7B-Base, 96 prompts, 39 random sets) would make you withdraw the claim?
8. Each planted bug: which test caught it and how it changed the colleague's conclusion.

## Self-check against the rubric

Score yourself with the [experiment rubric](../templates/experiment-rubric.md) (pass: 10 of 14 with no zero):

| # | Criterion | What "2" looks like here |
|---|---|---|
| 1 | Question and decision | the claim's scope and the decision it informs written before the runs |
| 2 | Controls | layer-matched random sets of the candidate's size from outside it, plus the any-layer control reported; the reference distribution named; the same prompts in every arm |
| 3 | Axis and budget parity | equal prompts and metric for every intervention; enough random draws for the stated significance |
| 4 | Correctness | the harness's derivation tests and `test_interp.py` pass; the rehearsal recovers the known head; the three bugs found with their tests |
| 5 | Uncertainty | bootstrap intervals for every effect; the candidate's rank among random sets |
| 6 | Conclusion matches evidence | the claim no stronger than the card; "specific" only if the off-target check supports it |
| 7 | Limits | scale, behaviour, ablation reference, necessity vs sufficiency, and what would change the answer |
