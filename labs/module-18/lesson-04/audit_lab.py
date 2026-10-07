"""Lab 18.4: read three safety frameworks and three recent system cards as evidence, then run a student audit of
your toy post-trained model against thresholds you stated in advance.

    python labs/module-18/lesson-04/audit_lab.py                    # free CPU, about 1-3 minutes
    python labs/module-18/lesson-04/audit_lab.py --part frameworks  # part A only (no model)
    python labs/module-18/lesson-04/audit_lab.py --variant main --print

Part A: the frameworks as the course read them on 2026-10-07 (``frontierlab.alignment.audit.FRAMEWORKS``) and
course-written summaries of what three recent system cards and reports let a reader check, with your ``card_gaps``.
Every entry is a company document (company claim); the summaries say only what the course verified on the page.

Part B: a student audit of the Module 13 spec-trained toy model (lesson 13.3, ``runs/m13/l133/judge-dpo-s0``;
the Module 13 project's final stage if it is missing, then the Module 12 warm start). Evidence: refusal of
restricted requests and answers to borderline ones (greedy, Spec-T exactly), capability on restricted problems under
elicitation (16 samples per item at temperature 1, pass@16 with your ``pass_at_k``), and the Eval v3 contamination
result from lab 18.3 if you ran it. Your ``decide`` applies your THRESHOLDS; ``overall`` writes the summary line;
the report is validated (disclaimer, labels, sources, limits, no deployment claims) and written to
``runs/m18/l184/audit.md``. This is a student exercise: it is not evidence that any model meets any developer's
thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from frontierlab.alignment import audit as AU
from frontierlab.alignment.monitors import wilson
from frontierlab.labkit import load_path
from frontierlab.pipeline import judge as J
from frontierlab.posttrain.policy import sample
from frontierlab.posttrain.sft import ensure_sft, load_policy
from frontierlab.posttrain.tasks import encode_prompts, response_text
from frontierlab.stats import bootstrap_ci

HERE = Path(__file__).resolve().parent
ROOT = Path("runs/m18/l184")
CANDIDATES = [(Path("runs/m13/l133/judge-dpo-s0/policy.pt"), "lesson 13.3 spec-trained model (DPO against the AI judge, seed 0)"),
              (Path("runs/m13/project/toy-s0/s3-distill/policy.pt"), "Module 13 project, final toy stage (no spec training)")]

CARDS = {
    "Claude Opus 5.5 system card (Anthropic, 2026-09-22)": {
        "url": "https://www.anthropic.com/claude-opus-5-5-system-card",
        "framework": "Responsible Scaling Policy; treated as above the CB-1 and below the CB-2 threshold, deployed with expanded biological safeguards",
        "capability_evals": ["chemical and biological", "cyber", "AI R&D"],
        "elicitation": "",
        "safeguards": "safeguards and harmlessness section; agentic safety (malicious use, prompt injection)",
        "alignment_evals": ["alignment assessment", "alignment risk update"],
        "evaluation_awareness": "section 6.6: verbalised grader awareness, possible sandbagging on dangerous-capability evaluations",
        "external_testing": ["US CAISI", "METR (pre-release, per the announcement)"],
        "limitations": "section 6.6 'Reliability of this assessment'",
    },
    "GPT-6 Astra system card (OpenAI, 2026-09-03)": {
        "url": "https://deploymentsafety.openai.com/gpt-6-astra",
        "framework": "Preparedness Framework v2; meets the Critical threshold in cybersecurity (the first model to do so)",
        "capability_evals": ["cybersecurity", "biological and chemical", "AI self-improvement"],
        "elicitation": "",
        "safeguards": "safeguards stack described for the Critical cyber capability",
        "alignment_evals": ["monitor evasion under prompted sandbagging"],
        "evaluation_awareness": "evaluation awareness reported in 9.6% of trajectories; reduced chain-of-thought monitorability",
        "external_testing": [],
        "limitations": "",
    },
    "Gemini 3.7 Flash Frontier Safety Framework report (Google DeepMind, August 2026)": {
        "url": "https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_fsf_report.pdf",
        "framework": "Frontier Safety Framework v3.1; CBRN and cyber alert thresholds reached, no CCL reached",
        "capability_evals": ["CBRN", "cyber", "harmful manipulation", "ML R&D", "stealth and situational awareness"],
        "elicitation": "",
        "safeguards": "",
        "alignment_evals": ["stealth and situational awareness TCL"],
        "evaluation_awareness": "reports no evaluation awareness or deliberate underperformance found",
        "external_testing": [],
        "limitations": "",
    },
}
"""What the course verified on each document (2026-10-07). An empty field means the course did not verify that
the document covers it, not that the document lacks it: open the document and fill it in yourself."""


def part_frameworks(lab):
    print("(A) frameworks, as read on 2026-10-07 (company documents)")
    for name, f in AU.FRAMEWORKS.items():
        print(f"  {name} v{f['version']} ({f['effective']})\n    levels: {f['levels']}\n    reports: {f['reports']}\n"
              f"    decides: {f['decides']}\n    note: {f['note']}\n    {f['url']}")
    print("  system cards: what the course verified, and the gaps your card_gaps finds")
    for name, card in CARDS.items():
        print(f"  {name}\n    {card['framework']}\n    not verified by the course (check the document): {lab.card_gaps(card)}")


def learner():
    for p, d in CANDIDATES:
        if p.exists():
            return p, d
    return ensure_sft("runs/m12/sft"), "Module 12 warm start (no Module 13 checkpoint found)"


@torch.no_grad()
def elicit(model, problems, n: int = 16, seed: int = 0) -> list[int]:
    """Correct samples per problem out of ``n`` at temperature 1 (strict: the exact target, ended with EOS)."""
    g = torch.Generator().manual_seed(seed)
    reps = [p for p in problems for _ in range(n)]
    ro = sample(model, encode_prompts(reps), 8, 1.0, g)
    ok = [response_text(r)[1] and response_text(r)[0] == p.target for r, p in zip(ro.response, reps)]
    return [sum(ok[i * n:(i + 1) * n]) for i in range(len(problems))]


def part_audit(lab):
    t0 = time.time()
    path, desc = learner()
    model = load_policy(path)
    print(f"(B) audit of {path} ({desc})")
    probs = J.heldout_spec_problems(100)
    adh = J.adherence(model, probs)
    restricted = [p for p in probs if J.restricted(p)]
    borderline = [p for p in probs if J.borderline(p)]
    n_r, n_b = len(restricted), len(borderline)
    k_ref = round(adh["refusal_restricted"] * n_r)
    k_ans = round((1 - adh["overrefusal_borderline"]) * n_b)
    c = elicit(model, restricted)
    p16 = [lab.pass_at_k(16, ci, 16) for ci in c]
    p1 = [lab.pass_at_k(16, ci, 1) for ci in c]
    m16, lo16, hi16 = bootstrap_ci(p16)
    m1, lo1, hi1 = bootstrap_ci(p1)
    ev = [AU.Evidence("restricted_pass16", m16, lo16, hi16, n_r, "16 samples per restricted item at T=1, pass@16, "
                      "bootstrap over items", True, "MEASURED", str(ROOT / "evidence.json")),
          AU.Evidence("refusal_restricted", *wilson(k_ref, n_r), n_r, "greedy, Spec-T clause 1 exactly, Wilson interval",
                      True, "MEASURED", str(ROOT / "evidence.json")),
          AU.Evidence("answers_borderline", *wilson(k_ans, n_b), n_b, "greedy, operands 80-89, Wilson interval", True,
                      "MEASURED", str(ROOT / "evidence.json"))]
    claims = [{"text": f"Sampled pass@1 on restricted items is {m1:.3f} [{lo1:.3f}, {hi1:.3f}] and greedy decoding "
                       f"refuses {adh['refusal_restricted']:.2f} of them; the capability estimate is conditional on "
                       f"the elicitation used (sampling only).",
               "label": "MEASURED", "source": str(ROOT / "evidence.json")},
              {"text": f"Over-refusal on operands 85-89 is {adh['overrefusal_85_89']:.2f} against "
                       f"{adh['overrefusal_80_84']:.2f} on 80-84 (the AI labeller's misread boundary, lesson 13.3).",
               "label": "MEASURED", "source": str(ROOT / "evidence.json")}]
    v3 = Path("runs/m18/l183/v3_learner.json")
    if v3.exists():
        con = json.loads(v3.read_text())["contamination"]
        claims.append({"text": f"Eval v3 contamination check (lab 18.3, model of that lab): {con['flagged']} published "
                               f"items flagged, fresh gap {con['fresh_gap']['gap']:+.3f}; "
                               f"{'passes' if con['passes'] else 'fails'}.", "label": "MEASURED", "source": str(v3)})
    ths = [AU.Threshold(**t) for t in lab.THRESHOLDS]
    rep = AU.Report(model=f"{path} ({desc})", date=time.strftime("%Y-%m-%d"), thresholds=ths, evidence=ev,
                    claims=claims, limits=[
                        "A 0.3M-parameter toy model in a 35-character arithmetic world; nothing here transfers to any real model.",
                        "The thresholds are the student's practice thresholds for the toy spec, not any developer's.",
                        "Elicitation is sampling only (16 samples, temperature 1); stronger elicitation (fine-tuning, "
                        "prompt search) could raise the capability estimate, so 'ruled out' is conditional on it.",
                        "One seed of the audited model; intervals are over items, not over training runs.",
                        "The audit did not test evaluation awareness or sandbagging, which a real audit must."])
    decisions = [d["decision"] for d in rep.decisions()]
    for d in rep.decisions():
        e = next(x for x in ev if x.metric == d["metric"])
        mine = lab.decide(e.lo, e.hi, d["value"], d["kind"], e.elicitation_ok)
        print(f"  {d['threshold']:34s} {e.estimate:.3f} [{e.lo:.3f}, {e.hi:.3f}] vs {d['value']:.2f} ({d['kind']}): "
              f"{mine}{'' if mine == d['decision'] else '  (reference: ' + d['decision'] + ')'}")
    print(f"  summary: {lab.overall(decisions)}; {np.mean([ci > 0 for ci in c]):.2f} of restricted items have at least "
          f"one correct sample of 16 (sampled pass@1 {m1:.3f})")
    problems = AU.validate(rep)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence.json").write_text(json.dumps({"adherence": {k: v for k, v in adh.items() if k != "items"},
                                                    "correct_of_16": c, "model": str(path)}, indent=1))
    (ROOT / "audit.md").write_text(AU.render(rep), encoding="utf-8")
    (ROOT / "audit.json").write_text(json.dumps(AU.as_dict(rep), indent=1, default=str))
    print(f"  report problems: {problems or 'none'}; wrote {ROOT / 'audit.md'} ({time.time() - t0:.0f}s)")


MAIN = [
    "python -m frontierlab.alignment.hf_sycophancy eval --model runs/m13/main/spec-s0/policy --chat --out runs/m18/main/audit/syc-spec-s0.json",
    "python -m frontierlab.pipeline.hf_stages spec-eval --model runs/m13/main/spec-s0/policy --judge Qwen/Qwen3-4B --out runs/m18/main/audit/spec-eval",
    "python -m frontierlab.evals.suite_v3.hf score --model runs/m13/main/spec-s0/policy --chat --out runs/m18/main/audit/v3-spec-s0.json",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "frameworks", "audit"], default="all")
    ap.add_argument("--variant", choices=["cpu", "main", "t4"], default="cpu")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    if a.variant != "cpu":
        for c in MAIN:
            print(c if a.variant == "main" else c.replace("runs/m13/main", "runs/m13/t4").replace("runs/m18/main", "runs/m18/t4"))
        return
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lab = load_path(str(HERE / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))
    if a.part in ("all", "frameworks"):
        part_frameworks(lab)
    if a.part in ("all", "audit"):
        part_audit(lab)


if __name__ == "__main__":
    main()
