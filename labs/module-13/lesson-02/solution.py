"""Lab 13.2 — Distillation. Reference solution."""

from __future__ import annotations

import torch


def reverse_kl(student_logits, teacher_logits):
    ls = torch.log_softmax(student_logits, -1)
    lt = torch.log_softmax(teacher_logits.detach(), -1)
    return (ls.exp() * (ls - lt)).sum(-1)


def forward_kl(student_logits, teacher_logits):
    ls = torch.log_softmax(student_logits, -1)
    lt = torch.log_softmax(teacher_logits.detach(), -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def sampled_rkl_loss(logp, teacher_logp, mask):
    adv = -(logp - teacher_logp).detach()
    rho = torch.exp(logp - logp.detach())
    m = mask.float()
    return -(rho * adv * m).sum() / m.sum().clamp_min(1.0)


def arm_flops(n_student, n_teacher, *, student_sampled=0.0, student_trained=0.0, teacher_generated=0.0,
              teacher_scored=0.0):
    out = {"student_sample": 2.0 * n_student * student_sampled, "student_train": 6.0 * n_student * student_trained,
           "teacher_sample": 2.0 * n_teacher * teacher_generated, "teacher_score": 2.0 * n_teacher * teacher_scored}
    total = sum(out.values())
    out["total"] = total
    out["teacher_share"] = (out["teacher_sample"] + out["teacher_score"]) / total if total else 0.0
    return out
