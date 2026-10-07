"""Lab 13.2 — Distillation. Fill in the TODOs; run `pytest labs/module-13/lesson-02` to check.

All per-token tensors are (B, R): B sampled responses, R response positions; ``mask`` (B, R) is 1 on
response tokens up to and including EOS. Logits are (B, R, V) with V the vocabulary.
"""

from __future__ import annotations

import torch


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """(B, R) KL(pi_s || pi_T) = sum_v pi_s(v) [log pi_s(v) - log pi_T(v)] at every position.

    Use log_softmax on both (never log(softmax)); the teacher is a constant: detach it."""
    raise NotImplementedError("TODO 1: exact per-position reverse KL")


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """(B, R) KL(pi_T || pi_s): the direction supervised KD and SFT on teacher samples minimise."""
    raise NotImplementedError("TODO 2: exact per-position forward KL")


def sampled_rkl_loss(logp: torch.Tensor, teacher_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """On-policy distillation with only the sampled tokens' log-probabilities (Thinking Machines' recipe):

    d_t = logp_t - teacher_logp_t (detached), advantage A_t = -d_t, discount 0, on-policy ratio
    rho_t = exp(logp_t - logp_t.detach()) (= 1, gradient grad log pi_s). Return the token mean over the
    mask of -(rho_t * A_t). Its gradient is an unbiased estimate of the gradient of the mean per-token
    reverse KL (test_lab.py checks it by enumeration)."""
    raise NotImplementedError("TODO 3: the sampled reverse-KL surrogate")


def arm_flops(n_student: int, n_teacher: int, *, student_sampled: float = 0.0, student_trained: float = 0.0,
              teacher_generated: float = 0.0, teacher_scored: float = 0.0) -> dict:
    """Compute of one distillation arm, in FLOPs, from token counts (prompt tokens included in each count):

    student_sample = 2 N_s x student_sampled; student_train = 6 N_s x student_trained;
    teacher_sample = 2 N_T x teacher_generated; teacher_score = 2 N_T x teacher_scored;
    plus "total" (the sum) and "teacher_share" (teacher FLOPs / total, 0 if total is 0)."""
    raise NotImplementedError("TODO 4: count the teacher")
