"""Distillation from scratch (lesson 13.2): SFT on teacher outputs vs on-policy distillation.

**Off-policy (sequence-level) distillation** (Kim and Rush 2016; Qwen3's "off-policy distillation"):
sample responses from the teacher, then fine-tune the student on them with the ordinary SFT loss
(:func:`teacher_examples` + :func:`frontierlab.pipeline.seqs.train_sft`). The student learns on the
teacher's distribution of prefixes, not its own, so its own mistakes at inference lead it into prefixes it
never trained on (the train/inference mismatch GKD names; Agarwal et al. 2023, section 1).

**On-policy distillation** (GKD, Agarwal et al. 2023; Thinking Machines 2025; Qwen3 section 4.5): sample
from the *student*, ask the teacher for its log-probabilities of the same tokens, and lower the per-token
reverse KL

    KL_t = sum_v pi_s(v | y_<t) [log pi_s(v | y_<t) - log pi_T(v | y_<t)]      at every response position t.

Two estimators of its gradient (:func:`opd_loss`):

* ``"sampled"`` (Thinking Machines): only the sampled token's log-probabilities are needed. With
  d_t = log pi_s(y_t) - log pi_T(y_t) and the advantage A_t = -d_t (detached), the policy-gradient
  surrogate ``-A_t * rho_t`` (rho_t = pi_s / pi_sampler, = 1 on-policy, gradient = grad log pi_s) has
  expectation over y_t ~ pi_s equal to grad KL_t: E[d_t grad log pi_s(y_t)] = grad KL_t, because
  E[grad log pi_s] = 0. The discount factor is zero: each token is credited only with its own KL term,
  not with the KL of the tokens it leads to.
* ``"exact"`` (GKD with the reverse KL): the full KL_t over the vocabulary, differentiated directly.
  Needs the teacher's full logits at every position: free on a 35-token vocabulary, a (B, R, 151,936)
  tensor for Qwen3.

Both need the teacher and student to assign probabilities to the *same tokens*, so they share a tokenizer
(INFERENCE from the definition; cross-tokenizer distillation needs an alignment step).

Compute (charged to a :class:`~frontierlab.pipeline.compute.Ledger`): student sampling 2 N_s per token,
teacher scoring 2 N_T per token (one forward over the student's tokens, no generation), student update
6 N_s per token. Off-policy distillation instead pays 2 N_T per *generated* teacher token (sampling,
which on a GPU is far slower per FLOP than scoring).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from frontierlab.model import LM
from frontierlab.pipeline.compute import Ledger, n_params
from frontierlab.pipeline.seqs import train_sft
from frontierlab.posttrain.policy import masked_mean, sample, token_logprobs
from frontierlab.posttrain.sft import load_policy, policy_config, save_policy
from frontierlab.posttrain.tasks import TAGS, encode_prompts, score, split_problems


# --------------------------------------------------------------------------- the teacher

def teacher_config():
    """A larger model of the same layout and tokenizer as the toy student (about 1.2M parameters, 4x the student)."""
    return policy_config(hidden_size=160, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=40, intermediate_size=448)


def train_teacher(out: str | Path, steps: int = 2500, batch: int = 64, lr: float = 3e-3, seed: int = 0) -> dict:
    """SFT the teacher on gold targets (all operations and tags, training triples only) to high accuracy."""
    from frontierlab.pipeline import toy
    from frontierlab.posttrain.sft import greedy_accuracy
    from frontierlab.posttrain.tasks import problems_from
    out = Path(out)
    torch.manual_seed(seed)
    model = LM(teacher_config())
    ex = toy.gold_examples(toy.train_problems(40000, seed=seed + 500))
    led = Ledger()
    res = train_sft(model, ex, steps=steps, batch=batch, lr=lr, seed=seed, ledger=led, log_every=250)
    _, held = split_problems(2)
    acc = {f"{op}{t}": greedy_accuracy(model, problems_from(held, t, 2, op)[:300]) for op in "+-" for t in TAGS}
    meta = {"steps": steps, "params": n_params(model), "heldout_greedy_acc": acc, "seconds": res["seconds"],
            "train_flops": led.total()}
    save_policy(model, out / "policy.pt", meta)
    (out / "teacher.json").write_text(json.dumps(meta, indent=1))
    return meta


def ensure_teacher(out: str | Path = "runs/m13/teacher", **kw) -> Path:
    path = Path(out) / "policy.pt"
    if not path.exists():
        print(f"training the distillation teacher -> {path} (a few minutes on a laptop)")
        train_teacher(out, **kw)
    return path


# --------------------------------------------------------------------------- divergences

def exact_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """(B, R) KL(pi_s || pi_T) at every position from logits (B, R, V); differentiable in the student."""
    ls = torch.log_softmax(student_logits, -1)
    lt = torch.log_softmax(teacher_logits.detach(), -1)
    return (ls.exp() * (ls - lt)).sum(-1)


def exact_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """(B, R) KL(pi_T || pi_s): the supervised-KD direction (mass-covering)."""
    ls = torch.log_softmax(student_logits, -1)
    lt = torch.log_softmax(teacher_logits.detach(), -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def opd_loss(logp: torch.Tensor, sampler_logp: torch.Tensor, teacher_logp: torch.Tensor, mask: torch.Tensor,
             kind: str = "sampled", student_logits: torch.Tensor | None = None,
             teacher_logits: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """On-policy distillation loss (token mean over ``mask``) and diagnostics.

    ``logp`` (B, R): current student log-probabilities of the sampled tokens (with gradient);
    ``sampler_logp`` (B, R): the same under the policy that sampled them (detached; equal to logp on-policy);
    ``teacher_logp`` (B, R): the teacher's log-probabilities of the sampled tokens.
    """
    m = mask.float()
    d = (sampler_logp - teacher_logp).detach()                      # per-token sampled reverse KL
    if kind == "sampled":
        adv = -d
        rho = torch.exp(logp - sampler_logp.detach())
        loss = -masked_mean(rho * adv, m)
    elif kind == "exact":
        if student_logits is None or teacher_logits is None:
            raise ValueError("the exact reverse KL needs both models' logits")
        loss = masked_mean(exact_reverse_kl(student_logits, teacher_logits), m)
    else:
        raise ValueError(kind)
    with torch.no_grad():
        diag = {"rkl_sampled": float(masked_mean(d, m))}
        if student_logits is not None and teacher_logits is not None:
            diag["rkl_exact"] = float(masked_mean(exact_reverse_kl(student_logits.detach(), teacher_logits), m))
    return loss, diag


# --------------------------------------------------------------------------- the two ways to distil

@torch.no_grad()
def teacher_examples(teacher, problems: list, n: int = 1, temperature: float = 1.0, seed: int = 0,
                     ledger: Ledger | None = None, max_new: int = 8) -> list:
    """Off-policy distillation data: ``n`` teacher samples per problem as SFT examples (teacher_sample cost)."""
    from frontierlab.pipeline import toy
    texts = toy.sample_texts(teacher, problems, n, temperature, seed, max_new, ledger, "teacher_sample")
    return [toy.example(p, t, f) for p, ts in zip(problems, texts) for t, f in ts]


def _logits(model, tokens, prompt_len, temperature=1.0):
    return model(tokens).logits[:, prompt_len - 1:-1].float() / temperature


def train_opd(student, teacher, problem_fn, steps: int, prompts: int = 16, group: int = 8, temperature: float = 1.0,
              lr: float = 3e-4, kind: str = "sampled", seed: int = 0, max_new: int = 8, grad_clip: float = 1.0,
              ledger: Ledger | None = None, log=None, log_every: int = 10, loss_fn=None) -> dict:
    """On-policy distillation: every step samples ``prompts * group`` responses from the current student,
    scores them with the teacher and takes one gradient step (so the ratio is exactly 1).

    ``problem_fn(step) -> list[Problem]`` gives the step's prompts (training triples only). ``loss_fn(logp,
    teacher_logp, mask, student_logits, teacher_logits) -> loss`` replaces the course loss (labs pass the
    learner's); the logged diagnostics always come from :func:`opd_loss`."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    teacher.eval()
    Ns, Nt = n_params(student), n_params(teacher)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    t0, hist = time.perf_counter(), []
    for step in range(1, steps + 1):
        probs = [p for p in problem_fn(step) for _ in range(group)]
        ro = sample(student, encode_prompts(probs), max_new, temperature, gen)
        toks = float(ro.tokens.numel())
        with torch.no_grad():
            t_logits = _logits(teacher, ro.tokens, ro.prompt_len, temperature)
            t_logp = torch.log_softmax(t_logits, -1).gather(-1, ro.response[..., None]).squeeze(-1)
        student.train()
        s_logits = _logits(student, ro.tokens, ro.prompt_len, temperature)
        logp = torch.log_softmax(s_logits, -1).gather(-1, ro.response[..., None]).squeeze(-1)
        loss, d = opd_loss(logp, logp.detach(), t_logp, ro.mask, kind, s_logits, t_logits)
        if loss_fn is not None:
            loss = loss_fn(logp, t_logp, ro.mask, s_logits, t_logits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()
        student.eval()
        if ledger is not None:
            ledger.forward("student_sample", Ns, toks)
            ledger.forward("teacher_score", Nt, toks)
            ledger.train("student_train", Ns, toks)
        if step % log_every == 0 or step == steps:
            row = {"stage": f"opd-{kind}", "step": step, "loss": float(loss.detach()), **d,
                   "pass": float(score(ro.response, probs).mean()), "len": float(ro.lengths.mean()),
                   "seconds": round(time.perf_counter() - t0, 2)}
            hist.append(row)
            if log is not None:
                log.log(**row)
    return {"steps": steps, "history": hist, "final": hist[-1] if hist else {},
            "seconds": round(time.perf_counter() - t0, 2)}


def rl_ledger(run_dir: str | Path, n: int, prompt_len: int = 7, epochs: int = 1) -> Ledger:
    """Compute of a Module 12 RL run (``frontierlab.posttrain.rl``) from its metrics: sampling (2 N per prompt
    and response token), three frozen forward passes per step (behaviour, reference, current) and the
    update (6 N per token)."""
    from frontierlab.metrics.jsonl import read_jsonl
    rows = [r for r in read_jsonl(Path(run_dir) / "metrics.jsonl") if r["split"] == "train"]
    cfg = json.loads(json.dumps(torch.load(Path(run_dir) / "checkpoint.pt", weights_only=False)["config"]))
    B = cfg["prompts"] * cfg["group"]
    led = Ledger()
    for r in rows:
        toks = B * (prompt_len + r["len"])
        led.forward("student_sample", n, toks)
        led.forward("reference", n, 3 * toks)
        led.train("student_train", n, epochs * toks)
    return led
