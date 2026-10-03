"""Chunked (fused-linear) cross-entropy: the loss without the (B, T, V) logits tensor (lesson 02.2).

The ordinary loss computes ``logits = h @ Wᵀ`` for all N = B·T positions at once, casts to fp32,
and autograd keeps an N × V fp32 tensor alive until the backward pass (``log_softmax`` saves its
output). At Baseline-0's V = 32768, B·T = 32·1024 that is 4 GiB — more than all the transformer
layers' activations together.

The gradient of mean cross-entropy with respect to the logits has a closed form,

    dL/dz_i = (softmax(z_i) - onehot(y_i)) / n_valid

so the loss and *all* gradients can be produced chunk by chunk in the forward pass: for each chunk
of ``chunk_size`` rows, compute its logits, its loss, ``g = dL/dz`` for those rows, then
``dL/dh_chunk = g @ W`` and ``dL/dW += gᵀ @ h_chunk``, and drop the chunk's logits. Peak extra
memory is one chunk of logits (``chunk_size × V``) instead of ``N × V``. The backward pass only
scales the stored gradients by the incoming gradient (1.0 for a loss you call ``.backward()`` on).

This is the idea behind fused linear cross-entropy kernels (Liger Kernel) and Cut Cross-Entropy
(Wijmans et al., 2024), here in plain PyTorch: it saves memory, it does not fuse kernels, and it
adds no FLOPs (the GEMMs are the same as in the ordinary forward + backward, just done earlier).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _upcast(z: torch.Tensor) -> torch.Tensor:
    """Logits in at least fp32 (bf16/fp16 are upcast, fp64 stays fp64 for the float64 tests)."""
    return z.float() if z.dtype in (torch.float16, torch.bfloat16) else z


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, targets, chunk_size: int, ignore_index: int):
        # h (N, C), w (V, C), targets (N,) long
        N = h.shape[0]
        valid = targets != ignore_index
        n_valid = valid.sum().clamp(min=1)
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_h = torch.zeros_like(h) if need_h else None
        grad_w = (torch.zeros(w.shape, dtype=torch.promote_types(w.dtype, torch.float32), device=w.device)
                  if need_w else None)
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        for s in range(0, N, chunk_size):
            hc, tc, vc = h[s:s + chunk_size], targets[s:s + chunk_size], valid[s:s + chunk_size]
            z = _upcast(hc @ w.t())                                   # (n, V) >= fp32, this chunk only
            lse = torch.logsumexp(z, dim=-1)
            safe_t = torch.where(vc, tc, torch.zeros_like(tc))
            tok = lse - z.gather(1, safe_t[:, None]).squeeze(1)
            loss = loss + (tok * vc).sum()
            if need_h or need_w:
                g = torch.softmax(z, dim=-1)                          # reuse z's memory: g replaces z
                g.scatter_add_(1, safe_t[:, None], -torch.ones_like(tok)[:, None])
                g.mul_((vc.to(g.dtype) / n_valid)[:, None])
                if need_h:
                    grad_h[s:s + chunk_size] = (g.to(w.dtype) @ w).to(h.dtype)
                if need_w:
                    grad_w.add_(g.t().to(hc.dtype) @ hc)
        ctx.save_for_backward(grad_h if need_h else torch.empty(0), grad_w if need_w else torch.empty(0))
        ctx.flags = (need_h, need_w, w.dtype)
        return loss / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        need_h, need_w, w_dtype = ctx.flags
        gh = grad_h * grad_out.to(grad_h.dtype) if need_h else None
        gw = (grad_w * grad_out).to(w_dtype) if need_w else None
        return gh, gw, None, None, None


def chunked_cross_entropy(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                          chunk_size: int = 4096, ignore_index: int = -100) -> torch.Tensor:
    """Mean cross-entropy of ``h @ weight.T`` against ``targets`` without materialising all logits.

    Shapes: h (N, C) or (B, T, C); weight (V, C); targets (N,) or (B, T). Equals
    ``F.cross_entropy((h @ weight.T).float(), targets, ignore_index=ignore_index)`` (in float64 the
    logits stay float64) in value and in
    the gradients for ``h`` and ``weight`` (checked in ``tests/test_perf.py``). It must be the last
    op before ``.backward()``: gradients are computed in the forward pass and only scaled afterwards.
    """
    if h.dim() == 3:
        h = h.reshape(-1, h.shape[-1])
        targets = targets.reshape(-1)
    return _ChunkedLinearCE.apply(h, weight, targets, chunk_size, ignore_index)


def reference_cross_entropy(h, weight, targets, ignore_index: int = -100):
    """The ordinary loss, materialising all logits in fp32 (what ``frontierlab.model.LM`` does)."""
    if h.dim() == 3:
        h, targets = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
    return F.cross_entropy(_upcast(h @ weight.t()), targets, ignore_index=ignore_index)


def hidden_states(model, idx: torch.Tensor) -> torch.Tensor:
    """Final-norm hidden states (B, T, C) of a ``frontierlab.model.LM``, without the output head.

    Mirrors ``LM.forward`` exactly (positions 0..T-1, no cache) so the loss below is the same loss.
    """
    B, T = idx.shape
    positions = torch.arange(T, device=idx.device)
    x = model.model.embed_tokens(idx)
    for layer in model.model.layers:
        x = layer(x, positions, None)
    return model.model.norm(x)


def lm_loss_chunked(model, idx: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """``model(idx, labels=idx).loss`` computed with :func:`chunked_cross_entropy`.

    Positions 0..T-2 predict tokens 1..T-1, averaged over B·(T-1) tokens, as in ``LM.forward``.
    """
    h = hidden_states(model, idx)[:, :-1]
    return chunked_cross_entropy(h.reshape(-1, h.shape[-1]), model.lm_head.weight,
                                 idx[:, 1:].reshape(-1), chunk_size)
