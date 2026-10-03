"""Why MLA needs a separate RoPE key: absorption works without RoPE on the compressed key, fails with it.

    python labs/module-03/lesson-01/why_decoupled.py

One head, latent width d_c = 16, key width d = 8, 6 positions, float64. The score between query t and
key s is q_t . k_s. Absorption precomputes ONE matrix W_UK^T and applies it to each query once, so that
q_t . (W_UK c_s) = (W_UK^T q_t) . c_s for every cached s. If RoPE rotates the up-projected key,
k_s = R_s W_UK c_s, the matrix between q_t and c_s is R_s W_UK, a different matrix for every cached
position: there is nothing position-free to fold into the query.
"""

import torch

from frontierlab.attention.base import RotaryEmbedding, apply_rope

torch.manual_seed(0)
d, d_c, T = 8, 16, 6
W_UK = torch.randn(d, d_c, dtype=torch.float64)
c = torch.randn(T, d_c, dtype=torch.float64)                   # cached latents, one per position
q = torch.randn(T, d, dtype=torch.float64)                     # queries (already projected)
cos, sin = RotaryEmbedding(d, 10000.0)(torch.arange(T))
rope = lambda x: apply_rope(x[None, None], cos.double(), sin.double())[0, 0]   # noqa: E731  rotate rows by position

causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
# 1. no RoPE on the compressed key: absorbed == naive
naive = q @ (c @ W_UK.T).T
absorbed = (q @ W_UK) @ c.T
print(f"no RoPE on W_UK c:   max |naive - absorbed| = {(naive - absorbed)[causal].abs().max():.2e}")

# 2. RoPE on queries and on the up-projected keys (what plain RoPE attention would do)
k = rope(c @ W_UK.T)                                           # R_s W_UK c_s
qr = rope(q)                                                   # R_t q_t
naive = qr @ k.T
absorbed = (qr @ W_UK) @ c.T                                   # the only position-free fold available
print(f"RoPE on W_UK c:      max |naive - absorbed| = {(naive - absorbed)[causal].abs().max():.2e}"
      f"   (scores themselves are ~{naive[causal].abs().mean():.1f})")

# 3. decoupled RoPE: content part absorbed, a separate small RoPE key shared by all heads
d_r = 4
W_KR = torch.randn(d_r, d_c, dtype=torch.float64)              # stands in for W_KR h_s (any per-token vector)
q_r = torch.randn(T, d_r, dtype=torch.float64)
cos_r, sin_r = RotaryEmbedding(d_r, 10000.0)(torch.arange(T))
rope_r = lambda x: apply_rope(x[None, None], cos_r.double(), sin_r.double())[0, 0]   # noqa: E731
k_rope = rope_r(c @ W_KR.T)
naive = q @ (c @ W_UK.T).T + rope_r(q_r) @ k_rope.T
absorbed = (q @ W_UK) @ c.T + rope_r(q_r) @ k_rope.T
print(f"decoupled RoPE key:  max |naive - absorbed| = {(naive - absorbed)[causal].abs().max():.2e}")
