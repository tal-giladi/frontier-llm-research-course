"""The toy model of superposition (Elhage et al. 2022, "Demonstrating Superposition"), from scratch (17.1).

``n`` features, each zero with probability ``S`` and otherwise uniform on [0, 1], are squeezed into ``m < n``
dimensions and read back out:

    h = W x            W (m, n)
    x' = ReLU(Wᵀ h + b)
    L = E_x Σ_i I_i (x_i − x'_i)²          I_i = importance of feature i

When features are dense (S near 0) the model keeps the ``m`` most important features, one per dimension,
and drops the rest. When they are sparse it represents more features than it has dimensions, accepting
interference between features that are rarely active together. The paper calls the switch a phase change.

Measures (paper section "The Geometry of Superposition"):

* ``norms`` — ‖W_i‖; a feature is "represented" when its norm is near 1 (the course uses > 0.5).
* ``dimensions per feature`` — D* = m / ‖W‖_F².
* ``feature dimensionality`` — D_i = ‖W_i‖⁴ / Σ_j (Ŵ_i · W_j)²: 1 for a feature with a dimension to
  itself, 1/2 for an antipodal pair, 2/5 for a pentagon.
* ``interference`` — Σ_{j≠i} (Ŵ_i · W_j)², what other features write onto feature i's direction.
"""

from __future__ import annotations

import torch


def sample(batch: int, n: int, sparsity: float, gen: torch.Generator) -> torch.Tensor:
    """(batch, n): each entry 0 with probability ``sparsity``, else uniform on [0, 1]."""
    x = torch.rand(batch, n, generator=gen)
    keep = torch.rand(batch, n, generator=gen) >= sparsity
    return x * keep


class ToyModel(torch.nn.Module):
    def __init__(self, n: int, m: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(m, n, generator=g) * (2 / (n + m)) ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):                       # x (B, n) -> (B, n)
        return torch.relu((x @ self.W.T) @ self.W + self.b)


def train(n: int = 20, m: int = 5, sparsity: float = 0.9, importance_decay: float = 0.9, steps: int = 3000,
          batch: int = 1024, lr: float = 1e-2, seed: int = 0) -> ToyModel:
    """Train one toy model. Importance I_i = decay^i (feature 0 most important)."""
    torch.manual_seed(seed)
    model = ToyModel(n, m, seed)
    imp = importance_decay ** torch.arange(n, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        x = sample(batch, n, sparsity, g)
        loss = (imp * (x - model(x)) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def stats(model: ToyModel, threshold: float = 0.5) -> dict:
    W = model.W.detach().double()                   # (m, n)
    m, n = W.shape
    norms = W.norm(dim=0)                           # (n,)
    unit = W / norms.clamp_min(1e-12)
    overlap = unit.T @ W                            # (n, n): Ŵ_i · W_j
    dim_i = norms ** 4 / (overlap ** 2).sum(dim=1).clamp_min(1e-12)
    interference = (overlap ** 2).sum(dim=1) - (overlap.diagonal() ** 2)
    represented = int((norms > threshold).sum())
    return {"norms": norms.tolist(), "represented": represented, "dims_per_feature": float(m / (W ** 2).sum()),
            "feature_dimensionality": dim_i.tolist(), "interference": interference.tolist(),
            "superposition": represented > m}
