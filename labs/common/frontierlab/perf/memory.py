"""Activation memory: measure it on any device, and take GPU memory snapshots (lesson 02.2).

Two ways to see memory:

* :class:`SavedTensors` works on CPU and GPU. It installs ``torch.autograd.graph.saved_tensors_hooks``
  so every tensor autograd saves for the backward pass is reported to it during the forward pass.
  Saved tensors are exactly the *activation memory* of training (plus any parameters an op saves,
  which are excluded when you pass the model). Storages are de-duplicated, so a tensor saved by two
  ops, or several views of one storage, are counted once.
* :func:`memory_snapshot` (CUDA only) wraps ``torch.cuda.memory._record_memory_history`` and
  ``torch.cuda.memory._dump_snapshot``. The pickle it writes opens in the interactive viewer at
  https://pytorch.org/memory_viz and shows every allocation with the Python stack that made it.
  These functions start with an underscore: they are documented ("Understanding CUDA Memory Usage")
  but their arguments can change between releases; this wrapper was checked against torch 2.14.1.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import torch


class SavedTensors:
    """Context manager that records the bytes autograd saves for backward.

    >>> with SavedTensors(exclude=model.parameters()) as st:
    ...     loss = model(x, labels=x).loss
    >>> st.total_bytes, st.largest(3)
    """

    def __init__(self, exclude=()):
        self._exclude = {p.untyped_storage().data_ptr() for p in exclude}
        self._seen: dict[int, tuple[int, tuple, torch.dtype]] = {}
        self._hooks = None

    def _pack(self, t: torch.Tensor):
        try:
            st = t.untyped_storage()
            ptr = st.data_ptr()
            if ptr not in self._exclude and ptr not in self._seen:
                self._seen[ptr] = (st.nbytes(), tuple(t.shape), t.dtype)
        except (RuntimeError, NotImplementedError):     # tensors without storage (e.g. some subclasses)
            pass
        return t

    @staticmethod
    def _unpack(t):
        return t

    def __enter__(self):
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self._hooks.__exit__(*exc)
        return False

    @property
    def total_bytes(self) -> int:
        return sum(n for n, _, _ in self._seen.values())

    @property
    def count(self) -> int:
        return len(self._seen)

    def largest(self, k: int = 5) -> list[tuple[int, tuple, str]]:
        """The ``k`` largest saved storages as (bytes, shape, dtype)."""
        return [(n, s, str(d)) for n, s, d in sorted(self._seen.values(), key=lambda r: -r[0])[:k]]

    def by_shape(self) -> Counter:
        """Total bytes per (shape, dtype), to see which kinds of tensor dominate."""
        c = Counter()
        for n, s, d in self._seen.values():
            c[(s, str(d))] += n
        return c


def saved_activation_bytes(model, loss_fn) -> SavedTensors:
    """Run ``loss_fn()`` (a forward pass returning a loss) and return what autograd saved.

    Parameters of ``model`` are excluded so the number is activation memory only. The loss is
    discarded without a backward pass.
    """
    with SavedTensors(exclude=model.parameters()) as st:
        loss = loss_fn()
    del loss
    return st


@contextmanager
def memory_snapshot(path: str, max_entries: int = 100_000):
    """Record every CUDA allocation inside the block and write a snapshot pickle to ``path``.

    Open the file at https://pytorch.org/memory_viz. Recording costs a little time per allocation;
    do it for a couple of steps, not a whole run.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("memory snapshots need CUDA; on CPU use SavedTensors")
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


def cuda_peak_bytes(fn, device: str = "cuda") -> int:
    """Peak allocated CUDA memory while ``fn()`` runs (allocated tensors, not the allocator's cache)."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def logits_bytes(B: int, T: int, V: int, bytes_per: int = 4) -> int:
    """Size of the (B, T, V) logits tensor; ``frontierlab.model.LM`` casts logits to fp32 (4 bytes)."""
    return B * T * V * bytes_per
