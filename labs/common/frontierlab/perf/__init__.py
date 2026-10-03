"""Performance engineering tools (Module 2): where the time and the memory of a training step go.

* :mod:`~frontierlab.perf.roofline` — hardware peaks, arithmetic intensity, roofline bounds, a
  per-op step-time model and the simple FLOPs ÷ (peak × MFU) prediction (lesson 02.1).
* :mod:`~frontierlab.perf.profiling` — ``torch.profiler`` wrappers that summarise top operators,
  CPU vs GPU time and GPU busy fraction (lesson 02.2).
* :mod:`~frontierlab.perf.memory` — saved-activation accounting on any device and CUDA memory
  snapshots (lesson 02.2).
* :mod:`~frontierlab.perf.chunked_ce` — cross-entropy without the (B, T, V) logits tensor (02.2, 02.4).
* :mod:`~frontierlab.perf.dist` — multi-process launch, data-parallel step timing, communication
  arithmetic (lesson 02.3).
* :mod:`~frontierlab.perf.timing` — the benchmark harness: warm-up, sync, repeats, intervals,
  interleaved comparisons (lesson 02.4).
"""

from frontierlab.perf.chunked_ce import chunked_cross_entropy, lm_loss_chunked, reference_cross_entropy
from frontierlab.perf.memory import SavedTensors, memory_snapshot, saved_activation_bytes
from frontierlab.perf.profiling import busy_fraction, profile_steps, summarize
from frontierlab.perf.roofline import HARDWARE, Hardware, Op, predicted_step_time, roofline_time, step_time_model
from frontierlab.perf.timing import Timing, benchmark, interleaved, speedup, sync

__all__ = ["HARDWARE", "Hardware", "Op", "SavedTensors", "Timing", "benchmark", "busy_fraction",
           "chunked_cross_entropy", "interleaved", "lm_loss_chunked", "memory_snapshot", "predicted_step_time",
           "profile_steps", "reference_cross_entropy", "roofline_time", "saved_activation_bytes", "speedup",
           "step_time_model", "summarize", "sync"]
