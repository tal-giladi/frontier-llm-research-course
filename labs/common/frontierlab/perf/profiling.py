"""Thin wrappers over ``torch.profiler`` that answer "where did the step's time go?" (lesson 02.2).

:func:`profile_steps` warms up outside the profiler, then records ``steps`` calls of a step
function, each inside a ``record_function("step")`` range. :func:`summarize` turns the result into
a small dict: the top operators by self time, total CPU time and (on GPU) total kernel time, the
number of kernel launches, and how busy the GPU was inside the profiled window.

Reading the numbers:

* ``self`` time of an operator excludes its children, so the self times add up to the total.
* On GPU, CPU time is the time Python and the dispatcher spent *queueing* kernels; device time is
  the time kernels actually ran. A step whose kernels are short compared with their launch cost
  (~a few µs each) is **launch-bound**: the GPU waits for the CPU (``device_busy_fraction`` well
  below 1). Fusion and CUDA graphs attack that.
* On CPU there is no separate device: operator times are the computation itself.
"""

from __future__ import annotations

from torch.profiler import ProfilerActivity, profile, record_function

import torch


def profile_steps(step_fn, steps: int = 3, warmup: int = 2, device: str = "cpu", record_shapes: bool = False,
                  profile_memory: bool = False, trace_path: str | None = None):
    """Run ``step_fn()`` ``warmup`` times unprofiled, then ``steps`` times under the profiler.

    Returns the ``torch.profiler.profile`` object. ``trace_path`` (optional) writes a Chrome/Perfetto
    trace (open it at https://ui.perfetto.dev) with the CPU and GPU timelines.
    """
    on_gpu = torch.device(device).type == "cuda"
    for _ in range(warmup):
        step_fn()
    if on_gpu:
        torch.cuda.synchronize()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_gpu else [])
    with profile(activities=acts, record_shapes=record_shapes, profile_memory=profile_memory) as prof:
        for _ in range(steps):
            with record_function("step"):
                step_fn()
        if on_gpu:
            torch.cuda.synchronize()
    if trace_path:
        prof.export_chrome_trace(trace_path)
    return prof


def busy_fraction(intervals: list[tuple[float, float]], window: tuple[float, float] | None = None) -> float:
    """Fraction of ``window`` covered by the union of ``intervals`` (start, end); overlaps count once.

    With kernel intervals and the step window this is how busy the GPU was; ``1 - busy`` is idle time
    (waiting for the CPU, for data, or for communication).
    """
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    lo, hi = window if window is not None else (iv[0][0], max(e for _, e in iv))
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / (hi - lo) if hi > lo else 0.0


def summarize(prof, top: int = 10) -> dict:
    """Top operators and totals from a ``profile_steps`` result. Times in milliseconds.

    Keys: ``steps``, ``step_ms`` (wall time of the "step" ranges, averaged), ``cpu_self_ms`` (sum of
    self CPU time over operators), ``device_ms`` (sum of kernel time; 0 on CPU), ``n_ops`` (operator
    calls), ``n_kernels`` (GPU kernel launches; 0 on CPU), ``device_busy_fraction`` (GPU only) and
    ``top``: a list of ``(name, self_ms, calls, share)`` sorted by self device time on GPU, by self
    CPU time on CPU.
    """
    ka = prof.key_averages()
    steps = [e for e in prof.events() if e.name == "step"]
    n_steps = max(1, len(steps))
    on_gpu = any(e.self_device_time_total > 0 for e in ka)
    ops = [e for e in ka if e.key != "step"]
    key = (lambda e: e.self_device_time_total) if on_gpu else (lambda e: e.self_cpu_time_total)
    total = sum(key(e) for e in ops) or 1.0
    rows = sorted(ops, key=key, reverse=True)[:top]
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    out = {
        "steps": n_steps,
        "step_ms": sum(e.time_range.end - e.time_range.start for e in steps) / n_steps / 1e3,
        "cpu_self_ms": sum(e.self_cpu_time_total for e in ops) / 1e3 / n_steps,
        "device_ms": sum(e.self_device_time_total for e in ops) / 1e3 / n_steps,
        "n_ops": sum(e.count for e in ops if e.key.startswith("aten::")) // n_steps,
        "n_kernels": len(kernels) // n_steps,
        "top": [(e.key, key(e) / 1e3 / n_steps, e.count // n_steps, key(e) / total) for e in rows],
    }
    if on_gpu and steps:
        window = (min(e.time_range.start for e in steps), max(e.time_range.end for e in steps))
        out["device_busy_fraction"] = busy_fraction([(k.time_range.start, k.time_range.end) for k in kernels],
                                                    window)
    return out


def format_summary(s: dict) -> str:
    lines = [f"{s['steps']} steps, {s['step_ms']:.2f} ms/step wall, {s['n_ops']} aten ops/step"
             + (f", {s['n_kernels']} kernels/step, GPU busy {s['device_busy_fraction']:.0%}"
                if "device_busy_fraction" in s else "")]
    for name, ms, calls, share in s["top"]:
        lines.append(f"  {share:6.1%}  {ms:9.3f} ms  {calls:5d}x  {name}")
    return "\n".join(lines)
