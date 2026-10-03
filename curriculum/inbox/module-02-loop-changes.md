# Module 2 — proposed changes to shared files (for the main session)

Module 2 does not edit `frontierlab/train/loop.py` or `frontierlab/model/lm.py`. These are the changes it
proposes, with exact code. None of the Module 2 labs depend on them (the labs call
`frontierlab.perf` directly), but later modules will want them.

## 1. `train/loop.py`: `--loss chunked` and `--ce-chunk` (lesson 02.4 result)

Add to `build_parser()`:

```python
    ap.add_argument("--loss", choices=["plain", "chunked"], default="plain",
                    help="chunked = frontierlab.perf.chunked_ce (no (B,T,V) logits; lesson 02.2/02.4)")
    ap.add_argument("--ce-chunk", type=int, default=4096, help="rows per chunk for --loss chunked")
```

In `main()`, replace the micro-batch body

```python
            with torch.autocast(device_type=x.device.type, dtype=autocast, enabled=autocast is not None):
                out = fwd(x, labels=x)
            (out.loss / a.grad_accum).backward()
            loss_sum += out.loss.item() / a.grad_accum
```

with

```python
            with torch.autocast(device_type=x.device.type, dtype=autocast, enabled=autocast is not None):
                if a.loss == "chunked":
                    from frontierlab.perf.chunked_ce import lm_loss_chunked
                    loss = lm_loss_chunked(model, x, a.ce_chunk)
                else:
                    loss = fwd(x, labels=x).loss
            (loss / a.grad_accum).backward()
            loss_acc += loss.detach() / a.grad_accum          # stays on device: no sync per micro-batch
```

with `loss_acc = torch.zeros((), device=a.device)` instead of `loss_sum = 0.0` before the micro-batch
loop, and `loss_sum = loss_acc.item()` once after it (see change 2). Note `lm_loss_chunked` takes the
uncompiled `model`; with `--compile` either compile `frontierlab.perf.chunked_ce.hidden_states` or
leave the chunked path eager. The run card should record `args.loss` (it does automatically via `vars(a)`).

## 2. `train/loop.py`: no `.item()` per micro-batch

The current loop calls `out.loss.item()` after every micro-batch, which forces a CPU–GPU
synchronisation each time and stops the CPU from queueing the next micro-batch's kernels (visible as a
gap in the GPU timeline, lesson 02.2). Accumulate on device as in change 1 and call `.item()` only
once per optimizer step — or, better, only on logging steps:

```python
        if step % a.log_every == 0 or step == a.steps:
            loss_sum = loss_acc.item()
```

(`gnorm` is also a device tensor; `float(gnorm)` is already only on logging steps.) The logged
values do not change.

## 3. `train/loop.py`: `--profile-steps`

```python
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="after warm-up, profile this many optimizer steps and write <run>/trace.json "
                         "and <run>/profile.txt (lesson 02.2)")
    ap.add_argument("--memory-snapshot", action="store_true",
                    help="CUDA only: record allocations during the profiled steps to <run>/mem.pickle")
```

Wrap the body of the `while step < stop:` loop so that when `a.profile_steps` is set and
`step == start_step + 3` (three warm-up steps after start or resume), a `torch.profiler.profile`
context (CPU + CUDA activities) is entered, and it is exited after `a.profile_steps` further steps,
followed by `prof.export_chrome_trace(str(a.run / "trace.json"))` and writing
`frontierlab.perf.profiling.format_summary(frontierlab.perf.profiling.summarize(prof))` to
`<run>/profile.txt`. With `--memory-snapshot`, use `frontierlab.perf.memory.memory_snapshot(a.run /
"mem.pickle")` around the same steps. Profiling must not change the training result: the RNG states
are untouched, so `tests/test_train.py::test_exact_resume` still holds.

## 4. `model/lm.py`: `.float()` downcasts float64 logits

`LM.forward` does `logits = self.lm_head(...).float()`. For a float64 model (the correctness suite
copies models to float64) this *downcasts* the logits to float32, so float64 checks of the loss and
of gradients are limited to float32 precision (about 1e-7 relative). Module 2's tests work around it;
a one-line fix keeps fp64 as fp64 and still upcasts bf16/fp16:

```python
        logits = self.lm_head(self.model.norm(x))
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
```

## 5. Lesson 01.1 wording (Shapes and cost)

01.1 says "The logits tensor dominates activation memory at small width". Measured in Module 2
(lesson 02.2, `SavedTensors`, Baseline-0, B = 1, T = 1024): the fp32 logits saved by `log_softmax` are
128 MiB of 1,080 MiB saved activations (12%) in fp32; with the returned logits tensor counted, about a
quarter. Proposed replacement sentence:

> The logits tensor is the largest single activation at small width: at B = 32, T = 1024 it is
> $32 \cdot 1024 \cdot 32768 \cdot 4$ bytes ≈ 4 GiB in fp32, and the loss keeps a second copy of the same
> size. Module 2 measures this and shows how to avoid both.

## 6. `frontierlab/flops.py` `PEAK_BF16`

`"T4-FP16": 65e12` and `"L4": 121e12` match the datasheets (dense). `"H100-PCIe": 756e12` could not be
checked on NVIDIA's current H100 page, which now lists H100 SXM and H100 NVL (NVL: 1,671 TFLOPS BF16
with sparsity, i.e. ~835 dense). Consider replacing the PCIe entry with `"H100-NVL": 835e12` or
citing a PCIe datasheet. `frontierlab.perf.roofline.HARDWARE` carries peaks *and* bandwidths with
source URLs; `flops.PEAK_BF16` could be derived from it.
