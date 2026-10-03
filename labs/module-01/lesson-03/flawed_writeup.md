# Experiment report: "GatedMix" beats SwiGLU

*A fictional internal write-up for lab 01.3. Every number is invented. Find the flaws; do not fix
the prose, write a corrected experiment contract instead.*

## Summary

We replaced Baseline-0's SwiGLU feed-forward with **GatedMix**, our new gated mixing layer. GatedMix
reaches a held-out loss of **3.391** against SwiGLU's **3.412**, a 0.021-nat improvement. We
recommend adopting GatedMix for all future models, including the frontier-scale run next quarter.

## Setup

- **Baseline:** the Baseline-0 run `b0-seed0` from last month (learning rate 3e-3, warmup 200, cosine,
  9,500 steps, Data-v0).
- **GatedMix:** Baseline-0 with every SwiGLU block replaced by GatedMix. GatedMix's inner width was
  set to 3,328 (instead of 2,816) "so it has room to mix", which brings the model to 140M
  parameters instead of 122M. While we were at it we also moved the warmup to 500 steps, which we
  have found to be more stable.
- **Tuning:** we swept GatedMix's learning rate over {1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3} and kept the
  best. The baseline was not re-tuned, since its learning rate is the one we always use.
- **Training length:** GatedMix was still improving at 9,500 steps, so we let it run to 10,500.
- **Data:** Data-v0, re-prepared this week with the latest FineWeb-Edu snapshot (the old shards had
  been deleted to save disk).

## Evaluation

- We evaluated every 500 steps on the **test** split and report the best checkpoint of each run.
  The learning rate for GatedMix was also chosen by test loss.
- Evaluation used 128 windows for GatedMix and the 256 windows logged in `b0-seed0`'s run, because
  evaluation is slow with the larger model.
- One run per arm (seed 0). The improvement of 0.021 is larger than the 0.01 we think of as
  meaningful, so the result is significant.

## Conclusion

GatedMix is better than SwiGLU. The gain should grow with scale, because larger models benefit more
from better mixing.
