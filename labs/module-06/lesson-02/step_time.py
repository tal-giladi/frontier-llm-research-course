"""Lab 06.2, step 5: the wall-clock cost of the n-stream residual, measured with the Module 2 harness.

    python labs/module-06/lesson-02/step_time.py                                   # CPU, toy, 16 x 128
    python labs/module-06/lesson-02/step_time.py --device cuda --preset pilot-30m --batch 16 --seq 1024 --vocab 32768 --bf16

One training step (forward, backward, AdamW step) of b0, HC and mHC (n = 4, t_max = 20) on the same batch,
interleaved rounds (``frontierlab.perf.interleaved``), and the paired per-round time ratio against b0 with a
95% bootstrap interval (``frontierlab.perf.speedup``). Next to it: the FLOP ratio from
``frontierlab.blocks.accounting`` and the residual's memory traffic from mHC Table 2. The difference between
the time ratio and the FLOP ratio is the point of lesson 06.2: HC/mHC cost memory traffic and small kernels,
not FLOPs. mHC reports 6.7% time overhead at n = 4 *with* fused kernels, recomputation and DualPipe overlap
(section 4.3); this course's implementation is unfused PyTorch, so expect much more.
"""

import argparse

import torch

from frontierlab.blocks import BlockLM, accounting, with_blocks
from frontierlab.blocks.hyperconn import residual_io_elements
from frontierlab.model import PRESETS
from frontierlab.perf import interleaved, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="toy")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    base = PRESETS[a.preset](vocab_size=a.vocab)
    arms = {"b0": with_blocks(base), "hc": with_blocks(base, residual="hc"), "mhc": with_blocks(base, residual="mhc")}
    torch.manual_seed(0)
    x = torch.randint(0, a.vocab, (a.batch, a.seq), device=a.device)
    steps = {}
    for name, cfg in arms.items():
        torch.manual_seed(0)
        model = BlockLM(cfg).to(a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step(model=model, opt=opt):
            with torch.autocast(device_type=torch.device(a.device).type, dtype=torch.bfloat16, enabled=a.bf16):
                loss = model(x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        steps[name] = step
    t = interleaved(steps, warmup=3, rounds=a.rounds, device=a.device)
    C = base.hidden_size
    f0 = accounting.flops_per_token(arms["b0"], a.seq)
    io0 = residual_io_elements(1, C, "plain")
    print(f"{a.preset}, batch {a.batch} x {a.seq}, {a.device}{' bf16' if a.bf16 else ''}, {a.rounds} interleaved rounds")
    print(f"{'arm':4s} {'median step':>12s} {'time ratio vs b0 [95% CI]':>28s} {'FLOP ratio':>10s} {'residual I/O ratio':>18s}")
    for name in arms:
        r = speedup(t[name], t["b0"]) if name != "b0" else {"speedup": 1.0, "ci": (1.0, 1.0)}
        fr = accounting.flops_per_token(arms[name], a.seq) / f0
        io = residual_io_elements(4, C, "hc") if name != "b0" else io0
        ior = (io["read"] + io["write"]) / (io0["read"] + io0["write"])
        print(f"{name:4s} {t[name].median * 1e3:10.1f} ms {r['speedup']:12.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
              f" {fr:10.3f} {ior:18.1f}")


if __name__ == "__main__":
    main()
