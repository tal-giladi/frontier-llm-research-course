"""Lab 07.1, step 3: what Muon costs — Newton-Schulz FLOPs per step (projected) and measured step times.

    python labs/module-07/lesson-01/cost_table.py                  # free CPU: table + toy and pilot-10m timings
    python labs/module-07/lesson-01/cost_table.py --device cuda --presets pilot-30m baseline0 --batch 32 --seq 1024
                                                                    # main path (not run in this build)

Part 1 (arithmetic, any machine): for each preset, the Newton-Schulz FLOPs of one optimizer step
(``frontierlab.optim.cost``), the training FLOPs of one step at the stated batch
(``frontierlab.attention.accounting.flops_per_token`` x tokens), their ratio and Jordan's bound T·m/B.
Part 2 (measured, the device you run on): forward + backward + optimizer step for MuonAdamW in its AdamW
configuration and in its Muon configuration, interleaved rounds with warm-up and synchronisation
(``frontierlab.perf``), and the paired step-time ratio with a bootstrap interval; also the optimizer step
alone. These two numbers feed the equal-wall-clock arm of the module project.
"""

import argparse
import copy
import platform

import torch

from frontierlab.attention.accounting import flops_per_token
from frontierlab.model import LM, PRESETS
from frontierlab.optim import cost
from frontierlab.optim.muon import make_optimizer
from frontierlab.perf.timing import interleaved, speedup


def arithmetic(presets, vocab, tokens, seq):
    print(f"Part 1 - arithmetic (vocab {vocab}, {tokens:,} tokens per optimizer step, T = {seq})")
    print(f"{'preset':10s} {'Muon mats':>9s} {'NS GFLOP/step':>14s} {'train GFLOP/step':>17s} {'NS share':>9s} {'T*m/B':>8s}")
    for name in presets:
        cfg = PRESETS[name](vocab_size=vocab)
        shapes = cost.muon_shapes(cfg)
        ns = cost.shape_flops(shapes, 5)
        train = flops_per_token(cfg, seq) * tokens
        print(f"{name:10s} {len(shapes):9d} {ns / 1e9:14.2f} {train / 1e9:17.1f} {ns / train:9.2%} "
              f"{cost.jordan_bound(cfg.hidden_size, tokens):8.2%}")


def measure(name, vocab, batch, seq, device, rounds, dtype):
    cfg = PRESETS[name](vocab_size=vocab)
    torch.manual_seed(0)
    m_adam = LM(cfg).to(device)
    m_muon = copy.deepcopy(m_adam)
    o_adam = make_optimizer(m_adam, optimizer="adamw", lr=1e-3)
    o_muon = make_optimizer(m_muon, optimizer="muon", lr=1e-3)
    x = torch.randint(0, vocab, (batch, seq), generator=torch.Generator().manual_seed(0)).to(device)
    ac = torch.bfloat16 if (dtype == "bf16" and device.startswith("cuda")) else None

    def step(model, opt):
        def f():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=x.device.type, dtype=ac, enabled=ac is not None):
                loss = model(x, labels=x).loss
            loss.backward()
            opt.step()
        return f

    t = interleaved({"adamw": step(m_adam, o_adam), "muon": step(m_muon, o_muon)}, warmup=3, rounds=rounds,
                    device=device)
    sp = speedup(t["muon"], t["adamw"])          # > 1 means the AdamW step is faster
    only = interleaved({"adamw": o_adam.step, "muon": o_muon.step}, warmup=2, rounds=rounds, device=device)
    print(f"{name:10s} AdamW step {t['adamw'].median * 1e3:8.1f} ms   Muon step {t['muon'].median * 1e3:8.1f} ms   "
          f"Muon/AdamW {sp['speedup']:.3f} [{sp['ci'][0]:.3f}, {sp['ci'][1]:.3f}]   "
          f"optimizer only: AdamW {only['adamw'].median * 1e3:.1f} ms, Muon {only['muon'].median * 1e3:.1f} ms")
    return t, only


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", nargs="*", default=["toy", "pilot-10m"])
    ap.add_argument("--table-presets", nargs="*", default=["toy", "pilot-10m", "pilot-30m", "pilot-70m", "baseline0"])
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args()
    arithmetic(a.table_presets, a.vocab, a.batch * a.seq, a.seq)
    print("\nmain-path batch for comparison (32 x 1024 = 32,768 tokens; 16 x 32,768 = 524,288 with accumulation):")
    arithmetic(["pilot-30m", "baseline0"], 32768, 32 * 1024, 1024)
    arithmetic(["pilot-30m", "baseline0"], 32768, 16 * 32768, 1024)
    print(f"\nPart 2 - measured on {a.device} ({platform.processor() or platform.machine()}, torch {torch.__version__}, "
          f"{torch.get_num_threads()} threads), batch {a.batch} x {a.seq}, {a.rounds} interleaved rounds")
    for name in a.presets:
        measure(name, a.vocab, a.batch, a.seq, a.device, a.rounds, a.dtype)


if __name__ == "__main__":
    main()
