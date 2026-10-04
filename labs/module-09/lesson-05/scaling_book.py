"""Worked numbers for lesson 09.5 (extension): torus topology, sharded-matmul communication, storage bandwidth.

    python labs/module-09/lesson-05/scaling_book.py

Every number is arithmetic on published specifications (labelled with their source); nothing here is measured.
TPU figures are the JAX Scaling Book's TPU table (https://jax-ml.github.io/scaling-book/tpus/, read 2026-10-04);
H100 NVLink is NVIDIA's (company claim), with the achieved bus bandwidth an ASSUMPTION as in 09.1; 3FS is DeepSeek's
README (company claim).
"""

from __future__ import annotations

from pathlib import Path

from frontierlab.labkit import load_target

lab = load_target(str(Path(__file__).parent / "test_lab.py"))

TPU = {  # Scaling Book TPU table: bf16 FLOP/s per chip, HBM bytes/s, ICI one-way bytes/s per link
    "v4p": {"flops": 2.75e14, "hbm": 1.2e12, "ici": 4.5e10, "pod": (16, 16, 16)},
    "v5p": {"flops": 4.59e14, "hbm": 2.8e12, "ici": 9.0e10, "pod": (16, 20, 28)},
}


def main():
    print("1. Topology")
    for name, d in (("TPU v4 cube 4x4x4", (4, 4, 4)), ("v4p pod 16x16x16", TPU["v4p"]["pod"]),
                    ("v5p pod 16x20x28", TPU["v5p"]["pod"])):
        worst = sum(n // 2 for n in d)
        print(f"   {name:18s} chips {d[0] * d[1] * d[2]:6,d}  worst-case hops {lab.torus_hops((0,) * 3, tuple(n // 2 for n in d), d):3d}"
              f" (formula {worst})  bisection links {lab.bisection_links(d):5,d} (without wraparound {lab.bisection_links(d, False):,d})")

    print("\n2. All-gather of Llama 3 8B's weights in bf16 (16 GB) over one axis")
    for name, chips in (("v5p, axis of 16", 16), ("v5p, axis of 4", 4)):
        t = lab.collective_time("allgather", 16e9, chips, TPU["v5p"]["ici"])
        print(f"   {name:16s} {t * 1e3:7.1f} ms  (one-way {TPU['v5p']['ici'] / 1e9:.0f} GB/s per link, both ring directions)")
    t_h100 = 16e9 * 7 / 8 / 300e9
    print(f"   8x H100 NVLink   {t_h100 * 1e3:7.1f} ms  (ASSUMED 300 GB/s achieved bus bandwidth, lesson 09.1)")

    print("\n3. Sharded matmuls (Scaling Book notation, mesh axes X, Y)")
    cases = [(("X", ""), ("", "Y"), "A[I_X, J] . B[J, K_Y]"), (("", "X"), ("", ""), "A[I, J_X] . B[J, K]"),
             (("", "X"), ("X", ""), "A[I, J_X] . B[J_X, K]"), (("X", ""), ("", "X"), "A[I_X, J] . B[J, K_X]")]
    for lhs, rhs, txt in cases:
        print(f"   {txt:24s} -> {lab.matmul_comm(lhs, rhs)}")
    print("   Megatron TP is case 3 for the second matmul of an MLP: W_in column-sharded, W_out row-sharded,"
          " one all-reduce (or reduce-scatter with SP) per MLP.")

    print("\n4. Checkpoint bandwidth: a 1T-parameter model with fp32 master weights and AdamW state (12 bytes/param)")
    size = 1e12 * 12
    for name, bw in (("3FS peak read, 180 storage nodes (6.6 TiB/s, company claim)", 6.6 * 2**40),
                     ("Llama 3 Tectonic sustained (2 TB/s, section 3.3.1)", 2e12),
                     ("Llama 3 Tectonic peak (7 TB/s)", 7e12)):
        print(f"   {name:58s} {size / bw:6.1f} s")


if __name__ == "__main__":
    main()
