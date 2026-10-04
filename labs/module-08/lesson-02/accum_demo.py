"""Lab 08.2, step 2: why DeepSeek-V3 promotes FP8 GEMM partial sums to FP32 every 128 elements.

    python labs/module-08/lesson-02/accum_demo.py

A MODEL of the accumulator (frontierlab.precision.accum): the running sum keeps ``mant_bits`` bits after the
leading one and drops the rest after every add (truncation), the products of E4M3 inputs are exact. DeepSeek-V3
section 3.3.2 states the H800 keeps "around 14 bits" and reports a maximum relative error of nearly 2% for
K = 4096; the hardware's exact behaviour is not public, so the bit count that reproduces it is a fit, not a fact.
Inputs are non-negative (uniform [0, 1)), the case where the sum grows like K. Runtime: about 1 minute on a laptop.
"""

import argparse

from frontierlab.precision.accum import accumulation_error


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="dot products per setting")
    a = ap.parse_args(argv)
    Ks = (128, 512, 1024, 2048, 4096, 8192)
    print("max relative error over", a.n, "dot products (truncating accumulator model)\n")
    print(f"{'mant bits':>9s} {'promote':>8s} | " + " ".join(f"K={K:<6d}" for K in Ks))
    for bits in (13, 14, 15, 23):
        for nc in (None, 128):
            errs = [accumulation_error(K, n=a.n, mant_bits=bits, promote_every=nc)["max_rel_err"] for K in Ks]
            print(f"{bits:9d} {str(nc):>8s} | " + " ".join(f"{e * 100:7.3f}%" for e in errs))
    print("\nround-to-nearest accumulator instead of truncation, 13 bits, no promotion:")
    print(" ".join(f"K={K}: {accumulation_error(K, n=a.n, mant_bits=13, mode='rne')['max_rel_err'] * 100:.3f}%" for K in Ks))


if __name__ == "__main__":
    main()
