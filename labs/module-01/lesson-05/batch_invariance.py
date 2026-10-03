"""Batch invariance on your machine: does a row's result depend on what else is in the batch? (lab 01.5)

    python labs/module-01/lesson-05/batch_invariance.py              # CPU, ~10 s
    python labs/module-01/lesson-05/batch_invariance.py --device cuda  # GPU (main path)

1. Floating-point addition is not associative: the order of a sum changes its last bits.
2. ``torch.mm(a[:1], b)`` vs ``torch.mm(a, b)[:1]``: the same row, computed alone or inside a batch.
   The Thinking Machines post ("Defeating Nondeterminism in LLM Inference", 2025) shows this on a GPU;
   this script measures it on your hardware.
3. The course model: logits of one sequence run alone vs as row 0 of batches of 2..32 (fp32).
Nothing here is random between runs: rerun the script and you get the same numbers. That is the
point of the post: the variation comes from batch composition (server load), not from "random"
kernels.
"""

import argparse

import torch

from frontierlab.model import LM, toy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device
    print(f"torch {torch.__version__}, device {dev}, threads {torch.get_num_threads()}")

    x = [0.1, 0.2, 0.3]
    print(f"\n1. (0.1 + 0.2) + 0.3 = {(x[0] + x[1]) + x[2]!r}   0.1 + (0.2 + 0.3) = {x[0] + (x[1] + x[2])!r}")

    g = torch.Generator(device="cpu").manual_seed(0)
    A = torch.randn(2048, 2048, generator=g).to(dev)
    B = torch.randn(2048, 2048, generator=g).to(dev)
    ref = torch.mm(A, B)[:1]
    print("\n2. max |mm(a[:n], b)[0] - mm(a, b)[0]|, fp32")
    for n in (1, 2, 8, 64, 512, 2048):
        d = (torch.mm(A[:n], B)[:1] - ref).abs().max().item()
        print(f"   batch rows {n:5d}: {d:.3e}")

    torch.manual_seed(0)
    model = LM(toy(vocab_size=512)).to(dev).eval()
    seqs = torch.randint(0, 512, (32, 64), generator=g).to(dev)
    with torch.no_grad():
        alone = model(seqs[:1]).logits
        print("\n3. course model, max |logits of sequence 0 alone - inside a batch|, fp32")
        for n in (2, 4, 8, 16, 32):
            d = (model(seqs[:n]).logits[:1] - alone).abs().max().item()
            print(f"   batch {n:3d}: {d:.3e}   bitwise identical: {d == 0.0}")
        again = model(seqs[:1]).logits
        print(f"   same batch twice: max diff {(again - alone).abs().max().item():.3e}  (run-to-run, same inputs)")


if __name__ == "__main__":
    main()
