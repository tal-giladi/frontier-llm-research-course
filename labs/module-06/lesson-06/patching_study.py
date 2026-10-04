"""Lab 06.6: entropy patching on Data-v0 text, compared with the course's BPE tokenizer.

    python labs/module-06/lesson-06/patching_study.py              # about 2 minutes on a laptop

1. Decode Data-v0 token windows back to UTF-8 bytes with the course tokenizer: about 2 MB of training text for
   the entropy model, 100 KB of validation text to patch.
2. Fit count-based byte models of order 1, 2 and 3 (the "entropy model"; BLT uses a 100M byte transformer,
   section 4.2) and compute the next-byte entropy H(x_i) of every validation byte.
3. Where is entropy high? Mean H of the first byte after a space vs of bytes inside words.
4. For target average patch sizes 2, 4, 4.5, 6 and 8 bytes: the global and the monotonic threshold that achieve
   it (bisection), the achieved size, the share of one-byte patches, and FLOPs per byte for a latent transformer
   of Baseline-0's size against the BPE model, with your lab's cost formula.
5. One sentence printed with its patch boundaries.
"""

import os
from pathlib import Path

from tokenizers import Tokenizer

from frontierlab.blocks.patching import NgramByteModel, patch_lengths, threshold_for_size
from frontierlab.data.loader import TokenData
from frontierlab.data.prepare import DEFAULT_OUT
from frontierlab.flops import flops_per_token
from frontierlab.labkit import load_path
from frontierlab.model import baseline0

lab = load_path(str(Path(__file__).parent / f"{os.environ.get('LAB_TARGET', 'lab')}.py"))


def text_bytes(split: str, n_tokens: int, tok: Tokenizer) -> tuple[bytes, int]:
    d = TokenData(split)
    ids = d.window(0, min(n_tokens, len(d) - 1)).tolist()
    data = tok.decode(ids).encode("utf-8")
    return data, len(ids)


def main():
    tok = Tokenizer.from_file(str(DEFAULT_OUT / "tokenizer.json"))
    train, _ = text_bytes("train", 500_000, tok)
    val, val_tokens = text_bytes("val", 25_000, tok)
    bpe = len(val) / val_tokens
    print(f"entropy-model text {len(train):,} bytes; validation text {len(val):,} bytes = {val_tokens:,} BPE tokens "
          f"({bpe:.2f} bytes per BPE token)")
    Hs = {}
    for order in (1, 2, 3):
        model = NgramByteModel(train, order=order)
        Hs[order] = model.entropies(val)
        print(f"order-{order} byte model: mean H {sum(Hs[order]) / len(val):.3f} nats/byte "
              f"({sum(Hs[order]) / len(val) / 0.6931:.3f} bits)")
    H = Hs[2]
    after_space = [H[i] for i in range(1, len(val)) if val[i - 1:i] == b" "]
    inside = [H[i] for i in range(1, len(val)) if val[i - 1:i].isalpha() and val[i:i + 1].isalpha()]
    print(f"order 2: mean H of the first byte after a space {sum(after_space) / len(after_space):.3f}; "
          f"inside words {sum(inside) / len(inside):.3f}  (BLT: 'the first bytes in words are typically most difficult')")
    b0 = baseline0(vocab_size=32768)
    global_per_patch = flops_per_token(b0, 1024, training=False)          # a Baseline-0-sized latent model
    local_per_byte = global_per_patch / 20                                 # course assumption: local models ~5% of it
    print(f"\nlatent model = Baseline-0 forward {global_per_patch / 1e6:.0f} MFLOP per patch; local models assumed "
          f"{local_per_byte / 1e6:.0f} MFLOP per byte (a stated assumption)")
    print(f"BPE model (one Baseline-0 step per token): {global_per_patch / bpe / 1e6:.1f} MFLOP per byte")
    print(f"{'target':>6s} {'mode':>9s} {'theta':>7s} {'size':>6s} {'1-byte':>7s} {'MFLOP/byte':>11s}")
    for target in (2.0, 4.0, 4.5, 6.0, 8.0):
        for mode in ("global", "monotonic"):
            th = threshold_for_size(H, target, mode)
            starts = lab.patch_starts(H, th, mode)
            size = lab.mean_patch_size(starts, len(val))
            lens = patch_lengths(starts, len(val))
            one = sum(1 for x in lens if x == 1) / len(lens)
            print(f"{target:6.1f} {mode:>9s} {th:7.3f} {size:6.2f} {one:7.2f} "
                  f"{lab.flops_per_byte(global_per_patch, local_per_byte, size) / 1e6:11.1f}")
    th = threshold_for_size(H, 4.5, "global")
    start = val.find(b". ") + 2
    piece = val[start:start + 120]
    Hp = H[start:start + 120]
    cuts = set(lab.patch_starts(Hp, th, "global"))
    shown = "".join(("|" if i in cuts and i else "") + chr(b) if b < 128 else "?" for i, b in enumerate(piece))
    print(f"\nglobal threshold for 4.5-byte patches ({th:.3f} nats), patch starts marked with |:\n{shown}")


if __name__ == "__main__":
    main()
