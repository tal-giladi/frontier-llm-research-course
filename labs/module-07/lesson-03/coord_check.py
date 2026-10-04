"""Lab 07.3, step 2: the coordinate check — does one training step change activations by the same amount at every width?

    python labs/module-07/lesson-03/coord_check.py            # free CPU, under a minute

For widths 64, 128, 256 and 512 (head_dim fixed at 32, heads and SwiGLU width scaled), builds the toy model in
SP (every weight N(0, 0.02^2), one learning rate) and in µP relative to width 128 (``frontierlab.optim.mup``),
takes 5 AdamW steps at learning rate 1e-2 on the same Data-v0 batches, and prints the RMS of the logits and of the
last block's output before training and the RMS of their *change* after 5 steps. In µP the change should be
roughly flat across width; in SP it grows with width (TP V section 3; the "coordinate check" of the mup package).
"""

import torch

from frontierlab.data.loader import TokenData
from frontierlab.model import toy
from frontierlab.optim import mup
from frontierlab.optim.muon import make_optimizer
from frontierlab.optim.stabilizers import OptLM

WIDTHS = (64, 128, 256, 512)


@torch.no_grad()
def outputs(model, x):
    acts = {}
    h = model.model.layers[-1].register_forward_hook(lambda m, a, o: acts.__setitem__("last", o))
    was = model.training
    model.eval()
    logits = model(x).logits
    model.train(was)
    h.remove()
    return logits.clone(), acts["last"].clone()


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


def run(width, param, data, lr=1e-2, steps=5, base=128):
    torch.manual_seed(0)
    cfg = mup.width_config(toy(vocab_size=data.meta["vocab_size"]), width)
    model = OptLM(cfg)
    scales = None
    if param == "mup":
        mup.apply_mup(model, base)
        scales = mup.lr_scales(model, base, "adamw")
    opt = make_optimizer(model, optimizer="adamw", lr=lr, weight_decay=0.0, lr_scales=scales)
    probe = data.batch(8, 128, torch.Generator().manual_seed(99))
    l0, h0 = outputs(model, probe)
    gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        x = data.batch(16, 128, gen)
        opt.zero_grad(set_to_none=True)
        model(x, labels=x).loss.backward()
        opt.step()
    l1, h1 = outputs(model, probe)
    return {"logits": rms(l0), "dlogits": rms(l1 - l0), "last": rms(h0), "dlast": rms(h1 - h0)}


def main():
    data = TokenData("train")
    print("RMS before training and RMS of the change after 5 AdamW steps at lr 1e-2 (muP base width 128)")
    print(f"{'width':>6s} {'param':>5s} {'logits':>9s} {'dlogits':>9s} {'last block':>11s} {'dlast':>9s}")
    res = {}
    for p in ("sp", "mup"):
        for w in WIDTHS:
            r = res[(p, w)] = run(w, p, data)
            print(f"{w:6d} {p:>5s} {r['logits']:9.4f} {r['dlogits']:9.4f} {r['last']:11.4f} {r['dlast']:9.4f}")
    for p in ("sp", "mup"):
        g = res[(p, WIDTHS[-1])]["dlogits"] / res[(p, WIDTHS[0])]["dlogits"]
        h = res[(p, WIDTHS[-1])]["dlast"] / res[(p, WIDTHS[0])]["dlast"]
        print(f"{p}: change at width {WIDTHS[-1]} / change at width {WIDTHS[0]}: logits {g:.2f}, last block {h:.2f}")


if __name__ == "__main__":
    main()
