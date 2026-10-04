---
id: "06.6"
module: 6
minutes: 25
practice_minutes: 35
prerequisites: ["06.4"]
objectives:
  - Explain how the Byte Latent Transformer replaces a fixed tokenizer by entropy-based byte patches and why the average patch size, not the vocabulary, sets its cost.
  - Implement byte entropy, the global and approximately monotonic patch-boundary rules and BLT's FLOPs-per-byte cost model, and verify them against the shared code.
  - Measure where byte entropy is high in Data-v0 text and how thresholds map to patch sizes, and compare the resulting FLOPs per byte with the course's BPE model.
volatility: concept
sources:
  - title: "Pagnoni et al. (Meta) — Byte Latent Transformer: Patches Scale Better Than Tokens (sections 2.1–2.3, 3, 4.2–4.5)"
    url: https://arxiv.org/abs/2412.09871
last_verified: "2026-10-04"
---

# 06.6 · Tokenizer-free models: Byte Latent Transformer

Extension: every model in this course reads tokens from a fixed BPE vocabulary, so every token costs one full pass of the transformer, whether it is the hard first word of a sentence or the obvious end of a long word. The Byte Latent Transformer (BLT) reads raw bytes, groups them into variable-length patches where a small byte model finds them hard to predict, and runs the large transformer once per patch. This lesson builds the patching rule and measures it on Data-v0's text; it does not train a BLT.

## Why this matters at a frontier lab

Tokenizers are fixed before pretraining and are hard to change: they decide how many steps a text costs, how numbers and rare scripts are split, and how robust the model is to typos. BLT reports that byte-level models with entropy patches can match a tokenized model's training-FLOP-controlled performance up to 8B parameters and 4T training bytes "while using up to 50% fewer flops at inference" (section 1), and that patch size is a new scaling axis: at a fixed inference budget, larger patches free compute for a larger latent model (Figure 1). The question it raises for a lab is not "should we drop the tokenizer tomorrow" but *what does compute per byte depend on, and where would a learned segmentation spend it?*

## The idea

### Patches instead of tokens

BLT has three parts (section 3): a light **local encoder** that turns bytes into patch representations (with cross-attention from patches to bytes and hashed byte n-gram embeddings, section 3.2.1 — the same lookup trick as Engram, lesson 06.4); a large **latent transformer** that runs autoregressively over patches with a block-causal mask; and a light **local decoder** that predicts the next patch's bytes. The latent transformer "consumes the bulk of the flops", so the number of patches sets the cost (section 3.1).

### Where to cut: entropy

A small byte LM $p_e$ gives the next-byte entropy (section 2.3, Eq. 1, with the minus sign the printed equation omits)

$$H(x_i) = -\sum_{v \in \mathcal{V}} p_e(x_i = v \mid x_{1..i-1}) \log p_e(x_i = v \mid x_{1..i-1}).$$

Two rules turn entropies into boundaries:

- **global:** byte $i$ starts a patch if $H(x_i) > \theta_g$;
- **approximately monotonic:** byte $i$ starts a patch if $H(x_i) - H(x_{i-1}) > \theta_r$, i.e. where entropy jumps up, breaking "approximate monotonically decreasing entropy within the patch".

The threshold is set to reach a target average patch size on the training data (section 4.3); context length is kept equal in bytes across patch sizes, so larger patches do not get an unfair advantage. The idea is that "the first bytes in words are typically most difficult" (section 2.2): one latent step for the hard choice, and the easy rest of the word inside the same patch.

### Cost per byte

With $F_g$ FLOPs per latent step, $F_\ell$ per byte for the local models and average patch size $\bar p$:

$$\text{FLOPs per byte} = \frac{F_g}{\bar p} + F_\ell .$$

A BPE model is the same formula with $\bar p$ = bytes per token and $F_\ell \approx 0$. BLT counts the embedding lookup as 0 FLOPs (section 4.5).

## Worked example

### Entropy of three distributions

- Uniform over 4 bytes: $H = \ln 4 = 1.386$ nats.
- $(0.9, 0.1)$: $-(0.9 \ln 0.9 + 0.1 \ln 0.1) = 0.0948 + 0.2303 = 0.325$ nats.
- Uniform over all 256 byte values: $\ln 256 = 5.545$ nats — the maximum.

### Cutting ten bytes

Entropies $H = (3.1, 0.2, 0.4, 2.2, 1.9, 0.3, 2.8, 0.1, 0.05, 1.4)$.

- Global, $\theta_g = 1.5$: starts at 0, 3, 4, 6 → patches of 3, 1, 2, 4 bytes; mean 2.5.
- Monotonic, $\theta_r = 1.0$: jumps $H_i - H_{i-1}$ are $-2.9, 0.2, 1.8, -0.3, -1.6, 2.5, -2.7, -0.05, 1.35$: starts at 0, 3, 6, 9 → patches 3, 3, 3, 1; mean 2.5. Byte 4 (1.9) does not start a patch under the monotonic rule: it is high, but lower than the byte before it.

### Cost of Baseline-0 as a latent model

Baseline-0's forward pass at $T = 1024$ is 263M FLOPs per step (lesson 01.1). Run once per BPE token at 3.99 bytes per token (Data-v0, measured below): 65.9M FLOPs per byte. As a BLT latent model with 4.5-byte patches and local models assumed at 13M FLOPs per byte (5% of a latent step; a course assumption, BLT's local models are sized per experiment): $263/4.5 + 13 = 71$M FLOPs per byte — *more* than BPE; with 8-byte patches 46M — 30% less. The saving comes from patches longer than the tokenizer's tokens, not from bytes as such.

## Shapes and cost

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| bytes | (B, S_bytes) | uint8 / int64 ids 0–255 | CPU (data loader) |
| next-byte entropies | (B, S_bytes) | fp32 | wherever the entropy model runs; BLT computes them "during dataloading" (section 2.3) |
| patch starts | variable per sequence | int64 | CPU |
| latent sequence | (B, S_bytes / $\bar p$, C) | bf16 | GPU |

The entropy model is a per-byte cost of its own; BLT's default is a 100M-parameter, 14-layer byte transformer with a 512-byte sliding window (section 4.2), run in preprocessing. The course uses a count-based order-2 byte model — "when the receptive field of the model is small enough, the trained entropy model can be encoded in an efficient lookup table" (section 4.2).

## Build it

`labs/common/frontierlab/blocks/patching.py`: `NgramByteModel` (count-based byte model, add-α smoothed), `patch_starts` (global and monotonic), `patch_lengths`, `threshold_for_size` (bisection on θ), `latent_flops_per_byte`. Tests in `test_blocks.py` check the boundary rules on a hand-made sequence (the worked example), the threshold search, and the entropy of a deterministic context (near 0) and of an unseen one ($\ln 256$).

## What the evidence says

- **Entropy patching — PROMISING, one lab.** PUBLICLY DOCUMENTED by Meta (BLT): FLOP-controlled scaling to 8B parameters and 4T bytes; BLT models with patch size 6 and 8 "quickly overtake scaling trends of bpe Llama 2 and 3" at fixed inference FLOPs (Figure 1); robustness to noisy input and character-level awareness (section 1). The large run used entropy patches of average size 4.5 with the entropy context reset at newlines and the monotonic rule, because global thresholds suffered "entropy drift" in repetitive content (section 4.4).
- **Not adopted** by the frontier open models in this course's reference table, which all use BPE vocabularies (INFERENCE from their published configs; see `references/frontier-models-2026-10.md`).
- **Open questions:** training stability and infrastructure for variable-length patches at larger scale; how patching interacts with long-context attention and KV-cache size; whether the gain holds against stronger tokenizers.

## Lab

**Folder:** [`labs/module-06/lesson-06/`](../../labs/module-06/) · **Time:** about 35 minutes · **Pass check:** `pytest labs/module-06/lesson-06` passes; your notes contain the entropy comparison (after a space vs inside words), the threshold table, the FLOPs-per-byte comparison with BPE, and one patched sentence with a comment on where the cuts fall.

This lab measures, it does not compare methods, so it has no experiment contract. Its one modelling assumption — the local models' cost of 5% of a latent step — is printed with the result; change it and see what moves.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | not needed (CPU arithmetic and counting) | as Free CPU |
| Free GPU | not needed | as Free CPU |
| Free CPU | laptop; measured below | `python labs/module-06/lesson-06/patching_study.py` |

### Steps

1. **Implement** `entropy`, `patch_starts`, `mean_patch_size` and `flops_per_byte`; run `pytest labs/module-06/lesson-06`.
2. **Run** `python labs/module-06/lesson-06/patching_study.py`.
3. **Explain** in three sentences: why the first byte after a space has higher entropy; why the global rule leaves more one-byte patches than the monotonic rule at the same size; and at what patch size a Baseline-0-sized latent model starts to cost fewer FLOPs per byte than the BPE model.

Measured in this build (free CPU: Windows 11, 16-thread laptop, torch 2.14.1+cpu, 2026-10-04; 29 s):

| Measurement | Result |
|---|---|
| text | 1.93 MB of decoded Data-v0 training text for the entropy model; 99,644 bytes of validation text = 25,000 BPE tokens (**3.99 bytes per token**) |
| mean next-byte entropy, order 1 / 2 / 3 | 2.544 / 2.216 / 2.211 nats (3.67 / 3.20 / 3.19 bits) |
| order 2: first byte after a space vs inside a word | **3.250 vs 2.037 nats** |
| patch size 4.5: global θ 2.657, achieved 4.54, 16% one-byte patches; monotonic θ 0.816, achieved 4.47, 5% one-byte | 71 MFLOP/byte (BPE: 65.9) |
| patch size 8: global 8.48 (13% one-byte); monotonic 7.99 (1%) | 44–46 MFLOP/byte |

A sentence at the 4.5-byte global threshold: `Fortunately, |the |International |O|r|ganization |of |S|tandar|dization, |more |commonly |referred |to |as |IS|O|,| |created |a |framework |in` — cuts fall at word starts, and the order-2 model, which only sees two bytes back, also cuts inside rare words ("O|r|ganization", "S|tandar|dization"): a better entropy model would not, which is why BLT uses a transformer for it.

<details>
<summary>Reference solution</summary>

`labs/module-06/lesson-06/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-06/lesson-06`.

</details>

## Common mistakes

- **Dropping the minus sign.** $\sum p \log p$ is negative; a threshold on it cuts in the wrong places.
- **Comparing patch sizes at equal *patches* per batch.** Larger patches then see more bytes; BLT equalises bytes per batch (section 4.3).
- **Treating the entropy model as free.** It runs on every byte, in preprocessing or online; at inference it is part of the latency.
- **Assuming bytes are cheaper than tokens.** At the same latent model, BLT costs fewer FLOPs per byte only when its patches are longer than the tokenizer's tokens (here, longer than about 4 bytes plus the local models' share).

## References

- A. Pagnoni et al. (Meta), *Byte Latent Transformer: Patches Scale Better Than Tokens*, sections 1, 2.1–2.3, 3, 4.2–4.5, Figure 1. https://arxiv.org/abs/2412.09871
- Shared code: `labs/common/frontierlab/blocks/patching.py`. Software versions: [references/versions.md](../../references/versions.md).

## Next

[06.7 · Non-autoregressive and latent reasoning](lesson-07.md)
