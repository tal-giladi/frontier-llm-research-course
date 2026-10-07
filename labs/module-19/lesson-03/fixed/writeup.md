# Research note: QK-Clip and QK-norm at toy scale (revised)

*The revised version of the fictional note in `../flawed/`, for lab 19.3. The runs exist only as EXAMPLE run cards in
`cards/`, and every number is invented. It is what the note should have said given evidence of this kind.*

## Summary

At the toy preset (1.8M parameters) with Muon at learning rate 1e-2, 300 steps and three seeds per arm, QK-Clip
($\tau = 15$) and QK-norm reach the same held-out loss within our resolution: the paired difference (QK-Clip minus
QK-norm) is −0.004 nats, 95% interval [−0.016, +0.008] over seeds, against a seed noise floor of 0.0286 nats
(lesson 01.4). Both bound the maximum attention logit (QK-Clip by construction, at about $\tau$). We cannot
distinguish them on loss at this scale; the Recipe-R decision should rest on the cost and compatibility argument
(QK-Clip works with MLA, QK-norm does not), and on a `pilot-30m` run if loss matters.

## Setup

- Both arms: toy preset, Muon with AdamW for embeddings and norms, learning rate 1e-2, cosine with 50 warm-up steps,
  Data-v0 at the CPU size, 300 steps of 16 × 128 tokens, seeds 0, 1 and 2 (initialisation and data order shared by
  seed across arms).
- QK-Clip arms: QK-norm off, QK-Clip with $\tau = 15$. QK-norm arms: QK-norm on, no clip.
- Evaluation: the final checkpoint of every run, 256 fixed windows of the validation split. The test split was not
  used.

## Result

Figure 1 is specified in `figure.yaml`: final held-out loss per arm, one point per seed, with the paired interval.

## Claims register

```yaml claims
- id: c1
  text: "At 1.8M parameters, QK-Clip (tau 15) and QK-norm reach the same held-out loss within 0.02 nats."
  metric: val_loss
  arms: {qk-clip: [clip-s0, clip-s1, clip-s2], qk-norm: [norm-s0, norm-s1, norm-s2]}
  changed: [config.qk_norm, optim.qk_clip, optim.qk_norm]
  axis: tokens
  value: -0.004
  interval: [-0.016, 0.008]
  checkpoint: {qk-clip: final, qk-norm: final}
  selection_split: val
  scope: "toy preset (1.8M parameters), 300 steps, Data-v0 CPU size, Muon at lr 1e-2"
```

## Limits

One size; one learning rate; 0.6M tokens per run; three seeds resolve differences of about 0.07 nats at 80% power
(minimum detectable effect for 3 seeds and a seed std of 0.0286), so "same within 0.02" is bounded by the interval
above, not by power; Kimi K2's logit growth appeared at 9B activated parameters and nothing here tests that regime.
