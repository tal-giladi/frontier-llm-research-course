# Experiment contract

Fill this in **before** you run anything that compares two or more things. Keep it next to your run
cards. If you change any line after seeing results, say so in the "Changes after results" section —
that is allowed, hiding it is not.

## Question and decision

- **Question:** one sentence. (Example: "At 32K context, does MLA give lower decode memory than GQA without losing more than 0.02 nats of held-out loss?")
- **Decision it informs:** what you will do differently depending on the answer.

## Hypothesis

- **Hypothesis:**
- **Status:** established effect / reported effect (cite) / may not appear at this scale.

## Baseline

- **Baseline run card:** path or name.
- **How it was tuned:** what was swept, how many runs, on which split.

## What changes and what is held fixed

- **Changed variable:** exactly one, unless this is an integration or factorial experiment (then list the cells).
- **Held fixed:** data version and hashes, tokens, tokenizer, seed set, evaluation version, hardware type, software versions.

## Comparison axis

Choose one and say why: **equal tokens**, **equal parameters**, **equal training FLOPs** or **equal wall-clock**. Say what this axis does *not* answer (for example: equal tokens says nothing about which is cheaper to train).

## Budget

- GPU type × count, GPU-hours, wall-clock, projected or measured cost.
- Extra compute counted in the comparison: teacher / judge / verifier / data generator.
- Tuning budget per arm (the baseline gets the same).

## Metrics and decision rule

- **Primary metric** and its uncertainty method (seeds, paired bootstrap over items, interval level).
- **Secondary metrics.**
- **Noise floor:** seed std of the baseline on the primary metric, and the minimum detectable effect with your seed count.
- **Decision rule, stated now:** (Example: "Adopt MLA if the 95% CI of the loss difference lies within ±0.02 and decode memory is at least 30% lower.")

## Correctness checks (must pass before results count)

- [ ] gradient check / causal check / cached-decode agreement / recurrent-vs-parallel equivalence / reference agreement (as applicable)
- [ ] same data, same eval windows, same seeds as the baseline
- [ ] budget parity verified from the run cards

## Fallback evidence

If the effect may not appear at this scale: which provided checkpoint or trace you will analyse instead, and the label you will use ("analysis of provided traces", not "reproduction").

## Limits of the conclusion

Scale, data, architecture, hardware, and what result would change your answer.

## Changes after results

What you changed after seeing results, and why.
