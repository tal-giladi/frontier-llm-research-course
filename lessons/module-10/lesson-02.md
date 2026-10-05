---
id: "10.2"
module: 10
minutes: 35
practice_minutes: 120
prerequisites: ["10.1", "01.4"]
objectives:
  - Describe the FineWeb-Edu, DCLM and Nemotron-CC quality classifiers (labels, model, threshold, ensemble) from their reports, and separate what each report measured from what it inferred.
  - Train a hashed bag-of-n-grams quality classifier in pure PyTorch on LLM-written educational scores, and report its imitation accuracy correctly, including why a fixed threshold on a regressor can fail.
  - Judge a filter by a matched training ablation at equal tokens with seeds, on held-out sets chosen so the filter's own target does not decide the result.
  - Explain, with the repetition each threshold forces at a fixed budget, why aggressive filtering that wins at a short horizon can lose at a long one.
volatility: concept
sources:
  - title: "Penedo et al., The FineWeb Datasets (section 4: FineWeb-Edu classifier, 460k Llama-3-70B annotations, threshold 3, F1 82%)"
    url: https://arxiv.org/abs/2406.17557
  - title: "Li et al., DataComp-LM (section 4.4 and Table 4: fastText OH-2.5 + ELI5 classifier, top 10%)"
    url: https://arxiv.org/abs/2406.11794
  - title: "Su et al., Nemotron-CC (abstract, sections 2.1-2.2, 3.2: classifier ensemble, 90% removal, 15T-token horizon)"
    url: https://arxiv.org/abs/2412.02595
  - title: "Muennighoff et al., Scaling Data-Constrained Language Models (abstract)"
    url: https://arxiv.org/abs/2305.16264
  - title: "HuggingFaceFW/fineweb-edu-llama3-annotations (the annotation dataset)"
    url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations
last_verified: "2026-10-04"
---

# 10.2 · Model-based quality filtering

The best-known open pretraining datasets of 2024 got most of their gains from one step: a small classifier that scores every web page and keeps the top few percent. This lesson builds such a classifier from LLM-written labels, and then does what the classifier's own accuracy cannot do: decides whether filtering by it produces better training data, with a matched training ablation at equal tokens, seeds and held-out sets that the filter's target does not favour.

## Why this matters at a frontier lab

A filter decides most of what a model reads. It is also easy to evaluate badly. Its validation F1 says how well it imitates its labeller, not whether the kept pages train a better model. Its effect looks largest on evaluations that resemble its target (a filter for "educational" pages wins on held-out educational pages by construction). And its cost moves with the training horizon: a filter that keeps 10% of the web is excellent for a 1T-token run and may be the wrong choice for a 15T-token run that would have to repeat the same pages many times. Getting any of these wrong locks a weaker recipe into every run that follows.

## The idea

### Three published designs

**FineWeb-Edu** (FineWeb paper, section 4; PUBLICLY DOCUMENTED). Llama-3-70B-Instruct scored "460,000 randomly sampled webpages" from one Common Crawl snapshot "on a scale from 0 to 5" for educational value. The classifier is "a linear regression model on top of the Snowflake-arctic-embed-m embedding model", fine-tuned on 410,000 of the annotations with the embedding layers frozen. The authors "chose a minimum threshold of 3"; the classifier "achieved an F1 score of 82%" on its validation set (F1 of the binary decision at 3). Filtering FineWeb gives FineWeb-Edu, "a 1.3-trillion token collection" (from FineWeb's 15T). On MMLU the score "increases from 33% to 37%", and FineWeb-Edu "can match the final performance of Matrix with almost 10x fewer tokens".

**DCLM** (section 4.4, Table 4; PUBLICLY DOCUMENTED). A fastText classifier over word uni- and bigrams, with positives from OpenHermes 2.5 and the r/ExplainLikeImFive subreddit and negatives "randomly sampled" from the RefinedWeb reproduction, "∼400k documents split equally". Keeping "the top-10% of examples" beat 15% and 20%. The filter was chosen by training: at the 1B-1x scale, models trained on each filter's output scored 26.1 (PageRank), 28.6 (AskLLM), 29.0 (perplexity filtering) and 30.2 (the fastText OH-2.5 + ELI5 filter) on the CORE average. DCLM-Baseline then trained a 7B model to 63.7% MMLU on 2.6T tokens.

**Nemotron-CC** (sections 2.1–2.2; PUBLICLY DOCUMENTED). Three classifiers "each of which has different high-quality preferences": one trained on Mistral 8x22B-instruct annotations, one on Nemotron-340B-instruct annotations, and the DCLM fastText classifier, combined by taking the **maximum** score, then split into 20 buckets of about 5% of documents each. The paper's motivation is the cost of aggressive filtering: FineWeb-Edu and DCLM "remove around 90% of the data", and both "contain around 80% near-duplicates (1T and 0.2T unique tokens, respectively)", so training on them "for many trillions of tokens implies seeing essentially the same samples many times", with diminishing returns after about four epochs (Muennighoff et al.). The results: at 1T tokens an 8B model on a high-quality subset improves MMLU by 5.6 over DCLM; the full 6.3T-token dataset (4.4T unique real tokens plus 1.9T synthetic) "matches DCLM on MMLU" with "four times more unique real tokens"; and an 8B model trained for 15T tokens, 7.2T of them from Nemotron-CC, beats Llama 3.1 8B by 5 MMLU points.

Read the last result carefully. It compares with Llama 3.1 8B, a different recipe, not with a DCLM-only run at 15T. That aggressive filtering loses at long horizons is the paper's argument from unique-token counts plus the repetition result, not an ablation at 15T (the 15T comparison: company claim; the mechanism: INFERENCE, plausible).

### The classifier, written out

All three classifiers are cheap scorers trained to imitate an expensive judgement. The course version is the fastText-shaped one, in pure PyTorch. Let $G(d)$ be the set of hashed word uni- and bigrams of document $d$, each mapped to one of $K = 2^{18}$ buckets. With one weight $w_g$ per bucket and a bias $b$,

$$\hat s(d) = b + \frac{1}{|G(d)|} \sum_{g \in G(d)} w_g,$$

which is exactly `torch.nn.EmbeddingBag(K, 1, mode="mean")` plus a bias. Trained as **regression** on the annotator's score $s \in \{0, \dots, 5\}$ (FineWeb-Edu's target), the loss is $\frac{1}{N}\sum (\hat s(d) - s(d))^2$; as **binary** logistic regression on "positive source vs random web" it is DCLM's setup.

### Imitation accuracy is not data quality

The classifier's metrics answer one question: does it rank pages the way the labeller would? Three facts limit what that says:

1. **Thresholds on a regressor shrink toward the mean.** A regressor trained with squared error predicts values pulled toward the average label. If only 9% of pages score 3 or more, a weak regressor may almost never predict 3, so "F1 at 3" collapses even when the ranking is good. Rank correlation (Spearman) and F1 at a matched positive rate measure the ranking; the filter keeps a *fraction*, so ranking is what matters.
2. **The labeller defines the target.** An "educational" labeller prefers textbook-like pages; a filter for it will lower held-out loss on textbook-like text whether or not the model gets better at anything else.
3. **Quantity is part of quality.** At a fixed budget $D$ tokens and a pool of $U$ unique tokens, keeping the top fraction $q$ forces $D / (qU)$ epochs. Halving $q$ doubles the repetition.

So a filter is evaluated by training on its output and comparing, at equal tokens, against training on the unfiltered pool, with seeds, on held-out sets that are not the filter's own target. That is how DCLM chose its filter (Table 4), and it is the lab.

## Worked example

### The score by hand

Buckets: a document has the words "cell divides cell", so $G = \{h(\text{cell}), h(\text{divides}), h(\text{cell divides}), h(\text{divides cell})\}$ (a set: "cell" counts once). With weights $0.8, 0.4, 1.2, 0.0$ and bias $1.1$: $\hat s = 1.1 + (0.8 + 0.4 + 1.2 + 0.0)/4 = 1.7$.

### F1 at a threshold

Five validation pages with true scores $(4, 3, 5, 1, 0)$ and predictions $(3.5, 2.0, 4.0, 3.1, 1.0)$. At 3: predicted positive $\{1, 3, 4\}$ (pages 1, 3 and 4 by position: 3.5, 4.0, 3.1), true positive $\{1, 2, 3\}$ (4, 3, 5). True positives 2, precision $2/3$, recall $2/3$, F1 $= 0.667$.

### Repetition forced by a threshold

Pool $U = 6.8$M tokens, budget $D = 1.02$M tokens (the lab's CPU numbers, rounded). Keeping 30%: $1.02 / (0.3 \cdot 6.8) = 0.5$ epochs. Keeping 10%: 1.5 epochs. Keeping 3%: 5.0 epochs, past the four-epoch point. The same arithmetic at frontier scale: a 15T-token run on a 1.3T-token filtered dataset is about 11.5 epochs if nothing else is added (INFERENCE from the stated sizes; real runs mix many sources).

## Shapes and cost

| Object | Shape, dtype, device | Notes |
|---|---|---|
| features of one document | (unique buckets,) int64, CPU | about 1,000 for a 600-word page |
| classifier weights | (2^18, 1) float32 + scalar bias | 1 MiB |
| a training batch | flat ids (Σ lengths,) int64 + offsets (batch,) int64 | `EmbeddingBag` input format |
| filtered source | Data-v0 layout, written by `make_source_subset` | records the selection and its SHA-256 |

Cost of scoring. The course classifier is dominated by Python word hashing: measured below, features for 23,505 annotated pages in about 20 s and the fit in about 35 s on CPU. At frontier scale the scorer runs once over every candidate page, so its cost per token matters: an embedding model of about $N_c = 10^8$ parameters costs about $2 N_c = 2 \times 10^8$ FLOPs per token in a forward pass, $3 \times 10^{21}$ FLOPs for 15T tokens, against $6 \cdot 8 \times 10^9 \cdot 15 \times 10^{12} = 7.2 \times 10^{23}$ for training an 8B model on them (arithmetic from the stated sizes, not a reported figure). A fastText-style bag model is orders of magnitude cheaper per token than either, which is one reason DCLM used it.

## Build it

```python
from frontierlab.datax import quality
from frontierlab.datax.mixture import make_source_subset

feats = [quality.features(t) for t in texts]                    # hashed uni+bigrams
clf = quality.fit(None, scores, mode="regression", feats=feats)  # EmbeddingBag(2**18, 1, "mean") + bias, AdamW
pred = quality.predict(clf, feats=pool_feats)
keep = quality.top_fraction(pred, 0.10)                          # indices of the top 10%
make_source_subset("labs/common/data/m10/web", "runs/m10/l102/src-top10", keep, note="classifier top 10%")
```

The subset is an ordinary source folder (Data-v0 layout, provenance of the parent plus the selection), so every arm of the ablation goes through the same mixture sampler, wrapper and run card. Correctness checks in `labs/common/tests/test_datax.py`: the classifier recovers a planted signal (Spearman above 0.8 on held-out synthetic pages), `spearman` equals SciPy's, `top_fraction` breaks ties deterministically; the lab test checks your score function against the `EmbeddingBag` forward.

## What the evidence says

- **Model-based quality filtering of web text: ESTABLISHED.** Independent groups report gains from it (FineWeb-Edu, DCLM, Nemotron-CC and others), each with training ablations of its own design (PUBLICLY DOCUMENTED, sections cited above).
- **Which labeller and which threshold: MODEL-SPECIFIC.** Each report chose its labels and threshold by its own evaluations; the choices do not transfer automatically to another tokenizer, model size or benchmark set.
- **Aggressive filtering hurts at long horizons: PROMISING.** Nemotron-CC's argument (unique tokens, repetition) is sound and its 15T result is strong, but the 15T comparison is against a different model's recipe, not a controlled ablation (company claim for the comparison; INFERENCE for the mechanism).
- **Ensembles of classifiers with different preferences: PROMISING** (Nemotron-CC section 2.2, one lab).
- **Course measurement (free CPU, 2026-10-04, other jobs running):** classifier Spearman 0.59 with the Llama-3 labels on 4,713 held-out annotations, F1 0.02 at the fixed threshold 3 and 0.50 at a matched positive rate; at 1.02M training tokens, keeping the top 30% of an 8,000-document web pool was inconclusive on every held-out set (Wikipedia −0.037, 95% CI [−0.124, +0.050]); keeping 10% (1.2 epochs) or 3% (4.3 epochs) made web held-out loss and LAMBADA clearly worse, and 3% also Wikipedia (+0.091 [+0.000, +0.182]). A course-scale null for moderate filtering and a measured cost for aggressive filtering; neither is evidence about 1B+ models.

## Lab

**Folder:** [`labs/module-10/lesson-02/`](../../labs/module-10/) · **Time:** about 2 hours (about 70 minutes of it unattended) · **Pass check:** `pytest labs/module-10/lesson-02` passes; `filter_ablation.py` prints the classifier metrics and the seed-level table; your write-up applies the decision rule to each arm and explains the repetition column.

### Experiment contract

- **Question:** does keeping the classifier's top 30%, 10% or 3% of a web pool lower held-out loss on text that is not the filter's target, at equal training tokens? Decision informed: whether and how hard Data-v1 filters its web source.
- **Hypothesis:** moderate filtering (30%) helps on Wikipedia held-out loss and LAMBADA; 3% repeats its data 5 times and helps less or hurts; all filtered arms win on FineWeb-Edu held-out loss (the target) and lose on unfiltered web held-out loss (the pool's own distribution). Status: reported effects at 1B+ scale (DCLM Table 4, Nemotron-CC); may not appear at 1.8M parameters.
- **Baseline:** `unfiltered`, the same first 8,000 web documents without filtering. Learning rate 3e-3 (course default for the toy preset), not tuned per arm; equal tuning budget (none) for all arms.
- **Changed variable:** which documents of the pool are kept. **Controlled:** pool, tokenizer, toy preset, 500 steps of 16 × 128 tokens (1.02M tokens), warmup 50, cosine, seeds 0–2 (same initial weights per seed across arms; data order differs by arm because the kept documents differ), evaluation sets and windows.
- **Comparison axis:** equal training tokens. It does not answer the cost of scoring the pool, which the arms do not pay in training FLOPs; state it separately.
- **Budget:** free CPU, 12 training runs of about 4–5 minutes plus scoring, about 70 minutes (measured below).
- **Metrics and decision rule:** primary: Wikipedia validation loss (256 windows of 128 tokens), seed-level paired 95% interval of (arm − unfiltered). Adopt an arm if the upper bound is below 0; reject if the lower bound is above 0; else inconclusive. Secondary: Data-v0 validation (the target distribution), web validation (the pool distribution), LAMBADA log-probability on 1,000 passages; Holm across the three arms on the primary metric if you claim more than one.
- **Correctness checks:** `pytest labs/common/tests/test_datax.py -k quality`; every arm's `mixture_accounting.json` shows the planned tokens; the subset meta of each arm records its selection hash; the classifier's validation pages are not training pages of the classifier (split by SHA-1 bucket).
- **Fallback evidence:** none; a null or negative result is a valid result.
- **Limits:** one labeller, one pool, one model size, 1M tokens per run; the classifier is far weaker than FineWeb-Edu's (no embedding model); Wikipedia stands in for "downstream" and is not a capability benchmark.

### Variants

| Variant | Hardware | What you run |
|---|---|---|
| Main path | 1× H100. Not run in this build; part of the Module 10 pilot | prepare a 2M-document web pool (`sources prepare web --docs 2000000`, about 4 GB streamed, PROJECTED), then `filter_ablation.py --variant main` (pilot-30m, 6,000 steps × 64 × 1,024 = 393M tokens per run, 12 runs). PROJECTED: 12 runs × 3.93e8 tokens × 0.321 GFLOP/token (`flops_per_token(pilot_30m(32768), 1024)`) ÷ (989e12 FLOP/s × an assumed 0.3 MFU) = 1.4 GPU-hours, plus scoring the pool |
| Free GPU (Colab/Kaggle T4) | T4, fp32 | `filter_ablation.py --variant t4` with a 200,000-document pool |
| Free CPU | laptop; measured: classifier 77 s, 12 runs of about 260 s, scoring about 1 min per run (about 70 min in all) | the steps below |

### Steps

1. **Implement** the six TODOs in `lab.py`: hashed features, the linear score, F1 at a threshold, the top-fraction filter, the epochs a budget forces, the decision rule. Run `pytest labs/module-10/lesson-02`.
2. **Prepare** the annotations and the pool (lesson 10.1 prepared `web` and `wiki`): `python -m frontierlab.datax.sources prepare annot --docs 24000` (45 MB, 59 s in this build).
3. **Classifier only:** `python labs/module-10/lesson-02/filter_ablation.py --classifier-only`. Write down F1 at 3, F1 at the matched rate and Spearman, and explain the gap between the two F1 numbers in one sentence.
4. **Ablation** (unattended): `python labs/module-10/lesson-02/filter_ablation.py`.
5. **Write up:** the decision for each arm on the primary metric; the sign pattern on the three secondary metrics and what it says about evaluating a filter on its own target; the epochs column against the four-epoch point; and whether you would filter Data-v1's web source, and how hard.

<details>
<summary>Hint for TODO 2</summary>

`EmbeddingBag(mode="mean")` averages the rows of the weight matrix selected by the document's bucket ids: `bias + weights[feats].mean()`.

</details>

<details>
<summary>What the build's run gave (compare after your write-up)</summary>

Measured 2026-10-04/05 on the build laptop (torch 2.14.1 CPU). Classifier: features for 23,505 annotated pages in 22 s, fit (6 epochs) in 55 s. Training: 12 runs of 500 steps × 16 × 128 = 1.02M tokens, about 260 s each on a quiet machine (several ran longer with other jobs or a sleeping laptop), scoring about 1 minute per run (3 held-out sets of 256 windows and 1,000 LAMBADA passages).

Classifier on the 20% validation annotations (label histogram over all 23,505: 2,941 / 13,275 / 5,201 / 1,687 / 401 / 0 for scores 0–5; 8.9% at 3 or more): RMSE 0.668 (predicting the mean would give 0.841), Spearman 0.595, **F1 at score 3: 0.019**, F1 at the threshold that matches the true positive rate: 0.50. The regressor almost never predicts 3 or more: shrinkage, exactly the failure the lesson describes; ranking by it still works, and FineWeb-Edu's 82% F1 came from a much stronger embedding model. Pool: the first 8,000 FineWeb training documents (6.68M tokens); kept tokens 2.59M (top 30%), 0.85M (10%), 0.24M (3%); epochs at the 1.02M-token budget 0.15, 0.39, 1.20 and 4.31.

Per-seed mean held-out loss (seeds 0, 1, 2) and the seed-level difference from `unfiltered` (95% t-interval; LAMBADA is log-probability, higher is better):

| Arm | Wikipedia (primary) | FineWeb-Edu (target) | web (pool) | LAMBADA | decision (primary) |
|---|---|---|---|---|---|
| unfiltered | 6.524, 6.530, 6.513 | 6.389, 6.393, 6.357 | 6.333, 6.328, 6.302 | −17.02, −16.84, −16.84 | — |
| top 30% | −0.037 [−0.124, +0.050] | −0.037 [−0.161, +0.086] | +0.043 [−0.072, +0.157] | −0.17 [−0.71, +0.37] | inconclusive |
| top 10% | +0.005 [−0.133, +0.142] | −0.003 [−0.161, +0.155] | +0.166 [+0.006, +0.326] | −0.58 [−0.92, −0.24] | inconclusive |
| top 3% | +0.091 [+0.000, +0.182] | −0.013 [−0.064, +0.037] | +0.266 [+0.214, +0.318] | −0.84 [−1.15, −0.54] | reject |

What it shows. (1) The seed std of the unfiltered arm is small (Wikipedia 0.008, FineWeb-Edu 0.020), but the paired differences vary much more: the arms train on different documents in a different order, so pairing by seed shares only the initial weights; with 3 seeds the intervals are about ±0.1 and the top-30% result is a null, not a finding (its seed-0 run alone looks like a 0.08 gain). (2) The pattern of the hypothesis holds where the effect is large: the harder the filter, the worse the model is on the distribution it removed (web) and on LAMBADA, and at 4.3 epochs it is also worse on Wikipedia: aggressive filtering lost at this budget, the course-scale shape of Nemotron-CC's argument (INFERENCE: the cause could be repetition or the narrower distribution; the design does not separate them). (3) No arm gained on the classifier's own target (FineWeb-Edu) beyond noise: a weak classifier and a 1.02M-token budget. For Data-v1 the evidence supports no filter, or at most a mild one, on the web source; the project's example recipe keeps the unfiltered web pool.

</details>

<details>
<summary>Reference solution</summary>

`labs/module-10/lesson-02/solution.py`. Check it with `LAB_TARGET=solution pytest labs/module-10/lesson-02`.

</details>

## Common mistakes

- **Choosing the filter by classifier accuracy.** It measures imitation of the labeller; train on the output and compare.
- **Evaluating only on the target distribution.** A filter for educational pages wins on educational held-out text by construction; add held-out sets it does not favour.
- **Comparing filtered and unfiltered arms at "one epoch each".** The arms then see different token counts; fix the tokens and report the repetition each filter forces.
- **Thresholding a regressor at the label's cut-off.** Shrinkage makes the threshold meaningless; keep a fraction by rank, or calibrate the threshold on held-out labels.
- **Training the classifier on documents that are also evaluation documents.** Check the classifier's training pages against the held-out sets with the 10.1 tools.
- **Forgetting the labeller's licence terms.** The labels of the annotation dataset were written by Llama-3-70B-Instruct; record that in the provenance and check the model licence's conditions on outputs before releasing a model trained with them.

## References

- G. Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024, section 4. https://arxiv.org/abs/2406.17557
- J. Li et al., *DataComp-LM: In search of the next generation of training sets for language models*, 2024, section 4.4, Table 4, Table 8. https://arxiv.org/abs/2406.11794
- D. Su et al., *Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset*, 2024, abstract, sections 2.1–2.2, 3.2. https://arxiv.org/abs/2412.02595
- N. Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023. https://arxiv.org/abs/2305.16264
- Annotation dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations (revision `72df4c92fb1b48beceb16016e8f695ec40a6c3a5`).
- Software versions used in this lab: [references/versions.md](../../references/versions.md).

## Next

[10.3 · Synthetic and rephrased data](lesson-03.md)
