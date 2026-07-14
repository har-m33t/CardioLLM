# Whole-Corpus Elastic Net CVD Classifier — Write-up

**Status:** completed real run against the full ARCHS4 human corpus
(`human_gene_v2.latest.h5`), 2026-07-13. All numeric fields below are from
that run (`eda/dataset/cvd_data/elasticnet_out/`). The design decisions
carry through to any future re-run; the tractability-driven scale-down of
the model stage is documented explicitly in its own section.

---

## Scope and design choice

No curated CVD cohort was built for this stage. Instead, a broad CVD
keyword net is applied directly to per-sample metadata (`title`,
`source_name_ch1`, `characteristics_ch1`) across the *entire* single-cell-
filtered ARCHS4 human corpus. Every sample gets a weak label:
`cvd_related = 1` if any keyword hits any of those fields, else 0. The
elastic net then does double duty — finding which genes predict CVD-
relatedness *and* implicitly distinguishing CVD samples from every other
tissue/condition/cell-line represented in ARCHS4.

**Why this rather than a curated cohort.** Building a clean case/control
cohort would require manual per-study review of hundreds of GEO series,
which is beyond the scope of this pass. The keyword-only approach is
deliberately simple: it exposes what a weak-supervision elastic net can
pull out with no biology added in, which is a useful lower bound on the
signal recoverable from the transcriptome alone.

## Weak-label composition

- **Keyword list (16 terms, case-insensitive substring):** cardiovasc,
  cardiac, heart failure, myocardial infarct, coronary artery,
  atherosclerosis, cardiomyopathy, arrhythmia, atrial fibrillation,
  hypertension, ischemic heart, aortic, vascular disease, congestive
  heart, cardiac hypertrophy, cardiac fibrosis.
- **Fields searched:** `title`, `source_name_ch1`, `characteristics_ch1`.
- **Whole-corpus size:** 1,098,771 human samples.
- **Positive count / base rate (whole corpus):** 10,557 / 0.96%.
- **Bulk pool (single-cell samples excluded, threshold 0.5):** 779,009
  samples (29.1% of the corpus excluded as single-cell). Positives
  surviving into the bulk pool: 8,725 (base rate 1.12%); bulk negatives
  available: 770,284.
- **Per-field contribution to positives (whole corpus):** title=1,011,
  source_name_ch1=4,598, characteristics_ch1=8,641 (a sample can match on
  more than one field).

## Class imbalance handling

Two mechanisms combined:
1. Keep all bulk-pool positives; subsample negatives at **3:1** neg:pos
   with a fixed seed. Actual training pool: **34,900 samples** = 8,725
   positives + 26,175 negatives. (The originally-sketched 10:1 ratio was
   reduced to 3:1 for compute tractability — 3:1 already keeps the
   negative class dominant while holding the pool small enough to
   materialise and fit; documented, not a hidden default.)
2. `class_weight="balanced"` inside `LogisticRegressionCV` — belt-and-
   suspenders, because the weak label is noisy and the residual 3:1
   imbalance still biases loss.

The training population is described as "CVD-matched samples plus a
random 3:1 subsample of the bulk-corpus rest," *not* "the corpus" —
this distinction is called out again in the metrics section.

## Cross-validation strategy

`StratifiedGroupKFold` with `groups = source_series_id` and stratification
on the CVD label. Same-series leakage is the single most dangerous
shortcut here: with a whole-corpus weak label, a model can score high
by memorising study-specific batch signatures instead of learning
biology. Grouping folds by series kills that shortcut. Verified: 12,597
unique series spread across 5 outer folds with `series_leakage_check`
passing (every series in exactly one outer fold; ~2,500 series/fold).

Inner CV for hyperparameter selection is also grouped-by-series. This
is done by precomputing grouped inner splits and passing them as `cv=`
to `LogisticRegressionCV`, since sklearn's `LogisticRegressionCV.fit`
does not accept a `groups=` argument.

## Model

`sklearn.pipeline.Pipeline` of `StandardScaler` → `LogisticRegressionCV`
with:
- penalty = `elasticnet`, solver = `saga`
- l1_ratios = [0.1, 0.5, 0.9]
- Cs = 3 (log-spaced by sklearn default)
- class_weight = balanced, max_iter = 200, tol = 1e-2
- scoring = average_precision (PR-AUC — matches the primary reporting metric)
- refit = True on best inner-CV hyperparameters
- inner CV = 3 grouped folds

Standardisation is fit on the training fold only inside the Pipeline —
no leakage. Per-fold hyperparameter picks (all folds converged):

| Fold | chosen C | chosen l1_ratio | nonzero β | wall |
|------|----------|-----------------|-----------|------|
| 0    | 1.0      | 0.9             | 1500      | 66s  |
| 1    | 1.0      | 0.9             | 1500      | 56s  |
| 2    | 1.0      | 0.9             | 1500      | 52s  |
| 3    | 1.0      | 0.1             | 1500      | 57s  |
| 4    | 1.0      | 0.5             | 1500      | 57s  |

All folds selected C = 1.0, so the elastic net applied no additional
sparsity *within* the per-fold feature set — every one of the 1,500
selected genes kept a nonzero coefficient. Sparsity in this run therefore
comes from the variance pre-selection (below), not from L1 shrinkage.

## Scale-down for tractability (model stage only)

The `label → subsample → load_expression → splits` stages run against the
full corpus unchanged. The `fit` stage, however, is intractable at full
scale on a single 24 GB machine: saga elastic-net over the full
34,900 × 49,231 dense matrix does not converge in a bounded iteration
budget (a single fit did not converge at max_iter=200 and took ~77s), and
the nested-CV grid runs ~28 fits per outer fold. Three changes make it
finish reliably (~4.5 min end-to-end for all 5 folds), validated against a
held-out fold before the full run (single-fit ROC-AUC 0.945, PR-AUC 0.876):

1. **Per-fold variance feature reduction, 49,231 → 1,500 genes.** The
   top-1,500 most-variable genes are selected *on each training fold only*
   (leakage-safe — variance is never measured on the held-out rows). The
   winning coefficients are mapped back into the full 49,231-gene space
   (zeros elsewhere) so downstream aggregation still sees full-width
   vectors. Configurable via `--top-k-genes` (0 disables).
2. **Looser convergence tolerance, `tol = 1e-2`** (sklearn default 1e-4).
   This was the decisive lever: saga now converges in ~50 iterations
   instead of hitting the cap, cutting a single fit from ~77s to ~8s with
   negligible metric change for a weak-label screening model.
3. **Memory-mapped matrix + reduced grid** (Cs 10→3, l1_ratios 5→3, inner
   folds 5→3). mmap keeps peak RSS bounded to a single fold's slice
   (~11 GB) instead of doubling the full 6.9 GB matrix into RAM.

This is a deliberate, documented trade of exhaustive regularization-path
search for a run that completes; the held-out metrics show the reduced
model still recovers strong CVD signal.

## Performance (mean ± std across 5 outer folds)

Metrics on the matched 3:1 training pool, threshold 0.5. **Real-world
deployment on the true unbalanced corpus (~1% base rate) would need
separate calibration.**

| Metric        | Value          | Notes                                     |
|---------------|----------------|-------------------------------------------|
| PR-AUC        | 0.873 ± 0.020  | **primary metric** — imbalance-robust     |
| ROC-AUC       | 0.943 ± 0.008  | reported for comparability                |
| Accuracy      | 0.910 ± 0.026  | do not lead with — imbalance-sensitive    |
| Sensitivity   | 0.791 ± 0.063  | recall on CVD-matched                     |
| Specificity   | 0.950 ± 0.007  | recall on non-CVD                         |
| F1            | 0.813 ± 0.040  | @ threshold 0.5                           |

See `plots/pr_curve.png` for the primary read of performance and
`plots/roc_curve.png` for the standard companion. `plots/calibration_curve.png`
shows how well raw scores map to true positive rates.

## Gene signal ranking + ClinGen HCVD cross-check

Coefficients aggregated across outer folds: mean, median, and fraction
of folds where the gene had nonzero β. Full ranked list is
`gene_signal/gene_signal_ranking.csv` (n = 49,231 genes; each fold
contributes nonzero β only for its variance-selected 1,500). Top-30 with
ClinGen HCVD membership highlighted in `plots/top_genes_coefficients.png`.

**Top signal.** The largest |mean β| genes are FCGR2A, GFRA1, BGN, **TTN**,
ROBO1, **GATA4**, CHCHD2, RPL23AP42, NEAT1, HSPG2. TTN (titin,
cardiomyopathy) and GATA4 (cardiac transcription factor) are both known
ClinGen HCVD genes appearing in the top 6.

**ClinGen recovery.** The ClinGen HCVD starter set has 44 unique genes; 48
rows of the low-count-filtered feature matrix carry a ClinGen symbol (4
symbols — KCNQ1, MYH11, GATA4, CACNA1C — appear twice on the ARCHS4 gene
axis, so row-count exceeds unique-gene-count). Of these, the top 100 model
coefficients recover 5 (TTN, GATA4, ACTA2, MYH7 already inside the top 30).
Additional model-selected candidates beyond the ClinGen list dominate the
top ranks (FCGR2A, GFRA1, BGN, ...) — the "in_clingen_hcvd" column is
`False` for these.

The ClinGen list is applied strictly post-hoc, never as a pre-filter
(pre-filtering would make recovery circular).

## Limitations we're explicit about

- **Label noise is real.** A keyword match on `title`/`characteristics_ch1`
  will catch cell-line studies that mention cardiac gene names,
  family-history annotations, and off-topic uses of "cardiac" (e.g.
  "cardiac stress echo test") — this is the accepted tradeoff of
  skipping cohort curation. Manual spot-checks of the top false
  positives are out of scope for this pass.
- **The training population is not the corpus.** Metrics describe
  performance on a matched 3:1 pool. Applied to the true corpus with
  its actual base rate (~1%), precision would drop substantially.
  This is why PR-AUC leads over ROC-AUC in reporting.
- **The feature set is variance-preselected per fold.** The elastic net
  fits on the 1,500 most-variable genes of each training fold, not all
  49,231. Genes with low variance in the pool (many ClinGen HCVD genes
  among them) can never receive a nonzero coefficient, which caps ClinGen
  recovery. A full-feature run would need more compute than a single
  machine provides; see the scale-down section.
- **No additional L1 sparsity.** Every fold selected C = 1.0, so the model
  kept all 1,500 pre-selected genes. A finer/wider Cs grid could induce
  sparser, more interpretable signatures at higher compute cost.
- **No confounder control beyond series grouping.** Tissue, condition,
  and study year are all correlated with the CVD label but not
  balanced across folds beyond the group-CV constraint. Downstream
  work with a curated cohort would need those.
- **Low-count filter is pool-specific.** The 10%-detection gene mask
  is computed on the training pool, not the whole corpus — see
  `expression/load_manifest.json`. Rerunning against a different pool
  will produce a different feature set.

## Reproducibility

All seeds fixed and logged (`logs/run_manifest.json`). Full pipeline:

```
python -m elasticnet.train \
  --archs4-h5-path /path/to/human_gene_v2.latest.h5 \
  --outdir eda/dataset/cvd_data/elasticnet_out \
  --negative-ratio 3 --n-outer-folds 5 --n-inner-folds 3 \
  --max-iter 200 --tol 0.01 --top-k-genes 1500 --seed 20260707
```

To re-fit only (reusing existing label/subsample/expression/splits
artifacts), add `--only fit,evaluate,gene_signal,plots`.
