# Task 6 — EDA Agent

Implements **Task 6** from `.claude/EDA_CLAUDE_TASKS.md`: runs the full
QC/EDA suite on the human-approved labeled matrix and writes plots,
tabular summaries, per-plot written interpretations, and a JSON audit log
for Task 7 (Reporting) to consume.

Task 6 is where the pipeline first *combines* Task 4's normalized matrix
with Task 5's reviewed labels. The CLI refuses to read Task 5's raw
`label_proposals.csv` — the filename must contain `.reviewed.` (the
convention Task 5's README instructs the reviewer to use), or the caller
must pass `--allow-unreviewed-labels` explicitly. That guarantees a
forgotten review step trips this CLI instead of silently poisoning
everything downstream.

## What it does

For one dataset (`archs4`, `recount3_GTEX_HEART`, etc.):

1. **Load** — read the Task 4 normalized matrix + sample metadata, join
   against the reviewed labels, drop unlabeled samples.
2. **Cohort composition** — per-label counts, per-series counts,
   demographics (sex/race/age/tissue) if the sample metadata carries them.
3. **Per-sample QC** — per-sample library size, genes-detected-per-sample,
   and optional biotype composition (needs `--gene-biotype-tsv`).
4. **Sample relationships** — PCA on the top-variance genes, sample-sample
   correlation heatmap with hierarchical-linkage ordering, and optional
   t-SNE on the PCA scores.
5. **Confounder screen** — top-K PCs vs. each covariate (`series_id`,
   `label`, `sex`, `tissue`, …). Categorical → eta²; continuous → r².
   Anything ≥ `--confounder-flag-threshold` gets flagged in the log.
6. **LLM interpretation** — 2-4 sentences per plot from Claude Opus 4.8,
   grounded on the underlying summary stats (not the image — the model
   is text-only here). Skippable with `--disable-llm-interpretation`.

Every step returns a dataclass report; `EDALog` accumulates them into
`eda_run_log_{dataset}.json` alongside the plot inventory and
interpretations.

## Inputs

From Task 4:

* `cvd_matrix_{dataset}_normalized.parquet` — genes × samples, canonical
  Ensembl gene ID as the row index, `sample_id` as columns.
* `cvd_sample_meta_{dataset}.parquet` — sample-indexed metadata.

From Task 5, *after human review*:

* `label_proposals_{dataset}.reviewed.csv` — must contain at least
  `sample_id`, `proposed_label`, `confidence`. Any extra columns (evidence
  quote, reviewer notes) are propagated to the audit log unchanged.

Optional:

* `--gene-biotype-tsv <path>` — TSV/CSV with `ensembl_id` and `biotype`
  columns. Enables the biotype-composition QC. Version suffixes on
  `ensembl_id` are stripped so the mapping matches Task 4's canonical
  (versionless) IDs.

## Outputs

Written under `{output_dir}/{dataset}/`:

* `eda_plots/` — one PNG per plot (see below).
* `eda_summary_stats_{dataset}.csv` — tall-format
  `(metric, key, value)` triples. Cohort counts, QC five-number summaries,
  per-PC variance ratios, flagged confounders. Easy to append to.
* `per_sample_qc_{dataset}.csv` — library size + genes-detected + label,
  one row per sample.
* `pca_scores_{dataset}.csv` — top-K PCA scores per sample.
* `eda_interpretations_{dataset}.json` — LLM interpretation text keyed by
  plot name.
* `eda_run_log_{dataset}.json` — structured audit trail. Fields:
  `config`, `inputs`, `outputs`, `steps` (per-step reports),
  `plots` (name → path), `interpretations`, `flagged_confounders`,
  `environment`, `warnings`, `errors`. Task 7 reads this file directly.

### Plot inventory

| File | What it shows |
|---|---|
| `cohort_labels.png` | Bar of samples per reviewed label. |
| `cohort_series.png` | Bar of samples per source series (top 20). |
| `qc_library_size.png` | Histogram of per-sample summed expression. |
| `qc_genes_detected.png` | Histogram of genes detected per sample. |
| `qc_biotype_share.png` *(optional)* | Boxplot of per-sample biotype share (top 10). |
| `pca_by_{covariate}.png` | PC1 vs. PC2 colored by label / series_id / sex / tissue. |
| `tsne_by_{covariate}.png` | t-SNE (on PCA scores) colored the same way. |
| `sample_correlation.png` | Sample-sample Pearson-r heatmap, hierarchical order. |
| `confounder_screen.png` | Top PCs × covariates association (η² / r²). |

## Install

Everything except `anthropic` is already in the base install
(`pandas`, `numpy`, `scikit-learn==1.2.2`, `scipy`, `matplotlib`,
`pyarrow`). If starting from a fresh venv:

```bash
source .venv/bin/activate
pip install pyarrow matplotlib scipy         # only pyarrow was newly needed on this box
uv pip install anthropic                     # only if you want LLM interpretation
```

## Credentials

* `ANTHROPIC_API_KEY` — required for LLM interpretation. Rerun with
  `--disable-llm-interpretation` to skip the step; plots and stats CSVs
  are still written and the audit log just has empty interpretation
  entries.

## Run

Assuming Task 4 has emitted `cvd_matrix_archs4_normalized.parquet` and
Task 5's proposals have been human-reviewed to
`label_proposals_archs4.reviewed.csv`:

```bash
python -m cvd_eda.eda.run \
    --dataset      archs4 \
    --matrix       cvd_eda/logs/task4_out/cvd_matrix_archs4_normalized.parquet \
    --sample-meta  cvd_eda/logs/task4_out/cvd_sample_meta_archs4.parquet \
    --labels       cvd_eda/logs/label_proposals_archs4.reviewed.csv \
    --output-dir   cvd_eda/logs/task6_out/ \
    --llm-cache    cvd_eda/logs/eda_llm_cache/
```

Offline / no-LLM run (still writes plots + stats):

```bash
python -m cvd_eda.eda.run ... --disable-llm-interpretation
```

Offline smoke test — no real data, no API calls, exits 0 if all 39 checks
pass:

```bash
python -m cvd_eda.eda.smoke_test
```

The smoke test fabricates a matrix with a *planted* batch effect (a
series-driven +/-1.0 shift on 60 genes) so the confounder screen has a
correct answer to verify against — the run fails if `series_id` doesn't
show up in `flagged_confounders`.

## CLI overrides

Every field on `EDAConfig` has a matching flag; the important ones:

| Flag | Default | Purpose |
|---|---|---|
| `--top-variable-genes` | 5000 | Cap on genes fed to PCA / t-SNE / corr. |
| `--n-pca-components` | 10 | Also caps t-SNE input dimensionality. |
| `--no-tsne` | off | Skip t-SNE (large N: t-SNE is the slow step). |
| `--tsne-perplexity` | 30 | sklearn default; must be < n_samples. |
| `--confounder-flag-threshold` | 0.30 | eta²/r² threshold for flagging. |
| `--top-pcs-for-confounder-screen` | 5 | How many PCs to screen against. |
| `--heatmap-sample-cap` | 200 | Downsample the corr heatmap above this. |
| `--min-label-confidence` | 0.0 | Additional confidence gate on reviewed labels. |
| `--allow-unreviewed-labels` | off | Bypass the `.reviewed.` filename check. |
| `--disable-llm-interpretation` | off | Skip the Anthropic calls. |
| `--model` | `claude-opus-4-8` | Same default as Task 5. |

## Design decisions

### Review-file gate

Task 5's contract is that its output requires human review. Making Task 6
refuse to read a filename that doesn't contain `.reviewed.` turns a
policy into a runtime error — a forgotten review step is caught the
moment someone runs the CLI. The escape hatch (`--allow-unreviewed-labels`)
is intentionally verbose so it never gets typed by accident.

### Confounder screen: eta² and r², not p-values

We want *effect sizes*, not significance. On a cohort where every
covariate is significantly non-zero (any real dataset with N > 50), a
p-value can't distinguish "PC1 mostly reflects series_id" from "PC1
weakly correlates with series_id". eta² / r² is bounded in [0, 1] and
directly comparable across covariates. The flagging threshold defaults
to 0.30 — low enough to catch shape-of-the-effect problems, high enough
that half the covariates don't get flagged on any dataset.

### Top-variable-gene reduction feeds PCA, t-SNE, *and* the correlation heatmap

All three are downstream of the same 5000-gene reduction so the plots
tell a consistent story: what dominates PC1 is what drove the corr
heatmap ordering. Using the full matrix on any one of them and the
reduced matrix on the others is a subtle way to make plots disagree.

### LLM sees numbers, not the image

Each interpretation call receives a JSON blob of summary statistics for
the plot, not an image. That's cheaper, deterministic across models
(no vision-specific behavior), and — importantly — the LLM's job here is
to state what the numbers imply for a downstream decision, not to
describe what the plot *looks like*. Grounding on numbers keeps the
interpretation directly checkable against the CSV.

### Sample-sample correlation heatmap uses hierarchical linkage order

Row/column order matters for a heatmap. Sorted by sample ID gives a
uninformative diagonal-only picture; hierarchical linkage on
`1 - correlation` distance reveals block structure. We report the leaf
order in the audit log so Task 7 (or a human) can reproduce it.

### Biotype composition is optional

Task 4's matrix carries only canonical Ensembl IDs. Rather than force a
`mygene` lookup at runtime (network-dependent, slow, non-deterministic),
we take a `--gene-biotype-tsv` on the CLI. Skip the flag → skip the plot
+ QC row; the log records that it was skipped and why.

### No dataset merging in this task

Each dataset produces its own subdir under `output_dir`. Merging ARCHS4
and RECOUNT3 is deliberately *not* done here: their alignment differences
are EDA signal, and any merge decision should be made by a human reading
the two run logs side by side (Task 7 territory).

## Module layout

```
cvd_eda/eda/
├── __init__.py            # public re-exports
├── config.py              # EDAConfig dataclass (all thresholds)
├── loaders.py             # Task 4 parquet + Task 5 reviewed labels → LabeledDataset
├── cohort.py              # cohort composition summary
├── qc.py                  # per-sample library size, genes detected, biotype share
├── relationships.py       # PCA, sample-sample correlation, t-SNE
├── confounders.py         # top-PC vs. covariate association screen
├── plotting.py            # matplotlib/seaborn PNG generation
├── interpret.py           # Anthropic-backed per-plot interpretation
├── logging_utils.py       # EDALog JSON accumulator
├── run.py                 # argparse CLI (this is the entrypoint)
├── smoke_test.py          # offline pipeline smoke test (39 checks)
└── README.md              # this file
```

## Non-goals

* No batch correction (ComBat / harmony). The confounder screen is
  intended to *flag* the batch problem so the reviewer can decide
  whether correction is warranted before the elastic-net stage.
* No differential expression / GSEA. Those live downstream of the
  elastic-net feature-selection stage the reporting agent gates.
* No cross-dataset merging (see design note above).
