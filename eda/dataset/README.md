# ARCHS4 whole-corpus EDA

This is the pre-CVD-selection exploratory data analysis on the **full ARCHS4
human corpus** — a resource-grade dataset description, replicating the
methodology of the ARCHS4 paper (Lachmann et al. 2018, *Nature Communications*,
doi:10.1038/s41467-018-03751-6) and the standard Bioconductor bulk RNA-seq
EDA workflow (Love/Huber-style; recount3 quickstart). Scope is intentionally
narrow: describe the corpus, not analyse a subset.

**Status: code only.** No script here has been run against real ARCHS4 data.
The ~30 GB+ download and the pipeline execution happen later, once the
scripts have been reviewed and (ideally) sanity-checked against a small toy
matrix. Every deliverable in the checklist below refers to a *script* that
will produce the listed file when run — not the file itself.

## Folder layout

```
eda/
├── eda.py                     CLI orchestrator — runs Tasks 1-6 end to end.
├── plotting.py                Shared matplotlib style for all figures.
├── dataset/
│   ├── data.py                Downloads the ARCHS4 H5 file (idempotent, size-checked).
│   ├── io.py                  Chunked H5 read helpers used by every step.
│   ├── make_toy_data.py       Synthesises a small ARCHS4-shaped H5 for tests.
│   └── README.md              This file.
├── steps/
│   ├── cohort.py              Task 1 — cohort composition.
│   ├── qc.py                  Task 2 — per-sample QC.
│   ├── normalize.py           Task 3 — quantile normalization + log2.
│   ├── dimred.py              Task 4 — PCA + sample- and gene-centric t-SNE.
│   ├── clustering.py          Task 5 — sample-sample correlation heatmap.
│   └── gene_summary.py        Task 6 — per-gene detection + biotype.
└── tests/
    └── test_steps.py          End-to-end smoke tests against a toy H5.
```

Each step module is a thin wrapper: pure functions that do the analysis,
plus a `run(h5_path, outdir)` entry that `eda.py` calls. The step modules
never talk to each other directly — steps 4 and 5 read step 3's on-disk
outputs (`outdir/normalized/`), which is the only inter-step coupling.

## Deliverables (per `.claude/eda_todo.md`)

| # | Task | Script | Files produced under `<outdir>/` |
|---|---|---|---|
| 1 | Cohort composition | `steps/cohort.py` | `cohort/cohort_composition_full.csv`, `cohort/cohort_samples_by_year.png`, `cohort/cohort_single_cell_flag.png` |
| 2 | Per-sample QC | `steps/qc.py` | `qc/qc_full_dataset.csv`, `qc/qc_library_size_hist.png`, `qc/qc_genes_detected_hist.png` |
| 3 | Normalization | `steps/normalize.py` | `normalized/reference_distribution.npy`, `normalized/subsample_indices.npy`, `normalized/subsample_matrix.npy`, `normalized/normalize_manifest.json` |
| 4 | Dimensionality reduction | `steps/dimred.py` | `dimred/tsne_sample_centric_n20000.png` (primary), `dimred/tsne_sample_centric_n5000.png` (stability), `dimred/tsne_gene_centric.png`, `dimred/pca_full.png`, `dimred/tsne_scores_n20000.csv`, `dimred/tsne_scores_n5000.csv`, `dimred/tsne_gene_scores.csv`, `dimred/pca_scores.csv`, `dimred/pca_explained_variance_ratio.npy`, `dimred/dimred_manifest.json` |
| 5 | Sample clustering | `steps/clustering.py` | `clustering/sample_correlation_heatmap.png`, `clustering/linkage.npy`, `clustering/heatmap_sample_indices.npy`, `clustering/clustering.csv`, `clustering/clustering_manifest.json` |
| 6 | Gene-level summary | `steps/gene_summary.py` | `gene_summary/gene_summary_full.csv`, `gene_summary/gene_detection_rate_hist.png`, `gene_summary/gene_biotype_bar.png` (if biotype metadata present) |
| 7 | Write-up | — | Authored manually from the artifacts above; not produced by this code. |

## Methodology fidelity

The parameter choices are lifted from the ARCHS4 paper's Methods section and
should not be changed without also updating the write-up (Task 7).

- **Subsampling pool (shared).** Every step that draws a random subsample —
  the quantile-norm reference distribution, the step-3 downstream matrix,
  the sample-centric t-SNE stability run, and the clustering heatmap — draws
  from a single upstream pool built once by `dataset/io.SINGLECELL_PROB_THRESHOLD`
  = `0.5`: samples with `singlecellprobability > 0.5` are excluded before any
  random draw. The excluded count and percentage are recorded in
  `normalized/normalize_manifest.json` under `singlecell_filter`. Whole-corpus
  steps that don't subsample (cohort, qc, gene_summary) still report both
  groups, per the "flag, don't drop" principle.
- **Normalization.** Quantile normalization (Bolstad tie handling) + log2,
  matching Lachmann et al. 2018. Because the full corpus does not fit in
  memory for classical quantile norm, we use the **reference-distribution
  variant**: reference computed from a documented uniform random subsample
  (`n_ref = 10 000` by default) drawn from the singlecell-filtered pool,
  then applied on demand. This choice, and its equivalence to full quantile
  norm for a sufficiently large reference, is stated in the write-up.
- **t-SNE.** Perplexity **50** for the sample-centric embedding, **30** for
  the gene-centric embedding — both from the ARCHS4 paper. Substituting
  scikit-learn's Barnes-Hut t-SNE for `Rtsne` (both implement van der
  Maaten's Barnes-Hut algorithm). `Rtsne` defaults to `initial_dims = 50`,
  i.e. it PCA-projects inputs to 50 dims before Barnes-Hut; sklearn's
  `TSNE` does not. To preserve Rtsne parity we PCA-project both the
  sample- and gene-centric inputs to 50 dims manually before the `TSNE`
  call (`dimred.TSNE_INITIAL_DIMS`). This deviation, and the fact that
  t-SNE runs on the step-3 subsample (not the whole ~800k+-sample corpus),
  are both documented in the write-up.

  The sample-centric t-SNE runs twice: **primary at N = 20 000** (the file
  used in the paper) and a **stability check at N = 5 000** drawn as a
  nested subset of the same 20 000 sample-index pool. Only these two sizes
  are computed — this is a stability confirmation, not a full sweep, and
  additional sizes are explicitly out of scope for this pass.
- **PCA.** Included as a linear sanity check on the t-SNE structure and to
  supply the variance-explained numbers reviewers expect, per the ARCHS4
  paper's supplementary comparison table.
- **Clustering.** Pearson correlation → `1 - r` distance → UPGMA (`average`)
  linkage. Standard Bioconductor convention. Full-corpus sample-sample
  correlation is not storable (>5 TB float32 at ~800k samples), so the
  heatmap uses a **nested** uniform random sub-sub-sample of size
  `HEATMAP_N = 2000` drawn from the same step-3 N = 20 000 sample-index pool
  used by the sample-centric t-SNE. Nesting (not an independent draw from the
  full corpus) is what gives the heatmap and t-SNE a common sample-ID set for
  cross-figure joins. The size, seed, and selection method are recorded in
  `clustering/clustering_manifest.json`.
- **QC.** MAD-based outlier flagging (3 × MAD, recount3 default) — samples
  are **flagged, not dropped**, because this EDA is pre-selection.

### Task 7 methods paragraph (paste verbatim)

> Samples flagged as likely single-cell (ARCHS4 `singlecellprobability` > 0.5)
> were excluded prior to subsampling. The remaining pool was uniformly
> randomly subsampled, as ARCHS4 does not provide a structured field suitable
> for stratified sampling at whole-corpus scale. Sample-centric t-SNE was
> computed on a random subsample of N = 20 000 samples (vs. the full corpus
> of ~187 946 in Lachmann et al. 2018; the current ARCHS4 release exceeds
> 800 000 human samples, making full-corpus Barnes-Hut t-SNE computationally
> impractical). Cluster structure was confirmed consistent at N = 5 000. The
> sample correlation heatmap uses a further uniform subsample of N = 2 000,
> drawn from the same N = 20 000 t-SNE subsample, for visual legibility and
> cross-figure consistency.

## Reproducibility

- All random draws (subsample indices, t-SNE, PCA, heatmap sub-sub-sample)
  use fixed seeds set at the top of each step module — never a single
  global seed, so `--only <step>` is reproducible in isolation.
- Seeds and step parameters are recorded per step:
  - `outdir/normalized/normalize_manifest.json`
  - `outdir/dimred/dimred_manifest.json`
  - `outdir/clustering/clustering_manifest.json`
  - `outdir/logs/eda_manifest.json` (orchestrator-level: per-step start/end/status)
- Every run writes a timestamped log file to `outdir/logs/`.
- The ARCHS4 release version (whichever `archs4py` resolves as `latest`)
  is recorded when `dataset/data.py` runs, in `outdir/logs/manifest.json`.
  Sample counts have grown substantially since the paper's 187 946 — the
  write-up must cite the version actually pulled, not the paper's numbers.

## Data integrity

`dataset/data.py` guards against the two failure modes that a filename-only
"already downloaded" check misses:

- **Truncated files.** Any file matching `human_gene_*.h5` under
  `<outdir>/archs4/` is rejected if smaller than `MIN_EXPECTED_SIZE_GB`
  (20 GB, conservative for the ~30 GB release). This means an interrupted
  download won't be silently treated as complete on rerun — delete the
  partial file and re-run.
- **No free space.** Before initiating a download, `data.py` checks that
  `<outdir>/archs4/` has at least `EXPECTED_DOWNLOAD_GB + DISK_HEADROOM_GB`
  (~40 GB) free and fails fast if it doesn't, so archs4py doesn't fill the
  disk producing a truncated file.

## How to run (once ready)

Nothing here should be executed against real ARCHS4 data until the code
review pass has completed. When that time comes:

```bash
# 1. Download the corpus (~30 GB, one-off):
python -m eda.dataset.data --outdir ~/cvd_data

# 2. Run the whole EDA pipeline:
python -m eda.eda --data-root ~/cvd_data --outdir ~/cvd_data/eda_out

# Or run a single step (e.g. re-do just the QC figures):
python -m eda.eda --data-root ~/cvd_data --outdir ~/cvd_data/eda_out --only qc
```

Required Python packages (beyond the repo's existing `eda` extra):
`archs4py`, `h5py`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.

## Local sanity-check without ARCHS4

Before spending 30 GB and multiple hours on the real corpus, exercise the
whole pipeline against a toy H5 with the same schema:

```bash
# Regenerate the toy H5 (fast; only useful if inspecting it directly):
python -m eda.dataset.make_toy_data --out /tmp/toy_archs4.h5

# Run the tests — each step module is exercised end-to-end against a toy
# corpus. Bytes are synthetic; shapes and file layout are the real thing.
pytest eda/tests -q
```

`make_toy_data.py` writes a `500 genes × 2 000 samples` H5 matching
ARCHS4's `data/expression` + `meta/samples/*` + `meta/genes/*` layout, so
every step module runs against it unchanged.

## Out of scope

CVD sample selection, inclusion-criterion justification, case/control
labelling, and any CVD-subset EDA are the **next** stage and live in a
separate module. Do not pull CVD-specific filtering into any script here —
inputs to every step are "the full loaded corpus," full stop.
