# cvd_eda — ARCHS4 + RECOUNT3 CVD EDA pipeline

Multi-agent workflow for loading, curating, and running EDA over ARCHS4 and
RECOUNT3 gene expression data with a cardiovascular-disease focus. Task
breakdown lives in `.claude/EDA_CLAUDE_TASKS.md`; this directory holds the
scripts and outputs each task's agent produces.

This is a **separate deliverable from TinyLLaVA_Factory** — it happens to
live in the same repo but is excluded from the `tinyllava` wheel via
`pyproject.toml` and does not touch any TinyLLaVA code.

## Layout

```
cvd_eda/
├── README.md                        (this file — pipeline overview)
├── .gitignore                       (excludes data/ and logs/)
├── ingestion/
│   ├── README.md                    (Task 1 docs)
│   └── archs4_ingest.py             (Task 1 — ARCHS4 ingestion)
├── task2_recount3/                  (Task 2 — RECOUNT3 ingestion; see its README)
│   ├── R/                           (enumerate_projects.R, pull_and_export.R)
│   ├── python/orchestrate.py        (Rscript driver + log aggregator)
│   ├── config/candidate_projects.yaml
│   ├── run.sh                       (one-shot entrypoint)
│   ├── setup.sh                     (installs recount3 / arrow in R)
│   └── README.md
├── processing/                     (Task 4 — data processing & cleaning; see its README)
├── data/                            (created at runtime; not committed)
│   ├── human_gene_v2.5.h5           (~45 GB — do not commit)
│   ├── archs4_raw.h5 -> human_gene_v2.5.h5   (stable symlink)
│   └── recount3_raw/                (Task 2 output — one Parquet trio per project)
└── logs/                            (created at runtime; not committed)
    ├── ingestion_log_archs4.json    (Task 1 audit trail)
    └── ingestion_log_recount3.json  (Task 2 audit trail)
```

Everything under `data/` and `logs/` is `.gitignore`d — the H5 alone is 45 GB.
`data/` should point at scratch (e.g. `$SCRATCH/archs4/`) rather than the
repo's own filesystem on the HPC login node.

## Task status

| # | Task                              | Owner    | Status         |
|---|-----------------------------------|----------|----------------|
| 1 | Ingestion — ARCHS4                | Claude   | **Implemented** — see `ingestion/README.md` |
| 2 | Ingestion — RECOUNT3              | Claude   | **Implemented** — see `task2_recount3/README.md` |
| 3 | Metadata curation (CVD relevance) | Claude   | **Implemented** — see `curation/README.md` |
| 4 | Data processing & cleaning        | Claude   | **Implemented** — see `processing/README.md` |
| 5 | Labeling (⚠ human review gate)     | Claude   | **Implemented** — see `labeling/README.md` |
| 6 | EDA                               | —        | Not started    |
| 7 | Reporting                         | —        | Not started    |

## Running Task 1

See `ingestion/README.md` — briefly:

```bash
source .venv/bin/activate
uv pip install h5py archs4py requests           # if not already installed
export CVD_EDA_DATA_DIR=$SCRATCH/archs4          # 45 GB — needs a big volume
python -m cvd_eda.ingestion.archs4_ingest
```

Log lands at `cvd_eda/logs/ingestion_log_archs4.json`. Task 7 (reporting)
will read every `ingestion_log_*.json` / `processing_log_*.json` from that
directory, so keep the layout.

## Running Task 2

Requires R (>= 4.2) on `PATH` — on the HPC, `module load R/4.4.0` first.
Full details in `task2_recount3/README.md`. Briefly:

```bash
module load R/4.4.0                  # site-specific; get Rscript on PATH
cvd_eda/task2_recount3/setup.sh      # installs recount3 + arrow + jsonlite

source .venv/bin/activate
pip install pyyaml                   # only Python dep beyond stdlib

# Ingests every project in task2_recount3/config/candidate_projects.yaml.
# GTEx HEART is seeded; SRA candidates are appended by Task 3.
cvd_eda/task2_recount3/run.sh                       # -> cvd_eda/data/recount3_raw/
cvd_eda/task2_recount3/run.sh $SCRATCH/cvd_recount3 # override output dir
cvd_eda/task2_recount3/run.sh $SCRATCH/cvd_recount3 --force  # re-ingest existing
```

Per-project Parquet trios (`{project}_counts.parquet`,
`{project}_coldata.parquet`, `{project}_rowdata.parquet`) land in the output
dir alongside `available_projects_catalog.parquet` and
`ingestion_log_recount3.json`. The log's schema is documented in
`task2_recount3/README.md`.

## Running Task 3

Curates per-sample metadata (from Task 1 / Task 2) for CVD relevance using a
two-tier keyword net plus an LLM triage pass for ambiguous samples. Full
details in `curation/README.md`. Briefly:

```bash
source .venv/bin/activate
uv pip install h5py pandas pyarrow anthropic   # if not already installed
export ANTHROPIC_API_KEY=sk-ant-...             # required unless --disable-llm

# ARCHS4
python -m cvd_eda.curation \
    --dataset archs4 \
    --input   "$CVD_EDA_DATA_DIR/archs4_raw.h5" \
    --llm-cache cvd_eda/logs/curation_llm_cache/

# RECOUNT3 (--input is repeatable; pass every coldata parquet Task 2 exported)
python -m cvd_eda.curation \
    --dataset recount3 \
    --input cvd_eda/data/recount3_raw/HEART_coldata.parquet \
    --llm-cache cvd_eda/logs/curation_llm_cache/
```

Writes `cvd_eda/logs/cvd_relevance_{dataset}.csv` (schema:
`sample_id, matched_keyword, llm_relevance, confidence, reasoning,
source_series_id`) plus `curation_log_{dataset}.json`. Task 4 and Task 5 both
consume the CSV from that path.

## Running Task 4

Turns raw counts into a CVD-subset, deduped, gene-ID-harmonized, filtered,
normalized matrix per dataset. Full details in `processing/README.md`.
Briefly:

```bash
source .venv/bin/activate
uv pip install h5py pandas pyarrow      # if not already installed

# ARCHS4
python -m cvd_eda.processing.run \
    --dataset archs4 \
    --archs4-h5 "$CVD_EDA_DATA_DIR/archs4_raw.h5" \
    --relevance-csv cvd_eda/logs/cvd_relevance_archs4.csv \
    --output-dir   cvd_eda/logs/task4_out/

# RECOUNT3 (one invocation processes every project in the directory)
python -m cvd_eda.processing.run \
    --dataset recount3 \
    --recount3-counts-dir cvd_eda/data/recount3_raw/ \
    --relevance-csv       cvd_eda/logs/cvd_relevance_recount3.csv \
    --output-dir          cvd_eda/logs/task4_out/
```

Writes `cvd_matrix_{dataset}_normalized.parquet`,
`cvd_sample_meta_{dataset}.parquet`, and `processing_log_{dataset}.json`
per dataset. The processing log records the exact config used, per-step
sample/gene counts, and the normalization method, so Task 7 (reporting) can
reconstruct the provenance without re-running.

Offline smoke test (fabricates synthetic data, drives the whole pipeline):

```bash
python -m cvd_eda.processing.smoke_test
```
