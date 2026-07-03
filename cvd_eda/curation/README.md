# Task 3 — Metadata-Curation Agent (CVD relevance)

Owner: Claude Code · Package: `cvd_eda.curation` · Entrypoint: `python -m cvd_eda.curation`

Implements the "Metadata-Curation Agent (CVD relevance)" stage of
`.claude/EDA_CLAUDE_TASKS.md`. Decides, per sample, whether the metadata is
CVD-relevant — with an explicit LLM triage step for the ambiguous middle band
so nothing borderline is silently included or dropped.

## Three-pass design

1. **Keyword net** (`keywords.py`). A high-recall regex over the sample's
   free-text metadata. Terms are split into two tiers:
   - **`STRONG_KEYWORDS`** — unambiguous disease phrases and study-shorthand
     acronyms (`myocardial infarct*`, `heart failure`, `HCM`, `atrial
     fibrillation`, …). A strong hit calls the sample **yes** without an
     LLM call — the phrase itself is decisive.
   - **`AMBIGUOUS_KEYWORDS`** — anatomical/physiological single tokens
     (`cardiac`, `aortic`, `hypertension`, …) that appear both in disease
     studies and in healthy-tissue references. A hit here goes to the LLM.
   - **No match** → sample is called **no** without an LLM call.

2. **LLM triage** (`llm.py`). Only ambiguous samples are sent to the model.
   The system prompt teaches a three-way distinction (yes/no/uncertain) with
   explicit guidance that healthy controls run inside a CVD study count as
   "yes" and that CVD-adjacent tissue from a non-CVD study counts as "no".
   The model returns structured JSON so downstream code doesn't parse prose.
   - Default model: `claude-haiku-4-5-20251001` — three-way classification
     over thousands of samples is a Haiku-shaped workload.
   - On-disk cache keyed by `sha256(model || prompt)`; identical samples
     across ARCHS4 and RECOUNT3 are classified once.
   - `--disable-llm` skips the stage entirely and marks every ambiguous
     sample `uncertain` with confidence 0 — for smoke tests, or when you
     want to defer everything borderline to the human review at Task 5.

3. **(Optional) GEO series context** (`geo.py`). If sample text is shorter
   than 60 chars *and* `--use-geo-fetch` is on, we pull the parent GSE's
   title + summary from NCBI Entrez and hand it to the LLM as extra context.
   Off by default: E-utilities is rate-limited, not every `series_id` is a
   GSE, and sample-level metadata is usually informative enough. Task 5
   (labeling) does GEO fetching too, on a much smaller volume.

## Layout

```
cvd_eda/curation/
├── README.md           (this file)
├── __init__.py
├── __main__.py         (enables `python -m cvd_eda.curation`)
├── keywords.py         (two-tier keyword net + KeywordMatch dataclass)
├── metadata.py         (ARCHS4 H5 + RECOUNT3 parquet loaders → shared schema)
├── llm.py              (LLMClassifier: Anthropic Messages + JSON parse + cache)
├── geo.py              (optional NCBI E-utilities GSE fetcher + on-disk cache)
└── curate.py           (orchestrator + argparse CLI + JSON audit log)
```

Loaders normalize to a shared 3-column shape (`sample_id`, `source_series_id`,
`text`); the rest of the pipeline is dataset-agnostic from that point on.

## Installing runtime deps

Not all Task 3 dependencies are in `pyproject.toml` (they are only needed by
the CVD EDA side pipeline, not TinyLLaVA):

```bash
source .venv/bin/activate
uv pip install h5py pandas pyarrow anthropic
```

- `h5py` is required for the ARCHS4 loader.
- `pandas` + `pyarrow` are required for the RECOUNT3 loader (Task 2's
  parquet outputs) and for CSV writing.
- `anthropic` is required unless you always run with `--disable-llm`.

## Running

### ARCHS4

```bash
# Points at Task 1's stable symlink (or the versioned filename).
python -m cvd_eda.curation \
    --dataset archs4 \
    --input   "$CVD_EDA_DATA_DIR/archs4_raw.h5" \
    --llm-cache cvd_eda/logs/curation_llm_cache/
```

Writes:

- `cvd_eda/logs/cvd_relevance_archs4.csv` — one row per sample.
- `cvd_eda/logs/curation_log_archs4.json` — audit trail for Task 7.

### RECOUNT3

```bash
# --input is repeatable; pass every coldata parquet Task 2 exported.
python -m cvd_eda.curation \
    --dataset recount3 \
    --input cvd_eda/data/recount3_raw/HEART_coldata.parquet \
    --input cvd_eda/data/recount3_raw/SRP123456_coldata.parquet \
    --llm-cache cvd_eda/logs/curation_llm_cache/
```

### Smoke test without an API key

```bash
python -m cvd_eda.curation --dataset archs4 --input path/to/archs4_raw.h5 --disable-llm
```

Every ambiguous sample becomes `uncertain` with confidence 0.0. Useful for
verifying wiring before you spend anything on the LLM stage.

## Output CSV schema

Exactly matches `.claude/EDA_CLAUDE_TASKS.md`:

| Column             | Meaning                                                                          |
|--------------------|----------------------------------------------------------------------------------|
| `sample_id`        | GEO accession (ARCHS4) or `sample_id` from Task 2's coldata (RECOUNT3).          |
| `matched_keyword`  | Representative keyword hit; empty when no match.                                 |
| `llm_relevance`    | `yes` \| `no` \| `uncertain`.                                                    |
| `confidence`       | 0.0–1.0. Keyword-only decisions: 0.9 (yes) / 0.95 (no). LLM decisions: the model's self-reported score, clamped. `--disable-llm` fallback: 0.0. |
| `reasoning`        | 1–3 sentences quoting the phrase the decision hinges on.                         |
| `source_series_id` | GSE (ARCHS4) / SRP / GTEx / TCGA project accession; `""` if the loader couldn't find one. |

Task 4 filters by `llm_relevance == "yes" AND confidence >= 0.7`; Task 5
consumes the `yes`/high-confidence subset to propose labels.

## JSON log schema (`curation_log_{dataset}.json`)

Task 7 (reporting) reads this. Fields:

| Field                         | Meaning                                                            |
|-------------------------------|--------------------------------------------------------------------|
| `task`                        | Always `"3-metadata-curation"`                                     |
| `dataset`                     | `"archs4"` or `"recount3"`                                         |
| `inputs`                      | Input paths as given on the CLI                                    |
| `output_csv`                  | Path to the CSV this run wrote                                     |
| `model`                       | Anthropic model id used (null if `--disable-llm`)                  |
| `confidence_threshold`        | Threshold used to compute `flagged_below_threshold` in stats       |
| `use_geo_fetch`, `disable_llm`| CLI flags recorded verbatim                                        |
| `run_started_utc`, `run_finished_utc` | ISO-8601                                                   |
| `stats`                       | See table below                                                    |
| `keyword_net`                 | The strong/ambiguous keyword lists actually used, so a future re-run knows what changed |
| `notes`                       | Free-text warnings (e.g. "LLM disabled — N samples marked uncertain") |

`stats` fields:

| Field                    | Meaning                                                    |
|--------------------------|------------------------------------------------------------|
| `total`                  | Samples processed                                          |
| `keyword_strong`         | Samples with a strong keyword hit (auto-yes)               |
| `keyword_ambiguous`      | Samples sent to the LLM                                    |
| `keyword_none`           | Samples with no keyword hit (auto-no)                      |
| `llm_calls`              | LLM requests actually sent (excludes cache hits)           |
| `llm_cache_hits`         | Samples resolved from the LLM cache                        |
| `llm_yes`/`llm_no`/`llm_uncertain` | LLM verdict breakdown                            |
| `flagged_below_threshold`| Samples below `confidence_threshold` (need human review)   |
| `elapsed_sec`            | Wall-clock time for the run                                |

## Design decisions

- **Strong hits bypass the LLM.** The task spec explicitly says the LLM is
  for *ambiguous/borderline* matches. Sending an obvious "heart failure"
  sample to the LLM burns budget on a decision the regex has already made.
- **Keyword-only "no" is confident.** A sample with no CVD vocabulary at all
  is almost certainly not CVD-relevant. We call it 0.95, not 1.0, to leave
  room for the (rare) case where a series description at Task 5 revives it.
- **Anthropic Messages over other providers.** Aligns with the existing
  agent stack in this project. Haiku is priced right for this volume; a
  ~1M-sample ARCHS4 filter with ~15% ambiguous rate is a few dollars at
  Haiku 4.5's rate card, and the on-disk cache means reruns are free.
- **CSV, not Parquet.** Task 4 and Task 5 both read it with `pd.read_csv`;
  the file is small (rows-times-few-KB) and hand-inspectable during human
  review, which matters for the labeling gate at Task 5.
- **GEO fetch is opt-in, not automatic.** Sample-level metadata is usually
  enough. Task 5 does GEO fetching on the labeling subset (much smaller
  volume), so the incremental value at Task 3 is low. When you do turn it
  on, cache hits make reruns free.
- **Uniqueness of `sample_id` is enforced at load time.** ARCHS4
  reprocesses / resubmits happen; a duplicate id would silently confuse
  Task 4's join. We raise loudly so it gets fixed at the source.

## Interaction with sibling tasks

- **Task 1 (`cvd_eda.ingestion.archs4_ingest`)** — Task 3 reads the H5 it
  produces. The stable `archs4_raw.h5` symlink is what you pass to
  `--input`.
- **Task 2 (`cvd_eda.recount3`)** — Task 3 reads every
  `{project}_coldata.parquet` it exports. Task 2's R export promotes the
  sample rownames to a real `sample_id` column, which is exactly what
  Task 3 keys on.
- **Task 4 (`cvd_eda.processing`)** — reads
  `cvd_eda/logs/cvd_relevance_{dataset}.csv` via `--relevance-csv` and
  filters by `llm_relevance == "yes"` and `confidence >= 0.7`.
- **Task 5 (`cvd_eda.labeling`)** — reads the same CSV via `--input`,
  further narrows to the high-confidence "yes" subset, and proposes
  labels for the human review gate.
- **Task 7 (reporting)** — reads `curation_log_{dataset}.json` for the
  audit trail.

## Not implemented here

- Any actual outcome labeling (case/control/subtype) — that's Task 5, and
  its output is the human-review gate.
- Any GEO fetching beyond the optional per-series-description helper. Task 5
  has its own `geo.py` for label evidence.
- Any batch effects / QC — Task 6 owns the EDA.
