# Task 5 — Labeling Agent (case/control/subtype) ⚠️

Implements **Task 5** from `.claude/EDA_CLAUDE_TASKS.md`: propose the outcome
label each CVD-relevant sample will carry into the elastic-net stage, with a
verbatim evidence quote for every proposal and a hard human-review checkpoint
before Task 6 is allowed to consume the output.

> ⚠️ **The output is a *proposal* file. Task 6 must not read `label_proposals.csv`
> until a human has reviewed the `uncertain` rows and spot-checked the rest.**
> The orchestrator ends every run with a `STOP — Task 5 output requires human
> review` banner on stdout so the checkpoint is impossible to miss in logs.

## What it does

For each row in Task 3's `cvd_relevance_{dataset}.csv` that the upstream
classifier called `yes` with `confidence >= --min-confidence`:

1. Optionally fetch the parent GEO series description via
   [`cvd_eda.task3_curation.geo.GEOSeriesFetcher`][geofetcher]. The fetcher
   is reused (not duplicated) so both Task 3 and Task 5 share one on-disk
   cache and one NCBI throttling policy.
2. Ask the LLM to propose one of:
   - `case` / `control` — for two-arm disease-vs-healthy studies.
   - A specific subtype: `HCM`, `DCM`, `HF`, `HFrEF`, `HFpEF`, `MI`, `CAD`,
     `AF`, `PAH`, `AS`, `other_CVD`.
   - `uncertain` — when the metadata does not support a confident label.
3. Write one row to `label_proposals.csv`.

The prompt requires the LLM to return a verbatim `evidence_quote` from the
input text — so a reviewer can grep the source metadata for the quote as a
fast spot-check without re-reading the whole series.

[geofetcher]: ../task3_curation/geo.py

## Input

Task 3's `cvd_relevance_{dataset}.csv`. Required columns:

- `sample_id`
- `llm_relevance` (yes/no/uncertain)
- `confidence` (0.0–1.0)
- `source_series_id`

Optional but used when present: `text`, `matched_keyword`, `reasoning`. Any
other columns are ignored.

## Output

`label_proposals.csv` with columns declared in `schema.py::CSV_COLUMNS`:

| Column | Notes |
|---|---|
| `sample_id` | Copied from Task 3 |
| `proposed_label` | One of `schema.LABEL_VOCAB` |
| `confidence` | 0.0–1.0; anchor: ≥0.9 = explicit in sample-level fields; ~0.5 = inferred from series design; ≤0.3 = ambiguous |
| `evidence_quote` | Verbatim substring of the input the model quoted |
| `uncertain_reason` | Populated iff `proposed_label == "uncertain"` |
| `source_series_id` | Copied from Task 3 |
| `model` | Anthropic model ID that produced the row |
| `cached` | `True` if the row was served from the on-disk cache |

Plus a JSON run log (`--log`) capturing model, thresholds, elapsed time,
per-label counts, LLM call / cache-hit counts. Task 7's Reporting Agent
reads this file directly.

## Install

Anthropic SDK is not in the base install. From the repo root:

```bash
source .venv/bin/activate
uv pip install anthropic
```

`requests` and `urllib.request` (used by the GEO fetcher) are stdlib /
already installed.

## Credentials

- `ANTHROPIC_API_KEY` — required. Or run `ant auth login` once and the SDK's
  zero-arg client picks up the OAuth profile automatically. Unlike Task 3
  there is no `--disable-llm` escape hatch: Task 5 is the highest-leverage
  step and running without an LLM would just produce empty proposals.
- `NCBI_EMAIL` (or `--ncbi-email`) — recommended by NCBI so they can contact
  you before rate-limiting rather than silently 429-ing.
- `NCBI_API_KEY` (or `--ncbi-api-key`) — optional; raises E-utilities rate
  limit from 3 req/s to 10 req/s.

## Run

Assuming Task 3 has emitted `cvd_relevance_archs4.csv`:

```bash
python -m cvd_eda.labeling.run \
    --input     cvd_eda/logs/cvd_relevance_archs4.csv \
    --output    cvd_eda/logs/label_proposals_archs4.csv \
    --log       cvd_eda/logs/task5_run_log_archs4.json \
    --llm-cache cvd_eda/logs/label_cache/ \
    --geo-cache cvd_eda/logs/geo_cache/ \
    --min-confidence 0.7
```

Smoke test:

```bash
python -m cvd_eda.labeling.run ... --max-samples 25 --log-level DEBUG
```

To skip GEO fetching (sample-level metadata only):

```bash
python -m cvd_eda.labeling.run ... --no-geo-fetch
```

## Model choice

Defaults to `claude-opus-4-8`. Task 3 uses Haiku for its bulk relevance pass
because that's a cheap three-way classification over thousands of samples.
Task 5 is a smaller volume (only the yes-with-high-confidence subset) *and*
the highest-leverage step in the pipeline — a mislabel here silently
corrupts every downstream elastic-net result. We do not downgrade for cost.

Override with `--model` only when reproducing a prior run.

## Design notes

- **Bias toward `uncertain`.** The system prompt tells the model to prefer
  flagging over guessing. A wrong label here corrupts everything downstream;
  a flagged one costs a reviewer a minute.
- **Series description is context, not authority.** When the sample-level
  fields are silent and the label has to be inferred from the study design,
  confidence is anchored around 0.5 so the reviewer can filter on it.
- **Verbatim evidence quote.** The quote must appear as a substring of the
  input. Reviewers can `grep` for it against the source metadata.
- **JSON parsed from text, not `output_config.format`.** Matches Task 3's
  `curation/llm.py` convention. If both stages ever need to switch to
  structured outputs, do it in a single PR against both files.
- **Reused GEO fetcher.** `cvd_eda.task3_curation.geo.GEOSeriesFetcher` is
  imported rather than duplicated. A future refactor could hoist this into a
  shared `cvd_eda/geo.py`; skipping that today because the file is fine
  where it lives and the import path is short.
- **On-disk cache keyed by `sha256(model || system || prompt)`.** Reruns
  with unchanged inputs are free (matches Task 3).

## Human-review workflow

1. Open `label_proposals.csv` sorted by `confidence` ascending.
2. Walk the `uncertain` rows first. For each, read `uncertain_reason` and
   pull the source GEO series at
   `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={source_series_id}`.
   Either correct the label in-place or drop the sample.
3. Spot-check the confident rows: sample 5–10% at random, verify that the
   `evidence_quote` actually appears in the source metadata and matches the
   label.
4. Save the reviewed file as `label_proposals.reviewed.csv` and point Task 6
   at that path. Never let Task 6 read the raw proposal file.

The design of this step is worth defending in code review: this is the
single place in the pipeline where automated labels get wrong in a way that
corrupts every downstream number silently. The human gate is the whole
point of the task, not a nice-to-have.
