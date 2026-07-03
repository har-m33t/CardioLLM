"""Task 3 — Metadata-Curation Agent (CVD relevance).

See ``cvd_eda/curation/README.md`` for the design and CLI. Consumes per-sample
metadata from Task 1 (ARCHS4 HDF5) and Task 2 (RECOUNT3 coldata Parquet) and
emits ``cvd_relevance_{dataset}.csv`` + ``curation_log_{dataset}.json`` that
Tasks 4, 5, and 7 read.
"""
