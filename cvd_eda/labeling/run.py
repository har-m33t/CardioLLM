"""CLI entrypoint for Task 5.

Typical invocation::

    source .venv/bin/activate
    uv pip install anthropic     # if not already installed

    python -m cvd_eda.labeling.run \\
        --input  cvd_eda/logs/cvd_relevance_archs4.csv \\
        --output cvd_eda/logs/label_proposals_archs4.csv \\
        --log    cvd_eda/logs/task5_run_log_archs4.json \\
        --llm-cache cvd_eda/logs/label_cache/ \\
        --geo-cache cvd_eda/logs/geo_cache/

Requires ``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile) in the
environment. See :file:`cvd_eda/labeling/README.md` for the review workflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cvd_eda.labeling.label import RunConfig, run
from cvd_eda.labeling.llm import DEFAULT_MODEL


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cvd_eda.labeling.run",
        description="Task 5: propose case/control/subtype labels for CVD-relevant samples. "
                    "Output requires human review before Task 6.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Task 3 output CSV (cvd_relevance_{dataset}.csv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write label_proposals.csv.",
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="Path to write the run log (JSON, consumed by Task 7).",
    )
    parser.add_argument(
        "--llm-cache",
        required=True,
        type=Path,
        help="Directory for the on-disk LLM cache. Reruns with same model + "
             "prompt hit this cache and skip billing.",
    )
    parser.add_argument(
        "--geo-cache",
        type=Path,
        default=None,
        help="Directory for the GEO series-description cache. If omitted or "
             "--no-geo-fetch is passed, the LLM only sees the sample-level "
             "metadata from Task 3.",
    )
    parser.add_argument(
        "--no-geo-fetch",
        dest="use_geo_fetch",
        action="store_false",
        help="Skip fetching GEO series descriptions. Only sample-level "
             "metadata from Task 3 is passed to the LLM.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Only samples with Task 3 llm_relevance=='yes' AND "
             "confidence >= this threshold get labeled (default: 0.7).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on how many samples to label (useful for smoke tests).",
    )
    parser.add_argument(
        "--ncbi-email",
        default=None,
        help="Contact email for NCBI E-utilities. Recommended by NCBI so they "
             "can contact you before rate-limiting.",
    )
    parser.add_argument(
        "--ncbi-api-key",
        default=None,
        help="NCBI API key. Raises the E-utilities rate limit from 3 to 10 req/s.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input.exists():
        print(f"error: input CSV not found: {args.input}", file=sys.stderr)
        return 2

    config = RunConfig(
        input_csv=args.input,
        output_csv=args.output,
        log_path=args.log,
        llm_cache_dir=args.llm_cache,
        geo_cache_dir=args.geo_cache,
        model=args.model,
        min_relevance_confidence=args.min_confidence,
        max_samples=args.max_samples,
        use_geo_fetch=args.use_geo_fetch and args.geo_cache is not None,
        ncbi_email=args.ncbi_email,
        ncbi_api_key=args.ncbi_api_key,
    )

    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
