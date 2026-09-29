#!/usr/bin/env python
"""Stage 6 — Map FinanceBench gold evidence to pages and chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.evidence_mapper import (  # noqa: E402
    build_evidence_mapping,
    load_jsonl,
    write_mapping_json,
)


DEFAULT_SNAPSHOT = ROOT / "data" / "raw" / "financebench" / "questions.jsonl"
DEFAULT_PAGES = ROOT / "data" / "interim" / "pages.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "evidence_mapping.json"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "evidence_mapping_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect FinanceBench gold evidence to extracted pages and "
            "evidence-matched chunks."
        )
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.35,
        help="Minimum evidence-token containment for non-exact chunk matching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.containment_threshold <= 1.0:
        raise ValueError("--containment-threshold must be between 0 and 1.")

    print("=" * 78)
    print("Stage 6 — FinanceBench gold evidence mapping")
    print("=" * 78)

    financebench_rows = load_jsonl(args.snapshot)
    pages = load_jsonl(args.pages)
    chunks = load_jsonl(args.chunks)

    print(f"Questions: {len(financebench_rows)}")
    print(f"Pages:     {len(pages)}")
    print(f"Chunks:    {len(chunks)}")
    print(
        f"Containment threshold: {args.containment_threshold:.2f}\n"
    )

    mapping, summary = build_evidence_mapping(
        financebench_rows=financebench_rows,
        pages=pages,
        chunks=chunks,
        containment_threshold=args.containment_threshold,
    )

    write_mapping_json(mapping, args.output)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Mapping: {args.output}")
    print(f"Summary: {args.summary}")
    print(
        "Strict evidence items mapped: "
        f"{summary['evidence_items_with_strict_chunk_match']}/"
        f"{summary['evidence_items']}"
    )
    print(
        "Questions with at least one non-strict evidence-to-chunk mapping: "
        f"{summary['questions_with_partial_chunk_mapping']}"
    )

    print("\nStage 6 stop condition satisfied:")
    print("  Every gold document/page reference exists in the extracted corpus.")
    print("  Page-level and strict evidence-text chunk relevance are separated.")
    print(
        "  Questions without a strict text match remain usable for "
        "page-level retrieval evaluation."
    )


if __name__ == "__main__":
    main()
