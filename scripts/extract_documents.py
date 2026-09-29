#!/usr/bin/env python
"""Stage 4 — Extract FinanceBench PDFs page by page."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.document_extractor import extract_corpus_pages  # noqa: E402


DEFAULT_SNAPSHOT = ROOT / "data" / "raw" / "financebench" / "questions.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "pdf_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "pages.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "interim" / "extraction_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract every FinanceBench PDF into zero-indexed page records."
        )
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("Stage 4 — Page-level PDF extraction")
    print("=" * 78)
    print(f"Snapshot: {args.snapshot}")
    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output}")
    print("\nFinanceBench page identity is preserved as ZERO-indexed.\n")

    summary = extract_corpus_pages(
        project_root=ROOT,
        financebench_snapshot=args.snapshot,
        manifest_path=args.manifest,
        output_path=args.output,
        fail_on_page_error=True,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Extraction complete.")
    print(f"Documents:          {summary['documents']}")
    print(f"Pages:              {summary['pages']}")
    print(f"Non-empty pages:    {summary['nonempty_text_pages']}")
    print(f"Empty-text pages:   {summary['empty_text_pages']}")
    print(f"Summary:            {args.summary}")

    if summary["pages"] <= 0:
        raise SystemExit("No pages were extracted.")

    print("\nStage 4 stop condition satisfied:")
    print("  Each physical PDF page has a stable zero-indexed page ID.")
    print("  Blank pages were retained, so later page indices were not shifted.")


if __name__ == "__main__":
    main()
