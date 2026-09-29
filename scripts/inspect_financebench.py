from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.financebench_loader import (
    EXPECTED_OPEN_SOURCE_ROWS,
    build_unique_document_table,
    load_financebench,
    save_financebench_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load FinanceBench from Hugging Face, validate it, inspect one "
            "example and save a local JSONL snapshot."
        )
    )
    parser.add_argument(
        "--show-row",
        type=int,
        default=0,
        help="Row index to display (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing local snapshot if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("Stage 1 — FinanceBench Hugging Face inspection")
    print("=" * 78)

    print("\nLoading PatronusAI/financebench (split='train')...")
    dataset = load_financebench()
    row_count = len(dataset)

    if row_count != EXPECTED_OPEN_SOURCE_ROWS:
        raise RuntimeError(
            "Unexpected FinanceBench row count. "
            f"Expected {EXPECTED_OPEN_SOURCE_ROWS}, received {row_count}. "
            "Stop and inspect the upstream dataset before continuing, because "
            "a changed benchmark would alter the experiment."
        )

    print(f"\nLoaded FinanceBench: {row_count} questions")

    print("\nDataset object:")
    print(dataset)

    print("\nColumns:")
    for number, column in enumerate(dataset.column_names, start=1):
        print(f"  {number:>2}. {column}")

    if not 0 <= args.show_row < row_count:
        raise IndexError(
            f"--show-row must be between 0 and {row_count - 1}; "
            f"received {args.show_row}."
        )

    print(f"\nExample row [{args.show_row}]:")
    pprint(dataset[args.show_row], sort_dicts=False)

    snapshot_path = save_financebench_snapshot(
        dataset,
        overwrite=args.overwrite,
    )
    print(f"\nLocal snapshot: {snapshot_path}")

    df = dataset.to_pandas()
    documents = build_unique_document_table(df)

    print("\nMeasured dataset overview:")
    print(f"  Questions:         {len(df)}")
    print(f"  Unique companies:  {df['company'].nunique(dropna=True)}")
    print(f"  Unique documents:  {df['doc_name'].nunique(dropna=True)}")

    print("\nquestion_type distribution:")
    print(df["question_type"].value_counts(dropna=False).to_string())

    print("\nquestion_reasoning distribution:")
    print(df["question_reasoning"].value_counts(dropna=False).to_string())

    print("\nFirst 10 unique source documents:")
    print(documents.head(10).to_string(index=False))

    print("\nStage 1 stop condition satisfied:")
    print(f"  Loaded FinanceBench: {row_count} questions")
    print("  Required fields are present.")
    print("  Local snapshot is available for Stages 2 and 3.")


if __name__ == "__main__":
    main()
