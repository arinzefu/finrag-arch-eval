"""
FinanceBench loading and local snapshot utilities.

Stage 1 goals
-------------
1. Load the open-source FinanceBench split from Hugging Face.
2. Validate the fields required by this project.
3. Save a local JSONL snapshot for downstream stages.
4. Provide a local loader so later experiments do not repeatedly depend on
   the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import Dataset, load_dataset


DATASET_ID = "PatronusAI/financebench"
DATASET_SPLIT = "train"
EXPECTED_OPEN_SOURCE_ROWS = 150

REQUIRED_COLUMNS = {
    "financebench_id",
    "company",
    "doc_name",
    "question",
    "answer",
    "question_type",
    "question_reasoning",
    "evidence",
    "justification",
    "doc_link",
}


def get_project_root() -> Path:
    """Return the repository root inferred from this module's location."""
    return Path(__file__).resolve().parents[2]


def get_default_snapshot_path() -> Path:
    """Return the canonical local FinanceBench snapshot path."""
    return (
        get_project_root()
        / "data"
        / "raw"
        / "financebench"
        / "questions.jsonl"
    )


def validate_required_columns(columns: Iterable[str]) -> None:
    """Raise if any project-required FinanceBench fields are absent."""
    available = set(columns)
    missing = sorted(REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(
            "FinanceBench is missing required columns: " + ", ".join(missing)
        )


def load_financebench(
    dataset_id: str = DATASET_ID,
    split: str = DATASET_SPLIT,
) -> Dataset:
    """Load FinanceBench from Hugging Face and validate its schema."""
    dataset = load_dataset(dataset_id, split=split)
    validate_required_columns(dataset.column_names)
    return dataset


def save_financebench_snapshot(
    dataset: Dataset,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Save FinanceBench as newline-delimited JSON and return the path."""
    path = Path(output_path) if output_path else get_default_snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        return path

    dataset.to_json(
        str(path),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return path


def load_local_financebench(
    snapshot_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the Stage 1 local JSONL snapshot as a pandas DataFrame."""
    path = Path(snapshot_path) if snapshot_path else get_default_snapshot_path()

    if not path.exists():
        raise FileNotFoundError(
            f"FinanceBench snapshot was not found at:\n  {path}\n"
            "Run `python scripts/inspect_financebench.py` first."
        )

    df = pd.read_json(path, lines=True)
    validate_required_columns(df.columns)
    return df


def build_unique_document_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row-per-document table from the FinanceBench snapshot.

    Before deduplicating, verify that repeated ``doc_name`` values do not
    disagree on source-document metadata.
    """
    required = {"doc_name", "doc_link"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot build document table; missing columns: "
            + ", ".join(sorted(missing))
        )

    metadata_columns = [
        column
        for column in [
            "company",
            "doc_type",
            "doc_period",
            "doc_link",
            "gics_sector",
        ]
        if column in df.columns
    ]

    inconsistencies: list[str] = []
    grouped = df.groupby("doc_name", dropna=False)

    for column in metadata_columns:
        counts = grouped[column].nunique(dropna=False)
        bad_docs = counts[counts > 1].index.tolist()
        if bad_docs:
            inconsistencies.append(
                f"{column}: {', '.join(map(str, bad_docs[:10]))}"
            )

    if inconsistencies:
        raise ValueError(
            "Inconsistent metadata found for duplicate doc_name values:\n"
            + "\n".join(inconsistencies)
        )

    columns = ["doc_name", *metadata_columns]
    return (
        df[columns]
        .drop_duplicates(subset=["doc_name"])
        .sort_values("doc_name", kind="stable")
        .reset_index(drop=True)
    )
