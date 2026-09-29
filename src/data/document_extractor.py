"""
Corpus-level PDF extraction for FinanceBench.

Inputs
------
* ``data/raw/financebench/questions.jsonl`` from Stage 1
* ``data/raw/pdf_manifest.csv`` and PDFs from Stage 3

Output
------
* ``data/interim/pages.jsonl``

The writer is atomic: pages are first written to ``.tmp`` and then moved into
place only after the full corpus has been extracted and validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.pdf_loader import iter_pdf_pages


VALID_LOCAL_STATUSES = {"downloaded", "skipped_existing"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return rows


def resolve_local_path(
    local_path: str,
    *,
    project_root: str | Path,
) -> Path:
    """Resolve a manifest path whether it is absolute or repo-relative."""
    raw = Path(local_path)
    if raw.is_absolute():
        return raw
    return Path(project_root) / raw


def validate_manifest_for_corpus(
    financebench_rows: list[dict[str, Any]],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate that every FinanceBench source document has one usable manifest row.
    """
    required_columns = {
        "doc_name",
        "company",
        "doc_type",
        "doc_period",
        "local_path",
        "download_status",
    }
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(
            "PDF manifest is missing columns: " + ", ".join(missing_columns)
        )

    required_docs = {
        str(row["doc_name"])
        for row in financebench_rows
        if row.get("doc_name") is not None
    }

    duplicated = manifest.loc[
        manifest["doc_name"].duplicated(keep=False), "doc_name"
    ].astype(str).unique().tolist()
    if duplicated:
        raise ValueError(
            "PDF manifest contains duplicate doc_name rows: "
            + ", ".join(sorted(duplicated)[:20])
        )

    manifest_docs = set(manifest["doc_name"].astype(str))
    missing_docs = sorted(required_docs - manifest_docs)
    if missing_docs:
        raise ValueError(
            "Required FinanceBench documents are absent from the manifest: "
            + ", ".join(missing_docs[:20])
        )

    relevant = manifest[
        manifest["doc_name"].astype(str).isin(required_docs)
    ].copy()

    bad_status = relevant[
        ~relevant["download_status"].isin(VALID_LOCAL_STATUSES)
    ]
    if not bad_status.empty:
        details = ", ".join(
            f"{row.doc_name}={row.download_status}"
            for row in bad_status.itertuples()
        )
        raise ValueError(
            "Cannot extract corpus because some documents are unresolved: "
            + details
        )

    return relevant.sort_values("doc_name", kind="stable").reset_index(drop=True)


def extract_corpus_pages(
    *,
    project_root: str | Path,
    financebench_snapshot: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    fail_on_page_error: bool = True,
) -> dict[str, Any]:
    """
    Extract every physical page from every required FinanceBench PDF.

    Returns a measured extraction summary.
    """
    project_root = Path(project_root)
    snapshot_rows = load_jsonl(financebench_snapshot)
    manifest = pd.read_csv(manifest_path)
    manifest = validate_manifest_for_corpus(snapshot_rows, manifest)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if temp_path.exists():
        temp_path.unlink()

    total_pages = 0
    empty_pages = 0
    doc_page_counts: dict[str, int] = {}

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as writer:
            for row in manifest.itertuples(index=False):
                pdf_path = resolve_local_path(
                    str(row.local_path),
                    project_root=project_root,
                )

                if not pdf_path.exists():
                    raise FileNotFoundError(
                        f"Manifest points to missing PDF for {row.doc_name}: "
                        f"{pdf_path}"
                    )

                pages_for_doc = 0

                for page in iter_pdf_pages(
                    pdf_path,
                    doc_name=str(row.doc_name),
                    company=None if pd.isna(row.company) else str(row.company),
                    doc_type=None if pd.isna(row.doc_type) else str(row.doc_type),
                    doc_period=(
                        None if pd.isna(row.doc_period) else row.doc_period
                    ),
                    fail_on_page_error=fail_on_page_error,
                ):
                    payload = page.to_dict()
                    writer.write(
                        json.dumps(payload, ensure_ascii=False) + "\n"
                    )

                    total_pages += 1
                    pages_for_doc += 1
                    if page.extraction_status == "empty":
                        empty_pages += 1

                if pages_for_doc == 0:
                    raise RuntimeError(
                        f"No pages were emitted for document {row.doc_name}."
                    )

                # Stage 3 stores pypdf page count when available.
                if hasattr(row, "pdf_pages") and not pd.isna(row.pdf_pages):
                    expected_pages = int(row.pdf_pages)
                    if pages_for_doc != expected_pages:
                        raise RuntimeError(
                            f"Page-count mismatch for {row.doc_name}: "
                            f"manifest={expected_pages}, "
                            f"extracted={pages_for_doc}"
                        )

                doc_page_counts[str(row.doc_name)] = pages_for_doc

        temp_path.replace(output_path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    summary = {
        "documents": len(doc_page_counts),
        "pages": total_pages,
        "empty_text_pages": empty_pages,
        "nonempty_text_pages": total_pages - empty_pages,
        "pages_by_document": doc_page_counts,
        "output_path": str(output_path),
        "page_indexing": "zero_based",
    }

    return summary
