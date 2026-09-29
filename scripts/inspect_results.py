#!/usr/bin/env python
"""Print a compact, human-readable view of pipeline JSONL results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return rows


def compact_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def score_fields(chunk: dict[str, Any]) -> str:
    values = []
    for label, key in (
        ("dense", "dense_rank"),
        ("sparse", "sparse_rank"),
        ("rrf", "rrf_rank"),
        ("rerank", "reranker_rank"),
    ):
        if chunk.get(key) is not None:
            values.append(f"{label}={chunk[key]}")
    return ", ".join(values) or f"rank={chunk.get('rank', '?')}"


def milliseconds(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def print_result(row: dict[str, Any], *, top_chunks: int, text_chars: int) -> None:
    diagnostics = row.get("retrieval_diagnostics") or {}
    filters = diagnostics.get("query_filters") or {}
    parameters = row.get("experiment_parameters") or {}
    final_chunks = list(row.get("retrieved_chunks") or [])[:top_chunks]
    reranked = diagnostics.get("reranked_candidates") or []
    truncated = sum(
        bool(chunk.get("reranker_input_truncated")) for chunk in reranked
    )
    windowed = sum(
        int(chunk.get("reranker_window_count") or 1) > 1 for chunk in reranked
    )

    print("=" * 88)
    print(f"{row.get('architecture', '?')} | {row.get('question_id', '?')}")
    print(f"Question:  {row.get('question', '')}")
    print(f"Gold:      {row.get('gold_answer', '')}")
    print(f"Generated: {row.get('generated_answer', '')}")
    if filters:
        print(
            "Filters:   "
            f"company={filters.get('company')!r}, "
            f"fiscal_year={filters.get('fiscal_year')!r}"
        )
    if diagnostics.get("reranker_query"):
        print(
            "Rerank q:  "
            + compact_text(diagnostics["reranker_query"], 320)
        )
    setting_fields = (
        "dense_candidate_k",
        "sparse_candidate_k",
        "fusion_candidate_k",
        "reranker_candidate_k",
        "rrf_k",
        "final_context_k",
    )
    settings = [
        f"{key}={parameters[key]}"
        for key in setting_fields
        if key in parameters
    ]
    if settings:
        print("Settings:  " + ", ".join(settings))

    candidate_count = row.get(
        "candidate_count_before_reranking",
        row.get("candidate_count_before_selection"),
    )
    if candidate_count is not None:
        print(f"Candidates: {candidate_count}")
    if reranked:
        print(
            f"Reranker:  truncated={truncated}/{len(reranked)}, "
            f"multi-window={windowed}/{len(reranked)}"
        )

    print("Final chunks:")
    if not final_chunks:
        print("  [none]")
    for position, chunk in enumerate(final_chunks, start=1):
        year = chunk.get("fiscal_year", "?")
        document = chunk.get("doc_name", "?")
        page = chunk.get("page", "?")
        print(
            f"  {position}. {chunk.get('chunk_id', '?')} | "
            f"{document} | FY{year} | page {page} | {score_fields(chunk)}"
        )
        if chunk.get("reranker_window_count") is not None:
            print(
                "     reranker input: "
                f"tokens={chunk.get('reranker_input_tokens', '?')}, "
                f"truncated={chunk.get('reranker_input_truncated', False)}, "
                f"windows={chunk.get('reranker_window_count', 1)}, "
                f"best_window={chunk.get('reranker_best_window_index', 0)}"
            )
        print(f"     {compact_text(chunk.get('text'), text_chars)}")

    print(
        "Latency:   "
        f"retrieval={milliseconds(row.get('retrieval_latency_ms'))}, "
        f"reranking={milliseconds(row.get('reranking_latency_ms'))}, "
        f"generation={milliseconds(row.get('generation_latency_ms'))}, "
        f"total={milliseconds(row.get('total_latency_ms'))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect pipeline results without printing full chunk JSON."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--top-chunks", type=int, default=5)
    parser.add_argument("--text-chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_chunks <= 0:
        raise ValueError("--top-chunks must be > 0.")
    if args.text_chars <= 0:
        raise ValueError("--text-chars must be > 0.")

    rows = load_results(args.result)
    if args.question_id:
        rows = [
            row for row in rows
            if str(row.get("question_id")) == str(args.question_id)
        ]
        if not rows:
            raise ValueError(f"Question ID not found: {args.question_id}")

    print(f"Result file: {args.result.resolve()}")
    print(f"Rows shown:  {len(rows)}")
    if any("experiment_parameters" not in row for row in rows):
        print(
            "WARNING: This is a legacy result without controlled experiment "
            "metadata. Rerun the pipeline before evaluating the fixes."
        )
    for row in rows:
        print_result(
            row,
            top_chunks=args.top_chunks,
            text_chars=args.text_chars,
        )


if __name__ == "__main__":
    main()
