#!/usr/bin/env python
"""Stage 15: score held-out P1-P3 retrieval at the gold evidence page level."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("P1", "P2", "P3")
DEFAULT_MANIFEST = (
    ROOT / "results" / "raw_runs" / "financebench-heldout-v1_manifest.json"
)
DEFAULT_EVIDENCE = ROOT / "data" / "processed" / "evidence_mapping.json"
DEFAULT_METRICS = ROOT / "results" / "evaluation" / "retrieval_metrics.csv"
DEFAULT_SUMMARY = ROOT / "results" / "aggregated" / "retrieval_summary.csv"
DEFAULT_PROVENANCE = (
    ROOT / "results" / "evaluation" / "retrieval_evaluation_manifest.json"
)

METRIC_FIELDS = (
    "run_id", "architecture", "question_id", "gold_page_count",
    "retrieved_chunk_count", "retrieved_unique_page_count",
    "gold_pages_retrieved_at_5", "first_gold_rank", "hit_at_1",
    "hit_at_3", "hit_at_5", "recall_at_5", "mrr_at_5",
    "gold_page_refs", "retrieved_page_refs_at_5",
)
SUMMARY_FIELDS = (
    "table_title", "run_id", "architecture", "question_count",
    "no_result_count", "hit_at_1", "hit_at_3", "hit_at_5",
    "recall_at_5", "mrr_at_5",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def page_ref(value: dict, *, label: str) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a page-reference object.")
    doc_name = value.get("doc_name")
    page = value.get("page")
    if not isinstance(doc_name, str) or not doc_name.strip():
        raise ValueError(f"{label} has no document name.")
    if isinstance(page, bool) or not isinstance(page, int) or page < 0:
        raise ValueError(f"{label} must have a nonnegative, zero-indexed page.")
    return doc_name, page


def encode_refs(refs: list[tuple[str, int]]) -> str:
    return json.dumps(
        [{"doc_name": doc, "page": page} for doc, page in refs],
        ensure_ascii=False, separators=(",", ":"),
    )


def score_question(
    *, run_id: str, architecture: str, question_id: str,
    chunks: list[dict], gold_refs: list[dict],
) -> dict:
    if not isinstance(chunks, list) or len(chunks) > 5:
        raise ValueError(f"{architecture} {question_id}: expected 0-5 final chunks.")
    if not isinstance(gold_refs, list) or not gold_refs:
        raise ValueError(f"{question_id}: no gold_page_refs available.")

    gold = [page_ref(ref, label=f"{question_id} gold") for ref in gold_refs]
    gold_set = set(gold)
    retrieved = []
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.get("rank") != rank:
            raise ValueError(
                f"{architecture} {question_id}: final chunk rank {rank} is invalid."
            )
        retrieved.append(
            page_ref(chunk, label=f"{architecture} {question_id} rank {rank}")
        )

    first_gold_rank = next(
        (rank for rank, ref in enumerate(retrieved, start=1) if ref in gold_set),
        None,
    )
    unique_hits = gold_set.intersection(retrieved)
    return {
        "run_id": run_id,
        "architecture": architecture,
        "question_id": question_id,
        "gold_page_count": len(gold_set),
        "retrieved_chunk_count": len(retrieved),
        "retrieved_unique_page_count": len(set(retrieved)),
        "gold_pages_retrieved_at_5": len(unique_hits),
        "first_gold_rank": first_gold_rank if first_gold_rank is not None else "",
        "hit_at_1": int(first_gold_rank is not None and first_gold_rank <= 1),
        "hit_at_3": int(first_gold_rank is not None and first_gold_rank <= 3),
        "hit_at_5": int(first_gold_rank is not None and first_gold_rank <= 5),
        "recall_at_5": len(unique_hits) / len(gold_set),
        "mrr_at_5": 1 / first_gold_rank if first_gold_rank else 0.0,
        "gold_page_refs": encode_refs(sorted(gold_set)),
        "retrieved_page_refs_at_5": encode_refs(retrieved),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def validate_manifest(manifest: dict, manifest_path: Path) -> list[str]:
    expected_ids = manifest.get("question_ids")
    if not isinstance(expected_ids, list) or not expected_ids:
        raise ValueError("Run manifest has no ordered question IDs.")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Run manifest contains duplicate question IDs.")
    if manifest.get("question_count") != len(expected_ids):
        raise ValueError("Run manifest question count does not match its IDs.")
    if not set(ARCHITECTURES).issubset(manifest.get("architectures", [])):
        raise ValueError("Run manifest does not include P1, P2, and P3.")
    question_source = Path(manifest["question_source"])
    if file_sha256(question_source) != manifest.get("question_source_sha256"):
        raise ValueError("Run question source differs from its manifest hash.")
    source_ids = [str(row["financebench_id"]) for row in load_jsonl(question_source)]
    if source_ids != expected_ids:
        raise ValueError("Question source order differs from run manifest.")
    for architecture in ARCHITECTURES:
        if architecture not in manifest.get("outputs", {}):
            raise ValueError(f"Missing {architecture} output in {manifest_path}.")
    return expected_ids


def evaluate(manifest: dict, evidence: dict, expected_ids: list[str]) -> tuple[list[dict], list[dict]]:
    run_id = str(manifest["run_id"])
    metrics = []
    summaries = []
    for architecture in ARCHITECTURES:
        path = Path(manifest["outputs"][architecture])
        raw_rows = load_jsonl(path)
        raw_ids = [str(row.get("question_id")) for row in raw_rows]
        if raw_ids != expected_ids:
            raise ValueError(f"{architecture} question IDs/order differ from run manifest.")
        arch_metrics = []
        for row in raw_rows:
            question_id = str(row["question_id"])
            if row.get("architecture") != architecture:
                raise ValueError(f"{path}: wrong architecture for {question_id}.")
            mapping = evidence.get(question_id)
            if not isinstance(mapping, dict) or not mapping.get("mapping_complete"):
                raise ValueError(f"Missing or incomplete evidence map for {question_id}.")
            arch_metrics.append(score_question(
                run_id=run_id,
                architecture=architecture,
                question_id=question_id,
                chunks=row.get("retrieved_chunks"),
                gold_refs=mapping.get("gold_page_refs"),
            ))
        metrics.extend(arch_metrics)
        count = len(arch_metrics)
        summaries.append({
            "table_title": "Table R1: Held-out page-level retrieval performance",
            "run_id": run_id,
            "architecture": architecture,
            "question_count": count,
            "no_result_count": sum(
                row["retrieved_chunk_count"] == 0 for row in arch_metrics
            ),
            **{
                key: sum(float(row[key]) for row in arch_metrics) / count
                for key in ("hit_at_1", "hit_at_3", "hit_at_5", "recall_at_5", "mrr_at_5")
            },
        })
    return metrics, summaries


def write_csv(rows: list[dict], fields: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provenance-output", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.metrics_output, args.summary_output, args.provenance_output)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Evaluation output paths must be distinct.")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Retrieval evaluation output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_ids = validate_manifest(manifest, args.manifest)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    metrics, summaries = evaluate(manifest, evidence, expected_ids)

    write_csv(metrics, METRIC_FIELDS, args.metrics_output)
    write_csv(summaries, SUMMARY_FIELDS, args.summary_output)
    provenance = {
        "title": "Stage 15: Held-out Page-Level Retrieval Evaluation",
        "run_id": manifest["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_unit": "(doc_name, zero-indexed page)",
        "architectures": list(ARCHITECTURES),
        "question_count_per_architecture": len(expected_ids),
        "metric_definitions": {
            "hit_at_k": "1 if any gold page occurs among the first k final chunks; else 0",
            "recall_at_5": "unique gold pages in the first 5 chunks / unique gold pages",
            "mrr_at_5": "reciprocal first gold chunk rank in the final top 5; 0 if absent",
        },
        "run_manifest": str(args.manifest.resolve()),
        "run_manifest_sha256": file_sha256(args.manifest),
        "evidence_mapping": str(args.evidence.resolve()),
        "evidence_mapping_sha256": file_sha256(args.evidence),
        "raw_output_sha256": {
            architecture: file_sha256(Path(manifest["outputs"][architecture]))
            for architecture in ARCHITECTURES
        },
        "experiment_settings": manifest.get("settings"),
        "outputs": {
            "per_question_title": "Stage 15: Page-Level Retrieval Metrics by Question",
            "per_question_csv": str(args.metrics_output.resolve()),
            "summary_title": "Table R1: Held-out page-level retrieval performance",
            "summary_csv": str(args.summary_output.resolve()),
        },
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.provenance_output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.provenance_output)

    print(provenance["title"])
    print(f"Questions per architecture: {len(expected_ids)}")
    print(f"Per-question results: {args.metrics_output}")
    print(f"Summary table:        {args.summary_output}")
    print(f"Provenance/parameters: {args.provenance_output}")
    print("\nTable R1: Held-out page-level retrieval performance")
    for row in summaries:
        print(
            f"{row['architecture']}: Hit@1={row['hit_at_1']:.3f} "
            f"Hit@3={row['hit_at_3']:.3f} Hit@5={row['hit_at_5']:.3f} "
            f"Recall@5={row['recall_at_5']:.3f} MRR@5={row['mrr_at_5']:.3f}"
        )


if __name__ == "__main__":
    main()
