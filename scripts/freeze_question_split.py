#!/usr/bin/env python
"""Freeze development-smoke and held-out FinanceBench question files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "financebench" / "questions.jsonl"
DEFAULT_CONFIG = ROOT / "configs" / "question_split.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "splits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path: Path) -> list[dict]:
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
        raise ValueError(f"No questions loaded from {path}.")
    return rows


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_jsonl(rows: list[dict], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the development and held-out evaluation split."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    development_ids = [str(value) for value in config["development_question_ids"]]
    if len(development_ids) != len(set(development_ids)):
        raise ValueError("Duplicate development question ID in split config.")

    questions = load_questions(args.source)
    all_ids = [str(row["financebench_id"]) for row in questions]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Question source contains duplicate financebench_id values.")
    missing = sorted(set(development_ids) - set(all_ids))
    if missing:
        raise ValueError("Development IDs missing from source: " + ", ".join(missing))

    development_set = set(development_ids)
    development = [
        row for row in questions if str(row["financebench_id"]) in development_set
    ]
    evaluation = [
        row for row in questions if str(row["financebench_id"]) not in development_set
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_path = args.output_dir / "development_smoke.jsonl"
    evaluation_path = args.output_dir / "common_evaluation.jsonl"
    manifest_path = args.output_dir / "question_split_manifest.json"
    existing = [
        path for path in (development_path, evaluation_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Frozen split output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    write_jsonl(development, development_path)
    write_jsonl(evaluation, evaluation_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": config["policy"],
        "source_path": project_relative(args.source),
        "source_sha256": sha256_file(args.source),
        "source_question_count": len(questions),
        "development_path": project_relative(development_path),
        "development_sha256": sha256_file(development_path),
        "development_question_ids": [
            str(row["financebench_id"]) for row in development
        ],
        "development_question_count": len(development),
        "evaluation_path": project_relative(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "evaluation_question_ids": [
            str(row["financebench_id"]) for row in evaluation
        ],
        "evaluation_question_count": len(evaluation),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Development questions: {len(development)} -> {development_path}")
    print(f"Evaluation questions:  {len(evaluation)} -> {evaluation_path}")
    print(f"Manifest:              {manifest_path}")


if __name__ == "__main__":
    main()
