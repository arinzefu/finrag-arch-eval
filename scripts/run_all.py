#!/usr/bin/env python
"""Run the controlled P0-P3 comparison over one fixed question set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import load_questions, parse_temperature  # noqa: E402
from src.generation.llm import OpenAIResponsesLLM  # noqa: E402
from src.pipelines.factory import create_pipeline  # noqa: E402
from src.pipelines.settings import DEFAULT_EXPERIMENT_SETTINGS  # noqa: E402


ARCHITECTURES = ("P0", "P1", "P2", "P3")
DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "financebench" / "questions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P0-P3 with one controlled parameter set."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "raw_runs"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", default="none")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError("Supply --model MODEL or set OPENAI_MODEL.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0.")
    run_id = str(args.run_id).strip()
    if not run_id or any(char in run_id for char in '<>:"/\\|?*'):
        raise ValueError("--run-id must be a non-empty filename-safe value.")

    questions = load_questions(args.questions)
    selected = questions if args.limit is None else questions[:args.limit]
    outputs = {
        architecture: args.output_dir / f"{run_id}_{architecture}.jsonl"
        for architecture in ARCHITECTURES
    }
    manifest_path = args.output_dir / f"{run_id}_manifest.json"
    existing = [path for path in [*outputs.values(), manifest_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Controlled run outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temperature = parse_temperature(args.temperature)
    llm = OpenAIResponsesLLM(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=temperature,
    )

    for architecture in ARCHITECTURES:
        print(f"\nRunning {architecture} over {len(selected)} fixed questions...")
        pipeline = create_pipeline(
            architecture,
            project_root=ROOT,
            llm=llm,
            device=args.device,
            settings=DEFAULT_EXPERIMENT_SETTINGS,
        )
        rows = []
        for question in selected:
            result = pipeline.answer(question)
            rows.append(result)
        write_jsonl_atomic(rows, outputs[architecture])

    question_bytes = args.questions.read_bytes()
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architectures": list(ARCHITECTURES),
        "question_source": str(args.questions.resolve()),
        "question_source_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "question_ids": [str(row["financebench_id"]) for row in selected],
        "question_count": len(selected),
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "temperature": temperature,
        "device": args.device,
        "settings": asdict(DEFAULT_EXPERIMENT_SETTINGS),
        "latency_scope": "per_question_excludes_pipeline_and_model_initialization",
        "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nControlled run complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
