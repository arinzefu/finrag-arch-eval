#!/usr/bin/env python
"""
Manual Stage 7 retrieval inspection.

Prints FinanceBench questions, their benchmark gold page references, and the
retriever's top-k global chunks. This is deliberately a qualitative smoke test,
not the final retrieval-metric implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.dense import DenseRetriever  # noqa: E402


DEFAULT_SNAPSHOT = ROOT / "data" / "raw" / "financebench" / "questions.jsonl"
DEFAULT_INDEX = ROOT / "data" / "indices" / "dense" / "faiss.index"
DEFAULT_METADATA = ROOT / "data" / "indices" / "dense" / "metadata.json"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions = read_jsonl(args.snapshot)
    if not questions:
        raise RuntimeError("No FinanceBench questions loaded.")

    n = min(args.n, len(questions))
    if n <= 0:
        raise ValueError("--n must be > 0.")

    # Deterministic spread across the benchmark rather than only first rows.
    if n == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            round(i * (len(questions) - 1) / (n - 1))
            for i in range(n)
        ]

    retriever = DenseRetriever(
        index_path=args.index,
        metadata_path=args.metadata,
        device=args.device,
    )

    for ordinal, question_index in enumerate(selected_indices, start=1):
        row = questions[question_index]
        print("\n" + "=" * 100)
        print(f"[{ordinal}/{n}] {row['financebench_id']}")
        print("QUESTION:")
        print(row["question"])

        print("\nGOLD DOCUMENT/PAGES:")
        evidence = row.get("evidence") or []
        gold_refs = list(
            dict.fromkeys(
                (
                    str(item["evidence_doc_name"]),
                    int(item["evidence_page_num"]),
                )
                for item in evidence
            )
        )
        for doc_name, page in gold_refs:
            print(f"  {doc_name} | zero-indexed page {page}")

        print(f"\nTOP-{args.k} DENSE RESULTS:")
        results = retriever.retrieve(row["question"], k=args.k)

        for result in results:
            snippet = " ".join(str(result["text"]).split())[:350]
            is_gold_page = (
                str(result["doc_name"]),
                int(result["page"]),
            ) in gold_refs

            marker = " <-- GOLD PAGE" if is_gold_page else ""
            print(
                f"\n  #{result['rank']} "
                f"score={result['score']:.4f} "
                f"{result['doc_name']} "
                f"p{result['page']} "
                f"{result['chunk_id']}{marker}"
            )
            print(f"     {snippet}")

    print("\n" + "=" * 100)
    print(
        "Manual smoke test complete. Inspect whether retrieved evidence is "
        "financially/question-relevant; quantitative Recall@k comes later."
    )


if __name__ == "__main__":
    main()
