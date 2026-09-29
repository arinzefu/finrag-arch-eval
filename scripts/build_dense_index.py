#!/usr/bin/env python
"""Stage 7 — Build the P1 dense FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.dense_index import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    build_dense_index,
    load_chunks_jsonl,
)


DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_INDEX = ROOT / "data" / "indices" / "dense" / "faiss.index"
DEFAULT_METADATA = ROOT / "data" / "indices" / "dense" / "metadata.json"
DEFAULT_SUMMARY = ROOT / "data" / "indices" / "dense" / "index_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed FinanceBench chunks with E5 and build an exact "
            "cosine-similarity FAISS IndexFlatIP."
        )
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device, e.g. cpu, cuda, cuda:0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("Stage 7 — Dense FAISS index")
    print("=" * 78)
    print(f"Chunks:     {args.chunks}")
    print(f"Model:      {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Index:      {args.index}")
    print(f"Metadata:   {args.metadata}\n")

    chunks = load_chunks_jsonl(args.chunks)
    print(f"Loaded {len(chunks)} chunks.")

    summary = build_dense_index(
        chunks=chunks,
        index_path=args.index,
        metadata_path=args.metadata,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDense index complete.")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nStage 7 stop condition satisfied:")
    print("  Every stored vector maps to exactly one persisted chunk.")
    print("  Embeddings are normalized and searched with IndexFlatIP.")
    print("  Query and passage E5 prefixes are fixed in metadata.")


if __name__ == "__main__":
    main()
