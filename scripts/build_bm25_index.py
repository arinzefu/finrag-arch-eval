#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.indexing.bm25_index import load_chunks_jsonl, build_bm25_index

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=Path, default=ROOT/"data/processed/chunks.jsonl")
    p.add_argument("--index", type=Path, default=ROOT/"data/indices/bm25/bm25.pkl")
    p.add_argument("--metadata", type=Path, default=ROOT/"data/indices/bm25/metadata.json")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--epsilon", type=float, default=0.25)
    args = p.parse_args()
    chunks = load_chunks_jsonl(args.chunks)
    metadata = build_bm25_index(
        chunks=chunks, index_path=args.index, metadata_path=args.metadata,
        k1=args.k1, b=args.b, epsilon=args.epsilon
    )
    print(f"BM25 chunks indexed: {len(chunks)}")
    print(f"Index: {args.index}")
    print(f"Metadata: {args.metadata}")
    print(json.dumps(metadata, indent=2))
if __name__ == "__main__":
    main()
