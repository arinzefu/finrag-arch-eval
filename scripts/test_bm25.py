#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.indexing.bm25_index import load_chunks_jsonl
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever

def show(title, results):
    print("\n"+title+"\n"+"-"*len(title))
    for r in results:
        snippet = " ".join(str(r["text"]).split())[:300]
        print(f"#{r['rank']} score={r['score']:.6f} {r['doc_name']} p{r['page']} {r['chunk_id']}")
        print(" ", snippet)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="capital expenditure FY2018 3M")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    chunks = load_chunks_jsonl(ROOT/"data/processed/chunks.jsonl")
    sparse = BM25Retriever.from_artifacts(
        index_path=ROOT/"data/indices/bm25/bm25.pkl",
        metadata_path=ROOT/"data/indices/bm25/metadata.json",
        chunks=chunks,
    )
    dense = DenseRetriever(
        index_path=ROOT/"data/indices/dense/faiss.index",
        metadata_path=ROOT/"data/indices/dense/metadata.json",
        device=args.device,
    )
    print("Query:", args.query)
    show(f"Top {args.k} BM25", sparse.retrieve(args.query, k=args.k))
    show(f"Top {args.k} Dense", dense.retrieve(args.query, k=args.k))
if __name__ == "__main__":
    main()
