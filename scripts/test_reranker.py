#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.indexing.bm25_index import load_chunks_jsonl
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.pipelines.settings import DEFAULT_EXPERIMENT_SETTINGS

def load_questions(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--question-index", type=int, default=0)
    p.add_argument("--device", default=None)
    args=p.parse_args()
    questions=load_questions(ROOT/"data/raw/financebench/questions.jsonl")
    if not 0 <= args.question_index < len(questions):
        raise IndexError("question-index outside FinanceBench range.")
    chunks=load_chunks_jsonl(ROOT/"data/processed/chunks.jsonl")
    dense=DenseRetriever(index_path=ROOT/"data/indices/dense/faiss.index", metadata_path=ROOT/"data/indices/dense/metadata.json", device=args.device)
    sparse=BM25Retriever.from_artifacts(index_path=ROOT/"data/indices/bm25/bm25.pkl", metadata_path=ROOT/"data/indices/bm25/metadata.json", chunks=chunks)
    settings=DEFAULT_EXPERIMENT_SETTINGS
    hybrid=HybridRetriever(
        dense,sparse,ReciprocalRankFusion(settings.rrf_k),
        dense_k=settings.p3_dense_k,sparse_k=settings.p3_sparse_k,
    )
    query=questions[args.question_index]["question"]
    candidates,diagnostics=hybrid.retrieve_with_diagnostics(
        query,k=settings.p3_fusion_k
    )
    reranker_query=diagnostics["processed_retrieval_query"]
    reranker=CrossEncoderReranker.from_pretrained(
        device=args.device,max_length=settings.reranker_max_length,
        batch_size=settings.reranker_batch_size,
        window_overlap=settings.reranker_window_overlap,
    )
    final=reranker.rerank(
        reranker_query,candidates,top_k=settings.final_context_k
    )
    print("QUESTION:\n",query)
    print("\nRERANKER QUERY:\n",reranker_query)
    print(f"\nHybrid candidates: {len(candidates)}")
    print(f"Reranked results: {len(final)}")
    if len(final) != min(settings.final_context_k,len(candidates)):
        raise RuntimeError("Unexpected reranked result count.")
    for r in final:
        snippet=" ".join(str(r["text"]).split())[:300]
        print(f"\n#{r['rank']} reranker={r['reranker_score']:.6f} first_stage={r.get('first_stage_score')} {r['doc_name']} p{r['page']}")
        print(" ",snippet)
if __name__=="__main__":
    main()
