from __future__ import annotations
from src.retrieval.base import BaseRetriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.query_processor import process_query

class HybridRetriever(BaseRetriever):
    def __init__(self, dense_retriever: BaseRetriever, sparse_retriever: BaseRetriever, fusion: ReciprocalRankFusion, *, dense_k: int = 10, sparse_k: int = 10) -> None:
        if dense_k <= 0 or sparse_k <= 0:
            raise ValueError("dense_k and sparse_k must be > 0.")
        self.dense, self.sparse, self.fusion = dense_retriever, sparse_retriever, fusion
        self.dense_k, self.sparse_k = int(dense_k), int(sparse_k)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        results, _ = self.retrieve_with_diagnostics(query, k=k)
        return results

    def retrieve_with_diagnostics(self, query: str, k: int = 5) -> tuple[list[dict], dict]:
        if k <= 0:
            raise ValueError("k must be > 0.")
        companies = set(getattr(self.dense, "companies", set())) | set(getattr(self.sparse, "companies", set()))
        processed = process_query(query, companies=companies)
        dense_results = self.dense.retrieve(query, k=self.dense_k)
        sparse_results = self.sparse.retrieve(query, k=self.sparse_k)
        fused = self.fusion.fuse(dense_results, sparse_results, source_names=("dense", "sparse"))
        for rank, row in enumerate(fused, start=1):
            row.setdefault("rrf_rank", rank)
            if "score" in row:
                row.setdefault("rrf_score", row["score"])
        def compact(rows: list[dict]) -> list[dict]:
            fields = ("chunk_id", "doc_name", "page", "company", "fiscal_year", "rank",
                      "score", "dense_rank", "dense_score", "sparse_rank", "sparse_score",
                      "rrf_rank", "rrf_score")
            return [{key: row[key] for key in fields if key in row} for row in rows]
        diagnostics = {
            "processed_retrieval_query": processed.retrieval_query,
            "query_filters": {
                "company": processed.company,
                "fiscal_year": processed.fiscal_year,
            },
            "dense_candidates": compact(dense_results),
            "sparse_candidates": compact(sparse_results),
            "rrf_union": compact(fused),
            "rrf_top_candidates": compact(fused[:int(k)]),
        }
        return fused[:int(k)], diagnostics
