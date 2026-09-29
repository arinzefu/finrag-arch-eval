from __future__ import annotations
from time import perf_counter
from typing import Any
from src.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.pipelines.base import (
    BasePipeline,
    compact_chunk_for_diagnostics,
    validate_question_record,
)

class RerankedRAGPipeline(BasePipeline):
    architecture = "P3"

    def __init__(self, llm, hybrid_retriever, reranker, *, candidate_k: int = 20, final_k: int = 5, experiment_parameters=None) -> None:
        super().__init__(llm, experiment_parameters=experiment_parameters)
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be > 0.")
        if final_k > candidate_k:
            raise ValueError("final_k cannot exceed candidate_k.")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_k = int(candidate_k)
        self.final_k = int(final_k)

    def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        validate_question_record(question)
        total_start = perf_counter()
        query = str(question["question"])

        start = perf_counter()
        if hasattr(self.hybrid_retriever, "retrieve_with_diagnostics"):
            candidates, diagnostics = self.hybrid_retriever.retrieve_with_diagnostics(query, k=self.candidate_k)
        else:
            candidates = self.hybrid_retriever.retrieve(query, k=self.candidate_k)
            diagnostics = {}
        retrieval_ms = (perf_counter() - start) * 1000.0

        reranker_query = str(
            diagnostics.get("processed_retrieval_query") or query
        )
        start = perf_counter()
        reranked = self.reranker.rerank(
            reranker_query, candidates, top_k=len(candidates)
        ) if candidates else []
        chunks = reranked[:self.final_k]
        reranking_ms = (perf_counter() - start) * 1000.0

        prompt = build_rag_prompt(query, chunks)
        llm_response, generation_ms = self._timed_generate(
            self.llm, system_prompt=SYSTEM_PROMPT, user_prompt=prompt
        )
        total_ms = (perf_counter() - total_start) * 1000.0

        result = self._base_result(
            question=question, architecture=self.architecture,
            generated_answer=llm_response.text, retrieved_chunks=chunks,
            retrieval_latency_ms=retrieval_ms, reranking_latency_ms=reranking_ms,
            generation_latency_ms=generation_ms, total_latency_ms=total_ms,
            llm_response=llm_response,
        )
        result["candidate_count_before_reranking"] = len(candidates)
        def compact(rows):
            return [compact_chunk_for_diagnostics(row) for row in rows]
        result["retrieval_diagnostics"] = {
            **diagnostics,
            "reranker_query": reranker_query,
            "first_stage_candidates": compact(candidates),
            "reranked_candidates": compact(reranked),
            "final_candidates": compact(chunks),
        }
        return result
