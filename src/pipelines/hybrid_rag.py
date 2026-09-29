from __future__ import annotations
from time import perf_counter
from typing import Any
from src.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.pipelines.base import (
    BasePipeline,
    compact_chunk_for_diagnostics,
    validate_question_record,
)
from src.retrieval.base import BaseRetriever

class HybridRAGPipeline(BasePipeline):
    architecture = "P2"

    def __init__(
        self,
        llm,
        retriever: BaseRetriever,
        *,
        candidate_k: int | None = None,
        final_k: int = 5,
        experiment_parameters=None,
    ) -> None:
        super().__init__(llm, experiment_parameters=experiment_parameters)
        candidate_k = final_k if candidate_k is None else candidate_k
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be > 0.")
        if final_k > candidate_k:
            raise ValueError("final_k cannot exceed candidate_k.")
        self.retriever = retriever
        self.candidate_k = int(candidate_k)
        self.final_k = int(final_k)

    def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        validate_question_record(question)
        total_start = perf_counter()
        start = perf_counter()
        query = str(question["question"])
        if hasattr(self.retriever, "retrieve_with_diagnostics"):
            candidates, diagnostics = self.retriever.retrieve_with_diagnostics(
                query, k=self.candidate_k
            )
        else:
            candidates = self.retriever.retrieve(query, k=self.candidate_k)
            diagnostics = {}
        chunks = candidates[:self.final_k]
        retrieval_ms = (perf_counter() - start) * 1000.0
        prompt = build_rag_prompt(question["question"], chunks)
        llm_response, generation_ms = self._timed_generate(
            self.llm, system_prompt=SYSTEM_PROMPT, user_prompt=prompt
        )
        total_ms = (perf_counter() - total_start) * 1000.0
        result = self._base_result(
            question=question, architecture=self.architecture,
            generated_answer=llm_response.text, retrieved_chunks=chunks,
            retrieval_latency_ms=retrieval_ms, reranking_latency_ms=None,
            generation_latency_ms=generation_ms, total_latency_ms=total_ms,
            llm_response=llm_response,
        )
        result["candidate_count_before_selection"] = len(candidates)
        result["retrieval_diagnostics"] = {
            **diagnostics,
            "first_stage_candidates": [
                compact_chunk_for_diagnostics(row) for row in candidates
            ],
            "final_candidates": [
                compact_chunk_for_diagnostics(row) for row in chunks
            ],
        }
        return result
