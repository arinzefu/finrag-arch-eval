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

class DenseRAGPipeline(BasePipeline):
    architecture = "P1"

    def __init__(
        self,
        llm,
        retriever: BaseRetriever,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        experiment_parameters=None,
    ) -> None:
        super().__init__(llm, experiment_parameters=experiment_parameters)
        candidate_k = top_k if candidate_k is None else candidate_k
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be > 0.")
        if top_k > candidate_k:
            raise ValueError("top_k cannot exceed candidate_k.")
        self.retriever = retriever
        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k)

    def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        validate_question_record(question)
        total_start = perf_counter()
        start = perf_counter()
        candidates = self.retriever.retrieve(
            str(question["question"]), k=self.candidate_k
        )
        chunks = candidates[:self.top_k]
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
            "first_stage_candidates": [
                compact_chunk_for_diagnostics(row) for row in candidates
            ],
            "final_candidates": [
                compact_chunk_for_diagnostics(row) for row in chunks
            ],
        }
        return result
