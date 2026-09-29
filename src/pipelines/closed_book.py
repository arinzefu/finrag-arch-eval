from __future__ import annotations
from time import perf_counter
from typing import Any
from src.generation.prompts import SYSTEM_PROMPT, build_closed_book_prompt
from src.pipelines.base import BasePipeline, validate_question_record

class ClosedBookPipeline(BasePipeline):
    architecture = "P0"

    def __init__(self, llm, *, experiment_parameters=None) -> None:
        super().__init__(llm, experiment_parameters=experiment_parameters)

    def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        validate_question_record(question)
        total_start = perf_counter()
        prompt = build_closed_book_prompt(question["question"])
        llm_response, generation_ms = self._timed_generate(
            self.llm, system_prompt=SYSTEM_PROMPT, user_prompt=prompt
        )
        total_ms = (perf_counter() - total_start) * 1000.0
        return self._base_result(
            question=question, architecture=self.architecture,
            generated_answer=llm_response.text, retrieved_chunks=[],
            retrieval_latency_ms=None, reranking_latency_ms=None,
            generation_latency_ms=generation_ms, total_latency_ms=total_ms,
            llm_response=llm_response,
        )
