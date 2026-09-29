from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from src.generation.llm import BaseLLM, LLMResponse


def validate_question_record(question: dict[str, Any]) -> None:
    required = {"financebench_id", "question", "answer"}
    missing = required - set(question)
    if missing:
        raise ValueError("Question record missing: " + ", ".join(sorted(missing)))
    if not str(question["question"]).strip():
        raise ValueError("Question text must not be empty.")


def slim_chunk_for_result(chunk: dict[str, Any]) -> dict[str, Any]:
    preferred = [
        "chunk_id", "doc_name", "page", "company", "fiscal_year",
        "document_type", "text", "rank", "score",
        "dense_score", "dense_rank", "sparse_score", "sparse_rank",
        "rrf_score", "reranker_score", "first_stage_score", "retrieval_method",
        "reranker_input_tokens", "reranker_input_truncated",
        "reranker_window_count", "reranker_best_window_index",
        "reranker_best_window_tokens",
    ]
    output = {key: chunk[key] for key in preferred if key in chunk}
    for key in ("component_scores", "component_ranks"):
        if key in chunk:
            output[key] = chunk[key]
    return output


def compact_chunk_for_diagnostics(chunk: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "chunk_id", "doc_name", "page", "company", "fiscal_year", "rank",
        "dense_rank", "dense_score", "sparse_rank", "sparse_score",
        "rrf_rank", "rrf_score", "reranker_rank", "reranker_score",
        "reranker_input_tokens", "reranker_input_truncated",
        "reranker_window_count", "reranker_best_window_index",
        "reranker_best_window_tokens",
    )
    return {key: chunk[key] for key in fields if key in chunk}


class BasePipeline(ABC):
    architecture: str

    def __init__(
        self,
        llm: BaseLLM,
        *,
        experiment_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.experiment_parameters = dict(experiment_parameters or {})

    @abstractmethod
    def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _timed_generate(
        llm: BaseLLM,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[LLMResponse, float]:
        start = perf_counter()
        response = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        return response, (perf_counter() - start) * 1000.0

    def _base_result(
        self,
        *,
        question: dict[str, Any],
        architecture: str,
        generated_answer: str,
        retrieved_chunks: list[dict[str, Any]],
        retrieval_latency_ms: float | None,
        reranking_latency_ms: float | None,
        generation_latency_ms: float,
        total_latency_ms: float,
        llm_response: LLMResponse,
    ) -> dict[str, Any]:
        result = {
            "question_id": str(question["financebench_id"]),
            "architecture": architecture,
            "question": str(question["question"]),
            "gold_answer": str(question["answer"]),
            "generated_answer": generated_answer,
            "retrieved_chunks": [slim_chunk_for_result(x) for x in retrieved_chunks],
            "retrieval_latency_ms": retrieval_latency_ms,
            "reranking_latency_ms": reranking_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "llm_provider": llm_response.provider,
            "llm_model": llm_response.model,
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "total_tokens": llm_response.total_tokens,
        }
        generation_parameters = {
            "model": getattr(self.llm, "model", llm_response.model),
            "max_output_tokens": getattr(self.llm, "max_output_tokens", None),
            "temperature": getattr(self.llm, "temperature", None),
        }
        result["generation_parameters"] = generation_parameters
        if self.experiment_parameters:
            result["experiment_parameters"] = dict(self.experiment_parameters)
        return result
