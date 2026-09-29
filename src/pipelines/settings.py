from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentSettings:
    """Frozen architecture settings for the P0-P3 comparison."""

    p1_dense_k: int = 5
    p2_dense_k: int = 10
    p2_sparse_k: int = 10
    p2_fusion_k: int = 5
    p3_dense_k: int = 20
    p3_sparse_k: int = 20
    p3_fusion_k: int = 20
    final_context_k: int = 5
    rrf_k: int = 60
    reranker_max_length: int = 512
    reranker_window_overlap: int = 64
    reranker_batch_size: int = 32

    def __post_init__(self) -> None:
        positive = {
            "p1_dense_k": self.p1_dense_k,
            "p2_dense_k": self.p2_dense_k,
            "p2_sparse_k": self.p2_sparse_k,
            "p2_fusion_k": self.p2_fusion_k,
            "p3_dense_k": self.p3_dense_k,
            "p3_sparse_k": self.p3_sparse_k,
            "p3_fusion_k": self.p3_fusion_k,
            "final_context_k": self.final_context_k,
            "rrf_k": self.rrf_k,
            "reranker_max_length": self.reranker_max_length,
            "reranker_batch_size": self.reranker_batch_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.final_context_k > self.p1_dense_k:
            raise ValueError("final_context_k cannot exceed p1_dense_k.")
        if self.final_context_k > self.p2_fusion_k:
            raise ValueError("final_context_k cannot exceed p2_fusion_k.")
        if self.final_context_k > self.p3_fusion_k:
            raise ValueError("final_context_k cannot exceed p3_fusion_k.")
        if self.reranker_window_overlap < 0:
            raise ValueError("reranker_window_overlap must be >= 0.")

    def result_metadata(self, architecture: str) -> dict[str, Any]:
        architecture = str(architecture).upper().strip()
        metadata: dict[str, Any] = {
            "protocol": "controlled_architecture_comparison_v1",
            "architecture": architecture,
            "final_context_k": self.final_context_k if architecture != "P0" else 0,
        }
        if architecture != "P0":
            metadata.update(
                {
                    "query_filter_policy": (
                        "question_only_company_and_unambiguous_year"
                    ),
                    "query_expansion_policy": "deterministic_financial_query_v2",
                    "generation_context_policy": "canonical_chunk_text",
                }
            )
        if architecture == "P1":
            metadata["dense_candidate_k"] = self.p1_dense_k
        elif architecture == "P2":
            metadata.update(
                {
                    "dense_candidate_k": self.p2_dense_k,
                    "sparse_candidate_k": self.p2_sparse_k,
                    "fusion_candidate_k": self.p2_fusion_k,
                    "rrf_k": self.rrf_k,
                }
            )
        elif architecture == "P3":
            metadata.update(
                {
                    "dense_candidate_k": self.p3_dense_k,
                    "sparse_candidate_k": self.p3_sparse_k,
                    "fusion_candidate_k": self.p3_fusion_k,
                    "rrf_k": self.rrf_k,
                    "reranker_candidate_k": self.p3_fusion_k,
                    "reranker_max_length": self.reranker_max_length,
                    "reranker_window_overlap": self.reranker_window_overlap,
                    "reranker_batch_size": self.reranker_batch_size,
                    "reranker_query_policy": (
                        "same_processed_query_as_first_stage"
                    ),
                }
            )
        return metadata


DEFAULT_EXPERIMENT_SETTINGS = ExperimentSettings()
