from __future__ import annotations
from pathlib import Path
from src.generation.llm import BaseLLM
from src.indexing.bm25_index import load_chunks_jsonl
from src.pipelines.closed_book import ClosedBookPipeline
from src.pipelines.dense_rag import DenseRAGPipeline
from src.pipelines.hybrid_rag import HybridRAGPipeline
from src.pipelines.reranked_rag import RerankedRAGPipeline
from src.pipelines.settings import DEFAULT_EXPERIMENT_SETTINGS, ExperimentSettings
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL

def create_pipeline(
    architecture: str,
    *,
    project_root: str | Path,
    llm: BaseLLM,
    device: str | None = None,
    settings: ExperimentSettings | None = None,
):
    root = Path(project_root)
    architecture = str(architecture).upper().strip()
    if architecture not in {"P0", "P1", "P2", "P3"}:
        raise ValueError(
            f"Unknown architecture {architecture!r}; use P0, P1, P2, or P3."
        )
    settings = settings or DEFAULT_EXPERIMENT_SETTINGS
    experiment_parameters = settings.result_metadata(architecture)

    if architecture == "P0":
        return ClosedBookPipeline(
            llm, experiment_parameters=experiment_parameters
        )

    dense = DenseRetriever(
        index_path=root / "data" / "indices" / "dense" / "faiss.index",
        metadata_path=root / "data" / "indices" / "dense" / "metadata.json",
        device=device,
    )

    if architecture == "P1":
        return DenseRAGPipeline(
            llm=llm,
            retriever=dense,
            candidate_k=settings.p1_dense_k,
            top_k=settings.final_context_k,
            experiment_parameters=experiment_parameters,
        )

    chunks = load_chunks_jsonl(root / "data" / "processed" / "chunks.jsonl")
    sparse = BM25Retriever.from_artifacts(
        index_path=root / "data" / "indices" / "bm25" / "bm25.pkl",
        metadata_path=root / "data" / "indices" / "bm25" / "metadata.json",
        chunks=chunks,
    )

    if architecture == "P2":
        hybrid = HybridRetriever(
            dense, sparse, ReciprocalRankFusion(k=settings.rrf_k),
            dense_k=settings.p2_dense_k,
            sparse_k=settings.p2_sparse_k,
        )
        return HybridRAGPipeline(
            llm=llm,
            retriever=hybrid,
            candidate_k=settings.p2_fusion_k,
            final_k=settings.final_context_k,
            experiment_parameters=experiment_parameters,
        )

    if architecture == "P3":
        hybrid = HybridRetriever(
            dense, sparse, ReciprocalRankFusion(k=settings.rrf_k),
            dense_k=settings.p3_dense_k,
            sparse_k=settings.p3_sparse_k,
        )
        reranker = CrossEncoderReranker.from_pretrained(
            DEFAULT_RERANKER_MODEL,
            device=device,
            max_length=settings.reranker_max_length,
            batch_size=settings.reranker_batch_size,
            window_overlap=settings.reranker_window_overlap,
        )
        return RerankedRAGPipeline(
            llm=llm, hybrid_retriever=hybrid, reranker=reranker,
            candidate_k=settings.p3_fusion_k,
            final_k=settings.final_context_k,
            experiment_parameters=experiment_parameters,
        )
