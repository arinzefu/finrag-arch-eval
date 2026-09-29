from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.retrieval.base import BaseRetriever
from src.retrieval.query_processor import (
    chunk_matches_query_filters,
    company_key,
    process_query,
)

DEFAULT_QUERY_PREFIX = "query: "


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required for dense retrieval.") from exc
    return faiss


def _load_sentence_transformer(model_name: str, device: Optional[str] = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for dense retrieval.") from exc

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        except RuntimeError as exc:
            last_exc = exc
            if "client has been closed" not in str(exc).lower() or attempt == 2:
                break
            _reset_huggingface_http_client()
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(
        "Failed to load the SentenceTransformer model. Retry the run, verify "
        "the model name in dense metadata, and ensure the model can be "
        "downloaded or is cached locally."
    ) from last_exc


def _reset_huggingface_http_client() -> None:
    try:
        from huggingface_hub.utils import _http
    except Exception:
        return

    close_session = getattr(_http, "close_session", None)
    if callable(close_session):
        close_session()


class DenseRetriever(BaseRetriever):
    """Load the persisted Stage 7 normalized FAISS index and metadata."""

    def __init__(
        self,
        *,
        index_path: str | Path,
        metadata_path: str | Path,
        device: Optional[str] = None,
    ) -> None:
        index_path, metadata_path = Path(index_path), Path(metadata_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dense metadata not found: {metadata_path}")

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "chunks" not in payload or "model_name" not in payload:
            raise ValueError("Dense metadata must contain 'chunks' and 'model_name'.")
        if payload.get("schema_version") != 2 or payload.get("retrieval_text_schema") != "document_header_v2":
            raise ValueError("Outdated dense index; recreate chunks and rebuild the dense index.")

        self.chunks = list(payload["chunks"])
        self.companies = {str(c["company"]) for c in self.chunks if c.get("company")}
        self.company_indices: dict[str, set[int]] = {}
        self.year_indices: dict[int, set[int]] = {}
        for idx, chunk in enumerate(self.chunks):
            if chunk.get("company"):
                self.company_indices.setdefault(company_key(chunk["company"]), set()).add(idx)
            if chunk.get("fiscal_year") is not None:
                self.year_indices.setdefault(int(chunk["fiscal_year"]), set()).add(idx)
        self.query_prefix = str(payload.get("query_prefix", DEFAULT_QUERY_PREFIX))
        self.faiss = _import_faiss()
        self.index = self.faiss.read_index(str(index_path))
        self.model = _load_sentence_transformer(str(payload["model_name"]), device=device)

        if int(self.index.ntotal) != len(self.chunks):
            raise RuntimeError(
                f"FAISS/metadata mismatch: {self.index.ntotal} vectors vs {len(self.chunks)} chunks."
            )

        expected_dim = payload.get("embedding_dimension")
        index_dim = getattr(self.index, "d", None)
        if expected_dim is not None and index_dim is not None and int(expected_dim) != int(index_dim):
            raise RuntimeError(
                f"Embedding dimension mismatch: metadata={expected_dim}, index={index_dim}."
            )

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query = str(query).strip()
        if not query:
            raise ValueError("query must not be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        processed = process_query(query, companies=self.companies)
        allowed: set[int] | None = None
        if processed.company:
            allowed = set(self.company_indices.get(processed.company, set()))
        if processed.fiscal_year is not None:
            year_allowed = set(self.year_indices.get(processed.fiscal_year, set()))
            allowed = year_allowed if allowed is None else allowed & year_allowed
        if allowed is not None and not allowed:
            return []
        k = min(int(k), len(allowed) if allowed is not None else len(self.chunks))
        embedding = self.model.encode(
            [self.query_prefix + processed.retrieval_query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding = np.asarray(embedding, dtype=np.float32, order="C")
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise RuntimeError(f"Unexpected query embedding shape: {embedding.shape}")
        index_dim = getattr(self.index, "d", None)
        if index_dim is not None and int(index_dim) != int(embedding.shape[1]):
            raise RuntimeError(
                f"Query embedding dimension {embedding.shape[1]} does not match FAISS index dimension {index_dim}."
            )

        self.faiss.normalize_L2(embedding)
        # IndexFlatIP returns globally sorted exact scores. Widen until k
        # matching-company results are found; a fixed top-50 can miss them.
        search_k = min(len(self.chunks), max(k, 50)) if allowed is not None else k
        while True:
            scores, indices = self.index.search(embedding, search_k)
            if (
                allowed is None
                or sum(int(i) in allowed for i in indices[0] if int(i) >= 0) >= k
                or search_k == len(self.chunks)
            ):
                break
            search_k = min(len(self.chunks), search_k * 2)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx < 0:
                continue
            if idx >= len(self.chunks):
                raise RuntimeError(f"FAISS returned invalid index {idx}.")
            if allowed is not None and idx not in allowed:
                continue
            if not chunk_matches_query_filters(self.chunks[idx], processed):
                continue
            rank = len(results) + 1
            result = dict(self.chunks[idx])
            result.update({
                "rank": rank,
                "score": float(score),
                "dense_score": float(score),
                "dense_rank": rank,
                "retrieval_method": "dense",
                "vector_index": idx,
            })
            results.append(result)
            if len(results) == k:
                break
        return results
