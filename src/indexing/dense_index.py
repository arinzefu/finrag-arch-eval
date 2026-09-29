"""
Build and persist an exact cosine-similarity FAISS index over FinanceBench chunks.

Default model
-------------
intfloat/e5-base-v2

For E5 asymmetric retrieval:
* documents are encoded as ``passage: <chunk text>``
* questions are encoded as ``query: <question>``

Embeddings are L2-normalized. FAISS ``IndexFlatIP`` therefore returns inner
products equivalent to cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_PASSAGE_PREFIX = "passage: "
DEFAULT_QUERY_PREFIX = "query: "


def load_chunks_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            required = {"chunk_id", "doc_name", "page", "text"}
            missing = required - set(chunk)
            if missing:
                raise ValueError(
                    f"Chunk line {line_number} is missing: "
                    + ", ".join(sorted(missing))
                )

            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            seen_ids.add(chunk_id)

            if not str(chunk["text"]).strip():
                raise ValueError(
                    f"Empty chunk text found for {chunk_id}."
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("No chunks were loaded.")

    return chunks


def _load_sentence_transformer(
    model_name: str,
    device: Optional[str] = None,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for Stage 7. "
            "Install requirements_stage4_7.txt."
        ) from exc

    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for Stage 7. "
            "Install requirements_stage4_7.txt."
        ) from exc
    return faiss


def build_dense_index(
    *,
    chunks: list[dict[str, Any]],
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Encode chunks and save an exact normalized inner-product FAISS index.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    # Ensure Stage 5 and Stage 7 use the same tokenizer/model family.
    tokenizer_names = {
        str(chunk.get("tokenizer_name"))
        for chunk in chunks
        if chunk.get("tokenizer_name") is not None
    }
    if tokenizer_names and tokenizer_names != {model_name}:
        raise ValueError(
            "Chunk tokenizer/model does not match dense embedding model. "
            f"Chunks use {sorted(tokenizer_names)}, Stage 7 requested "
            f"{model_name!r}. Recreate chunks or use the matching model."
        )

    prefix_values = {
        str(chunk.get("passage_prefix"))
        for chunk in chunks
        if chunk.get("passage_prefix") is not None
    }
    if prefix_values and prefix_values != {passage_prefix}:
        raise ValueError(
            "Chunk passage_prefix metadata does not match Stage 7: "
            f"{sorted(prefix_values)} vs {passage_prefix!r}"
        )

    model = _load_sentence_transformer(model_name, device=device)
    faiss = _import_faiss()

    if any(not str(chunk.get("retrieval_text") or "").strip() for chunk in chunks):
        raise ValueError("Chunks lack retrieval_text; rerun scripts/create_chunks.py.")
    texts = [str(chunk["retrieval_text"]) for chunk in chunks]

    # The chunker records the exact embedding input length, including header.
    if any(int(chunk.get("model_input_tokens_estimate", 10**9)) > int(chunk.get("chunk_size", 512)) for chunk in chunks):
        raise ValueError("Retrieval text exceeds the dense model token budget; recreate chunks.")

    dimension: int | None = None
    index = None

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        model_inputs = [
            passage_prefix + text for text in batch_texts
        ]

        embeddings = model.encode(
            model_inputs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
            order="C",
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Unexpected embedding shape: "
                f"{embeddings.shape}; expected a 2D array."
            )
        batch_dimension = int(embeddings.shape[1])
        if dimension is None:
            dimension = batch_dimension
            index = faiss.IndexFlatIP(dimension)
        elif batch_dimension != dimension:
            raise RuntimeError(
                "Embedding dimension changed between batches: "
                f"{batch_dimension} != {dimension}."
            )

        # SentenceTransformer normalization is requested above. Normalize again
        # through FAISS to remove numerical drift and guarantee cosine/IP
        # equivalence at persisted-index time.
        faiss.normalize_L2(embeddings)
        if index is None:
            raise RuntimeError("Dense index was not initialized.")
        index.add(embeddings)

    if index is None or dimension is None:
        raise RuntimeError("No embeddings were generated.")

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS ntotal={index.ntotal}, chunks={len(chunks)}."
        )

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    metadata = {
        "schema_version": 2,
        "retrieval_text_schema": "document_header_v2",
        "index_type": "IndexFlatIP",
        "metric": "cosine_similarity_via_normalized_inner_product",
        "normalized_embeddings": True,
        "model_name": model_name,
        "embedding_dimension": dimension,
        "passage_prefix": passage_prefix,
        "query_prefix": query_prefix,
        "chunk_count": len(chunks),
        "batch_size": batch_size,
        "chunks": chunks,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "model_name": model_name,
        "embedding_dimension": dimension,
        "chunk_count": len(chunks),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "index_type": "IndexFlatIP",
        "metric": "cosine_similarity_via_normalized_inner_product",
    }
