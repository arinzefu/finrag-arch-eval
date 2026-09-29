from __future__ import annotations
import hashlib, json, pickle
from pathlib import Path
from typing import Any
from src.retrieval.sparse import tokenize_financial_text, TOKENIZER_ID

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_EPSILON = 0.25

def load_chunks_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    chunks, seen = [], set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            required = {"chunk_id", "doc_name", "page", "text"}
            missing = required - set(chunk)
            if missing:
                raise ValueError(f"Chunk line {line_no} missing: {', '.join(sorted(missing))}")
            cid = str(chunk["chunk_id"])
            if cid in seen:
                raise ValueError(f"Duplicate chunk_id: {cid}")
            seen.add(cid)
            if not str(chunk["text"]).strip():
                raise ValueError(f"Empty chunk text: {cid}")
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No chunks loaded.")
    return chunks

def build_bm25_index(
    *,
    chunks: list[dict[str, Any]],
    index_path: str | Path,
    metadata_path: str | Path,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("Install rank-bm25 before building the sparse index.") from exc
    tokenized = []
    for chunk in chunks:
        if not str(chunk.get("retrieval_text") or "").strip():
            raise ValueError(f"Chunk {chunk['chunk_id']} lacks retrieval_text; recreate chunks.")
        tokens = tokenize_financial_text(str(chunk["retrieval_text"]))
        if not tokens:
            raise ValueError(f"Chunk {chunk['chunk_id']} produced no BM25 tokens.")
        tokenized.append(tokens)
    bm25 = BM25Okapi(tokenized, k1=float(k1), b=float(b), epsilon=float(epsilon))
    if int(bm25.corpus_size) != len(chunks):
        raise RuntimeError("BM25 corpus size mismatch.")
    index_path, metadata_path = Path(index_path), Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(index_path)
    metadata = {
        "schema_version": 2, "index_type": "BM25Okapi",
        "retrieval_text_schema": "document_header_v2",
        "tokenizer_id": TOKENIZER_ID,
        "k1": float(k1), "b": float(b), "epsilon": float(epsilon),
        "corpus_size": len(chunks),
        "chunk_ids": [str(x["chunk_id"]) for x in chunks],
        "corpus_sha256": hashlib.sha256(json.dumps(
            [(x["chunk_id"], x["retrieval_text"]) for x in chunks],
            ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
    tmpm = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmpm.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    tmpm.replace(metadata_path)
    return metadata
