from __future__ import annotations
import hashlib, json, pickle, re
from pathlib import Path
from typing import Any, Callable
import numpy as np
from src.retrieval.base import BaseRetriever
from src.retrieval.query_processor import chunk_matches_query_filters, process_query

_FINANCIAL_TOKEN_PATTERN = re.compile(r"[$£€]?[A-Za-z0-9]+(?:[&./-][A-Za-z0-9]+)*%?")
TOKENIZER_ID = "financial_regex_v1"

def tokenize_financial_text(text: str) -> list[str]:
    return [token.casefold() for token in _FINANCIAL_TOKEN_PATTERN.findall(str(text))]

class BM25Retriever(BaseRetriever):
    def __init__(self, bm25: Any, chunks: list[dict[str, Any]], *, tokenizer: Callable[[str], list[str]] = tokenize_financial_text) -> None:
        if bm25 is None:
            raise ValueError("bm25 must not be None.")
        if not chunks:
            raise ValueError("chunks must not be empty.")
        self.bm25 = bm25
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.companies = {str(c["company"]) for c in self.chunks if c.get("company")}
        corpus_size = getattr(bm25, "corpus_size", None)
        if corpus_size is not None and int(corpus_size) != len(self.chunks):
            raise ValueError("BM25/chunk corpus size mismatch.")

    @classmethod
    def from_artifacts(cls, *, index_path: str | Path, metadata_path: str | Path, chunks: list[dict[str, Any]]) -> "BM25Retriever":
        index_path, metadata_path = Path(index_path), Path(metadata_path)
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"BM25 metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("tokenizer_id") != TOKENIZER_ID:
            raise ValueError("BM25 tokenizer metadata does not match this retriever.")
        if metadata.get("schema_version") != 2 or metadata.get("retrieval_text_schema") != "document_header_v2":
            raise ValueError("Outdated BM25 index; recreate chunks and rebuild BM25.")
        expected_ids = [str(x) for x in metadata["chunk_ids"]]
        actual_ids = [str(x["chunk_id"]) for x in chunks]
        if expected_ids != actual_ids:
            raise ValueError("BM25 chunk order differs from chunks.jsonl; rebuild BM25.")
        actual_hash = hashlib.sha256(json.dumps(
            [(x["chunk_id"], x.get("retrieval_text")) for x in chunks],
            ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if metadata.get("corpus_sha256") != actual_hash:
            raise ValueError("BM25 corpus content differs from chunks.jsonl; rebuild BM25.")
        with index_path.open("rb") as handle:
            bm25 = pickle.load(handle)  # trusted local project artifact only
        return cls(bm25=bm25, chunks=chunks)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query = str(query).strip()
        if not query:
            raise ValueError("query must not be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")
        processed = process_query(query, companies=self.companies)
        tokens = self.tokenizer(processed.retrieval_query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        if scores.ndim != 1 or len(scores) != len(self.chunks):
            raise RuntimeError("BM25 score vector does not match chunk corpus.")
        if processed.company or processed.fiscal_year is not None:
            eligible = np.array([
                i for i, chunk in enumerate(self.chunks)
                if chunk_matches_query_filters(chunk, processed)
            ], dtype=int)
        else:
            eligible = np.arange(len(scores))
        if len(eligible) == 0:
            return []
        k = min(int(k), len(eligible))
        ranked_indices = eligible[np.lexsort((eligible, -scores[eligible]))[:k]]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            idx = int(idx)
            result = dict(self.chunks[idx])
            score = float(scores[idx])
            result.update({
                "rank": rank, "score": score, "sparse_score": score,
                "sparse_rank": rank, "retrieval_method": "bm25",
                "sparse_index": idx,
            })
            results.append(result)
        return results
