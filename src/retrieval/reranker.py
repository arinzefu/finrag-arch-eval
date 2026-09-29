from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_WINDOW_OVERLAP = 64


@dataclass(frozen=True)
class _RerankWindow:
    candidate_index: int
    window_index: int
    text: str
    token_count: int

class CrossEncoderReranker:
    def __init__(self, model: Any, *, batch_size: int = 32, window_overlap: int = DEFAULT_WINDOW_OVERLAP) -> None:
        if model is None:
            raise ValueError("model must not be None.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if window_overlap < 0:
            raise ValueError("window_overlap must be >= 0.")
        self.model, self.batch_size = model, int(batch_size)
        self.window_overlap = int(window_overlap)

    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_RERANKER_MODEL, *, device: Optional[str] = None, max_length: int | None = None, batch_size: int = 32, window_overlap: int = DEFAULT_WINDOW_OVERLAP):
        from sentence_transformers import CrossEncoder
        kwargs = {}
        if device is not None: kwargs["device"] = device
        if max_length is not None: kwargs["max_length"] = int(max_length)
        return cls(CrossEncoder(model_name, **kwargs), batch_size=batch_size, window_overlap=window_overlap)

    @staticmethod
    def _max_length(model: Any, tokenizer: Any) -> int | None:
        max_length = getattr(model, "max_length", None)
        if max_length is None and tokenizer is not None:
            max_length = getattr(tokenizer, "model_max_length", None)
        if max_length is None:
            return None
        max_length = int(max_length)
        return None if max_length >= 1000000 else max_length

    @staticmethod
    def _metadata_header(candidate: dict) -> str:
        parts = [
            f"Company: {candidate['company']}" if candidate.get("company") else "",
            f"Fiscal year: {candidate['fiscal_year']}" if candidate.get("fiscal_year") else "",
            f"Document type: {candidate['document_type']}" if candidate.get("document_type") else "",
            f"Document: {candidate.get('doc_name', 'unknown_document')}",
            f"Page: {candidate.get('page', 'unknown')}",
        ]
        return "\n".join(part for part in parts if part) + "\n\n"

    @staticmethod
    def _token_ids(tokenizer: Any, text: str) -> list[int]:
        return list(tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"])

    @classmethod
    def _pair_token_count(cls, tokenizer: Any, query: str, text: str) -> int:
        return len(
            tokenizer(
                str(query),
                str(text),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )

    @classmethod
    def _document_budget(cls, tokenizer: Any, query: str, max_length: int) -> int:
        query_tokens = len(cls._token_ids(tokenizer, query))
        try:
            special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
        except Exception:
            special_tokens = cls._pair_token_count(tokenizer, "", "")
        return max_length - query_tokens - special_tokens

    def _candidate_windows(
        self,
        *,
        query: str,
        candidate: dict,
        candidate_index: int,
        tokenizer: Any,
        max_length: int | None,
    ) -> list[_RerankWindow]:
        full_text = str(candidate.get("retrieval_text") or self._metadata_header(candidate) + candidate["text"])
        if tokenizer is None or max_length is None:
            return [_RerankWindow(candidate_index, 0, full_text, 0)]

        full_tokens = self._pair_token_count(tokenizer, query, full_text)
        candidate["reranker_input_tokens"] = full_tokens
        candidate["reranker_input_truncated"] = full_tokens > max_length
        if full_tokens <= max_length:
            return [_RerankWindow(candidate_index, 0, full_text, full_tokens)]

        doc_budget = self._document_budget(tokenizer, query, max_length)
        if doc_budget <= 0:
            return [_RerankWindow(candidate_index, 0, full_text, full_tokens)]

        header = self._metadata_header(candidate)
        header_tokens = len(self._token_ids(tokenizer, header))
        body_budget = doc_budget - header_tokens
        if body_budget < 32:
            header = ""
            body_budget = doc_budget
        if body_budget <= 0:
            return [_RerankWindow(candidate_index, 0, full_text, full_tokens)]

        body = str(candidate["text"])
        use_offsets = bool(getattr(tokenizer, "is_fast", False))
        encoded = tokenizer(
            body,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=use_offsets,
        )
        token_ids = list(encoded["input_ids"])
        if not token_ids:
            return [_RerankWindow(candidate_index, 0, full_text, full_tokens)]

        offsets = encoded.get("offset_mapping") if use_offsets else None
        overlap = min(self.window_overlap, max(0, body_budget - 1))
        windows: list[_RerankWindow] = []
        start = 0
        window_index = 0

        while start < len(token_ids):
            end = min(start + body_budget, len(token_ids))

            while end > start:
                if offsets:
                    window_body = body[int(offsets[start][0]):int(offsets[end - 1][1])].strip()
                else:
                    window_body = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
                window_text = (header + window_body).strip()
                token_count = self._pair_token_count(tokenizer, query, window_text)
                if token_count <= max_length:
                    break
                end -= 1
            else:
                break

            if window_text:
                windows.append(_RerankWindow(candidate_index, window_index, window_text, token_count))
                window_index += 1
            if end >= len(token_ids):
                break
            start = max(start + 1, end - overlap)

        return windows or [_RerankWindow(candidate_index, 0, full_text, full_tokens)]

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        query = str(query).strip()
        if not query:
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not documents:
            return []
        candidates = [dict(d) for d in documents]
        for i, d in enumerate(candidates):
            if "chunk_id" not in d or "text" not in d:
                raise ValueError(f"Candidate {i} lacks chunk_id or text.")
        tokenizer = getattr(self.model, "tokenizer", None)
        max_length = self._max_length(self.model, tokenizer)
        windows = [
            window
            for i, candidate in enumerate(candidates)
            for window in self._candidate_windows(
                query=query,
                candidate=candidate,
                candidate_index=i,
                tokenizer=tokenizer,
                max_length=max_length,
            )
        ]
        pairs = [(query, window.text) for window in windows]
        raw_scores = self.model.predict(
            pairs, batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.ndim != 1 or len(scores) != len(windows):
            raise RuntimeError("Expected one reranker score per scored window.")
        best_by_candidate: dict[int, tuple[float, _RerankWindow]] = {}
        window_counts = [0 for _ in candidates]
        for window, score in zip(windows, scores):
            score = float(score)
            window_counts[window.candidate_index] += 1
            current = best_by_candidate.get(window.candidate_index)
            if current is None or score > current[0]:
                best_by_candidate[window.candidate_index] = (score, window)

        for first_rank, d in enumerate(candidates, start=1):
            score, best_window = best_by_candidate[first_rank - 1]
            if "score" in d:
                d["first_stage_score"] = float(d["score"])
            d["reranker_score"] = score
            d["reranker_window_count"] = int(window_counts[first_rank - 1])
            d["reranker_best_window_index"] = int(best_window.window_index)
            d["reranker_best_window_tokens"] = int(best_window.token_count)
            if d.get("reranker_input_truncated"):
                d["reranker_best_text"] = best_window.text
            d["_first_stage_position"] = first_rank
        candidates.sort(key=lambda d: (-d["reranker_score"], d["_first_stage_position"], str(d["chunk_id"])))
        output = []
        for rank, d in enumerate(candidates[:min(top_k, len(candidates))], start=1):
            result = dict(d)
            result.pop("_first_stage_position", None)
            result.update({"rank": rank, "reranker_rank": rank,
                           "score": float(result["reranker_score"]), "retrieval_method": "cross_encoder_rerank"})
            output.append(result)
        return output
