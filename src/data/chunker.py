"""
Deterministic page-preserving token chunking.

Default configuration
---------------------
Embedding/tokenizer model: intfloat/e5-base-v2
Maximum model input:      512 tokens
Overlap:                   50 content tokens

The E5 model requires a ``passage: `` prefix for documents. The chunker reserves
space for this prefix and the tokenizer's special tokens. Therefore a nominal
``chunk_size=512`` means the final embedding-model input remains <=512 tokens;
content is not silently truncated later.

Chunks never cross PDF page boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


DEFAULT_TOKENIZER_MODEL = "intfloat/e5-base-v2"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PASSAGE_PREFIX = "passage: "


def document_metadata(page: dict[str, Any]) -> dict[str, Any]:
    """Infer public document identity; do not use benchmark evidence metadata."""
    doc_name = str(page["doc_name"])
    match = re.match(r"^(.+?)_(20\d{2})(?:Q[1-4])?_(10[-_]?K|10[-_]?Q)$", doc_name, re.I)
    company = str(page.get("company") or "").strip() or None
    year = None
    doc_type = str(page.get("doc_type") or "").strip() or None
    if match:
        company = match.group(1).replace("_", " ")
        year = int(match.group(2))
        doc_type = match.group(3).replace("_", "-").upper()
        if "-" not in doc_type:
            doc_type = doc_type[:2] + "-" + doc_type[2:]
    if year is None:
        period_match = re.search(r"\b(20\d{2})\b", str(page.get("doc_period") or ""))
        year = int(period_match.group(1)) if period_match else None
    return {"company": company, "fiscal_year": year, "document_type": doc_type}


def retrieval_header(page: dict[str, Any], metadata: dict[str, Any]) -> str:
    parts = [f"Company: {metadata['company']}" if metadata["company"] else "",
             f"Fiscal year: {metadata['fiscal_year']}" if metadata["fiscal_year"] else "",
             f"Document type: {metadata['document_type']}" if metadata["document_type"] else "",
             f"Document: {page['doc_name']}", f"Page: {page['page']}"]
    return "\n".join(part for part in parts if part) + "\n\n"


def safe_identifier(value: str) -> str:
    """Create a stable chunk-ID-safe form of a document identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Document identifier is empty after normalization.")
    return cleaned


def _trim_span(
    text: str,
    start_char: int,
    end_char: int,
) -> tuple[str, int, int]:
    """Trim boundary whitespace while keeping character offsets correct."""
    raw = text[start_char:end_char]
    left_trim = len(raw) - len(raw.lstrip())
    right_stripped = raw.rstrip()
    right_trim = len(raw) - len(right_stripped)

    adjusted_start = start_char + left_trim
    adjusted_end = end_char - right_trim
    return text[adjusted_start:adjusted_end], adjusted_start, adjusted_end


@dataclass
class TokenWindowChunker:
    """
    Page-local tokenizer window chunker.

    A tokenizer can be injected for tests. Otherwise Hugging Face
    ``AutoTokenizer`` is loaded lazily.
    """

    tokenizer_name: str = DEFAULT_TOKENIZER_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX
    tokenizer: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0.")

        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for Stage 5. "
                    "Install requirements_stage4_7.txt."
                ) from exc

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
            )

        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "Stage 5 requires a fast tokenizer because original-text "
                "character offsets must be preserved."
            )

        self.special_token_count = int(
            self.tokenizer.num_special_tokens_to_add(pair=False)
        )
        self.prefix_token_count = len(
            self.tokenizer(
                self.passage_prefix,
                add_special_tokens=False,
            )["input_ids"]
        )

        self.content_capacity = (
            self.chunk_size
            - self.special_token_count
            - self.prefix_token_count
        )

        if self.content_capacity <= 0:
            raise ValueError(
                "chunk_size is too small after reserving prefix/special tokens."
            )

        if self.chunk_overlap >= self.content_capacity:
            raise ValueError(
                "chunk_overlap must be smaller than the available content "
                f"capacity ({self.content_capacity})."
            )

        self.step_size = self.content_capacity - self.chunk_overlap

    def chunk_page(self, page: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk one extracted page without crossing its page boundary."""
        text = str(page.get("text") or "")
        if not text.strip():
            return []

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if len(token_ids) != len(offsets):
            raise RuntimeError("Tokenizer input_ids/offset_mapping length mismatch.")

        if not token_ids:
            return []

        doc_name = str(page["doc_name"])
        page_index = int(page["page"])
        doc_id = safe_identifier(doc_name)
        metadata = document_metadata(page)
        header = retrieval_header(page, metadata)
        header_tokens = len(self.tokenizer(header, add_special_tokens=False)["input_ids"])
        content_capacity = self.content_capacity - header_tokens
        if content_capacity <= self.chunk_overlap:
            raise ValueError("Chunk budget is too small for document header and overlap.")

        chunks: list[dict[str, Any]] = []
        token_start = 0
        chunk_index = 0

        while token_start < len(token_ids):
            token_end = min(
                token_start + content_capacity,
                len(token_ids),
            )

            # Fast-tokenizer offsets point back to exact source characters.
            start_char = int(offsets[token_start][0])
            end_char = int(offsets[token_end - 1][1])

            chunk_text, start_char, end_char = _trim_span(text, start_char, end_char)
            retrieval_text = header + chunk_text
            # Tokenization at the header/content boundary can differ slightly
            # from the sum of their independent token counts.
            while chunk_text and len(self.tokenizer(
                self.passage_prefix + retrieval_text, add_special_tokens=True
            )["input_ids"]) > self.chunk_size:
                token_end -= 1
                if token_end <= token_start:
                    raise ValueError("No room for chunk content after retrieval header.")
                chunk_text, start_char, end_char = _trim_span(
                    text, int(offsets[token_start][0]), int(offsets[token_end - 1][1])
                )
                retrieval_text = header + chunk_text

            if chunk_text:
                content_tokens = token_end - token_start
                model_input_tokens_estimate = len(self.tokenizer(
                    self.passage_prefix + retrieval_text, add_special_tokens=True
                )["input_ids"])

                if model_input_tokens_estimate > self.chunk_size:
                    raise RuntimeError(
                        "Chunk would exceed configured model input length: "
                        f"{model_input_tokens_estimate} > {self.chunk_size}"
                    )

                chunk_id = (
                    f"{doc_id}_p{page_index}_c{chunk_index:03d}"
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_name": doc_name,
                        "page": page_index,
                        "text": chunk_text,
                        "retrieval_text": retrieval_text,
                        "company": metadata["company"],
                        "fiscal_year": metadata["fiscal_year"],
                        "document_type": metadata["document_type"],
                        "doc_type": page.get("doc_type"),
                        "doc_period": page.get("doc_period"),
                        "chunk_index": chunk_index,
                        "token_start": token_start,
                        "token_end": token_end,
                        "content_token_count": content_tokens,
                        "char_start": start_char,
                        "char_end": end_char,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "content_capacity": content_capacity,
                        "tokenizer_name": self.tokenizer_name,
                        "passage_prefix": self.passage_prefix,
                        "model_input_tokens_estimate": (
                            model_input_tokens_estimate
                        ),
                    }
                )
                chunk_index += 1

            if token_end >= len(token_ids):
                break
            token_start = max(token_start + 1, token_end - self.chunk_overlap)

        return chunks

    def chunk_pages(
        self,
        pages: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Chunk multiple pages and verify global chunk-ID uniqueness."""
        chunks: list[dict[str, Any]] = []

        for page in pages:
            chunks.extend(self.chunk_page(page))

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError("Duplicate chunk_id values were generated.")

        return chunks
