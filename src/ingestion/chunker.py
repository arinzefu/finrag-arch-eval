"""Chunk cleaned filing text into retrieval-sized records."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any


TOKEN_RE = re.compile(r"\$?\(?-?\d+(?:,\d{3})*(?:\.\d+)?%?|\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9$]|\())")
BLANK_LINE_RE = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    filing_id: str
    chunk_index: int
    section_id: str
    section_title: str
    text: str
    token_count: int
    word_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in BLANK_LINE_RE.split(text or "") if part.strip()]


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text or "") if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def split_long_unit(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentence_units: list[str] = []
    for sentence in split_sentences(text):
        if estimate_tokens(sentence) <= max_tokens:
            sentence_units.append(sentence)
            continue

        tokens = TOKEN_RE.findall(sentence)
        for start in range(0, len(tokens), max_tokens):
            sentence_units.append(" ".join(tokens[start : start + max_tokens]))

    return sentence_units


def _overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    selected: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if selected and total + unit_tokens > overlap_tokens:
            break
        selected.append(unit)
        total += unit_tokens
        if total >= overlap_tokens:
            break

    return list(reversed(selected))


def chunk_text(
    text: str,
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    """Split text into chunks with paragraph and sentence boundaries where possible."""
    if max_tokens < 100:
        raise ValueError("max_tokens should be at least 100.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    units: list[str] = []
    for paragraph in split_paragraphs(text):
        units.extend(split_long_unit(paragraph, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = _overlap_units(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(part) for part in current)

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def chunk_sections(
    sections: list[dict[str, str]],
    *,
    filing_id: str,
    base_metadata: dict[str, Any],
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Chunk a list of section dictionaries returned by parse_10k.extract_sections."""
    chunks: list[Chunk] = []

    for section in sections:
        section_id = section.get("section_id", "unknown")
        section_title = section.get("section_title", "")
        for local_index, text in enumerate(
            chunk_text(
                section.get("text", ""),
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        ):
            chunk_index = len(chunks)
            identity = f"{filing_id}:{section_id}:{local_index}:{stable_hash(text)}"
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "section_id": section_id,
                    "section_title": section_title,
                    "section_chunk_index": local_index,
                }
            )
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{stable_hash(identity, 16)}",
                    filing_id=filing_id,
                    chunk_index=chunk_index,
                    section_id=section_id,
                    section_title=section_title,
                    text=text,
                    token_count=estimate_tokens(text),
                    word_count=count_words(text),
                    metadata=metadata,
                )
            )

    return chunks
