from __future__ import annotations

from typing import Any, Sequence


SYSTEM_PROMPT = """You are a financial question-answering assistant.
Answer the user's question accurately and concisely.
Preserve financial units, currencies, percentages, dates, signs, and numerical
values exactly when they matter to the answer.
Do not invent citations or source text.
Return the answer directly without unnecessary conversational filler."""


CLOSED_BOOK_TEMPLATE = """Question:
{question}

Answer:"""


RAG_TEMPLATE = """Use only the supplied context to answer the financial question.
If the answer cannot be determined from the supplied context, say:
"Insufficient evidence in the retrieved context."

Context:
{context}

Question:
{question}

Answer:"""


def build_closed_book_prompt(question: str) -> str:
    question = str(question).strip()
    if not question:
        raise ValueError("question must not be empty.")
    return CLOSED_BOOK_TEMPLATE.format(question=question)


def format_context(chunks: Sequence[dict[str, Any]]) -> str:
    if not chunks:
        return "[No retrieved context]"

    sections: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        # Reranker windows affect candidate scoring only. Every RAG pipeline
        # sends the same canonical chunk representation to the generator.
        text = str(chunk.get("text") or "").strip()
        if not text:
            raise ValueError(f"Retrieved chunk {position} has empty text.")

        metadata = [
            f"[Context {position}]",
            f"Document: {chunk.get('doc_name', 'unknown_document')}",
            f"Page (zero-indexed): {chunk.get('page', 'unknown')}",
            f"Chunk ID: {chunk.get('chunk_id', 'unknown_chunk')}",
        ]
        if chunk.get("company"):
            metadata.append(f"Company: {chunk['company']}")
        if chunk.get("fiscal_year"):
            metadata.append(f"Fiscal year: {chunk['fiscal_year']}")
        if chunk.get("document_type"):
            metadata.append(f"Document type: {chunk['document_type']}")
        sections.append("\n".join([*metadata, text]))

    return "\n\n".join(sections)


def build_rag_prompt(question: str, chunks: Sequence[dict[str, Any]]) -> str:
    question = str(question).strip()
    if not question:
        raise ValueError("question must not be empty.")
    return RAG_TEMPLATE.format(context=format_context(chunks), question=question)
