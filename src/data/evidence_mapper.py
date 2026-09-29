"""
Map FinanceBench gold evidence onto the extracted page/chunk corpus.

FinanceBench directly annotates:
* evidence_doc_name
* evidence_page_num (ZERO-indexed)
* evidence_text
* evidence_text_full_page

The page reference is authoritative. Chunk mapping is derived from evidence text
and is therefore stored separately from page-level gold relevance.

Output design
-------------
``gold_page_refs``:
    Authoritative benchmark document/page references.

``gold_page_chunk_ids``:
    Every chunk on an annotated gold page. Useful for page-level diagnostics.

``gold_chunk_ids``:
    Stricter chunks matched to the actual evidence text. These should be used
    only when ``mapping_complete`` is true (or with explicit handling of
    partial mappings).

This distinction avoids pretending that every chunk on a gold page is equally
relevant evidence.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return output


def normalize_for_match(text: str | None) -> str:
    """Normalize extraction differences while preserving words/numbers."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = text.replace(",", "")
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def multiset_token_containment(
    evidence_text: str,
    chunk_text: str,
) -> float:
    """
    Fraction of evidence tokens represented in a candidate chunk.

    Counter intersection is used so repeated numeric/financial tokens are not
    silently collapsed as they would be with a plain set.
    """
    evidence_tokens = normalize_for_match(evidence_text).split()
    chunk_tokens = normalize_for_match(chunk_text).split()

    if not evidence_tokens or not chunk_tokens:
        return 0.0

    evidence_counter = Counter(evidence_tokens)
    chunk_counter = Counter(chunk_tokens)
    matched = sum(
        min(count, chunk_counter.get(token, 0))
        for token, count in evidence_counter.items()
    )
    return matched / len(evidence_tokens)


def _build_chunk_lookup(
    chunks: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    lookup: dict[tuple[str, int], list[dict[str, Any]]] = {}

    seen_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)

        key = (str(chunk["doc_name"]), int(chunk["page"]))
        lookup.setdefault(key, []).append(chunk)

    for key in lookup:
        lookup[key].sort(
            key=lambda item: int(item.get("chunk_index", 0))
        )
    return lookup


def _build_page_lookup(
    pages: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for page in pages:
        key = (str(page["doc_name"]), int(page["page"]))
        if key in lookup:
            raise ValueError(f"Duplicate page identity: {key}")
        lookup[key] = page
    return lookup


def match_evidence_to_chunks(
    evidence_text: str,
    candidates: list[dict[str, Any]],
    *,
    containment_threshold: float = 0.35,
) -> dict[str, Any]:
    """
    Match one evidence annotation to chunks from its annotated page.

    Matching hierarchy:
    1. normalized exact containment;
    2. multiset evidence-token containment;
    3. no strict chunk match (page reference remains valid).
    """
    evidence_norm = normalize_for_match(evidence_text)
    if not evidence_norm:
        return {
            "matched_chunk_ids": [],
            "method": "no_evidence_text",
            "best_containment_score": 0.0,
            "scores": [],
        }

    exact_matches: list[str] = []
    scored: list[tuple[str, float]] = []

    for chunk in candidates:
        chunk_id = str(chunk["chunk_id"])
        chunk_norm = normalize_for_match(chunk.get("text", ""))

        # Either direction handles an evidence annotation longer than a chunk.
        if (
            chunk_norm
            and (
                evidence_norm in chunk_norm
                or (
                    len(chunk_norm.split()) >= 8
                    and chunk_norm in evidence_norm
                )
            )
        ):
            exact_matches.append(chunk_id)

        score = multiset_token_containment(
            evidence_text,
            str(chunk.get("text", "")),
        )
        scored.append((chunk_id, score))

    if exact_matches:
        return {
            "matched_chunk_ids": exact_matches,
            "method": "normalized_exact_containment",
            "best_containment_score": max(
                (score for _, score in scored),
                default=1.0,
            ),
            "scores": [
                {"chunk_id": chunk_id, "containment": score}
                for chunk_id, score in scored
            ],
        }

    best_score = max((score for _, score in scored), default=0.0)

    if best_score >= containment_threshold:
        # Keep chunks that clear the absolute threshold and are reasonably
        # close to the best candidate. This supports evidence spanning adjacent
        # overlapping chunks without labeling weakly related chunks.
        relative_floor = best_score * 0.80
        matched = [
            chunk_id
            for chunk_id, score in scored
            if score >= containment_threshold and score >= relative_floor
        ]
        return {
            "matched_chunk_ids": matched,
            "method": "token_containment",
            "best_containment_score": best_score,
            "scores": [
                {"chunk_id": chunk_id, "containment": score}
                for chunk_id, score in scored
            ],
        }

    return {
        "matched_chunk_ids": [],
        "method": "page_only_no_strict_chunk_match",
        "best_containment_score": best_score,
        "scores": [
            {"chunk_id": chunk_id, "containment": score}
            for chunk_id, score in scored
        ],
    }


def build_evidence_mapping(
    *,
    financebench_rows: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    containment_threshold: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build question-level page and chunk gold-evidence mappings.

    Missing benchmark pages are treated as hard errors. Failure to text-match a
    chunk is not a hard error because the authoritative page annotation remains
    available for page-level evaluation.
    """
    page_lookup = _build_page_lookup(pages)
    chunk_lookup = _build_chunk_lookup(chunks)

    mapping: dict[str, Any] = {}

    missing_page_refs: list[tuple[str, str, int]] = []
    questions_with_partial_chunk_mapping = 0
    evidence_items_total = 0
    evidence_items_strictly_mapped = 0

    for row in financebench_rows:
        question_id = str(row["financebench_id"])
        evidence_items = row.get("evidence") or []
        row_doc_name = row.get("doc_name")

        if not isinstance(evidence_items, list):
            raise ValueError(
                f"evidence must be a list for {question_id}, "
                f"received {type(evidence_items).__name__}"
            )

        question_page_refs: list[dict[str, Any]] = []
        page_chunk_ids: list[str] = []
        strict_chunk_ids: list[str] = []
        mapped_items: list[dict[str, Any]] = []
        all_items_strict = True

        for evidence_index, evidence in enumerate(evidence_items):
            evidence_items_total += 1

            if not isinstance(evidence, dict):
                raise ValueError(
                    f"evidence item {evidence_index} for {question_id} "
                    f"must be a dict, received {type(evidence).__name__}"
                )

            doc_name_value = (
                evidence.get("evidence_doc_name")
                or evidence.get("doc_name")
                or row_doc_name
            )
            if doc_name_value is None:
                raise ValueError(
                    f"Missing evidence document name for {question_id} "
                    f"evidence item {evidence_index}. Expected one of "
                    "'evidence_doc_name', evidence-level 'doc_name', "
                    "or row-level 'doc_name'."
                )

            if evidence.get("evidence_page_num") is None:
                raise ValueError(
                    f"Missing evidence_page_num for {question_id} "
                    f"evidence item {evidence_index}."
                )

            doc_name = str(doc_name_value)
            page_index = int(evidence["evidence_page_num"])
            key = (doc_name, page_index)

            if key not in page_lookup:
                missing_page_refs.append(
                    (question_id, doc_name, page_index)
                )
                all_items_strict = False
                continue

            page_ref = {"doc_name": doc_name, "page": page_index}
            if page_ref not in question_page_refs:
                question_page_refs.append(page_ref)

            candidates = chunk_lookup.get(key, [])
            current_page_chunk_ids = [
                str(chunk["chunk_id"]) for chunk in candidates
            ]
            page_chunk_ids.extend(current_page_chunk_ids)

            match = match_evidence_to_chunks(
                str(evidence.get("evidence_text") or ""),
                candidates,
                containment_threshold=containment_threshold,
            )
            matched_ids = match["matched_chunk_ids"]
            strict_chunk_ids.extend(matched_ids)

            strict_match = bool(matched_ids)
            if strict_match:
                evidence_items_strictly_mapped += 1
            else:
                all_items_strict = False

            mapped_items.append(
                {
                    "evidence_index": evidence_index,
                    "evidence_doc_name": doc_name,
                    "evidence_page_num": page_index,
                    "evidence_text": evidence.get("evidence_text"),
                    "page_exists": True,
                    "page_chunk_ids": current_page_chunk_ids,
                    "matched_chunk_ids": matched_ids,
                    "chunk_mapping_method": match["method"],
                    "best_containment_score": (
                        match["best_containment_score"]
                    ),
                    "candidate_scores": match["scores"],
                }
            )

        # Stable de-duplication preserving source order.
        page_chunk_ids = list(dict.fromkeys(page_chunk_ids))
        strict_chunk_ids = list(dict.fromkeys(strict_chunk_ids))
        gold_docs = list(
            dict.fromkeys(ref["doc_name"] for ref in question_page_refs)
        )

        if evidence_items and not all_items_strict:
            questions_with_partial_chunk_mapping += 1

        # Convenience fields mirror the simple schema when a question uses
        # exactly one gold document.
        single_gold_doc = gold_docs[0] if len(gold_docs) == 1 else None
        single_doc_pages = (
            sorted(
                {
                    int(ref["page"])
                    for ref in question_page_refs
                    if ref["doc_name"] == single_gold_doc
                }
            )
            if single_gold_doc is not None
            else []
        )

        mapping[question_id] = {
            "question": row.get("question"),
            "gold_answer": row.get("answer"),
            "gold_doc": single_gold_doc,
            "gold_docs": gold_docs,
            "gold_pages": single_doc_pages,
            "gold_page_refs": question_page_refs,
            "gold_page_chunk_ids": page_chunk_ids,
            "gold_chunk_ids": strict_chunk_ids,
            "mapping_complete": bool(evidence_items) and all_items_strict,
            "evidence_items": mapped_items,
        }

    if missing_page_refs:
        preview = "; ".join(
            f"{qid}:{doc}:p{page}"
            for qid, doc, page in missing_page_refs[:20]
        )
        raise RuntimeError(
            "FinanceBench gold evidence references pages absent from the "
            f"extracted corpus ({len(missing_page_refs)} missing): {preview}"
        )

    summary = {
        "questions": len(financebench_rows),
        "evidence_items": evidence_items_total,
        "evidence_items_with_strict_chunk_match": (
            evidence_items_strictly_mapped
        ),
        "questions_with_partial_chunk_mapping": (
            questions_with_partial_chunk_mapping
        ),
        "missing_gold_page_references": len(missing_page_refs),
        "containment_threshold": containment_threshold,
        "page_indexing": "zero_based",
        "strict_chunk_definition": (
            "evidence-text matched chunks only; page-level chunks are stored "
            "separately in gold_page_chunk_ids"
        ),
    }

    return mapping, summary


def write_mapping_json(
    mapping: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Atomically write the question-level evidence mapping."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    try:
        temp.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp.replace(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

