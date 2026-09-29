"""Deterministic financial query expansion and question-only entity extraction.

Company names are drawn from indexed document metadata, never from reference
answers or evidence annotations. Ambiguous mentions deliberately skip filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def company_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _company_pattern(company: str) -> re.Pattern[str]:
    words = re.findall(r"[A-Za-z0-9]+", company)
    body = r"[\W_]*".join(map(re.escape, words))
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.I)


@dataclass(frozen=True)
class ProcessedQuery:
    retrieval_query: str
    company: str | None
    fiscal_year: int | None


def _focus_question(question: str) -> str:
    """Remove answer-formatting boilerplate while preserving source hints."""
    focused = re.sub(
        r"^\s*assume\s+that\s+you\s+are\b[^.!?]*[.!?]\s*",
        "",
        question,
        count=1,
        flags=re.I,
    )
    focused = re.sub(
        r"\banswer\s+the\s+following\s+question\s+by\s+primarily\s+using\s+"
        r"information\s+that\s+is\s+shown\s+in\s+the\s+([^:]+):\s*",
        lambda match: f"{match.group(1).strip()}: ",
        focused,
        count=1,
        flags=re.I,
    )
    focused = re.sub(
        r"\bgive\s+a\s+response\s+to\s+the\s+question\s+by\s+relying\s+on\s+"
        r"the\s+details\s+shown\s+in\s+the\s+([^.!?]+)[.!?]?\s*$",
        lambda match: match.group(1).strip(),
        focused,
        count=1,
        flags=re.I,
    )
    focused = re.sub(
        r"\banswer\s+in\s+([^.!?]+)[.!?]?\s*$",
        lambda match: match.group(1).strip(),
        focused,
        count=1,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", focused).strip()


def _mentions_company(question: str, company: str) -> bool:
    if _company_pattern(company).search(question):
        return True
    key = company_key(company)
    # Metadata sometimes stores collapsed names such as AMERICANEXPRESS or
    # COCACOLA while questions use spaces or punctuation.
    return len(key) >= 3 and key in company_key(question)


def _extract_unambiguous_year(question: str) -> int | None:
    fiscal_years = set(
        re.findall(r"\b(?:FY\s*|fiscal\s+year\s+)(20\d{2})\b", question, re.I)
    )
    if len(fiscal_years) == 1:
        return int(next(iter(fiscal_years)))
    if len(fiscal_years) > 1:
        return None

    years = set(re.findall(r"\b(20\d{2})\b", question))
    return int(next(iter(years))) if len(years) == 1 else None


def _coerce_year(value: object) -> int | None:
    if value is None:
        return None
    match = re.search(r"\b(20\d{2})\b", str(value))
    return int(match.group(1)) if match else None


def chunk_matches_query_filters(chunk: dict, processed: ProcessedQuery) -> bool:
    if processed.company and company_key(chunk.get("company") or "") != processed.company:
        return False
    if processed.fiscal_year is not None:
        chunk_year = _coerce_year(chunk.get("fiscal_year") or chunk.get("doc_period"))
        if chunk_year != processed.fiscal_year:
            return False
    return True


def process_query(question: str, *, companies: Iterable[str] = ()) -> ProcessedQuery:
    question = str(question).strip()
    if not question:
        raise ValueError("question must not be empty.")

    # An alias is only accepted if it unambiguously identifies one company.
    matches = {
        company_key(name)
        for name in companies
        if name and _mentions_company(question, str(name))
    }
    company = next(iter(matches)) if len(matches) == 1 else None

    fiscal_year = _extract_unambiguous_year(question)
    focused_question = _focus_question(question)
    additions: list[str] = []
    if re.search(r"\b(?:PPNE|PPE|PP\s*&\s*E|P\s*P\s*&\s*E)\b", question, re.I):
        additions.append("net PP&E property plant and equipment net")
    if re.search(r"\bbalance\s+sheets?\b|\bstatement\s+of\s+financial\s+position\b", question, re.I):
        additions.append("consolidated balance sheet")
    if re.search(r"\b(?:CAPEX|capital expenditures?|capital spending)\b", question, re.I):
        additions.append("capital expenditure capital spending purchases of property plant and equipment")
    if re.search(r"\bcapital(?:\s+|-)+intens(?:ive|ity)\b", question, re.I):
        additions.append(
            "capital expenditures revenue property plant and equipment net "
            "total assets net income return on assets"
        )
    if fiscal_year is not None:
        additions.append(str(fiscal_year))
    return ProcessedQuery(
        " ".join([focused_question, *additions]), company, fiscal_year
    )
