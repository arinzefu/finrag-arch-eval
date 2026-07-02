"""Extract simple table artifacts from SEC filing HTML."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")
LINE_BREAK_TAGS = {"br", "p", "div", "li"}


def normalize_cell_text(value: str) -> str:
    """Normalize whitespace while preserving the readable cell value."""
    value = html.unescape(value or "")
    value = value.replace("\xa0", " ")
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def _safe_int(value: str | None, default: int = 1) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(parsed, default)


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._table_depth = 0
        self._current_table: dict[str, Any] | None = None
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None
        self._caption_parts: list[str] = []
        self._capture_caption = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs}

        if tag == "table":
            if self._table_depth == 0:
                self._current_table = {"caption": "", "rows": []}
                self._caption_parts = []
            self._table_depth += 1
            return

        if self._table_depth != 1:
            return

        if tag == "caption":
            self._capture_caption = True
        elif tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = {
                "text_parts": [],
                "is_header": tag == "th",
                "colspan": _safe_int(attrs_dict.get("colspan"), 1),
                "rowspan": _safe_int(attrs_dict.get("rowspan"), 1),
            }
        elif tag in LINE_BREAK_TAGS and self._current_cell is not None:
            self._current_cell["text_parts"].append(" ")

    def handle_data(self, data: str) -> None:
        if self._table_depth != 1:
            return

        if self._current_cell is not None:
            self._current_cell["text_parts"].append(data)
        elif self._capture_caption:
            self._caption_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "table":
            if self._table_depth == 1 and self._current_table is not None:
                self._current_table["caption"] = normalize_cell_text(" ".join(self._caption_parts))
                if self._current_table["rows"]:
                    self.tables.append(self._current_table)
                self._current_table = None
                self._current_row = None
                self._current_cell = None
                self._caption_parts = []
                self._capture_caption = False
            self._table_depth = max(0, self._table_depth - 1)
            return

        if self._table_depth != 1:
            return

        if tag == "caption":
            self._capture_caption = False
        elif tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            text = normalize_cell_text(" ".join(self._current_cell["text_parts"]))
            self._current_row.append(
                {
                    "text": text,
                    "is_header": self._current_cell["is_header"],
                    "colspan": self._current_cell["colspan"],
                    "rowspan": self._current_cell["rowspan"],
                }
            )
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None and self._current_table is not None:
            if any(cell["text"] for cell in self._current_row):
                self._current_table["rows"].append(self._current_row)
            self._current_row = None


def _cell_rows_to_text_rows(rows: list[list[dict[str, Any]]]) -> list[list[str]]:
    return [[cell["text"] for cell in row] for row in rows]


def _trim_empty_edges(rows: list[list[str]]) -> list[list[str]]:
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    padded = [row + [""] * (max_cols - len(row)) for row in rows]

    non_empty_cols = [
        idx
        for idx in range(max_cols)
        if any(row[idx].strip() for row in padded)
    ]
    if not non_empty_cols:
        return []

    start = min(non_empty_cols)
    end = max(non_empty_cols) + 1
    return [row[start:end] for row in padded]


def rows_to_markdown(rows: list[list[str]]) -> str:
    rows = _trim_empty_edges(rows)
    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    padded = [row + [""] * (column_count - len(row)) for row in rows]

    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").strip()

    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(escape(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(column_count)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def rows_to_plain_text(rows: list[list[str]]) -> str:
    rows = _trim_empty_edges(rows)
    return "\n".join("\t".join(cell.strip() for cell in row) for row in rows)


def extract_tables_from_html(
    html_text: str,
    *,
    source_file: str | Path | None = None,
    min_rows: int = 2,
    min_cols: int = 2,
) -> list[dict[str, Any]]:
    """Return normalized table records found in an HTML filing document."""
    parser = _HTMLTableParser()
    parser.feed(html_text)

    records: list[dict[str, Any]] = []
    for table_index, raw_table in enumerate(parser.tables):
        rows = _trim_empty_edges(_cell_rows_to_text_rows(raw_table["rows"]))
        if not rows:
            continue

        column_count = max(len(row) for row in rows)
        if len(rows) < min_rows or column_count < min_cols:
            continue

        records.append(
            {
                "table_index": table_index,
                "source_file": str(source_file) if source_file else None,
                "caption": raw_table.get("caption", ""),
                "row_count": len(rows),
                "column_count": column_count,
                "rows": rows,
                "text": rows_to_plain_text(rows),
                "markdown": rows_to_markdown(rows),
            }
        )

    return records
