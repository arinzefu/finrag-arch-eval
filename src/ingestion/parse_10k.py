#!/usr/bin/env python3
"""Parse raw SEC 10-K filings into processed text, chunks, and tables."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    from .chunker import chunk_sections, count_words, estimate_tokens
    from .table_parser import extract_tables_from_html
except ImportError:  # Allows: python src/ingestion/parse_10k.py
    from chunker import chunk_sections, count_words, estimate_tokens
    from table_parser import extract_tables_from_html


RAW_EXTENSIONS = {".htm", ".html", ".txt", ".xhtml", ".xml"}
SEC_HEADER_RE = re.compile(r"<SEC-HEADER>.*?</SEC-HEADER>", re.IGNORECASE | re.DOTALL)
SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
BLANK_LINE_RE = re.compile(r"\n\s*\n+")
ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}|\d{18}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ITEM_KEY_PATTERN = r"1A|1B|1C|7A|9A|9B|9C|10|11|12|13|14|15|16|1|2|3|4|5|6|7|8|9"
ITEM_HEADING_RE = re.compile(
    rf"^\s*(item\s+(?:{ITEM_KEY_PATTERN})\.?\s*(?:[-.:]\s*)?.{{0,160}})$",
    re.IGNORECASE,
)
ITEM_ID_RE = re.compile(rf"\bitem\s+({ITEM_KEY_PATTERN})\b", re.IGNORECASE)

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "tr",
    "ul",
}
SKIP_TAGS = {"script", "style", "noscript", "head"}


class _HTMLTextExtractor(HTMLParser):
    """Extract readable narrative text while skipping table bodies."""

    def __init__(self, *, skip_tables: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_tables = skip_tables
        self._skip_depth = 0
        self._table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "table" and self.skip_tables:
            self._table_depth += 1
            return
        if self._skip_depth or self._table_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "table" and self.skip_tables and self._table_depth:
            self._table_depth -= 1
            return
        if self._skip_depth or self._table_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._table_depth:
            return
        if data:
            self.parts.append(data)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relative_string(path: Path, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "filing"


def read_raw_text(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        cleaned = WHITESPACE_RE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
        elif lines and lines[-1] != "":
            lines.append("")
    normalized = "\n".join(lines)
    normalized = BLANK_LINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def html_to_text(raw_text: str, *, skip_tables: bool = True) -> str:
    raw_text = SEC_HEADER_RE.sub(" ", raw_text)
    raw_text = SCRIPT_STYLE_RE.sub(" ", raw_text)

    if "<" not in raw_text or ">" not in raw_text:
        return normalize_text(raw_text)

    parser = _HTMLTextExtractor(skip_tables=skip_tables)
    parser.feed(raw_text)
    extracted = "".join(parser.parts)

    if not extracted.strip():
        extracted = TAG_RE.sub(" ", raw_text)

    return normalize_text(extracted)


def normalize_section_title(line: str) -> str:
    line = normalize_text(line).replace("\n", " ")
    line = re.sub(r"\s+", " ", line)
    return line[:180].strip(" .-:")


def section_id_from_title(title: str) -> str:
    match = ITEM_ID_RE.search(title)
    if not match:
        return "front_matter"
    return "item_" + match.group(1).lower()


def looks_like_item_heading(line: str) -> bool:
    line = normalize_section_title(line)
    if len(line) < 5 or len(line) > 180:
        return False
    if not ITEM_HEADING_RE.match(line):
        return False
    if line.count(" ") > 24:
        return False
    return True


def extract_sections(text: str) -> list[dict[str, str]]:
    """Split cleaned filing text into broad 10-K item sections."""
    lines = text.splitlines()
    sections: list[dict[str, str]] = []
    current_title = "Front Matter"
    current_id = "front_matter"
    current_lines: list[str] = []

    def flush() -> None:
        section_text = normalize_text("\n".join(current_lines))
        if section_text:
            sections.append(
                {
                    "section_id": current_id,
                    "section_title": current_title,
                    "text": section_text,
                    "word_count": str(count_words(section_text)),
                    "token_count": str(estimate_tokens(section_text)),
                }
            )

    for line in lines:
        if looks_like_item_heading(line):
            heading = normalize_section_title(line)
            if current_lines:
                flush()
            current_title = heading
            current_id = section_id_from_title(heading)
            current_lines = [heading]
        else:
            current_lines.append(line)

    flush()

    if not sections:
        return [
            {
                "section_id": "full_text",
                "section_title": "Full Text",
                "text": text,
                "word_count": str(count_words(text)),
                "token_count": str(estimate_tokens(text)),
            }
        ]

    return sections


def load_raw_metadata(raw_dir: Path, root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metadata_path = raw_dir / "metadata.jsonl"
    by_resolved_path: dict[str, dict[str, Any]] = {}
    by_filename: dict[str, dict[str, Any]] = {}

    if not metadata_path.exists():
        return by_resolved_path, by_filename

    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        local_path = str(record.get("local_path", "")).strip()
        if local_path:
            candidates = [
                Path(local_path),
                root / local_path,
                raw_dir / local_path,
                raw_dir / Path(local_path).name,
            ]
            for candidate in candidates:
                try:
                    by_resolved_path[str(candidate.resolve())] = record
                except OSError:
                    continue
            by_filename[Path(local_path).name] = record

    return by_resolved_path, by_filename


def infer_metadata(path: Path) -> dict[str, Any]:
    filename = path.name
    accession_match = ACCESSION_RE.search(filename)
    date_match = DATE_RE.search(filename)
    ticker = path.parent.name.upper()

    accession = accession_match.group(0) if accession_match else ""
    if accession and "-" not in accession and len(accession) == 18:
        accession = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"

    return {
        "company_name": "",
        "ticker": ticker,
        "cik": "",
        "accession_number": accession,
        "form": "10-K",
        "filing_date": date_match.group(0) if date_match else "",
        "report_date": "",
        "primary_document": filename,
        "primary_doc_description": "",
        "source_url": "",
        "local_path": str(path),
        "bytes": path.stat().st_size,
    }


def metadata_for_path(
    path: Path,
    *,
    root: Path,
    path_metadata: dict[str, dict[str, Any]],
    filename_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved = str(path.resolve())
    record = path_metadata.get(resolved) or filename_metadata.get(path.name) or infer_metadata(path)
    metadata = dict(record)
    metadata.setdefault("bytes", path.stat().st_size)
    metadata["source_file"] = relative_string(path, root)
    return metadata


def make_filing_id(metadata: dict[str, Any], source_path: Path) -> str:
    ticker = str(metadata.get("ticker") or source_path.parent.name or "unknown").lower()
    accession = str(metadata.get("accession_number") or "").replace("-", "")
    suffix = accession or source_path.stem
    return safe_name(f"{ticker}_{suffix}")


def find_raw_filings(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    filings: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in RAW_EXTENSIONS:
            continue
        if path.name.lower().startswith("metadata"):
            continue
        filings.append(path)
    return filings


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def process_filing(
    path: Path,
    *,
    root: Path,
    metadata: dict[str, Any],
    chunk_tokens: int,
    overlap_tokens: int,
    min_table_rows: int,
    min_table_cols: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_text = read_raw_text(path)
    cleaned_text = html_to_text(raw_text, skip_tables=True)
    sections = extract_sections(cleaned_text)
    filing_id = make_filing_id(metadata, path)

    base_metadata = {
        "company_name": metadata.get("company_name", ""),
        "ticker": metadata.get("ticker", ""),
        "cik": metadata.get("cik", ""),
        "accession_number": metadata.get("accession_number", ""),
        "form": metadata.get("form", ""),
        "filing_date": metadata.get("filing_date", ""),
        "report_date": metadata.get("report_date", ""),
        "source_url": metadata.get("source_url", ""),
        "source_file": relative_string(path, root),
    }

    chunks = [
        chunk.to_dict()
        for chunk in chunk_sections(
            sections,
            filing_id=filing_id,
            base_metadata=base_metadata,
            max_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]

    tables = extract_tables_from_html(
        raw_text,
        source_file=relative_string(path, root),
        min_rows=min_table_rows,
        min_cols=min_table_cols,
    )
    for table_index, table in enumerate(tables):
        table.update(base_metadata)
        table["filing_id"] = filing_id
        table["table_id"] = f"{filing_id}_table_{table_index:04d}"

    filing_record = {
        "filing_id": filing_id,
        "metadata": base_metadata,
        "text": cleaned_text,
        "text_char_count": len(cleaned_text),
        "word_count": count_words(cleaned_text),
        "token_count": estimate_tokens(cleaned_text),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "table_count": len(tables),
        "sections": sections,
    }

    return filing_record, chunks, tables


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Process raw SEC 10-K filings into cleaned text, chunks, and tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=root / "data" / "raw" / "10k",
        help="Directory containing raw 10-K filing files. Defaults to data/raw/10k.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "processed",
        help="Directory where processed outputs are written. Defaults to data/processed.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=800,
        help="Approximate max tokens per text chunk. Defaults to 800.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=100,
        help="Approximate overlapping tokens between adjacent chunks. Defaults to 100.",
    )
    parser.add_argument(
        "--min-table-rows",
        type=int,
        default=2,
        help="Minimum table rows to keep. Defaults to 2.",
    )
    parser.add_argument(
        "--min-table-cols",
        type=int,
        default=2,
        help="Minimum table columns to keep. Defaults to 2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N raw filings. Useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    root = repo_root()
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()

    if args.chunk_tokens < 100:
        raise SystemExit("--chunk-tokens must be at least 100.")
    if args.overlap_tokens >= args.chunk_tokens:
        raise SystemExit("--overlap-tokens must be smaller than --chunk-tokens.")

    raw_files = find_raw_filings(raw_dir)
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    if not raw_files:
        raise SystemExit(f"No raw 10-K files found under {raw_dir}")

    path_metadata, filename_metadata = load_raw_metadata(raw_dir, root)

    all_chunks: list[dict[str, Any]] = []
    all_tables: list[dict[str, Any]] = []
    filing_summaries: list[dict[str, Any]] = []
    failures: list[str] = []

    print(f"Processing {len(raw_files)} filing(s) from {raw_dir}")
    print(f"Saving processed files under {processed_dir}")

    for raw_path in raw_files:
        try:
            metadata = metadata_for_path(
                raw_path,
                root=root,
                path_metadata=path_metadata,
                filename_metadata=filename_metadata,
            )
            filing_record, chunks, tables = process_filing(
                raw_path,
                root=root,
                metadata=metadata,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                min_table_rows=args.min_table_rows,
                min_table_cols=args.min_table_cols,
            )
        except Exception as exc:
            failures.append(f"{raw_path}: {exc}")
            continue

        ticker = safe_name(str(metadata.get("ticker") or raw_path.parent.name))
        filing_id = filing_record["filing_id"]

        filing_path = processed_dir / "filings" / ticker / f"{filing_id}.json"
        chunks_path = processed_dir / "chunks" / ticker / f"{filing_id}.jsonl"
        tables_path = processed_dir / "tables" / ticker / f"{filing_id}.jsonl"

        write_json(filing_path, filing_record)
        write_jsonl(chunks_path, chunks)
        write_jsonl(tables_path, tables)

        all_chunks.extend(chunks)
        all_tables.extend(tables)
        filing_summaries.append(
            {
                "filing_id": filing_id,
                **filing_record["metadata"],
                "processed_filing": relative_string(filing_path, root),
                "processed_chunks": relative_string(chunks_path, root),
                "processed_tables": relative_string(tables_path, root),
                "text_char_count": filing_record["text_char_count"],
                "word_count": filing_record["word_count"],
                "token_count": filing_record["token_count"],
                "section_count": filing_record["section_count"],
                "chunk_count": filing_record["chunk_count"],
                "table_count": filing_record["table_count"],
            }
        )

        print(
            f"[processed] {filing_id}: "
            f"{len(chunks)} chunk(s), {len(tables)} table(s)"
        )

    write_jsonl(processed_dir / "chunks.jsonl", all_chunks)
    write_jsonl(processed_dir / "tables.jsonl", all_tables)
    write_json(
        processed_dir / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "filing_count": len(filing_summaries),
            "chunk_count": len(all_chunks),
            "table_count": len(all_tables),
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "filings": filing_summaries,
        },
    )

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)

    print(
        f"\nDone: {len(filing_summaries)} filing(s), "
        f"{len(all_chunks)} chunk(s), {len(all_tables)} table(s)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
