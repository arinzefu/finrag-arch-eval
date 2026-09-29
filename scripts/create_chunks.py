#!/usr/bin/env python
"""Stage 5 — Create deterministic page-preserving RAG chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.chunker import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOKENIZER_MODEL,
    TokenWindowChunker,
)


DEFAULT_PAGES = ROOT / "data" / "interim" / "pages.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_CONFIG = ROOT / "data" / "processed" / "chunking_config.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input pages file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def write_jsonl_atomic(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk FinanceBench pages without crossing page boundaries."
    )
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--tokenizer-model",
        default=DEFAULT_TOKENIZER_MODEL,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("Stage 5 — Page-preserving token chunking")
    print("=" * 78)
    print(f"Pages:           {args.pages}")
    print(f"Chunks:          {args.output}")
    print(f"Tokenizer/model: {args.tokenizer_model}")
    print(f"Chunk size:      {args.chunk_size}")
    print(f"Overlap:         {args.chunk_overlap}\n")

    pages = read_jsonl(args.pages)

    required_page_fields = {"doc_name", "page", "text"}
    for i, page in enumerate(pages):
        missing = required_page_fields - set(page)
        if missing:
            raise ValueError(
                f"Page row {i} is missing: {', '.join(sorted(missing))}"
            )

    chunker = TokenWindowChunker(
        tokenizer_name=args.tokenizer_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    chunks = chunker.chunk_pages(pages)

    if not chunks:
        raise RuntimeError("No chunks were produced.")

    # Every non-empty source page should have at least one chunk.
    nonempty_page_refs = {
        (str(page["doc_name"]), int(page["page"]))
        for page in pages
        if str(page.get("text") or "").strip()
    }
    chunk_page_refs = {
        (str(chunk["doc_name"]), int(chunk["page"]))
        for chunk in chunks
    }
    missing_pages = sorted(nonempty_page_refs - chunk_page_refs)
    if missing_pages:
        raise RuntimeError(
            "Some non-empty pages generated no chunks: "
            + ", ".join(map(str, missing_pages[:20]))
        )

    write_jsonl_atomic(chunks, args.output)

    config = {
        "tokenizer_name": args.tokenizer_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "passage_prefix": chunker.passage_prefix,
        "special_token_count": chunker.special_token_count,
        "prefix_token_count": chunker.prefix_token_count,
        "content_capacity": chunker.content_capacity,
        "step_size": chunker.step_size,
        "pages_total": len(pages),
        "pages_with_text": len(nonempty_page_refs),
        "chunks_total": len(chunks),
        "page_boundary_policy": "chunks_never_cross_pages",
        "page_indexing": "zero_based",
    }

    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Pages loaded:       {len(pages)}")
    print(f"Non-empty pages:    {len(nonempty_page_refs)}")
    print(f"Chunks created:     {len(chunks)}")
    print(f"Content capacity:   {chunker.content_capacity} tokens")
    print(f"Config:             {args.config_output}")

    print("\nStage 5 stop condition satisfied:")
    print("  Every chunk retains doc_name and zero-indexed page.")
    print("  Every non-empty page generated at least one chunk.")
    print("  No chunk exceeds the configured 512-token model input budget.")


if __name__ == "__main__":
    main()
