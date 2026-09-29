from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.document_downloader import (  # noqa: E402
    download_documents,
    summarize_manifest,
)
from src.data.financebench_loader import (  # noqa: E402
    build_unique_document_table,
    load_local_financebench,
)


DEFAULT_PDF_DIR = ROOT / "data" / "raw" / "pdfs"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "raw" / "pdf_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate the unique source PDFs referenced by the "
            "local FinanceBench snapshot."
        )
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"PDF directory (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest CSV (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for transient HTTP errors (default: 3).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay between documents in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing valid PDFs.",
    )
    parser.add_argument(
        "--no-github-fallback",
        action="store_true",
        help=(
            "Disable fallback to the official FinanceBench GitHub PDF corpus "
            "when a dataset doc_link fails."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("Stage 3 — FinanceBench source-document acquisition")
    print("=" * 78)

    df = load_local_financebench()
    documents = build_unique_document_table(df)

    print(f"\nQuestions in local snapshot: {len(df)}")
    print(f"Unique documents to resolve: {len(documents)}")
    print(f"PDF directory: {args.pdf_dir}")
    print(f"Manifest: {args.manifest}")
    print(
        "Official GitHub fallback: "
        f"{'disabled' if args.no_github_fallback else 'enabled'}"
    )
    print()

    manifest = download_documents(
        documents,
        project_root=ROOT,
        pdf_dir=args.pdf_dir,
        manifest_path=args.manifest,
        timeout=args.timeout,
        retries=args.retries,
        overwrite=args.overwrite,
        delay_seconds=args.delay,
        allow_official_github_fallback=not args.no_github_fallback,
    )

    print("\nManifest summary:")
    summary = summarize_manifest(manifest)
    print("  No documents were processed." if summary.empty else summary.to_string())

    failures = manifest.loc[
        manifest["download_status"].eq("failed")
    ].copy()

    print(f"\nManifest written to: {args.manifest}")
    print(f"Unresolved failures: {len(failures)}")

    if not failures.empty:
        print("\nFailed documents:")
        columns = [
            "doc_name",
            "company",
            "source_url",
            "local_path",
            "error",
        ]
        print(failures[columns].to_string(index=False))

        print(
            "\nStage 3 stop condition NOT satisfied. "
            "Resolve or explicitly document every failure before Stage 4."
        )
        raise SystemExit(1)

    print("\nStage 3 stop condition satisfied:")
    print("  Every required source document is present and parseable.")
    print("  No document failure was silently ignored.")


if __name__ == "__main__":
    main()
