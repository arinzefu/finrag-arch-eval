"""
FinanceBench source-document downloader.

Design principles
-----------------
* Work from the Stage 1 local snapshot.
* Download one PDF per unique ``doc_name``.
* Use FinanceBench's ``doc_link`` as the primary source.
* Optionally fall back to the official FinanceBench GitHub PDF corpus.
* Skip already-valid local files unless ``overwrite=True``.
* Verify that downloaded bytes are parseable as a PDF.
* Record every success, skip and failure in a CSV manifest.
* Never silently ignore a missing or corrupted source document.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import pandas as pd
import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_TIMEOUT_SECONDS = 60
PDF_MAGIC = b"%PDF"
OFFICIAL_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "patronus-ai/financebench/main/pdfs"
)


@dataclass
class DownloadRecord:
    # Requested manifest fields
    doc_name: str
    company: Optional[str]
    doc_type: Optional[str]
    doc_period: Optional[int | str]
    source_url: str
    local_path: str
    download_status: str

    # Additional reproducibility/debugging fields
    resolved_url: Optional[str] = None
    download_source: Optional[str] = None
    http_status: Optional[int] = None
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    pdf_pages: Optional[int] = None
    error: Optional[str] = None


def build_http_session(
    *,
    retries: int = 3,
    backoff_factor: float = 1.0,
) -> requests.Session:
    """Create a requests session with retries for transient HTTP errors."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "FinanceBench-RAG-Architecture-Evaluation/1.0 "
                "(academic benchmark acquisition)"
            )
        }
    )
    return session


def safe_pdf_filename(doc_name: str) -> str:
    """Convert FinanceBench ``doc_name`` into a safe local PDF filename."""
    cleaned = str(doc_name).strip()
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")

    if not cleaned:
        raise ValueError("doc_name produced an empty filename.")

    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"

    return cleaned


def official_github_pdf_url(doc_name: str) -> str:
    """Construct the official FinanceBench GitHub raw-PDF URL."""
    filename = safe_pdf_filename(doc_name)
    return f"{OFFICIAL_GITHUB_RAW_BASE}/{quote(filename)}"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_pdf(path: Path) -> tuple[bool, Optional[int], Optional[str]]:
    """Validate a PDF using both its signature and pypdf parsing."""
    if not path.exists():
        return False, None, "File does not exist."

    if path.stat().st_size < 5:
        return False, None, "File is empty or too small to be a PDF."

    try:
        with path.open("rb") as handle:
            if handle.read(4) != PDF_MAGIC:
                return False, None, "File does not begin with the %PDF signature."
    except OSError as exc:
        return False, None, f"{type(exc).__name__}: {exc}"

    try:
        reader = PdfReader(str(path), strict=False)
        page_count = len(reader.pages)
        if page_count <= 0:
            return False, 0, "PDF parsed but contains zero pages."

        # Access first/last page objects to catch obvious page-tree corruption.
        _ = reader.pages[0]
        _ = reader.pages[-1]
        return True, page_count, None

    except (PdfReadError, OSError, ValueError, IndexError) as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _optional_scalar(row: pd.Series, column: str):
    """Return a clean scalar value from a DataFrame row or None."""
    if column not in row.index:
        return None
    value = row[column]
    return None if pd.isna(value) else value


def _attempt_download(
    *,
    url: str,
    destination: Path,
    session: requests.Session,
    timeout: int,
    chunk_size: int = 1024 * 1024,
) -> tuple[bool, Optional[int], Optional[str]]:
    """Stream one URL to ``destination`` and return status information."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, None, f"Unsupported URL scheme: {parsed.scheme!r}"

    try:
        with session.get(
            url,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        ) as response:
            status = response.status_code
            if status != 200:
                return False, status, f"HTTP {status}"

            with destination.open("wb") as handle:
                for block in response.iter_content(chunk_size=chunk_size):
                    if block:
                        handle.write(block)

        return True, status, None

    except requests.RequestException as exc:
        destination.unlink(missing_ok=True)
        return False, None, f"{type(exc).__name__}: {exc}"
    except OSError as exc:
        destination.unlink(missing_ok=True)
        return False, None, f"{type(exc).__name__}: {exc}"


def download_one_document(
    row: pd.Series,
    *,
    project_root: Path,
    pdf_dir: Path,
    session: requests.Session,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    overwrite: bool = False,
    allow_official_github_fallback: bool = True,
) -> DownloadRecord:
    """Download and validate one unique FinanceBench document."""
    doc_name = str(row["doc_name"])
    primary_url = str(row["doc_link"]).strip()

    pdf_dir.mkdir(parents=True, exist_ok=True)
    destination = pdf_dir / safe_pdf_filename(doc_name)

    try:
        relative_path = destination.resolve().relative_to(project_root.resolve())
        manifest_local_path = relative_path.as_posix()
    except ValueError:
        manifest_local_path = str(destination.resolve())

    base = {
        "doc_name": doc_name,
        "company": _optional_scalar(row, "company"),
        "doc_type": _optional_scalar(row, "doc_type"),
        "doc_period": _optional_scalar(row, "doc_period"),
        "source_url": primary_url,
        "local_path": manifest_local_path,
    }

    if destination.exists() and not overwrite:
        valid, page_count, validation_error = validate_pdf(destination)
        if valid:
            return DownloadRecord(
                **base,
                download_status="skipped_existing",
                resolved_url=primary_url,
                download_source="existing_local_file",
                file_size_bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                pdf_pages=page_count,
            )

        destination.unlink(missing_ok=True)
        print(
            "          existing file invalid; re-downloading "
            f"({validation_error})"
        )

    candidate_urls: list[tuple[str, str]] = []

    if primary_url and primary_url.lower() not in {"none", "nan"}:
        candidate_urls.append(("financebench_doc_link", primary_url))

    if allow_official_github_fallback:
        fallback_url = official_github_pdf_url(doc_name)
        if fallback_url not in {url for _, url in candidate_urls}:
            candidate_urls.append(("official_financebench_github", fallback_url))

    if not candidate_urls:
        return DownloadRecord(
            **base,
            download_status="failed",
            error="No usable source URL is available.",
        )

    attempt_errors: list[str] = []
    temporary = destination.with_suffix(".pdf.part")

    for source_name, url in candidate_urls:
        temporary.unlink(missing_ok=True)

        downloaded, status, request_error = _attempt_download(
            url=url,
            destination=temporary,
            session=session,
            timeout=timeout,
        )

        if not downloaded:
            attempt_errors.append(
                f"{source_name}: {request_error or 'download failed'}"
            )
            continue

        valid, page_count, validation_error = validate_pdf(temporary)
        if not valid:
            attempt_errors.append(
                f"{source_name}: downloaded bytes failed PDF validation "
                f"({validation_error})"
            )
            temporary.unlink(missing_ok=True)
            continue

        temporary.replace(destination)

        return DownloadRecord(
            **base,
            download_status="downloaded",
            resolved_url=url,
            download_source=source_name,
            http_status=status,
            file_size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            pdf_pages=page_count,
        )

    temporary.unlink(missing_ok=True)
    return DownloadRecord(
        **base,
        download_status="failed",
        error=" | ".join(attempt_errors),
    )


def download_documents(
    documents: pd.DataFrame,
    *,
    project_root: str | Path,
    pdf_dir: str | Path,
    manifest_path: str | Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = 3,
    overwrite: bool = False,
    delay_seconds: float = 0.0,
    allow_official_github_fallback: bool = True,
) -> pd.DataFrame:
    """Download all unique source PDFs and persist a manifest after each row."""
    required = {"doc_name", "doc_link"}
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(
            "Document table is missing required fields: "
            + ", ".join(sorted(missing))
        )

    documents = (
        documents
        .drop_duplicates(subset=["doc_name"])
        .sort_values("doc_name", kind="stable")
        .reset_index(drop=True)
    )

    project_root = Path(project_root)
    pdf_dir = Path(pdf_dir)
    manifest_path = Path(manifest_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    session = build_http_session(retries=retries)
    records: list[DownloadRecord] = []
    total = len(documents)

    try:
        for number, (_, row) in enumerate(documents.iterrows(), start=1):
            print(f"[{number:>3}/{total}] {row['doc_name']}")

            record = download_one_document(
                row,
                project_root=project_root,
                pdf_dir=pdf_dir,
                session=session,
                timeout=timeout,
                overwrite=overwrite,
                allow_official_github_fallback=(
                    allow_official_github_fallback
                ),
            )
            records.append(record)

            print(f"          -> {record.download_status}")
            if record.download_source:
                print(f"             source: {record.download_source}")
            if record.error:
                print(f"             error: {record.error}")

            manifest = pd.DataFrame(asdict(item) for item in records)
            manifest.to_csv(manifest_path, index=False)

            if delay_seconds > 0 and number < total:
                time.sleep(delay_seconds)

    finally:
        session.close()

    return pd.DataFrame(asdict(item) for item in records)


def summarize_manifest(manifest: pd.DataFrame) -> pd.Series:
    """Return counts for each download status."""
    if manifest.empty:
        return pd.Series(dtype="int64")
    return manifest["download_status"].value_counts(dropna=False).sort_index()
