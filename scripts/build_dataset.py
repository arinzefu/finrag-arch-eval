

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
DEFAULT_SLEEP_SECONDS = 0.2
MAX_RETRIES = 4


@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    cik: int
    aliases: tuple[str, ...] = ()

    @property
    def cik10(self) -> str:
        return f"{self.cik:010d}"

    @property
    def slug(self) -> str:
        return self.ticker.lower().replace(".", "-")


COMPANIES: tuple[Company, ...] = (
    Company("Apple Inc.", "AAPL", 320193, ("apple",)),
    Company("Microsoft Corporation", "MSFT", 789019, ("microsoft", "mircosoft")),
    Company("Amazon.com, Inc.", "AMZN", 1018724, ("amazon",)),
    Company("Meta Platforms, Inc.", "META", 1326801, ("meta", "facebook")),
    Company("NVIDIA Corporation", "NVDA", 1045810, ("nvidia",)),
    Company("JPMorgan Chase & Co.", "JPM", 19617, ("jp morgan", "jpmorgan", "jp morgan chase")),
    Company("Bank of America Corporation", "BAC", 70858, ("bank of america", "bofa")),
    Company("BlackRock, Inc.", "BLK", 1364742, ("blackrock",)),
    Company("The Goldman Sachs Group, Inc.", "GS", 886982, ("goldman sachs",)),
    Company("State Street Corporation", "STT", 93751, ("state street",)),
    Company("Broadcom Inc.", "AVGO", 1730168, ("broadcom",)),
    Company("Berkshire Hathaway Inc.", "BRK-B", 1067983, ("berkshire hathaway", "berkshire")),
    Company("Eli Lilly and Company", "LLY", 59478, ("eli lilly", "eli lily", "lilly")),
    Company("Walmart Inc.", "WMT", 104169, ("walmart", "wal-mart")),
    Company("Morgan Stanley", "MS", 895421, ("morgan stanley",)),
    Company("Blackstone Inc.", "BX", 1393818, ("blackstone",)),
    Company("General Motors Company", "GM", 1467858, ("general motors",)),
    Company("Ford Motor Company", "F", 37996, ("ford",)),
    Company("Wells Fargo & Company", "WFC", 72971, ("wells fargo",)),
    Company("Hilton Worldwide Holdings Inc.", "HLT", 1585689, ("hilton worldwide", "hilton")),
)


@dataclass(frozen=True)
class Filing:
    company_name: str
    ticker: str
    cik: int
    accession_number: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    primary_doc_description: str
    source_url: str
    local_path: str
    bytes: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_selector(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def company_lookup() -> dict[str, Company]:
    lookup: dict[str, Company] = {}
    for company in COMPANIES:
        keys = {
            company.ticker,
            company.ticker.replace("-", "."),
            company.name,
            company.slug,
            *company.aliases,
        }
        for key in keys:
            lookup[normalize_selector(key)] = company
    return lookup


def selected_companies(selectors: Iterable[str] | None) -> list[Company]:
    if not selectors:
        return list(COMPANIES)

    lookup = company_lookup()
    selected: list[Company] = []
    seen: set[str] = set()
    unknown: list[str] = []

    for selector in selectors:
        key = normalize_selector(selector)
        company = lookup.get(key)
        if company is None:
            unknown.append(selector)
            continue
        if company.ticker not in seen:
            selected.append(company)
            seen.add(company.ticker)

    if unknown:
        valid = ", ".join(company.ticker for company in COMPANIES)
        raise SystemExit(f"Unknown company selector(s): {', '.join(unknown)}. Valid tickers: {valid}")

    return selected


def build_request(url: str, user_agent: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": user_agent,
        },
    )


def fetch_bytes(url: str, user_agent: str, sleep_seconds: float) -> bytes:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            wait = min(2 ** (attempt - 1), 12)
            time.sleep(wait)

        try:
            with urllib.request.urlopen(build_request(url, user_agent), timeout=60) as response:
                payload = response.read()
            time.sleep(sleep_seconds)
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except urllib.error.URLError as exc:
            last_error = exc

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_json(url: str, user_agent: str, sleep_seconds: float) -> dict:
    payload = fetch_bytes(url, user_agent, sleep_seconds)
    return json.loads(payload.decode("utf-8"))


def find_10k_filings(
    company: Company,
    user_agent: str,
    *,
    max_filings: int,
    include_amendments: bool,
    sleep_seconds: float,
) -> list[dict[str, str]]:
    submissions = fetch_json(
        SEC_SUBMISSIONS_URL.format(cik=company.cik10),
        user_agent,
        sleep_seconds,
    )

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    accepted_forms = {"10-K", "10-KT"}
    if include_amendments:
        accepted_forms.update({"10-K/A", "10-KT/A"})

    filings: list[dict[str, str]] = []
    for idx, form in enumerate(forms):
        if form not in accepted_forms:
            continue

        primary_document = primary_documents[idx]
        if not primary_document:
            continue

        accession = accessions[idx]
        compact_accession = accession.replace("-", "")
        source_url = SEC_ARCHIVES_URL.format(
            cik_int=company.cik,
            accession=compact_accession,
            document=primary_document,
        )
        filings.append(
            {
                "accession_number": accession,
                "form": form,
                "filing_date": filing_dates[idx],
                "report_date": report_dates[idx] if idx < len(report_dates) else "",
                "primary_document": primary_document,
                "primary_doc_description": descriptions[idx] if idx < len(descriptions) else "",
                "source_url": source_url,
            }
        )
        if len(filings) >= max_filings:
            break

    return filings


def safe_document_name(company: Company, filing: dict[str, str]) -> str:
    suffix = Path(filing["primary_document"]).suffix or ".html"
    accession = filing["accession_number"].replace("-", "")
    filing_date = filing["filing_date"]
    return f"{filing_date}_{company.ticker}_{accession}{suffix}".replace("/", "-")


def write_bytes(path: Path, payload: bytes, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return True


def write_metadata(raw_dir: Path, filings: list[Filing]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_jsonl = raw_dir / "metadata.jsonl"
    metadata_csv = raw_dir / "metadata.csv"
    manifest_json = raw_dir / "manifest.json"

    with metadata_jsonl.open("w", encoding="utf-8") as handle:
        for filing in filings:
            handle.write(json.dumps(asdict(filing), ensure_ascii=False) + "\n")

    fieldnames = list(asdict(filings[0]).keys()) if filings else [
        "company_name",
        "ticker",
        "cik",
        "accession_number",
        "form",
        "filing_date",
        "report_date",
        "primary_document",
        "primary_doc_description",
        "source_url",
        "local_path",
        "bytes",
    ]
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for filing in filings:
            writer.writerow(asdict(filing))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "SEC EDGAR company submissions and filing archives",
        "raw_filings_dir": str(raw_dir),
        "filing_count": len(filings),
        "companies": [asdict(company) for company in COMPANIES],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_dataset_readme(data_dir: Path) -> None:
    qa_dir = data_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    readme = qa_dir / "README.md"
    if readme.exists():
        return

    readme.write_text(
        "\n".join(
            [
                "# Dataset Notes",
                "",
                "Raw 10-K filings are downloaded from SEC EDGAR by `scripts/build_dataset.py`.",
                "",
                "Expected raw layout:",
                "",
                "```text",
                "data/raw/10k/",
                "  metadata.csv",
                "  metadata.jsonl",
                "  manifest.json",
                "  <ticker>/",
                "    <filing-date>_<ticker>_<accession>.htm",
                "```",
                "",
                "`questions.jsonl` should contain gold QA pairs with category and evidence labels.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw 10-K filings for the FinRAG architecture evaluation dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
        help="Dataset directory. Defaults to <repo>/data.",
    )
    parser.add_argument(
        "--company",
        action="append",
        dest="companies",
        help="Company selector by ticker or name. Repeat to select a subset. Defaults to all configured companies.",
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        default=1,
        help="Number of recent annual filings to download per company. Defaults to 1.",
    )
    parser.add_argument(
        "--include-amendments",
        action="store_true",
        help="Include 10-K/A and 10-KT/A amendments in addition to annual filings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing filing files.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay after each SEC request. Defaults to 0.2 seconds.",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help="SEC User-Agent string. Defaults to SEC_USER_AGENT from .env or the environment.",
    )
    return parser.parse_args()


def main() -> int:
    root = repo_root()
    load_dotenv(root / ".env")
    args = parse_args()

    if args.max_filings < 1:
        raise SystemExit("--max-filings must be at least 1.")

    user_agent = args.user_agent or os.environ.get("SEC_USER_AGENT")
    if not user_agent or "@" not in user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT in .env or pass --user-agent with a descriptive value "
            "that includes contact information, for example: "
            "'finrag-arch-eval/0.1 your.name@example.com'."
        )

    companies = selected_companies(args.companies)
    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw" / "10k"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_readme(data_dir)

    downloaded: list[Filing] = []
    failures: list[str] = []

    print(f"Saving filings under: {raw_dir}")
    print(f"Companies: {', '.join(company.ticker for company in companies)}")

    for company in companies:
        try:
            filings = find_10k_filings(
                company,
                user_agent,
                max_filings=args.max_filings,
                include_amendments=args.include_amendments,
                sleep_seconds=args.sleep_seconds,
            )
        except Exception as exc:
            failures.append(f"{company.ticker}: could not read submissions ({exc})")
            continue

        if not filings:
            failures.append(f"{company.ticker}: no 10-K filings found")
            continue

        for filing in filings:
            local_path = raw_dir / company.slug / safe_document_name(company, filing)
            if args.dry_run:
                print(f"[dry-run] {company.ticker}: {filing['source_url']} -> {local_path}")
                payload_size = 0
            else:
                try:
                    if local_path.exists() and not args.overwrite:
                        payload_size = local_path.stat().st_size
                        print(f"[exists] {company.ticker}: {local_path}")
                    else:
                        payload = fetch_bytes(
                            filing["source_url"],
                            user_agent,
                            args.sleep_seconds,
                        )
                        write_bytes(local_path, payload, args.overwrite)
                        payload_size = len(payload)
                        print(f"[saved] {company.ticker}: {local_path}")
                except Exception as exc:
                    failures.append(f"{company.ticker}: could not download {filing['accession_number']} ({exc})")
                    continue

            downloaded.append(
                Filing(
                    company_name=company.name,
                    ticker=company.ticker,
                    cik=company.cik,
                    accession_number=filing["accession_number"],
                    form=filing["form"],
                    filing_date=filing["filing_date"],
                    report_date=filing["report_date"],
                    primary_document=filing["primary_document"],
                    primary_doc_description=filing["primary_doc_description"],
                    source_url=filing["source_url"],
                    local_path=str(local_path.relative_to(root) if local_path.is_relative_to(root) else local_path),
                    bytes=payload_size,
                )
            )

    if not args.dry_run:
        write_metadata(raw_dir, downloaded)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)

    print(f"\nRecorded filings: {len(downloaded)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
