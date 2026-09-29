"""Shared, read-only access to the frozen common experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/raw_runs/financebench-heldout-v1_manifest.json"
ARCHITECTURES = ("P0", "P1", "P2", "P3")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path = MANIFEST) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = manifest["question_ids"]
    if not ids or len(ids) != len(set(ids)) or len(ids) != manifest["question_count"]:
        raise ValueError("Invalid frozen question list.")
    source = Path(manifest["question_source"])
    if sha256(source) != manifest["question_source_sha256"]:
        raise ValueError("Frozen question source hash differs.")
    if [r["financebench_id"] for r in load_jsonl(source)] != ids:
        raise ValueError("Frozen question source IDs/order differ.")
    result = {}
    for arch in ARCHITECTURES:
        if arch not in manifest["architectures"]:
            raise ValueError(f"Missing architecture {arch} in manifest.")
        rows = load_jsonl(Path(manifest["outputs"][arch]))
        if [r["question_id"] for r in rows] != ids or any(
            r["architecture"] != arch for r in rows
        ):
            raise ValueError(f"{arch} raw IDs/order/architecture differ.")
        result[arch] = rows
    return manifest, result


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_provenance(path: Path, title: str, manifest_path: Path, inputs: dict[str, Path],
                     parameters: dict, outputs: dict[str, Path]) -> None:
    payload = {
        "title": title,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256(manifest_path),
        "inputs": {name: {"path": str(p.resolve()), "sha256": sha256(p)}
                   for name, p in inputs.items()},
        "parameters": parameters,
        "outputs": {name: str(p.resolve()) for name, p in outputs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def refuse_existing(*paths: Path) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("Evaluation output already exists: " + ", ".join(existing))
