#!/usr/bin/env python
"""Stage 18: summarize recorded canonical-run latency without new inference."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_run,
                                       refuse_existing, write_csv, write_provenance)

METRICS = ROOT / "results/evaluation/latency_metrics.csv"
SUMMARY = ROOT / "results/aggregated/latency_summary.csv"
PROVENANCE = ROOT / "results/evaluation/latency_evaluation_manifest.json"
COMPONENTS = ("retrieval", "reranking", "generation", "total")
FIELDS = ["run_id", "architecture", "question_id"] + [f"{c}_latency_ms" for c in COMPONENTS]
SUMMARY_FIELDS = ["table_title", "run_id", "architecture", "component", "question_count",
                  "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms"]


def evaluate(manifest: dict, runs: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    metrics, summary = [], []
    for arch in ARCHITECTURES:
        rows = []
        for raw in runs[arch]:
            row = {"run_id": manifest["run_id"], "architecture": arch,
                   "question_id": raw["question_id"]}
            for component in COMPONENTS:
                value = raw.get(f"{component}_latency_ms")
                applicable = component not in ("retrieval", "reranking") or (
                    component == "retrieval" and arch != "P0") or (
                    component == "reranking" and arch == "P3")
                if applicable:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                        raise ValueError(f"Invalid {component} latency for {arch} {raw['question_id']}")
                    row[f"{component}_latency_ms"] = value
                else:
                    row[f"{component}_latency_ms"] = ""
            rows.append(row)
        metrics.extend(rows)
        for component in COMPONENTS:
            values = [float(r[f"{component}_latency_ms"]) for r in rows
                      if r[f"{component}_latency_ms"] != ""]
            if not values:
                continue
            summary.append({"table_title": "Table L1: Canonical per-question latency (ms)",
                            "run_id": manifest["run_id"], "architecture": arch,
                            "component": component, "question_count": len(values),
                            "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                            "std_ms": statistics.stdev(values) if len(values) > 1 else 0,
                            "min_ms": min(values), "max_ms": max(values)})
    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    refuse_existing(METRICS, SUMMARY, PROVENANCE)
    manifest, runs = load_run(args.manifest)
    metrics, summary = evaluate(manifest, runs)
    write_csv(METRICS, metrics, FIELDS)
    write_csv(SUMMARY, summary, SUMMARY_FIELDS)
    write_provenance(PROVENANCE, "Stage 18: Canonical-run latency", args.manifest,
                     {a: Path(manifest["outputs"][a]) for a in ARCHITECTURES},
                     {"unit": "milliseconds", "std": "sample standard deviation (ddof=1)",
                      "scope": manifest.get("latency_scope"),
                      "unavailable": "dense/BM25/fusion component timings were not recorded; no inferred values"},
                     {"per_question": METRICS, "summary": SUMMARY})
    print("Table L1: Canonical per-question latency (ms)")
    for row in summary:
        if row["component"] == "total":
            print(f"{row['architecture']}: mean={row['mean_ms']:.1f} "
                  f"median={row['median_ms']:.1f} std={row['std_ms']:.1f}")


if __name__ == "__main__":
    main()
