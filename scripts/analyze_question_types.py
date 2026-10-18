#!/usr/bin/env python
"""Stage 19: native FinanceBench category summaries from saved evaluations."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_jsonl, load_run,
                                       read_csv, refuse_existing, write_csv, write_provenance)

ANSWER = ROOT / "results/evaluation/answer_metrics_v2.csv"
RETRIEVAL = ROOT / "results/evaluation/retrieval_metrics.csv"
FAITHFULNESS = ROOT / "results/evaluation/faithfulness_metrics.csv"
LATENCY = ROOT / "results/evaluation/latency_metrics.csv"
SUMMARY = ROOT / "results/aggregated/question_type_summary.csv"
PROVENANCE = ROOT / "results/aggregated/question_type_summary_manifest.json"
METRICS = ("exact_match", "token_f1", "numeric_accuracy", "numeric_precision", "numeric_recall",
           "numeric_f1", "answer_correctness", "answer_completeness", "exact_abstention",
           "bertscore_precision", "bertscore_recall", "bertscore_f1",
           "hit_at_5", "recall_at_5", "mrr_at_5", "context_faithfulness",
           "context_hallucination", "gold_evidence_support", "gold_evidence_hallucination",
           "total_latency_ms")
FIELDS = ["table_title", "run_id", "grouping", "category", "architecture", "question_count"]
FIELDS.extend(("semantic_scored_count", "semantic_non_abstained_scored_count",
               "judge_error_count", "correct_count", "answer_correctness_lower_bound",
               "answer_correctness_upper_bound"))
for metric in METRICS:
    FIELDS.extend((f"{metric}_n", f"{metric}_mean"))


def keyed_rows(path: Path, manifest: dict, architectures: set[str]) -> dict[tuple[str, str], dict]:
    expected = {(arch, qid) for arch in architectures for qid in manifest["question_ids"]}
    keyed = {}
    for row in read_csv(path):
        key = (row["architecture"], row["question_id"])
        if key in keyed or row.get("run_id") != manifest["run_id"]:
            raise ValueError(f"Duplicate or wrong-run row in {path}: {key}")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError(f"{path} has incomplete or extra architecture/question rows.")
    return keyed


def evaluate(manifest: dict, native: dict, tables: dict[str, dict]) -> list[dict]:
    groups = defaultdict(list)
    for arch in ARCHITECTURES:
        for qid in manifest["question_ids"]:
            labels = native[qid]
            combined = {"architecture": arch, "question_id": qid}
            for table in tables.values():
                combined.update(table.get((arch, qid), {}))
            for grouping in ("question_type", "question_reasoning"):
                label = labels.get(grouping)
                if not isinstance(label, str) or not label.strip():
                    if grouping == "question_reasoning" and label is None:
                        continue  # Native source leaves this field unannotated for some questions.
                    raise ValueError(f"Missing {grouping} for {qid}")
                groups[(grouping, label, arch)].append(combined)
    output = []
    for (grouping, category, arch), members in sorted(groups.items()):
        row = {"table_title": "Table Q1: Native FinanceBench category performance",
               "run_id": manifest["run_id"], "grouping": grouping, "category": category,
               "architecture": arch, "question_count": len(members)}
        for metric in METRICS:
            values = [float(member[metric]) for member in members
                      if member.get(metric, "") not in ("", None)]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite {metric} in {grouping}/{category}/{arch}")
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = sum(values) / len(values) if values else ""
        errors = [member for member in members if member.get("answer_quality_label") == "judge_error"]
        if any(member.get(metric, "") not in ("", None) for member in errors
               for metric in ("answer_correctness", "answer_completeness")):
            raise ValueError("Judge-error semantic scores must remain missing, not zero")
        semantic_members = [member for member in members
                            if member.get("answer_correctness", "") not in ("", None)]
        correct_count = sum(float(member["answer_correctness"]) == 1 for member in semantic_members)
        row.update(semantic_scored_count=len(semantic_members), judge_error_count=len(errors),
                   semantic_non_abstained_scored_count=sum(
                       member.get("exact_abstention") in (0, "0", False) for member in semantic_members),
                   correct_count=correct_count)
        has_quality = all("answer_quality_label" in member for member in members)
        row["answer_correctness_lower_bound"] = correct_count / len(members) if has_quality else ""
        row["answer_correctness_upper_bound"] = (
            (correct_count + len(errors)) / len(members) if has_quality else "")
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace category summary and provenance after validated staging")
    args = parser.parse_args()
    if not args.overwrite:
        refuse_existing(SUMMARY, PROVENANCE)
    manifest, _ = load_run(args.manifest)
    required = (ANSWER, RETRIEVAL, FAITHFULNESS, LATENCY)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Stage 19 requires completed evaluation CSVs; missing: "
                         + ", ".join(str(path) for path in missing))
    source = Path(manifest["question_source"])
    native = {row["financebench_id"]: row for row in load_jsonl(source)}
    if set(native) != set(manifest["question_ids"]):
        raise ValueError("Native label IDs differ from the frozen evaluation split.")
    tables = {"answer": keyed_rows(ANSWER, manifest, set(ARCHITECTURES)),
              "retrieval": keyed_rows(RETRIEVAL, manifest, set(ARCHITECTURES[1:])),
              "faithfulness": keyed_rows(FAITHFULNESS, manifest, set(ARCHITECTURES)),
              "latency": keyed_rows(LATENCY, manifest, set(ARCHITECTURES))}
    rows = evaluate(manifest, native, tables)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="categories-", dir=SUMMARY.parent) as temporary:
        staged_summary = Path(temporary) / SUMMARY.name
        staged_provenance = Path(temporary) / PROVENANCE.name
        write_csv(staged_summary, rows, FIELDS)
        write_provenance(staged_provenance, "Stage 19: Native FinanceBench category performance", args.manifest,
                     {"native_labels": source, "answer_metrics": ANSWER, "retrieval_metrics": RETRIEVAL,
                      "faithfulness_metrics": FAITHFULNESS, "latency_metrics": LATENCY},
                     {"categories": ["question_type", "question_reasoning"],
                      "label_coverage": {grouping: sum(isinstance(native[qid].get(grouping), str)
                                                        and bool(native[qid][grouping].strip())
                                                        for qid in manifest["question_ids"])
                                         for grouping in ("question_type", "question_reasoning")},
                      "missing_reasoning_policy": "Exclude source-null reasoning labels from reasoning groups; retain those questions in question_type groups",
                      "aggregation": "arithmetic mean over applicable observations; NA excluded, denominator saved per metric",
                      "answer_quality": "post-hoc model-judged whole-answer correctness and essential-reference-unit completeness; semantic means exclude judge_error rows, per-metric n and error counts saved; correctness bounds over all category questions treat every judge error as incorrect/correct respectively; BERTScore is similarity, not accuracy",
                      "derived_labels": "none"}, {"summary": staged_summary})
        # Keep persisted output paths valid after moving the staged files.
        payload = json.loads(staged_provenance.read_text(encoding="utf-8"))
        payload["outputs"]["summary"] = str(SUMMARY.resolve())
        staged_provenance.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
        staged_summary.replace(SUMMARY)
        staged_provenance.replace(PROVENANCE)
    print(f"Table Q1: Native FinanceBench category performance: {len(rows)} rows in {SUMMARY}")


if __name__ == "__main__":
    main()
