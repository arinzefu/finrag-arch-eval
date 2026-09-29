#!/usr/bin/env python
"""Stage 21: unweighted quality-latency trade-offs for the frozen architectures."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_question_types import keyed_rows
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_run, read_csv,
                                       refuse_existing, write_csv, write_provenance)

SOURCES = {
    "answer": ROOT / "results/evaluation/answer_metrics_v2.csv",
    "retrieval": ROOT / "results/evaluation/retrieval_metrics.csv",
    "faithfulness": ROOT / "results/evaluation/faithfulness_metrics.csv",
    "latency": ROOT / "results/evaluation/latency_metrics.csv",
}
ABSTENTIONS = ROOT / "results/statistics/abstention_summary.csv"
SUMMARY = ROOT / "results/aggregated/tradeoff_summary.csv"
FRONTIERS = ROOT / "results/aggregated/pareto_frontiers.csv"
PAIRS = ROOT / "results/aggregated/paired_tradeoffs.csv"
PROVENANCE = ROOT / "results/aggregated/tradeoff_summary_manifest.json"

OBJECTIVES = (
    ("answer_token_f1", "token_f1", ARCHITECTURES),
    ("context_faithfulness", "context_faithfulness", ARCHITECTURES[1:]),
    ("recall_at_5", "recall_at_5", ARCHITECTURES[1:]),
    ("gold_evidence_support", "gold_evidence_support", ARCHITECTURES),
)
METRIC_SOURCES = (
    ("token_f1", "answer"), ("numeric_accuracy", "answer"),
    ("gold_evidence_support", "faithfulness"),
    ("context_faithfulness", "faithfulness"),
    ("recall_at_5", "retrieval"), ("total_latency_ms", "latency"),
    ("abstained", "faithfulness"),
)
SUMMARY_FIELDS = [
    "table_title", "run_id", "architecture", "question_count", "exact_abstention_count",
    "exact_abstention_rate", "answer_token_f1_mean", "numeric_accuracy_mean",
    "numeric_accuracy_n", "hit_at_5_mean", "recall_at_5_mean", "mrr_at_5_mean",
    "retrieval_n", "context_faithfulness_mean", "context_faithfulness_n",
    "gold_evidence_support_mean", "gold_evidence_support_n", "total_latency_mean_ms",
    "total_latency_median_ms", "pareto_answer_token_f1", "pareto_context_faithfulness",
    "pareto_recall_at_5", "pareto_gold_evidence_support",
]
FRONTIER_FIELDS = ["table_title", "run_id", "objective", "architecture",
                   "quality_mean", "quality_scored_n", "quality_coverage_rate",
                   "exact_abstention_rate", "mean_total_latency_ms", "pareto_nondominated",
                   "dominated_by"]
PAIR_FIELDS = ["table_title", "run_id", "architecture_a", "architecture_b", "metric",
               "pair_count", "excluded_pair_count", "mean_a_paired", "mean_b_paired",
               "mean_difference_b_minus_a", "percent_change_from_a"]


def values(table: dict, question_ids: list[str], arch: str, metric: str) -> list[float]:
    result = []
    for qid in question_ids:
        value = table[(arch, qid)][metric]
        if value != "":
            result.append(float(value))
    return result


def dominates(a_quality: float, a_latency: float,
              b_quality: float, b_latency: float) -> bool:
    """A dominates B when no worse on both axes and strictly better on one."""
    return (a_quality >= b_quality and a_latency <= b_latency
            and (a_quality > b_quality or a_latency < b_latency))


def validate_abstentions(manifest: dict, faithfulness: dict) -> None:
    rows = read_csv(ABSTENTIONS)
    if len(rows) != len(ARCHITECTURES) or {r["architecture"] for r in rows} != set(ARCHITECTURES):
        raise ValueError("Stage 20 abstention summary has wrong architecture coverage")
    for row in rows:
        arch = row["architecture"]
        if row["run_id"] != manifest["run_id"] or int(row["question_count"]) != len(manifest["question_ids"]):
            raise ValueError(f"Stage 20 abstention summary does not match frozen run: {arch}")
        records = [faithfulness[(arch, qid)] for qid in manifest["question_ids"]]
        count = sum(int(r["abstained"]) for r in records)
        context_n = sum(r["context_faithfulness"] != "" for r in records)
        gold_n = sum(r["gold_evidence_support"] != "" for r in records)
        if (count != int(row["exact_abstention_count"])
                or context_n != int(row["context_scored_count"])
                or gold_n != int(row["gold_scored_count"])):
            raise ValueError(f"Stage 20 abstention counts disagree with per-question scores: {arch}")


def evaluate(manifest: dict, tables: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ids = manifest["question_ids"]
    n = len(ids)
    summary = []
    for arch in ARCHITECTURES:
        answer = tables["answer"]
        faithfulness = tables["faithfulness"]
        latency = values(tables["latency"], ids, arch, "total_latency_ms")
        numeric = values(answer, ids, arch, "numeric_accuracy")
        context = values(faithfulness, ids, arch, "context_faithfulness")
        gold = values(faithfulness, ids, arch, "gold_evidence_support")
        abstentions = sum(int(faithfulness[(arch, qid)]["abstained"]) for qid in ids)
        retrieved = arch != "P0"
        summary.append({
            "table_title": "Table T1: Architecture quality, coverage, and latency",
            "run_id": manifest["run_id"], "architecture": arch, "question_count": n,
            "exact_abstention_count": abstentions, "exact_abstention_rate": abstentions / n,
            "answer_token_f1_mean": statistics.mean(values(answer, ids, arch, "token_f1")),
            "numeric_accuracy_mean": statistics.mean(numeric) if numeric else "",
            "numeric_accuracy_n": len(numeric),
            "hit_at_5_mean": statistics.mean(values(tables["retrieval"], ids, arch, "hit_at_5")) if retrieved else "",
            "recall_at_5_mean": statistics.mean(values(tables["retrieval"], ids, arch, "recall_at_5")) if retrieved else "",
            "mrr_at_5_mean": statistics.mean(values(tables["retrieval"], ids, arch, "mrr_at_5")) if retrieved else "",
            "retrieval_n": n if retrieved else 0,
            "context_faithfulness_mean": statistics.mean(context) if context else "",
            "context_faithfulness_n": len(context),
            "gold_evidence_support_mean": statistics.mean(gold) if gold else "",
            "gold_evidence_support_n": len(gold),
            "total_latency_mean_ms": statistics.mean(latency),
            "total_latency_median_ms": statistics.median(latency),
            **{f"pareto_{name}": "" for name, _, _ in OBJECTIVES},
        })
    by_arch = {row["architecture"]: row for row in summary}
    frontiers = []
    for objective, quality_field, eligible in OBJECTIVES:
        for arch in eligible:
            row = by_arch[arch]
            quality = float(row[f"{objective}_mean"])
            latency = float(row["total_latency_mean_ms"])
            dominated_by = [other for other in eligible if other != arch and dominates(
                float(by_arch[other][f"{objective}_mean"]),
                float(by_arch[other]["total_latency_mean_ms"]), quality, latency)]
            scored_n = (row["context_faithfulness_n"] if objective == "context_faithfulness"
                        else row["gold_evidence_support_n"] if objective == "gold_evidence_support"
                        else row["retrieval_n"] if objective == "recall_at_5" else n)
            nondominated = int(not dominated_by)
            row[f"pareto_{objective}"] = nondominated
            frontiers.append({
                "table_title": "Table T2: Unweighted quality-latency Pareto frontiers",
                "run_id": manifest["run_id"], "objective": objective, "architecture": arch,
                "quality_mean": quality, "quality_scored_n": scored_n,
                "quality_coverage_rate": scored_n / n,
                "exact_abstention_rate": row["exact_abstention_rate"],
                "mean_total_latency_ms": latency, "pareto_nondominated": nondominated,
                "dominated_by": ",".join(dominated_by),
            })

    pairs = []
    for arch_a, arch_b in zip(ARCHITECTURES, ARCHITECTURES[1:]):
        for metric, source in METRIC_SOURCES:
            if arch_a == "P0" and metric in {"context_faithfulness", "recall_at_5"}:
                continue
            a, b = [], []
            for qid in ids:
                av = tables[source][(arch_a, qid)][metric]
                bv = tables[source][(arch_b, qid)][metric]
                if av != "" and bv != "":
                    a.append(float(av))
                    b.append(float(bv))
            if not a:
                raise ValueError(f"No paired observations for {arch_a}/{arch_b} {metric}")
            mean_a, mean_b = statistics.mean(a), statistics.mean(b)
            pairs.append({
                "table_title": "Table T3: Adjacent-architecture paired trade-offs",
                "run_id": manifest["run_id"], "architecture_a": arch_a,
                "architecture_b": arch_b, "metric": metric, "pair_count": len(a),
                "excluded_pair_count": n - len(a), "mean_a_paired": mean_a,
                "mean_b_paired": mean_b, "mean_difference_b_minus_a": mean_b - mean_a,
                "percent_change_from_a": (100 * (mean_b - mean_a) / mean_a
                                          if metric == "total_latency_ms" and mean_a else ""),
            })
    return summary, frontiers, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    refuse_existing(SUMMARY, FRONTIERS, PAIRS, PROVENANCE)
    manifest, _ = load_run(args.manifest)
    inputs = {**SOURCES, "abstentions": ABSTENTIONS}
    missing = [path for path in inputs.values() if not path.exists()]
    if missing:
        raise SystemExit("Stage 21 requires completed evaluation files; missing: "
                         + ", ".join(str(path) for path in missing))
    tables = {name: keyed_rows(path, manifest,
               set(ARCHITECTURES[1:] if name == "retrieval" else ARCHITECTURES))
              for name, path in SOURCES.items()}
    validate_abstentions(manifest, tables["faithfulness"])
    summary, frontiers, pairs = evaluate(manifest, tables)
    write_csv(SUMMARY, summary, SUMMARY_FIELDS)
    write_csv(FRONTIERS, frontiers, FRONTIER_FIELDS)
    write_csv(PAIRS, pairs, PAIR_FIELDS)
    write_provenance(PROVENANCE, "Stage 21: Unweighted architecture trade-offs",
                     args.manifest, inputs,
                     {"objectives": [{"name": name, "quality_metric": metric,
                                     "eligible_architectures": list(eligible)}
                                    for name, metric, eligible in OBJECTIVES],
                      "pareto_rule": "Maximize macro mean quality, minimize mean total latency; at least one strict improvement",
                      "weights": "none", "support_scores": "Conditional on scored factual units; NA abstentions not imputed as zero",
                      "coverage": "Scored denominator and exact-abstention rate shown alongside support frontiers",
                      "paired_differences": "B minus A on shared non-NA question IDs; descriptive, no additional significance tests",
                      "latency_scope": "Recorded per-question total; includes model/API/network variability"},
                     {"summary": SUMMARY, "frontiers": FRONTIERS, "paired_tradeoffs": PAIRS})
    print(f"Table T1: {len(summary)} architecture rows in {SUMMARY}")
    print(f"Table T2: {len(frontiers)} Pareto rows in {FRONTIERS}")
    print(f"Table T3: {len(pairs)} adjacent-architecture rows in {PAIRS}")


if __name__ == "__main__":
    main()
