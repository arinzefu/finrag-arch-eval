#!/usr/bin/env python
"""Stage 20: prespecified paired tests on the frozen P0-P3 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest, rankdata, wilcoxon

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_question_types import keyed_rows
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_run,
                                       refuse_existing, write_csv, write_provenance)

SOURCES = {
    "answer": ROOT / "results/evaluation/answer_metrics_v2.csv",
    "retrieval": ROOT / "results/evaluation/retrieval_metrics.csv",
    "faithfulness": ROOT / "results/evaluation/faithfulness_metrics.csv",
    "latency": ROOT / "results/evaluation/latency_metrics.csv",
}
RESULTS = ROOT / "results/statistics/hypothesis_tests.csv"
ABSTENTIONS = ROOT / "results/statistics/abstention_summary.csv"
PROVENANCE = ROOT / "results/statistics/hypothesis_tests_manifest.json"
BOOTSTRAP_REPLICATES = 10_000
ALPHA = 0.05

# A positive B-minus-A difference is the hypothesized direction in every row.
# H0's P0/P1 gold-support contrast also tests H1 (higher support means less
# hallucination), so it is tested once rather than duplicated.
COMPARISONS = tuple(
    ("H0,H1" if arch == "P1" else "H0", "P0", arch,
     "gold_evidence_support", "faithfulness", "scored_units_only")
    for arch in ARCHITECTURES[1:]
) + tuple(
    ("H0", "P0", arch, metric, source, population)
    for arch in ARCHITECTURES[1:]
    for metric, source, population in (
        ("exact_match", "answer", "all_questions"),
        ("token_f1", "answer", "all_questions"),
        ("numeric_accuracy", "answer", "gold_numeric_questions"),
        ("total_latency_ms", "latency", "all_questions"),
    )
) + tuple(
    ("H2", "P1", "P2", metric, source, population)
    for metric, source, population in (
        ("hit_at_5", "retrieval", "all_questions"),
        ("recall_at_5", "retrieval", "all_questions"),
        ("mrr_at_5", "retrieval", "all_questions"),
        ("numeric_accuracy", "answer", "gold_numeric_questions"),
    )
) + (
    ("H3", "P2", "P3", "context_faithfulness", "faithfulness", "scored_units_only"),
    ("H3", "P2", "P3", "total_latency_ms", "latency", "all_questions"),
)
BINARY_METRICS = {"exact_match", "hit_at_5"}
FIELDS = ["table_title", "run_id", "hypothesis", "architecture_a", "architecture_b",
          "metric", "population", "test", "pair_count", "excluded_pair_count",
          "mean_a_paired", "mean_b_paired", "mean_difference_b_minus_a",
          "difference_ci95_low", "difference_ci95_high", "p_value_two_sided",
          "holm_p_value", "significant_holm_0_05", "rank_biserial_b_minus_a",
          "a_only_success", "b_only_success"]
ABSTENTION_FIELDS = ["table_title", "run_id", "architecture", "question_count",
                     "exact_abstention_count", "exact_abstention_rate",
                     "context_scored_count", "gold_scored_count"]


def paired_values(table: dict[tuple[str, str], dict], question_ids: list[str],
                  arch_a: str, arch_b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for qid in question_ids:
        a = table[(arch_a, qid)][metric]
        b = table[(arch_b, qid)][metric]
        if a == "" or b == "":
            continue
        values = (float(a), float(b))
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {metric} for {qid}")
        left.append(values[0])
        right.append(values[1])
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def paired_bootstrap_ci(differences: np.ndarray, seed: int,
                        replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    if not len(differences):
        raise ValueError("Cannot bootstrap zero pairs")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_test(a: np.ndarray, b: np.ndarray, binary: bool) -> tuple[str, float, float | str, int | str, int | str]:
    differences = b - a
    if binary:
        if not np.all(np.isin(a, [0, 1])) or not np.all(np.isin(b, [0, 1])):
            raise ValueError("McNemar requires binary paired observations")
        a_only = int(np.sum((a == 1) & (b == 0)))
        b_only = int(np.sum((a == 0) & (b == 1)))
        discordant = a_only + b_only
        p_value = binomtest(min(a_only, b_only), discordant, 0.5).pvalue if discordant else 1.0
        return "McNemar exact, two-sided", float(p_value), "", a_only, b_only
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return "Wilcoxon signed-rank, two-sided", 1.0, 0.0, "", ""
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    effect = (positive - negative) / (positive + negative)
    p_value = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided",
                       method="auto").pvalue
    return "Wilcoxon signed-rank, two-sided", float(p_value), effect, "", ""


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def evaluate(manifest: dict, tables: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    question_ids = manifest["question_ids"]
    results = []
    for hypothesis, arch_a, arch_b, metric, source, population in COMPARISONS:
        a, b = paired_values(tables[source], question_ids, arch_a, arch_b, metric)
        if not len(a):
            raise ValueError(f"No valid pairs for {arch_a}/{arch_b} {metric}")
        difference = b - a
        seed_material = f"{manifest['run_id']}:{arch_a}:{arch_b}:{metric}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        low, high = paired_bootstrap_ci(difference, seed)
        test, p_value, effect, a_only, b_only = paired_test(a, b, metric in BINARY_METRICS)
        results.append({
            "table_title": "Table S1: Prespecified paired architecture tests",
            "run_id": manifest["run_id"], "hypothesis": hypothesis,
            "architecture_a": arch_a, "architecture_b": arch_b, "metric": metric,
            "population": population, "test": test, "pair_count": len(a),
            "excluded_pair_count": len(question_ids) - len(a),
            "mean_a_paired": float(a.mean()), "mean_b_paired": float(b.mean()),
            "mean_difference_b_minus_a": float(difference.mean()),
            "difference_ci95_low": low, "difference_ci95_high": high,
            "p_value_two_sided": p_value, "holm_p_value": "",
            "significant_holm_0_05": "", "rank_biserial_b_minus_a": effect,
            "a_only_success": a_only, "b_only_success": b_only,
        })
    for row, adjusted in zip(results, holm_adjust([r["p_value_two_sided"] for r in results])):
        row["holm_p_value"] = adjusted
        row["significant_holm_0_05"] = int(adjusted < ALPHA)

    faithfulness = tables["faithfulness"]
    abstentions = []
    for arch in ARCHITECTURES:
        records = [faithfulness[(arch, qid)] for qid in question_ids]
        count = sum(int(record["abstained"]) for record in records)
        abstentions.append({
            "table_title": "Table S2: Abstention and support-score coverage",
            "run_id": manifest["run_id"], "architecture": arch,
            "question_count": len(records), "exact_abstention_count": count,
            "exact_abstention_rate": count / len(records),
            "context_scored_count": sum(record["context_faithfulness"] != "" for record in records),
            "gold_scored_count": sum(record["gold_evidence_support"] != "" for record in records),
        })
    return results, abstentions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    refuse_existing(RESULTS, ABSTENTIONS, PROVENANCE)
    manifest, _ = load_run(args.manifest)
    missing = [path for path in SOURCES.values() if not path.exists()]
    if missing:
        raise SystemExit("Stage 20 requires completed evaluation CSVs; missing: "
                         + ", ".join(str(path) for path in missing))
    tables = {name: keyed_rows(path, manifest,
               set(ARCHITECTURES[1:] if name == "retrieval" else ARCHITECTURES))
              for name, path in SOURCES.items()}
    rows, abstentions = evaluate(manifest, tables)
    write_csv(RESULTS, rows, FIELDS)
    write_csv(ABSTENTIONS, abstentions, ABSTENTION_FIELDS)
    write_provenance(PROVENANCE, "Stage 20: Prespecified paired architecture tests",
                     args.manifest, SOURCES,
                     {"comparisons": [{"hypothesis": h, "architecture_a": a,
                                       "architecture_b": b, "metric": m,
                                       "population": pop} for h, a, b, m, _, pop in COMPARISONS],
                      "direction": "architecture_b minus architecture_a; positive is hypothesized",
                      "tests": "two-sided exact McNemar for binary; two-sided Wilcoxon signed-rank with zero differences removed for scores",
                      "effect": "signed-rank biserial (W+ - W-) / (W+ + W-), nonzero differences only; NA for McNemar",
                      "missing_pairs": "pairwise complete cases, never zero-impute abstentions or inapplicable numeric scores",
                      "confidence_interval": "paired percentile bootstrap of mean B-A difference",
                      "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                      "bootstrap_seed": "first 8 bytes of SHA256(run_id:architecture_a:architecture_b:metric)",
                      "multiple_testing": "Holm adjustment over all prespecified rows",
                      "alpha": ALPHA, "numpy_version": np.__version__,
                      "scipy_version": scipy.__version__},
                     {"hypothesis_tests": RESULTS, "abstention_summary": ABSTENTIONS})
    print(f"Table S1: {len(rows)} paired tests saved to {RESULTS}")
    print(f"Table S2: Abstention coverage saved to {ABSTENTIONS}")


if __name__ == "__main__":
    main()
