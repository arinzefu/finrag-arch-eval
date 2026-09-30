#!/usr/bin/env python
"""Stage 22: publish titled figures and tables from frozen results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_run,
                                       read_csv, refuse_existing, sha256, write_csv)
from src.retrieval.reranker import DEFAULT_RERANKER_MODEL

INPUTS = {
    "retrieval": ROOT / "results/aggregated/retrieval_summary.csv",
    "answers": ROOT / "results/aggregated/answer_summary_v2.csv",
    "faithfulness": ROOT / "results/aggregated/faithfulness_summary.csv",
    "latency": ROOT / "results/aggregated/latency_summary.csv",
    "categories": ROOT / "results/aggregated/question_type_summary.csv",
    "statistics": ROOT / "results/statistics/hypothesis_tests.csv",
    "tradeoffs": ROOT / "results/aggregated/tradeoff_summary.csv",
    "frontiers": ROOT / "results/aggregated/pareto_frontiers.csv",
    "category_manifest": ROOT / "results/aggregated/question_type_summary_manifest.json",
    "statistics_manifest": ROOT / "results/statistics/hypothesis_tests_manifest.json",
    "tradeoff_manifest": ROOT / "results/aggregated/tradeoff_summary_manifest.json",
    "reranker_source": ROOT / "src/retrieval/reranker.py",
}
FIGURE_TITLES = {
    "figure01_architecture_overview": "Figure 1. Frozen architecture overview",
    "figure02_recall_at_5": "Figure 2. Gold-page Recall@5",
    "figure03_answer_correctness": "Figure 3. Held-out answer correctness",
    "figure04_evidence_support": "Figure 4. Evidence support and scored-answer coverage",
    "figure05_total_latency": "Figure 5. Recorded total latency",
    "figure06_answer_quality_latency": "Figure 6. Answer quality versus latency",
    "figure06b_retrieval_latency": "Figure 6b. Retrieval quality versus latency",
    "figure06c_faithfulness_latency": "Figure 6c. Context faithfulness versus latency",
    "figure07_native_reasoning": "Figure 7. Answer quality by native reasoning category",
}
TABLE_TITLES = {
    "table1_architecture_configuration": "Table 1: Frozen architecture configuration",
    "table2_overall_results": "Table 2: Overall held-out results and coverage",
    "table3_question_category_results": "Table 3: Native FinanceBench category results",
    "table4_hypothesis_tests": "Table 4: Paired hypothesis tests",
}
MANIFEST_OUTPUT = ROOT / "results/final_outputs_manifest.json"
COLORS = {"P0": "#444444", "P1": "#0072B2", "P2": "#D55E00", "P3": "#009E73"}
TABLE1_FIELDS = ["table_title", "run_id", "architecture", "retrieval_design",
                 "dense_candidate_k", "bm25_candidate_k", "fusion_method", "rrf_k",
                 "fusion_candidate_k", "reranker_model", "reranker_max_length",
                 "reranker_window_overlap", "reranker_batch_size", "final_context_k",
                 "llm_model", "generation_max_output_tokens", "temperature_policy",
                 "prompt_policy"]
TABLE2_FIELDS = ["table_title", "run_id", "architecture", "question_count",
                 "hit_at_5", "recall_at_5", "mrr_at_5", "exact_match", "token_f1",
                 "numeric_accuracy", "numeric_applicable_n", "context_faithfulness",
                 "context_hallucination", "context_scored_n", "gold_evidence_support",
                 "gold_evidence_hallucination", "gold_scored_n", "exact_abstention_count",
                 "exact_abstention_rate", "total_latency_mean_ms", "total_latency_median_ms",
                 "total_latency_std_ms"]
TABLE3_FIELDS = ["table_title", "run_id", "grouping", "category", "architecture",
                 "question_count", "token_f1_mean", "token_f1_n", "numeric_accuracy_mean",
                 "numeric_accuracy_n", "recall_at_5_mean", "recall_at_5_n",
                 "context_faithfulness_mean", "context_faithfulness_n",
                 "gold_evidence_support_mean", "gold_evidence_support_n",
                 "total_latency_ms_mean", "total_latency_ms_n"]


def keyed_architecture(rows: list[dict], expected: tuple[str, ...], run_id: str,
                       source: str) -> dict[str, dict]:
    if len(rows) != len(expected) or {r["architecture"] for r in rows} != set(expected):
        raise ValueError(f"{source} has missing or duplicate architecture rows")
    if any(r["run_id"] != run_id for r in rows):
        raise ValueError(f"{source} belongs to a different run")
    return {r["architecture"]: r for r in rows}


def load_inputs(manifest: dict, manifest_path: Path = MANIFEST) -> dict:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage 22 inputs missing: " + ", ".join(missing))
    run_id = manifest["run_id"]
    data = {}
    stage_manifests = {name: json.loads(INPUTS[name].read_text(encoding="utf-8"))
                       for name in ("category_manifest", "statistics_manifest", "tradeoff_manifest")}
    if any(item["run_manifest_sha256"] != sha256(manifest_path) for item in stage_manifests.values()):
        raise ValueError("An upstream evaluation manifest belongs to a different frozen run")
    for name in ("retrieval", "answers", "faithfulness", "tradeoffs"):
        expected = ARCHITECTURES[1:] if name == "retrieval" else ARCHITECTURES
        data[name] = keyed_architecture(read_csv(INPUTS[name]), expected, run_id, name)
    latency_rows = read_csv(INPUTS["latency"])
    if any(row["run_id"] != run_id for row in latency_rows):
        raise ValueError("Latency summary belongs to a different run")
    latency = {(row["architecture"], row["component"]): row for row in latency_rows}
    if len(latency) != len(latency_rows) or any((arch, "total") not in latency for arch in ARCHITECTURES):
        raise ValueError("Latency summary has duplicate or missing total rows")
    data["latency"] = latency

    categories = read_csv(INPUTS["categories"])
    if not categories or any(row["run_id"] != run_id for row in categories):
        raise ValueError("Native category table is empty or from another run")
    category_keys = {(r["grouping"], r["category"], r["architecture"]) for r in categories}
    if len(category_keys) != len(categories):
        raise ValueError("Duplicate native category row")
    for grouping in ("question_type", "question_reasoning"):
        group_rows = [r for r in categories if r["grouping"] == grouping]
        labels = {r["category"] for r in group_rows}
        if not labels:
            raise ValueError(f"No native {grouping} groups")
        for label in labels:
            members = [r for r in group_rows if r["category"] == label]
            if {r["architecture"] for r in members} != set(ARCHITECTURES):
                raise ValueError(f"Incomplete native category {grouping}/{label}")
            if len({int(r["question_count"]) for r in members}) != 1:
                raise ValueError(f"Unequal category counts for {grouping}/{label}")
        coverage = sum(int(r["question_count"]) for r in group_rows if r["architecture"] == "P0")
        expected_coverage = stage_manifests["category_manifest"]["parameters"]["label_coverage"][grouping]
        if coverage != expected_coverage:
            raise ValueError(f"Unexpected native {grouping} coverage: {coverage}")
    data["categories"] = categories

    statistics = read_csv(INPUTS["statistics"])
    expected_tests = len(stage_manifests["statistics_manifest"]["parameters"]["comparisons"])
    if len(statistics) != expected_tests or any(row["run_id"] != run_id for row in statistics):
        raise ValueError("Paired-test table is incomplete or from another run")
    data["statistics"] = statistics
    frontiers = read_csv(INPUTS["frontiers"])
    expected_frontiers = sum(len(item["eligible_architectures"])
                             for item in stage_manifests["tradeoff_manifest"]["parameters"]["objectives"])
    if len(frontiers) != expected_frontiers or any(row["run_id"] != run_id for row in frontiers):
        raise ValueError("Stage 21 Pareto table is incomplete or from another run")
    data["frontiers"] = frontiers

    for arch in ARCHITECTURES:
        answer, faith, trade = data["answers"][arch], data["faithfulness"][arch], data["tradeoffs"][arch]
        total = data["latency"][(arch, "total")]
        for left, right in ((answer["token_f1"], trade["answer_token_f1_mean"]),
                            (faith["gold_evidence_support"], trade["gold_evidence_support_mean"]),
                            (total["mean_ms"], trade["total_latency_mean_ms"])):
            if not math.isclose(float(left), float(right), abs_tol=1e-9):
                raise ValueError(f"Cross-stage summary mismatch for {arch}")
        if arch != "P0":
            if not math.isclose(float(data["retrieval"][arch]["recall_at_5"]),
                                float(trade["recall_at_5_mean"]), abs_tol=1e-9):
                raise ValueError(f"Retrieval/trade-off mismatch for {arch}")
    return data


def architecture_configuration(manifest: dict) -> list[dict]:
    settings = manifest["settings"]
    model = manifest["model"]
    temperature = ("API default (unset)" if manifest["temperature"] is None
                   else str(manifest["temperature"]))
    result = []
    for arch in ARCHITECTURES:
        row = dict.fromkeys(TABLE1_FIELDS, "")
        row.update(table_title=TABLE_TITLES["table1_architecture_configuration"],
                   run_id=manifest["run_id"], architecture=arch, llm_model=model,
                   generation_max_output_tokens=manifest["max_output_tokens"],
                   temperature_policy=temperature,
                   final_context_k=0 if arch == "P0" else settings["final_context_k"],
                   prompt_policy="closed_book" if arch == "P0" else "shared_rag_p1_p3")
        if arch == "P0":
            row["retrieval_design"] = "closed_book"
        elif arch == "P1":
            row.update(retrieval_design="dense", dense_candidate_k=settings["p1_dense_k"])
        elif arch == "P2":
            row.update(retrieval_design="hybrid_rrf", dense_candidate_k=settings["p2_dense_k"],
                       bm25_candidate_k=settings["p2_sparse_k"], fusion_method="RRF",
                       rrf_k=settings["rrf_k"], fusion_candidate_k=settings["p2_fusion_k"])
        else:
            row.update(retrieval_design="hybrid_rrf_cross_encoder",
                       dense_candidate_k=settings["p3_dense_k"],
                       bm25_candidate_k=settings["p3_sparse_k"], fusion_method="RRF",
                       rrf_k=settings["rrf_k"], fusion_candidate_k=settings["p3_fusion_k"],
                       reranker_model=DEFAULT_RERANKER_MODEL,
                       reranker_max_length=settings["reranker_max_length"],
                       reranker_window_overlap=settings["reranker_window_overlap"],
                       reranker_batch_size=settings["reranker_batch_size"])
        result.append(row)
    return result


def build_tables(manifest: dict, data: dict) -> dict[str, tuple[list[dict], list[str]]]:
    table1 = architecture_configuration(manifest)
    table2 = []
    for arch in ARCHITECTURES:
        answer, faith, trade = data["answers"][arch], data["faithfulness"][arch], data["tradeoffs"][arch]
        retrieval = data["retrieval"].get(arch, {})
        latency = data["latency"][(arch, "total")]
        table2.append({
            "table_title": TABLE_TITLES["table2_overall_results"], "run_id": manifest["run_id"],
            "architecture": arch, "question_count": manifest["question_count"],
            "hit_at_5": retrieval.get("hit_at_5", ""),
            "recall_at_5": retrieval.get("recall_at_5", ""),
            "mrr_at_5": retrieval.get("mrr_at_5", ""),
            "exact_match": answer["exact_match"], "token_f1": answer["token_f1"],
            "numeric_accuracy": answer["numeric_accuracy"],
            "numeric_applicable_n": answer["numeric_applicable_count"],
            "context_faithfulness": faith["context_faithfulness"],
            "context_hallucination": faith["context_hallucination"],
            "context_scored_n": faith["context_applicable_count"],
            "gold_evidence_support": faith["gold_evidence_support"],
            "gold_evidence_hallucination": faith["gold_evidence_hallucination"],
            "gold_scored_n": faith["gold_applicable_count"],
            "exact_abstention_count": trade["exact_abstention_count"],
            "exact_abstention_rate": trade["exact_abstention_rate"],
            "total_latency_mean_ms": latency["mean_ms"],
            "total_latency_median_ms": latency["median_ms"],
            "total_latency_std_ms": latency["std_ms"],
        })
    table3 = [{field: (TABLE_TITLES["table3_question_category_results"] if field == "table_title"
                       else row.get(field, "")) for field in TABLE3_FIELDS}
              for row in data["categories"]]
    with INPUTS["statistics"].open(encoding="utf-8", newline="") as handle:
        source_fields = list(csv.DictReader(handle).fieldnames or [])
    table4 = [{**row, "table_title": TABLE_TITLES["table4_hypothesis_tests"]}
              for row in data["statistics"]]
    return {
        "table1_architecture_configuration": (table1, TABLE1_FIELDS),
        "table2_overall_results": (table2, TABLE2_FIELDS),
        "table3_question_category_results": (table3, TABLE3_FIELDS),
        "table4_hypothesis_tests": (table4, source_fields),
    }


def plot_architecture(manifest: dict, configuration: list[dict]):
    fig, ax = plt.subplots(figsize=(11.8, 3.5))
    ax.axis("off")
    labels = ["Dense", "BM25", "Fusion", "Reranker", "Final context", "LLM"]
    matrix = []
    for row in configuration:
        fusion = (f"RRF k={row['rrf_k']}\ntop {row['fusion_candidate_k']}"
                  if row["fusion_method"] else "-")
        reranker = (f"Cross-encoder\n{row['reranker_max_length']} tokens"
                    if row["reranker_model"] else "-")
        matrix.append([f"top {row['dense_candidate_k']}" if row["dense_candidate_k"] else "-",
                       f"top {row['bm25_candidate_k']}" if row["bm25_candidate_k"] else "-",
                       fusion, reranker,
                       str(row["final_context_k"]) if row["final_context_k"] else "-",
                       row["llm_model"]])
    table = ax.table(cellText=matrix, rowLabels=list(ARCHITECTURES), colLabels=labels,
                     cellLoc="center", rowLoc="center", bbox=[.02, .16, .96, .74])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D3D9DD")
        if row == 0:
            cell.set_facecolor("#DCE7EB")
            cell.set_text_props(weight="bold")
        elif col == -1:
            cell.set_facecolor(COLORS[ARCHITECTURES[row - 1]])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if matrix[row - 1][col] != "-" else "#F4F6F7")
    fig.suptitle(FIGURE_TITLES["figure01_architecture_overview"], fontsize=13, weight="bold")
    fig.text(.5, .04,
             f"Frozen run: {manifest['run_id']} | Shared output limit: {manifest['max_output_tokens']} tokens | "
             "P1-P3 use one RAG prompt builder", ha="center", fontsize=9, color="#4B5563")
    return fig


def plot_recall(data: dict):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    archs = ARCHITECTURES[1:]
    scores = [float(data["retrieval"][arch]["recall_at_5"]) for arch in archs]
    bars = ax.bar(archs, scores, color=[COLORS[a] for a in archs], width=.62)
    ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=4)
    ax.set_ylim(0, max(.5, max(scores) * 1.25))
    ax.set_ylabel("Gold-page Recall@5")
    ax.set_title(FIGURE_TITLES["figure02_recall_at_5"], weight="bold")
    ax.text(.5, -.16, "145 held-out questions per architecture; page-level evidence relevance",
            transform=ax.transAxes, ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def plot_answers(data: dict):
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    positions = np.arange(len(ARCHITECTURES))
    measures = (("Exact match", "exact_match", "#718096"),
                ("Token F1", "token_f1", "#0072B2"),
                ("Numeric accuracy", "numeric_accuracy", "#D55E00"))
    width = .24
    for index, (label, field, color) in enumerate(measures):
        scores = [float(data["answers"][arch][field]) for arch in ARCHITECTURES]
        bars = ax.bar(positions + (index - 1) * width, scores, width=width,
                      label=label, color=color)
        ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=2, fontsize=8)
    ax.set_xticks(positions, ARCHITECTURES)
    ax.set_ylim(0, .25)
    ax.set_ylabel("Mean score")
    ax.set_title(FIGURE_TITLES["figure03_answer_correctness"], weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(.5, -.16, "Numeric accuracy uses the corrected v2 extractor (112 applicable questions per architecture)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def plot_faithfulness(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharey=True)
    for ax, archs, field, count_field, subtitle in (
        (axes[0], ARCHITECTURES[1:], "context_faithfulness", "context_applicable_count",
         "Retrieved-context support"),
        (axes[1], ARCHITECTURES, "gold_evidence_support", "gold_applicable_count",
         "Annotated-evidence support"),
    ):
        scores = [float(data["faithfulness"][arch][field]) for arch in archs]
        labels = [f"{arch}\nn={data['faithfulness'][arch][count_field]}" for arch in archs]
        bars = ax.bar(labels, scores, color=[COLORS[a] for a in archs], width=.62)
        ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=3, fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Mean supported factual-unit fraction")
    fig.suptitle(FIGURE_TITLES["figure04_evidence_support"], weight="bold")
    fig.text(.5, .02, "Conditional on scored answers; P0 has no retrieved context. See judge audit and abstention counts.",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, .08, 1, .94))
    return fig


def plot_latency(data: dict):
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    totals = [data["latency"][(arch, "total")] for arch in ARCHITECTURES]
    means = [float(row["mean_ms"]) / 1000 for row in totals]
    medians = [float(row["median_ms"]) / 1000 for row in totals]
    stds = [float(row["std_ms"]) / 1000 for row in totals]
    positions = np.arange(len(ARCHITECTURES))
    ax.bar(positions, means, yerr=stds, color=[COLORS[a] for a in ARCHITECTURES],
           capsize=4, width=.6, label="Mean +/- 1 SD")
    ax.scatter(positions, medians, marker="D", color="white", edgecolor="#222222",
               zorder=3, label="Median")
    ax.set_xticks(positions, ARCHITECTURES)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.18)
    ax.set_ylabel("Recorded total latency (seconds)")
    ax.set_title(FIGURE_TITLES["figure05_total_latency"], weight="bold")
    ax.legend(frameon=False)
    ax.text(.5, -.16, "145 questions per architecture; pipeline/model initialization excluded",
            transform=ax.transAxes, ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def plot_tradeoff(data: dict, objective: str, figure_key: str, ylabel: str, ymax: float):
    rows = [row for row in data["frontiers"] if row["objective"] == objective]
    if not rows:
        raise ValueError(f"Missing Pareto objective {objective}")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    nondominated = sorted((row for row in rows if row["pareto_nondominated"] == "1"),
                          key=lambda row: float(row["mean_total_latency_ms"]))
    if len(nondominated) > 1:
        ax.plot([float(row["mean_total_latency_ms"]) / 1000 for row in nondominated],
                [float(row["quality_mean"]) for row in nondominated],
                color="#90989E", linestyle="--", linewidth=1, zorder=1)
    for row in rows:
        arch = row["architecture"]
        x = float(row["mean_total_latency_ms"]) / 1000
        y = float(row["quality_mean"])
        on_frontier = row["pareto_nondominated"] == "1"
        ax.scatter(x, y, s=125, marker="o", edgecolor=COLORS[arch], linewidth=2,
                   facecolor=COLORS[arch] if on_frontier else "white", zorder=3)
        ax.annotate(f"{arch} (n={row['quality_scored_n']})", (x, y),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, max(float(row["mean_total_latency_ms"]) for row in rows) / 1000 * 1.22)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Mean recorded total latency (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(FIGURE_TITLES[figure_key], weight="bold")
    note = ("Filled: Pareto nondominated; open: dominated. Support is conditional on scored answers."
            if objective == "context_faithfulness"
            else "Filled: Pareto nondominated; open: dominated. Point estimates only.")
    ax.text(.5, -.18, note, transform=ax.transAxes, ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, .07, 1, 1))
    return fig


def plot_native_reasoning(data: dict):
    rows = [row for row in data["categories"] if row["grouping"] == "question_reasoning"]
    categories = sorted({row["category"] for row in rows})
    lookup = {(row["category"], row["architecture"]): row for row in rows}
    matrix = np.array([[float(lookup[(category, arch)]["token_f1_mean"])
                        for arch in ARCHITECTURES] for category in categories])
    counts = [int(lookup[(category, "P0")]["question_count"]) for category in categories]
    fig, ax = plt.subplots(figsize=(9.8, max(5.1, .5 * len(categories) + 1.6)))
    display = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(.35, float(matrix.max())), aspect="auto")
    ax.set_xticks(range(len(ARCHITECTURES)), ARCHITECTURES)
    ax.set_yticks(range(len(categories)),
                  [f"{textwrap.fill(category, 26)} (n={n})" for category, n in zip(categories, counts)])
    for i in range(len(categories)):
        for j in range(len(ARCHITECTURES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > display.norm.vmax * .55 else "#142027")
    ax.set_title(FIGURE_TITLES["figure07_native_reasoning"], weight="bold")
    fig.colorbar(display, ax=ax, label="Mean answer token F1", shrink=.8)
    fig.text(.5, .02, f"Native reasoning labels only; {sum(counts)} of 145 questions labeled",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def build_figures(manifest: dict, data: dict, configuration: list[dict]) -> dict:
    return {
        "figure01_architecture_overview": plot_architecture(manifest, configuration),
        "figure02_recall_at_5": plot_recall(data),
        "figure03_answer_correctness": plot_answers(data),
        "figure04_evidence_support": plot_faithfulness(data),
        "figure05_total_latency": plot_latency(data),
        "figure06_answer_quality_latency": plot_tradeoff(data, "answer_token_f1",
                                                            "figure06_answer_quality_latency", "Mean answer token F1", .23),
        "figure06b_retrieval_latency": plot_tradeoff(data, "recall_at_5",
                                                      "figure06b_retrieval_latency", "Gold-page Recall@5", .52),
        "figure06c_faithfulness_latency": plot_tradeoff(data, "context_faithfulness",
                                                         "figure06c_faithfulness_latency", "Mean context support", 1.0),
        "figure07_native_reasoning": plot_native_reasoning(data),
    }


def output_paths() -> dict[str, Path]:
    paths = {f"{name}.{extension}": ROOT / "results/figures" / f"{name}.{extension}"
             for name in FIGURE_TITLES for extension in ("png", "pdf")}
    paths.update({f"{name}.csv": ROOT / "results/tables" / f"{name}.csv"
                  for name in TABLE_TITLES})
    return paths


def render_to_staging(staging: Path, manifest: dict, data: dict,
                      tables: dict[str, tuple[list[dict], list[str]]]) -> dict[str, Path]:
    staged = {}
    for name, (rows, fields) in tables.items():
        path = staging / f"{name}.csv"
        write_csv(path, rows, fields)
        staged[path.name] = path
    figures = build_figures(manifest, data, tables["table1_architecture_configuration"][0])
    try:
        if set(figures) != set(FIGURE_TITLES):
            raise ValueError("Figure renderer and title list differ")
        for name, fig in figures.items():
            for extension in ("png", "pdf"):
                path = staging / f"{name}.{extension}"
                fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
                staged[path.name] = path
    finally:
        for fig in figures.values():
            plt.close(fig)
    if set(staged) != set(output_paths()) or any(path.stat().st_size == 0 for path in staged.values()):
        raise ValueError("Final output staging is incomplete")
    return staged


def publish(manifest_path: Path, manifest: dict, staged: dict[str, Path]) -> None:
    destinations = output_paths()
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    for name, destination in destinations.items():
        with staged[name].open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    payload = {
        "title": "Final figures and tables",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "run_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                   for name, path in INPUTS.items()},
        "parameters": {
            "figure_formats": ["png", "pdf"], "figure_png_dpi": 220,
            "matplotlib_version": matplotlib.__version__,
            "numpy_version": np.__version__,
            "plot_data": "persisted summaries only; architecture configuration from frozen run manifest",
            "native_reasoning_coverage": "source-labeled questions only, never post hoc inferred",
            "support_policy": "conditional on answers with scored factual units; coverage and abstention shown separately",
            "pareto_policy": "unweighted point-estimate frontiers; no final architecture winner inferred",
            "latency_policy": "recorded per-question total; excludes pipeline/model initialization",
            "judge_caveat": "see results/evaluation/faithfulness_audit_report.md",
        },
        "outputs": {
            name: {"title": (FIGURE_TITLES[name.rsplit(".", 1)[0]]
                             if name.endswith((".png", ".pdf"))
                             else TABLE_TITLES[name.rsplit(".", 1)[0]]),
                   "path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in destinations.items()
        },
    }
    with MANIFEST_OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--check-only", action="store_true",
                        help="validate inputs and show planned outputs without creating files")
    args = parser.parse_args()
    if not args.check_only:
        refuse_existing(*output_paths().values(), MANIFEST_OUTPUT)
    manifest, _ = load_run(args.manifest)
    data = load_inputs(manifest, args.manifest)
    tables = build_tables(manifest, data)
    if args.check_only:
        print(f"Stage 22 inputs valid for {manifest['run_id']}.")
        print(f"Planned output: {len(FIGURE_TITLES)} figures (PNG and PDF), "
              f"{len(TABLE_TITLES)} CSV tables, one provenance manifest.")
        for name, path in output_paths().items():
            print(f"  {name}: {path}")
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "savefig.dpi": 220})
    with tempfile.TemporaryDirectory(prefix="stage22-", dir=ROOT / "results") as temporary:
        staged = render_to_staging(Path(temporary), manifest, data, tables)
        publish(args.manifest, manifest, staged)
    print(f"Saved {len(FIGURE_TITLES)} titled figures as PNG and PDF in {ROOT / 'results/figures'}")
    print(f"Saved {len(TABLE_TITLES)} titled CSV tables in {ROOT / 'results/tables'}")
    print(f"Provenance: {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
