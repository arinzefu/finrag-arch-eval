"""Read-only companion for the FinanceBench P0-P3 notebook.

From a notebook started in the repository root::

    from scripts.notebook_report import load_report
    report = load_report()
    report.show("study")
    report.show("architectures")
    report.show("retrieval")
    report.show("answers")
    report.show("faithfulness")
    report.show("latency")
    report.show("categories")
    report.show("statistics")
    report.show("tradeoffs")
    report.show("audit")
    report.show("interpretation")

Call report.show_all() for the same walkthrough in one cell. Methods that
return DataFrames are available for custom notebook cells. Nothing here runs
retrieval, generation, judging, or evaluation, and nothing writes files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.evaluation_common import ARCHITECTURES, ROOT, sha256

FIGURES = {
    "1": "figure01_architecture_overview",
    "2": "figure02_recall_at_5",
    "3": "figure03_answer_correctness",
    "4": "figure04_evidence_support",
    "5": "figure05_total_latency",
    "6": "figure06_answer_quality_latency",
    "6b": "figure06b_retrieval_latency",
    "6c": "figure06c_faithfulness_latency",
    "7": "figure07_native_reasoning",
}
TABLE_PATHS = {
    "architecture": "results/tables/table1_architecture_configuration.csv",
    "overall": "results/tables/table2_overall_results.csv",
    "category": "results/tables/table3_question_category_results.csv",
    "tests": "results/tables/table4_hypothesis_tests.csv",
    "retrieval": "results/aggregated/retrieval_summary.csv",
    "answers": "results/aggregated/answer_summary_v2.csv",
    "faithfulness": "results/aggregated/faithfulness_summary.csv",
    "latency": "results/aggregated/latency_summary.csv",
    "tradeoffs": "results/aggregated/tradeoff_summary.csv",
    "frontiers": "results/aggregated/pareto_frontiers.csv",
    "paired_tradeoffs": "results/aggregated/paired_tradeoffs.csv",
}
QUESTION_METRICS = {
    "retrieval": "results/evaluation/retrieval_metrics.csv",
    "answers": "results/evaluation/answer_metrics_v2.csv",
    "faithfulness": "results/evaluation/faithfulness_metrics.csv",
    "latency": "results/evaluation/latency_metrics.csv",
}
SECTION_ORDER = ("study", "architectures", "metrics", "overall", "retrieval", "answers",
                 "faithfulness", "latency", "categories", "statistics", "tradeoffs",
                 "audit", "interpretation")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing report columns: {missing}")
    return frame.loc[:, columns].copy()


class FinanceBenchReport:
    """Frozen, notebook-friendly view of the controlled P0-P3 experiment."""

    def __init__(self, root: Path = ROOT, *, verify_hashes: bool = True) -> None:
        self.root = Path(root).resolve()
        self.run_manifest = _read_json(self.root / "results/raw_runs/financebench-heldout-v1_manifest.json")
        self.final_manifest = _read_json(self.root / "results/final_outputs_manifest.json")
        self.corpus_freeze = _read_json(self.root / "configs/corpus_freeze.json")
        self.faithfulness_protocol = _read_json(
            self.root / "results/evaluation/faithfulness_evaluation_manifest.json")
        self.statistics_protocol = _read_json(
            self.root / "results/statistics/hypothesis_tests_manifest.json")
        self.category_protocol = _read_json(
            self.root / "results/aggregated/question_type_summary_manifest.json")
        self.audit_selection = _read_json(
            self.root / "results/evaluation/faithfulness_audit_selection.json")
        self.tables = {name: pd.read_csv(self.root / path)
                       for name, path in TABLE_PATHS.items()}
        self._question_tables: dict[str, pd.DataFrame] = {}
        self._raw_runs: dict[str, dict[str, dict]] | None = None
        self._validate(verify_hashes)

    def _validate(self, verify_hashes: bool) -> None:
        run_id = self.run_manifest["run_id"]
        if self.final_manifest["run_id"] != run_id:
            raise ValueError("Final outputs belong to a different frozen run")
        if len(self.run_manifest["question_ids"]) != self.run_manifest["question_count"]:
            raise ValueError("Frozen question count does not match question IDs")
        for name, frame in self.tables.items():
            if frame.empty or not frame["run_id"].eq(run_id).all():
                raise ValueError(f"{name} is empty or belongs to another run")
        for name in ("architecture", "overall", "answers", "faithfulness", "tradeoffs"):
            frame = self.tables[name]
            if len(frame) != 4 or set(frame["architecture"]) != set(ARCHITECTURES):
                raise ValueError(f"{name} does not cover P0-P3 exactly once")
        if not pd.isna(self.tables["overall"].set_index("architecture").loc["P0", "context_faithfulness"]):
            raise ValueError("P0 context faithfulness must be NA, not zero")
        if not pd.isna(self.tables["overall"].set_index("architecture").loc["P0", "hit_at_5"]):
            raise ValueError("P0 retrieval must be NA, not zero")
        category = self.tables["category"]
        expected_coverage = self.category_protocol["parameters"]["label_coverage"]
        for grouping, coverage in expected_coverage.items():
            subset = category.loc[(category["grouping"] == grouping)
                                  & (category["architecture"] == "P0")]
            if int(subset["question_count"].sum()) != coverage:
                raise ValueError(f"Native {grouping} coverage differs from provenance")
        expected_tests = len(self.statistics_protocol["parameters"]["comparisons"])
        if len(self.tables["tests"]) != expected_tests:
            raise ValueError("Paired test count differs from Stage 20 provenance")
        if verify_hashes:
            source = self.root / "results/raw_runs/financebench-heldout-v1_manifest.json"
            if sha256(source) != self.final_manifest["run_manifest"]["sha256"]:
                raise ValueError("Frozen run manifest hash has changed")
            for name in ("retrieval", "answers", "faithfulness", "latency", "categories",
                         "statistics", "tradeoffs", "frontiers"):
                record = self.final_manifest["inputs"][name]
                path = Path(record["path"])
                if not path.exists() or sha256(path) != record["sha256"]:
                    raise ValueError(f"Published input missing or modified: {name}")
            for name, record in self.final_manifest["outputs"].items():
                folder = "figures" if name.endswith((".png", ".pdf")) else "tables"
                path = self.root / "results" / folder / name
                if not path.exists() or sha256(path) != record["sha256"]:
                    raise ValueError(f"Final output missing or modified: {path}")

    def dataframe(self, name: str) -> pd.DataFrame:
        """Return an independent copy of a saved result table by TABLE_PATHS key."""
        if name not in self.tables:
            raise KeyError(f"Unknown table {name!r}; available: {', '.join(TABLE_PATHS)}")
        return self.tables[name].copy()

    def study_design(self) -> str:
        """Plain-language description of the comparison and frozen corpus."""
        run = self.run_manifest
        corpus = self.corpus_freeze
        table = self.tables["architecture"]
        llm = table["llm_model"].unique().tolist()
        tokens = table["generation_max_output_tokens"].unique().tolist()
        temperature = table["temperature_policy"].unique().tolist()
        if not (len(llm) == len(tokens) == len(temperature) == 1):
            raise ValueError("Shared generation parameters are inconsistent")
        return (
            "# Controlled financial RAG architecture comparison\n\n"
            "**Research question:** Which retrieval architecture best balances retrieval quality, "
            "faithfulness, answer correctness, and latency for financial QA? This is a design-selection "
            "study, not a new RAG method.\n\n"
            f"- Frozen run: `{run['run_id']}`; {run['question_count']} held-out FinanceBench questions "
            f"answered by each of P0-P3 ({run['question_count'] * len(ARCHITECTURES)} answers). "
            "Five development questions were not in this run.\n"
            f"- Frozen corpus: {corpus['corpus']['chunk_count']:,} chunks; SHA256 "
            f"`{corpus['corpus']['file_sha256']}`. Dense model: "
            f"`{corpus['dense_index']['embedding_model']}` "
            f"({corpus['dense_index']['embedding_dimension']} dimensions); "
            f"BM25 tokenizer: `{corpus['bm25_index']['tokenizer_id']}`.\n"
            f"- Shared answer model: `{llm[0]}`; output limit: {tokens[0]} tokens; "
            f"temperature: {temperature[0]}. P1-P3 share one RAG prompt and final context size of 5.\n"
            "- The raw run, evaluation tables, figures, and tables are persisted. "
            "This notebook only reads them."
        )

    def architectures(self) -> pd.DataFrame:
        """Table 1: controlled/common settings and the retrieval differences."""
        return self.dataframe("architecture").drop(columns=["table_title", "run_id"])

    def overall_results(self) -> pd.DataFrame:
        """Table 2: every architecture's main metrics, denominators, and latency."""
        return self.dataframe("overall").drop(columns=["table_title", "run_id"])

    def comparison_matrix(self) -> pd.DataFrame:
        """One row per architecture, combining frozen controls and headline outcomes."""
        config = self.tables["architecture"].set_index("architecture")
        overall = self.tables["overall"].set_index("architecture")
        retrieval = self.tables["retrieval"].set_index("architecture")
        answers = self.tables["answers"].set_index("architecture")
        latency = self.tables["latency"].set_index(["architecture", "component"])
        run = self.run_manifest
        corpus = self.corpus_freeze["corpus"]

        def applicable_int(value: Any) -> Any:
            return int(value) if pd.notna(value) else pd.NA

        def proportion(value: Any) -> Any:
            return round(float(value), 3) if pd.notna(value) else pd.NA

        rows = []
        for arch in ARCHITECTURES:
            settings = config.loc[arch]
            result = overall.loc[arch]
            answer = answers.loc[arch]
            evidence = retrieval.loc[arch] if arch in retrieval.index else None
            total = latency.loc[(arch, "total"), "mean_ms"]
            rerank = (latency.loc[(arch, "reranking"), "mean_ms"]
                      if (arch, "reranking") in latency.index else pd.NA)
            rows.append({
                ("Shared controls", "Held-out n"): int(run["question_count"]),
                ("Shared controls", "Corpus chunks"): int(corpus["chunk_count"]),
                ("Shared controls", "Corpus SHA-256 (12)"): corpus["file_sha256"][:12],
                ("Shared controls", "Answer LLM"): settings["llm_model"],
                ("Shared controls", "Max output tokens"): int(settings["generation_max_output_tokens"]),
                ("Shared controls", "Temperature"): settings["temperature_policy"],
                ("Design", "Retrieval"): settings["retrieval_design"],
                ("Design", "Dense k"): applicable_int(settings["dense_candidate_k"]),
                ("Design", "BM25 k"): applicable_int(settings["bm25_candidate_k"]),
                ("Design", "RRF k"): applicable_int(settings["rrf_k"]),
                ("Design", "Fusion pool"): applicable_int(settings["fusion_candidate_k"]),
                ("Design", "Reranker"): settings["reranker_model"],
                ("Design", "Final context k"): int(settings["final_context_k"]),
                ("Design", "Prompt policy"): settings["prompt_policy"],
                ("Retrieval", "Hit@1"): proportion(evidence["hit_at_1"]) if evidence is not None else pd.NA,
                ("Retrieval", "Hit@3"): proportion(evidence["hit_at_3"]) if evidence is not None else pd.NA,
                ("Retrieval", "Hit@5"): proportion(result["hit_at_5"]),
                ("Retrieval", "Recall@5"): proportion(result["recall_at_5"]),
                ("Retrieval", "MRR@5"): proportion(result["mrr_at_5"]),
                ("Answer", "Exact match"): proportion(result["exact_match"]),
                ("Answer", "Token F1"): proportion(result["token_f1"]),
                ("Answer", "Numeric accuracy"): proportion(result["numeric_accuracy"]),
                ("Answer", "Numeric precision"): proportion(answer["numeric_precision"]),
                ("Answer", "Numeric F1"): proportion(answer["numeric_f1"]),
                ("Answer", "Numeric n"): int(result["numeric_applicable_n"]),
                ("Evidence", "Context support"): proportion(result["context_faithfulness"]),
                ("Evidence", "Context hallucination"): proportion(result["context_hallucination"]),
                ("Evidence", "Context n"): int(result["context_scored_n"]),
                ("Evidence", "Gold support"): proportion(result["gold_evidence_support"]),
                ("Evidence", "Gold hallucination"): proportion(result["gold_evidence_hallucination"]),
                ("Evidence", "Gold n"): int(result["gold_scored_n"]),
                ("Evidence", "Exact abstentions"): int(result["exact_abstention_count"]),
                ("Latency", "Mean total (s)"): round(float(total) / 1000, 2),
                ("Latency", "Mean reranking (s)"):
                    round(float(rerank) / 1000, 2) if pd.notna(rerank) else pd.NA,
            })
        matrix = pd.DataFrame(rows, index=list(ARCHITECTURES))
        matrix.columns = pd.MultiIndex.from_tuples(matrix.columns, names=["Group", "Measure"])
        matrix.index.name = "Architecture"
        return matrix

    def metric_glossary(self) -> pd.DataFrame:
        """Definitions and applicability for each main evaluation family."""
        return pd.DataFrame([
            ("Hit@1/3/5", "P1-P3", "Binary: any annotated gold evidence page in final top-k chunks."),
            ("Recall@5", "P1-P3", "Unique annotated gold pages in final top 5 / total annotated gold pages."),
            ("MRR@5", "P1-P3", "Reciprocal rank of the first annotated gold page in final top 5; otherwise 0."),
            ("Exact match", "P0-P3", "Normalized answer-string equality; strict surface metric."),
            ("Token F1", "P0-P3", "Bag-of-token overlap with the gold answer after fixed normalization."),
            ("Numeric accuracy", "P0-P3; gold-numeric answers", "Matched gold numeric entities / gold numeric entities; corrected v2 extractor."),
            ("Numeric precision / F1", "P0-P3; applicable answers", "Precision penalizes extra generated numeric entities; F1 combines precision and recall."),
            ("Context faithfulness", "P1-P3; scored answers", "Supported factual answer units / all factual units, against actual final retrieved chunks."),
            ("Gold-evidence support", "P0-P3; scored answers", "Same unit measure against FinanceBench annotated evidence."),
            ("Hallucination", "P0-P3 where support applies", "One minus the corresponding support score; not defined for an answer without factual units."),
            ("Exact abstention", "P0-P3", "Exact refusal phrase count; shown separately, not scored as zero support."),
            ("Latency", "P0-P3", "Recorded per-question component/total time; excludes model and pipeline initialization."),
        ], columns=["Metric", "Applies to", "Definition"])

    def retrieval(self) -> pd.DataFrame:
        """Stage 15 page-level Hit@1/3/5, Recall@5, and MRR@5."""
        return _compact(self.tables["retrieval"], ["architecture", "question_count",
            "no_result_count", "hit_at_1", "hit_at_3", "hit_at_5", "recall_at_5", "mrr_at_5"])

    def answers(self) -> pd.DataFrame:
        """Stage 16 v2 exact, token, and full numeric metric family."""
        return _compact(self.tables["answers"], ["architecture", "question_count",
            "numeric_applicable_count", "exact_match", "token_f1", "numeric_accuracy",
            "numeric_precision", "numeric_recall", "numeric_f1"])

    def faithfulness(self) -> pd.DataFrame:
        """Stage 17 support/hallucination with exact-abstention and coverage counts."""
        return _compact(self.tables["faithfulness"], ["architecture", "question_count",
            "abstention_count", "context_applicable_count", "context_faithfulness",
            "context_hallucination", "gold_applicable_count", "gold_evidence_support",
            "gold_evidence_hallucination", "judge_model", "judge_protocol",
            "context_guard_override_count", "gold_guard_override_count"])

    def latency(self) -> pd.DataFrame:
        """Stage 18 component and total timing; inapplicable components remain NA."""
        return _compact(self.tables["latency"], ["architecture", "component",
            "question_count", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms"])

    def categories(self, grouping: str = "question_reasoning") -> pd.DataFrame:
        """Table 3 filtered to one native FinanceBench grouping."""
        if grouping not in ("question_type", "question_reasoning"):
            raise ValueError("grouping must be 'question_type' or 'question_reasoning'")
        frame = self.tables["category"]
        columns = ["grouping", "category", "architecture", "question_count",
                   "token_f1_mean", "numeric_accuracy_mean", "numeric_accuracy_n",
                   "recall_at_5_mean", "context_faithfulness_mean",
                   "context_faithfulness_n", "gold_evidence_support_mean",
                   "gold_evidence_support_n", "total_latency_ms_mean"]
        return _compact(frame.loc[frame["grouping"] == grouping], columns).reset_index(drop=True)

    def hypotheses(self) -> pd.DataFrame:
        """The dissertation's substantive H0-H3 predictions, not statistical nulls."""
        return pd.DataFrame([
            ("H0", "Closed-book P0 has most hallucination, least accuracy, and lowest latency.", "P0 versus P1-P3"),
            ("H1", "Dense P1 improves faithfulness over P0 but may miss exact terms/numbers.", "P0 versus P1"),
            ("H2", "Hybrid P2 improves retrieval on numeric/terminology-heavy questions.", "P1 versus P2"),
            ("H3", "Reranked hybrid P3 improves text-based faithfulness at a latency cost.", "P2 versus P3"),
        ], columns=["Hypothesis", "Substantive prediction", "Primary contrast"])

    def hypothesis_tests(self, hypothesis: str | None = None) -> pd.DataFrame:
        """All 21 fixed paired contrasts or those tagged with H0, H1, H2, H3."""
        frame = self.tables["tests"]
        if hypothesis is not None:
            if hypothesis not in ("H0", "H1", "H2", "H3"):
                raise ValueError("hypothesis must be H0, H1, H2, or H3")
            frame = frame.loc[frame["hypothesis"].str.split(",").map(lambda tags: hypothesis in tags)]
        return _compact(frame, ["hypothesis", "architecture_a", "architecture_b", "metric",
            "population", "test", "pair_count", "excluded_pair_count",
            "mean_difference_b_minus_a", "difference_ci95_low", "difference_ci95_high",
            "p_value_two_sided", "holm_p_value", "significant_holm_0_05",
            "rank_biserial_b_minus_a", "a_only_success", "b_only_success"]).reset_index(drop=True)

    def tradeoffs(self) -> pd.DataFrame:
        """Stage 21 quality, coverage, latency, and unweighted frontier flags."""
        return self.dataframe("tradeoffs").drop(columns=["table_title", "run_id"])

    def frontiers(self, objective: str | None = None) -> pd.DataFrame:
        """Pareto membership and explicit dominators for four quality axes."""
        frame = self.tables["frontiers"]
        if objective is not None:
            choices = set(frame["objective"])
            if objective not in choices:
                raise ValueError(f"objective must be one of {sorted(choices)}")
            frame = frame.loc[frame["objective"] == objective]
        return _compact(frame, ["objective", "architecture", "quality_mean",
            "quality_scored_n", "quality_coverage_rate", "exact_abstention_rate",
            "mean_total_latency_ms", "pareto_nondominated", "dominated_by"]).reset_index(drop=True)

    def paired_tradeoffs(self) -> pd.DataFrame:
        """Descriptive adjacent-architecture differences on shared applicable IDs."""
        return self.dataframe("paired_tradeoffs").drop(columns=["table_title", "run_id"])

    def figure_path(self, number: str | int) -> Path:
        """Return a saved PNG by figure number (1, ..., 6b, 6c, 7)."""
        key = str(number).lower().strip()
        if key not in FIGURES:
            raise KeyError(f"Unknown figure {number!r}; available: {', '.join(FIGURES)}")
        path = self.root / "results/figures" / f"{FIGURES[key]}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _load_question_data(self) -> None:
        if self._raw_runs is not None:
            return
        expected = set(self.run_manifest["question_ids"])
        raw_runs: dict[str, dict[str, dict]] = {}
        for arch in ARCHITECTURES:
            path = self.root / "results/raw_runs" / f"{self.run_manifest['run_id']}_{arch}.jsonl"
            with path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            keyed = {record["question_id"]: record for record in records}
            if len(records) != len(expected) or set(keyed) != expected:
                raise ValueError(f"Frozen raw observations incomplete for {arch}")
            raw_runs[arch] = keyed
        for name, relative in QUESTION_METRICS.items():
            frame = pd.read_csv(self.root / relative)
            expected_arch = set(ARCHITECTURES[1:] if name == "retrieval" else ARCHITECTURES)
            keys = set(zip(frame["architecture"], frame["question_id"]))
            expected_keys = {(arch, qid) for arch in expected_arch for qid in expected}
            if (len(keys) != len(frame) or keys != expected_keys
                    or not frame["run_id"].eq(self.run_manifest["run_id"]).all()):
                raise ValueError(f"Per-question {name} metrics are incomplete or from another run")
            self._question_tables[name] = frame.set_index(["architecture", "question_id"])
        self._raw_runs = raw_runs

    def audit_sample(self) -> pd.DataFrame:
        """Eight IDs selected by hash before their individual v4 judgments were read."""
        self._load_question_data()
        assert self._raw_runs is not None
        ids = self.audit_selection["question_ids"]
        if set(ids) - set(self.run_manifest["question_ids"]):
            raise ValueError("Audit selection contains IDs outside the frozen run")
        return pd.DataFrame([
            {"question_id": qid, "question": self._raw_runs["P0"][qid]["question"],
             "gold_answer": self._raw_runs["P0"][qid]["gold_answer"]}
            for qid in ids
        ])

    def case_study(self, question_id: str) -> dict[str, Any]:
        """One frozen question with gold answer, all P0-P3 answers, scores, and final chunks."""
        self._load_question_data()
        assert self._raw_runs is not None
        if question_id not in self._raw_runs["P0"]:
            raise KeyError(f"Question {question_id!r} is not in the frozen held-out run")
        source = self._raw_runs["P0"][question_id]
        rows = []
        for arch in ARCHITECTURES:
            raw = self._raw_runs[arch][question_id]
            answers = self._question_tables["answers"].loc[(arch, question_id)]
            faith = self._question_tables["faithfulness"].loc[(arch, question_id)]
            latency = self._question_tables["latency"].loc[(arch, question_id)]
            retrieval = (self._question_tables["retrieval"].loc[(arch, question_id)]
                         if arch != "P0" else None)
            chunks = raw["retrieved_chunks"]
            rows.append({
                "architecture": arch,
                "generated_answer": raw["generated_answer"],
                "final_chunk_ids": "; ".join(chunk["chunk_id"] for chunk in chunks),
                "final_pages": "; ".join(f"{chunk['doc_name']} p{chunk['page']}" for chunk in chunks),
                "hit_at_5": retrieval["hit_at_5"] if retrieval is not None else pd.NA,
                "token_f1": answers["token_f1"],
                "numeric_accuracy": answers["numeric_accuracy"],
                "context_faithfulness": faith["context_faithfulness"],
                "gold_evidence_support": faith["gold_evidence_support"],
                "abstained": faith["abstained"],
                "total_latency_ms": latency["total_latency_ms"],
            })
        return {"question_id": question_id, "question": source["question"],
                "gold_answer": source["gold_answer"], "answers": pd.DataFrame(rows)}

    def limitations(self) -> str:
        """Study-specific cautions to accompany every interpretation."""
        reasoning_n = self.category_protocol["parameters"]["label_coverage"]["question_reasoning"]
        question_n = self.run_manifest["question_count"]
        return (
            "## Interpretation safeguards\n\n"
            "- **Support is model-assisted, not ground truth.** The fixed `gpt-4o-mini` v4 judge "
            "scored factual answer units. The preselected eight-question audit found a clear wrong-year "
            "Boeing table value marked supported and an invented P0 answer marked nonfactual. See "
            "`results/evaluation/faithfulness_audit_report.md`.\n"
            "- **Abstentions are not zeros.** Context/gold support means include only answers with "
            "scored factual units. Show their `n` and the exact-abstention count alongside the means. "
            "P0 has no context score.\n"
            "- **Surface metrics are strict.** Exact match and token F1 can penalize correct paraphrases. "
            "Numeric accuracy uses the corrected v2 extractor; abbreviated dates remain a known limitation.\n"
            "- **Latency is observational.** Recorded per-question time excludes setup but includes "
            "API/network variation. Sequential architecture runs do not isolate a pure causal hardware cost.\n"
            f"- **Subgroup coverage is incomplete.** Native `question_reasoning` exists for "
            f"{reasoning_n}/{question_n} questions. No numeric/terminology flags were assigned "
            "after viewing outcomes.\n"
            "- **Stage 20 was not blinded preregistration.** The comparison list was fixed before "
            "its tests ran, but earlier held-out aggregate results were visible. The 21 p-values use "
            "a global Holm adjustment. Nonsignificance does not prove equivalence.\n"
            "- **Pareto membership is descriptive.** It uses point-estimate mean quality and latency "
            "without subjective weights; differing scored-answer coverage matters."
        )

    def interpretation(self) -> str:
        """Numerical reading of H0-H3 and the trade-off, without declaring a winner."""
        overall = self.tables["overall"].set_index("architecture")
        tests = self.tables["tests"]
        p2_p3 = tests.loc[(tests["architecture_a"] == "P2")
                          & (tests["architecture_b"] == "P3")]
        faith = p2_p3.loc[p2_p3["metric"] == "context_faithfulness"].iloc[0]
        latency = p2_p3.loc[p2_p3["metric"] == "total_latency_ms"].iloc[0]
        p1_p2_recall = tests.loc[(tests["architecture_a"] == "P1")
                                 & (tests["architecture_b"] == "P2")
                                 & (tests["metric"] == "recall_at_5")].iloc[0]
        frontiers = self.frontiers()
        context_frontier = ", ".join(frontiers.loc[
            (frontiers["objective"] == "context_faithfulness")
            & (frontiers["pareto_nondominated"] == 1), "architecture"])
        recall_frontier = ", ".join(frontiers.loc[
            (frontiers["objective"] == "recall_at_5")
            & (frontiers["pareto_nondominated"] == 1), "architecture"])
        return (
            "## Reading the results\n\n"
            f"- **H0 (closed book):** P0's mean total latency was "
            f"{overall.loc['P0', 'total_latency_mean_ms']/1000:.2f} s, versus "
            f"{overall.loc['P1', 'total_latency_mean_ms']/1000:.2f} s for P1. "
            "Thus P0 was not the fastest observed architecture. Its gold-evidence support "
            f"mean was {overall.loc['P0', 'gold_evidence_support']:.3f} "
            f"(n={int(overall.loc['P0', 'gold_scored_n'])}), with the judge caveats above.\n"
            "- **H1 (dense RAG):** P1 improved strict numeric accuracy over P0, but the "
            "paired gold-support difference was not significant after Holm adjustment. "
            "Its mean answer token F1 was not higher than P0's.\n"
            f"- **H2 (hybrid):** P2 Recall@5 was {overall.loc['P2', 'recall_at_5']:.3f}, "
            f"below P1's {overall.loc['P1', 'recall_at_5']:.3f}; the paired "
            f"difference was {p1_p2_recall['mean_difference_b_minus_a']:+.3f} "
            f"(Holm p={p1_p2_recall['holm_p_value']:.3f}). This run does not establish "
            "the proposed hybrid retrieval advantage.\n"
            f"- **H3 (reranking):** P3 added {latency['mean_difference_b_minus_a']/1000:.2f} s "
            "per question versus P2 (Holm-adjusted p<0.001), while the paired "
            f"context-support difference was {faith['mean_difference_b_minus_a']:+.3f} "
            f"over {int(faith['pair_count'])} scored pairs "
            f"(Holm p={faith['holm_p_value']:.3f}). The evidence does not establish "
            "a faithfulness gain commensurate with that cost.\n"
            f"- **Design choice:** Point-estimate context-support/latency frontier: "
            f"{context_frontier}; Recall@5/latency frontier: {recall_frontier}. "
            "These are different objectives, and no single scalar score or definitive winner "
            "is imposed. Consider the audit, abstentions, and practical latency limit in the discussion."
        )

    def show_case(self, question_id: str) -> None:
        """Display an inspectable, side-by-side frozen question in IPython."""
        from IPython.display import Markdown, display

        case = self.case_study(question_id)
        display(Markdown(f"### {case['question_id']}\n\n**Question:** {case['question']}\n\n"
                         f"**Gold:** {case['gold_answer']}"))
        with pd.option_context("display.max_colwidth", 180, "display.max_columns", 30):
            display(case["answers"].round(3))

    def show_figure(self, number: str | int) -> None:
        """Display a persisted, hash-checked Stage 22 figure in IPython."""
        from IPython.display import Image, display

        display(Image(filename=str(self.figure_path(number)), width=950))

    def show(self, section: str) -> None:
        """Display one narrative section, tables, and its saved scientific figures."""
        from IPython.display import Markdown, display

        section = section.lower().strip()
        if section not in SECTION_ORDER:
            raise KeyError(f"Unknown section {section!r}; use one of {', '.join(SECTION_ORDER)}")

        def table(frame: pd.DataFrame) -> None:
            with pd.option_context("display.max_rows", 100, "display.max_columns", 30,
                                   "display.max_colwidth", 100, "display.width", 180):
                display(frame.round(3))

        if section == "study":
            display(Markdown(self.study_design()))
            return
        if section == "architectures":
            display(Markdown("## P0-P3 architecture and controlled parameters\n\n"
                "P0 is closed-book. P1 adds dense retrieval. P2 adds BM25 with RRF. "
                "P3 reranks hybrid candidates with a cross-encoder. All four use the same "
                "answer model and output limit; P1-P3 share the RAG prompt and final context size."))
            table(self.architectures())
            self.show_figure("1")
            return
        if section == "metrics":
            display(Markdown("## Evaluation definitions\n\nRetrieval is evaluated at the "
                "annotated `(document, page)` level for P1-P3 only. Strict answer-overlap "
                "scores, model-assisted evidence support, and recorded latency answer "
                "different questions. Blank/NA is inapplicable or unscored, never zero."))
            table(self.metric_glossary())
            return
        if section == "overall":
            display(Markdown("## Overall held-out results\n\nThis is the central comparison "
                "table. Support scores are conditional on scored factual units; the relevant "
                "denominator and exact abstentions are shown beside each rate."))
            table(self.overall_results())
            return
        if section == "retrieval":
            display(Markdown("## Retrieval quality (Stage 15)\n\nP0 has no retrieval "
                "metric. Hit@k, Recall@5, and MRR@5 use annotated gold pages, not document "
                "names alone. Each architecture contributes 145 questions."))
            table(self.retrieval())
            self.show_figure("2")
            return
        if section == "answers":
            display(Markdown("## Answer correctness (Stage 16 v2)\n\nExact match and "
                "token F1 are strict surface comparisons. Numeric values are normalized, "
                "but the v2 correction prevents the company name `3M` from being read "
                "as 3 million. Numeric scores have their own applicable denominator."))
            table(self.answers())
            self.show_figure("3")
            return
        if section == "faithfulness":
            protocol = self.faithfulness_protocol["parameters"]
            display(Markdown("## Evidence support and hallucination (Stage 17)\n\n"
                f"One fixed `{protocol['judge_model']}` judge and `{protocol['judge_protocol']}` "
                "rubric evaluated P0-P3. Context support uses the final retrieved chunks "
                "(P1-P3); gold support uses FinanceBench annotated evidence (P0-P3). "
                "Hallucination is one minus support for scored answers. Exact abstentions "
                "and answers with no scored factual units are not assigned zero support."))
            table(self.faithfulness())
            self.show_figure("4")
            return
        if section == "latency":
            display(Markdown("## Recorded latency (Stage 18)\n\nComponent, mean, "
                "median, standard deviation, minimum, and maximum are in milliseconds. "
                "Inapplicable components are NA; dense/BM25/fusion sub-times were not "
                "recorded separately. Setup is excluded, but API/network effects remain."))
            table(self.latency())
            self.show_figure("5")
            return
        if section == "categories":
            coverage = self.category_protocol["parameters"]["label_coverage"]
            display(Markdown("## Native FinanceBench categories (Stage 19)\n\n"
                f"`question_type` labels cover {coverage['question_type']} questions; "
                f"`question_reasoning` labels cover {coverage['question_reasoning']}. "
                "The remaining reasoning labels are source-null and excluded only from "
                "reasoning groups. No outcome-informed derived labels were created."))
            display(Markdown("### Question type"))
            table(self.categories("question_type"))
            display(Markdown("### Question reasoning"))
            table(self.categories("question_reasoning"))
            self.show_figure("7")
            return
        if section == "statistics":
            display(Markdown("## Paired hypothesis analysis (Stage 20)\n\nThese H0-H3 "
                "labels name the dissertation's substantive predictions, not statistical "
                "nulls. Every contrast pairs the same question IDs. McNemar exact handles "
                "binary outcomes; Wilcoxon signed-rank handles scores; mean differences "
                "have seeded paired-bootstrap 95% intervals. Holm adjustment spans all "
                "21 comparisons. Scored-answer tests use complete pairs only."))
            table(self.hypotheses())
            table(self.hypothesis_tests())
            return
        if section == "tradeoffs":
            display(Markdown("## Unweighted quality-latency trade-offs (Stage 21)\n\n"
                "The Pareto views maximize one quality score and minimize mean total "
                "latency; no arbitrary weighted score is used. Filled points are "
                "nondominated point estimates. Conditional support coverage and "
                "abstentions must be considered with those flags."))
            table(self.tradeoffs())
            table(self.frontiers())
            display(Markdown("### Adjacent-architecture differences on shared applicable questions"))
            table(self.paired_tradeoffs())
            for number in ("6", "6b", "6c"):
                self.show_figure(number)
            return
        if section == "audit":
            display(Markdown("## Preselected evidence-judge audit\n\nEight non-pilot "
                "questions were selected by a hash of IDs before per-question v4 "
                "judgments were read. These cases are illustrative and not an estimated "
                "judge-error rate."))
            table(self.audit_sample())
            audit_path = self.root / "results/evaluation/faithfulness_audit_report.md"
            display(Markdown(audit_path.read_text(encoding="utf-8")))
            self.show_case("financebench_id_10285")
            return
        display(Markdown(self.interpretation()))
        display(Markdown(self.limitations()))

    def show_all(self) -> None:
        """Display the complete, ordered dissertation results walkthrough."""
        for section in SECTION_ORDER:
            self.show(section)


def load_report(root: Path = ROOT, *, verify_hashes: bool = True) -> FinanceBenchReport:
    """Load and validate the persisted experiment without recomputing it."""
    return FinanceBenchReport(root, verify_hashes=verify_hashes)
