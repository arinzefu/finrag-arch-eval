"""Read-only companion for the FinanceBench P0-P3 notebook.

From a notebook started in the repository root::

    from scripts.notebook_report import load_report
    report = load_report()
    report.show("study")
    report.show("architectures")
    report.show("retrieval")
    report.show("answers")
    report.show("faithfulness")
    report.show("abstentions")
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
ABSTENTION_TABLE_PATHS = {
    "overlap": "results/aggregated/abstention_overlap.csv",
    "pairwise": "results/aggregated/abstention_pairwise.csv",
    "evidence_coverage": "results/aggregated/abstention_evidence_coverage.csv",
    "question_details": "results/aggregated/abstention_question_details.csv",
    "question_patterns": "results/aggregated/abstention_question_patterns.csv",
    "support_summary": "results/aggregated/abstention_support_summary.csv",
    "categories": "results/aggregated/abstention_category_summary.csv",
}
QUESTION_METRICS = {
    "retrieval": "results/evaluation/retrieval_metrics.csv",
    "answers": "results/evaluation/answer_metrics_v2.csv",
    "faithfulness": "results/evaluation/faithfulness_metrics.csv",
    "latency": "results/evaluation/latency_metrics.csv",
}
SECTION_ORDER = ("study", "architectures", "metrics", "overall", "retrieval", "answers",
                 "faithfulness", "abstentions", "latency", "categories", "statistics", "tradeoffs",
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
        quality_path = self.root / "results/evaluation/answer_quality_evaluation_manifest.json"
        self.answer_quality_protocol = _read_json(quality_path) if quality_path.exists() else None
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
        self._abstention_tables: dict[str, pd.DataFrame] | None = None
        self._verify_hashes = verify_hashes
        self._validate(verify_hashes)

    @property
    def has_answer_quality(self) -> bool:
        """Whether post-hoc semantic and BERTScore results have been published."""
        return {"answer_correctness", "answer_completeness", "bertscore_f1",
                "semantic_scored_count", "bertscore_scored_count"}.issubset(self.tables["answers"])

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
        self._validate_answer_quality(verify_hashes)
        if verify_hashes:
            source = self.root / "results/raw_runs/financebench-heldout-v1_manifest.json"
            if sha256(source) != self.final_manifest["run_manifest"]["sha256"]:
                raise ValueError("Frozen run manifest hash has changed")
            for name in ("retrieval", "answers", "faithfulness", "latency", "categories",
                         "statistics", "tradeoffs", "frontiers",
                         *(name for name in ("answer_quality_manifest",)
                           if name in self.final_manifest["inputs"])):
                record = self.final_manifest["inputs"][name]
                path = Path(record["path"])
                if not path.exists() or sha256(path) != record["sha256"]:
                    raise ValueError(f"Published input missing or modified: {name}")
            for name, record in self.final_manifest["outputs"].items():
                folder = "figures" if name.endswith((".png", ".pdf")) else "tables"
                path = self.root / "results" / folder / name
                if not path.exists() or sha256(path) != record["sha256"]:
                    raise ValueError(f"Final output missing or modified: {path}")

    def _validate_answer_quality(self, verify_hashes: bool) -> None:
        """Follow the quality manifest through its frozen run, journals and outputs."""
        protocol = self.answer_quality_protocol
        if protocol is None:
            if self.has_answer_quality:
                raise ValueError("Semantic answer metrics are missing their quality manifest")
            return
        if protocol.get("run_id") != self.run_manifest["run_id"]:
            raise ValueError("Answer-quality evaluation belongs to a different frozen run")
        run_record = protocol["run_manifest"]
        run_path = self.root / "results/raw_runs/financebench-heldout-v1_manifest.json"
        if Path(run_record["path"]).resolve() != run_path.resolve():
            raise ValueError("Answer-quality evaluation references a different run manifest")
        required_inputs = {"evidence", "reference_rubrics", "judgments", "bertscore",
                           "semantic_source", "bertscore_source", "evaluator_source", *ARCHITECTURES}
        if not required_inputs.issubset(protocol["inputs"]):
            raise ValueError("Answer-quality provenance omits required input journals or sources")
        outputs = protocol["outputs"]
        for name, relative in (("per_question", QUESTION_METRICS["answers"]),
                               ("summary", TABLE_PATHS["answers"])):
            if name not in outputs or Path(outputs[name]["path"]).resolve() != (self.root / relative).resolve():
                raise ValueError(f"Answer-quality provenance references a different {name} output")
        if self.has_answer_quality and "answer_quality_manifest" not in self.final_manifest["inputs"]:
            raise ValueError("Published semantic answer metrics lack quality-manifest provenance")
        if verify_hashes:
            if not run_path.exists() or sha256(run_path) != run_record["sha256"]:
                raise ValueError("Answer-quality frozen-run manifest hash has changed")
            for kind in ("inputs", "outputs"):
                for name, record in protocol[kind].items():
                    path = Path(record["path"])
                    if not path.exists() or sha256(path) != record["sha256"]:
                        raise ValueError(f"Answer-quality {kind} missing or modified: {name}")

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
            "faithfulness, answer correctness and latency for financial QA? This is a design-selection "
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
                ("Answer", "Model-judged correctness"): proportion(result.get("answer_correctness")),
                ("Answer", "Reference-unit coverage proxy"): proportion(result.get("answer_completeness")),
                ("Answer", "Semantic scored n"): applicable_int(result.get("semantic_scored_count")),
                ("Answer", "Judge errors"): applicable_int(answer.get("judge_error_count")),
                ("Answer", "Correctness lower bound"):
                    proportion(answer.get("answer_correctness_lower_bound")),
                ("Answer", "Correctness upper bound"):
                    proportion(answer.get("answer_correctness_upper_bound")),
                ("Answer", "BERTScore F1 similarity"): proportion(result.get("bertscore_f1")),
                ("Answer", "BERTScore n"): applicable_int(result.get("bertscore_scored_count")),
                ("Answer", "Correctness, non-abstained"):
                    proportion(result.get("answer_correctness_non_abstained")),
                ("Answer", "Non-abstained n"): applicable_int(result.get("non_abstained_count")),
                ("Answer", "Semantic non-abstained scored n"):
                    applicable_int(answer.get("semantic_non_abstained_scored_count")),
                ("Answer", "Numeric accuracy"): proportion(result["numeric_accuracy"]),
                ("Answer", "Numeric precision"): proportion(answer["numeric_precision"]),
                ("Answer", "Numeric F1"): proportion(answer["numeric_f1"]),
                ("Answer", "Numeric n"): int(result["numeric_applicable_n"]),
                ("Evidence", "Context support"): proportion(result["context_faithfulness"]),
                ("Evidence", "Context hallucination proxy"): proportion(result["context_hallucination"]),
                ("Evidence", "Context n"): int(result["context_scored_n"]),
                ("Evidence", "Gold support"): proportion(result["gold_evidence_support"]),
                ("Evidence", "Gold hallucination proxy"): proportion(result["gold_evidence_hallucination"]),
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
            ("Whole-answer correctness", "P0-P3; valid semantic judgments", "Post-hoc model-judged binary fully-correct rate against the question, reference answer and annotated evidence. Exact abstentions score zero; unresolved judge errors have no score. Full-question lower/upper bounds retain their uncertainty. Not a human-verified accuracy rate."),
            ("Reference-unit coverage (completeness proxy)", "P0-P3; valid semantic judgments", "Post-hoc model-judged fraction of generated reference units covered correctly, macro-averaged over scored questions. Minimal requested units were intended, but a convenience audit found optional supporting details included and compatible paraphrases rejected. Extra incorrect claims may lower correctness without lowering coverage."),
            ("BERTScore P/R/F1", "P0-P3; all questions", "Contextual token-embedding similarity to the gold answer. Precision/recall are embedding matches, not factual precision/recall; F1 is similarity, not an accuracy percentage."),
            ("Numeric accuracy", "P0-P3; gold-numeric answers", "Matched gold numeric entities / gold numeric entities; corrected v2 extractor."),
            ("Numeric precision / F1", "P0-P3; applicable answers", "Precision penalizes extra generated numeric entities; F1 combines precision and recall."),
            ("Context support", "P1-P3; scored answers", "Model-judged supported factual units / scored factual units against final retrieved chunks; macro mean across applicable answers."),
            ("Gold-evidence support", "P0-P3; scored answers", "Same model-judged unit measure against FinanceBench annotated evidence; applicability can differ from context."),
            ("Hallucination proxy", "P0-P3 where support applies", "One minus corresponding support over the same scored answers; unsupported units are not independently verified false answers."),
            ("Exact abstention", "P0-P3", "Exact refusal phrase count; shown separately, not scored as zero support."),
            ("Latency", "P0-P3", "Recorded per-question component/total time; excludes model and pipeline initialization."),
        ], columns=["Metric", "Applies to", "Definition"])

    def retrieval(self) -> pd.DataFrame:
        """Stage 15 page-level Hit@1/3/5, Recall@5, and MRR@5."""
        return _compact(self.tables["retrieval"], ["architecture", "question_count",
            "no_result_count", "hit_at_1", "hit_at_3", "hit_at_5", "recall_at_5", "mrr_at_5"])

    def answers(self) -> pd.DataFrame:
        """Legacy lexical and numeric diagnostic scores, preserved unchanged."""
        return _compact(self.tables["answers"], ["architecture", "question_count",
            "numeric_applicable_count", "exact_match", "token_f1", "numeric_accuracy",
            "numeric_precision", "numeric_recall", "numeric_f1"])

    def answer_quality(self) -> pd.DataFrame:
        """Whole-answer semantic judgments and reference similarity with denominators."""
        frame = self.tables["answers"]
        columns = ["architecture", "question_count",
            "answer_correctness", "answer_completeness", "semantic_scored_count",
            "bertscore_precision", "bertscore_recall", "bertscore_f1", "bertscore_scored_count",
            "exact_abstention_count", "non_abstained_count",
            "answer_correctness_non_abstained", "answer_completeness_non_abstained",
            "bertscore_f1_non_abstained"]
        columns.extend(column for column in ("judge_error_count",
            "answer_correctness_lower_bound", "answer_correctness_upper_bound",
            "semantic_non_abstained_scored_count") if column in frame)
        return _compact(frame, columns)

    def answer_quality_labels(self) -> pd.DataFrame:
        """Mutually exclusive model-judged outcomes; exact refusal is a separate flag."""
        frame = self.tables["answers"]
        columns = ["architecture", "question_count", "correct_count",
            "partially_correct_count", "incorrect_count", "failure_to_answer_count",
            "exact_abstention_count"]
        if "judge_error_count" in frame:
            columns.append("judge_error_count")
        return _compact(frame, columns)

    def answer_quality_scoring_note(self) -> str:
        """Make unresolved judgments and semantic denominators explicit."""
        frame = self.tables["answers"]
        if "judge_error_count" not in frame or not frame["judge_error_count"].fillna(0).sum():
            return ("Semantic means use valid judgments, including zero-valued exact abstentions. "
                    "The scored n is shown for every architecture.")
        counts = ", ".join(f"{row.architecture}: {int(row.judge_error_count)}"
                           for row in frame.itertuples())
        return ("**Partial semantic scoring:** unresolved judge errors were " + counts + ". "
                "These responses have no correctness or coverage score and are excluded from "
                "semantic means; they are not imputed as incorrect. Exact abstentions remain "
                "valid zero scores. The full-question correctness lower bound is correct/n; "
                "the upper bound is (correct + judge errors)/n. These bounds cover unresolved "
                "judgments, not undetected judge mistakes. Conditional semantic means also "
                "exclude judge errors and have their own valid non-abstained n.")

    def common_answer_quality(self) -> pd.DataFrame:
        """Compare all architectures on IDs where P1-P3 did not exact-abstain."""
        self._load_question_data()
        frame = self._question_tables["answers"].reset_index()
        rag = frame.loc[frame["architecture"].isin(ARCHITECTURES[1:])]
        grouped = rag.groupby("question_id")["exact_abstention"]
        shared_ids = grouped.sum().loc[lambda values: values == 0].index
        selected = frame.loc[frame["question_id"].isin(shared_ids)]
        metrics = ["answer_correctness", "answer_completeness", "bertscore_f1", "token_f1"]
        rows = []
        for arch in ARCHITECTURES:
            members = selected.loc[selected["architecture"] == arch]
            row = {"architecture": arch, "common_question_count": len(members)}
            for metric in metrics:
                row[metric] = members[metric].mean()
                row[f"{metric}_n"] = int(members[metric].notna().sum())
            missing = int(members["answer_correctness"].isna().sum())
            row["judge_error_count"] = missing
            row["answer_correctness_lower_bound"] = (
                members["answer_correctness"].sum() / len(members) if len(members) else pd.NA)
            row["answer_correctness_upper_bound"] = (
                (members["answer_correctness"].sum() + missing) / len(members)
                if len(members) else pd.NA)
            rows.append(row)
        return pd.DataFrame(rows)

    def faithfulness(self) -> pd.DataFrame:
        """Stage 17 support/hallucination with exact-abstention and coverage counts."""
        return _compact(self.tables["faithfulness"], ["architecture", "question_count",
            "abstention_count", "context_applicable_count", "context_faithfulness",
            "context_hallucination", "gold_applicable_count", "gold_evidence_support",
            "gold_evidence_hallucination", "judge_model", "judge_protocol",
            "context_guard_override_count", "gold_guard_override_count"])

    def support_results(self) -> pd.DataFrame:
        """Model-judged support and complementary proxies with scored-answer counts."""
        frame = self.tables["faithfulness"]

        def score_with_n(score: Any, count: Any) -> str:
            return f"{score:.3f} ({int(count)})" if pd.notna(score) else "NA"

        return pd.DataFrame([
            {"ID": row["architecture"], "Abstentions": int(row["abstention_count"]),
             "Context support (scored n)": score_with_n(
                 row["context_faithfulness"], row["context_applicable_count"]),
             "Context hallucination proxy": row["context_hallucination"],
             "Gold support (scored n)": score_with_n(
                 row["gold_evidence_support"], row["gold_applicable_count"]),
             "Gold hallucination proxy": row["gold_evidence_hallucination"]}
            for row in frame.to_dict("records")
        ])

    def _load_abstention_data(self) -> None:
        """Read the separately versioned diagnostic exports without regenerating them."""
        if self._abstention_tables is not None:
            return
        manifest_path = self.root / "results/aggregated/abstention_analysis_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Saved abstention analysis is missing. Run "
                "`python scripts/analyze_abstentions.py` once before reading this section."
            )
        manifest = _read_json(manifest_path)
        run_id = self.run_manifest["run_id"]
        if manifest["run_id"] != run_id:
            raise ValueError("Abstention analysis belongs to another frozen run")
        if self._verify_hashes:
            for record in [manifest["run_manifest"], *manifest["inputs"].values(),
                           *manifest["outputs"].values()]:
                path = Path(record["path"])
                if not path.is_absolute():
                    path = self.root / path
                if not path.exists() or sha256(path) != record["sha256"]:
                    raise ValueError(f"Abstention analysis input/output missing or modified: {path}")
        tables = {name: pd.read_csv(self.root / path,
                  dtype={"pattern_p1_p2_p3": "string"}
                  if name in ("overlap", "question_patterns") else None)
                  for name, path in ABSTENTION_TABLE_PATHS.items()}
        for name, frame in tables.items():
            if frame.empty or not frame["run_id"].eq(run_id).all():
                raise ValueError(f"Abstention {name} is empty or belongs to another run")
        question_n = self.run_manifest["question_count"]
        if (len(tables["overlap"]) != 8
                or tables["overlap"]["question_count"].sum() != question_n):
            raise ValueError("Abstention overlap must partition the frozen question set")
        support = tables["support_summary"]
        if len(support) != 4 or set(support["architecture"]) != set(ARCHITECTURES):
            raise ValueError("Abstention support summary must cover P0-P3 exactly once")
        if not support["question_count"].eq(question_n).all():
            raise ValueError("Abstention support denominators differ from frozen question count")
        self._abstention_tables = tables

    def abstentions(self, name: str = "support_summary") -> pd.DataFrame:
        """Copy one saved abstention table, including full question-level CSV exports."""
        if name not in ABSTENTION_TABLE_PATHS:
            raise KeyError(f"Unknown abstention table {name!r}; available: "
                           f"{', '.join(ABSTENTION_TABLE_PATHS)}")
        self._load_abstention_data()
        assert self._abstention_tables is not None
        return self._abstention_tables[name].drop(columns=["run_id"]).copy()

    def abstention_interpretation(self) -> str:
        """Describe observed refusal overlap and evidence coverage without causal claims."""
        overlap = self.abstentions("overlap")
        all_abstained = overlap.loc[
            overlap[["p1_abstained", "p2_abstained", "p3_abstained"]].eq(1).all(axis=1),
            "question_count"].sum()
        none_abstained = overlap.loc[
            overlap[["p1_abstained", "p2_abstained", "p3_abstained"]].eq(0).all(axis=1),
            "question_count"].sum()
        question_n = self.run_manifest["question_count"]
        mixed = question_n - all_abstained - none_abstained
        summary = self.abstentions("support_summary").set_index("architecture")
        counts = ", ".join(
            f"{arch}: {int(summary.loc[arch, 'exact_abstentions'])}/{question_n}"
            for arch in ARCHITECTURES)
        observations = []
        evidence = self.abstentions("evidence_coverage")
        for arch in ARCHITECTURES[1:]:
            rows = evidence.loc[evidence["architecture"] == arch].set_index("exact_abstention")
            abstained, answered = rows.loc[1], rows.loc[0]
            observations.append(
                f"{arch}: {int(abstained['final_gold_page_hit'])}/"
                f"{int(abstained['question_count'])} exact abstentions retained an annotated "
                f"gold page in the final context; {int(answered['no_final_gold_page_hit'])}/"
                f"{int(answered['question_count'])} non-abstentions had no final gold-page hit."
            )
        return (
            "## Abstention patterns and evidence diagnostics\n\n"
            "P1-P3 were instructed to use only the supplied context and return "
            "`Insufficient evidence in the retrieved context.` when the answer could "
            "not be determined. The saved refusals do not record a more specific "
            "explanation of that decision.\n\n"
            "**Definition:** An exact abstention matches the frozen evaluator's complete "
            "refusal phrase after its fixed normalization. Partial answers or answers "
            "containing a caveat remain non-abstentions unless the whole answer matches. "
            "Non-abstention therefore does not establish completeness or correctness.\n\n"
            f"Exact abstentions were {counts}. Across P1-P3, **{int(all_abstained)}** "
            f"questions were abstained by all three, **{int(none_abstained)}** by none, and "
            f"**{int(mixed)}** had mixed decisions. Equal totals can mask different question "
            "sets; the eight patterns are mutually exclusive and the pairwise table shows "
            "both one-sided disagreements and shared decisions. P0 is the closed-book "
            "baseline and has no retrieval evidence diagnostic.\n\n"
            + "\n\n".join(observations)
            + "\n\n**How to read the reasons:** The saved diagnostics locate observable "
            "evidence-coverage conditions: an annotated gold page can be absent from the "
            "candidate union, lost before pre-final selection, lost at final selection, "
            "or retained in the final context. They describe associations with abstention, "
            "not the model's internal reason for refusing. A page hit does not establish "
            "that the retrieved chunk contains every fact needed for the answer; a miss "
            "can still contain equivalent evidence outside the annotations. Empty retrieval "
            "is shown separately and overlaps the no-final-hit group. Compare exact "
            "abstentions with non-abstentions before attributing refusals to retrieval.\n\n"
            "**Retrieval stages:** P1's observed candidate pool is dense top 5 and is "
            "also its final context. P2 combines dense top 10 and BM25 top 10; its "
            "pre-final RRF top 5 is the final context. P3 combines dense top 20 and "
            "BM25 top 20, keeps RRF top 20, then selects the reranked top 5. A loss "
            "before pre-final selection is therefore an RRF cutoff; a final-selection "
            "loss in P3 is the reranked top-5 cutoff. Different candidate budgets mean "
            "union misses cannot be compared as a controlled component effect. "
            "A P1 pool miss refers only to its observed dense top 5."
        )

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

    def category_answer_quality(self, grouping: str = "question_reasoning") -> pd.DataFrame:
        """Native category semantic quality, similarity and support, each with its own n."""
        if grouping not in ("question_type", "question_reasoning"):
            raise ValueError("grouping must be 'question_type' or 'question_reasoning'")
        frame = self.tables["category"].loc[self.tables["category"]["grouping"] == grouping]
        measures = (
            ("Model-judged correctness (n)", "answer_correctness"),
            ("Reference-unit coverage proxy (n)", "answer_completeness"),
            ("BERTScore F1 similarity (n)", "bertscore_f1"),
            ("Context support (n)", "context_faithfulness"),
            ("Gold support (n)", "gold_evidence_support"),
        )
        rows = []
        for source in frame.to_dict("records"):
            row = {key: source[key] for key in ("category", "architecture", "question_count")}
            for label, metric in measures:
                value, count = source[f"{metric}_mean"], source[f"{metric}_n"]
                row[label] = f"{value:.3f} ({int(count)})" if pd.notna(value) else "NA"
            row["Exact abstention rate"] = source["exact_abstention_mean"]
            for column, label in (("judge_error_count", "Judge errors"),
                    ("answer_correctness_lower_bound", "Correctness lower bound"),
                    ("answer_correctness_upper_bound", "Correctness upper bound")):
                if column in source:
                    row[label] = source[column]
            rows.append(row)
        return pd.DataFrame(rows)

    def hypotheses(self) -> pd.DataFrame:
        """The substantive P0-P3 predictions, not statistical nulls."""
        return pd.DataFrame([
            ("P0", "Closed-book P0 has most hallucination, least accuracy, and lowest latency.", "P0 versus P1-P3"),
            ("P1", "Dense P1 improves faithfulness over P0 but may miss exact terms/numbers.", "P0 versus P1"),
            ("P2", "Hybrid P2 improves retrieval on numeric/terminology-heavy questions.", "P1 versus P2"),
            ("P3", "Reranked hybrid P3 improves text-based faithfulness at a latency cost.", "P2 versus P3"),
        ], columns=["Hypothesis", "Substantive prediction", "Primary contrast"])

    def hypothesis_tests(self, hypothesis: str | None = None) -> pd.DataFrame:
        """All paired contrasts, optionally filtered by frozen H0-H3 or P0-P3 alias."""
        frame = self.tables["tests"]
        if hypothesis is not None:
            if hypothesis not in ("H0", "H1", "H2", "H3", "P0", "P1", "P2", "P3"):
                raise ValueError("hypothesis must be H0-H3 or its P0-P3 alias")
            frozen_tag = "H" + hypothesis[1]
            frame = frame.loc[frame["hypothesis"].str.split(",").map(lambda tags: frozen_tag in tags)]
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
                "answer_quality_label": answers.get("answer_quality_label", pd.NA),
                "answer_correctness": answers.get("answer_correctness", pd.NA),
                "answer_completeness": answers.get("answer_completeness", pd.NA),
                "bertscore_f1": answers.get("bertscore_f1", pd.NA),
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
            "Whole-answer correctness and reference-unit coverage (a completeness proxy) are "
            "post-hoc model judgments, not human-verified labels. A convenience audit found "
            "optional supporting values included in reference units and compatible phrasing "
            "rejected. P0's 'over 60 years' for 3M was rejected despite evidence of 65 years "
            "(financebench_id_01858); its correct Adobe improvement conclusion was penalized "
            "for omitted supporting percentages (financebench_id_00591). The audit is not a "
            "representative sample or judge error-rate estimate, and no scores were manually "
            "corrected. BERTScore is configuration-specific semantic similarity "
            "and can reward shared language despite a wrong entity, period, polarity or number. "
            "It is not a percentage of correct answers. Conditional response scores compare "
            "different question subsets unless the same-question table is used. The original "
            "paired tests retain their frozen metric family; the new semantic results are "
            "descriptive and do not extend those prespecified significance claims."
        )

    def interpretation(self) -> str:
        """Numerical reading of P0-P3 and the trade-off, without declaring a winner."""
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
        semantic_reading = ""
        if self.has_answer_quality:
            summaries = self.tables["answers"].set_index("architecture")
            semantic_reading = (
                "- **Post-hoc semantic assessment:** Valid-judgment model-judged fully-correct "
                "answer rates were " + ", ".join(
                    f"{arch}: {summaries.loc[arch, 'answer_correctness']:.3f} "
                    f"({int(summaries.loc[arch, 'correct_count'])}/"
                    f"{int(summaries.loc[arch, 'semantic_scored_count'])})"
                    for arch in ARCHITECTURES)
                + ". The reference-unit coverage proxy, outcome-count and same-question tables "
                "show what these binary rates omit. BERTScore is complementary similarity. "
                "Missing judge scores and all-question correctness bounds are reported separately. "
                "These post-hoc results are descriptive; the frozen paired tests below "
                "remain tests of their original metrics.\n"
            )
        return (
            "## Reading the results\n\n"
            + semantic_reading +
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
            display(Markdown("## Retrieval quality\n\nP0 has no retrieval "
                "metric. Hit@k, Recall@5, and MRR@5 use annotated gold pages, not document "
                "names alone. Each architecture contributes 145 questions."))
            table(self.retrieval())
            self.show_figure("2")
            return
        if section == "answers":
            if not self.has_answer_quality:
                display(Markdown("## Answer diagnostics\n\nThe saved outputs currently contain "
                    "lexical and numerical diagnostics. The post-hoc semantic-correctness, "
                    "reference-completeness and BERTScore evaluation has not yet been published."))
                table(self.answers())
                self.show_figure("3")
                return
            display(Markdown("## Answer correctness, completeness and reference similarity\n\n"
                "A post-hoc model judge compares each saved answer with the question, gold "
                "answer and annotated gold evidence. Whole-answer correctness is the fraction "
                "labelled fully correct; partial, incorrect and failure-to-answer outcomes count "
                "as zero among valid judgments. Reference-unit coverage is a completeness proxy: "
                "the fraction of generated reference units judged correctly covered, averaged "
                "over scored questions. Minimal question-required units were intended, but a "
                "convenience audit found optional supporting details in some rubrics and "
                "compatible paraphrases rejected. Coverage can reach one when all reference "
                "units are present even if added contradictory reasoning makes "
                "the whole answer incorrect. These are automated model judgments without "
                "independent human validation; they are not directly comparable to a "
                "human-assessed benchmark accuracy rate. BERTScore uses contextual token embeddings to compare "
                "the generated and reference answers; its P/R/F1 are similarity measures, not "
                "factual precision, recall or percentages of correct answers.\n\n"
                "The valid-judgment means include exact abstentions. Unresolved judge errors "
                "have missing scores and are excluded, with full-question correctness bounds "
                "shown separately. The conditional columns exclude "
                "only exact refusals and use each pipeline's own response subset; a longer refusal "
                "can still be a failure to answer. Scored denominators are shown explicitly."))
            table(self.answer_quality())
            display(Markdown(self.answer_quality_scoring_note()))
            display(Markdown("**Observed judge limitations:** P0's 'over 60 years' was rejected "
                "although the 3M evidence says 65 years (`financebench_id_01858`). Its correct "
                "Adobe FCF-improvement conclusion was penalized for omitting supporting "
                "percentages (`financebench_id_00591`). These convenience-audit examples do "
                "not estimate error prevalence. Scores were not manually corrected; the "
                "audit and source hashes are saved in `results/evaluation/answer_quality_v3_audit.md`."))
            if self.answer_quality_protocol is not None:
                protocol = self.answer_quality_protocol["parameters"]
                bert = protocol["bertscore"]
                display(Markdown(f"The saved post-hoc protocol uses `{protocol['judge_model']}` "
                    f"and `{protocol['judge_protocol']}` for semantic judgments. Each question "
                    "has one architecture-blind reference-unit rubric shared by P0-P3. "
                    "The semantic scores remain automated judgments and have not been "
                    "independently validated by human annotators. "
                    f"BERTScore uses `{bert['model_id']}`, revision `{bert['model_revision']}`, "
                    f"layer {bert['num_layers']}, IDF weighting `{bert['idf']}` and baseline "
                    f"rescaling `{bert['rescale_with_baseline']}`. Results depend on this "
                    "configuration and should not be directly compared with different article "
                    "datasets or judge rubrics."))
            display(Markdown("### Model-judged outcome counts\n\nThe four substantive outcome "
                "counts plus unresolved judge errors partition each architecture's questions. "
                "A judge error is an evaluation failure, not an answer outcome. Failure to answer is broader "
                "than the frozen whole-response exact-abstention detector."))
            table(self.answer_quality_labels())
            display(Markdown("### Same-question comparison\n\nThe following compares all "
                "four architectures on the same question IDs where P1, P2 and P3 did not "
                "exact-abstain. It is selected by response behaviour and does not represent "
                "the entire benchmark. Missing judge scores are excluded metric by metric; "
                "the valid scored n may therefore differ even within this shared question set."))
            table(self.common_answer_quality())
            display(Markdown("### Lexical and numerical diagnostics\n\nExact match and "
                "token F1 measure surface overlap. Numeric values use the corrected v2 "
                "extractor, with a separate applicable denominator. These original scores are "
                "retained alongside the new semantic assessment."))
            table(self.answers())
            self.show_figure("3")
            return
        if section == "faithfulness":
            protocol = self.faithfulness_protocol["parameters"]
            display(Markdown("## Model-judged factual-unit support and hallucination proxies\n\n"
                f"One fixed `{protocol['judge_model']}` judge and `{protocol['judge_protocol']}` "
                "rubric evaluated P0-P3. Context support uses the final retrieved chunks "
                "(P1-P3); gold support uses FinanceBench annotated evidence (P0-P3). "
                "For each answer with at least one judge-scored factual unit, support is the "
                "proportion of those units judged supported. Each reported support score is "
                "a question-level macro mean over applicable answers. The hallucination "
                "proxy is the proportion judged unsupported: **1 − support**. Each "
                "support–proxy pair therefore sums to one by definition; for example, "
                "0.850 + 0.150 = 1. These are two views of the same judgments. Unsupported "
                "units are not independently verified false answers.\n\n"
                "**Table caption:** Model-judged factual-unit support and hallucination "
                "proxies. Context refers to the final retrieved chunks; gold refers to "
                "annotated FinanceBench evidence. Parentheses give the number of scored answers."))
            table(self.support_results())
            display(Markdown("The applicable sets differ across architectures and evidence "
                "sources; compare the scored denominators and abstention counts alongside "
                "these conditional means. Context and gold scores should not be added. "
                "Exact abstentions and other answers without scored factual units are "
                "excluded from the corresponding mean. P0 has no context score because "
                "it received no retrieved chunks. The saved figure below uses the original "
                "evaluation column labels; its hallucination series are the same proxies."))
            self.show_figure("4")
            return
        if section == "abstentions":
            display(Markdown(self.abstention_interpretation()))
            display(Markdown("### Abstention and scored-answer coverage\n\n"
                "Scored n counts answers with at least one factual unit scored against "
                "that evidence source. Other non-abstentions can still be unscored; "
                "subtracting exact refusals alone does not recover a support denominator."))
            table(_compact(self.abstentions(), ["architecture", "question_count",
                "exact_abstentions", "exact_abstention_rate", "non_abstentions",
                "context_scored_n", "context_unscored_non_abstentions",
                "gold_scored_n", "gold_unscored_non_abstentions"]))
            display(Markdown("### Joint P1-P3 decisions\n\n"
                "Pattern bits are ordered P1, P2, P3: 1 means exact abstention and "
                "0 means non-abstention. The eight rows partition all questions."))
            table(self.abstentions("overlap").drop(columns=["question_ids_json"]))
            display(Markdown("### Pairwise overlap and disagreement\n\n"
                "Jaccard overlap is both abstained / either abstained. Each pair's both, "
                "A-only, B-only, and neither counts sum to the frozen question count."))
            table(self.abstentions("pairwise"))
            questions = self.abstentions("question_patterns")
            mixed = questions.loc[~questions["pattern_p1_p2_p3"].isin(["000", "111"])]
            display(Markdown("### Questions with different decisions\n\n"
                "One example per observed mixed pattern is shown below. The full saved "
                "question-pattern table includes every question and all four generated answers."))
            table(_compact(mixed.groupby("pattern_p1_p2_p3", sort=True).head(1),
                ["question_id", "question", "pattern_p1_p2_p3",
                 "p1_abstained", "p2_abstained", "p3_abstained"]))
            display(Markdown("### Evidence conditions for both decision groups\n\n"
                "The three loss-location columns partition no-final-gold-page hits. "
                "They are page-level observations, not verified explanations of model intent."))
            table(self.abstentions("evidence_coverage"))
            display(Markdown("### Native question categories\n\n"
                "Rates use each architecture/category's full question count. Missing native "
                "labels are retained as a separate category, and categories are descriptive."))
            table(self.abstentions("categories"))
            display(Markdown("All question IDs, P0-P3 answers, refusal flags, and diagnostic "
                "conditions are saved in `abstention_question_details.csv` and "
                "`abstention_question_patterns.csv`. Inspect them with "
                "`report.abstentions('question_details')` or "
                "`report.abstentions('question_patterns')`; filter the saved rows to compare "
                "the exact questions on which decisions differed."))
            return
        if section == "latency":
            display(Markdown("## Recorded latency\n\nComponent, mean, "
                "median, standard deviation, minimum, and maximum are in milliseconds. "
                "Inapplicable components are NA; dense/BM25/fusion sub-times were not "
                "recorded separately. Setup is excluded, but API/network effects remain."))
            table(self.latency())
            self.show_figure("5")
            return
        if section == "categories":
            coverage = self.category_protocol["parameters"]["label_coverage"]
            display(Markdown("## Native FinanceBench categories\n\n"
                f"`question_type` labels cover {coverage['question_type']} questions; "
                f"`question_reasoning` labels cover {coverage['question_reasoning']}. "
                "The remaining reasoning labels are source-null and excluded only from "
                "reasoning groups. No outcome-informed derived labels were created."))
            display(Markdown("### Question type"))
            if self.has_answer_quality and "answer_correctness_mean" in self.tables["category"]:
                table(self.category_answer_quality("question_type"))
                display(Markdown("Correctness/reference-unit coverage proxy/BERTScore means "
                    "use each native category's valid scored answers; unresolved judge scores "
                    "are excluded. Evidence support is conditional on scored factual "
                    "units; parentheses show each metric's own n. These metrics are not summed. "
                    "Only the benchmark's existing category labels are used."))
            display(Markdown("### Question-type lexical, numeric and retrieval diagnostics"))
            table(self.categories("question_type"))
            display(Markdown("### Question reasoning"))
            if self.has_answer_quality and "answer_correctness_mean" in self.tables["category"]:
                table(self.category_answer_quality("question_reasoning"))
            display(Markdown("### Reasoning-category lexical, numeric and retrieval diagnostics"))
            table(self.categories("question_reasoning"))
            self.show_figure("7")
            return
        if section == "statistics":
            display(Markdown("## Paired hypothesis analysis\n\nThese P0-P3 "
                "labels name the substantive predictions, not statistical "
                "nulls. Every contrast pairs the same question IDs. McNemar exact handles "
                "binary outcomes; Wilcoxon signed-rank handles scores; mean differences "
                "have seeded paired-bootstrap 95% intervals. Holm adjustment spans all "
                "21 original comparisons. Scored-answer tests use complete pairs only. "
                "The added post-hoc correctness, completeness and BERTScore results are "
                "descriptive; they have not been inserted into the original prespecified "
                "test family."))
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
        """Display the complete, ordered results walkthrough."""
        for section in SECTION_ORDER:
            self.show(section)


def load_report(root: Path = ROOT, *, verify_hashes: bool = True) -> FinanceBenchReport:
    """Load and validate the persisted experiment without recomputing it."""
    return FinanceBenchReport(root, verify_hashes=verify_hashes)
