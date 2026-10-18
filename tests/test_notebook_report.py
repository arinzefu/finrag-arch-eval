from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.evaluation_common import ROOT, sha256
from scripts.notebook_report import FinanceBenchReport, SECTION_ORDER, load_report


class AnswerQualityReportTests(unittest.TestCase):
    def test_quality_provenance_rejects_modified_journal_and_wrong_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = FinanceBenchReport.__new__(FinanceBenchReport)
            report.root = root
            report.run_manifest = {"run_id": "r"}
            report.tables = {"answers": pd.DataFrame()}
            report.final_manifest = {"inputs": {}}
            run = root / "results/raw_runs/financebench-heldout-v1_manifest.json"
            run.parent.mkdir(parents=True)
            run.write_text("{}", encoding="utf-8")
            inputs = {}
            for name in ("evidence", "reference_rubrics", "judgments", "bertscore",
                         "semantic_source", "bertscore_source", "evaluator_source",
                         "P0", "P1", "P2", "P3"):
                path = root / f"{name}.txt"
                path.write_text(name, encoding="utf-8")
                inputs[name] = {"path": str(path), "sha256": sha256(path)}
            outputs = {}
            for name, relative in (("per_question", "results/evaluation/answer_metrics_v2.csv"),
                                   ("summary", "results/aggregated/answer_summary_v2.csv")):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
                outputs[name] = {"path": str(path), "sha256": sha256(path)}
            report.answer_quality_protocol = {"run_id": "r", "inputs": inputs,
                "outputs": outputs, "run_manifest": {"path": str(run), "sha256": sha256(run)}}
            report._validate_answer_quality(True)
            Path(inputs["judgments"]["path"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "judgments"):
                report._validate_answer_quality(True)
            # Disabling content hashes never permits a different frozen run.
            report.answer_quality_protocol["run_id"] = "different"
            with self.assertRaisesRegex(ValueError, "different frozen run"):
                report._validate_answer_quality(False)

    def test_common_subset_uses_question_ids_not_each_pipeline_subset(self) -> None:
        report = FinanceBenchReport.__new__(FinanceBenchReport)
        report._raw_runs = {}  # Synthetic per-question tables already loaded.
        rows = []
        for arch in ("P0", "P1", "P2", "P3"):
            for qid, score in (("shared", 1.0), ("different", 0.0)):
                rows.append({"architecture": arch, "question_id": qid,
                    "exact_abstention": int(arch == "P2" and qid == "different"),
                    "answer_correctness": score, "answer_completeness": score,
                    "bertscore_f1": score, "token_f1": score})
        report._question_tables = {"answers": pd.DataFrame(rows).set_index(
            ["architecture", "question_id"])}
        result = report.common_answer_quality()
        self.assertTrue(result["common_question_count"].eq(1).all())
        self.assertTrue(result["answer_correctness"].eq(1).all())
        self.assertTrue(result["bertscore_f1_n"].eq(1).all())

    def test_common_subset_retains_missing_judgment_counts_and_bounds(self) -> None:
        report = FinanceBenchReport.__new__(FinanceBenchReport)
        report._raw_runs = {}
        rows = []
        for arch in ("P0", "P1", "P2", "P3"):
            for qid, score in (("scored", 1.0), ("judge_error", float("nan"))):
                rows.append({"architecture": arch, "question_id": qid,
                    "exact_abstention": 0, "answer_correctness": score,
                    "answer_completeness": score, "bertscore_f1": .8, "token_f1": .5})
        report._question_tables = {"answers": pd.DataFrame(rows).set_index(
            ["architecture", "question_id"])}
        result = report.common_answer_quality()
        self.assertTrue(result["common_question_count"].eq(2).all())
        self.assertTrue(result["answer_correctness"].eq(1).all())
        self.assertTrue(result["answer_correctness_n"].eq(1).all())
        self.assertTrue(result["bertscore_f1_n"].eq(2).all())
        self.assertTrue(result["judge_error_count"].eq(1).all())
        self.assertTrue(result["answer_correctness_lower_bound"].eq(.5).all())
        self.assertTrue(result["answer_correctness_upper_bound"].eq(1).all())

    def test_partial_semantic_note_does_not_impute_judge_errors(self) -> None:
        report = FinanceBenchReport.__new__(FinanceBenchReport)
        report.tables = {"answers": pd.DataFrame([
            {"architecture": "P0", "judge_error_count": 0},
            {"architecture": "P1", "judge_error_count": 2}])}
        note = report.answer_quality_scoring_note()
        self.assertIn("P1: 2", note)
        self.assertIn("not imputed as incorrect", note)
        self.assertIn("not undetected judge mistakes", note)

    def test_category_quality_retains_different_scoring_denominators(self) -> None:
        report = FinanceBenchReport.__new__(FinanceBenchReport)
        report.tables = {"category": pd.DataFrame([{
            "grouping": "question_type", "category": "native category", "architecture": "P0",
            "question_count": 10, "answer_correctness_mean": .3, "answer_correctness_n": 10,
            "answer_completeness_mean": .4, "answer_completeness_n": 10,
            "bertscore_f1_mean": .8, "bertscore_f1_n": 10,
            "context_faithfulness_mean": float("nan"), "context_faithfulness_n": 0,
            "gold_evidence_support_mean": .9, "gold_evidence_support_n": 3,
            "exact_abstention_mean": 0,
        }])}
        row = report.category_answer_quality("question_type").iloc[0]
        self.assertEqual(row["Model-judged correctness (n)"], "0.300 (10)")
        self.assertEqual(row["Gold support (n)"], "0.900 (3)")
        self.assertEqual(row["Context support (n)"], "NA")
        with self.assertRaises(ValueError):
            report.category_answer_quality("addition")


@unittest.skipUnless((ROOT / "results/final_outputs_manifest.json").exists(),
                     "Stage 22 outputs are required for the notebook report")
class NotebookReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = load_report()

    def test_all_architectures_and_na_are_preserved(self) -> None:
        self.assertEqual(self.report.architectures()["architecture"].tolist(),
                         ["P0", "P1", "P2", "P3"])
        overall = self.report.overall_results().set_index("architecture")
        self.assertTrue(pd.isna(overall.loc["P0", "hit_at_5"]))
        self.assertTrue(pd.isna(overall.loc["P0", "context_faithfulness"]))
        self.assertEqual(self.report.retrieval().shape[0], 3)
        self.assertEqual(self.report.answers().shape[0], 4)
        self.assertEqual(self.report.faithfulness().shape[0], 4)

    def test_all_report_sections_have_saved_data(self) -> None:
        self.assertEqual(len(SECTION_ORDER), 14)
        self.assertEqual(SECTION_ORDER[SECTION_ORDER.index("faithfulness") + 1], "abstentions")
        self.assertEqual(len(self.report.categories("question_type")), 12)
        self.assertEqual(len(self.report.categories("question_reasoning")), 36)
        self.assertEqual(len(self.report.hypothesis_tests()), 21)
        self.assertEqual(len(self.report.hypothesis_tests("P1")), 1)
        self.assertEqual(len(self.report.frontiers()), 14)
        self.assertEqual(len(self.report.paired_tradeoffs()), 19)
        self.assertTrue(self.report.figure_path("6c").exists())

    def test_comparison_matrix_preserves_controls_and_applicability(self) -> None:
        matrix = self.report.comparison_matrix()
        self.assertEqual(matrix.index.tolist(), ["P0", "P1", "P2", "P3"])
        self.assertEqual(matrix.loc["P0", ("Shared controls", "Answer LLM")], "gpt-4o-mini")
        self.assertTrue(matrix[("Shared controls", "Answer LLM")].eq("gpt-4o-mini").all())
        self.assertTrue(pd.isna(matrix.loc["P0", ("Design", "RRF k")]))
        self.assertTrue(pd.isna(matrix.loc["P1", ("Design", "RRF k")]))
        self.assertEqual(matrix.loc["P2", ("Design", "RRF k")], 60)
        self.assertEqual(matrix.loc["P3", ("Design", "RRF k")], 60)
        self.assertTrue(pd.isna(matrix.loc["P0", ("Retrieval", "Hit@5")]))
        self.assertTrue(pd.isna(matrix.loc["P0", ("Evidence", "Context support")]))
        self.assertEqual(matrix.loc["P3", ("Evidence", "Context n")], 86)
        self.assertEqual(matrix.loc["P3", ("Latency", "Mean total (s)")], 5.63)

    def test_support_display_names_evidence_and_complementary_proxies(self) -> None:
        table = self.report.support_results().set_index("ID")
        self.assertEqual(table.loc["P0", "Context support (scored n)"], "NA")
        self.assertEqual(table.loc["P1", "Context support (scored n)"], "0.850 (85)")
        self.assertEqual(table.loc["P3", "Gold support (scored n)"], "0.652 (85)")
        source = self.report.faithfulness()
        for evidence, support, proxy in (
            ("Context", "context_faithfulness", "context_hallucination"),
            ("Gold", "gold_evidence_support", "gold_evidence_hallucination"),
        ):
            scored = source.dropna(subset=[support, proxy])
            self.assertTrue(((scored[support] + scored[proxy]) - 1).abs().lt(1e-12).all())
            self.assertIn(f"{evidence} hallucination proxy", table)

    def test_abstention_tables_cover_both_groups_and_all_joint_patterns(self) -> None:
        question_n = self.report.run_manifest["question_count"]
        overlap = self.report.abstentions("overlap")
        self.assertEqual(len(overlap), 8)
        self.assertEqual(int(overlap["question_count"].sum()), question_n)
        self.assertIn("000", overlap["pattern_p1_p2_p3"].tolist())
        evidence = self.report.abstentions("evidence_coverage")
        self.assertEqual(len(evidence), 6)
        for _, group in evidence.groupby("architecture"):
            self.assertEqual(set(group["exact_abstention"]), {0, 1})
            self.assertEqual(int(group["question_count"].sum()), question_n)
        self.assertEqual(len(self.report.abstentions("question_details")), 4 * question_n)
        self.assertEqual(len(self.report.abstentions("question_patterns")), question_n)
        overlap.loc[0, "question_count"] = -1
        self.assertGreaterEqual(self.report.abstentions("overlap").loc[0, "question_count"], 0)
        narrative = self.report.abstention_interpretation()
        self.assertIn("not the model's internal reason", narrative)
        self.assertIn("non-abstentions", narrative)

    def test_preselected_case_shows_all_four_frozen_answers(self) -> None:
        sample = self.report.audit_sample()
        self.assertEqual(len(sample), 8)
        case = self.report.case_study(sample.iloc[0]["question_id"])
        self.assertEqual(case["answers"]["architecture"].tolist(),
                         ["P0", "P1", "P2", "P3"])
        self.assertTrue(pd.isna(case["answers"].iloc[0]["hit_at_5"]))
        self.assertIn("gold_answer", case)

    def test_invalid_selection_does_not_silently_fall_back(self) -> None:
        with self.assertRaises(ValueError):
            self.report.categories("derived_numeric")
        with self.assertRaises(ValueError):
            self.report.hypothesis_tests("H4")
        with self.assertRaises(KeyError):
            self.report.case_study("not-a-frozen-question")
        with self.assertRaises(KeyError):
            self.report.figure_path("8")
        with self.assertRaises(KeyError):
            self.report.abstentions("inferred_model_motive")

    def test_complete_walkthrough_renders_without_writing(self) -> None:
        with patch("IPython.display.display") as display:
            self.report.show_all()
        self.assertGreater(display.call_count, 30)


if __name__ == "__main__":
    unittest.main()
