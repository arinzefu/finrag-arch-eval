from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.evaluation_common import ROOT
from scripts.notebook_report import SECTION_ORDER, load_report


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
        self.assertEqual(len(SECTION_ORDER), 13)
        self.assertEqual(len(self.report.categories("question_type")), 12)
        self.assertEqual(len(self.report.categories("question_reasoning")), 36)
        self.assertEqual(len(self.report.hypothesis_tests()), 21)
        self.assertEqual(len(self.report.hypothesis_tests("H1")), 1)
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

    def test_complete_walkthrough_renders_without_writing(self) -> None:
        with patch("IPython.display.display") as display:
            self.report.show_all()
        self.assertGreater(display.call_count, 30)


if __name__ == "__main__":
    unittest.main()
