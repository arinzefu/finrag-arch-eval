from __future__ import annotations

import unittest

from scripts.analyze_question_types import evaluate


class QualityCategoryCoverageTests(unittest.TestCase):
    def test_judge_errors_are_missing_and_have_all_question_bounds(self) -> None:
        manifest = {"run_id": "r", "question_ids": ["correct", "unscored", "refused"]}
        native = {qid: {"question_type": "native", "question_reasoning": None}
                  for qid in manifest["question_ids"]}
        answers = {
            ("P0", "correct"): {"answer_quality_label": "correct", "answer_correctness": 1,
                                "answer_completeness": 1, "exact_abstention": 0, "bertscore_f1": .9},
            ("P0", "unscored"): {"answer_quality_label": "judge_error", "answer_correctness": "",
                                 "answer_completeness": "", "exact_abstention": 0, "bertscore_f1": .8},
            ("P0", "refused"): {"answer_quality_label": "failure_to_answer", "answer_correctness": 0,
                                "answer_completeness": 0, "exact_abstention": 1, "bertscore_f1": .1},
        }
        row = next(row for row in evaluate(manifest, native, {"answer": answers})
                   if row["architecture"] == "P0")
        self.assertEqual(row["question_count"], 3)
        self.assertEqual(row["answer_correctness_n"], 2)
        self.assertEqual(row["answer_correctness_mean"], .5)
        self.assertEqual(row["judge_error_count"], 1)
        self.assertEqual(row["semantic_non_abstained_scored_count"], 1)
        self.assertEqual(row["answer_correctness_lower_bound"], 1 / 3)
        self.assertEqual(row["answer_correctness_upper_bound"], 2 / 3)
        self.assertEqual(row["bertscore_f1_n"], 3)
        answers[("P0", "unscored")]["answer_correctness"] = 0
        with self.assertRaisesRegex(ValueError, "must remain missing"):
            evaluate(manifest, native, {"answer": answers})

    def test_all_judge_errors_have_no_semantic_mean(self) -> None:
        manifest = {"run_id": "r", "question_ids": ["q"]}
        native = {"q": {"question_type": "native", "question_reasoning": None}}
        answers = {("P1", "q"): {"answer_quality_label": "judge_error",
                                  "answer_correctness": "", "answer_completeness": "",
                                  "exact_abstention": "0", "bertscore_f1": .8}}
        row = next(row for row in evaluate(manifest, native, {"answer": answers})
                   if row["architecture"] == "P1")
        self.assertEqual(row["answer_correctness_mean"], "")
        self.assertEqual(row["semantic_scored_count"], 0)
        self.assertEqual(row["answer_correctness_lower_bound"], 0)
        self.assertEqual(row["answer_correctness_upper_bound"], 1)


if __name__ == "__main__":
    unittest.main()
