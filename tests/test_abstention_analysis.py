from __future__ import annotations

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_abstentions import (
    analyze,
    is_exact_abstention,
    keyed_csv,
    retrieval_diagnostics,
    validate_support,
)
from scripts.evaluate_retrieval import score_question


def support_row(**updates: object) -> dict:
    row = {
        "run_id": "test-run", "architecture": "P1", "question_id": "q1",
        "abstained": "0", "judge_model": "fixed-judge",
        "judge_protocol": "answer-unit-passage-v4",
        "context_claim_count": "4", "context_supported_count": "3",
        "context_faithfulness": ".75", "context_hallucination": ".25",
        "gold_claim_count": "2", "gold_supported_count": "1",
        "gold_evidence_support": ".5", "gold_evidence_hallucination": ".5",
    }
    row.update(updates)
    return row


def chunk(chunk_id: str, page: int, rank: int = 1) -> dict:
    return {"chunk_id": chunk_id, "doc_name": "filing", "page": page,
            "rank": rank}


class ExactAbstentionTests(unittest.TestCase):
    def test_frozen_exact_phrases_and_normalization(self) -> None:
        for phrase in (
            "Insufficient evidence in the retrieved context.",
            "  INSUFFICIENT EVIDENCE.  ",
            "I don't know", "I do not know.",
            "Cannot determine from the provided information",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_exact_abstention(phrase))

    def test_nonexact_answers_include_mixed_refusals_and_wrong_answers(self) -> None:
        for answer in (
            "0", "The answer is 123.",
            "Revenue rose, but insufficient evidence in the retrieved context.",
            "I cannot find the exact figures; please provide the balance sheet.",
            "Insufficient evidence!",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(is_exact_abstention(answer))


class SupportValidationTests(unittest.TestCase):
    def test_support_and_proxy_use_the_same_scored_units(self) -> None:
        validate_support(support_row(), "P1")
        with self.assertRaises(ValueError):
            validate_support(support_row(context_hallucination=".2"), "P1")
        with self.assertRaises(ValueError):
            validate_support(support_row(context_supported_count="2"), "P1")

    def test_zero_scored_units_require_na_instead_of_zero_support(self) -> None:
        row = support_row(context_claim_count="0", context_supported_count="0",
                          context_faithfulness="", context_hallucination="")
        validate_support(row, "P1")
        with self.assertRaises(ValueError):
            validate_support({**row, "context_faithfulness": "0",
                              "context_hallucination": "1"}, "P1")
        with self.assertRaises(ValueError):
            validate_support(support_row(context_faithfulness="",
                                        context_hallucination=""), "P1")

    def test_counts_cannot_be_negative_or_exceed_the_denominator(self) -> None:
        for changes in (
            {"gold_claim_count": "-2"},
            {"gold_supported_count": "3"},
            {"gold_evidence_support": "nan"},
            {"context_faithfulness": "inf"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_support(support_row(**changes), "P1")

    def test_p0_context_is_inapplicable(self) -> None:
        row = support_row(architecture="P0", context_claim_count="",
                          context_supported_count="", context_faithfulness="",
                          context_hallucination="")
        validate_support(row, "P0")
        with self.assertRaises(ValueError):
            validate_support(support_row(architecture="P0"), "P0")

    def test_exact_abstentions_cannot_have_scored_factual_units(self) -> None:
        with self.assertRaises(ValueError):
            validate_support(support_row(abstained="1"), "P1")
        row = support_row(abstained="1", context_claim_count="0",
                          context_supported_count="0", context_faithfulness="",
                          context_hallucination="", gold_claim_count="0",
                          gold_supported_count="0", gold_evidence_support="",
                          gold_evidence_hallucination="")
        validate_support(row, "P1")


class KeyedCsvTests(unittest.TestCase):
    def write_rows(self, directory: str, rows: list[dict]) -> Path:
        path = Path(directory) / "metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "run_id", "architecture", "question_id"))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_rejects_duplicate_wrong_run_missing_and_extra_rows(self) -> None:
        good = {"run_id": "test-run", "architecture": "P1", "question_id": "q1"}
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, [good])
            self.assertEqual(set(keyed_csv(path, "test-run", ["q1"], ("P1",))),
                             {("P1", "q1")})
            for rows in (
                [good, good],
                [{**good, "run_id": "different-run"}],
                [],
                [good, {**good, "question_id": "q2"}],
            ):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    keyed_csv(self.write_rows(directory, rows), "test-run",
                              ["q1"], ("P1",))


class RetrievalDiagnosticTests(unittest.TestCase):
    def record(self, union: list[dict], pre: list[dict], final: list[dict],
               architecture: str = "P3") -> tuple[dict, dict]:
        record = {
            "architecture": architecture, "question_id": "q1",
            "retrieved_chunks": final,
            "retrieval_diagnostics": {
                "rrf_union": union, "rrf_top_candidates": pre,
                "first_stage_candidates": pre, "final_candidates": final,
                "reranked_candidates": final + [c for c in pre if c not in final],
            },
        }
        metrics = score_question(run_id="test-run", architecture=architecture,
                                 question_id="q1", chunks=final,
                                 gold_refs=[{"doc_name": "filing", "page": 7}])
        return record, metrics

    def diagnose(self, union: list[dict], pre: list[dict], final: list[dict]) -> dict:
        record, metrics = self.record(union, pre, final)
        return retrieval_diagnostics(record, {("filing", 7)}, metrics)

    def test_stages_distinguish_missing_candidates_from_selection_losses(self) -> None:
        other, gold = chunk("other", 2), chunk("gold", 7, 2)
        five_other = [chunk(f"other{i}", i, i) for i in range(1, 6)]
        sixth_gold = chunk("sixth-gold", 7, 6)
        fixtures = (
            ([other], [other], [other], "gold_page_absent_candidate_union"),
            ([other, gold], [other], [other], "gold_page_lost_before_pre_final"),
            (five_other + [sixth_gold], five_other + [sixth_gold], five_other,
             "gold_page_lost_at_final_selection"),
            ([other, gold], [other, gold], [other, gold], "final_gold_page_hit"),
            ([], [], [], "empty_retrieval"),
        )
        for union, pre, final, category in fixtures:
            with self.subTest(category=category):
                self.assertEqual(self.diagnose(union, pre, final)["diagnostic_category"],
                                 category)

    def test_same_page_cannot_hide_a_chunk_outside_the_candidate_pool(self) -> None:
        record, metrics = self.record([chunk("original", 7)],
                                      [chunk("replacement", 7)],
                                      [chunk("replacement", 7)])
        with self.assertRaises(ValueError):
            retrieval_diagnostics(record, {("filing", 7)}, metrics)

    def test_reusing_a_chunk_id_on_another_page_fails(self) -> None:
        record, metrics = self.record([chunk("same", 2)], [chunk("same", 2)],
                                      [chunk("same", 7)])
        with self.assertRaises(ValueError):
            retrieval_diagnostics(record, {("filing", 7)}, metrics)

    def test_saved_final_count_and_hit_must_match_the_actual_context(self) -> None:
        record, metrics = self.record([chunk("gold", 7)], [chunk("gold", 7)],
                                      [chunk("gold", 7)])
        for changes in ({"retrieved_chunk_count": "2"}, {"hit_at_5": "0"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                retrieval_diagnostics(record, {("filing", 7)}, {**metrics, **changes})

    def test_rrf_saved_selection_and_duplicate_candidates_are_validated(self) -> None:
        record, metrics = self.record([chunk("gold", 7)], [chunk("gold", 7)],
                                      [chunk("gold", 7)])
        mismatched = copy.deepcopy(record)
        mismatched["retrieval_diagnostics"]["rrf_top_candidates"] = []
        duplicate = copy.deepcopy(record)
        duplicate["retrieval_diagnostics"]["rrf_union"] *= 2
        for invalid in (mismatched, duplicate):
            with self.subTest(record=invalid), self.assertRaises(ValueError):
                retrieval_diagnostics(invalid, {("filing", 7)}, metrics)

    def test_final_diagnostics_and_reranking_must_describe_actual_selection(self) -> None:
        other, gold = chunk("other", 2), chunk("gold", 7, 2)
        record, metrics = self.record([other, gold], [other, gold], [other, gold])
        wrong_final = copy.deepcopy(record)
        wrong_final["retrieval_diagnostics"]["final_candidates"] = [other]
        wrong_order = copy.deepcopy(record)
        wrong_order["retrieval_diagnostics"]["reranked_candidates"] = [gold, other]
        missing_reranked = copy.deepcopy(record)
        missing_reranked["retrieval_diagnostics"]["reranked_candidates"] = [other]
        for invalid in (wrong_final, wrong_order, missing_reranked):
            with self.subTest(record=invalid), self.assertRaises(ValueError):
                retrieval_diagnostics(invalid, {("filing", 7)}, metrics)

    def test_p2_preserves_rrf_order_when_selecting_final_chunks(self) -> None:
        other, gold = chunk("other", 2), chunk("gold", 7, 2)
        final = [chunk("gold", 7), chunk("other", 2, 2)]
        record, metrics = self.record([other, gold], [other, gold], final,
                                      architecture="P2")
        with self.assertRaises(ValueError):
            retrieval_diagnostics(record, {("filing", 7)}, metrics)


class FrozenAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = analyze()
        cls.tables = cls.result["tables"]

    def test_overlap_matches_the_frozen_145_questions(self) -> None:
        actual = {r["pattern_p1_p2_p3"]: r["question_count"]
                  for r in self.tables["overlap"]}
        self.assertEqual(actual, {"000": 68, "001": 7, "010": 7, "011": 5,
                                  "100": 4, "101": 3, "110": 8, "111": 43})
        partition = [q for r in self.tables["overlap"]
                     for q in json.loads(r["question_ids_json"])]
        self.assertEqual(len(partition), len(set(partition)))
        self.assertEqual(set(partition), set(self.result["manifest"]["question_ids"]))

    def test_pairwise_counts_capture_both_directions_of_disagreement(self) -> None:
        expected = {
            ("P1", "P2"): (51, 7, 12, 75),
            ("P1", "P3"): (46, 12, 12, 75),
            ("P2", "P3"): (48, 15, 10, 72),
        }
        for row in self.tables["pairwise"]:
            key = row["architecture_a"], row["architecture_b"]
            values = tuple(row[k] for k in ("both_abstained", "a_only_abstained",
                                            "b_only_abstained", "neither_abstained"))
            self.assertEqual(values, expected[key])
            self.assertEqual(sum(values), 145)
            self.assertAlmostEqual(row["jaccard_abstentions"],
                                   values[0] / sum(values[:3]))

    def test_evidence_comparison_retains_non_abstentions_and_page_hit_refusals(self) -> None:
        expected = {
            ("P1", 1): (58, 9, 49), ("P1", 0): (87, 49, 38),
            ("P2", 1): (63, 6, 57), ("P2", 0): (82, 40, 42),
            ("P3", 1): (58, 10, 48), ("P3", 0): (87, 48, 39),
        }
        for row in self.tables["evidence_coverage"]:
            self.assertEqual((row["question_count"], row["final_gold_page_hit"],
                              row["no_final_gold_page_hit"]),
                             expected[row["architecture"], row["exact_abstention"]])
            self.assertEqual(row["no_final_gold_page_hit"], sum(row[k] for k in (
                "gold_page_absent_candidate_union", "gold_page_lost_before_pre_final",
                "gold_page_lost_at_final_selection")))
            self.assertLessEqual(row["empty_retrieval"], row["no_final_gold_page_hit"])

    def test_support_denominators_exclude_more_than_exact_abstentions(self) -> None:
        expected = {"P0": (0, 0, 112), "P1": (58, 85, 83),
                    "P2": (63, 80, 81), "P3": (58, 86, 85)}
        for row in self.tables["support_summary"]:
            self.assertEqual((row["exact_abstentions"], row["context_scored_n"],
                              row["gold_scored_n"]), expected[row["architecture"]])
            for source in ("context", "gold"):
                support, proxy = row[f"{source}_support"], row[f"{source}_hallucination_proxy"]
                if support == "":
                    self.assertEqual(proxy, "")
                    self.assertEqual(row[f"{source}_scored_n"], 0)
                else:
                    self.assertAlmostEqual(support + proxy, 1.0)
        p0 = next(r for r in self.tables["support_summary"] if r["architecture"] == "P0")
        self.assertEqual(p0["gold_unscored_non_abstentions"], 33)

    def test_question_details_preserve_partial_refusals_without_relabeling(self) -> None:
        row = next(r for r in self.tables["question_details"]
                   if r["architecture"] == "P1"
                   and r["question_id"] == "financebench_id_00517")
        self.assertEqual(row["exact_abstention"], 0)
        self.assertIn("Insufficient evidence in the retrieved context.", row["generated_answer"])
        self.assertEqual(row["context_score_status"], "no_scored_factual_units")
        self.assertEqual(row["context_support"], "")


if __name__ == "__main__":
    unittest.main()
