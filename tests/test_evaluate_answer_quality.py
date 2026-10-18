from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts import answer_semantics as semantic
from scripts import evaluate_answer_quality as quality
from scripts import evaluate_answers as legacy
from scripts.evaluation_common import ARCHITECTURES, write_csv


def semantic_result(statuses: tuple[str, ...], label: str, *, attempted: bool = True) -> dict:
    reference = {"essential_units": [
        {"unit_id": f"U{i}", "requirement": f"Essential answer item {i}."}
        for i in range(1, len(statuses) + 1)], "rationale": "Fixed question rubric."}
    return semantic.parse_judgment({
        "correctness_label": label, "attempted_answer": attempted,
        "has_contradiction": False, "unit_assessments": [
            {"unit_id": f"U{i}", "status": status, "rationale": "Compared with required answer."}
            for i, status in enumerate(statuses, 1)], "rationale": "Compared with the gold answer.",
    }, reference)


def fixtures() -> tuple[dict, dict, dict, dict]:
    manifest = {"run_id": "frozen-test-run"}
    runs, judgments, bert_rows = {}, {}, []
    labels = [
        semantic_result(("answered", "answered"), "correct"),
        semantic_result(("answered", "missing"), "partially_correct"),
        semantic_result(("incorrect", "missing"), "incorrect"),
        semantic_result(("missing", "missing"), "failure_to_answer", attempted=False),
    ]
    for arch in ARCHITECTURES:
        runs[arch] = []
        for index, label in enumerate(labels, 1):
            qid = f"q{index}"
            exact = arch == "P3" or index == 4 and arch != "P0"
            generated = ("Insufficient evidence in the retrieved context." if exact
                         else "Data Center" if index == 1
                         else "First requested item" if index == 2
                         else "A wrong answer" if index == 3
                         else "You would calculate this using financial statements.")
            raw = {"question_id": qid, "question": "Identify the requested answer.",
                   "gold_answer": "Data Center", "generated_answer": generated}
            runs[arch].append(raw)
            judgments[arch, qid] = {"judgment": labels[3] if exact else label}
            bert_rows.append({"architecture": arch, "question_id": qid,
                              "bertscore_precision": index / 10,
                              "bertscore_recall": index / 10,
                              "bertscore_f1": index / 10,
                              "bertscore_candidate_token_count": 8,
                              "bertscore_reference_token_count": 4,
                              "bertscore_truncated": False})
    return manifest, runs, judgments, {"rows": bert_rows}


class QualityAssemblyTests(unittest.TestCase):
    def test_aggregate_correctness_uses_all_questions_and_non_abstention_denominators(self) -> None:
        manifest, runs, judgments, bert = fixtures()
        with patch.object(quality, "validate_legacy") as validate:
            rows, summaries = quality.assemble(manifest, runs, judgments, bert)
        validate.assert_called_once()
        self.assertEqual(len(rows), 16)
        p1 = next(row for row in summaries if row["architecture"] == "P1")
        self.assertEqual((p1["question_count"], p1["semantic_scored_count"],
                          p1["bertscore_scored_count"]), (4, 4, 4))
        self.assertEqual((p1["correct_count"], p1["partially_correct_count"], p1["incorrect_count"],
                          p1["failure_to_answer_count"]), (1, 1, 1, 1))
        self.assertEqual((p1["exact_abstention_count"], p1["non_abstained_count"]), (1, 3))
        self.assertEqual(p1["answer_correctness"], .25)
        self.assertAlmostEqual(p1["answer_correctness_non_abstained"], 1 / 3)
        self.assertEqual(p1["answer_completeness"], .375)
        self.assertEqual(p1["answer_completeness_non_abstained"], .5)
        self.assertAlmostEqual(p1["bertscore_f1"], .25)
        self.assertAlmostEqual(p1["bertscore_f1_non_abstained"], .2)

    def test_broader_failure_remains_in_non_exact_abstention_denominator(self) -> None:
        manifest, runs, judgments, bert = fixtures()
        with patch.object(quality, "validate_legacy"):
            rows, summaries = quality.assemble(manifest, runs, judgments, bert)
        p0 = next(row for row in summaries if row["architecture"] == "P0")
        procedural = next(row for row in rows if (row["architecture"], row["question_id"]) == ("P0", "q4"))
        self.assertEqual(procedural["answer_quality_label"], "failure_to_answer")
        self.assertEqual(procedural["exact_abstention"], 0)
        self.assertEqual((p0["failure_to_answer_count"], p0["exact_abstention_count"]), (1, 0))
        self.assertEqual(p0["non_abstained_count"], 4)
        self.assertEqual(p0["answer_correctness_non_abstained"], .25)

    def test_all_exact_abstentions_produce_zero_overall_and_undefined_conditional_scores(self) -> None:
        manifest, runs, judgments, bert = fixtures()
        with patch.object(quality, "validate_legacy"):
            rows, summaries = quality.assemble(manifest, runs, judgments, bert)
        p3 = next(row for row in summaries if row["architecture"] == "P3")
        self.assertEqual((p3["failure_to_answer_count"], p3["exact_abstention_count"],
                          p3["non_abstained_count"]), (4, 4, 0))
        self.assertEqual((p3["answer_correctness"], p3["answer_completeness"]), (0, 0))
        for metric in ("answer_correctness", "answer_completeness", "bertscore_f1"):
            self.assertEqual(p3[f"{metric}_non_abstained"], "")
        self.assertTrue(all(row["exact_abstention"] for row in rows if row["architecture"] == "P3"))

    def test_existing_metrics_are_reproduced_without_quality_field_changes(self) -> None:
        manifest, runs, judgments, bert = fixtures()
        old_rows, old_summaries = legacy.evaluate(manifest, runs)
        with patch.object(quality, "validate_legacy"):
            rows, summaries = quality.assemble(manifest, runs, judgments, bert)
        for before, after in zip(old_rows, rows):
            self.assertEqual({k: after[k] for k in legacy.FIELDS}, before)
        for before, after in zip(old_summaries, summaries):
            for key in legacy.SUMMARY_FIELDS:
                if key != "table_title":
                    self.assertEqual(after[key], before[key])

    def test_publish_assembly_requires_complete_judgment_and_bertscore_coverage(self) -> None:
        for missing_kind in ("semantic", "bertscore"):
            manifest, runs, judgments, bert = fixtures()
            if missing_kind == "semantic":
                judgments.pop(("P1", "q2"))
            else:
                bert["rows"].pop()
            with self.subTest(missing_kind=missing_kind), patch.object(quality, "validate_legacy"), self.assertRaises(ValueError):
                quality.assemble(manifest, runs, judgments, bert)

    def test_invalid_judge_is_missing_not_silently_counted_as_incorrect(self) -> None:
        manifest, runs, judgments, bert = fixtures()
        judgments["P1", "q1"] = {"judgment": None, "judge_error": "Three invalid assessments"}
        with patch.object(quality, "validate_legacy"):
            rows, summaries = quality.assemble(manifest, runs, judgments, bert)
        p1 = next(row for row in summaries if row["architecture"] == "P1")
        invalid = next(row for row in rows if (row["architecture"], row["question_id"]) == ("P1", "q1"))
        self.assertEqual((invalid["answer_quality_label"], invalid["answer_correctness"],
                          invalid["answer_completeness"]), ("judge_error", "", ""))
        self.assertEqual((p1["semantic_scored_count"], p1["judge_error_count"],
                          p1["semantic_non_abstained_scored_count"]), (3, 1, 2))
        self.assertEqual((p1["correct_count"], p1["incorrect_count"]), (0, 1))
        self.assertEqual((p1["answer_correctness_lower_bound"], p1["answer_correctness_upper_bound"]), (0, .25))
        self.assertAlmostEqual(p1["answer_completeness"], .5 / 3)
        self.assertEqual(p1["answer_completeness_non_abstained"], .25)
        self.assertAlmostEqual(p1["bertscore_f1"], .25)
        self.assertEqual(p1["bertscore_scored_count"], 4)


class LegacyProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest, self.runs, _, _ = fixtures()
        self.rows, _ = legacy.evaluate(self.manifest, self.runs)

    def test_unchanged_legacy_columns_pass_even_when_csv_has_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            saved = [{**row, "answer_correctness": .5} for row in self.rows]
            write_csv(path, saved, legacy.FIELDS + ["answer_correctness"])
            quality.validate_legacy(self.rows, path)

    def test_changed_published_numeric_or_overlap_scores_are_rejected(self) -> None:
        for field in ("token_f1", "exact_match", "gold_numeric_count"):
            saved = [dict(row) for row in self.rows]
            saved[0][field] += .1
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "metrics.csv"
                write_csv(path, saved, legacy.FIELDS)
                with self.assertRaisesRegex(ValueError, "Published legacy score changed"):
                    quality.validate_legacy(self.rows, path)

    def test_missing_and_duplicate_legacy_keys_are_rejected(self) -> None:
        for saved in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.subTest(count=len(saved)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "metrics.csv"
                write_csv(path, saved, legacy.FIELDS)
                with self.assertRaisesRegex(ValueError, "coverage"):
                    quality.validate_legacy(self.rows, path)


class CheckpointTests(unittest.TestCase):
    def test_invalid_responses_exhaust_budget_without_imputing_a_judgment(self) -> None:
        response = SimpleNamespace(status="completed", output_text="{}", model="gpt-4o-mini-test",
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=2))
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        def reject(payload):
            raise ValueError("Conflicting label")
        with tempfile.TemporaryDirectory() as directory, patch.object(quality, "INVALID", Path(directory)/"invalid.jsonl"):
            with self.assertRaises(quality.JudgeValidationError) as raised:
                asyncio.run(quality.structured_call(client, system="fixed rule", prompt="fixed evidence",
                            output_format={}, parser=reject, identity={"kind": "answer", "question_id": "q1"}))
            self.assertEqual(raised.exception.usage["parse_attempts"], 3)
            self.assertEqual(len(quality.load_jsonl(quality.INVALID)), 3)
        self.assertEqual(create.await_count, 3)
        self.assertIn("Do not invent", create.call_args.kwargs["input"])

    def test_matching_checkpoint_loads_and_missing_checkpoint_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            self.assertEqual(quality.load_records(path, ("question_id",), {"protocol": "v1"}), {})
            quality.append_json(path, {"question_id": "q1", "protocol": "v1", "result": 1})
            result = quality.load_records(path, ("question_id",), {"protocol": "v1"})
            self.assertEqual(result["q1",]["result"], 1)

    def test_duplicate_identity_and_protocol_or_source_changes_are_rejected(self) -> None:
        common = {"protocol": "v1", "model": "fixed-judge", "request_source": "hash1"}
        valid = {"question_id": "q1", **common}
        cases = [[valid, valid]]
        for key in common:
            cases.append([{**valid, key: "changed"}])
        for records in cases:
            with self.subTest(records=records), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "duplicate or protocol/source mismatch"):
                    quality.load_records(path, ("question_id",), common)


if __name__ == "__main__":
    unittest.main()
