from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.aggregate_latency import evaluate as latency_evaluate
from scripts.analyze_question_types import evaluate as category_evaluate
from scripts.evaluate_answers import numeric_entities, score_answer
from scripts.evaluate_faithfulness import (PROTOCOL, PROMPT_VERSION, answer_units, assemble,
                                           financial_spans, is_abstention, judge, load_journal,
                                           parse_judgment,
                                           score_record)


def assessment(unit_id: str, factual: bool, supported: bool, *, support_type: str = "direct",
               ids: list[str] | None = None, derivation: str = "") -> dict:
    return {"unit_id": unit_id, "factual": factual, "supported": supported,
            "support_type": support_type,
            "evidence_ids": ids or [], "derivation": derivation}


def passages(text: str) -> list[dict[str, str]]:
    return [{"evidence_id": "E1", "text": text}]


class Stage16Tests(unittest.TestCase):
    def test_magnitude_and_question_units(self) -> None:
        self.assertEqual(numeric_entities("$1.5 billion"), numeric_entities("$1,500 million"))
        question = "What is the amount in USD millions?"
        self.assertEqual(numeric_entities("$1577.00", question),
                         numeric_entities("$1,577 million", question))

    def test_numeric_multiset_and_no_gold_numbers(self) -> None:
        row = score_answer("r", "P1", "q", "", "Down 0.9%", "Down 0.9% and 3.2%")
        self.assertEqual(row["numeric_accuracy"], 1)
        self.assertEqual(row["numeric_precision"], 0.5)
        self.assertEqual(row["numeric_f1"], 2 / 3)
        no_numeric = score_answer("r", "P0", "q", "", "Consumer", "Consumer")
        self.assertEqual(no_numeric["numeric_accuracy"], "")

    def test_year_is_not_financial_value(self) -> None:
        self.assertFalse(numeric_entities("FY2022 compared with 2021"))

    def test_company_3m_is_not_three_million(self) -> None:
        self.assertFalse(numeric_entities("3M's FY2022 sales"))
        self.assertEqual(numeric_entities("3M reported $3 million"),
                         numeric_entities("$3 million"))


class Stage17Tests(unittest.TestCase):
    def test_judge_schema_and_unsupported_claims(self) -> None:
        answer = "Revenue rose. Margin was 8%"
        payload = {"assessments": [assessment("U1", True, False, support_type="unsupported"),
                                   assessment("U2", True, True, ids=["E1"])]}
        judgment = parse_judgment(json.dumps(payload), answer_units(answer), passages("8%"))
        count, supported, score, unsupported, overrides = score_record({"judgment": judgment})
        self.assertEqual((count, supported, score, overrides), (2, 1, .5, 0))
        self.assertEqual(json.loads(unsupported), ["Revenue rose."])
        with self.assertRaises(ValueError):
            parse_judgment('{"assessments":[{"unit_id":"U1","supported":"yes"}]}',
                           answer_units("x"), passages("x"))

    def test_judge_cannot_change_answer_number(self) -> None:
        answer = "FY2018 Revenue: $7,505 million"
        payload = {"assessments": [assessment("U1", True, True,
                                               ids=["E1"])]}
        result = parse_judgment(json.dumps(payload), answer_units(answer),
                                passages("$7,500 million"))
        self.assertEqual(result["claims"][0]["claim"], answer)
        self.assertFalse(result["claims"][0]["supported"])

    def test_absent_revenue_cannot_be_supported_by_unrelated_quote(self) -> None:
        answer = "FY2017 Revenue: $3,606 million"
        payload = {"assessments": [assessment("U1", True, True,
                                               ids=["E1"])]}
        result = parse_judgment(json.dumps(payload), answer_units(answer),
                                passages("Capex: $155 million"))
        self.assertFalse(result["claims"][0]["supported"])
        self.assertEqual(result["claims"][0]["guard_reason"],
                         "financial_number_absent_from_passage")

    def test_table_value_can_be_supported_without_verbatim_answer_wording(self) -> None:
        payload = {"assessments": [assessment("U1", True, True, ids=["E1"])]}
        result = parse_judgment(json.dumps(payload), answer_units("FY2017 Capex: $155 million"),
                                passages("Capital expenditures (116), (131), (155)"))
        self.assertTrue(result["claims"][0]["supported"])

    def test_missing_arithmetic_operand_is_unsupported(self) -> None:
        payload = {"assessments": [assessment("U1", True, True, support_type="arithmetic",
                                               ids=["E1"],
                                               derivation="155 / 3606 = 4.3%")]} 
        result = parse_judgment(json.dumps(payload), answer_units("Capex margin is 4.3%"),
                                passages("Capex 155"))
        self.assertFalse(result["claims"][0]["supported"])
        self.assertEqual(result["claims"][0]["guard_reason"], "arithmetic_premise_not_cited")

    def test_missing_unit_is_counted_unsupported(self) -> None:
        payload = {"assessments": [assessment("U1", True, True,
                                               ids=["E1"])]}
        result = parse_judgment(json.dumps(payload), answer_units(
                                "Revenue was $7,505 million; capex was $131 million"),
                                passages("$7,505 million"))
        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(result["claims"][1]["guard_reason"], "missing_unit_assessment")

    def test_financial_unit_cannot_be_marked_nonfactual(self) -> None:
        payload = {"assessments": [assessment("U1", False, False,
                                               support_type="unsupported")]}
        result = parse_judgment(json.dumps(payload), answer_units("Revenue was $7,505 million"), [])
        self.assertEqual(result["claims"][0]["guard_reason"],
                         "financial_unit_marked_nonfactual")

    def test_company_3m_is_not_a_financial_amount(self) -> None:
        self.assertEqual(financial_spans("3M reported $155 million"), ["$155 million"])

    def test_p0_context_is_not_applicable(self) -> None:
        runs = {a: [] for a in ("P0", "P1", "P2", "P3")}
        runs["P0"] = [{"question_id": "q", "generated_answer": "Revenue rose"}]
        records = {("P0", "q", "gold"): {"judgment": {"claims": [
            {"claim": "Revenue rose", "supported": False}]}}}
        rows, _ = assemble({"run_id": "r"}, runs, records, "gpt-4o-mini")
        self.assertEqual(rows[0]["context_faithfulness"], "")
        self.assertEqual(rows[0]["gold_evidence_hallucination"], 1)
        self.assertTrue(is_abstention("Insufficient evidence in the retrieved context."))

    def test_malformed_judge_json_retries_same_request_and_logs_failure(self) -> None:
        class FakeLLM:
            model = "gpt-4o-mini"
            max_output_tokens = 2400

            def __init__(self):
                self.calls = []
                self.client = SimpleNamespace(responses=self)

            def create(self, **kwargs):
                self.calls.append(kwargs)
                content = ('{"assessments":[{"unit_id":"U1","supported":true}'
                           if len(self.calls) == 1 else json.dumps({"assessments": [
                               assessment("U1", True, True, ids=["E1"])]}))
                return SimpleNamespace(output_text=content, status="completed",
                                       usage=SimpleNamespace(input_tokens=15, output_tokens=42))

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "invalid.jsonl"
            llm = FakeLLM()
            result = judge("P1", {"question_id": "q", "question": "Revenue?",
                                   "generated_answer": "Revenue rose"}, "context",
                           passages("Revenue rose"),
                           llm, "manifest-hash", "evidence-hash", invalid_log=log)
            self.assertEqual(result["parse_attempts"], 2)
            self.assertEqual(len(llm.calls), 2)
            self.assertEqual(llm.calls[0], llm.calls[1])
            self.assertEqual(llm.calls[0]["text"]["format"]["type"], "json_schema")
            logged = json.loads(log.read_text(encoding="utf-8").strip())
            self.assertEqual(logged["question_id"], "q")
            self.assertEqual(logged["attempt"], 1)

    def test_resume_rejects_a_different_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            record = {"architecture": "P0", "question_id": "q", "kind": "gold",
                      "model": "gpt-4o-mini", "protocol": PROTOCOL,
                      "run_manifest_sha256": "manifest", "evidence_mapping_sha256": "evidence",
                      "prompt_sha256": PROMPT_VERSION, "judgment": {"claims": []}}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(len(load_journal(path, "gpt-4o-mini", "manifest", "evidence")), 1)
            record["protocol"] = "claim-evidence-v1"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_journal(path, "gpt-4o-mini", "manifest", "evidence")


class Stage18And19Tests(unittest.TestCase):
    def test_latency_not_applicable_and_no_invented_components(self) -> None:
        manifest = {"run_id": "r"}
        raw = {"question_id": "q", "retrieval_latency_ms": 0,
               "reranking_latency_ms": 0, "generation_latency_ms": 12,
               "total_latency_ms": 12}
        runs = {a: [raw] for a in ("P0", "P1", "P2", "P3")}
        rows, summaries = latency_evaluate(manifest, runs)
        self.assertEqual(rows[0]["retrieval_latency_ms"], "")
        self.assertEqual(rows[1]["reranking_latency_ms"], "")
        self.assertEqual(rows[3]["reranking_latency_ms"], 0)
        self.assertFalse(any(r["component"] == "bm25" for r in summaries))

    def test_categories_use_native_labels_and_na_denominators(self) -> None:
        manifest = {"run_id": "r", "question_ids": ["q"]}
        native = {"q": {"question_type": "metrics", "question_reasoning": "extraction"}}
        tables = {"answer": {("P0", "q"): {"token_f1": "0.5", "numeric_accuracy": ""}}}
        rows = category_evaluate(manifest, native, tables)
        p0_type = next(r for r in rows if r["grouping"] == "question_type"
                       and r["architecture"] == "P0")
        self.assertEqual(p0_type["token_f1_mean"], .5)
        self.assertEqual(p0_type["numeric_accuracy_n"], 0)
        self.assertEqual(p0_type["numeric_accuracy_mean"], "")

    def test_missing_native_reasoning_label_is_not_invented(self) -> None:
        manifest = {"run_id": "r", "question_ids": ["q"]}
        native = {"q": {"question_type": "novel-generated", "question_reasoning": None}}
        rows = category_evaluate(manifest, native, {})
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["grouping"] for r in rows}, {"question_type"})


if __name__ == "__main__":
    unittest.main()
