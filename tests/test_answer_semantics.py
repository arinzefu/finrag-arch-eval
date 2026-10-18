from __future__ import annotations

import json
import unittest

from scripts.answer_semantics import (exact_abstention_result, is_exact_abstention,
                                      judgment_prompt, parse_judgment, parse_reference,
                                      reference_prompt)


def reference(count: int = 2) -> dict:
    return {"essential_units": [
        {"unit_id": f"U{i}", "requirement": f"Required member {i}."}
        for i in range(1, count + 1)], "rationale": "Each requested list member is essential."}


def judgment(statuses: tuple[str, ...], label: str, *, attempted: bool = True,
             contradiction: bool = False) -> dict:
    return {"correctness_label": label, "attempted_answer": attempted,
            "has_contradiction": contradiction, "unit_assessments": [
                {"unit_id": f"U{i}", "status": status, "rationale": "Compared with reference."}
                for i, status in enumerate(statuses, 1)],
            "rationale": "Assessment follows the supplied reference."}


class AnswerSemanticsTests(unittest.TestCase):
    def test_complete_paraphrase_is_correct(self) -> None:
        result = parse_judgment(json.dumps(judgment(("answered", "answered"), "correct")), reference())
        self.assertEqual((result["semantic_correctness"], result["completeness"]), (1, 1))

    def test_partial_requested_list_has_recall_like_completeness(self) -> None:
        result = parse_judgment(judgment(("answered", "missing"), "partially_correct"), reference())
        self.assertEqual(result["semantic_correctness"], 0)
        self.assertEqual(result["completeness"], .5)

    def test_wrong_requested_unit_is_incorrect_even_with_correct_progress(self) -> None:
        result = parse_judgment(judgment(("answered", "incorrect"), "incorrect"), reference())
        self.assertEqual(result["correctness_label"], "incorrect")
        self.assertEqual(result["completeness"], .5)

    def test_correct_final_answer_with_wrong_reasoning_is_incorrect(self) -> None:
        result = parse_judgment(judgment(("answered", "answered"), "incorrect", contradiction=True),
                                reference())
        self.assertEqual(result["semantic_correctness"], 0)
        self.assertEqual(result["completeness"], 1)

    def test_non_exact_procedural_nonanswer_can_be_failure(self) -> None:
        result = parse_judgment(judgment(("missing", "missing"), "failure_to_answer", attempted=False),
                                reference())
        self.assertEqual(result["completeness"], 0)

    def test_single_required_conclusion_is_complete_without_optional_reference_narrative(self) -> None:
        minimal_reference = {"essential_units": [
            {"unit_id": "U1", "requirement": "Yes: the company's dividend trend is stable/increasing."}],
            "rationale": "The question asks for a conclusion; the count of years is optional supporting context."}
        result = parse_judgment(judgment(("answered",), "correct"), minimal_reference)
        self.assertEqual((result["essential_unit_count"], result["matched_essential_unit_count"],
                          result["semantic_correctness"], result["completeness"]), (1, 1, 1, 1))

    def test_generic_method_gets_no_credit_under_company_specific_conclusion_rubric(self) -> None:
        minimal_reference = {"essential_units": [
            {"unit_id": "U1", "requirement": "No: the company's quick-ratio-based liquidity is below healthy."}],
            "rationale": "Neither a generic formula nor a generic threshold answers the company-specific question."}
        result = parse_judgment(judgment(("missing",), "failure_to_answer", attempted=False),
                                minimal_reference)
        self.assertEqual((result["matched_essential_unit_count"], result["completeness"]), (0, 0))

    def test_substantive_wrong_attempt_without_matched_units_is_incorrect(self) -> None:
        result = parse_judgment(judgment(("missing", "incorrect"), "incorrect"), reference())
        self.assertEqual(result["completeness"], 0)

    def test_exact_abstention_detector_matches_frozen_protocol(self) -> None:
        for answer in ("Insufficient evidence in the retrieved context.", "insufficient evidence",
                       " I DON'T KNOW... ", "I do not know", "Cannot determine from the provided information."):
            self.assertTrue(is_exact_abstention(answer))
        self.assertFalse(is_exact_abstention("I don't know, but Data Center grew 64%."))
        self.assertFalse(is_exact_abstention("Here is how to calculate it. Insufficient evidence."))
        result = exact_abstention_result(reference())
        self.assertEqual((result["correctness_label"], result["semantic_correctness"],
                          result["completeness"]), ("failure_to_answer", 0, 0))

    def test_missing_duplicate_unknown_or_invalid_units_are_rejected(self) -> None:
        base = judgment(("answered", "answered"), "correct")
        variants = []
        missing = json.loads(json.dumps(base))
        missing["unit_assessments"].pop()
        variants.append(missing)
        for value in ("U1", "U3"):
            wrong_id = json.loads(json.dumps(base))
            wrong_id["unit_assessments"][1]["unit_id"] = value
            variants.append(wrong_id)
        wrong_status = json.loads(json.dumps(base))
        wrong_status["unit_assessments"][1]["status"] = "mostly_correct"
        variants.append(wrong_status)
        for item in variants:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_judgment(item, reference())

    def test_labels_must_agree_with_statuses(self) -> None:
        for statuses, label, attempted, contradiction in (
            (("answered", "missing"), "correct", True, False),
            (("answered", "incorrect"), "partially_correct", True, False),
            (("answered", "answered"), "correct", True, True),
            (("missing", "missing"), "failure_to_answer", True, False),
            (("answered", "missing"), "failure_to_answer", False, False),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_judgment(judgment(statuses, label, attempted=attempted,
                                        contradiction=contradiction), reference())

    def test_boolean_string_and_missing_rationale_are_rejected(self) -> None:
        base = judgment(("answered", "answered"), "correct")
        base["attempted_answer"] = "true"
        with self.assertRaises(ValueError):
            parse_judgment(base, reference())
        base = judgment(("answered", "answered"), "correct")
        base["rationale"] = " "
        with self.assertRaises(ValueError):
            parse_judgment(base, reference())

    def test_reference_must_have_nonempty_unique_consecutive_units(self) -> None:
        variants = [reference(0)]
        duplicate = reference()
        duplicate["essential_units"][1]["requirement"] = "required  member 1."
        variants.append(duplicate)
        wrong_id = reference()
        wrong_id["essential_units"][1]["unit_id"] = "U4"
        variants.append(wrong_id)
        for item in variants:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_reference(item)

    def test_prompts_are_architecture_blind_and_reference_independent_of_generated_answer(self) -> None:
        raw = {"question": "Which segment grew fastest?", "gold_answer": "Data Center",
               "generated_answer": "Data Center, up 64%.", "architecture": "P3",
               "retrieved_chunks": [{"text": "untrusted retrieval sentinel"}]}
        mapping = {"mapping_complete": True, "evidence_items": [
            {"evidence_doc_name": "AMD_2022_10K", "evidence_page_num": 12,
             "evidence_text": "Data Center revenue grew 64%."}]}
        before = reference_prompt(raw, mapping)
        raw["generated_answer"] = "A different answer."
        self.assertEqual(before, reference_prompt(raw, mapping))
        for prompt in (before, judgment_prompt(raw, mapping, reference(1))):
            self.assertNotIn("P3", prompt)
            self.assertNotIn("untrusted retrieval sentinel", prompt)
            self.assertIn("annotated_evidence", prompt)
        self.assertNotIn("generated_answer", before)

    def test_incomplete_evidence_fails_instead_of_silently_dropping_evidence(self) -> None:
        with self.assertRaises(ValueError):
            reference_prompt({"question": "Q?", "gold_answer": "A"},
                             {"mapping_complete": False, "evidence_items": []})

    def test_assessments_are_normalized_to_fixed_reference_order(self) -> None:
        payload = judgment(("answered", "missing"), "partially_correct")
        payload["unit_assessments"].reverse()
        result = parse_judgment(payload, reference())
        self.assertEqual([unit["unit_id"] for unit in result["unit_assessments"]], ["U1", "U2"])


if __name__ == "__main__":
    unittest.main()
