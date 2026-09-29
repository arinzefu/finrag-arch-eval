from __future__ import annotations

import unittest

from scripts.evaluate_retrieval import score_question


def chunk(doc_name: str, page: int, rank: int) -> dict:
    return {"doc_name": doc_name, "page": page, "rank": rank}


def gold(doc_name: str, page: int) -> dict:
    return {"doc_name": doc_name, "page": page}


class RetrievalEvaluationTests(unittest.TestCase):
    def score(self, chunks: list[dict], refs: list[dict]) -> dict:
        return score_question(
            run_id="test-run", architecture="P2", question_id="q1",
            chunks=chunks, gold_refs=refs,
        )

    def test_same_document_wrong_page_is_not_relevant(self) -> None:
        result = self.score([chunk("filing", 4, 1)], [gold("filing", 5)])
        self.assertEqual(result["hit_at_5"], 0)
        self.assertEqual(result["recall_at_5"], 0.0)
        self.assertEqual(result["mrr_at_5"], 0.0)

    def test_duplicate_chunks_on_one_gold_page_count_once(self) -> None:
        result = self.score(
            [chunk("filing", 1, 1), chunk("filing", 1, 2),
             chunk("filing", 2, 3), chunk("other", 2, 4)],
            [gold("filing", 1), gold("filing", 2)],
        )
        self.assertEqual(result["gold_pages_retrieved_at_5"], 2)
        self.assertEqual(result["retrieved_unique_page_count"], 3)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["mrr_at_5"], 1.0)

    def test_first_gold_at_rank_four_affects_hits_and_mrr(self) -> None:
        result = self.score(
            [chunk("other", i, i) for i in (1, 2, 3)]
            + [chunk("filing", 7, 4), chunk("other", 5, 5)],
            [gold("filing", 7), gold("filing", 8)],
        )
        self.assertEqual(
            (result["hit_at_1"], result["hit_at_3"], result["hit_at_5"]),
            (0, 0, 1),
        )
        self.assertEqual(result["first_gold_rank"], 4)
        self.assertEqual(result["recall_at_5"], 0.5)
        self.assertEqual(result["mrr_at_5"], 0.25)

    def test_missing_gold_and_invalid_rank_fail(self) -> None:
        with self.assertRaises(ValueError):
            self.score([chunk("filing", 1, 1)], [])
        with self.assertRaises(ValueError):
            self.score([chunk("filing", 1, 2)], [gold("filing", 1)])

    def test_empty_retrieval_is_a_measured_miss(self) -> None:
        result = self.score([], [gold("filing", 1)])
        self.assertEqual(result["retrieved_chunk_count"], 0)
        self.assertEqual(result["first_gold_rank"], "")
        self.assertEqual(result["hit_at_5"], 0)
        self.assertEqual(result["recall_at_5"], 0.0)
        self.assertEqual(result["mrr_at_5"], 0.0)


if __name__ == "__main__":
    unittest.main()
