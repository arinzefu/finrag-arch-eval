from __future__ import annotations

import unittest

from scripts.analyze_tradeoffs import dominates, evaluate


class Stage21Tests(unittest.TestCase):
    def test_pareto_rule_including_equal_points(self) -> None:
        self.assertTrue(dominates(.8, 100, .7, 100))
        self.assertTrue(dominates(.8, 100, .8, 110))
        self.assertFalse(dominates(.8, 100, .8, 100))
        self.assertFalse(dominates(.8, 110, .7, 100))

    def test_coverage_and_pairwise_na(self) -> None:
        ids = ["q1", "q2"]
        answer, retrieval, faithfulness, latency = {}, {}, {}, {}
        for arch in ("P0", "P1", "P2", "P3"):
            for qid in ids:
                answer[(arch, qid)] = {"token_f1": ".5", "numeric_accuracy": "" if qid == "q2" else ".5"}
                faithfulness[(arch, qid)] = {
                    "abstained": "1" if arch == "P2" and qid == "q2" else "0",
                    "context_faithfulness": "" if arch == "P0" or arch == "P2" and qid == "q2" else ".8",
                    "gold_evidence_support": "" if arch == "P2" and qid == "q2" else ".6",
                }
                latency[(arch, qid)] = {"total_latency_ms": str({"P0": 90, "P1": 100, "P2": 200, "P3": 300}[arch])}
                if arch != "P0":
                    retrieval[(arch, qid)] = {"hit_at_5": "1", "recall_at_5": ".5", "mrr_at_5": ".5"}
        summary, frontiers, pairs = evaluate(
            {"run_id": "r", "question_ids": ids},
            {"answer": answer, "retrieval": retrieval, "faithfulness": faithfulness, "latency": latency})
        p0, _, p2, _ = summary
        self.assertEqual(p0["context_faithfulness_mean"], "")
        self.assertEqual(p0["pareto_context_faithfulness"], "")
        self.assertEqual(p2["context_faithfulness_n"], 1)
        self.assertEqual(p2["exact_abstention_rate"], .5)
        self.assertFalse(any(r["architecture"] == "P0" and r["objective"] == "recall_at_5"
                             for r in frontiers))
        p1_p2_context = next(r for r in pairs if r["architecture_a"] == "P1"
                             and r["architecture_b"] == "P2"
                             and r["metric"] == "context_faithfulness")
        self.assertEqual((p1_p2_context["pair_count"], p1_p2_context["excluded_pair_count"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
