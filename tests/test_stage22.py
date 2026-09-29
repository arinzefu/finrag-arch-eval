from __future__ import annotations

import io
import unittest

import matplotlib.pyplot as plt

from scripts.generate_final_outputs import (ARCHITECTURES, FIGURE_TITLES, TABLE_TITLES,
                                            architecture_configuration, keyed_architecture,
                                            output_paths, plot_architecture, plot_tradeoff)


class Stage22Tests(unittest.TestCase):
    def test_configuration_uses_frozen_settings(self) -> None:
        manifest = {
            "run_id": "r", "model": "same-llm", "max_output_tokens": 256,
            "temperature": None, "settings": {
                "p1_dense_k": 5, "p2_dense_k": 10, "p2_sparse_k": 10,
                "p2_fusion_k": 5, "p3_dense_k": 20, "p3_sparse_k": 20,
                "p3_fusion_k": 20, "final_context_k": 5, "rrf_k": 60,
                "reranker_max_length": 512, "reranker_window_overlap": 64,
                "reranker_batch_size": 32,
            },
        }
        rows = architecture_configuration(manifest)
        self.assertEqual([r["architecture"] for r in rows], list(ARCHITECTURES))
        self.assertEqual([r["llm_model"] for r in rows], ["same-llm"] * 4)
        self.assertEqual(rows[0]["final_context_k"], 0)
        self.assertEqual(rows[1]["dense_candidate_k"], 5)
        self.assertEqual(rows[2]["fusion_candidate_k"], 5)
        self.assertEqual(rows[3]["fusion_candidate_k"], 20)
        self.assertEqual(rows[3]["reranker_max_length"], 512)
        self.assertEqual(rows[3]["temperature_policy"], "API default (unset)")

    def test_duplicate_or_wrong_run_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            keyed_architecture([{"architecture": "P0", "run_id": "r"}] * 2,
                               ("P0", "P1"), "r", "test")
        with self.assertRaises(ValueError):
            keyed_architecture([{"architecture": "P0", "run_id": "other"}],
                               ("P0",), "r", "test")

    def test_output_inventory_and_in_memory_figures(self) -> None:
        paths = output_paths()
        self.assertEqual(len(paths), 2 * len(FIGURE_TITLES) + len(TABLE_TITLES))
        self.assertEqual(len(set(paths.values())), len(paths))
        manifest = {
            "run_id": "r", "model": "same-llm", "max_output_tokens": 256,
            "temperature": None, "settings": {
                "p1_dense_k": 5, "p2_dense_k": 10, "p2_sparse_k": 10,
                "p2_fusion_k": 5, "p3_dense_k": 20, "p3_sparse_k": 20,
                "p3_fusion_k": 20, "final_context_k": 5, "rrf_k": 60,
                "reranker_max_length": 512, "reranker_window_overlap": 64,
                "reranker_batch_size": 32,
            },
        }
        config = architecture_configuration(manifest)
        fig = plot_architecture(manifest, config)
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            self.assertGreater(len(buffer.getvalue()), 5_000)
        finally:
            plt.close(fig)
        data = {"frontiers": [
            {"objective": "recall_at_5", "architecture": "P1", "quality_mean": ".4",
             "quality_scored_n": "2", "mean_total_latency_ms": "100",
             "pareto_nondominated": "1"},
            {"objective": "recall_at_5", "architecture": "P2", "quality_mean": ".3",
             "quality_scored_n": "2", "mean_total_latency_ms": "200",
             "pareto_nondominated": "0"},
        ]}
        fig = plot_tradeoff(data, "recall_at_5", "figure06b_retrieval_latency",
                            "Recall@5", .5)
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            self.assertGreater(len(buffer.getvalue()), 5_000)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
