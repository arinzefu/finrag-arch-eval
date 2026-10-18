from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from scripts.generate_final_outputs import (ARCHITECTURES, FIGURE_TITLES, TABLE_TITLES,
                                            architecture_configuration, keyed_architecture,
                                            output_paths, plot_answers, plot_architecture,
                                            plot_tradeoff, publish)


class Stage22Tests(unittest.TestCase):
    def test_missing_semantic_scores_are_displayed_as_na_with_valid_n(self) -> None:
        answer = {"answer_correctness": .4, "answer_completeness": .6, "bertscore_f1": .8,
                  "token_f1": .2, "exact_match": .01, "numeric_accuracy": .15,
                  "semantic_scored_count": 145}
        answers = {arch: answer.copy() for arch in ARCHITECTURES}
        answers["P0"].update(answer_correctness="", answer_completeness="", semantic_scored_count=0)
        figure = plot_answers({"answers": answers})
        try:
            self.assertTrue(any(text.get_text() == "NA" for text in figure.axes[0].texts))
            self.assertEqual(figure.axes[0].get_xticklabels()[0].get_text(), "P0\nn=0")
            self.assertTrue(any("exclude judge errors" in text.get_text() for text in figure.texts))
        finally:
            plt.close(figure)

    def test_extended_answer_figure_keeps_distinct_metric_panels(self) -> None:
        answer = {"answer_correctness": .4, "answer_completeness": .6, "bertscore_f1": .8,
                  "token_f1": .2, "exact_match": .01, "numeric_accuracy": .15}
        figure = plot_answers({"answers": {arch: answer for arch in ARCHITECTURES}})
        try:
            self.assertEqual(len(figure.axes), 3)
            self.assertIn("Model-judged", figure.axes[0].get_title())
            self.assertIn("similarity", figure.axes[1].get_title())
            self.assertEqual(len(figure.axes[0].patches), 8)
            self.assertTrue(all(axis.get_ylim()[1] > 1 for axis in figure.axes))
        finally:
            plt.close(figure)

    def test_overwrite_publishes_staged_files_and_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            stage = folder / "stage"
            stage.mkdir()
            name = "table2_overall_results.csv"
            old = folder / name
            old.write_text("old", encoding="utf-8")
            current = stage / name
            current.write_text("new", encoding="utf-8")
            run = folder / "run.json"
            run.write_text("{}", encoding="utf-8")
            final = folder / "final_outputs_manifest.json"
            final.write_text("{}", encoding="utf-8")
            with patch("scripts.generate_final_outputs.output_paths", return_value={name: old}), \
                    patch("scripts.generate_final_outputs.MANIFEST_OUTPUT", final), \
                    patch("scripts.generate_final_outputs.INPUTS", {}):
                with self.assertRaises(FileExistsError):
                    publish(run, {"run_id": "r"}, {name: current})
                publish(run, {"run_id": "r"}, {name: current}, overwrite=True)
            self.assertEqual(old.read_text(encoding="utf-8"), "new")
            manifest = json.loads(final.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "r")
            self.assertEqual(len(manifest["outputs"][name]["sha256"]), 64)

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
