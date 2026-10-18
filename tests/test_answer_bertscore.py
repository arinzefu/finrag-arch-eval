"""Offline protocol tests; the live run exercises the actual BERTScore model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import answer_bertscore as bs


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeTokenizer:
    model_max_length = 6

    def encode(self, text, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is True
        return [0] + text.split() + [1]

    def build_inputs_with_special_tokens(self, tokens):
        return [0] + tokens + [1]


class FakeScorer:
    calls = []
    settings = []
    output = None

    def __init__(self, **kwargs):
        self.settings.append(kwargs)
        self._tokenizer = FakeTokenizer()
        self._model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=6))

    def score(self, candidates, references, **kwargs):
        self.calls.append((candidates, references, kwargs))
        columns = self.output if self.output is not None else (
            [0.6] * len(candidates), [0.8] * len(candidates), [0.7] * len(candidates))
        return tuple(FakeTensor(values) for values in columns), "official-local-path-hash"


class BERTScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.snapshot = Path(self.temp.name)
        (self.snapshot / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
        (self.snapshot / "vocab.txt").write_text("word", encoding="utf-8")
        (self.snapshot / "model.safetensors").write_bytes(b"test weight fingerprint")
        FakeScorer.calls, FakeScorer.settings, FakeScorer.output = [], [], None
        self.resolve = patch.object(bs, "_resolve_snapshot", return_value=self.snapshot)
        self.resolve_mock = self.resolve.start()
        self.addCleanup(self.resolve.stop)
        self.runtime = patch.object(bs, "_load_runtime", return_value=(
            FakeScorer, lambda *args: "canonical-model-hash"))
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.versions = patch.object(bs, "_package_versions", return_value={"bert-score": "test"})
        self.versions.start()
        self.addCleanup(self.versions.stop)

    def test_refusal_is_scored_normally_and_order_is_preserved(self):
        candidates = ["Insufficient evidence.", "Data Center"]
        references = ["$10 million", "Data Center"]
        rows, metadata = bs.score_pairs(candidates, references)
        self.assertEqual(FakeScorer.calls[0][:2], (candidates, references))
        self.assertEqual([row["bertscore_f1"] for row in rows], [0.7, 0.7])
        self.assertEqual(rows[0]["bertscore_candidate_token_count"], 4)
        self.assertEqual(rows[0]["bertscore_truncated"], 0)
        self.assertEqual(metadata["pair_count"], 2)
        self.assertEqual(metadata["model_revision"], bs.DEFAULT_MODEL_REVISION)
        self.assertEqual(metadata["canonical_library_hash"], "canonical-model-hash")
        self.assertEqual(len(metadata["artifact_sha256"]["model.safetensors"]), 64)
        self.assertEqual(list(rows[0]), bs.BERTSCORE_FIELDS)
        self.assertFalse(FakeScorer.settings[0]["idf"])
        self.assertFalse(FakeScorer.settings[0]["rescale_with_baseline"])
        self.assertFalse(FakeScorer.settings[0]["use_fast_tokenizer"])
        json.dumps(metadata)  # Manifest must remain JSON-serializable.

    def test_overlength_pair_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "exceed the 6-token limit"):
            bs.score_pairs(["one two three four five"], ["reference"])
        self.assertFalse(FakeScorer.calls)

    def test_explicit_truncation_records_original_lengths(self):
        rows, metadata = bs.score_pairs(["one two three four five"], ["reference"],
                                        allow_truncation=True)
        self.assertEqual(rows[0]["bertscore_candidate_token_count"], 7)
        self.assertEqual(rows[0]["bertscore_truncated"], 1)
        self.assertEqual(metadata["truncated_pair_count"], 1)
        self.assertTrue(metadata["allow_truncation"])

    def test_unknown_or_mutable_revision_rejected(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            bs.score_pairs(["answer"], ["reference"], model_revision="main")
        self.assertFalse(self.resolve_mock.called)

    def test_pair_validation_does_not_load_runtime(self):
        cases = [([], []), (["answer"], []), (["answer"], [None]), ("answer", ["reference"])]
        for candidates, references in cases:
            with self.subTest(candidates=candidates, references=references):
                with self.assertRaises((TypeError, ValueError)):
                    bs.score_pairs(candidates, references)
        self.assertFalse(self.resolve_mock.called)

    def test_invalid_batch_size_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bs.score_pairs(["answer"], ["reference"], batch_size=value)

    def test_empty_text_is_passed_to_library(self):
        rows, _ = bs.score_pairs([""], ["reference"])
        self.assertEqual(rows[0]["bertscore_candidate_token_count"], 2)
        self.assertEqual(FakeScorer.calls[0][0], [""])

    def test_nonfinite_results_cannot_be_published(self):
        FakeScorer.output = ([0.5], [0.6], [float("nan")])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bs.score_pairs(["answer"], ["reference"])

    def test_score_shape_is_validated(self):
        FakeScorer.output = ([0.5, 0.4], [0.6], [0.5])
        with self.assertRaisesRegex(ValueError, "unexpected number"):
            bs.score_pairs(["answer"], ["reference"])

    def test_model_fingerprint_changes_score_hash(self):
        _, first = bs.score_pairs(["answer"], ["reference"])
        (self.snapshot / "vocab.txt").write_text("different", encoding="utf-8")
        _, second = bs.score_pairs(["answer"], ["reference"])
        self.assertNotEqual(first["score_hash"], second["score_hash"])
        self.assertEqual(first["input_pairs_sha256"], second["input_pairs_sha256"])

    def test_roberta_slow_tokenizer_prefix_matches_official_preprocessing(self):
        class RobertaTokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                self.prefix = kwargs.get("add_prefix_space")
                return super().encode(text, **kwargs)

        tokenizer = RobertaTokenizer()
        self.assertEqual(bs._token_count(tokenizer, "  text  "), 3)
        self.assertTrue(tokenizer.prefix)

    def test_tokenizer_sentinel_uses_model_limit(self):
        tokenizer = SimpleNamespace(model_max_length=10**30)
        config = SimpleNamespace(max_position_embeddings=512)
        self.assertEqual(bs._input_limit(tokenizer, config), 512)


if __name__ == "__main__":
    unittest.main()
