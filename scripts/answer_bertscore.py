"""Reference-based BERTScore with explicit model provenance and length checks.

This is an embedding similarity measure, not a correct-answer rate. All supplied
answer/reference pairs are scored, including refusals. The default uses the
official BERTScore layer selection for BERT base (layer 9), with uniform token
weights and no baseline rescaling. It is deliberately smaller than BERTScore's
default English RoBERTa-large model so that the evaluation can run on CPU.

Dependencies are imported only when scoring. Model resolution is cache-only:
download the pinned snapshot separately before calling this module.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL_TYPE = "bert-base-uncased"
DEFAULT_MODEL_REVISION = "1dbc166cf8765166998eff31ade2eb64c8a40076"
DEFAULT_NUM_LAYERS = 9
BERTSCORE_FIELDS = [
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "bertscore_candidate_token_count", "bertscore_reference_token_count",
    "bertscore_truncated",
]
PROTOCOL = "reference-bertscore-v1"


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifacts(path: Path) -> dict[str, str]:
    """Fingerprint tokenizer/configuration and the available model weights."""
    names = {
        "config.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.json", "vocab.txt", "vocab.json", "merges.txt",
        "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
        "pytorch_model.bin.index.json", "sentencepiece.bpe.model", "spiece.model",
        "added_tokens.json",
    }
    artifacts = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.name in names or file.name.startswith(
                ("model-", "pytorch_model-")) and file.suffix in (".safetensors", ".bin")):
            artifacts[file.name] = _sha256(file)
    if "config.json" not in artifacts:
        raise ValueError(f"Model snapshot is missing config.json: {path}")
    return artifacts


def _resolve_snapshot(model_type: str, model_revision: str) -> Path:
    # BERTScorer does not expose a revision argument. Passing its model loader an
    # immutable local snapshot is how we enforce the selected revision.
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=model_type, revision=model_revision,
                                  local_files_only=True))
    if path.name.lower() != model_revision.lower():
        raise ValueError("Resolved model snapshot does not match the pinned revision")
    return path.resolve()


def _load_runtime():
    from bert_score import BERTScorer
    from bert_score.utils import get_hash

    return BERTScorer, get_hash


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("bert-score", "transformers", "torch", "tokenizers",
                    "huggingface-hub", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _token_count(tokenizer: Any, text: str) -> int:
    """Match BERTScore's slow-tokenizer preprocessing, without truncation."""
    text = text.strip()
    if not text:
        return len(tokenizer.build_inputs_with_special_tokens([]))
    kwargs = {"add_special_tokens": True, "truncation": False}
    # The official implementation sets this for the slow GPT-2/RoBERTa classes.
    if any(cls.__name__ in ("GPT2Tokenizer", "RobertaTokenizer")
           for cls in type(tokenizer).__mro__):
        kwargs["add_prefix_space"] = True
    return len(tokenizer.encode(text, **kwargs))


def _input_limit(tokenizer: Any, config: Any) -> int:
    limits = []
    for value in (getattr(tokenizer, "model_max_length", None),
                  getattr(config, "max_position_embeddings", None)):
        # Hugging Face uses an enormous integer sentinel for an unknown limit.
        if isinstance(value, int) and 0 < value < 1_000_000:
            limits.append(value)
    if not limits:
        raise ValueError("Cannot establish a finite model/tokenizer input limit")
    return min(limits)


def score_pairs(candidates: Sequence[str], references: Sequence[str], *,
                model_type: str = DEFAULT_MODEL_TYPE,
                num_layers: int = DEFAULT_NUM_LAYERS,
                model_revision: str = DEFAULT_MODEL_REVISION,
                device: str = "cpu", batch_size: int = 8,
                nthreads: int = 4, allow_truncation: bool = False
                ) -> tuple[list[dict], dict]:
    """Return BERTScore rows in input order and a JSON-serializable manifest.

    ``model_revision`` must be a full Hugging Face commit SHA. No dependency or
    model download occurs here. The default rejects inputs exceeding the model
    limit instead of silently scoring a prefix; opting into truncation records
    token counts and truncation flags for every pair.

    Scores are the library's raw greedy cosine precision, recall and F1, with
    IDF disabled, baseline rescaling disabled, and the slow tokenizer selected.
    In particular, an abstention is not replaced with a fabricated zero.
    """
    if isinstance(candidates, (str, bytes)) or isinstance(references, (str, bytes)):
        raise TypeError("candidates and references must be sequences of strings")
    candidates, references = list(candidates), list(references)
    if len(candidates) != len(references) or not candidates:
        raise ValueError("Provide the same positive number of candidates and references")
    if not all(isinstance(text, str) for text in candidates + references):
        raise TypeError("Every candidate and reference must be a string")
    for name, value in (("num_layers", num_layers), ("batch_size", batch_size),
                        ("nthreads", nthreads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision):
        raise ValueError("model_revision must be an immutable 40-character commit SHA")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a Hugging Face model identifier")

    snapshot = _resolve_snapshot(model_type, model_revision)
    artifacts = _model_artifacts(snapshot)
    scorer_class, get_hash = _load_runtime()
    scorer = scorer_class(model_type=str(snapshot), num_layers=num_layers,
                          batch_size=batch_size, nthreads=nthreads,
                          device=device, lang="en", all_layers=False, idf=False,
                          rescale_with_baseline=False, use_fast_tokenizer=False)
    limit = _input_limit(scorer._tokenizer, scorer._model.config)
    counts = [( _token_count(scorer._tokenizer, candidate),
                _token_count(scorer._tokenizer, reference))
              for candidate, reference in zip(candidates, references)]
    truncated = [i for i, (candidate_count, reference_count) in enumerate(counts)
                 if candidate_count > limit or reference_count > limit]
    if truncated and not allow_truncation:
        preview = ", ".join(map(str, truncated[:10]))
        raise ValueError(f"{len(truncated)} BERTScore pair(s) exceed the {limit}-token "
                         f"limit (zero-based indices: {preview}); no scores were published")

    (precision, recall, f1), library_hash = scorer.score(
        candidates, references, batch_size=batch_size, return_hash=True)
    values = [tensor.detach().cpu().tolist() for tensor in (precision, recall, f1)]
    if any(len(column) != len(candidates) for column in values):
        raise ValueError("BERTScore returned an unexpected number of scores")
    rows = []
    for i, (candidate_count, reference_count) in enumerate(counts):
        scores = [float(column[i]) for column in values]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"BERTScore returned a non-finite value for pair {i}")
        rows.append(dict(zip(BERTSCORE_FIELDS, scores + [candidate_count,
                         reference_count, int(candidate_count > limit or reference_count > limit)])))

    settings = {"protocol": PROTOCOL, "model_id": model_type,
                "model_revision": model_revision.lower(), "num_layers": num_layers,
                "idf": False, "rescale_with_baseline": False,
                "all_layers": False, "use_fast_tokenizer": False,
                "language": "en", "allow_truncation": allow_truncation,
                "max_input_tokens": limit, "artifact_sha256": artifacts,
                "package_versions": _package_versions()}
    metadata = dict(settings)
    metadata.update({
        "score_hash": _digest_json(settings),
        "canonical_library_hash": get_hash(model_type, num_layers, False, False, False, False),
        "library_hash": library_hash, "local_snapshot": str(snapshot),
        "device": device, "batch_size": batch_size, "nthreads": nthreads,
        "pair_count": len(rows), "truncated_pair_count": len(truncated),
        "max_candidate_token_count": max(pair[0] for pair in counts),
        "max_reference_token_count": max(pair[1] for pair in counts),
        "input_pairs_sha256": _digest_json(list(zip(candidates, references))),
        "interpretation": "Raw contextual-token similarity; not answer accuracy, "
                          "factual correctness, completeness or a percentage of correct answers.",
        "abstention_handling": "Score supplied refusal text against its reference normally",
        "model_selection": "BERT base layer 9 is the official BERTScore layer selection; "
                           "selected for CPU feasibility rather than the English default "
                           "RoBERTa-large layer 17.",
    })
    return rows, metadata
