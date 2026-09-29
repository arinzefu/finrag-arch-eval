#!/usr/bin/env python
"""Stage 16: deterministic answer overlap and numeric correctness, P0-P3."""

from __future__ import annotations

import argparse
import re
import string
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_run,
                                       refuse_existing, write_csv, write_provenance)

METRICS = ROOT / "results/evaluation/answer_metrics_v2.csv"
SUMMARY = ROOT / "results/aggregated/answer_summary_v2.csv"
PROVENANCE = ROOT / "results/evaluation/answer_evaluation_v2_manifest.json"
FIELDS = ["run_id", "architecture", "question_id", "exact_match", "token_f1",
          "gold_numeric_count", "generated_numeric_count", "matched_numeric_count",
          "numeric_accuracy", "numeric_precision", "numeric_recall", "numeric_f1"]
SUMMARY_FIELDS = ["table_title", "run_id", "architecture", "question_count", "numeric_applicable_count",
                  "exact_match", "token_f1", "numeric_accuracy", "numeric_precision",
                  "numeric_recall", "numeric_f1"]

NUMBER = re.compile(
    r"(?<![\w.])(?P<open>\()?\s*(?P<currency>\$|USD\s*|£|GBP\s*|€|EUR\s*)?"
    r"(?P<sign>[+-])?\s*(?P<number>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s*(?P<close>\))?\s*(?P<magnitude>thousand|million|billion|trillion|k|m|bn)?"
    r"\s*(?P<percent>%|percent\b|percentage points?\b)?(?![\w.])",
    re.IGNORECASE,
)
MAGNITUDES = {"thousand": 1000, "k": 1000, "million": 10**6, "m": 10**6,
              "billion": 10**9, "bn": 10**9, "trillion": 10**12}


def normalize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [token for token in text.split() if token not in ("a", "an", "the")]


def numeric_entities(text: str, question: str = "") -> Counter:
    question_lower = question.lower()
    question_unit = None
    if re.search(r"\b(?:usd|us dollars?)\s+millions?\b", question_lower):
        question_unit = ("USD", Decimal(10**6))
    elif re.search(r"\b(?:usd|us dollars?)\s+billions?\b", question_lower):
        question_unit = ("USD", Decimal(10**9))
    entities = Counter()
    for match in NUMBER.finditer(text):
        if match.group(0).strip().lower() == "3m":
            continue  # The company name is not a financial magnitude.
        raw = match.group("number")
        value = Decimal(raw.replace(",", ""))
        magnitude = match.group("magnitude")
        percent = match.group("percent")
        currency = match.group("currency")
        if value == value.to_integral() and 1900 <= value <= 2100 and not any(
            (magnitude, percent, currency)
        ):
            continue  # Filing years are metadata, not financial quantities.
        if match.group("sign") == "-" or (match.group("open") and match.group("close")):
            value = -value
        if magnitude:
            value *= MAGNITUDES[magnitude.lower()]
        if percent:
            unit = "percent" if percent.lower() in ("%", "percent") else "percentage_points"
        elif currency:
            unit = {"$": "USD", "£": "GBP", "€": "EUR"}.get(currency.strip(), currency.strip().upper())
            if question_unit and unit == question_unit[0] and not magnitude:
                value *= question_unit[1]
        elif question_unit:
            unit, multiplier = question_unit
            if not magnitude:
                value *= multiplier
        else:
            unit = "number"
        entities[(value.normalize(), unit)] += 1
    return entities


def score_answer(run_id: str, arch: str, question_id: str, question: str,
                 gold: str, generated: str) -> dict:
    gold_tokens, generated_tokens = normalize(gold), normalize(generated)
    overlap = sum((Counter(gold_tokens) & Counter(generated_tokens)).values())
    token_f1 = (2 * overlap / (len(gold_tokens) + len(generated_tokens))
                if gold_tokens or generated_tokens else 1.0)
    gold_nums = numeric_entities(gold, question)
    gen_nums = numeric_entities(generated, question)
    gold_count, gen_count = sum(gold_nums.values()), sum(gen_nums.values())
    matched = sum((gold_nums & gen_nums).values())
    recall = matched / gold_count if gold_count else ""
    precision = matched / gen_count if gen_count else (0.0 if gold_count else "")
    numeric_f1 = (2 * matched / (gold_count + gen_count)
                  if gold_count + gen_count else "")
    return {"run_id": run_id, "architecture": arch, "question_id": question_id,
            "exact_match": int(gold_tokens == generated_tokens), "token_f1": token_f1,
            "gold_numeric_count": gold_count, "generated_numeric_count": gen_count,
            "matched_numeric_count": matched, "numeric_accuracy": recall,
            "numeric_precision": precision, "numeric_recall": recall, "numeric_f1": numeric_f1}


def evaluate(manifest: dict, runs: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    metrics, summaries = [], []
    for arch in ARCHITECTURES:
        rows = []
        for raw in runs[arch]:
            rows.append(score_answer(manifest["run_id"], arch, raw["question_id"],
                                     raw["question"], raw["gold_answer"], raw["generated_answer"]))
        metrics.extend(rows)
        applicable = [r for r in rows if r["gold_numeric_count"]]
        summary = {"table_title": "Table A1 (v2): Held-out answer correctness", "run_id": manifest["run_id"],
                   "architecture": arch, "question_count": len(rows),
                   "numeric_applicable_count": len(applicable)}
        for field in ("exact_match", "token_f1"):
            summary[field] = sum(r[field] for r in rows) / len(rows)
        for field in ("numeric_accuracy", "numeric_precision", "numeric_recall", "numeric_f1"):
            summary[field] = (sum(r[field] for r in applicable) / len(applicable)
                              if applicable else "")
        summaries.append(summary)
    return metrics, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    refuse_existing(METRICS, SUMMARY, PROVENANCE)
    manifest, runs = load_run(args.manifest)
    metrics, summaries = evaluate(manifest, runs)
    write_csv(METRICS, metrics, FIELDS)
    write_csv(SUMMARY, summaries, SUMMARY_FIELDS)
    write_provenance(PROVENANCE, "Stage 16 v2: Held-out answer correctness", args.manifest,
                     {a: Path(manifest["outputs"][a]) for a in ARCHITECTURES},
                     {"normalization": "lowercase; strip ASCII punctuation/articles; whitespace tokens",
                      "numeric_accuracy": "matched gold numeric entities / gold numeric entities; NA if no gold numerics",
                      "numeric_matching": "multiset exact Decimal after explicit magnitude/currency/percent normalization; standalone years and company name 3M excluded",
                      "revision_from_v1": "Exclude bare company token 3M from numeric entities; preserve v1 outputs unchanged",
                      "question_unit_inference": "USD millions/billions in question applies to unqualified answer values"},
                     {"per_question": METRICS, "summary": SUMMARY})
    print("Table A1 (v2): Held-out answer correctness")
    for row in summaries:
        print(f"{row['architecture']}: EM={row['exact_match']:.3f} F1={row['token_f1']:.3f} "
              f"numeric={row['numeric_accuracy']:.3f} (n={row['numeric_applicable_count']})")


if __name__ == "__main__":
    main()
