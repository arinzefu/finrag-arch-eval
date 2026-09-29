#!/usr/bin/env python
"""Stage 17: resumable fixed-judge answer-unit support against retrieved and gold evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import (ARCHITECTURES, MANIFEST, ROOT, load_run, refuse_existing,
                                       sha256, write_csv, write_provenance)
from scripts.evaluate_answers import NUMBER
from src.generation.llm import OpenAIResponsesLLM

EVIDENCE = ROOT / "data/processed/evidence_mapping.json"
PROTOCOL = "answer-unit-passage-v4"
JOURNAL = ROOT / "results/evaluation/faithfulness_judgments_v4.jsonl"
INVALID_JOURNAL = ROOT / "results/evaluation/faithfulness_invalid_responses_v4.jsonl"
METRICS = ROOT / "results/evaluation/faithfulness_metrics.csv"
SUMMARY = ROOT / "results/aggregated/faithfulness_summary.csv"
PROVENANCE = ROOT / "results/evaluation/faithfulness_evaluation_manifest.json"
FIELDS = ["run_id", "architecture", "question_id", "judge_model", "judge_protocol", "abstained",
          "context_claim_count", "context_supported_count", "context_faithfulness",
          "context_hallucination", "context_unsupported_claims", "context_guard_override_count", "gold_claim_count",
          "gold_supported_count", "gold_evidence_support", "gold_evidence_hallucination",
          "gold_unsupported_claims", "gold_guard_override_count"]
SUMMARY_FIELDS = ["table_title", "run_id", "architecture", "question_count", "abstention_count",
                  "context_applicable_count", "context_faithfulness", "context_hallucination",
                  "gold_applicable_count", "gold_evidence_support", "gold_evidence_hallucination",
                  "judge_model", "judge_protocol", "context_guard_override_count",
                  "gold_guard_override_count"]
SYSTEM_PROMPT = """You are a financial QA evidence auditor. Evaluate ONLY the numbered Answer units supplied by the user; do not create, rewrite, or import claims from the Evidence. Return one assessment for every unit_id. Mark factual=false for headings, instructions, generic procedures, formulas without asserted values, and pure abstentions. For factual units, judge the entire unit against ONLY the supplied Evidence passages, not outside knowledge or the question. A unit with any unsupported factual assertion is unsupported. For supported units, cite the IDs of passages that together entail every factual assertion; never invent passage IDs. Use support_type direct for directly stated facts and arithmetic only for results calculated from numbers in cited passages. For arithmetic, write derivation as numeric operands followed by = result, and cite every operand. Absent, contradictory, wrong-year, wrong-unit, or uncited units are unsupported. Return JSON only."""
SCHEMA = {
    "type": "object", "properties": {"assessments": {"type": "array", "items": {
        "type": "object", "properties": {
            "unit_id": {"type": "string"}, "factual": {"type": "boolean"},
            "supported": {"type": "boolean"},
            "support_type": {"type": "string", "enum": ["direct", "arithmetic", "unsupported"]},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "derivation": {"type": "string"},
        }, "required": ["unit_id", "factual", "supported", "support_type",
                        "evidence_ids", "derivation"],
        "additionalProperties": False,
    }}}, "required": ["assessments"], "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "name": "financial_answer_units_v4",
          "strict": True, "schema": SCHEMA}
PROMPT_VERSION = hashlib.sha256(
    json.dumps({"prompt": SYSTEM_PROMPT, "format": FORMAT,
                "unit_splitter": "line-semicolon-sentence-v1"}, sort_keys=True).encode()
).hexdigest()


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().rstrip(".") in {
        "insufficient evidence in the retrieved context", "insufficient evidence",
        "i don't know", "i do not know", "cannot determine from the provided information",
    }


def gold_passages(mapping: dict) -> list[dict[str, str]]:
    items = mapping.get("evidence_items")
    if not mapping.get("mapping_complete") or not isinstance(items, list) or not items:
        raise ValueError("Gold evidence mapping is incomplete.")
    return [{"evidence_id": f"E{index}",
             "text": f"[{item['evidence_doc_name']} page {item['evidence_page_num']}]\n{item['evidence_text']}"}
            for index, item in enumerate(items, 1)]


def context_passages(chunks: list[dict]) -> list[dict[str, str]]:
    return [{"evidence_id": f"E{index}", "text": f"[{c['chunk_id']}]\n{c['text']}"}
            for index, c in enumerate(chunks, 1)]


def serialized_passages(passages: list[dict[str, str]]) -> str:
    return json.dumps(passages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def answer_units(answer: str) -> list[dict[str, str]]:
    units = []
    for line in answer.splitlines():
        line = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", line.strip())
        if not line:
            continue
        for clause in re.split(r";\s+", line):
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])", clause):
                if part.strip():
                    units.append({"unit_id": f"U{len(units) + 1}", "text": part.strip()})
    return units


def financial_spans(text: str) -> list[str]:
    return [match.group(0).strip() for match in NUMBER.finditer(text)
            if any(match.group(field) for field in ("currency", "magnitude", "percent"))
            and not (match.group("magnitude") and match.group("magnitude").lower() in ("m", "k")
                     and not match.group("currency")
                     and match.start("magnitude") == match.end("number"))]


def number_keys(text: str) -> set[str]:
    return {re.sub(r"\D", "", match.group("number")) for match in NUMBER.finditer(text)}


def parse_judgment(raw_text: str, units: list[dict[str, str]],
                   passages: list[dict[str, str]]) -> dict:
    payload = json.loads(raw_text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object.")
    assessments = payload.get("assessments")
    if not isinstance(assessments, list):
        raise ValueError("Judge response has no assessments list.")
    by_id = {}
    unknown_or_duplicate_ids = []
    known_ids = {unit["unit_id"] for unit in units}
    for item in assessments:
        if (not isinstance(item, dict) or not isinstance(item.get("unit_id"), str)
                or not isinstance(item.get("factual"), bool)
                or not isinstance(item.get("supported"), bool)
                or item.get("support_type") not in ("direct", "arithmetic", "unsupported")
                or not isinstance(item.get("derivation"), str)
                or not isinstance(item.get("evidence_ids"), list)
                or any(not isinstance(q, str) for q in item["evidence_ids"])):
            raise ValueError("Judge response has an invalid unit assessment.")
        unit_id = item["unit_id"]
        if unit_id not in known_ids or unit_id in by_id:
            unknown_or_duplicate_ids.append(unit_id)
            continue
        by_id[unit_id] = item
    checked, nonfactual = [], []
    passage_by_id = {item["evidence_id"]: item["text"] for item in passages}
    for unit in units:
        unit_id, claim = unit["unit_id"], unit["text"]
        item = by_id.get(unit_id)
        if item is None:
            checked.append({"unit_id": unit_id, "claim": claim, "judge_supported": False,
                            "supported": False, "support_type": "unsupported",
                            "evidence_ids": [], "derivation": "",
                            "guard_reason": "missing_unit_assessment"})
            continue
        if not item["factual"] and not financial_spans(claim):
            nonfactual.append(unit_id)
            continue
        judge_supported = item["supported"]
        cited_ids = [value.strip() for value in item["evidence_ids"]]
        cited_text = "\n".join(passage_by_id.get(value, "") for value in cited_ids)
        supported = judge_supported
        guard_reason = "financial_unit_marked_nonfactual" if not item["factual"] else ""
        if not item["factual"]:
            supported = False
        if supported and item["support_type"] == "unsupported":
            supported, guard_reason = False, "support_type_conflict"
        elif supported and (not cited_ids or any(value not in passage_by_id for value in cited_ids)):
            supported, guard_reason = False, "missing_or_unknown_evidence_id"
        elif supported and item["support_type"] == "direct":
            cited_keys = number_keys(cited_text)
            if any(not number_keys(span).issubset(cited_keys) for span in financial_spans(claim)):
                supported, guard_reason = False, "financial_number_absent_from_passage"
        elif supported and item["support_type"] == "arithmetic":
            premises = number_keys(item["derivation"].split("=", 1)[0])
            cited_keys = number_keys(cited_text)
            if not item["derivation"].strip() or not premises or not premises.issubset(cited_keys):
                supported, guard_reason = False, "arithmetic_premise_not_cited"
        checked.append({"unit_id": unit_id, "claim": claim, "judge_supported": judge_supported,
                        "supported": supported, "support_type": item["support_type"],
                        "evidence_ids": cited_ids, "derivation": item["derivation"],
                        "guard_reason": guard_reason})
    return {"claims": checked, "units": units, "nonfactual_unit_ids": nonfactual,
            "unknown_or_duplicate_unit_ids": unknown_or_duplicate_ids}


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_journal(path: Path, model: str, manifest_hash: str, evidence_hash: str) -> dict[tuple[str, str, str], dict]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            key = (record["architecture"], record["question_id"], record["kind"])
            if key in records:
                raise ValueError(f"Duplicate judgment at line {number}: {key}")
            if any((record.get("protocol") != PROTOCOL,
                    record["model"] != model,
                    record["run_manifest_sha256"] != manifest_hash,
                    record["evidence_mapping_sha256"] != evidence_hash,
                    record["prompt_sha256"] != PROMPT_VERSION)):
                raise ValueError("Existing judge journal has different model, data, or rubric.")
            if not isinstance(record.get("judgment", {}).get("claims"), list):
                raise ValueError(f"Invalid saved judgment at line {number}.")
            records[key] = record
    return records


def append_invalid_response(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def judge(arch: str, raw: dict, kind: str, passages: list[dict[str, str]], llm: OpenAIResponsesLLM,
          manifest_hash: str, evidence_hash: str, *,
          invalid_log: Path = INVALID_JOURNAL, max_parse_attempts: int = 3) -> dict:
    if max_parse_attempts < 1:
        raise ValueError("max_parse_attempts must be positive.")
    units = answer_units(raw["generated_answer"])
    prompt = (f"Question (not evidence): {raw['question']}\n\n"
              f"Answer units (evaluate these IDs only):\n"
              + "\n".join(f"{unit['unit_id']}: {unit['text']}" for unit in units)
              + f"\n\nEvidence type: {kind}\nEvidence passages:\n"
              + ("\n\n".join(f"{item['evidence_id']}: {item['text']}" for item in passages)
                 or "[NO EVIDENCE RETRIEVED]"))
    for attempt in range(1, max_parse_attempts + 1):
        response = llm.client.responses.create(
            model=llm.model, instructions=SYSTEM_PROMPT, input=prompt,
            max_output_tokens=llm.max_output_tokens, text={"format": FORMAT},
        )
        response_text = str(getattr(response, "output_text", "") or "")
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        try:
            if getattr(response, "status", None) != "completed":
                raise ValueError(f"Response status: {getattr(response, 'status', None)}")
            judgment = parse_judgment(response_text, units, passages)
            break
        except ValueError as exc:
            append_invalid_response(invalid_log, {
                "architecture": arch, "question_id": raw["question_id"], "kind": kind,
                "model": llm.model, "protocol": PROTOCOL, "prompt_sha256": PROMPT_VERSION,
                "attempt": attempt, "parse_error": str(exc),
                "response_text": response_text, "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            if attempt == max_parse_attempts:
                raise RuntimeError(
                    f"Judge returned an invalid response {max_parse_attempts} times for "
                    f"{arch} {raw['question_id']} {kind}; inspect {invalid_log}. "
                    "Completed judgments remain in the checkpoint."
                ) from exc
    return {"architecture": arch, "question_id": raw["question_id"], "kind": kind,
            "model": llm.model, "protocol": PROTOCOL, "run_manifest_sha256": manifest_hash,
            "evidence_mapping_sha256": evidence_hash, "prompt_sha256": PROMPT_VERSION,
            "answer_sha256": fingerprint(raw["generated_answer"]),
            "reference_sha256": fingerprint(serialized_passages(passages)),
            "judgment": judgment,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "parse_attempts": attempt}


def score_record(record: dict | None) -> tuple[object, object, object, object, int]:
    if record is None:
        return "", "", "", "[]", 0
    claims = record["judgment"]["claims"]
    if not claims:
        return 0, 0, "", "[]", 0
    supported = sum(item["supported"] for item in claims)
    unsupported = [item["claim"] for item in claims if not item["supported"]]
    overrides = sum(bool(item.get("guard_reason")) for item in claims)
    return (len(claims), supported, supported / len(claims),
            json.dumps(unsupported, ensure_ascii=False), overrides)


def assemble(manifest: dict, runs: dict, records: dict, model: str) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for arch in ARCHITECTURES:
        arch_rows = []
        for raw in runs[arch]:
            qid = raw["question_id"]
            c_count, c_supported, c_score, c_unsupported, c_overrides = score_record(
                records.get((arch, qid, "context")) if arch != "P0" else None)
            g_count, g_supported, g_score, g_unsupported, g_overrides = score_record(
                records.get((arch, qid, "gold")))
            row = {"run_id": manifest["run_id"], "architecture": arch, "question_id": qid,
                   "judge_model": model, "judge_protocol": PROTOCOL,
                   "abstained": int(is_abstention(raw["generated_answer"])),
                   "context_claim_count": c_count, "context_supported_count": c_supported,
                   "context_faithfulness": c_score,
                   "context_hallucination": 1 - c_score if c_score != "" else "",
                   "context_unsupported_claims": c_unsupported if arch != "P0" else "",
                   "context_guard_override_count": c_overrides if arch != "P0" else "",
                   "gold_claim_count": g_count, "gold_supported_count": g_supported,
                   "gold_evidence_support": g_score,
                   "gold_evidence_hallucination": 1 - g_score if g_score != "" else "",
                   "gold_unsupported_claims": g_unsupported,
                   "gold_guard_override_count": g_overrides}
            rows.append(row)
            arch_rows.append(row)
        summary = {"table_title": "Table F1: Unit-level evidence support and hallucination",
                   "run_id": manifest["run_id"], "architecture": arch,
                   "question_count": len(arch_rows), "abstention_count": sum(r["abstained"] for r in arch_rows),
                   "judge_model": model, "judge_protocol": PROTOCOL,
                   "context_guard_override_count": sum(r["context_guard_override_count"] or 0 for r in arch_rows),
                   "gold_guard_override_count": sum(r["gold_guard_override_count"] for r in arch_rows)}
        for prefix, key in (("context", "context_faithfulness"), ("gold", "gold_evidence_support")):
            vals = [r[key] for r in arch_rows if r[key] != ""]
            summary[f"{prefix}_applicable_count"] = len(vals)
            summary[key] = sum(vals) / len(vals) if vals else ""
            hkey = "context_hallucination" if prefix == "context" else "gold_evidence_hallucination"
            summary[hkey] = 1 - summary[key] if vals else ""
        summaries.append(summary)
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, help="Checkpoint first N questions per architecture; no final CSV until complete")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.model != "gpt-4o-mini":
        parser.error("The agreed fixed judge model is gpt-4o-mini; use a new protocol for another model.")
    refuse_existing(METRICS, SUMMARY, PROVENANCE)
    manifest, runs = load_run(args.manifest)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    manifest_hash, evidence_hash = sha256(args.manifest), sha256(args.evidence)
    records = load_journal(JOURNAL, args.model, manifest_hash, evidence_hash)
    expected_keys = {(arch, qid, kind) for arch in ARCHITECTURES
                     for qid in manifest["question_ids"]
                     for kind in (("gold",) if arch == "P0" else ("gold", "context"))}
    if not set(records).issubset(expected_keys):
        raise ValueError("Judge journal contains questions outside the frozen run.")
    todo = []
    for arch in ARCHITECTURES:
        for raw in runs[arch][:args.limit]:
            mapping = evidence.get(raw["question_id"])
            if not isinstance(mapping, dict):
                raise ValueError(f"Missing gold evidence for {raw['question_id']}")
            references = {"gold": gold_passages(mapping)}
            if arch != "P0":
                references["context"] = context_passages(raw["retrieved_chunks"])
            for kind, passages in references.items():
                key = (arch, raw["question_id"], kind)
                existing = records.get(key)
                if existing:
                    if (existing["answer_sha256"] != fingerprint(raw["generated_answer"])
                            or existing["reference_sha256"] != fingerprint(serialized_passages(passages))):
                        raise ValueError(f"Saved judgment source changed: {key}")
                elif is_abstention(raw["generated_answer"]):
                    records[key] = {"architecture": arch, "question_id": raw["question_id"],
                                    "kind": kind, "model": args.model, "protocol": PROTOCOL,
                                    "run_manifest_sha256": manifest_hash,
                                    "evidence_mapping_sha256": evidence_hash, "prompt_sha256": PROMPT_VERSION,
                                    "answer_sha256": fingerprint(raw["generated_answer"]),
                                    "reference_sha256": fingerprint(serialized_passages(passages)),
                                    "judgment": {"claims": []}, "input_tokens": 0,
                                    "output_tokens": 0, "parse_attempts": 0}
                    todo.append(records[key])
                else:
                    todo.append((arch, raw, kind, passages))
    needs_api = any(isinstance(task, tuple) for task in todo)
    if needs_api and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for remaining judge calls; no outputs changed.")
    llm = OpenAIResponsesLLM(model=args.model, max_output_tokens=2400) if needs_api else None
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as handle:
        for task in todo:
            record = (judge(*task, llm, manifest_hash, evidence_hash)
                      if isinstance(task, tuple) else task)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records[(record["architecture"], record["question_id"], record["kind"])] = record
    expected = len(expected_keys)
    print(f"Saved judgments: {len(records)}/{expected} in {JOURNAL}")
    if args.limit is not None:
        pilot_runs = {arch: runs[arch][:args.limit] for arch in ARCHITECTURES}
        pilot_rows, pilot_summary = assemble(manifest, pilot_runs, records, args.model)
        pilot_path = ROOT / f"results/smoke/faithfulness_v4_{args.limit}_preview.csv"
        if not pilot_path.exists():
            write_csv(pilot_path, pilot_rows, FIELDS)
        print(f"Pilot preview: {pilot_path}")
        print("Pilot only | architecture | abstentions | context support (n) | gold support (n) | guard overrides")
        for item in pilot_summary:
            print(f"{item['architecture']} | {item['abstention_count']} | "
                  f"{item['context_faithfulness']} ({item['context_applicable_count']}) | "
                  f"{item['gold_evidence_support']} ({item['gold_applicable_count']}) | "
                  f"{item['context_guard_override_count'] + item['gold_guard_override_count']}")
    if set(records) != expected_keys:
        print("Partial checkpoint only. Rerun without --limit to complete Stage 17.")
        return
    metrics, summary = assemble(manifest, runs, records, args.model)
    write_csv(METRICS, metrics, FIELDS)
    write_csv(SUMMARY, summary, SUMMARY_FIELDS)
    write_provenance(PROVENANCE, "Stage 17: Unit-level evidence support", args.manifest,
                     {"gold_evidence": args.evidence, "judgments": JOURNAL,
                      **{a: Path(manifest["outputs"][a]) for a in ARCHITECTURES}},
                     {"judge_model": args.model, "judge_protocol": PROTOCOL,
                      "prompt_and_schema_sha256": PROMPT_VERSION,
                      "response_format": "strict JSON schema via Responses text.format",
                      "judge_max_output_tokens": 2400, "judge_temperature": None,
                      "invalid_response_policy": "retry identical request up to 3 times; log invalid responses separately",
                      "source_guards": ["scored units are segmented deterministically from answer",
                                        "cited passage IDs exist in the supplied reference",
                                        "direct financial numbers occur in cited passages",
                                        "arithmetic operands occur in cited passages",
                                        "missing unit assessments and unscored financial units counted unsupported"],
                      "previous_pilots": ["faithfulness_judgments.jsonl retained, not reused",
                                          "faithfulness_invalid_responses_v2.jsonl retained, not reused",
                                          "faithfulness_judgments_v3.jsonl retained, not reused"],
                      "unit_score": "supported factual answer units / all factual answer units; macro mean over applicable answers",
                      "hallucination": "1 - claim support; empty-claim answers NA",
                      "P0_context": "not applicable", "gold_reference": "FinanceBench annotated evidence_text",
                      "context_reference": "actual final retrieved chunk text"},
                     {"per_question": METRICS, "summary": SUMMARY, "judgments": JOURNAL})
    print("Table F1: Unit-level evidence support and hallucination")
    for row in summary:
        print(f"{row['architecture']}: gold support={row['gold_evidence_support']:.3f} "
              f"(n={row['gold_applicable_count']}), context support={row['context_faithfulness']}")


if __name__ == "__main__":
    main()
