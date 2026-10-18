#!/usr/bin/env python
"""Augment saved answers with reference correctness, completeness and BERTScore.

This is a post-hoc evaluation, not a new generation run. Completed judgments and
BERTScore are checkpointed; current answer CSVs are replaced only after all 580
rows have been evaluated. Unresolved invalid judgments retain missing semantic
scores and explicit judge_error labels. --overwrite archives previous outputs.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import answer_semantics as semantic
from scripts import evaluate_answers as legacy
from scripts.answer_bertscore import score_pairs
from scripts.evaluation_common import (
    ARCHITECTURES, MANIFEST, ROOT, load_jsonl, load_run, read_csv, sha256,
    write_csv, write_provenance,
)

EVIDENCE = ROOT / "data/processed/evidence_mapping.json"
REFERENCES = ROOT / "results/evaluation/answer_reference_units_v3.jsonl"
JUDGMENTS = ROOT / "results/evaluation/answer_quality_judgments_v3.jsonl"
INVALID = ROOT / "results/evaluation/answer_quality_invalid_responses_v3.jsonl"
BERT_CACHE = ROOT / "results/evaluation/answer_bertscore_v1.json"
QUALITY_MANIFEST = ROOT / "results/evaluation/answer_quality_evaluation_manifest.json"
QUALITY_FIELDS = [
    "exact_abstention", "answer_quality_label", "answer_correctness", "answer_completeness",
    "answer_attempted", "answer_has_contradiction", "essential_unit_count",
    "matched_essential_unit_count", "incorrect_essential_unit_count", "missing_essential_unit_count",
    "answer_quality_rationale", "judge_model", "judge_protocol",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "bertscore_candidate_token_count", "bertscore_reference_token_count", "bertscore_truncated",
]
SUMMARY_QUALITY_FIELDS = [
    "non_abstained_count", "exact_abstention_count", "correct_count", "partially_correct_count",
    "incorrect_count", "failure_to_answer_count", "judge_error_count", "semantic_scored_count",
    "semantic_non_abstained_scored_count", "bertscore_scored_count",
    "answer_correctness_lower_bound", "answer_correctness_upper_bound",
    "answer_correctness", "answer_correctness_non_abstained", "answer_completeness",
    "answer_completeness_non_abstained", "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "bertscore_f1_non_abstained", "judge_model", "judge_protocol",
]


class JudgeValidationError(RuntimeError):
    """The judge produced no usable assessment after the fixed retry budget."""

    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.usage = usage


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path: Path, identity_fields: tuple[str, ...], common: dict) -> dict:
    result = {}
    if not path.exists():
        return result
    for row in load_jsonl(path):
        key = tuple(row[k] for k in identity_fields)
        if key in result or any(row.get(k) != v for k, v in common.items()):
            raise ValueError(f"Checkpoint duplicate or protocol/source mismatch: {path} {key}")
        result[key] = row
    return result


def validate_legacy(rows: list[dict], path: Path = legacy.METRICS) -> None:
    """An extension may not silently change previously published overlap scores."""
    if not path.exists():
        return
    old_rows = read_csv(path)
    old = {(r["architecture"], r["question_id"]): r for r in old_rows}
    if len(old) != len(old_rows) or set(old) != {(r["architecture"], r["question_id"]) for r in rows}:
        raise ValueError("Current answer CSV has different question/architecture coverage")
    for row in rows:
        saved = old[row["architecture"], row["question_id"]]
        for key in legacy.FIELDS:
            a, b = str(row[key]), saved[key]
            if key in ("run_id", "architecture", "question_id") or a == "" or b == "":
                same = a == b
            else:
                same = math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
            if not same:
                raise ValueError(f"Published legacy score changed: {row['architecture']} {row['question_id']} {key}")


def bert_scores(flat: list[dict], manifest: dict, *, batch_size: int, device: str) -> dict:
    from scripts.answer_bertscore import DEFAULT_MODEL_REVISION, DEFAULT_MODEL_TYPE, DEFAULT_NUM_LAYERS
    settings = {"model_type": DEFAULT_MODEL_TYPE, "model_revision": DEFAULT_MODEL_REVISION,
                "num_layers": DEFAULT_NUM_LAYERS, "idf": False, "rescale_with_baseline": False}
    source_hash = semantic.fingerprint(canonical([
        [r["architecture"], r["question_id"], r["generated_answer"], r["gold_answer"]] for r in flat]))
    if BERT_CACHE.exists():
        cache = json.loads(BERT_CACHE.read_text(encoding="utf-8"))
        if (cache["run_id"] != manifest["run_id"] or cache["source_sha256"] != source_hash
                or cache["settings"] != settings or len(cache["rows"]) != len(flat)):
            raise ValueError("BERTScore checkpoint has different sources/configuration")
        for raw, score in zip(flat, cache["rows"]):
            if (raw["architecture"], raw["question_id"]) != (score["architecture"], score["question_id"]):
                raise ValueError("BERTScore checkpoint row order mismatch")
            if not all(math.isfinite(score[k]) for k in ("bertscore_precision", "bertscore_recall", "bertscore_f1")):
                raise ValueError("Invalid BERTScore checkpoint value")
        print(f"Reused {len(flat)} BERTScore pairs from checkpoint", flush=True)
        return cache
    print(f"Computing real BERTScore for {len(flat)} saved answer/reference pairs", flush=True)
    import torch
    torch.set_num_threads(4)
    scores, metadata = score_pairs([r["generated_answer"] for r in flat],
                                  [r["gold_answer"] for r in flat],
                                  device=device, batch_size=batch_size, nthreads=4)
    rows = [{"architecture": r["architecture"], "question_id": r["question_id"], **s}
            for r, s in zip(flat, scores)]
    cache = {"run_id": manifest["run_id"], "source_sha256": source_hash,
             "settings": settings, "metadata": metadata, "rows": rows}
    save_json(BERT_CACHE, cache)
    print(f"Saved {len(rows)} actual BERTScore results", flush=True)
    return cache


async def structured_call(client, *, system: str, prompt: str, output_format: dict,
                          parser, identity: dict) -> tuple[dict, dict]:
    retry_feedback = ""
    for attempt in range(1, 4):
        actual_prompt = prompt + retry_feedback
        response = await client.responses.create(
            model=semantic.JUDGE_MODEL, instructions=system, input=actual_prompt,
            text={"format": output_format}, max_output_tokens=2400, temperature=0,
        )
        raw_text = str(response.output_text or "")
        usage = {"input_tokens": getattr(response.usage, "input_tokens", None),
                 "output_tokens": getattr(response.usage, "output_tokens", None),
                 "response_model": response.model, "parse_attempts": attempt,
                 "final_request_sha256": semantic.fingerprint(actual_prompt)}
        try:
            if response.status != "completed":
                raise ValueError(f"Incomplete judge response: {response.status}")
            return parser(raw_text), usage
        except (ValueError, TypeError) as exc:
            append_json(INVALID, {**identity, "attempt": attempt, "error": str(exc),
                                 "response_text": raw_text, **usage})
            retry_feedback = (
                "\n\nVALIDATION FEEDBACK (not additional evidence): Your previous output was "
                "internally inconsistent or invalid. Reassess every field against the same "
                "question and evidence. Do not mechanically change only the label; correct "
                "unit assessments, attempt and contradiction flags as appropriate, then "
                "apply the stated classification rules. The fixed essential units define "
                "the required coverage: do not introduce an additional missing requirement "
                "only in the overall rationale. If every fixed unit is answered and there "
                "is no contradiction, an attempted answer meets the stated correct rule. "
                "If you believe required content is absent, reassess the relevant supplied "
                "unit as missing and explain which part of that unit is absent. If you "
                "believe an extra claim is false, reassess the contradiction flag against "
                "the evidence. Do not invent missing content or contradictions to justify "
                "a label. Return the complete JSON object.\n"
                + canonical({"validation_error": str(exc), "previous_output": raw_text}))
            if attempt == 3:
                raise JudgeValidationError(f"Judge failed validation three times: {identity}; {exc}", usage) from exc
    raise AssertionError("Unreachable retry branch")


async def semantic_scores(manifest: dict, runs: dict, evidence: dict, manifest_path: Path,
                          *, concurrency: int, limit: int | None) -> tuple[dict, dict]:
    common = {"protocol": semantic.PROTOCOL, "model": semantic.JUDGE_MODEL,
              "prompt_sha256": semantic.PROMPT_VERSION, "run_manifest_sha256": sha256(manifest_path),
              "evidence_mapping_sha256": sha256(EVIDENCE)}
    refs = load_records(REFERENCES, ("question_id",), common)
    judgments = load_records(JUDGMENTS, ("architecture", "question_id"), common)
    expected = {(a, q) for a in ARCHITECTURES for q in manifest["question_ids"]}
    if not set(judgments) <= expected or not set(refs) <= {(q,) for q in manifest["question_ids"]}:
        raise ValueError("Semantic checkpoints contain questions outside this frozen run")
    selected_ids = manifest["question_ids"][:limit]
    raw_lookup = {(a, r["question_id"]): r for a in ARCHITECTURES for r in runs[a]}
    # Validate saved payloads against their request hashes even when no call is needed.
    for (qid,), record in refs.items():
        prompt = semantic.reference_prompt(raw_lookup["P0", qid], evidence[qid])
        if record["request_sha256"] != semantic.fingerprint(prompt):
            raise ValueError(f"Reference rubric input changed: {qid}")
        semantic.parse_reference(record["reference"])
    for (arch, qid), record in judgments.items():
        reference = refs[qid,]["reference"]
        prompt = semantic.judgment_prompt(raw_lookup[arch, qid], evidence[qid], reference)
        if record["request_sha256"] != semantic.fingerprint(prompt):
            raise ValueError(f"Answer judge input changed: {arch} {qid}")
        if record.get("judgment") is None:
            if not record.get("judge_error") or record.get("parse_attempts") != 3:
                raise ValueError(f"Invalid failed-judgment checkpoint: {arch} {qid}")
            continue
        rebuilt = semantic.parse_judgment({k: record["judgment"][k] for k in (
            "correctness_label", "attempted_answer", "has_contradiction", "unit_assessments", "rationale")}, reference)
        if rebuilt != record["judgment"]:
            raise ValueError(f"Derived semantic scores do not match saved assessments: {arch} {qid}")
    need_refs = [q for q in selected_ids if (q,) not in refs]
    need_judges = [(a, q) for q in selected_ids for a in ARCHITECTURES
                   if (a, q) not in judgments and not semantic.is_exact_abstention(raw_lookup[a, q]["generated_answer"])]
    if (need_refs or need_judges) and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing. BERTScore checkpoint is retained; current answer CSVs unchanged.")
    client = None
    if need_refs or need_judges:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(timeout=90, max_retries=3)
    semaphore = asyncio.Semaphore(concurrency)

    async def make_reference(qid: str) -> None:
        async with semaphore:
            prompt = semantic.reference_prompt(raw_lookup["P0", qid], evidence[qid])
            parsed, usage = await structured_call(client, system=semantic.REFERENCE_SYSTEM_PROMPT,
                prompt=prompt, output_format=semantic.REFERENCE_FORMAT, parser=semantic.parse_reference,
                identity={"kind": "reference", "question_id": qid})
            record = {**common, "question_id": qid, "request_sha256": semantic.fingerprint(prompt),
                      "reference": parsed, **usage}
            append_json(REFERENCES, record)
            refs[qid,] = record
            if len(refs) % 10 == 0 or len(refs) == len(selected_ids):
                print(f"Reference rubrics: {len(refs)}/{len(manifest['question_ids'])}", flush=True)

    async def make_judgment(arch: str, qid: str) -> None:
        async with semaphore:
            raw, reference = raw_lookup[arch, qid], refs[qid,]["reference"]
            prompt = semantic.judgment_prompt(raw, evidence[qid], reference)
            if semantic.is_exact_abstention(raw["generated_answer"]):
                parsed = semantic.exact_abstention_result(reference)
                usage = {"input_tokens": 0, "output_tokens": 0, "parse_attempts": 0,
                         "response_model": "deterministic_exact_abstention"}
            else:
                try:
                    parsed, usage = await structured_call(client, system=semantic.JUDGMENT_SYSTEM_PROMPT,
                        prompt=prompt, output_format=semantic.JUDGMENT_FORMAT,
                        parser=lambda payload: semantic.parse_judgment(payload, reference),
                        identity={"kind": "answer", "architecture": arch, "question_id": qid})
                except JudgeValidationError as exc:
                    parsed, usage = None, {**exc.usage, "judge_error": str(exc)}
                    print(f"Unscored semantic judgment: {arch} {qid}; invalid responses retained", flush=True)
            record = {**common, "architecture": arch, "question_id": qid,
                      "request_sha256": semantic.fingerprint(prompt), "judgment": parsed, **usage}
            append_json(JUDGMENTS, record)
            judgments[arch, qid] = record
            if len(judgments) % 20 == 0 or len(judgments) == len(expected):
                print(f"Whole-answer judgments: {len(judgments)}/{len(expected)}", flush=True)

    async def run_tasks(coroutines) -> None:
        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    try:
        await run_tasks(make_reference(q) for q in need_refs)
        await run_tasks(make_judgment(a, q) for q in selected_ids for a in ARCHITECTURES
                        if (a, q) not in judgments)
    finally:
        if client is not None:
            await client.close()
    return refs, judgments


def assemble(manifest: dict, runs: dict, judgments: dict, bert: dict) -> tuple[list[dict], list[dict]]:
    rows, summaries = legacy.evaluate(manifest, runs)
    validate_legacy(rows)
    bert_lookup = {(r["architecture"], r["question_id"]): r for r in bert["rows"]}
    raw_lookup = {(a, r["question_id"]): r for a in ARCHITECTURES for r in runs[a]}
    expected = set(raw_lookup)
    if set(judgments) != expected or set(bert_lookup) != expected:
        raise ValueError("Publishing requires a semantic evaluation record and BERTScore result for every saved answer")
    for row in rows:
        key = row["architecture"], row["question_id"]
        record = judgments[key]
        j = record["judgment"]
        semantic_fields = ({
            "answer_quality_label": j["correctness_label"], "answer_correctness": j["semantic_correctness"],
            "answer_completeness": j["completeness"], "answer_attempted": int(j["attempted_answer"]),
            "answer_has_contradiction": int(j["has_contradiction"]),
            **{k: j[k] for k in ("essential_unit_count", "matched_essential_unit_count",
                                 "incorrect_essential_unit_count", "missing_essential_unit_count")},
            "answer_quality_rationale": j["rationale"],
        } if j is not None else {
            "answer_quality_label": "judge_error", "answer_correctness": "", "answer_completeness": "",
            "answer_attempted": "", "answer_has_contradiction": "", "essential_unit_count": "",
            "matched_essential_unit_count": "", "incorrect_essential_unit_count": "",
            "missing_essential_unit_count": "", "answer_quality_rationale": record["judge_error"],
        })
        row.update({
            "exact_abstention": int(semantic.is_exact_abstention(raw_lookup[key]["generated_answer"])),
            **semantic_fields, "judge_model": semantic.JUDGE_MODEL,
            "judge_protocol": semantic.PROTOCOL,
            **{k: v for k, v in bert_lookup[key].items() if k.startswith("bertscore_")},
        })
    for summary in summaries:
        selected = [r for r in rows if r["architecture"] == summary["architecture"]]
        non_abstained = [r for r in selected if not r["exact_abstention"]]
        valid = [r for r in selected if r["answer_quality_label"] != "judge_error"]
        valid_non_abstained = [r for r in valid if not r["exact_abstention"]]
        n = len(selected)
        summary["table_title"] = "Table A1: Post-hoc whole-answer correctness, completeness and similarity"
        summary.update(non_abstained_count=len(non_abstained), exact_abstention_count=n-len(non_abstained),
                       semantic_scored_count=len(valid), judge_error_count=n-len(valid),
                       semantic_non_abstained_scored_count=len(valid_non_abstained), bertscore_scored_count=n,
                       judge_model=semantic.JUDGE_MODEL, judge_protocol=semantic.PROTOCOL)
        for label in semantic.LABELS:
            summary[{"correct": "correct_count", "partially_correct": "partially_correct_count",
                     "incorrect": "incorrect_count", "failure_to_answer": "failure_to_answer_count"}[label]] = sum(
                         r["answer_quality_label"] == label for r in selected)
        for metric in ("answer_correctness", "answer_completeness", "bertscore_precision", "bertscore_recall", "bertscore_f1"):
            scored = [r for r in selected if r[metric] != ""]
            summary[metric] = sum(r[metric] for r in scored) / len(scored) if scored else ""
        for metric in ("answer_correctness", "answer_completeness", "bertscore_f1"):
            scored = [r for r in non_abstained if r[metric] != ""]
            summary[f"{metric}_non_abstained"] = sum(r[metric] for r in scored) / len(scored) if scored else ""
        summary["answer_correctness_lower_bound"] = summary["correct_count"] / n
        summary["answer_correctness_upper_bound"] = (summary["correct_count"] + summary["judge_error_count"]) / n
    return rows, summaries


def backup_outputs() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/archive" / f"before_answer_quality_{stamp}"
    candidates = [legacy.METRICS, legacy.SUMMARY, legacy.PROVENANCE, QUALITY_MANIFEST,
                  ROOT / "notebooks/02_architecture_comparison_results.ipynb",
                  ROOT / "results/final_outputs_manifest.json",
                  ROOT / "results/aggregated/question_type_summary.csv",
                  ROOT / "results/aggregated/question_type_summary_manifest.json",
                  ROOT / "results/statistics/hypothesis_tests_manifest.json",
                  ROOT / "results/aggregated/tradeoff_summary_manifest.json"]
    candidates.extend((ROOT / "results/tables").glob("*.csv"))
    candidates.extend((ROOT / "results/figures").glob("*.png"))
    candidates.extend((ROOT / "results/figures").glob("*.pdf"))
    for source in candidates:
        if source.exists():
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return destination


def publish(manifest_path: Path, manifest: dict, runs: dict, refs: dict, judgments: dict,
            bert: dict, *, overwrite: bool) -> Path:
    if not overwrite and any(p.exists() for p in (legacy.METRICS, legacy.SUMMARY, legacy.PROVENANCE, QUALITY_MANIFEST)):
        raise FileExistsError("Current answer evaluation exists; --overwrite is required")
    rows, summaries = assemble(manifest, runs, judgments, bert)
    inputs = {"evidence": EVIDENCE, "reference_rubrics": REFERENCES, "judgments": JUDGMENTS,
              **({"invalid_responses": INVALID} if INVALID.exists() else {}),
              "bertscore": BERT_CACHE, "semantic_source": Path(semantic.__file__),
              "bertscore_source": ROOT / "scripts/answer_bertscore.py",
              "evaluator_source": Path(__file__),
              **{a: Path(manifest["outputs"][a]) for a in ARCHITECTURES}}
    for name, filename in (("qualitative_audit", "answer_quality_v3_audit.md"),
                           ("qualitative_audit_cases", "answer_quality_v3_audit.csv")):
        audit_path = ROOT / "results/evaluation" / filename
        if audit_path.exists():
            inputs[name] = audit_path
    input_records = {k: {"path": str(v.resolve()), "sha256": sha256(v)} for k, v in inputs.items()}
    run_record = {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}
    backup = backup_outputs()
    parameters = {
        "evaluation_status": "post-hoc extension; saved generation outputs unchanged",
        "judge_model": semantic.JUDGE_MODEL, "judge_protocol": semantic.PROTOCOL,
        "prompt_and_schema_sha256": semantic.PROMPT_VERSION,
        "reference_rubric_policy": "one architecture-blind essential-unit rubric per question, shared across P0-P3",
        "answer_correctness": "1 only for fully correct model-judged whole answer; other valid labels 0; judge_error missing; macro mean over valid judgments",
        "answer_completeness": "model-judged correctly answered reference units / generated reference units; completeness proxy; macro mean over valid judgments",
        "unresolved_judgments": "three validation failures yield judge_error and missing semantic scores; never imputed; scored counts reported; all-question correctness lower/upper bounds assign unresolved 0/1 respectively",
        "abstention_policy": "frozen exact refusal detector retained separately from broader model-judged failure_to_answer",
        "conditional_denominator": "valid scored non-exact-abstention responses, including noncanonical failures and partial answers; conditional BERTScore uses all non-exact-abstention responses; overall BERTScore covers every saved answer including exact refusals",
        "judge_evidence": "question, reference answer, annotated evidence and shared rubric; architecture and actual retrieved context withheld",
        "judge_validation": "strict schema, complete unit IDs, label/count consistency; up to three parse attempts; no independent human validation claimed",
        "pilot_policy": "gpt-4o-mini v1/v2 pilots retained but not reused; user-selected v3 mini judge with shared generated units and consistency feedback; spot checks found optional-detail penalties, compatible paraphrase errors and rubric scope omissions; exploratory model judgments, not independently verified accuracy",
        "legacy_policy": "original exact-match, token-F1 and numeric columns reproduced and checked unchanged",
        "bertscore": bert["metadata"], "previous_outputs_archive": str(backup.resolve()),
        "article_comparison": "different datasets, judges and metric definitions are not directly comparable",
    }
    write_csv(legacy.METRICS, rows, legacy.FIELDS + QUALITY_FIELDS)
    write_csv(legacy.SUMMARY, summaries, legacy.SUMMARY_FIELDS + SUMMARY_QUALITY_FIELDS)
    write_provenance(legacy.PROVENANCE, "Post-hoc extended answer correctness evaluation", manifest_path,
                     inputs, parameters, {"per_question": legacy.METRICS, "summary": legacy.SUMMARY})
    payload = {"title": "Post-hoc whole-answer correctness, completeness and BERTScore",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": manifest["run_id"],
               "run_manifest": run_record,
               "inputs": input_records,
               "parameters": parameters,
               "outputs": {k: {"path": str(v.resolve()), "sha256": sha256(v)} for k, v in
                           {"per_question": legacy.METRICS, "summary": legacy.SUMMARY}.items()}}
    save_json(QUALITY_MANIFEST, payload)
    print(f"Replaced current answer CSVs; previous outputs archived at {backup}", flush=True)
    return backup


def refresh_reports(manifest_path: Path) -> None:
    # Existing prespecified tests/trade-off values use only legacy columns, which
    # validate_legacy checked unchanged. Refresh their input provenance honestly.
    for path in (ROOT / "results/statistics/hypothesis_tests_manifest.json",
                 ROOT / "results/aggregated/tradeoff_summary_manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["inputs"].values():
            source = Path(record["path"])
            if source.resolve() == legacy.METRICS.resolve():
                record["sha256"] = sha256(source)
        payload["parameters"]["answer_quality_extension"] = (
            "Input CSV gained post-hoc columns; all tested legacy values verified unchanged. "
            "Existing prespecified tests and trade-off values unchanged; new metrics descriptive.")
        save_json(path, payload)
    from scripts import analyze_question_types, generate_final_outputs
    old_argv = sys.argv
    try:
        sys.argv = ["analyze_question_types.py", "--manifest", str(manifest_path), "--overwrite"]
        analyze_question_types.main()
        sys.argv = ["generate_final_outputs.py", "--manifest", str(manifest_path), "--overwrite"]
        generate_final_outputs.main()
    finally:
        sys.argv = old_argv
    print("Refreshed category summaries, final figures/tables and provenance", flush=True)


def rerun_notebook() -> None:
    """Execute the read-only reporting cells and persist their actual MIME output."""
    import ast
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from scripts.notebook_report import load_report
    path = ROOT / "notebooks/02_architecture_comparison_results.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    shell = InteractiveShell.instance()
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source)
        with capture_output() as captured:
            result = shell.run_cell(source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Reporting cell {index} failed: {captured.stdout}\n{captured.stderr}")
        outputs = []
        for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if value:
                outputs.append({"output_type": "stream", "name": name, "text": value.splitlines(True)})
        outputs.extend({"output_type": "display_data", "data": out.data, "metadata": out.metadata}
                       for out in captured.outputs)
        count += 1
        cell.update(execution_count=count, outputs=outputs)
    nbformat.validate(nbformat.from_dict(notebook))
    load_report().abstentions()
    save_json(path, notebook)
    print(f"Executed and saved {count} notebook reporting cells", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--bertscore-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true", help="Checkpoint semantic judgments without publishing")
    parser.add_argument("--limit", type=int, help="Judge first N questions per architecture, checkpoint only")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--refresh-reports", action="store_true")
    parser.add_argument("--rerun-notebook", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env",
                        help="local credential file; values are never included in provenance")
    args = parser.parse_args()
    if args.concurrency < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Concurrency, batch size and limit must be positive")
    if args.judge_only and args.bertscore_only:
        parser.error("Choose at most one checkpoint-only mode")
    if args.rerun_notebook and (not args.refresh_reports or args.limit or args.judge_only or args.bertscore_only):
        parser.error("--rerun-notebook requires a full run with --refresh-reports")
    try:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    except ImportError:
        pass  # Environment-configured keys do not require python-dotenv.
    manifest, runs = load_run(args.manifest)
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "results/evaluation/faithfulness_evaluation_manifest.json").read_text(encoding="utf-8"))
    for arch in ARCHITECTURES:
        if sha256(Path(manifest["outputs"][arch])) != protocol["inputs"][arch]["sha256"]:
            raise ValueError(f"Frozen raw answers changed: {arch}")
    if sha256(EVIDENCE) != protocol["inputs"]["gold_evidence"]["sha256"]:
        raise ValueError("Frozen annotated evidence changed")
    legacy_rows, _ = legacy.evaluate(manifest, runs)
    validate_legacy(legacy_rows)
    if args.check_only:
        print(f"Frozen sources and {len(legacy_rows)} legacy rows validated. API key available: {bool(os.getenv('OPENAI_API_KEY'))}")
        return
    flat = [r for a in ARCHITECTURES for r in runs[a]]
    bert = None if args.judge_only else bert_scores(flat, manifest, batch_size=args.batch_size, device=args.device)
    if args.bertscore_only:
        return
    refs, judgments = asyncio.run(semantic_scores(manifest, runs, evidence, args.manifest,
                                concurrency=args.concurrency, limit=args.limit))
    if args.judge_only:
        print("Semantic checkpoint saved; current metrics CSVs were not replaced")
        return
    if args.limit is not None:
        print("Pilot checkpoint only; current metrics CSVs were not replaced")
        return
    publish(args.manifest, manifest, runs, refs, judgments, bert, overwrite=args.overwrite)
    if args.refresh_reports:
        refresh_reports(args.manifest)
    if args.rerun_notebook:
        rerun_notebook()


if __name__ == "__main__":
    main()
