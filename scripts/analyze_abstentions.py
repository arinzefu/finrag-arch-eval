#!/usr/bin/env python
"""Post-hoc abstention overlap and evidence diagnostics from the saved frozen run.

No retrieval, generation, or judging is run. Diagnostic categories describe
observed gold-page availability, not verified causes of a model's decision.
Run: python scripts/analyze_abstentions.py [--check-only | --overwrite]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_common import (
    ARCHITECTURES, MANIFEST, ROOT, load_jsonl, load_run, read_csv, sha256, write_csv,
)

RAG_ARCHITECTURES = ARCHITECTURES[1:]
OUTPUT_NAMES = {
    "overlap": "abstention_overlap.csv",
    "pairwise": "abstention_pairwise.csv",
    "evidence_coverage": "abstention_evidence_coverage.csv",
    "question_details": "abstention_question_details.csv",
    "question_patterns": "abstention_question_patterns.csv",
    "support_summary": "abstention_support_summary.csv",
    "categories": "abstention_category_summary.csv",
}
PROVENANCE_NAME = "abstention_analysis_manifest.json"
EXACT_PHRASES = frozenset({
    "insufficient evidence in the retrieved context", "insufficient evidence",
    "i don't know", "i do not know", "cannot determine from the provided information",
})
SUPPORT_CAPTION = (
    "Model-judged factual-unit support and hallucination proxies. Context refers "
    "to the final retrieved chunks; gold refers to annotated FinanceBench evidence. "
    "Scored-answer counts are reported separately for each evidence source."
)


def is_exact_abstention(answer: str) -> bool:
    """Mirror the frozen Stage 17 detector, including its punctuation policy."""
    return answer.strip().lower().rstrip(".") in EXACT_PHRASES


def keyed_csv(path: Path, run_id: str, ids: list[str],
              architectures: tuple[str, ...]) -> dict[tuple[str, str], dict]:
    expected = {(a, q) for a in architectures for q in ids}
    result = {}
    for row in read_csv(path):
        key = row["architecture"], row["question_id"]
        if row["run_id"] != run_id or key in result:
            raise ValueError(f"Duplicate or wrong-run row in {path}: {key}")
        result[key] = row
    if set(result) != expected:
        raise ValueError(f"Incomplete or extra architecture/question rows in {path}")
    return result


def flag(value: str | int) -> int:
    if str(value) not in ("0", "1"):
        raise ValueError(f"Invalid binary flag: {value!r}")
    return int(value)


def validate_support(row: dict, architecture: str) -> None:
    """Check applicability, unit counts, and the support/proxy complement."""
    abstained = flag(row["abstained"])
    for source, support_key, proxy_key in (
        ("context", "context_faithfulness", "context_hallucination"),
        ("gold", "gold_evidence_support", "gold_evidence_hallucination"),
    ):
        count, supported = row[f"{source}_claim_count"], row[f"{source}_supported_count"]
        support, proxy = row[support_key], row[proxy_key]
        if architecture == "P0" and source == "context":
            if any(v != "" for v in (count, supported, support, proxy)):
                raise ValueError("P0 context scores must be NA")
            continue
        if count == supported == support == proxy == "" and abstained:
            continue
        n, s = int(count), int(supported)
        if n < 0 or not 0 <= s <= n or (abstained and n):
            raise ValueError("Invalid factual-unit counts or scored exact abstention")
        if not n:
            if support != "" or proxy != "":
                raise ValueError("Answers without scored factual units must have NA scores")
            continue
        a, b = float(support), float(proxy)
        if not (math.isclose(a, s / n, abs_tol=1e-12)
                and math.isclose(b, 1 - a, abs_tol=1e-12)):
            raise ValueError("Support/proxy scores disagree with factual-unit counts")


def page_refs(candidates: list[dict]) -> set[tuple[str, int]]:
    refs = set()
    for candidate in candidates:
        page = candidate["page"]
        if (not isinstance(candidate["doc_name"], str) or not candidate["doc_name"]
                or isinstance(page, bool) or not isinstance(page, int) or page < 0):
            raise ValueError("Invalid zero-indexed document/page reference")
        refs.add((candidate["doc_name"], page))
    return refs


def chunk_ids(candidates: list[dict]) -> set[str]:
    result = {c["chunk_id"] for c in candidates}
    if len(result) != len(candidates):
        raise ValueError("Duplicate chunks in a retrieval stage")
    return result


def retrieval_diagnostics(record: dict, gold: set[tuple[str, int]], saved: dict) -> dict:
    arch = record["architecture"]
    final = record["retrieved_chunks"]
    diagnostics = record["retrieval_diagnostics"]
    pre = diagnostics["first_stage_candidates"]
    union = pre if arch == "P1" else diagnostics["rrf_union"]
    def identity_order(stage: list[dict]) -> list[tuple]:
        return [(c["chunk_id"], c["doc_name"], c["page"]) for c in stage]

    if arch != "P1" and identity_order(diagnostics["rrf_top_candidates"]) != identity_order(pre):
        raise ValueError("Saved RRF selection differs from pre-final candidates")
    if identity_order(diagnostics["final_candidates"]) != identity_order(final):
        raise ValueError("Saved diagnostic final candidates differ from final context")
    selection = diagnostics["reranked_candidates"] if arch == "P3" else pre
    if arch == "P3" and (chunk_ids(selection) != chunk_ids(pre)
                         or set(identity_order(selection)) != set(identity_order(pre))):
        raise ValueError("Reranked candidates must preserve the pre-final candidate set")
    if identity_order(final) != identity_order(selection[:5]):
        raise ValueError("Final context differs from the saved top-five selection")
    if not chunk_ids(final) <= chunk_ids(pre) <= chunk_ids(union):
        raise ValueError("Final/pre-final/union candidates are not nested")
    # Check document/page identity too: a reused chunk ID must not change pages.
    identities = {}
    for stage in (union, pre, final):
        page_refs(stage)
        for c in stage:
            identity = (c["doc_name"], c["page"])
            if c["chunk_id"] in identities and identities[c["chunk_id"]] != identity:
                raise ValueError("Chunk page identity changed across retrieval stages")
            identities[c["chunk_id"]] = identity
    union_hit, pre_hit, final_hit = [int(bool(page_refs(s) & gold))
                                    for s in (union, pre, final)]
    if (not gold or len(final) > 5 or int(saved["retrieved_chunk_count"]) != len(final)
            or int(saved["gold_page_count"]) != len(gold)
            or flag(saved["hit_at_5"]) != final_hit):
        raise ValueError(f"Saved final-context retrieval score mismatch: {arch}")
    if not final:
        category = "empty_retrieval"
    elif not union_hit:
        category = "gold_page_absent_candidate_union"
    elif not pre_hit:
        category = "gold_page_lost_before_pre_final"
    elif not final_hit:
        category = "gold_page_lost_at_final_selection"
    else:
        category = "final_gold_page_hit"
    return {
        "diagnostic_category": category, "final_gold_page_hit": final_hit,
        "empty_retrieval": int(not final), "candidate_union_gold_page_hit": union_hit,
        "pre_final_gold_page_hit": pre_hit, "gold_page_count": len(gold),
        "candidate_union_chunk_count": len(union), "pre_final_chunk_count": len(pre),
        "final_chunk_count": len(final),
        "gold_page_refs_json": json.dumps(sorted(gold)),
        "final_chunk_ids_json": json.dumps([c["chunk_id"] for c in final]),
        "final_page_refs_json": json.dumps(sorted(page_refs(final))),
    }


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def build_tables(manifest: dict, raw: dict, faith: dict, retrieval: dict,
                 evidence: dict, native: dict) -> dict[str, list[dict]]:
    ids, run_id = manifest["question_ids"], manifest["run_id"]
    details, wide = [], []
    patterns = defaultdict(list)
    for position, qid in enumerate(ids):
        question = native[qid]
        pattern_row = {"run_id": run_id, "question_id": qid,
                       "question": question["question"], "gold_answer": question["answer"]}
        flags = []
        for arch in ARCHITECTURES:
            answer, score = raw[arch][position], faith[arch, qid]
            if (answer["question"] != question["question"]
                    or answer["gold_answer"] != question["answer"]):
                raise ValueError(f"Raw/native question or gold answer mismatch: {arch} {qid}")
            validate_support(score, arch)
            abstained = flag(score["abstained"])
            if abstained != is_exact_abstention(answer["generated_answer"]):
                raise ValueError(f"Saved abstention flag disagrees with answer: {arch} {qid}")
            row = {
                "run_id": run_id, "question_id": qid, "architecture": arch,
                "question": question["question"], "gold_answer": question["answer"],
                "generated_answer": answer["generated_answer"],
                "company": question["company"], "question_type": question["question_type"],
                "question_reasoning": question.get("question_reasoning") or "unannotated",
                "exact_abstention": abstained,
            }
            for source, support_key, proxy_key in (
                ("context", "context_faithfulness", "context_hallucination"),
                ("gold", "gold_evidence_support", "gold_evidence_hallucination"),
            ):
                support = score[support_key]
                row[f"{source}_factual_unit_count"] = score[f"{source}_claim_count"]
                row[f"{source}_support"] = float(support) if support != "" else ""
                row[f"{source}_hallucination_proxy"] = (
                    float(score[proxy_key]) if score[proxy_key] != "" else "")
                row[f"{source}_score_status"] = (
                    "not_applicable_no_context" if arch == "P0" and source == "context"
                    else "exact_abstention" if abstained else "scored" if support != ""
                    else "no_scored_factual_units")
            if arch == "P0":
                row.update({key: "" for key in (
                    "final_gold_page_hit", "empty_retrieval", "candidate_union_gold_page_hit",
                    "pre_final_gold_page_hit", "gold_page_count", "candidate_union_chunk_count",
                    "pre_final_chunk_count", "gold_page_refs_json", "final_page_refs_json")})
                row.update(diagnostic_category="no_retrieval_baseline", final_chunk_count=0,
                           final_chunk_ids_json="[]")
            else:
                mapping = evidence[qid]
                if not mapping["mapping_complete"]:
                    raise ValueError(f"Incomplete gold mapping: {qid}")
                row.update(retrieval_diagnostics(answer, page_refs(mapping["gold_page_refs"]),
                                                 retrieval[arch, qid]))
                flags.append(str(abstained))
            details.append(row)
            pattern_row[f"{arch.lower()}_abstained"] = abstained
            pattern_row[f"{arch.lower()}_generated_answer"] = answer["generated_answer"]
        pattern = "".join(flags)
        patterns[pattern].append(qid)
        pattern_row["pattern_p1_p2_p3"] = pattern
        wide.append(pattern_row)
    overlap = [{"run_id": run_id, "pattern_p1_p2_p3": f"{p:03b}",
                "p1_abstained": int(bool(p & 4)), "p2_abstained": int(bool(p & 2)),
                "p3_abstained": int(bool(p & 1)), "question_count": len(patterns[f"{p:03b}"]),
                "question_rate": len(patterns[f"{p:03b}"]) / len(ids),
                "question_ids_json": json.dumps(patterns[f"{p:03b}"])} for p in range(8)]
    pairwise = []
    for a, b in itertools.combinations(RAG_ARCHITECTURES, 2):
        counts = Counter((flag(faith[a, q]["abstained"]), flag(faith[b, q]["abstained"]))
                         for q in ids)
        union = len(ids) - counts[0, 0]
        pairwise.append({
            "run_id": run_id, "architecture_a": a, "architecture_b": b,
            "question_count": len(ids), "both_abstained": counts[1, 1],
            "a_only_abstained": counts[1, 0], "b_only_abstained": counts[0, 1],
            "neither_abstained": counts[0, 0], "either_abstained": union,
            "jaccard_abstentions": counts[1, 1] / union if union else "",
            "disagreement_rate": (counts[1, 0] + counts[0, 1]) / len(ids),
        })
    coverage = []
    for arch, abstained in itertools.product(RAG_ARCHITECTURES, (0, 1)):
        group = [d for d in details if d["architecture"] == arch
                 and d["exact_abstention"] == abstained]
        misses = [d for d in group if not d["final_gold_page_hit"]]
        row = {
            "run_id": run_id, "architecture": arch, "exact_abstention": abstained,
            "question_count": len(group), "question_rate": len(group) / len(ids),
            "final_gold_page_hit": len(group) - len(misses),
            "no_final_gold_page_hit": len(misses),
            "empty_retrieval": sum(d["empty_retrieval"] for d in group),
            "gold_page_absent_candidate_union": sum(not d["candidate_union_gold_page_hit"] for d in misses),
            "gold_page_lost_before_pre_final": sum(d["candidate_union_gold_page_hit"]
                                                   and not d["pre_final_gold_page_hit"] for d in misses),
            "gold_page_lost_at_final_selection": sum(d["pre_final_gold_page_hit"] for d in misses),
            "final_gold_page_hit_rate": mean([d["final_gold_page_hit"] for d in group]),
        }
        coverage.append(row)
    support_summary = []
    for arch in ARCHITECTURES:
        selected = [d for d in details if d["architecture"] == arch]
        n_abstained = sum(d["exact_abstention"] for d in selected)
        row = {"run_id": run_id, "architecture": arch, "question_count": len(ids),
               "exact_abstentions": n_abstained, "exact_abstention_rate": n_abstained / len(ids),
               "non_abstentions": len(ids) - n_abstained}
        for source in ("context", "gold"):
            applicable = [d for d in selected if d[f"{source}_support"] != ""]
            row[f"{source}_scored_n"] = len(applicable)
            row[f"{source}_unscored_non_abstentions"] = (
                "" if source == "context" and arch == "P0"
                else len(ids) - n_abstained - len(applicable))
            row[f"{source}_support"] = mean([d[f"{source}_support"] for d in applicable])
            row[f"{source}_hallucination_proxy"] = mean(
                [d[f"{source}_hallucination_proxy"] for d in applicable])
        protocols = {(faith[arch, q]["judge_model"], faith[arch, q]["judge_protocol"]) for q in ids}
        if len(protocols) != 1:
            raise ValueError(f"Mixed judge protocols for {arch}")
        row["judge_model"], row["judge_protocol"] = protocols.pop()
        support_summary.append(row)
    categories = []
    for arch in ARCHITECTURES:
        for grouping in ("question_type", "question_reasoning"):
            groups = defaultdict(list)
            for d in details:
                if d["architecture"] == arch:
                    groups[d[grouping]].append(d)
            for category, group in sorted(groups.items()):
                n = sum(d["exact_abstention"] for d in group)
                categories.append({"run_id": run_id, "architecture": arch, "grouping": grouping,
                                   "category": category, "question_count": len(group),
                                   "exact_abstentions": n, "non_abstentions": len(group) - n,
                                   "exact_abstention_rate": n / len(group)})
    return dict(overlap=overlap, pairwise=pairwise, evidence_coverage=coverage,
                question_details=details, question_patterns=wide,
                support_summary=support_summary, categories=categories)


def analyze(manifest_path: Path = MANIFEST, *, root: Path = ROOT) -> dict:
    manifest, raw = load_run(manifest_path)
    ids, run_id = manifest["question_ids"], manifest["run_id"]
    inputs = {
        "analysis_source": Path(__file__).resolve(),
        "question_source": Path(manifest["question_source"]),
        "faithfulness": root / "results/evaluation/faithfulness_metrics.csv",
        "faithfulness_summary": root / "results/aggregated/faithfulness_summary.csv",
        "faithfulness_protocol": root / "results/evaluation/faithfulness_evaluation_manifest.json",
        "retrieval": root / "results/evaluation/retrieval_metrics.csv",
        "evidence_mapping": root / "data/processed/evidence_mapping.json",
        **{f"raw_{a}": Path(manifest["outputs"][a]) for a in ARCHITECTURES},
    }
    protocol = json.loads(inputs["faithfulness_protocol"].read_text(encoding="utf-8"))
    if protocol["run_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Faithfulness evaluation used a different frozen run")
    for name, path in [(a, inputs[f"raw_{a}"]) for a in ARCHITECTURES] + [
        ("gold_evidence", inputs["evidence_mapping"])
    ]:
        if sha256(path) != protocol["inputs"][name]["sha256"]:
            raise ValueError(f"Frozen evaluation input changed: {name}")
    faith = keyed_csv(inputs["faithfulness"], run_id, ids, ARCHITECTURES)
    retrieval = keyed_csv(inputs["retrieval"], run_id, ids, RAG_ARCHITECTURES)
    native = {r["financebench_id"]: r for r in load_jsonl(inputs["question_source"])}
    evidence = json.loads(inputs["evidence_mapping"].read_text(encoding="utf-8"))
    if not set(ids) <= set(evidence):
        raise ValueError("Gold evidence mapping misses evaluation questions")
    tables = build_tables(manifest, raw, faith, retrieval, evidence, native)
    # Reproduce the published conditional support means and scored denominators.
    summaries = read_csv(inputs["faithfulness_summary"])
    if len(summaries) != 4 or {s["architecture"] for s in summaries} != set(ARCHITECTURES):
        raise ValueError("Published support summary must cover P0-P3 once")
    saved = {s["architecture"]: s for s in summaries}
    checks = {"exact_abstentions": "abstention_count", "context_scored_n": "context_applicable_count",
              "gold_scored_n": "gold_applicable_count", "context_support": "context_faithfulness",
              "context_hallucination_proxy": "context_hallucination",
              "gold_support": "gold_evidence_support", "gold_hallucination_proxy": "gold_evidence_hallucination"}
    for row in tables["support_summary"]:
        published = saved[row["architecture"]]
        if published["run_id"] != run_id or int(published["question_count"]) != len(ids):
            raise ValueError("Published support summary belongs to a different question set")
        for derived_key, saved_key in checks.items():
            a, b = row[derived_key], published[saved_key]
            if (a == "") != (b == "") or (a != "" and not math.isclose(float(a), float(b), abs_tol=1e-12)):
                raise ValueError(f"Derived support summary disagrees with published {saved_key}")
    return {"manifest": manifest, "inputs": inputs, "tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/aggregated")
    parser.add_argument("--check-only", action="store_true", help="Validate and summarize without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this analysis's derived outputs")
    args = parser.parse_args()
    paths = {k: args.output_dir / name for k, name in OUTPUT_NAMES.items()}
    provenance = args.output_dir / PROVENANCE_NAME
    if not args.check_only and not args.overwrite and any(p.exists() for p in [*paths.values(), provenance]):
        parser.error("Derived analysis exists; use --overwrite to replace this analysis only")
    result = analyze(args.manifest)
    tables = result["tables"]
    print("Abstention overlap (P1/P2/P3, 1=exact abstention):",
          {r["pattern_p1_p2_p3"]: r["question_count"] for r in tables["overlap"]})
    for row in tables["evidence_coverage"]:
        print(row)
    if args.check_only:
        print("All inputs and derived denominators validated; no files written.")
        return
    # Only these new paths may be written; frozen inputs remain read-only.
    protected = {p.resolve() for p in [args.manifest, *result["inputs"].values()]}
    if any(p.resolve() in protected for p in [*paths.values(), provenance]):
        raise ValueError("An output path aliases a frozen input")
    for name, rows in tables.items():
        write_csv(paths[name], rows, list(rows[0]))
    payload = {
        "title": "Post-hoc descriptive abstention overlap and evidence diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": result["manifest"]["run_id"],
        "run_manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
        "inputs": {name: {"path": str(p.resolve()), "sha256": sha256(p)} for name, p in result["inputs"].items()},
        "outputs": {name: {"path": str(p.resolve()), "sha256": sha256(p)} for name, p in paths.items()},
        "parameters": {
            "question_count": result["manifest"]["question_count"],
            "architectures": list(ARCHITECTURES), "overlap_architectures": list(RAG_ARCHITECTURES),
            "exact_abstention_phrases": sorted(EXACT_PHRASES),
            "normalization": "strip whitespace, lowercase, strip trailing periods; frozen Stage 17 policy",
            "frozen_retrieval_settings": result["manifest"]["settings"],
            "support_table_caption": SUPPORT_CAPTION,
            "support_aggregation": "question-level macro mean over answers with at least one scored factual unit; same applicable set for each support/proxy pair",
            "hallucination_proxy": "1 - support; unsupported factual units, not independently verified false answers",
            "coverage_denominator": "question_count within architecture and exact_abstention group; question_rate uses the full frozen question set",
            "missing_native_reasoning": "explicit unannotated category; no inferred labels",
        },
        "interpretation": [
            "Diagnostic categories are observational and do not identify causal reasons for abstention.",
            "A gold-page hit does not prove that the selected chunk contains sufficient answer-bearing evidence; a miss does not rule out alternative valid evidence.",
            "No exact abstention does not guarantee a substantive or correct answer; partial refusals and nonfactual responses may remain.",
            "Observed candidate pools have different budgets: P1 dense top5; P2 dense10/sparse10, RRF top5; P3 dense20/sparse20, RRF top20, reranked top5.",
            "The three missing-page stage counts partition no_final_gold_page_hit; empty_retrieval overlaps those counts and must not be added again.",
            "Support/proxy pairs sum to one by definition. Context and gold refer to different evidence and must not be added together.",
            "Conditional support means exclude exact abstentions and other answers without scored factual units; different architectures/sources can have different applicable sets.",
        ],
    }
    temporary = provenance.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(provenance)
    print(f"Saved {len(paths)} CSVs and provenance in {args.output_dir}")


if __name__ == "__main__":
    main()
