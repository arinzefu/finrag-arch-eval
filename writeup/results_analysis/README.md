# Updated Results and Analysis

`results_and_analysis.tex` is the drop-in replacement for the supplied Results and Analysis, Discussion, and Limitations and Conclusion sections. Include it in the thesis with `\input{writeup/results_analysis/results_and_analysis}` (adjust the path for the thesis project), replacing the previous sections. The thesis preamble must load `graphicx`, `float`, and `amsmath`. Check section numbering and table labels when integrating it.

`main.tex` is a standalone preview wrapper. It adds an article document class, page geometry and preview formatting; do not paste that wrapper into an existing thesis.

`terminologies.tex` expands the supplied terminology section with the answer-quality, abstention and statistical terms used in the revised results. It retains the original evaluation and system table labels and adds tables for statistics, answer quality and abstention. Copy this section into the thesis or include it with `\input{writeup/results_analysis/terminologies}`, adjusting the path. The preview places this glossary before the results. Its fixed-width columns wrap definitions rather than shrinking the tables with `\resizebox`; the glossary itself requires only `float` for `[H]` placement.

## Data provenance

All paths below are relative to the repository root. The text describes the saved `financebench-heldout-v1` run of 145 questions across P0–P3; the additional evaluations did not regenerate answers.

- `results/aggregated/answer_summary_v2.csv` and `results/evaluation/answer_metrics_v2.csv`: current semantic, reference-coverage, BERTScore, lexical and numeric results, including valid denominators, unresolved judge-error counts and correctness bounds.
- `results/evaluation/answer_quality_evaluation_manifest.json`: evaluation configuration and hashes linking the current answer scores to their source records.
- `results/evaluation/answer_reference_units_v3.jsonl` and `results/evaluation/answer_quality_judgments_v3.jsonl`: shared reference rubrics and response-level semantic evaluation records.
- `results/evaluation/answer_quality_v3_audit.md` and its companion CSV: selected observed rubric and judge errors.
- `results/aggregated/abstention_*.csv` and `results/aggregated/abstention_analysis_manifest.json`: exact-abstention counts, overlap, retrieval-stage conditions, subgroup summaries and provenance.
- `results/aggregated/retrieval_summary.csv`, `results/aggregated/latency_summary.csv`, `results/aggregated/abstention_support_summary.csv` and `results/statistics/hypothesis_tests.csv`: retained retrieval, timing, factual-unit support and paired-comparison results.
- `results/raw_runs/financebench-heldout-v1_P0.jsonl` through `financebench-heldout-v1_P3.jsonl`, and `data/processed/evidence_mapping.json`: saved answers, retrieved contexts and annotated evidence used to check examples.

## Interpretation

The semantic, reference-coverage, BERTScore and expanded abstention analyses are post hoc and descriptive. GPT-4o-mini generated one reference rubric per question without seeing pipeline answers; it judged the 401 non-exact-abstention responses. The 179 exact abstentions received deterministic zero scores. There are 390 accepted model judgments and 11 unresolved errors, so semantic means use 569 valid results with explicit per-pipeline denominators. Correctness bounds account for unresolved judgments only; they are not confidence intervals and do not account for accepted judge mistakes.

The qualitative audit is a convenience inspection by the code assistant, not independent human validation. It found compatible answers rejected and optional details treated as required. Reference-unit coverage is therefore a completeness proxy. BERTScore is similarity, and the factual-unit hallucination proxies measure unsupported units rather than independently verified false answers. Annotated-page hits do not establish passage sufficiency, and observed retrieval conditions are not validated internal causes of refusal. The existing adjusted statistical tests do not apply to the newly added metrics or abstention subgroups.
