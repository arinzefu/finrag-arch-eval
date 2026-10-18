# Qualitative audit of answer-quality v3

Audit snapshot: 2026-10-17T23:55:54.147875+00:00.

This is a convenience audit of saved gpt-4o-mini rubric and judgment outputs by the code assistant. Cases were selected after inspection because they exposed specific problems. It is not an independent human validation, a representative sample, or an estimate of judge error prevalence. The CSV records observed issues without manually correcting any answer score. Retries and unresolved judge errors remain visible in the evaluation journals.

Whole-answer correctness is model judged. The `answer_completeness` field is best described as **model-judged reference-unit coverage (a completeness proxy)**. Minimal question-required units were intended, but the audit found optional supporting values in several generated rubrics and compatible phrasing rejected. Structural validity does not establish semantic validity.

## Observed cases

| Audit ID | Question | Architecture | Observation |
| --- | --- | --- | --- |
| AQV3-01 | `financebench_id_01858` | P0 | P0 says dividend increases continued for 'over 60 years'; annotated evidence says 65 consecutive years. The saved judge marks the answer incorrect because it does not reproduce 65. |
| AQV3-02 | `financebench_id_00591` | P0 | Question asks whether Adobe FCF conversion improved. P0 explicitly says yes/improved. The rubric embeds 143% and 156%; the judge acknowledges a correct assertion but marks it incorrect for omitting those figures. |
| AQV3-03 | `financebench_id_00807` | P0 | P0 supplies a formula and conditional instructions to evaluate 3M if its quick-ratio value is available. It gives no actual company result. The judge says it incorrectly implies a healthy profile and labels it incorrect with attempted_answer=true. |
| AQV3-04 | `financebench_id_01079` | P1;P2;P3 | The question asks FY2023/FY2022/FY2021 acquisitions; the gold answer groups the three named acquisitions under FY2023. The fixed units identify the acquisitions without fiscal-year qualifiers. P1-P3 place the Czech acquisition in FY2022. |
| AQV3-05 | `financebench_id_01079` | P1;P2;P3 | Saved invalid responses mark every fixed unit answered and has_contradiction=false, but choose partially_correct because FY2021 is not covered. This rationale requires scope absent from the fixed rubric. |
| AQV3-06 | `financebench_id_00222` | P0 | A healthy-liquidity yes/no question has a separate U2 requiring quick ratio 1.57. |
| AQV3-07 | `financebench_id_00678` | P0 | An improving-gross-margin yes/no question has five units: the conclusion, FY2022/FY2021 gross profits, and FY2022/FY2021 gross-margin percentages. |
| AQV3-08 | `financebench_id_00790` | P0 | A capital-intensity yes/no question has four units: the conclusion, two ROA values, and the fixed-assets/total-assets ratio. |
| AQV3-09 | `financebench_id_00215` | P0 | A capital-intensity yes/no question has a separate U2 requiring the 2.77 capital-intensity ratio. |
| AQV3-10 | `financebench_id_00216` | P0 | A healthy-liquidity yes/no question has a separate U2 requiring the approximately 0.54 quick ratio. |

## Consequences for interpretation

- Omitting optional supporting quantities can lower a sufficient answer's coverage or correctness score.
- Compatible paraphrases or approximations can be rejected. P0's "over 60 years" versus 65 is a concrete example.
- Incorrect/failure-to-answer labels can be confused even when their binary correctness scores are both zero.
- A question/reference scope discrepancy or missing period qualifier can affect all architectures through their shared rubric.
- Unresolved structured-judgment conflicts have missing semantic scores. Valid-judgment means and their n must be read with judge-error counts and full-question correctness bounds. Those bounds describe unresolved judgments; they do not bound undetected semantic judge errors.
- No scores were manually relabeled or replaced using these audit observations. Small architecture differences should not be treated as independently verified accuracy gains.

## Source hashes

The companion CSV includes each cited source path, its SHA-256 at this audit snapshot, and a SHA-256 for each cited record. Whole-file hashes were finalized after the evaluation journal reached all 580 architecture/question records. Future reruns can change journal files; the hashes identify this completed-run snapshot. Record hashes remain reproducible from the cited JSON record using UTF-8 JSON serialization with `sort_keys=True`, `ensure_ascii=False`, and separators `(",", ":")`. Record locators distinguish architecture, question, retry attempt, and request where applicable.

| Source | SHA-256 at audit |
| --- | --- |
| `data/processed/evidence_mapping.json` | `d8c3ee8155f047d865cc9e572a71ef949821e824c7b193e046b84b42e5eae15f` |
| `results/evaluation/answer_quality_invalid_responses_v3.jsonl` | `79feb07f642583f250cf70ec4bc73f33301d1a206955af7d07d0c21845f88f91` |
| `results/evaluation/answer_quality_judgments_v3.jsonl` | `b9e70d9da5dbb65d9d6766c4a66dcd675cc941d3698b885d147350adec6bb746` |
| `results/evaluation/answer_reference_units_v3.jsonl` | `28322cffdb5866cbc9f71b390e631b11f5df3b36983e50cea9294a9e0345ca7b` |
| `results/raw_runs/financebench-heldout-v1_P0.jsonl` | `35e69e5d0d5bb039c3ee91fa36eab874ab5799d8aad68a2a6b0d34ed255c9942` |
| `results/raw_runs/financebench-heldout-v1_P1.jsonl` | `fb15c3d6321f9fb3f534e23fee1a94f68ca36e4fd3802f054d87f8405610e676` |
| `results/raw_runs/financebench-heldout-v1_P2.jsonl` | `b117ba61634367f2e13b12043c8b5412c1250925829a1818c72c1f19c34be3b7` |
| `results/raw_runs/financebench-heldout-v1_P3.jsonl` | `bc37c4f40483786196c9f2b5c802eabbabad594f7ff9d2555ac72bfda85341e2` |

Companion CSV: `answer_quality_v3_audit.csv`; SHA-256 `affd6c3f73399b7bddc3403e46f7e3f0dc35956df5505e6e58130cf84120d076`.
