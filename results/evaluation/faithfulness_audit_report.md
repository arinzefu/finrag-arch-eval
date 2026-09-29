# Stage 17: Preselected v4 Judgment Audit

## Scope

The eight question IDs were fixed in `faithfulness_audit_selection.json` by a
hash of IDs before individual v4 judgments were inspected. This excludes the
five pilot questions used while developing the judge. The sample covers 32
P0-P3 answers and their 56 gold/context judgment records (P0 has no context
judgment). Answers, cited unit decisions, and the annotated evidence text were
checked against the frozen source. This is a qualitative spot check, not an
estimate of judge error prevalence or a replacement for a full human review.

| Question ID | Manual observation |
| --- | --- |
| `financebench_id_10285` | **Clear false positive.** Boeing's FY2018 PP&E is $12,645 million, while $12,672 million is the FY2017 column in the same balance-sheet row. P3 answered $12,672 million. The v4 judge marked this unit supported against both gold evidence and retrieved context because the number occurs in the cited table; the numeric-presence guard did not check column-year alignment. P1/P2 answered $12,645 million. |
| `financebench_id_03849` | P1/P3 abstained. P2's longer answer ended with "Insufficient evidence in the retrieved context." The gold-evidence judge marked that sentence as a supported factual unit, even though the annotated statements contain the inputs for the 7.9% answer. The exact-abstention rule did not classify this mixed answer as an abstention. P0 supplied no result and had no scored factual units. |
| `financebench_id_04854` | P1-P3 answered $3,215.4 million for General Mills FY2020 free cash flow. The annotated cash-flow statement gives 3,676.2 less 460.8 = 3,215.4; the gold answer is rounded to $3,215.00. Support is reasonable, while strict numeric equality may penalize the rounding difference. P0 gave no number. |
| `financebench_id_01935` | P0 described an acquisition that is not the filing's agenda and was marked unsupported. P1 abstained. P2/P3 identified the Amcor supplemental indentures, consistent with the annotated filing, and were marked supported. |
| `financebench_id_10130` | All retrieval systems abstained on Corning DPO despite a numerical gold answer. P0 restated the formula and requested input values; its two generic operand descriptions were marked gold-supported, although it never answered 63.86. This illustrates that evidence support is not answer completeness. |
| `financebench_id_01009` | The generated PepsiCo geography lists are partial or use different granularity from the gold list. Some were marked context-supported while gold support was NA or lower. A faithful statement about selected countries does not necessarily answer the requested complete geographic scope. |
| `financebench_id_00476` | The annotated filing lists no registered debt securities. P0 invented generic senior notes/bonds, but the judge labeled its three answer units nonfactual and assigned no gold-support score: a **clear missed hallucination**. P1-P3 abstained despite the evidence supporting "none." |
| `financebench_id_01912` | P1-P3 identified MGM China and the 44% revenue decline, consistent with the annotated evidence. P0 said "Asia-Pacific" instead of the named MGM China region, yet was marked supported; this is at least an imprecise entity match. |

## Interpretation

Table F1 is a **model-assisted unit-level evidence-support measure**, not an
independently verified accuracy or hallucination ground truth. Its rates are
conditional on non-abstaining answers with factual units and must be read with
abstention counts. The audit found both false support (notably adjacent-year
table columns) and missed factual claims, so small differences among P1-P3
faithfulness means are not by themselves decisive. Keep the v4 journal,
per-question metrics, and aggregate table unchanged; do not tune the judge
against this held-out sample and mix a new protocol into the frozen run.
