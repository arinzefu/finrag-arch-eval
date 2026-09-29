# Stage 17 V4 Pilot Audit: Five Held-Out Questions

Run: `financebench-heldout-v1`. Judge: `gpt-4o-mini`. Protocol:
`answer-unit-passage-v4`. This pilot contains five fixed question IDs per
architecture (35 context/gold judgment records) and is not a final estimate.

| Architecture | Exact abstentions | Context support (applicable n) | Gold support (applicable n) | Guard overrides |
| --- | ---: | ---: | ---: | ---: |
| P0 | 0 | NA (0) | 0.600 (5) | 10 |
| P1 | 4 | 1.000 (1) | 0.000 (1) | 1 |
| P2 | 2 | 0.667 (3) | 0.611 (3) | 15 |
| P3 | 2 | 0.917 (3) | 0.583 (3) | 1 |

There are 27 guard overrides: 12 financial units called nonfactual by the
model, 6 direct amounts absent from selected passages, 6 uncited arithmetic
premises, 2 missing/unknown evidence IDs, and 1 conflicting support label.
These are deliberately visible in the append-only v4 judgment journal.

## Case check: financebench_id_07966

P2 answered the Activision Blizzard capex/revenue question with revenue
figures of $3,606m (FY2017), $7,505m (FY2018), and $6,531m (FY2019). The
selected retrieved chunks contain none of these revenue values. V4 does not
credit those revenue units or their derived percentages. Against gold
evidence it also rejects the incorrect $7,505m and $6,531m figures while
crediting the cited capex values of $155m, $131m, and $116m.

The context judge incorrectly marked P2's three capex lines nonfactual. V4's
guard includes them as factual but conservatively scores them unsupported,
even though the selected context contains the values. This is a known false
negative and means the support score is not a perfect entailment measure.
P3's capex lines were marked factual and supported by its context. This
asymmetry is a reason to manually audit a preselected sample after the full
run and to report judge limitations, not to tune the evaluator further on
these held-out questions.

## Interpretation and decision

The pilot demonstrates that the previous false-positive failure (invented
revenue supported by context) is blocked. It also reveals conservative
false negatives and occasional model mistakes in factual-unit classification.
The score measures support of deterministic answer units, not atomic claims;
P0 gold support is not answer correctness. Five questions are far too few for
architecture ranking. Freeze v4 before running the remaining 140 questions;
retain this pilot, the earlier failed pilots, and all guard reasons for
methodological disclosure. Do not alter retrieval or generation using these
held-out observations. Correct the separately documented Stage 16 `3M`
numeric-extraction issue with versioned outputs before final combined tables.
