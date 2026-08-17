# C1 native-ligand control — result, and a failed prediction

Run 2026-08-17 on all 133 external targets, **before any generated molecule was analysed.**

## G2 re-confirmed

Median native Zn–donor distance **1.992 Å** (5–95%: 1.802–2.341), inside the registered
1.9–2.3 Å window. The coordinate frame is correct; no transform is required.

## Registered predictions vs observed

| quantity | registered | observed | verdict |
|---|---|---|---|
| valid-coordination rate | > 60% | **75.9%** | holds |
| primary violation rate | **< 5%** | **19.5%** | **FAILED** |
| V1 hard clash | — | 0.0% | — |
| V2 shell occupancy | — | 19.5% | — |
| V3 malformed donor | — | 56.4% | — |

**The registered <5% prediction for native ligands is wrong.** Reported as a failed
prediction, not rationalised away.

## Diagnosis — the threshold, not the frame

`ANALYSIS_PLAN.md` §9 anticipated this: *"C1 shows native ligands also violate at high rates
→ thresholds are too strict, or the frame mapping is subtly wrong despite G2."* The frame is
verified correct, so it is the threshold.

All 26 violating natives are V2 (non-donor in shell), never V1. Every offending atom is
**carbon** (28 instances), at median **2.63 Å**, tightly clustered in 2.47–2.66 Å — just
inside the 2.70 Å shell. **92% co-occur with a valid coordinating donor in the same
molecule.**

That is the geometric signature of chelation: in a bidentate carboxylate or hydroxamate, the
central carbon sits ~2.5–2.7 Å from the metal *because* its two oxygens coordinate at ~2.0 Å.
The carbon's position is forced by the coordination it enables. Counting it as "occupying the
metal's excluded volume" is a threshold artefact, not a chemical finding.

V3 at 56.4% has the same root: chelating donors often place one oxygen inside the accepted
range and its partner slightly outside.

## Consequence for the primary analysis — the design survives

The primary analysis was always **generated vs native, paired by target** (GLMM, §7). A
non-zero native rate does not invalidate that contrast; it calibrates it. C1 exists precisely
to establish the ceiling, and it has done so — the ceiling is 19.5%, not the ~0% assumed.

What changes: the *absolute* primary rate cannot be read as "how often the model behaves as
though the metal were absent", because ~20% of that rate is chelation geometry that real
ligands exhibit too. Only the generated-minus-native contrast supports that reading.

**No threshold is changed.** V1, V2, V3 and the coordination ranges stand exactly as
registered.
