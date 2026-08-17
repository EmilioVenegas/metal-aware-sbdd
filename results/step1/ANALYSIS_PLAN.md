# Pre-registered analysis plan — Step 1: metal coordination failure

**Committed before any generated molecule is evaluated.** Everything below — thresholds,
endpoints, predictions, controls, stopping rules — is fixed now. Deviations are recorded as
deviations, with reasons, not silently folded in.

---

## 1. Question

Do pocket-conditioned generative models place ligand atoms into the volume occupied by a
catalytic metal, and if so, is that failure **specific to metals** or merely an instance of
generally poor geometry?

## 2. Prerequisites — hard gates, checked before any analysis

**G1 — one pocket definition.** All arms use a single pocket definition (source pipeline and
distance cutoff), confirmed to be the one the base checkpoint was trained on. Recorded
explicitly here before generation: `DiffSBDD process_crossdock.py (.npz, --no_H, --dist_cutoff 8.0 Å from crossdocked_pocket10 inputs)`.

**G2 — coordinate frame alignment.** Metal positions are recovered from raw CrossDocked
receptors and mapped into the pocket frame. The transform is validated on **native ligands**:

> Median metal–donor distance for native ligands must fall in **1.9–2.3 Å**.

If it does not, the mapping is wrong and Step 1 stops until it is fixed. No generated-molecule
number is computed or reported before G2 passes. This gate exists because every downstream
number is meaningless under a bad transform, and a bad transform is easy to not notice.

## 3. The checker — thresholds fixed now

Donor atoms are N, O, S. Non-donors are C and halogens.

**Reference coordination distances** (PDB survey ranges, per metal–donor pair):

| pair | accepted range (Å) |
|---|---|
| Zn–N | 1.9 – 2.35 |
| Zn–O | 1.85 – 2.30 |
| Zn–S | 2.15 – 2.50 |
| Mg–O | 1.95 – 2.25 |
| Mn–N/O | 2.00 – 2.35 |
| Fe–N/O | 1.95 – 2.30 |
| Ca–O | 2.25 – 2.65 |
| Cu–N/O | 1.90 – 2.30 |

**Violation definitions:**

- **V1 hard clash** — any ligand heavy atom within **1.7 Å** of the metal centre.
- **V2 shell occupancy** — a **non-donor** heavy atom within **2.7 Å** of the metal centre.
- **V3 malformed coordination** — a donor atom inside the shell but outside its accepted
  range for that metal–donor pair.

**Valid coordination** — a donor atom at a distance inside the accepted range.

**Geometry** — coordination number, and RMS angular deviation from the ideal geometry for that
coordination number (4 → tetrahedral 109.5°, 6 → octahedral 90/180°). Reported descriptively;
not part of the primary endpoint, because coordination number is ambiguous when the protein
contributes ligands.

## 4. Endpoints

> **Pre-generation Amendment (2026-08-16):**
> 1. **Primary Subgroup — Catalytic Zinc (Zn):** Zinc is the pre-specified primary endpoint subgroup.
>    All 30 Zn test targets (100.0%) are confirmed catalytic (≥2 protein sidechain donors).
> 2. **Incidental Exclusion:** Incidental metal sites (<2 protein sidechain donors; e.g. surface buffer
>    additives) are excluded from the primary endpoint. In our survey, 38 of 74 Mg test targets (51.4%)
>    are incidental crystallisation additives or weakly bound ions, which would dilute a true catalytic
>    coordination effect.
> 3. **Secondary Subgroups:** Catalytic non-zinc metalloproteins (Mg, Ca, Mn, Fe, Co, Ni, Cu with ≥2
>    sidechain donors; 74 targets), and all-metal exploratory analysis.

**Primary:** proportion of generated molecules exhibiting **≥1 violation of V1 or V2** at catalytic
metal sites in the primary Zinc subgroup (reported on clean and full target sets). Binary per molecule.
Chosen because it directly operationalises "generates as though the metal were empty space" and is
robust to the coordination-number ambiguity above.

**Secondary:**
1. Primary violation rate across catalytic non-zinc metalloprotein subgroups (Mg, Ca, Mn, Fe, etc.).
2. Proportion forming ≥1 valid coordination bond.
3. V3 rate among molecules that place a donor in the shell.
4. Distributions of metal–donor distance and coordination number.

## 5. Controls

**C1 — native ligands.** The ceiling, and the G2 validation. Native ligands of metalloenzymes
frequently coordinate the metal directly, so this also calibrates what "good" looks like.

**C2 — protein-atom clash rate, same molecules.** Clash rate against ordinary protein heavy
atoms in the same pockets. Establishes the model's baseline geometric sloppiness.

**C3 — buried pseudo-atom control.** *This is the control that could detect the positive.*
In metal-free pockets, place a virtual point at buried positions matched to real metal sites
on burial depth, and measure how often generated ligands occupy the equivalent volume.

> If generated ligands occupy real metal sites at the **same** rate they occupy arbitrary
> matched buried points, then the model is not failing at metals specifically — it is simply
> filling buried volume, and the metal framing is wrong.

C3 is mandatory. The predecessor project's governing lesson was that a buriedness baseline
reached the 98.2nd percentile where the method under test scored 52.4 against a 52.2 floor.

## 6. Predictions, registered in advance

These may be wrong. That is the point of writing them down.

| quantity | predicted |
|---|---|
| Primary endpoint, generated molecules | **> 30%** |
| Primary endpoint, native ligands (C1) | **< 5%** |
| Valid-coordination rate, generated | **< 15%** |
| Valid-coordination rate, native | **> 60%** |
| Metal-site occupancy vs matched buried points (C3) | **within 1.3×** — i.e. we expect the effect to be only partly metal-specific |

The C3 prediction is deliberately unflattering. If metal sites are occupied at close to the
rate of arbitrary buried points, the honest conclusion is that the headline is about buried
volume, not metals, and the framing changes.

## 7. Analysis

- Molecules pooled within target; **target treated as the unit of resampling**.
- Bootstrap over targets, 10,000 resamples, BCa intervals.
- Paired comparisons by target wherever arms are compared.
- Report per-target values, not only the pooled mean.

## 8. Detection limit

Computed and recorded **before generation**, across test cohorts:

### Primary Subgroup: Catalytic Zinc (Zn)
- **Clean Catalytic Zn targets ($m = 3$):**
  - $N = 100$: $\text{MDE} = 49.62\%$ (delta = 0.4962, SE = 0.1093)
  - $N = 250$: $\text{MDE} = 47.74\%$ (delta = 0.4774, SE = 0.1051)
  - $N = 500$: $\text{MDE} = 47.10\%$ (delta = 0.4710, SE = 0.1037)
  *(Note: $m=3$ has high statistical uncertainty; motivates reporting full vs clean cohorts and/or retraining Arm A on the split).*
- **All Catalytic Zn targets ($m = 30$):**
  - $N = 100$: $\text{MDE} = 8.48\%$ (delta = 0.0848, SE = 0.0307)
  - $N = 250$: $\text{MDE} = 8.16\%$ (delta = 0.0816, SE = 0.0296)
  - $N = 500$: $\text{MDE} = 8.05\%$ (delta = 0.0805, SE = 0.0292)

### Secondary Subgroups:
- **Clean Catalytic All-Metal targets ($m = 18$):**
  - $N = 100$: $\text{MDE} = 11.23\%$ (delta = 0.1123, SE = 0.0388)
  - $N = 250$: $\text{MDE} = 10.80\%$ (delta = 0.1080, SE = 0.0373)
  - $N = 500$: $\text{MDE} = 10.66\%$ (delta = 0.1066, SE = 0.0368)
- **All Catalytic All-Metal targets ($m = 104$):**
  - $N = 100$: $\text{MDE} = 4.44\%$ (delta = 0.0444, SE = 0.0157)
  - $N = 250$: $\text{MDE} = 4.28\%$ (delta = 0.0428, SE = 0.0151)
  - $N = 500$: $\text{MDE} = 4.22\%$ (delta = 0.0422, SE = 0.0149)
- **All Test Metalloproteins ($m = 148$):**
  - $N = 100$: $\text{MDE} = 3.71\%$ (delta = 0.0371, SE = 0.0132)
  - $N = 250$: $\text{MDE} = 3.57\%$ (delta = 0.0357, SE = 0.0127)
  - $N = 500$: $\text{MDE} = 3.53\%$ (delta = 0.0353, SE = 0.0125)

**Operating Choice:** $N = 100$ molecules per target.
For all catalytic Zn ($m = 30$), $N = 100$ gives an MDE of **8.48%**, which is below the minimum
scientifically meaningful effect threshold ($\sim 10\text{--}15\%$) while keeping inference runtime
within the 8 GB GPU constraint (~12–15 hours per arm). For the clean Zn subset ($m = 3$), the design
cannot resolve effects below ~50%, requiring clean vs contaminated reporting or Arm A retraining.

## 9. What would falsify the Step 1 claim

- C3 shows metal sites are occupied at the rate of matched buried points → not metal-specific.
- C1 shows native ligands also violate at high rates → thresholds are too strict, or the
  frame mapping is subtly wrong despite G2.
- The primary endpoint on generated molecules is low (< 10%) → the model avoids metal sites
  in practice despite never seeing them, and the premise is weaker than assumed.

Any of these is reported as the result. None is grounds for adjusting thresholds after the
fact.
