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

## 4. Endpoints and Cohort Structure

> **Pre-generation Amendment (2026-08-16 — Overbuilt Clean External Benchmark & Cohort Restructuring):**
>
> 1. **PRIMARY COHORT — Overbuilt Clean External Catalytic Zinc ($m = 51$ independent 30% clusters / 217 targets):**
>    - 217 high-resolution catalytic Zinc metalloenzyme structures across diverse enzyme families (`data/external_zn_test.pt`), forming **51 independent sequence clusters at 30% sequence identity** (1 representative structure per cluster used for the benchmark, or pooled across cluster members).
>    - **Strict CrossDocked Independence:** 0 hits at $\ge 30\%$ sequence identity against ALL 14,480 CrossDocked training PDBs under complete 1,000-hit pagination search.
>    - **Catalytic Shell:** $\ge 2$ protein sidechain donors (His, Asp, Glu, Cys, etc.) within $\le 2.8$ Å of Zn.
>    - **Direct Native Inhibitor Coordination:** Directly coordinated authentic drug-like inhibitor ($\ge 8$ heavy atoms, non-amino-acid, non-solvent) with a donor atom (N, O, S, P, Cl, F) within **$\le 2.5$ Å** of the catalytic Zn.
>    - Serves as the primary, unconfounded test set for all headline claims.
>
> 2. **SECONDARY COHORT — CrossDocked Catalytic Zinc ($m = 30$):**
>    - All 30 catalytic Zinc targets from the CrossDocked test split (`data/metal_target_split.pt`), with contamination status tracked per target (27 seen in base training, 3 clean).
>
> 3. **CONSISTENCY SUBSET — Clean CrossDocked Catalytic Zinc ($m = 3$):**
>    - The 3 clean CrossDocked Zn targets. Evaluated strictly for directional consistency, never as a headline claim due to detection limits ($m = 3$).
>
> 4. **INCIDENTAL EXCLUSION:**
>    - Incidental metal sites (<2 protein sidechain donors) are excluded from the primary and secondary endpoints. (In CrossDocked, 38 of 74 Mg targets are incidental crystallisation additives or weakly bound ions, which would dilute true catalytic coordination effects).
>
> 5. **CONTAMINATION BIAS CHECK (Pre-registered Hypothesis):**
>    - We compare the primary violation rate on clean ($m = 18$) vs contaminated ($m = 86$) catalytic CrossDocked targets, pooled across metals and paired by target class where possible.
>    - **Registered Prediction:** $\text{Violation Rate}_{\text{contaminated}} \le \text{Violation Rate}_{\text{clean}}$.
>    - **Rationale:** Contaminated targets were included in the base model's training set with the metal deleted but with native, metal-coordinating ligand chemistry present. Model memorisation of native ligand shape/chemistry should suppress clashes with the virtual metal position.
>    - **Consequence / Demotion Rule:** If the empirical result falsifies this prediction (i.e. if $\text{Violation Rate}_{\text{contaminated}} > \text{Violation Rate}_{\text{clean}}$), the contaminated CrossDocked cohort cannot be described as conservative, and the secondary cohort must be demoted to exploratory-only status.

**Primary Endpoint:** Proportion of generated molecules exhibiting **≥1 violation of V1 or V2** at catalytic metal sites in the PRIMARY cohort (Overbuilt Clean External Catalytic Zinc, $m = 51$ independent sequence clusters). Binary per molecule.

**Secondary Endpoints:**
1. Primary violation rate on SECONDARY cohort (CrossDocked Catalytic Zinc, $m = 30$, reporting clean vs contaminated).
2. Primary violation rate across other catalytic metalloproteins (Mg, Ca, Mn, Fe, Co, Ni, Cu; 74 targets).
3. Proportion forming ≥1 valid coordination bond.
4. V3 rate among molecules that place a donor in the shell.
5. Distributions of metal–donor distance and coordination number.

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

- Molecules pooled within target; **protein sequence cluster (at 30% sequence identity, $m = 51$) treated as the unit of resampling**.
- Bootstrap over clusters, 10,000 resamples, BCa intervals, preventing pseudo-replication.
- Paired comparisons by cluster wherever arms are compared.
- Report per-cluster and per-target values, not only the pooled mean.

## 8. Detection limit

Computed and recorded **before generation**, across restructured cohorts:

### 1. PRIMARY Cohort: Overbuilt Clean External Catalytic Zn ($m = 51$ independent 30% clusters / 217 targets)
- **Cluster-level resampling ($m = 51$ independent clusters):**
  - $N = 100$: $\text{MDE} = \mathbf{6.41\%}$ (delta = 0.0641, SE = 0.0232)
  - $N = 250$: $\text{MDE} = \mathbf{6.17\%}$ (delta = 0.0617, SE = 0.0223)
  - $N = 500$: $\text{MDE} = \mathbf{6.09\%}$ (delta = 0.0609, SE = 0.0220)
- **Target-level reference ($m = 217$ targets):**
  - $N = 100$: $\text{MDE} = \mathbf{3.06\%}$ (delta = 0.0306, SE = 0.0110)
  - $N = 250$: $\text{MDE} = \mathbf{2.95\%}$ (delta = 0.0295, SE = 0.0106)
  - $N = 500$: $\text{MDE} = \mathbf{2.91\%}$ (delta = 0.0291, SE = 0.0105)

### 2. SECONDARY Cohort: All CrossDocked Catalytic Zn ($m = 30$)
- $N = 100$: $\text{MDE} = \mathbf{8.48\%}$ (delta = 0.0848, SE = 0.0307)
- $N = 250$: $\text{MDE} = \mathbf{8.16\%}$ (delta = 0.0816, SE = 0.0296)
- $N = 500$: $\text{MDE} = \mathbf{8.05\%}$ (delta = 0.0805, SE = 0.0292)

### 3. CONSISTENCY Cohort: Clean CrossDocked Catalytic Zn ($m = 3$)
- $N = 100$: $\text{MDE} = \mathbf{49.62\%}$ (delta = 0.4962, SE = 0.1093)
- $N = 250$: $\text{MDE} = \mathbf{47.74\%}$ (delta = 0.4774, SE = 0.1051)
- $N = 500$: $\text{MDE} = \mathbf{47.10\%}$ (delta = 0.4710, SE = 0.1037)
*(Directional consistency only; design cannot resolve effects below ~50% with $m=3$.)*

### 4. Contamination Bias Check Cohorts:
- **Clean Catalytic CrossDocked All-Metal ($m = 18$):**
  - $N = 100$: $\text{MDE} = \mathbf{11.23\%}$ (delta = 0.1123, SE = 0.0388)
  - $N = 250$: $\text{MDE} = \mathbf{10.80\%}$ (delta = 0.1080, SE = 0.0373)
  - $N = 500$: $\text{MDE} = \mathbf{10.66\%}$ (delta = 0.1066, SE = 0.0368)
- **Contaminated Catalytic CrossDocked All-Metal ($m = 86$):**
  - $N = 100$: $\text{MDE} = \mathbf{4.90\%}$ (delta = 0.0490, SE = 0.0174)
  - $N = 250$: $\text{MDE} = \mathbf{4.71\%}$ (delta = 0.0471, SE = 0.0167)
  - $N = 500$: $\text{MDE} = \mathbf{4.65\%}$ (delta = 0.0465, SE = 0.0165)
- **Full Catalytic CrossDocked All-Metal ($m = 104$):**
  - $N = 100$: $\text{MDE} = \mathbf{4.44\%}$ (delta = 0.0444, SE = 0.0157)
  - $N = 250$: $\text{MDE} = \mathbf{4.28\%}$ (delta = 0.0428, SE = 0.0151)
  - $N = 500$: $\text{MDE} = \mathbf{4.22\%}$ (delta = 0.0422, SE = 0.0149)

### Operating Choice
We select $\mathbf{N = 100}$ molecules per target.
- For the PRIMARY overbuilt clean external Zinc cohort under cluster-level resampling ($m = 51$ independent clusters), $N = 100$ yields $\text{MDE} = \mathbf{6.41\%}$, providing substantial margin below the minimum claimable effect threshold ($\sim 10\text{--}15\%$).
- Increasing $N$ from 100 to 250 reduces MDE by only $0.24$ percentage points while multiplying inference compute by $2.5\times$, confirming $N = 100$ is statistically sufficient and computationally optimal within our 8 GB GPU constraint.

## 9. What would falsify the Step 1 claim

- C3 shows metal sites are occupied at the rate of matched buried points → not metal-specific.
- C1 shows native ligands also violate at high rates → thresholds are too strict, or the
  frame mapping is subtly wrong despite G2.
- The primary endpoint on generated molecules is low (< 10%) → the model avoids metal sites
  in practice despite never seeing them, and the premise is weaker than assumed.

Any of these is reported as the result. None is grounds for adjusting thresholds after the
fact.
