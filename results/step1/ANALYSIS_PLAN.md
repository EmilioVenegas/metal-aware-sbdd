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

> **Pre-generation Amendment (2026-08-16 — Strict Positive-Definition Filter, Final):**
>
> All figures below supersede the earlier m=51 entry. Filters applied:
> non-polymer CCD type; not in metabolite/nucleotide/cofactor blacklist; RDKit
> drug-likeness (≥12 HA, QED>0.1, MW≤1000); X-ray resolution ≤2.5 Å (cryo-EM
> flagged separately); not a catalytic-dead mutant (title keyword); min coordinating-
> donor distance >1.75 Å (excludes covalent adducts); same-UniProt cluster merge.
>
> 1. **PRIMARY COHORT — Clean External Catalytic Zinc:**
>    - 134 targets surviving all strict filters (`data/external_zn_test_clean.pt`).
>    - **27 sequence clusters at 30% identity; 25 after same-UniProt merge.**
>    - **20 clusters are X-ray; 5 are cryo-EM (flagged separately, not counted in
>      the primary analysis).**
>    - Cluster report: `results/step1/external_zn_cluster_report.md`
>    - Strict CrossDocked independence: <30% seq identity to any training PDB.
>    - Catalytic shell: ≥2 protein sidechain donors within 2.8 Å of Zn.
>    - Coordination: ≥1 donor (N/O/S) within 2.5 Å; min coord dist >1.75 Å.
>
> 2. **SECONDARY COHORT — CrossDocked Catalytic Zinc (m=30):**
>    - All 30 catalytic Zinc targets from CrossDocked test split
>      (`data/metal_target_split.pt`), contamination status tracked per target
>      (27 seen in training, 3 clean).
>
> 3. **CONSISTENCY SUBSET — Clean CrossDocked Catalytic Zinc (m=3):**
>    - 3 clean CrossDocked Zn targets. Directional consistency only; never a
>      headline claim (MDE ~50% at m=3).
>
> 4. **Cryo-EM targets (m=5):** Evaluated and reported as a sensitivity check;
>    not pooled with X-ray primary due to lower coordination-geometry accuracy.

**Primary Endpoint:** Proportion of generated molecules exhibiting **≥1 violation of V1 or V2**
at catalytic metal sites in the PRIMARY X-ray cohort (m=20 independent clusters, n=134 targets).
Binary per molecule.

**Secondary Endpoints:**
1. Primary violation rate on SECONDARY cohort (CrossDocked Catalytic Zinc, m=30).
2. Primary violation rate across other catalytic metalloproteins (Mg, Ca, Mn, Fe, Co, Ni, Cu).
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

- Molecules pooled within target; **protein sequence cluster (at 30% identity, m=20 X-ray
  clusters) treated as the unit of resampling** for the primary analysis.
- Bootstrap over clusters, 10,000 resamples, BCa intervals, preventing pseudo-replication.
- Paired comparisons by cluster wherever arms are compared.
- Report per-cluster and per-target values, not only the pooled mean.
- Cryo-EM clusters (m=5) analysed in a separate sensitivity pass.

## 8. Detection limit

Computed and recorded **before generation**, using the final filtered set.

The SE of a proportion under cluster-level bootstrap resampling is `sqrt(p(1-p)/m)`.
N (molecules per target) does not appear in this formula once N is large enough that the
within-cluster estimate is stable; N=100 is sufficient at all predicted violation rates.
MDE at 80% power, two-tailed α=0.05: `(z_α/2 + z_β) × SE = 2.802 × SE`.

All MDE figures use `p=0.5` (maximum-variance, most conservative assumption).

### PRIMARY Cohort: Clean External Catalytic Zn

| Analysis level | m | MDE (80% power) |
|---|---|---|
| Cluster-level resampling — X-ray only (primary) | 20 | **31.3%** |
| Cluster-level resampling — all methods | 25 | **28.0%** |
| Target-level reference (ignores cluster structure) | 134 | **12.1%** |

> [!WARNING]
> **Marginal power warning (pre-registered).** The registered prediction is >30% violation
> rate. With m=20 X-ray clusters the cluster-level MDE is 31.3% — the predicted effect
> equals the MDE. Power against the registered alternative is therefore ~50%, not 80%.
>
> Consequence: a null result (violation rate <30%) is **not informative** at cluster level
> with m=20. The study is adequately powered only if the true rate exceeds ~40%.
>
> **Mitigation:** We report both cluster-level and target-level (n=134, MDE=12.1%)
> estimates. The target-level estimate treats each structure independently and risks
> pseudo-replication within enzyme families, but it can detect effects at the registered
> prediction. We pre-register that: (a) the cluster-level estimate is the headline number;
> (b) target-level is a sensitivity check; (c) agreement between the two is a prerequisite
> for any strong claim.

### SECONDARY Cohort: CrossDocked Catalytic Zn (m=30)
- MDE = **25.6%** (SE=0.0913) at 80% power

### CONSISTENCY Cohort: Clean CrossDocked Catalytic Zn (m=3)
- MDE = **80.9%** — directional consistency only; design cannot resolve effects at m=3.

### Operating choice
N = 100 molecules per target. Increasing N does not reduce cluster-level MDE.

## 9. What would falsify the Step 1 claim

- C3 shows metal sites are occupied at the rate of matched buried points → not metal-specific.
- C1 shows native ligands also violate at high rates → thresholds are too strict, or the
  frame mapping is subtly wrong despite G2.
- The primary endpoint on generated molecules is low (< 10%) → the model avoids metal sites
  in practice despite never seeing them, and the premise is weaker than assumed.
- Cluster-level and target-level estimates disagree by >2× → family-level confounding; the
  claim is demoted to "warrants further investigation."

Any of these is reported as the result. None is grounds for adjusting thresholds after the
fact.
