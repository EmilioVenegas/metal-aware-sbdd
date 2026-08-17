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

> **G2 result (2026-08-17, scripts/run_g2_validation.py):**
> Measured over 22 X-ray representative targets (re-downloaded CIF, minimum-distance
> Zn-ligand pair selection). N=31 coordinating donor atoms.
> - 5th pct: 1.823 Å | **Median: 2.183 Å** | 95th pct: 2.372 Å
> - **G2: PASS** (median within [1.9, 2.3] Å)
>
> One flag: **9UD7** A1EOY N08 remeasured at **1.73 Å** (< 1.75 Å covalent threshold).
> Build step had stored 1.87 Å due to non-minimum-distance Zn selection. G2's
> minimum-distance pair logic is correct. Applied pre-registered covalent exclusion rule:
> 9UD7 and its 4 cluster members (9U64, 9LZV, 9KMB, 9ISD) removed.
>
> **Final benchmark after G2: n=133 targets, m_xray=21 clusters, m_cryo=5, m_total=26.**

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

**C1 — native ligands (unpaired reference).** In the same external Zn pockets, apply the
checker to the native ligand coordinates. This is the **ceiling**: native ligands frequently
coordinate the metal directly. It also validates G2 — if the median native-ligand
metal-donor distance falls outside 1.9–2.3 Å, the frame mapping is wrong.

**C2 — protein-atom clash rate (paired within molecule).** For each generated molecule,
measure the clash rate against **ordinary protein heavy atoms** in the same pocket (same
threshold radii, same distance cutoffs). Paired: one protein-atom rate and one metal-site
rate per molecule, aggregated per cluster. Establishes the model's baseline geometric
sloppiness and ensures any metal-site signal is not simply "this model generates bad
geometry everywhere."

**C3 — burial-matched decoy control (within-pocket paired — PRIMARY).**
*This is the control that decides whether the metal framing survives.*

**Primary design (within-pocket, paired):** In the SAME metalloprotein pockets, with the
SAME generated molecules, compare occupancy of the **true metal site** against
**burial-matched decoy points** placed within that same pocket. Protocol:

1. For each target pocket, identify the true Zn position (from the raw PDB receptor).
2. Place K=5 decoy points per pocket at positions matched to the metal on:
   - burial depth (solvent-accessible surface area shell)
   - distance to pocket centroid (±0.5 Å)
   - Reject decoys within 2.0 Å of any protein heavy atom.
3. For each generated molecule, measure whether any heavy atom falls within 2.7 Å of the
   metal site AND within 2.7 Å of each decoy point. Occupancy is the per-molecule
   indicator.
4. Paired difference per cluster: `metal_occupancy − mean(decoy_occupancy)`.
5. Bootstrap over clusters (the paired difference is the unit).

Paired by design: same molecules, same pocket, only the reference point differs. This
eliminates between-pocket variability from the comparison.

**Secondary design (cross-pocket, unpaired):** Compare metal-site occupancy on metalloprotein
pockets against occupancy of burial-matched positions in metal-free pockets of similar depth.
This is the original design; it remains as a sensitivity check.

> If the within-pocket paired difference is not distinguishable from zero, then generated
> molecules do not preferentially occupy the metal site over matched buried space in the same
> pocket — the failure is not metal-specific, and the framing must change.

C3 is mandatory and is decided before any claim is made about metal-specificity.

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

## 6b. Sampling definition — Amendment 4 (2026-08-17, pre-generation)

**Blocking amendment. Fixed before any molecule is generated.**

**N = 100 *valid* molecules per target.** "Valid" is upstream DiffSBDD's own criterion:
`build_molecule` returns a molecule and RDKit sanitisation succeeds. This project does **not**
modify validity semantics — see `MODIFICATIONS.md`, which records that the predecessor fork's
`SanitizeMol` fallbacks were deliberately **not** ported precisely because they would change
this denominator.

**Recorded per target:** attempts, valid count, validity rate, and the failure mode of each
invalid sample. Sampling proceeds in batches until 100 valid molecules are obtained, with a
hard cap of **1000 attempts**; targets not reaching 100 valid within the cap are reported with
their actual N and flagged, never silently dropped.

**Pre-registered sensitivity analysis.** The primary endpoint is recomputed with invalid
samples included in the denominator (counted as non-violating, the conservative direction).

> **Rationale.** If molecules that fail to build are geometrically atypical near the metal —
> plausible, since atoms crowded into a coordination site produce strained or unbuildable
> bonding — then discarding them removes exactly the samples most likely to violate, biasing
> the endpoint toward the null in a way that is invisible in the valid-only analysis.

**Pre-registered check:** correlate per-target validity rate with per-target violation rate
across clusters. A strong negative correlation is evidence that invalidity and metal-site
occupancy share a cause, and in that case the all-attempts analysis becomes the headline
rather than the sensitivity check. This reassignment rule is fixed now, not after seeing the
correlation.

## 7. Analysis

**Primary analysis:** GLMM with cluster as a random effect (logistic link; cluster intercept
random), fit over all surviving targets within each cohort. Fixed effect of interest: arm
(generated vs. native / metal-site vs. decoy). The GLMM respects the nested structure
(molecules within targets within clusters) and yields the headline OR and 95% CI.

**Sensitivity bounds:**
- *Cluster-level bootstrap:* Aggregate violation rate per cluster; resample clusters
  (10,000 BCa bootstrap). This is the most conservative estimate — treats all within-cluster
  correlation as nuisance. Report alongside GLMM.
- *Target-level bootstrap:* Aggregate per target; resample targets. Ignores between-cluster
  structure (risks pseudo-replication in large enzyme families). Reported as a sensitivity
  check; headline is GLMM.

**Additional requirements:**
- Cryo-EM targets (m=5) analysed in a separate stratified pass; never pooled with X-ray
  primary without noting the stratum.
- Paired comparisons (C2, C3-primary) use paired bootstrap over the within-cluster
  difference, consistent with the within-pocket paired design.
- Report per-cluster and per-target values, not only the pooled mean.
- Agreement criterion (pre-registered): if GLMM OR and cluster-level bootstrap differ by
  more than 2× in the point estimate, the discrepancy is reported as evidence of
  family-level confounding and the claim is demoted.

## 8. Detection limit

Computed and recorded **before generation**, using the final filtered set.

The SE of a proportion under cluster-level bootstrap resampling is `sqrt(p(1-p)/m)`.
N (molecules per target) does not appear in this formula once N is large enough that the
within-cluster estimate is stable; N=100 is sufficient at all predicted violation rates.
MDE at 80% power, two-tailed α=0.05: `(z_α/2 + z_β) × SE = 2.802 × SE`.

**Amendment 1 (2026-08-17): Corrected MDE reporting.** The previous entry used
`SE = √(p(1−p)/m)` as if σ_d equalled the theoretical maximum for a single independent
proportion. For a paired comparison of proportions the relevant quantity is σ_d, the
between-cluster SD of the **paired difference** (D_i = arm1_rate_i − arm2_rate_i). The
theoretical bound σ_d=0.47 assumes the two arms are uncorrelated and both at p=0.5, which
is the worst case, not an operating estimate. The table below shows MDE over a range of σ_d.

### PRIMARY Cohort: Clean External Catalytic Zn (Amendment 3: resolution ≤2.8 Å)

**Final counts (post-G2):** n=133 targets, 26 clusters after same-UniProt merge; **21 X-ray clusters (primary), 5 cryo-EM (stratified subgroup).** (9UD7 cluster excluded at G2 for covalent coordination.)

MDE formula for paired comparison across m clusters: `MDE = (z_α/2 + z_β) × σ_d / √m`, where σ_d is the between-cluster SD of the paired difference (not the theoretical maximum for an independent proportion). N (molecules per target) does not appear once N≥100.

**Key:** `σ_d=0.47` is the theoretical bound (independent Bernoulli at p=0.5, uncorrelated); `σ_d=0.15–0.20` is the plausible operating range for a within-pocket paired design.

#### MDE table — Amendment 1

| σ_d | label | m=21 | m=26 |
|---|---|---|---|
| 0.15 | plausible-low (within-pocket paired) | 9.2% | 8.2% |
| 0.20 | plausible-mid | 12.2% | 11.0% |
| 0.30 | plausible-high (unpaired proxy) | 18.3% | 16.5% |
| 0.47 | theoretical bound (indep. proportions) | 28.7% | 25.8% |

#### Per-comparison power assessment

**C1: generated molecules vs native ligands**
- Expected δ ≈ 0.25 (registered: generated >30%, native <5%)
- σ_d ≈ 0.20–0.30 (unpaired, different populations)
- MDE at m=21, σ_d=0.20: **12.2%** → C1 is **well-powered** (expected δ >> MDE at all plausible σ_d)

**C2: metal-site clash vs protein-atom clash (within molecule, paired)**
- Expected δ ≈ 0.05–0.15 (metal site occupied more than typical buried protein atom)
- σ_d ≈ 0.10–0.15 (paired within same molecule; correlation reduces variance)
- MDE at m=21, σ_d=0.10: **6.1%** | σ_d=0.15: **9.2%** → C2 is **adequately powered** at plausible σ_d

**C3: metal-site occupancy vs burial-matched decoy (within-pocket paired — PRIMARY)**
- Expected δ ≈ 0.069 (registered 1.3× occupancy ratio; if metal occ=0.30, decoy=0.231)
- σ_d ≈ 0.15 (within-pocket paired; optimistic lower bound)
- MDE at m=21, σ_d=0.15: **9.2%** — exceeds expected δ of 6.9%
- m required to detect δ=0.069 at σ_d=0.15: **≥37 clusters**

> [!WARNING]
> **C3 is underpowered for the registered 1.3× effect.** With m=22 X-ray clusters and
> σ_d=0.15 (optimistic for paired within-pocket), MDE=9.0% > δ=6.9%. Power against the
> registered 1.3× alternative is <50%.
>
> **Pre-registered consequence:** If C3 shows a null result (paired difference not
> distinguishable from zero), this is **not interpretable** as absence of effect — the
> design cannot resolve the 1.3× prediction. C3 null → "unresolved" not "absent".
> A positive C3 result (occupancy ratio >1.3× detected) would be informative because
> it is harder to achieve than the MDE requires.
>
> **Mitigation recorded in advance:** The within-pocket paired design (K=5 decoys, same
> molecules) reduces σ_d substantially compared to a cross-pocket design. If the empirical
> σ_d measured from the data is ≤0.10, C3 becomes adequately powered at m=22 (MDE=5.9%).
> We will report the empirical σ_d alongside the result.

**C3 secondary (cross-pocket, unpaired):** σ_d≈0.30 expected; at m=22 MDE=17.9% — also underpowered for the 1.3× prediction. Reported as context, not evidence.

### SECONDARY Cohort: CrossDocked Catalytic Zn (m=30)
- MDE at σ_d=0.20: **10.2%** | σ_d=0.30: **15.4%** | σ_d=0.47: **24.0%**

### CONSISTENCY Cohort: Clean CrossDocked Catalytic Zn (m=3)
- MDE at σ_d=0.20: **32.4%** — directional consistency only; cannot resolve effects at m=3.

### Operating choice
N = 100 molecules per target. Increasing N does not reduce cluster-level MDE.

## 9. What would falsify the Step 1 claim

- **C3 positive:** Metal-site occupancy = decoy occupancy (paired difference ≈ 0) AND the
  empirical σ_d ≤ 0.10 (confirming the design had adequate power) → the excess occupancy at
  metal sites is not metal-specific; the framing must change.
- **C1 positive:** Native ligands violate at rates ≥10% → thresholds too strict or G2
  frame mapping subtly wrong despite passing the gate.
- **Primary endpoint low (<10%):** Model avoids metal sites in practice despite never seeing
  them during training; the premise is weaker than assumed.
- **GLMM vs cluster-bootstrap disagree >2×:** Family-level confounding; claim demoted to
  "warrants investigation in a broader, family-balanced set."

Any of these is reported as the result. None is grounds for adjusting thresholds after the
fact.
