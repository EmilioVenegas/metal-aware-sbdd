# Pre-Registration Plan: Step 2 Arm D (Inference-Time Coordination Constraints & Masked Inpainting)

**Date:** 2026-08-19  
**Status:** COMMITTED BEFORE RUNNING BENCHMARK GENERATION  
**Context:** Fourth arm of the Step 2 ablation study, testing whether inference-time geometric conditioning (masked inpainting of an open-coordination donor seed) applied to the metal-aware representation (Arm C: `checkpoints/arm_c_best.ckpt`) resolves the distance calibration dilemma without requiring retraining.

---

## 1. Core Scientific Hypothesis & Objective

- **Arm D Question:** Does enforcing an explicit catalytic donor seed at inference time (at 2.05 Å along the open coordination sphere derived from catalytic protein sidechain donors) close the remaining coordination and angular gap on top of Arm C?
- **Hypothesis:** Yes. Arm C demonstrated that the metal-aware representation directs atoms toward the catalytic metal (doubling heavy-atom shell contacts to 12,166 and doubling nitrogen donors to 1,375). However, because coordinate layers were frozen, the model suffered from a distance calibration dilemma, elevating V1 hard clashes (<1.70 Å) to 11.59% and keeping angular RMSD at 25.02°. Arm D pins the coordinating donor at the chemically ideal position ($d = 2.05\text{ \AA}$) and diffuses the remainder of the ligand around it, directly repairing the distance and angular distortion.
- **Does Arm D Need Training?** **NO.** Arm D is strictly an inference-time intervention using DiffSBDD's existing RePaint inpainting algorithm (`model.ddpm.inpaint`). No weights are modified, no gradient steps are taken, and no new training datasets are required.

---

## 2. Methodology & Algorithm

1. **Model Checkpoint:** `checkpoints/arm_c_best.ckpt` (Arm C: 16-element vocabulary + LoRA).
2. **Open Coordination Vector:** For each catalytic Zn target, extract protein sidechain donor positions from `data/protein_donors.json`. Compute:
   $$\vec{u}_{\text{open}} = - \frac{\sum_{k} (\vec{x}_{\text{donor}, k} - \vec{x}_{\text{Zn}}) / \|\vec{x}_{\text{donor}, k} - \vec{x}_{\text{Zn}}\|}{\|\sum_{k} (\vec{x}_{\text{donor}, k} - \vec{x}_{\text{Zn}}) / \|\vec{x}_{\text{donor}, k} - \vec{x}_{\text{Zn}}\|\|}$$
3. **Donor Seed Placement:** Place an Oxygen donor seed (index 2 in `atom_encoder`) at:
   $$\vec{x}_{\text{seed}} = \vec{x}_{\text{Zn}} + 2.05\text{ \AA} \cdot \vec{u}_{\text{open}}$$
4. **Masked Inpainting Sampling:** Fix node 0 at $\vec{x}_{\text{seed}}$ with one-hot Oxygen, and sample the remaining $N-1$ ligand nodes using `model.ddpm.inpaint` with RePaint resampling (`resamplings = 1`).
5. **Sampling Accounting:** $N = 100$ valid molecules per target across all 133 external Zn targets ($N = 13,300$ total molecules), with per-target attempt capping (cap = 1000) and deterministic per-target seeds.

---

## 3. Evaluation Cohort & Harness

- **Cohort:** 133 external catalytic Zn targets (`data/external_zn_test_clean.pt`), split into primary X-ray stratum ($m = 21$ clusters, $n = 127$ targets) and cryo-EM stratum ($m = 5$ clusters, $n = 6$ targets).
- **Harness:** Unmodified `scripts/coordination_checker.py` scoring V1 hard clash (<1.70 Å), V2 shell occupancy (<2.70 Å), V2-strict, valid coordination (1.90–2.35 Å for Zn–N/O), and combined-sphere angular RMSD.

---

## 4. Pre-Registered Predictions — **RETRACTED 2026-08-19, BEFORE ANY DATA WAS GENERATED**

The table below is the original draft. It is **withdrawn as scientifically invalid**. It is kept
here, not deleted, per this repository's rule that corrections are dated notes rather than silent
edits. No Arm D molecule had been scored when it was withdrawn (`arm_d_generation/sdf/` empty, no
manifest — the first launch was interrupted during model warm-up).

| Metric | Arm A | Arm B | Arm C | ~~Arm D Prediction~~ | Native C1 |
|---|:---:|:---:|:---:|:---:|:---:|
| ~~Valid Coordination Rate~~ | 19.98% | 10.58% | 24.05% | ~~> 45.0%~~ | 77.17% |
| ~~Primary Violation ($V1 \lor V2$)~~ | 18.38% | 11.39% | 25.74% | ~~< 20.0%~~ | 20.47% |
| ~~Angular RMSD to Ideal (All)~~ | 25.28° | 26.90° | 25.02° | ~~< 15.0°~~ | 18.04° |
| ~~Mean Coordination Count~~ | 0.215 | 0.110 | 0.260 | ~~> 0.80~~ | 0.874 |
| ~~A $\rightarrow$ C1 Gap Closed~~ | 0.0% | -16.4% | +7.1% | ~~> 40.0%~~ | 100.0% |

### Why it is invalid: the endpoints cannot fail

Two of the four endpoints are satisfied by the seed-placement arithmetic before a single denoising
step runs. Verified in code, not argued:

1. **Valid coordination is saturated by construction.** `COORD_RANGES[("ZN","O")] = (1.85, 2.30)`
   (`coordination_checker.py:31`). The seed is placed at exactly 2.05 Å — dead centre of that
   window — so `in_range = True`, hence `has_valid_coordination = cn_ligand > 0` is `True`
   deterministically. Inpainting *fixes* the seed (`lig_fixed`), so the model cannot move it.
2. **Angular RMSD is analytically minimised by the seed formula.** `coordination_rms_angle_dev` is
   computed over the **combined** sphere (protein donors + ligand donors,
   `coordination_checker.py:110-127`). `compute_open_coordination_seed` places the seed at the
   negative normalised sum of the protein-donor unit vectors — i.e. the arg-min of angular
   deviation for the 4th vertex given 3 fixed donors. The reported angle is therefore a property
   of the crystal structure's own donor geometry, not of the model.

**Measured floor — a one-atom "molecule", no diffusion, no DiffSBDD, all 133 targets:**

| Endpoint | Seed-alone floor | Retracted Arm D threshold | Arm C | Native C1 |
|---|:---:|:---:|:---:|:---:|
| Valid coordination rate | **100.00%** | > 45.0% | 24.05% | 77.17% |
| Mean angular RMSD | **13.22°** (median 12.64°) | < 15.0° | 25.02° | 18.04° |

Both retracted thresholds are cleared by arithmetic alone, and 13.22° *beats the native ceiling of
18.04°*. A model emitting pure noise for every non-seed atom would have "passed". The endpoints had
no detection capability, violating standards 2 and 3 in `AGENTS.md` (a control that could have
detected the positive; state the detection limit before running).

**Additional evidence that the endpoint cannot discriminate chemistry.** Pilot on 9ZSN, seeding at
the same 2.05 Å along (a) the open coordination vector and (b) a *random* direction:

| | Open vector | Random vector |
|---|:---:|:---:|
| Valid coordination, **as scored** | 100% | **100%** |
| V1 hard clash | 20% | **58%** |
| Disconnected multi-fragment molecules | 50% | **100%** |
| Valid coordination, **seed excluded** | 10% | 25% |

The headline endpoint is identical for a chemically reasoned vector and a random one. The clash and
connectivity endpoints are not. That difference is what Arm D can actually measure.

---

## 4b. Corrected Framing and Endpoints — registered 2026-08-19, before any molecule is scored

**Arm D is not a competitor to Arm C on coordination rate.** Seeding hands it the anchor for free,
so any as-scored coordination comparison is meaningless. Arm D is a **decomposition arm**: it grants
the coordination anchor by construction and asks what *remains* broken. It separates two failures
that Arm C's 24.05% confounds — "could not place a donor correctly" versus "cannot build a sane
ligand around a correctly placed donor."

### Primary endpoint (replaces valid-coordination rate)

**Seed-excluded valid coordination.** The fixed seed atom is removed from the molecule and every
endpoint recomputed over the model-generated atoms only. The seed becomes part of the *conditioning*
, not part of the measurement. This is directly comparable to Arm C's 24.05% on the same denominator.

### Co-primary endpoint (not saturated, model-dependent)

**V1 hard clash rate, as scored (< 1.70 Å).** The seed cannot cause a clash — it sits at 2.05 Å.
Every clash is a secondary atom the model placed. This is Arm C's actual failure mode (11.59%) and
it is fully preserved under seeding; the pilot shows Arm D at ~20%, i.e. plausibly *worse*.

### Secondary endpoints

- **Seed incorporation:** fraction of molecules where the seed is covalently bonded into a
  recognised zinc-binding group (carboxylate / hydroxamate / thiol / imidazole / sulfonamide, the
  five SMARTS already used in `scripts/run_smarts_baseline.py`) rather than left as a bare or
  disconnected atom.
- **Connectivity:** fraction of multi-fragment molecules (`Chem.GetMolFrags`). Pilot: 50%.
- **Quality cost vs Arm C:** validity rate, QED, SA. Off-distribution conditioning is expected to
  degrade these; the size of the degradation is the cost of the intervention.

### Retired endpoints — will not be reported as Arm D results

As-scored valid coordination rate; as-scored angular RMSD; as-scored mean coordination count; and
any "A → C1 gap closed" arithmetic built on them. All are seed-determined.

## 5. Mandatory Controls

Both are paired, run on the identical cohort and seeds, and are prerequisites for reporting any Arm
D number.

- **S1 — random-vector seed (the control that could detect the positive).** Identical donor element
  and identical 2.05 Å distance, direction drawn uniformly at random instead of the open coordination
  vector (`generate_arm_d.py --seed-mode random`). If Arm D matches S1 on an endpoint, that endpoint
  measures seed placement arithmetic and is reported as uninformative. Only endpoints where Arm D
  separates from S1 are attributable to the model.

  **Why this control is required.** S2 below establishes the arithmetic floor, but with a one-atom
  molecule — a reader can object that a real generated molecule behaves differently. S1 closes that:
  full diffusion, full molecule, same model, *only the seed direction changed*. If it still scores
  100% as-scored valid coordination, the endpoint provably cannot distinguish coordination chemistry
  from "an oxygen was placed at 2.05 Å." Conversely, the endpoints where the open vector *does*
  separate (pilot: V1 clash 20% vs 58%, multi-fragment 50% vs 100%) are the ones whose gains can be
  attributed to geometric reasoning rather than to seed placement. This is the same logic as Step 1's
  C3 decoy control ("is the metal site special, or merely buried?") applied one level up: "is the
  open coordination vector special, or merely 2.05 Å?"

  **Sampling depth: $N = 25$ valid molecules/target (Arm D itself stays at $N = 100$).** S1 only has
  to resolve a *contrast*, and the contrast is evaluated at cluster level ($m = 21$ primary X-ray
  clusters), where between-cluster variance dominates the within-target binomial term. Measured on
  the real Arm C vs Arm A paired cluster contrast for `has_valid_coordination`, by subsampling the
  existing $N = 100$ checker output:

  | Control $N$ | $\sigma_d$ | SE | MDE (80% power) | Inflation vs $N{=}100$ |
  |---:|---:|---:|---:|---:|
  | 100 | 0.0750 | 0.0164 | 4.59 pp | — |
  | 50 | 0.0896 | 0.0196 | 5.48 pp | 1.19× |
  | **25 (registered)** | **0.1108** | **0.0242** | **6.77 pp** | **1.48×** |
  | 10 | 0.1633 | 0.0356 | 9.98 pp | 2.18× |

  The $N = 100$ row reproduces the SE of 1.60% reported in `ARMC_RESULTS.md` §3.1, confirming the
  decomposition is computed on the same footing as the published contrasts. **Stated plainly: $N = 25$
  inflates the minimum detectable effect by 48%, from 4.59 pp to 6.77 pp.** That is accepted, not
  waved away, because every S1 contrast the pilot points to is far larger than 6.77 pp — V1 clash
  ~38 pp, multi-fragment ~50 pp, seed-excluded coordination ~15 pp, i.e. 2.2–7.4× the MDE. Because
  Arm D runs at $N = 100$ and only S1 at $N = 25$, 6.77 pp is a conservative bound on the
  Arm D − S1 contrast rather than an exact figure.

  **Detection limit declared in advance, per standard 3 in `AGENTS.md`: any Arm D − S1 difference
  below 6.77 pp is reported as unresolved, not as a null.** If an endpoint's observed separation
  lands under that bound, S1 is re-run at $N = 100$ for that endpoint before any claim is made.
- **S2 — seed-alone floor.** The one-atom baseline above (100.00%, 13.22°), reported alongside every
  endpoint so no Arm D claim can be made without clearing its own arithmetic floor.

### Decision Rules

1. **Primary.** Arm D is judged on **seed-excluded valid coordination vs Arm C's 24.05%**, cluster-
   level, GEE with cluster as the grouping factor, same as Arms B/C. Superiority requires beating
   Arm C *and* separating from control S1.
2. **Clash.** If as-scored V1 hard clash does not fall below Arm C's 11.59%, then seed inpainting
   does **not** repair the distance-calibration failure — it relocates it to the secondary atoms.
   Pilot evidence points this way; it is registered as the expected outcome, not a surprise.
3. **Honest-negative rule.** If seed-excluded coordination is at or below Arm C's 24.05% and clash
   is at or above 11.59%, Arm D is reported as a **negative result**: guaranteeing one contact by
   construction does not produce coordination chemistry, and inference-time pinning is not a
   substitute for coordinate-level learning. This is a publishable finding and will not be
   re-framed as a partial success.

---

## Amendment 1 — cluster-stratified sampling (recorded 2026-08-19, after an interim look)

**Disclosed up front: this design change was made after an unblinded interim look at 19 of 133
targets, which showed Arm D losing on both primary endpoints (seed-excluded coordination
18.63% vs Arm C's 29.95% on the same targets; V1 clash 36.21% vs 19.38%).** Nothing below
changes an endpoint, threshold, decision rule, the cohort definition, the checker, or the
resampling unit. Only *which targets are sampled, and how many* changes.

### What changed

§3 registered generation over all 133 cohort targets. Replaced with **`--cluster-cap 3`: at most
3 targets per sequence cluster, selected in cohort-file order** — 42 targets (36 in the primary
X-ray stratum), covering **21 of 21 X-ray clusters**. Control S1 is simultaneously **restored to
$N = 100$**, superseding the $N = 25$ registered in §5 above.

### Why — a fact about the cohort, not about the result

The registered analysis weights every cluster **equally** (cluster is the resampling unit, fixed
in Step 1's `ANALYSIS_PLAN.md` after an audit found an earlier 30-entry set collapsing to 6
proteins). But the X-ray stratum is severely skewed:

- **C07 holds 50 of 127 targets; C03 holds 32** — together 65% of the stratum.
- **11 clusters hold exactly one target**; 5 hold two.
- Consequence: generating in cohort-file order requires **117 of 133 targets to touch all 21
  clusters**. The interim 19 targets covered only **6 clusters, with 16 of the 19 inside C03**.

So sequential generation spends its GPU budget inside the two mega-clusters, which are
down-weighted to $1/21$ each at analysis time. This is a property of the cohort that was
knowable before any molecule was generated and should have been caught when §3 was written.

### Measured cost of the cap

Computed by subsampling the **published Arm C vs Arm A** contrast — i.e. from data independent
of Arm D — on `has_valid_coordination`, $m = 21$:

| Design | Targets | $\sigma_d$ | MDE (80%) | GPU |
|---|---:|---:|---:|---:|
| All targets, $N{=}100$ (as registered) | 127 | 0.0750 | 4.59 pp | 15.1 h |
| Cap 6/cluster | 50 | 0.0754 | 4.61 pp | 5.9 h |
| Cap 4/cluster | 41 | 0.0760 | 4.65 pp | 4.9 h |
| **Cap 3/cluster (adopted)** | **36** | **0.0768** | **4.70 pp** | **4.3 h** |
| Cap 2/cluster | 31 | 0.0779 | 4.77 pp | 3.7 h |

**The cap costs 0.11 pp of minimum detectable effect (4.59 → 4.70) for a 3.5× GPU reduction**,
because once ~3 targets estimate a cluster mean, between-cluster variance dominates and targets
4–50 inside C07 add almost nothing.

### Every inferential property is preserved or improved

| Property | As registered | After amendment |
|---|---|---|
| Clusters ($m$) | 21 | **21** (unchanged) |
| Primary MDE vs Arm C | 4.59 pp | 4.70 pp (+0.11) |
| **S1 MDE** | 6.77 pp ($N{=}25$) | **4.70 pp ($N{=}100$)** |
| Endpoints / thresholds / decision rules | — | **untouched** |
| Remaining GPU | 16.9 h | **8.7 h** |

### The obvious objection, stated rather than avoided

*"You looked, saw Arm D losing, and cut its budget. Would you have run all 133 if it were
winning?"* That charge cannot be fully refuted, and pretending otherwise would be worse than
recording it. Three things bound it:

1. The justification is the cluster size distribution — independent of any Arm D outcome — and
   the MDE table is derived from Arm C/Arm A data, not Arm D data.
2. The amendment **strengthens the control**, moving S1 from $N{=}25$ to $N{=}100$ and its MDE
   from 6.77 pp to 4.70 pp. Someone burying an unwanted result weakens controls; they do not
   upgrade the one control that could overturn their own arm.
3. Interim point estimates remain non-registered. The verdict is taken from the capped,
   21-cluster analysis, not from the 6-cluster interim.

Effects under test are **11.11 pp** and **22.40 pp** against a 4.70 pp MDE — 2.4× and 4.8×
margin. The cap does not put any live hypothesis below the detection limit.

### Superseded numbers

§5's "$N = 25$ / 6.77 pp bound" for S1 no longer applies; S1 runs at $N = 100$ and both contrasts
use the **4.70 pp** bound. The $N{=}25$ decomposition table in §5 is retained as the record of
why the reduced-$N$ route was considered and then made unnecessary.
