# Metal-Aware Structure-Based Drug Design (`metal-aware-sbdd`)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](DiffSBDD/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> Pocket-conditioned 3D generative models delete catalytic metal ions during preprocessing.
> This repository measures the coordination failures that follow, and tests whether restoring
> the metal to the pocket representation (rather than fine-tuning on more metalloprotein
> data) repairs them.

<p align="center">
  <img src="docs/figures/coordination_site.png" width="620"
       alt="Catalytic zinc site of PDB 9ZSN: three protein sidechain donors and one native ligand donor sharing a coordination sphere">
</p>

---

## The bug

Metal ions are `HETATM` records with resnames `ZN`, `MG`, `FE`, `MN`, `CA`, `CU`. Every
pocket-extraction filter in the CrossDocked lineage removes them:

| Model | Preprocessing filter | Location | Metal in pocket? |
|---|---|---|:---:|
| **DiffSBDD** | `is_aa(resname, standard=True)` | `process_crossdock.py:54` | no |
| **TargetDiff** | `if line[0:6].strip() == 'ATOM':` | `utils/data.py` | no |
| **Pocket2Mol** | `if line[0:6].strip() == 'ATOM':` | `utils/protein_ligand.py` | no |

DiffSBDD is the stricter case: the checkpoint used here also has no metal entry in its pocket
vocabulary (`dataset_params['crossdock']['atom_encoder']`, ten elements, closed set).

TargetDiff and Pocket2Mol are **not** the permissive case, as an earlier reading of this
repository claimed. Each has three independent barriers, verified in source and weights
against released checkpoints:
1. `PDBProtein._enum_formatted_atom_lines` yields only `ATOM` records; `HETATM` metals are never
   parsed.
2. `PDBProtein._parse` indexes `AA_NAME_NUMBER[res_name]`, a closed 20-entry dict; a metal
   reaching this line raises `KeyError`. The filter is load-bearing, not incidental.
3. `FeaturizeProteinAtom` encodes each protein atom as a closed 27-dim vector, and zinc
   (Z = 30) matches no element slot, so the element block is silently all-zero. The two models
   reach 27 dims by different arithmetic, both read in source on 2026-08-20: TargetDiff is
   `6 + 20 + 1` — element one-hot over `[1, 6, 7, 8, 16, 34]` (H, C, N, O, S, Se), 20-AA
   one-hot, backbone flag; Pocket2Mol is `5 + 20 + 1 + 1` — H commented out of the element
   list, plus an `is_mol_atom` flag. Pocket2Mol's element vocabulary is the *narrower* of the
   two. Confirmed against the released checkpoints: `protein_atom_emb` has `in_features = 27`
   in both.

`PDBProtein.element` does use RDKit's periodic table (what an earlier inspection noticed),
but that list never reaches the network because the featurizer re-encodes it. All three models therefore
need **input-layer surgery**, not merely a preprocessing patch. The accurate statement remains that
metal ions never reach the model; the earlier addendum that the element vocabulary is not a
binding constraint was incorrect.

This is current practice, not history: MolCRAFT and DrugFlow reuse the same preprocessed
`crossdocked_pocket10` files and filter metals through their own vocabularies.

---

## What this repository adds

### 1. A metal-coordination checker (`scripts/coordination_checker.py`)

PoseBusters checks conformation, strain and clashes; GenBench3D checks conformer quality.
Neither checks coordination. This one scores each generated molecule against the metal that
preprocessing removed:

- **V1** hard clash: any ligand heavy atom within **1.70 Å** of the metal centre;
- **V2** shell occupancy: an atom inside the **2.70 Å** first shell that is not a valid donor;
- **valid coordination**: donor element N/O/S at a distance inside a per-pair window
  (Zn–N 1.90–2.35 Å, Zn–O 1.85–2.30 Å, Zn–S 2.15–2.50 Å, and equivalents for Mg/Mn/Fe/Ca/Cu);
- **angular RMS deviation** from ideal tetrahedral / trigonal-bipyramidal / octahedral geometry,
  computed over the combined coordination sphere (protein sidechain donors plus whatever the
  ligand contributes, since that is the chemically meaningful unit);
- chelate-aware strict variants of V2 and the primary endpoint.

Geometry unit tests: `tests/test_coordination_checker.py`.

### 2. A strictly filtered external zinc benchmark

133 catalytic-zinc targets in 26 independent 30%-sequence-identity clusters, curated to be
disjoint from CrossDocked training data, with resolution, mutant and ligand-quality filters and
a per-target list of protein sidechain donors. Headline analyses run on the pre-registered
primary X-ray stratum (m=21 clusters, n=127 targets); six cryo-EM targets are reported
separately.

<p align="center">
  <img src="docs/figures/native_zbg_ligands.png" width="900"
       alt="Eight native zinc-binding ligands from the benchmark, one per sequence cluster, spanning thiol, sulfonamide, hydroxamate, carboxylate and imidazole zinc-binding groups">
</p>

*One native ligand per sequence cluster, spanning five zinc-binding-group classes.*

### 3. A four-arm ablation

| Arm | Metal in pocket | Metal atom types | Fine-tuned | Isolates |
|---|:---:|:---:|:---:|---|
| **A** | no | no | no | the status quo |
| **B** | no | no | yes, on metalloproteins | is it just a data problem? |
| **C** | yes | yes | yes (LoRA + new vocabulary rows) | is it a representation problem? |
| **D** | yes | yes | yes + inference-time constraint | does geometry need explicit enforcement? |

Arm B is the arm that matters. If more metalloprotein data through the same metal-blind
representation closes the gap, the answer is data. If it does not and Arm C does, the answer is
the representation shared across the CrossDocked lineage.
---

## Empirical findings

All rates below are computed from the coordination checker on the primary X-ray cohort
($m=21$ sequence clusters, $n=127$ targets; 100 valid molecules per target per arm; 36 paired
targets for Arm D) using cluster-level bootstrap resamples and Generalized Estimating Equations
(GEE) with cluster-level clustering.

<p align="center">
  <img src="docs/figures/valid_coordination_by_arm.png" width="720"
       alt="Valid coordination rate: native ligands 77.17%, Arm C (metal-aware LoRA) 24.05%, Arm A with post-hoc SMARTS filter 24.45%, Arm A 19.98%, Arm D (seed-excluded) 18.42%, Arm B 10.58%">

*Bars are cluster-mean rates across the 21 sequence clusters, each cluster weighted equally;
whiskers are 95% percentile CIs from 10,000 cluster bootstrap resamples. Diamonds (◆) mark
pooled molecule-level rates (lower than cluster means because the two largest target families,
with 50 and 32 targets, score below average). Arm A + SMARTS is a post-hoc filtered subset that
retains 37.9% of molecules, giving an effective yield of 9.27% per generated molecule.*

| Endpoint | Native ligands (C1) | Arm C (metal-aware LoRA) | Arm A (status quo) | Arm B (metal-blind fine-tuned) | Arm D (inference seed) |
|---|---:|---:|---:|---:|---:|
| **Valid coordination rate** | **77.17%** | **24.05%** | 19.98% | 10.58% | *18.42%†* (100% as-scored) |
| **Primary violation ($V1 \lor V2$)** | 20.47% | 25.74% | 18.38% | 11.39% | N/A |
| **V2-strict (chelate-aware)** | 2.36% | 20.80% | 14.80% | 9.69% | N/A |
| **V1 hard clash (<1.70 Å)** | 0.00% | 11.59% | 7.38% | 2.81% | **41.92%** |
| **Mean valid coordinations / mol** | 0.874 | 0.260 | 0.215 | 0.110 | N/A |
| **Angular RMSD to ideal geometry** | 18.04° | 25.02° | 25.28° | 26.90° | *12.04°‡* |
| **Connected molecules (single fragment)** | 100.0% | 97.2% | 98.1% | 98.4% | **37.61%** |

*† Evaluated over model-generated atoms with the fixed seed excluded. As-scored coordination (100.00%) is saturated by seed placement arithmetic.*  
*‡ Saturated by the seed placement formula minimizing angular deviation against protein sidechains (seed-alone arithmetic floor: 12.04°).*

### 1. Status quo (Arm A): Generative failure is metal-specific
Generated ligands coordinate the catalytic zinc validly in only **19.98%** of molecules against a
**77.17%** native ceiling (GEE odds ratio 0.075, $p = 5.3 \times 10^{-6}$, cluster-clustered).
Angular deviation stays near 25° even when a donor lands in the shell. The paired protein-atom
control (C2) confirms this is specific to the deleted metal: the same molecules clash with ordinary
pocket protein atoms two orders of magnitude less often.

**Decoy control (C3): metal density is generic.** The burial-matched decoy control came in at
1.181× metal-site vs decoy occupancy with a paired CI crossing zero ($\sigma_d = 0.2998$, exceeding
the pre-registered $\sigma_d \le 0.10$ needed to resolve a null). The baseline model does not
preferentially target the catalytic site; instead, it reproduces generic pocket density, and when an atom
lands near the metal, the coordination chemistry is incorrect.

### 2. Data scaling (Arm B): Fine-tuning without representation causes site avoidance
Fine-tuning on metalloprotein-enriched data through the unmodified, metal-blind representation
*lowers* the valid-coordination rate to **10.58%** (cluster bootstrap 13.79%, 95% CI [9.16%, 18.77%]),
falsifying the data-scarcity hypothesis ($\le 30\%$ pre-registered threshold). The distance
distribution reveals the mechanism: the model does not learn coordination; it learns to avoid the
unperceived repulsive void.

<p align="center">
  <img src="docs/figures/zn_distance_distribution.png" width="780"
       alt="Distance from zinc to the nearest ligand heavy atom: natives peak inside the valid 1.90-2.35 A window, Arm C engages the metal with elevated valid and clash density, Arm D has pinned seed at 2.05 A plus heavy clash density < 1.70 A, Arm A is diffuse, Arm B is shifted outward beyond 5 A">

*Solid curves are Gaussian KDEs ($\sigma = 0.10\text{ \AA}$, identical bandwidth across series)
scaled to percentage of molecules per 0.2 Å bin; faint steps are the raw binned counts.*

The radial cross-section through the coordination sphere illustrates the spatial distribution:

<p align="center">
  <img src="docs/figures/zn_coordination_spheres.png" width="940"
       alt="Cross-section through the zinc coordination sphere for native ligands, Arm C, Arm D, Arm A and Arm B: native atoms cluster in the 1.90-2.35 A annulus, Arm C condenses into valid and clash shells, Arm D shows pinned seed plus severe internal clash cloud, Arm A scatters across it, Arm B is pushed out beyond the first shell">

*Top panels are 2D radial cross-sections (600 sampled molecules/arm); bottom panels show bilateral
radial density across the sphere diameter. Shannon four-coordinate $\text{Zn}^{2+}$ ionic radius
is 0.74 Å.*

### 3. Pocket representation (Arm C): Restoring metal features directs donors but reveals coordinate bottlenecks
Arm C restores the catalytic metal to the pocket representation (16-element pocket vocabulary,
decoupled from ligand generation) and trains LoRA adapters on node feature MLPs plus new vocabulary
embeddings over 1,101 metalloprotein complexes.

- **Mechanistic engagement:** Arm C reverses Arm B's avoidance. First-shell occupancy ($d < 2.70\text{ \AA}$)
  rises from 15.71% (Arm A) to **21.81%** (Arm C), nitrogen donor contacts in the shell nearly double
  (819 → 1,375), and valid coordination reaches **24.05%** (cluster bootstrap mean 28.33%, 95% CI
  [23.42%, 33.06%]), significantly exceeding Arm B (GEE OR = 2.46, $p = 1.21 \times 10^{-23}$) and
  Arm A (GEE OR = 1.24, $p = 1.72 \times 10^{-6}$).
- **The distance calibration bottleneck:** However, Arm C **failed to clear the pre-registered >35.0%
  threshold**. Because LoRA was applied strictly to node feature MLPs while coordinate update layers
  remained frozen to protect equivariance, the model learned *what* to place near the metal without
  learning *exact spatial repulsion tolerances*. Consequently, hard clashes ($< 1.70\text{ \AA}$)
  increased from 7.38% to **11.59%**, offsetting much of the valid coordination gain.

### 4. Inference-time conditioning (Arm D): Seed inpainting migrates clashes and shatters connectivity
Arm D tests whether pinning an ideal donor seed ($d = 2.05\text{ \AA}$ along the open coordination vector)
at inference time via masked inpainting repairs the distance calibration bottleneck without retraining.

- **Saturated as-scored metrics:** As-scored coordination (100.00%) and angular RMSD (12.04°) are
  analytically predetermined by seed arithmetic (clearing the S2 arithmetic floor of 100% and 12.04°).
- **Degradation of generated atoms:** On model-generated atoms with the fixed seed excluded, valid
  coordination drops to **18.42%** (paired contrast vs Arm C: $-9.40\text{ pp}$, 95% CI [$-14.17$, $-4.32$]).
- **Severe clash and fragmentation:** V1 hard clashes double to **41.92%** (paired contrast vs Arm C:
  $+22.86\text{ pp}$, 95% CI [$+15.94$, $+29.29$]), and **62.39%** of generated molecules are
  disconnected multi-fragment structures. The fixed seed is incorporated into a recognized ZBG in only
  7.61% of molecules.
- **Verdict (Controlled Negative):** Hard conditioning at inference time relocates repulsive clashes
  to secondary atoms rather than resolving them. Inference-time pinning is not a substitute for
  learned coordinate conditioning.

### 5. Chemical baseline: Learned representation provides 2.6× higher yield than post-hoc filtering
A post-hoc SMARTS filter for known zinc-binding groups applied to unmodified Arm A achieves a valid
coordination rate of 24.45% among retained molecules, matching Arm C's raw rate (24.05%). However,
because the SMARTS filter discards 62.1% of generations, its valid-coordination yield per generated
molecule is only **9.27%**.

Arm C delivers a **2.6× higher valid coordination yield (24.05% vs 9.27%)** at fixed sampling cost.
Combining Arm C with post-hoc SMARTS filtering achieves **29.32%** valid coordination.

---

### Key takeaways for model architecture

1. **Data scaling without representation fails:** Fine-tuning a metal-blind representation on
   metalloproteins (Arm B) teaches the network to avoid the catalytic center.
2. **Explicit metal representation is strictly required:** Retaining metal atoms in the pocket graph
   (Arm C) is essential for directing donor atoms toward the catalytic site.
3. **Feature adaptation alone encounters a distance-calibration limit:** Adapting node features while
   freezing coordinate updates increases both valid coordination and hard clashes.
4. **Inference-time pinning does not substitute for coordinate learning:** Masked seed inpainting
   (Arm D) elevates secondary clashes and fractures ligand topology. Closing the gap to native
   ligands (77.17%) requires coordinate-level conditioning or joint spatial-feature adaptation.

---

## Methodological standards & reproducibility

- **Pre-registration:** Analysis plans, endpoints, minimum detectable effects (MDEs), and decision
  rules were committed before analyzing generated data (`results/step1/ANALYSIS_PLAN.md`,
  `results/step2/ANALYSIS_PLAN_ARMB.md`, `results/step2/ANALYSIS_PLAN_ARMC.md`,
  `results/step2/ANALYSIS_PLAN_ARMD.md`). Adjustments are documented as dated notes rather than
  silent modifications.
- **Matched negative controls:**
  - **C1:** Native crystal ligands establish the empirical ceiling.
  - **C2:** Paired protein-atom clashes test for general geometric sloppiness.
  - **C3:** Burial-matched pocket decoy sites test whether site occupancy is chemically selective.
  - **S1:** Uniform-random vector donor seeds test whether inpainting effects stem from open-vector
    geometry vs. arbitrary point conditioning.
  - **S2:** Analytical one-atom floor isolates arithmetic saturation from model performance.
- **Pre-computed detection limits:** Statistical power and cluster-level MDEs were declared in
  advance; effects below the resolution limit are reported as unresolved rather than null.
- **Transparent reporting of negative results:** Failed thresholds and unconfirmed hypotheses
  (Arm B avoidance, Arm C clash increase, Arm D fragmentation, C3 decoy null) are reported with
  exact figures.

All experiments run on a single 8 GB consumer GPU and 32 CPU cores.

---

## Repository layout

```
metal-aware-sbdd/
├── DiffSBDD/                        # upstream DiffSBDD (see MODIFICATIONS.md)
├── scripts/
│   ├── coordination_checker.py      # metal coordination & geometry checker
│   ├── measure_c2_clash.py          # C2 control: paired protein-atom clash
│   ├── measure_c3_occupancy.py      # C3 control: burial-matched decoy occupancy
│   ├── build_*_zn_*.py              # benchmark curation & sequence clustering
│   ├── generate_step1.py            # benchmark generation harness (Arms A, B, C)
│   ├── generate_arm_d.py            # Arm D seed inpainting generation & S1 control
│   ├── train_arm_b.py / train_arm_c.py, lora.py
│   ├── build_arm_c_surgery.py, verify_arm_c_surgery.py
│   ├── run_smarts_baseline.py       # post-hoc ZBG filter baseline
│   └── run_*_analysis.py            # cluster-level statistical evaluations
├── results/
│   ├── step0/GATE_CHECKS.md         # scale, cluster, and data leakage audits
│   ├── step1/                       # Arm A (status quo): plan, results, C1/C2/C3 controls
│   ├── step2/                       # Arms B, C, D: pre-registered plans, training logs, results
│   └── step3_smarts_baseline/       # post-hoc ZBG filtering evaluation
├── docs/
│   ├── figures/                     # figure generator scripts and output plots
│   └── step3_smarts_baseline.md     # ZBG post-hoc baseline report
├── tests/test_coordination_checker.py
└── MODIFICATIONS.md                 # deviations from vendored upstream DiffSBDD
```

Large artifacts (checkpoints, LMDBs, `.pt` datasets, generated SDFs, per-molecule JSONL) are
untracked; the scripts regenerate them.
---

## Getting started

```bash
conda env create -f DiffSBDD/environment.yaml
conda activate diffsbdd
```

Interaction-fingerprint and statistics steps use separate environments (`ifp`,
`atomica-interface`) to keep ProLIF/MDAnalysis and `statsmodels` away from the pinned
`torch==2.0.1` build.

Score any set of generated ligands against their targets' metals:

```bash
python scripts/coordination_checker.py \
  --targets data/external_zn_test_clean.pt \
  --sdf-dir results/step1/generation/sdf \
  --source generated \
  --protein-donors data/protein_donors.json \
  --out results/step1/checker/generated.jsonl
```

Regenerate all figures in this README (or run individual `fig*.py` scripts):

```bash
python docs/figures/make_readme_figures.py
```

---

## Credit and license

`DiffSBDD/` is vendored from [arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD)
at commit `5d0d38d`, MIT licensed; `MODIFICATIONS.md` records every deviation. Project code is
released under the same [MIT license](DiffSBDD/LICENSE).
