# Metal-Aware Structure-Based Drug Design (`metal-aware-sbdd`)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](DiffSBDD/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Benchmarking and repairing metal coordination failures in pocket-conditioned 3D generative diffusion models for structure-based drug design (SBDD).**

---

## Overview & Scientific Rationale

Metalloproteins represent roughly 30–40% of all structurally characterized enzymes and encompass critical drug targets across oncology, infectious diseases, and inflammation (e.g., matrix metalloproteinases, carbonic anhydrases, and histone deacetylases). In these systems, catalytic and structural metal ions ($Zn^{2+}$, $Mg^{2+}$, $Fe^{2+/3+}$, $Mn^{2+}$, etc.) are central determinants of pocket volume, electrostatics, and ligand affinity.

Despite this, modern pocket-conditioned 3D generative models for SBDD **delete metal ions during data preprocessing**:

| Model | Preprocessing Filter | Source Location | Metals in Pocket? |
|---|---|---|:---:|
| **DiffSBDD** | `is_aa(resname, standard=True)` | `process_crossdock.py:54` | ❌ No |
| **TargetDiff** | `if line[0:6].strip() == 'ATOM':` | `utils/data.py` | ❌ No |
| **Pocket2Mol** | `if line[0:6].strip() == 'ATOM':` | `utils/protein_ligand.py` | ❌ No |

Because metal ions are parsed as `HETATM` records in standard PDB/CIF files, every one of these filters strips them from the pocket representation. The generative model views the catalytic metal coordinate as empty pocket space, frequently generating hydrophobic scaffolds directly into the primary coordination sphere or missing coordination interactions entirely.

---

## Core Contributions & Methodology

1. **Metal Coordination Checker**
   - Implements rigorous geometric validation of metal-ligand coordination chemistry missing from standard benchmarks like PoseBusters and GenBench3D.
   - Evaluates Van der Waals clashes (V1), non-donor shell invasion (V2), malformed distances (V3), and angular geometry/denticity.
2. **Empirical Failure Measurement with Rigorous Controls (Step 1)**
   - Benchmarks generative models against a curated, strictly filtered test set of catalytic zinc proteins ($m=26$ independent sequence clusters).
   - Paired controls (native complexes and burial-matched decoy points) establish that coordination failure is specific to metal neglect rather than general 3D generative geometric noise.
3. **Multi-Arm Ablation Study (Step 2)**
   - **Arm A (Status Quo):** Base model evaluation.
   - **Arm B (Data-Only Fix):** Full fine-tuning on a metalloprotein-enriched dataset, maintaining the broken (metal-blind) representation. 
   - **Arm C (Representation Fix):** Retaining metal atoms during preprocessing, adding new metal atom types to the conditioning vocabulary, and adapting the model via targeted LoRA.
   - This isolates whether the field-wide failure is merely a training data oversight or a fundamental representation bug.
4. **Generalization across Modern Architectures (Step 3)**
   - Evaluation of the bug and fix mechanisms on recent state-of-the-art generative frameworks (TargetDiff, MolDiff, SurfGen, FlagGNN) to prove this is a universal SBDD bottleneck.

---

## Recent Advances & Project Status (August 2026)

- **Step 1 (Baseline Verification) is COMPLETE:** The status quo model (Arm A) successfully reproduces training-ligand density but completely fails at coordination chemistry, achieving only a 19.98% valid coordination rate with a 25-29° angular RMSD error.
- **Step 2 (Training Ablation) is COMPLETE:** 
  - **Arm B** (Novel Metalloprotein Split) finished full fine-tuning.
  - **Arm C** (Metal-Aware Vocabulary + LoRA) finished fine-tuning 9.5x faster than Arm B due to isolating the gradient to 0.7% of parameters (adapter matrices + new metal vocabulary embedding rows). 
  - *Data Hygiene Check:* Mathematically verified zero overlap (leakage) between the Arm C training splits and the external 133-target Zinc evaluation cohort.
- **Step 3 (Baseline De-risking) is COMPLETE:** We formally executed a "Kill Check" evaluating whether a cheap, post-hoc SMARTS filter (for known Zinc-Binding Groups) on Arm A's broken outputs could act as a practical bypass. The filtered molecules achieved only a **24.96%** valid coordination rate. This proves definitively that explicitly learning 3D geometric orientation is strictly required; a post-hoc functional group filter is not a viable substitute.

*Currently actively generating ligands from the Arm B & Arm C checkpoints for final 4-level GLMM statistical evaluation.*

---

## Repository Layout

```
metal-aware-sbdd/
├── DiffSBDD/                  # Vendored architecture & equivariant diffusion dynamics
├── scripts/                   # Curations, generation, checker, and analysis pipelines
│   ├── coordination_checker.py       # Geometric metal coordination checker
│   ├── generate_step1.py             # Model inference generation harness
│   ├── run_smarts_baseline.py        # Post-hoc functional group kill-check filter
│   ├── evaluate_step2_glmm.py        # 4-arm Generalized Estimating Equation (GEE) script
│   └── verify_independent_leakage.py # Dataset overlap verifier
├── results/                   # Experimental outputs and inference artifacts
│   ├── step1/                 # Arm A outputs and native controls
│   ├── step2/                 # Arm B and Arm C outputs and checkpoint generation logs
│   └── step3_smarts_baseline/ # SMARTS kill-check outputs
├── docs/                      # Research plan and progress tracking
│   ├── plan.md                # Master research plan and experimental gates
│   ├── modern_architectures.md# SOTA SBDD architectures for step 3 evaluation
│   ├── perspectives.md        # Alternative hypothesis and theoretical additions
│   └── step3_smarts_baseline.md # SMARTS kill check formal record
└── README.md
```

---

## Methodological Standards

This project adheres to strict pre-registration and falsification principles:
- **Pre-Registration**: Analysis plans, statistical endpoints (GLMM with cluster random effects, bootstrap bounds), and decision gates are committed to the repository *before* evaluating generated molecules.
- **Controlled Hypotheses**:
  - **C1 (Native Control)**: Validates that native ligands satisfy checker thresholds.
  - **C2 (Model Failure)**: Quantifies coordination failure rates in unconditioned models.
  - **C3 (Metal-Specific Decoy Control)**: Compares metal occupancy against burial-matched pocket decoy locations.
- **Minimum Detectable Effect (MDE)**: Statistical power and MDEs are calculated a priori for all comparisons across sequence clusters.

---

## Getting Started

### Prerequisites
- Linux OS
- CUDA-compatible GPU (8 GB+ VRAM supported)
- Conda / Mamba environment manager

### Environment Setup

Create and activate the environments:
```bash
conda env create -f DiffSBDD/environment.yaml
conda activate diffsbdd
```
*(Note: Some downstream evaluations like GLMM stats and ProLIF use the isolated `ifp` and `atomica-interface` environments).*

### Running the Metal Coordination Checker

To evaluate a generated ligand or native complex against a target protein:
```bash
python scripts/coordination_checker.py \
  --targets path/to/targets.pt \
  --sdf-dir path/to/generated_ligands_dir/ \
  --source generated \
  --protein-donors path/to/donors.json \
  --out results/coordination_summary.jsonl
```

---

## License

This project is licensed under the [MIT License](file:///home/emilio/Documents/metal-aware-sbdd/DiffSBDD/LICENSE).
