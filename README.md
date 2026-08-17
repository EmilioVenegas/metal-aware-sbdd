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

## Core Contributions

1. **Metal Coordination Checker ([`scripts/coordination_checker.py`](file:///home/emilio/Documents/metal-aware-sbdd/scripts/coordination_checker.py))**
   - Implements rigorous geometric validation of metal-ligand coordination chemistry missing from standard benchmarks like PoseBusters and GenBench3D.
   - Evaluates:
     - **V1 (Clash)**: Van der Waals clashes within the inner coordination sphere.
     - **V2 (Shell Occupancy)**: Non-donor atoms (e.g., carbon, halogen) invading the coordination sphere.
     - **V3 (Malformed Distance)**: Donor atoms present but at unphysical bond lengths.
     - **Valid Coordination & Denticity**: Correct element-pair donor distances, angular geometry, and coordination numbers.
2. **Empirical Failure Measurement with Rigorous Controls**
   - Benchmarks generative models against a curated, strictly filtered test set of catalytic zinc proteins ($m=22$ independent sequence clusters at 30% sequence identity).
   - Paired controls (native complexes and burial-matched decoy points) establish that coordination failure is specific to metal neglect rather than general 3D generative geometric noise.
3. **Representation-Aware Generative Modeling**
   - Restores explicit metal coordinates and element types to the pocket conditioning tensor.
   - A multi-arm ablation study isolating representation defects from training data volume effects.

---

## Repository Layout

```
metal-aware-sbdd/
├── DiffSBDD/                  # Vendored DiffSBDD architecture & equivariant diffusion dynamics
│   ├── equivariant_diffusion/ # EGNN dynamics and continuous diffusion modules
│   ├── configs/               # Model and dataset configuration files
│   └── analysis/              # SA_Score, docking, and molecular evaluation tools
├── scripts/                   # Dataset curation, generation, checker, and analysis pipelines
│   ├── coordination_checker.py       # Geometric metal coordination checker
│   ├── generate_step1.py             # DiffSBDD inference harness across targets
│   ├── place_c3_decoys.py            # Burial-matched decoy placement for paired controls
│   ├── measure_c3_occupancy.py       # C3 control occupancy evaluation
│   ├── build_strictly_clean_zn_set.py# External zinc benchmark dataset curator
│   └── refilter_zn_set.py            # Quality filters & sequence identity clustering
├── results/                   # Pre-registered analysis plans and experimental outputs
│   ├── step0/                 # Step 0 baseline reproducibility and verification
│   └── step1/                 # Step 1 external zinc benchmark and coordination results
│       ├── ANALYSIS_PLAN.md   # Pre-registered analysis protocol & power calculations
│       └── C1_RESULT.md       # Native coordination control results
├── docs/                      # Research plan and step-by-step progress tracking
│   ├── plan.md                # Master research plan and experimental gates
│   └── step0.md               # Step 0 setup and validation checklist
├── tests/                     # Unit tests for geometry thresholds and checkers
│   └── test_coordination_checker.py
└── README.md
```

---

## Methodological Standards

This project adheres to strict pre-registration and falsification principles:
- **Pre-Registration**: Analysis plans, statistical endpoints (GLMM with cluster random effects, bootstrap bounds), and decision gates are committed to the repository *before* evaluating generated molecules.
- **Controlled Hypotheses**:
  - **C1 (Native Control)**: Validates that native ligands satisfy checker thresholds.
  - **C2 (Model Failure)**: Quantifies coordination failure rates in unconditioned models.
  - **C3 (Metal-Specific Decoy Control)**: Compares metal occupancy against burial-matched pocket decoy locations to rule out generic pocket-filling artifacts.
- **Minimum Detectable Effect (MDE)**: Statistical power and MDEs are calculated a priori for all comparisons across sequence clusters.

---

## Getting Started

### Prerequisites
- Linux OS
- CUDA-compatible GPU (8 GB+ VRAM supported)
- Conda / Mamba environment manager

### Environment Setup

Create and activate the environment:
```bash
conda env create -f DiffSBDD/environment.yaml
conda activate diffsbdd
```

### Running Unit Tests

Run the geometric coordination test suite:
```bash
pytest tests/test_coordination_checker.py -v
```

### Running the Metal Coordination Checker

To evaluate a generated ligand or native complex against a target protein:
```bash
python scripts/coordination_checker.py \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.sdf \
  --metal-element ZN \
  --output-json results/coordination_summary.json
```

---

## License

This project is licensed under the [MIT License](file:///home/emilio/Documents/metal-aware-sbdd/DiffSBDD/LICENSE).
