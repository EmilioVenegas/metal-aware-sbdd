# Step 0 / Step 2 Pre-flight Gate Checks: Scale & Leakage Verification

**Date:** 2026-08-17  
**Scope:** Pre-flight data pipeline and pre-training gates for Step 2 zinc ablation study.

---

## 1. Summary of Gate Outcomes

| Gate | Requirement | Measured Result | Status |
|---|---|---|---|
| **Gate 1: Usable Zn Training Scale** | ≥ 2,000 usable metalloprotein complexes with catalytic Zn | **2,886 complexes** (within 8.0 Å pocket cutoff) / **2,514 complexes** (within 5.0 Å core shell) across **1,239 unique PDBs** | **PASSED** |
| **Gate 2: Non-leakage vs Eval Cohort** | Zero overlap in PDB ID, UniProt ID, sequence (>30%), or cluster | **0 PDB overlaps**, **0 UniProt overlaps**, **0 cluster overlaps**, all sequence identities < 30% | **PASSED** |

Both Gate 1 and Gate 2 have formally passed. Training for Arm B (metal-blind fine-tune) and Arm C (metal-aware fine-tune) is fully licensed without invoking the Binding MOAD fallback.

---

## 2. Gate 1: Training Data Scale Breakdown

### Methodology
1. Downloaded all source PDB structures for CrossDocked zinc entries directly from RCSB into `data/crossdocked_receptors/` (1,512/1,512 fetched, 100% success).
2. For each crystal structure, identified Zn ions and evaluated catalytic coordination:
   - Evaluated protein sidechain coordination: His (`ND1`/`NE2`), Asp (`OD1`/`OD2`), Glu (`OE1`/`OE2`), Cys (`SG`), Ser (`OG`), Thr (`OG1`), Tyr (`OH`), Asn (`OD1`/`ND2`), Gln (`OE1`/`NE2`), Lys (`NZ`), Met (`SD`) within ≤ 2.8 Å of Zn.
   - Identified all non-solvent bound ligand molecules (≥ 5 heavy atoms).
   - Computed minimum distance between any ligand heavy atom and catalytic Zn.

### Results
- **Total Zn PDBs analyzed:** 1,513 PDBs
- **PDBs with sidechain-coordinated catalytic Zn:** 1,509 (99.7%)
- **Complexes with ligand within 5.0 Å of catalytic Zn:** 2,514 complexes across 1,161 PDBs
- **Complexes with ligand within 8.0 Å of catalytic Zn (standard pocket cutoff):** 2,886 complexes across 1,239 PDBs
- **Gate threshold:** ≥ 2,000 complexes.
- **Outcome:** Gate 1 PASSED (2,886 > 2,000).

---

## 3. Gate 2: Leakage Verification Breakdown

### Methodology
Evaluated CrossDocked zinc training structures against the 133-target external zinc evaluation cohort (`data/external_zn_test_clean.pt`):
- 133 test targets across 26 target-disjoint clusters (<30% sequence identity)
- 28 unique UniProt accessions in the test cohort

### Checks
1. **Direct PDB ID Overlap:**
   - Test PDB IDs (133): `0` found in CrossDocked train (0 / 133 overlap).
2. **UniProt Accession Overlap:**
   - Test UniProt IDs (28): `0` found in CrossDocked train targets (0 / 28 overlap).
3. **Cluster / Gene Overlap:**
   - 26 evaluation clusters: `0` found in CrossDocked train clusters.
4. **Sequence Identity:**
   - Maximum sequence identity between any test target and CrossDocked training sequences is strictly < 30%.

- **Outcome:** Gate 2 PASSED (Zero leakage confirmed).

---

## 4. Preprocessing Pipeline Modifications

The metal-retention updates have been implemented and verified:
1. `DiffSBDD/process_crossdock.py`:
   - Updated residue filter to retain HETATM records with resnames in `{ZN, MG, FE, MN, CA, CU}` when ordered (no alternate-location ambiguity).
   - Capitalized element names during pocket atom extraction and mapped to `amino_acid_dict`.
2. `DiffSBDD/utils.py`:
   - Updated `get_pocket_from_ligand` to mirror `process_crossdock.py` exactly, ensuring inference-time pocket extraction retains the catalytic zinc.
3. Verified on real test PDBs (e.g. `9SSB.pdb`), confirming zinc ions are preserved in the extracted pocket representations.
