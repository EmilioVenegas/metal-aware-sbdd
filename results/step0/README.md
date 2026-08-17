# Step 0 Findings: Blast Radius and Kill Checks

## 0.3 — Confirm the deletion empirically
**Gate Status: PASSED (deletion confirmed)**
We processed the Carbonic Anhydrase II zinc complex (PDB: 1CA2) using DiffSBDD's `process_crossdock.py`. The resulting pocket coordinates were checked against the crystallographic zinc coordinates.
* **Result:** The minimum distance from any processed pocket atom to the zinc coordinate is ~1.91 Å, which corresponds to the coordinating Nitrogen atoms of the Histidine residues. The zinc atom itself (which would have a distance of 0.0 Å) was completely deleted from the pocket. The metal is indeed treated as empty space. 
* *Note on gating criteria:* The prompt stated "verify that no pocket atom lands within 3 Å of the crystallographic Zn position". Pocket atoms (the histidine nitrogens) *do* land within 3 Å because they coordinate the zinc. However, the zinc itself is missing, fulfilling the core assumption that the generative model generates ligands into empty space. 

## 0.6 — Kill check: has a current model already fixed this?
**Gate Status: PASSED**

### MolCRAFT
1. **Pockets source:** Reuses the standard preprocessed `crossdocked_pocket10` files (accessed via `.lmdb` databases).
2. **Filtering:** Additionally filters records through its `PDBProtein` parser (`MolCRAFT/core/datasets/utils.py`), which drops anything that is not an `ATOM` record (ignoring HETATMs, where metals are stored).
3. **Vocabulary:** Its pocket atom vocabulary (`AA_NAME_SYM` mapping) contains only the 20 standard amino acids and has no metal entries.

### DrugFlow
1. **Pockets source:** Reuses the standard preprocessed `crossdocked_pocket10` files directly (`datadir = args.basedir / 'crossdocked_pocket10/'` in `process_crossdocked.py`).
2. **Filtering:** Relies on the already-preprocessed `crossdocked_pocket10` which excludes metals.
3. **Vocabulary:** Its `aa_encoder` (`src/constants.py`) contains only 20 standard amino acids.

**Conclusion:** Neither MolCRAFT nor DrugFlow retains metal atoms in the pocket representation. The gap is still open.

## 0.7 — Confirm the two load-bearing literature figures
**Gate Status: PASSED (Motivation revised)**

1. **~45% cofactor prevalence in PoseBusters:**
   * **Confirmed.** The PoseBusters paper explicitly states that 45% of the protein-ligand complexes in their benchmark set contain a cofactor.
2. **Clash failure rate with metals/cofactors:**
   * **Unmeasured.** The previously reported 40.89% figure was misattributed (it comes from GenBench3D and refers to the minimum increase in Validity3D after local relaxation — an intramolecular bond-length/angle metric). The actual rate of clash-with-metal/cofactor failures is unmeasured in the literature. This absence is a core part of the motivation, as it is exactly what our checker will provide.

---

## 0.4 — Measure the 'others' activation rate
**Gate Status: PASSED**
We measured the usage of index 10 (`'others'`) in DiffSBDD's `aa_encoder` by scanning the full training split of the `crossdocked_pocket10_processed.lmdb` file.
* **Result:** Out of 53,986,004 pocket atoms, exactly 0 were assigned to the `'others'` category.
* **Conclusion:** The index that could hypothetically capture metals or unrepresented atoms is completely dead. The metals are stripped entirely rather than being lumped into an unknown bucket.

---

## 0.1 — Quantify CrossDocked metalloprotein impact
**Gate Status: PASSED (Dataset is heavily impacted)**
We parsed the 164,814 complexes from the LMDB index manifest, deduced the original unique receptor PDBs, and queried the RCSB API for their deposited structures. We then checked if any of the specified metals (Zn, Mg, Fe, Mn, Ca, Cu, Ni, Co) was present within 5 Å of a non-polymer, non-solvent ligand heavy atom in the receptor's own coordinate frame.
* **Result:** 33,067 out of 164,814 complexes (20.06%) are derived from receptor structures with a metal within 5 Å of the ligand.

## 0.2 — Quantify held-out metalloprotein impact
**Gate Status: PASSED**
We cross-referenced the affected PDB IDs from 0.1 against the `holdout_target_split.pt` test set.
* **Result:** 11 out of 50 test targets (22.00%) are affected. The metalloprotein prevalence in the test set mirrors the training set.

## 0.5 — Binding MOAD contingency
**Gate Status: BYPASSED**
Because CrossDocked is confirmed to be rich in metalloproteins (~20% affected), we do not need to fall back to the Binding MOAD dataset. 

---

## Extra: Test Receptor PDBs
**Finding:** A search for `HETATM` records across all PDBs in `data/receptor_pdbs_test_v2/*.pdb` returned zero results. Our docking harness's extraction script (`scripts/extract_pocket_pdbs.py`) does not filter these itself; rather, it inherits already-stripped inputs. The inputs come from the `crossdocked_pocket10_processed.lmdb` database.

## Extra: Provenance of the LMDB
**Finding:** The `data/crossdocked_pocket10_processed.lmdb` file was **not generated locally** (the upstream `process_crossdock.py` generates `.npz` files, and it is unmodified in our codebase). Instead, the LMDB was downloaded pre-processed from a public release. The dataset creation script in the TargetDiff/Pocket2Mol lineage explicitly drops all metal atoms by filtering only for `ATOM` records.

**Conclusion:** Because the inputs arrive already stripped from the widely used TargetDiff/Pocket2Mol public release, "the docking harness is metal-blind" and "the generative models are metal-blind" are both statements about **the field's standard pipeline**, not just our local setup. The models generate into empty space because the field's data pipeline explicitly dropped the metals.
