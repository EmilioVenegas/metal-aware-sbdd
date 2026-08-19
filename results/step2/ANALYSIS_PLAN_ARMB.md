# Pre-Registration Plan: Step 2 Arm B (Full Fine-Tune Baseline)

**Date:** 2026-08-17  
**Status:** COMMITTED BEFORE RUNNING TRAINING  
**Context:** Ablation study testing if the metal-coordination failure observed in Step 1 is simply a training data deficit (Arm B) vs a fundamental representation bottleneck (Arm C).

---

## 1. Core Hypothesis & Objective

- **Arm B Question:** Does continuing training on a 100% metalloprotein-enriched dataset—using the existing, metal-blind pocket representation—close the valid-coordination gap?
- **Hypothesis:** Fine-tuning on metalloproteins with a metal-blind representation will NOT close the valid coordination gap (or will offer marginal improvement at best). Because the pocket preprocessing deletes the metal ion, the model continues to treat the metal binding site as empty geometric space, unable to learn proper donor element preferences, coordination numbers, or angular geometries around the deleted ion.
- **Role in Ablation:** Arm B isolates data composition from representation. If Arm B fails to resolve the coordination deficit but Arm C (retained metal representation) succeeds, we conclusively prove that the pocket representation itself, rather than data availability, is the field-wide bottleneck.

---

## 2. Dataset & Cohort Definition

### 2.1 Training Set (Metalloprotein Split)
- **Source:** CrossDocked2020 training partition defined in `data/metal_target_split.pt` (`train_indices`).
- **Metalloprotein Selection:** All training complexes from receptor PDBs annotated with bound metals (`data/pdb_metals_map.json`: Zn, Mg, Ca, Fe, Mn, Cu, Ni, Co).
- **Size:** Exactly **24,339 metalloprotein complexes** across **563 distinct target protein clusters** (zero sequence identity overlap with validation or test sets).
- **Leakage Check:** 0 out of 133 external Zn test targets (`data/external_zn_test_clean.pt`) overlap with CrossDocked training PDBs or clusters (verified in §2.1 of Step 2).

### 2.2 Validation Set
- **Source:** CrossDocked2020 validation partition in `data/metal_target_split.pt` (`val_indices`).
- **Size:** Exactly **2,748 metalloprotein complexes** across 74 target protein clusters.

### 2.3 Evaluation Cohort (Fixed from Step 1)
- Scored on the **identical 133-target, 26-cluster external Zn cohort** (`data/external_zn_test_clean.pt`) using the exact Step 1 checker harness (`results/step1/checker/coordination_checker.py`), same thresholds (Zn–O/N: 1.85–2.30 Å, Zn–S: 2.15–2.55 Å, hard clash <1.70 Å), and same sample size ($N=100$ valid molecules/target).

---

## 3. Model Architecture & Training Parameters

### 3.1 Architecture
- **Base Checkpoint:** `checkpoints/crossdocked_fullatom_cond.ckpt` (epoch 999, global step 1,562,000; SHA256: `07f86764bf569aafbc40a9c15fc02de8e2550437dd0f17f657eab3abe66c372c`).
- **Vocabulary & Encoders:** Strict preservation of existing 10-element `atom_encoder` and `aa_encoder` (`{'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}`).
- **Model Parameters:** Unmodified 6-layer EGNN dynamics, full-atom pocket conditioning. No architectural surgery, no new vocabulary tokens.

### 3.2 Training Method
- **Method:** **Full fine-tune** (all 122 parameter tensors in `LigandPocketDDPM` receive gradients and update via AdamW; no freezing, no LoRA).
- **Optimizer:** AdamW (`lr = 1.0e-4`, `amsgrad = True`, `weight_decay = 1.0e-12`).
- **Batch Size:** Micro-batch size = 4, Gradient Accumulation steps = 4 $\rightarrow$ **Effective Batch Size = 16** (matching the original DiffSBDD training configuration).
- **Loss Objective:** Standard diffusion $L_2$ denoising score loss + polynomial-2 noise schedule ($T=500$ steps).

### 3.3 Planned Exposure
- **Epochs:** **5 full epochs** over the 24,339 metalloprotein training complexes.
- **Total Examples Seen:** $5 \times 24,339 = \mathbf{121,695}$ training examples.
- **Total Optimization Steps:** $\frac{121,695}{16} \approx \mathbf{7,606}$ optimization steps.

---

## 4. Intermediate Checkpoints & Trajectory Tracking

Checkpoints will be preserved at the following discrete intervals:
1. `checkpoints/arm_b_epoch1.ckpt`: After **1 epoch** (24,339 examples seen, ~1,521 steps) — early adaptation check.
2. `checkpoints/arm_b_epoch3.ckpt`: After **3 epochs** (73,017 examples seen, ~4,563 steps) — mid-training trajectory point.
3. `checkpoints/arm_b_epoch5.ckpt`: After **5 epochs** (121,695 examples seen, ~7,606 steps) — final planned training checkpoint.
4. `checkpoints/arm_b_best.ckpt`: Model state corresponding to minimum metalloprotein validation loss.

---

## 5. Pre-Registered Predictions

| Metric | Arm A (Status Quo) | Arm B Prediction (Fine-Tune) | Native Ceiling (C1) |
|---|---|---|---|
| **Valid-coordination rate** | 19.98% | **< 28.0%** (no major closure of 19.98% $\rightarrow$ 77.17% gap) | 77.17% |
| **Primary violation (V1/V2) rate** | 18.38% | **12.0% – 22.0%** (minor reduction from density fitting) | 0.00% |
| **Angular RMSD to ideal geometry** | 27.6° | **> 22.0°** (cannot learn angular coordination constraints) | ~11.4° |
| **Mean coordination count** | ~0.35 | **< 0.60** (remains severely under-coordinated) | 1.87 |

### Decision Rule
- If Arm B valid-coordination rate remains $\le 30.0\%$, the data-scarcity hypothesis is falsified, confirming the defect is representation-bound.
- If Arm B valid-coordination rate exceeds $50.0\%$, the representation-bottleneck hypothesis is rejected in favor of the data-scarcity hypothesis.

---

## Addendum (2026-08-17, recorded after training was already in progress — a deviation, not a silent correction)

**Training-set overlap with the base checkpoint's own pretraining data.** The claim in §2.1 that
this is "CrossDocked2020 training partition" data implicitly reads as *novel* exposure. It is
only partly novel. `data/metal_target_split.pt`'s `train_filenames` cover 14,329 unique PDB IDs;
checked against `timesplit_no_lig_or_rec_overlap_train` — the split file DiffSBDD's released
`crossdocked_fullatom_cond.ckpt` was itself pretrained on — **4,691 of those 14,329 PDB IDs
(32.7%) were already seen during the base checkpoint's pretraining.**

This does not invalidate the run (§2.1's leakage check, which guards against contaminating the
*133-target eval cohort*, is unaffected and remains correct). It does mean roughly a third of
Arm B's "fine-tuning exposure" is re-exposure to already-seen structures rather than new
metalloprotein signal. Decision, made explicitly rather than silently: **let the in-progress run
finish and report this fraction alongside the result**, rather than kill and restart on a
stricter novel-only subset. Revisit with a novel-only rerun only if Arm B's outcome is close to
the decision-rule boundary above, where the ambiguity would matter.

---

## Amendment 1 (2026-08-17, Pre-Registered Before Retraining Arm B)

**Strictly Novel Metalloprotein Retraining (Arm B Rerun):**
To cleanly address both the 32.7% seen-data caveat and the post-epoch-3 validation loss divergence observed in the initial run, Arm B is retrained on a strictly novel dataset partition:
1. **Filter:** All complexes whose receptor PDB ID appears in the base checkpoint pretraining splits (`DiffSBDD/data/timesplit_no_lig_or_rec_overlap_train` and `val`) are strictly excluded.
2. **Dataset Size:**
   - **Training:** Exactly **17,417 strictly novel metalloprotein complexes** across **2,254 unique PDB IDs** (100% novel to the base model, 0% leakage vs. 133 external test targets).
   - **Validation:** Exactly **1,647 strictly novel metalloprotein complexes**.
3. **Training & Checkpoint Strategy:**
   - **Epochs:** 5 full epochs (87,085 total examples seen, 5,440 optimizer steps).
   - **Checkpoints:** Saved at every epoch (`arm_b_epoch1.ckpt` through `arm_b_epoch5.ckpt`) plus `arm_b_best.ckpt` selected strictly on novel validation loss.
   - **Optimizer:** AdamW, `lr = 1.0e-4`, effective batch size = 16.

