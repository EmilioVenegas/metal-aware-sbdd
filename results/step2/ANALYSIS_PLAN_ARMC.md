# Pre-Registration Plan: Step 2 Arm C (Metal-Aware Pocket Representation, LoRA Fine-Tune)

**Date:** 2026-08-17
**Status:** COMMITTED BEFORE RUNNING TRAINING
**Context:** Ablation study testing whether the metal-coordination failure observed in Step 1 is
a representation bottleneck (Arm C) rather than a training-data deficit (Arm B, already run —
see `ARMB_RESULTS.md`).

---

## 1. Core Hypothesis & Objective

- **Arm C Question:** Does giving the model an atom type for the catalytic metal — retained in
  the pocket, with its own vocabulary entry — close the valid-coordination gap that Arm B
  (more data, same metal-blind representation) did not?
- **Hypothesis:** Yes, substantially, because the mechanism identified in Step 1 is specific:
  DiffSBDD has no atom representing the metal at all (`docs/plan.md`'s cross-model table), so no
  amount of additional metal-blind training data can teach it donor geometry around a position
  it cannot perceive. Arm C removes that specific bottleneck; Arm B could not.
- **Role in Ablation:** If Arm C closes the gap materially more than Arm B did, the claim is
  "the pocket representation is the bug, and it is inherited across the field" (plan.md). If Arm
  C performs similarly to Arm B, the representation fix alone is insufficient and the mechanism
  is more complicated than "the model cannot see the metal."

## 2. What Changed Relative to Arm B (engineering, already built and verified — see docs/step2.md §4)

- **Pocket vocabulary:** 10 → 16 element classes (+ Zn, Mg, Fe, Mn, Ca, Cu), genuinely decoupled
  from the ligand vocabulary (`dataset_params['crossdock_metal']`, `constants.py`).
- **Preprocessing:** metals retained in the pocket (`process_crossdock.py`, `utils.py`).
- **Weights:** `checkpoints/arm_c_surgered_init.ckpt` — base checkpoint weights copied in full
  except the resized `residue_encoder`/`residue_decoder` layers, whose overlapping block is
  copied and new rows/columns left at fresh initialization. Numerically verified identical to
  the base checkpoint on a metal-free pocket (`scripts/verify_arm_c_surgery.py`, max abs diff
  `0.000e+00`).
- **Training method:** LoRA (rank 8, alpha 16) on the final `node_mlp` layer of the last two
  `EquivariantBlock`s, plus the resized `residue_encoder`/`residue_decoder` left fully
  trainable (LoRA cannot adapt genuinely new parameters — see `docs/step2.md`'s gotcha).
  Gradient flow to both paths verified with two real optimizer steps
  (`scripts/verify_arm_c_gradient_flow.py`) — including confirming `residue_decoder` correctly
  receives *no* gradient (its output is discarded by `ConditionalDDPM`, traced in
  `conditional_model.py`), so it does not silently misrepresent what's actually being learned.
  **0.7% of parameters are trainable** (7,280 / 1,011,437) — LoRA's usual sense in which this is
  a much lighter-touch intervention than Arm B's full fine-tune.

## 3. Dataset (built for this arm — did not exist before, see `scripts/build_arm_c_dataset.py`)

**Not** the LMDB Arm B trains on (that one is metal-blind by construction — reused unmodified
because Arm B deliberately keeps the old representation). Arm C needs real metal-containing
pockets, which required building a new dataset:

- **Source:** each of the 1,512 RCSB-fetched receptor PDBs' own co-crystallized native ligand
  (not CrossDocked's cross-docked pose augmentation — no raw CrossDocked ligand SDFs exist on
  this machine; verified absent before choosing this path, see `build_arm_c_dataset.py`'s
  docstring).
- **Filter:** ≥1 sidechain-coordinated catalytic Zn in the receptor; a non-crystallization-
  additive HETATM group with ≥5 heavy atoms within 8.0 Å of a catalytic Zn, closest such group
  chosen when multiple qualify. Lighter-touch than Step 1's eval-cohort curation (no QED/
  resolution/mutant filters) — this is training data, not a benchmark claim.
- **Pocket construction:** the same (patched) `process_ligand_and_pocket` function Arm C
  training and future Arm C inference both call — training-time and inference-time pocket
  definitions are identical by construction, not just by intent.
- **Leakage:** any PDB ID present in the 133-target external Zn eval cohort is excluded before
  extraction (checked directly, not inherited from `GATE_CHECKS.md`'s separate check).
- **Sanity gate, enforced in the build script itself (`assert`, not just reported):** at least
  one example must show a retained metal atom in its encoded pocket, or the script aborts before
  saving anything. [Result recorded once the full-scale build finishes — see addendum below.]
- **Split:** deterministic 90/10 by PDB ID (sorted, every 10th to val — reproducible, no shuffle
  seed dependency).

## 4. Training Method & Parameters

- **LoRA:** rank 8, alpha 16 (scaling = 2.0), on `egnn.e_block_3.gcl_0.node_mlp.2` and
  `egnn.e_block_4.gcl_0.node_mlp.2`.
- **Fully trainable (not LoRA):** `residue_encoder` (on the loss path), `residue_decoder` (off
  the loss path for `ConditionalDDPM` — trainable for structural consistency, inert in practice).
- **Optimizer:** AdamW, `lr = 1e-3` — higher than Arm B's `1e-4` full-fine-tune LR, the standard
  reason being that only ~0.7% of parameters move here, versus 100% in Arm B.
- **Epochs:** 20 (vs. Arm B's 5) — the dataset is far smaller (~1,000-1,500 vs. 24,339
  examples), and LoRA + a small unfrozen sub-module typically need more passes over less data to
  converge. Checkpoints saved at ~epoch 5, ~epoch 10, and epoch 20 (final), plus best-val.
- **Effective batch size:** 16 (micro-batch 4 × accumulation 4), matching Arm B for comparability.

## 5. Evaluation Cohort (fixed from Step 1, unchanged from Arm B)

Scored on the identical 133-target, 26-cluster external Zn cohort
(`data/external_zn_test_clean.pt`) with the exact Step 1 checker harness
(`results/step1/checker/coordination_checker.py`), same thresholds, same N=100 valid
molecules/target.

## 6. Pre-Registered Predictions

| Metric | Arm A (Status Quo) | Arm B (Fine-Tune, observed) | Arm C Prediction (LoRA, metal-aware) | Native Ceiling (C1) |
|---|---|---|---|---|
| **Valid-coordination rate** | 19.98% | *pending checker eval* | **> 35%** (materially above both Arm A and Arm B's registered ceiling of 28%) | 77.17% |
| **Primary violation (V1/V2) rate** | 18.38% | *pending checker eval* | **< 15%** (below Arm A; metal now occupiable by a real donor rather than only by accident) | 0.00% |
| **Angular RMSD to ideal geometry** | 27.6° (all cohort) / 25.19° (primary X-ray, per `STEP1_RESULTS.md`) | *pending checker eval* | **< 20°** (some real coordination geometry should now be learnable, not just distance-matching) | ~11.4° |
| **Mean coordination count** | ~0.35 | *pending checker eval* | **> 0.7** | 1.87 |

**Note on Arm B's row:** Arm B's checker-based endpoints have not been measured yet (only its
training loss trajectory has — see `ARMB_RESULTS.md` §4). Both arms' checker evaluations should
be run and reported together so the three-way Arm A / B / C comparison is apples-to-apples, not
staggered across different evaluation sessions.

### Decision Rule

- If Arm C's valid-coordination rate exceeds Arm B's by a wide margin (informally, more than the
  gap between Arm A and Arm B, whatever that turns out to be) **and** clears the >35% threshold
  above, the representation-bottleneck hypothesis is supported: giving the model an atom type
  for the metal did what more metal-blind data could not.
- If Arm C performs comparably to Arm B (within the same rough range, both well short of >35%),
  the representation fix alone is insufficient — LoRA rank/target-layer choice, training data
  scale, or the surgery itself would need to be investigated as the constraining factor before
  concluding the representation-bottleneck hypothesis is wrong.
- If Arm C's valid-coordination rate exceeds native ligands' C1 ceiling in an implausible way, or
  if generation validity collapses (many fewer than 100 valid molecules per target reachable),
  that is itself a finding — likely an artifact of undertrained new vocabulary rows given the
  small dataset — and must be reported as such, not filtered out of the denominator.

---

## Addendum — dataset build result (recorded 2026-08-17)

- **Total Extracted Complexes:** 1,223 complexes (from 1,512 candidate PDBs).
- **Split:** 1,101 train examples (`data/arm_c_train.pt`) / 122 validation examples (`data/arm_c_val.pt`).
- **Retained Catalytic Metal Sanity Gate:** PASS. 1,100 / 1,101 train pockets (99.9%) and 121 / 122 val pockets (99.2%) contain at least one explicit metal atom in the expanded 16-element pocket encoding ($Z \ge 10$).
- **Zero Leakage:** 0 overlap with the 133 external Zn test targets.


---

## Addendum 2 — unblinded interim look, and one correction to §6 (recorded 2026-08-19)

**Disclosure.** An unblinded interim analysis was run on the 73 of 133 targets completed while
generation was still in progress: `results/step2/ARMC_INTERIM_73.md`, scored with the unmodified
Step 1 checker into `arm_c_generation/interim_checker_73.jsonl`. **No threshold, endpoint,
denominator, or decision rule above is changed by it.** The registered analysis remains the
133-target, 26-cluster, cluster-level analysis. The interim subset is not random — targets run in
descending PDB-ID order, so it covers 11 of 26 clusters with 60 of 73 targets drawn from two
(C07 n=35, C03 n=25) — and its point estimates are not the registered result.

**Correction to §6, factual, not a re-registration.** The Native-ceiling column gives
"Primary violation (V1/V2) rate = 0.00%". That is the native **V1 hard-clash** rate. The native
**primary (V1∨V2)** rate is **19.55%** on the full cohort — native ligand donors legitimately
occupy the 2.70 Å shell without landing inside the pre-registered distance window. Native
V2-strict is 1.37%. Arm C's registered `< 15%` prediction is unchanged; only the ceiling used for
any "fraction of the A→C1 gap closed" arithmetic on this endpoint is corrected.

**Denominator note, no new choice made.** §6's angular-RMSD comparator is Arm A's
"27.6° (all cohort) / 25.19° (primary X-ray)" — all-molecule values. The registered Arm C
`< 20°` is therefore an all-molecule endpoint and will be scored as one. Any conditional
(molecules with ≥1 valid coordination) angle reported alongside is a diagnostic, labelled as such.
