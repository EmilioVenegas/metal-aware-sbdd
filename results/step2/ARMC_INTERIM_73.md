# Arm C — interim (unblinded) look at 73/133 targets

**Date:** 2026-08-19. **Status: INTERIM, NOT THE REGISTERED RESULT.**
Generation (`scripts/generate_step1.py --checkpoint checkpoints/arm_c_best.ckpt`) was still
running when this was written: 73 of 133 targets complete, 7,300 molecules. ETA for the
remaining 60 targets ≈ 10.9 h at the trailing-20 mean of 653 s/target.

This document exists because `ANALYSIS_PLAN_ARMC.md` is a pre-registration and an unblinded
interim look must be on the record, not silent. **No threshold, endpoint, or decision rule in
`ANALYSIS_PLAN_ARMC.md` is changed by this document.** The registered analysis runs on all 133
targets with the cluster-level machinery, unchanged.

Interim scoring used the unmodified Step 1 checker
(`scripts/coordination_checker.py`, same thresholds, same `data/protein_donors.json`), output
at `results/step2/arm_c_generation/interim_checker_73.jsonl` (source label
`generated_arm_c_interim`, kept out of `checker_results.jsonl` so the final run is untouched).

---

## 0. The subset is not a random sample — read every number below with this caveat

Targets are processed in descending PDB-ID order, so the completed 73 cover **11 of 26
clusters**, and **60 of the 73 come from two clusters** (C07 n=35, C03 n=25). Thirteen
single- or double-member clusters and the 9-member C14 and 5-member C15 are entirely
unsampled. Per-target means are therefore dominated by two protein families; the cluster-level
mean (the registered unit of analysis) is already visibly smaller than the per-target mean, and
the final 133-target number can move substantially in either direction.

## 1. Matched-subset endpoints (same 73 targets for every arm)

All arms re-scored on the 73 completed targets — **not** the published full-cohort headlines,
which are not comparable to a partial run.

| endpoint | Arm A | Arm B | **Arm C (interim)** | Native C1 | Arm C pre-registered |
|---|---|---|---|---|---|
| valid-coordination rate | 17.67% | 8.53% | **22.53%** | 76.71% | **> 35%** |
| primary violation (V1∨V2) | 12.51% | 8.33% | **19.48%** | 26.03% | **< 15%** |
| V2-strict | 9.78% | 6.99% | **15.19%** | 1.37% | — |
| V1 hard clash (<1.70 Å) | 3.96% | 1.58% | **7.64%** | 0.00% | — |
| mean valid coordination count | 0.184 | 0.089 | **0.239** | 0.863 | **> 0.7** |
| mean shell contacts (<2.70 Å) | 0.589 | 0.323 | **0.802** | 1.781 | — |
| mean min-dist to metal (Å) | 3.45 | 4.61 | **3.19** | 2.02 | — |
| angular RMSD, molecules with ≥1 valid coordination | 16.36° | 15.78° | **17.93°** | 16.02° | **< 20°** |
| angular RMSD, all molecules with a defined angle | 26.07° | 27.72° | **25.74°** | 19.07° | — |

Per-target paired differences on the 73: Arm C − Arm A = **+4.86 pp** valid coordination
(sd 6.12) and **+6.97 pp** primary violation (sd 8.27). At the registered cluster level
(11 clusters here): **+2.47 pp** valid coordination (sd 6.89) — the CI plainly straddles zero
at this cluster count.

## 2. Distance-shell redistribution — the mechanistic read

Fraction of molecules by distance from the nearest ligand heavy atom to the catalytic Zn:

| bin (Å) | Arm A | Arm B | **Arm C** |
|---|---|---|---|
| < 1.70 (hard clash) | 4.0% | 1.6% | **7.6%** |
| 1.70 – 1.90 | 4.4% | 1.7% | **6.3%** |
| 1.90 – 2.35 (Zn–N/O valid window) | 19.2% | 9.9% | **22.9%** |
| 2.35 – 2.70 | 15.6% | 10.5% | **16.8%** |
| 2.70 – 3.50 | 23.0% | 19.4% | **19.8%** |
| 3.50 – 5.00 | 18.2% | 18.5% | **12.9%** |
| > 5.00 | 15.6% | 38.4% | **13.6%** |

Nearest-atom element when within 2.70 Å (counts): Arm A `O 2522, C 291, N 252, S 57, F 22`
(n=3,144); Arm C `O 2982, N 510, C 356, S 42, F 22, Cl 5, P 2` (n=3,919). Nitrogen donors in
the shell **doubled**.

Read plainly: **the metal atom type does change model behaviour, in the predicted direction —
the model now places atoms at the metal.** Shell occupancy rises 43.2% → 53.6%, mean min-dist
falls 3.45 → 3.19 Å, N-donor contacts double. But the *distance calibration* does not come with
it: the sub-1.70 Å clash bin grows almost as fast as the valid window (+3.7 pp vs +3.7 pp), so
violations rise in step with valid coordinations and the net gain in the registered primary
endpoint is small.

Arm B is the reverse and remains a clean negative: it moves ligands *away* from the metal
(38.4% of molecules > 5 Å, vs 15.6% for Arm A) — more metalloprotein data through a metal-blind
representation teaches avoidance, not coordination.

## 3. Three prior-document errors this look surfaced (all must be fixed before final scoring)

1. **`ANALYSIS_PLAN_ARMC.md` §6 lists the native-ceiling primary-violation rate as 0.00%.**
   That is the V1 hard-clash rate, not the primary (V1∨V2) rate. Native C1 primary violation is
   **19.55%** on the full cohort (26.03% on this 73-target subset), because native ligand donors
   legitimately occupy the shell without falling inside the pre-registered distance window.
   V2-strict, the chelate-aware form, is 1.37% for natives. Any "fraction of the A→C1 gap
   closed" computed on the primary-violation endpoint against a 0.00% ceiling is wrong.
2. **`ARMB_EVALUATION.md` §5 reports Arm B angular RMS deviation as `Mean = 0.00°,
   Median = 0.00°`, scored "CONFIRMED" against a registered `> 22.0°`.** That number is not
   reproducible from `arm_b_generation/checker_results.jsonl`: 6,854 records carry a non-null
   angle, none of them zero, mean **26.88°**, median 29.28°; conditioning on ≥1 valid
   coordination gives 16.31° over 1,327 molecules. The registered verdict survives
   (26.88° > 22°) but the reported figure is wrong and must be corrected in that file.
   **The denominator does not need a new decision and must not be re-chosen now:**
   `ANALYSIS_PLAN_ARMC.md` §6 fixes the comparator as Arm A's `27.6° (all cohort) /
   25.19° (primary X-ray)`, which are all-molecule values, so the registered Arm C `< 20°`
   is an all-molecule endpoint. On that denominator Arm C is **25.74°** on the interim
   subset — failing, and essentially unchanged from Arm A's 26.07°. The conditional
   denominator (17.93°, molecules with ≥1 valid coordination) is a diagnostic only, and
   is reported as such because it shows the *shape* of the shift: geometry conditional on
   coordinating is no better either.
3. **`ARMB_EVALUATION.md`'s native-ceiling column gives "Mean Coordination Count 1.87" and
   "Angular RMSD ~11.40°". Neither reproduces from `native_c1.jsonl`.** On the primary X-ray
   cohort (n=127 native ligands) the checker gives mean `n_valid_coordination` = **0.874**,
   mean `coordination_number_total` = 4.29 (the metal's whole shell, protein donors included),
   mean `n_shell_contacts` = 1.748, and angular RMSD **18.04°** all-molecule / 14.94°
   conditional on ≥1 valid coordination. No field-and-denominator combination in the checker
   output yields 1.87 or 11.40°. Note also that the same table row mixes definitions: Arm B's
   "0.11" *is* `n_valid_coordination`, so the native "1.87" is not the same quantity even if
   its provenance is found. Use recomputed values; do not propagate these two.

## 4. Intervention integrity — verified during the run, PASS

The concern that Arm C might be silently generating with a metal-blind pocket (which would make
the whole run Arm B with LoRA) was checked directly against the live checkpoint:

- `checkpoints/arm_c_best.ckpt` `hyper_parameters`: `dataset='crossdock_metal'`,
  `pocket_representation='full-atom'`.
- `pocket_type_encoder` is the 16-entry `aa_encoder` (`… 'Zn':10, 'Mg':11, 'Fe':12, 'Mn':13,
  'Ca':14, 'Cu':15`); `lig_type_encoder` is the unchanged 10-entry dict; **distinct objects** —
  the ligand generator cannot emit Zn.
- `utils.get_pocket_from_ligand` → `prepare_pocket` on the first 12 completed targets yields
  1–2 metal-class atoms per pocket, and the class-10 atom sits at **0.00 Å** from the
  `zn_coord` the checker scores against. The metal the evaluation measures is the metal the
  model sees.

## 4b. The pre-registered kill baseline is currently live — SMARTS ≈ Arm C on rate

`docs/plan.md` names the post-hoc zinc-binding-group SMARTS filter as "the most likely real
threat," and `docs/step3_smarts_baseline.md` already ran it on the full Arm A cohort (24.96%
valid coordination among 5,116 retained molecules). Re-run here on the **same 73 targets**, so
it is comparable to the interim Arm C number (`scripts/run_smarts_baseline.py`'s five ZBG
patterns, applied per molecule and joined to the checker records by `mol_index`):

| | retained | valid-coord **rate** among retained | valid-coord **yield** per generated molecule |
|---|---|---|---|
| Arm A, unfiltered | 7,300 (100%) | 17.67% | 17.67% |
| **Arm A + SMARTS** | **2,837 (38.9%)** | **21.89%** | **8.51%** |
| **Arm C, unfiltered** | 7,300 (100%) | **22.53%** | **22.53%** |
| Arm C + SMARTS | 2,446 (33.5%) | 27.56% | 9.23% |

**On the rate criterion the kill baseline matches Arm C** (21.89% vs 22.53%) — a cheap
substructure filter over the *unmodified* base model recovers as much coordination validity as
the representation fix does. That is exactly the pre-registered failure condition.

The one distinction that survives, and it must be stated carefully rather than leaned on: the
SMARTS baseline is a *filter*, discarding 61% of generations, so its yield of validly
coordinating molecules per molecule generated is **8.51% against Arm C's 22.53%** — 2.6×. If
the final 133-target result reproduces this pattern, the defensible claim is about *yield at
fixed sampling cost*, not about validity rate, and the paper's framing has to change
accordingly. It is not a rescue of the "representation fixes coordination chemistry" claim.

## 5. Leading explanation if the final number lands near this one

Not to be treated as established — these are the hypotheses to test, in order.

1. **Checkpoint selection was made on noise.** Per-epoch validation is a *single* stochastic
   VLB sample over 122 examples (`train_arm_c.py:evaluate_validation`, no fixed timestep or
   seed). The 20 epoch values are
   `[27.7, 15.7, 71.1, 40.5, 35.8, 34.1, -6.4, 57.3, 13.2, 31.4, 19.1, 30.8, 35.3, 48.1, 32.8, 55.3, 7.6, 15.3, 72.5, 60.4]`
   — mean 34.9, **sd 21.2**, range −6.4 to 72.5, with no trend. `arm_c_best.ckpt` is epoch 7,
   the single most extreme low outlier in a noise-dominated series. Training loss is flat at
   0.176–0.180 across all 20 epochs. **The molecules now being generated come from a checkpoint
   selected essentially at random among epochs.**
2. **The adapted layers cannot fix distances.** LoRA touches only `node_mlp.2` of
   `e_block_3`/`e_block_4` — the *feature* pathway. The coordinate-update pathway was
   deliberately left frozen to protect equivariance. The model can therefore change *what* it
   puts near the metal (N-donor count doubled) but has limited capacity to change *where* — the
   exact split the distance histogram shows.
3. **Scale.** 1,101 training complexes, 7,280 trainable parameters, 8.8 minutes of training.
   Undertraining is not excluded by any evidence in hand.

Per `ANALYSIS_PLAN_ARMC.md`'s own decision rule, a result in this range means "the
representation fix alone is insufficient — LoRA rank/target-layer choice, training data scale,
or the surgery itself would need to be investigated as the constraining factor **before**
concluding the representation-bottleneck hypothesis is wrong." That is the branch this interim
look points at, and (1) above is the first thing to rule out, because it is cheap: score
`arm_c_epoch20.ckpt` on a subset and see whether epoch choice moves the endpoint at all.
