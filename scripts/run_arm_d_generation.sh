#!/bin/bash
# Step 2 Arm D: inference-time coordination constraint on the Arm C checkpoint.
#
# Runs TWO generation passes, because the Arm D endpoint that looks most impressive is
# satisfied by arithmetic. The seed sits at 2.05 A and COORD_RANGES[("ZN","O")] is
# (1.85, 2.30), so a one-atom "molecule" scores 100.00% valid coordination and 13.22 deg
# angular RMSD over the 133-target cohort -- better than the 18.04 deg native ceiling,
# with no model involved. See results/step2/ANALYSIS_PLAN_ARMD.md section 4.
#
#   PASS 1  --seed-mode open    Arm D proper: seed on the open coordination vector.
#   PASS 2  --seed-mode random  Control S1: same donor, same 2.05 A, random direction.
#
# S1 is the control that could detect the positive. On the 9ZSN pilot both passes scored
# 100% as-scored valid coordination -- identical -- while V1 clash was 20% vs 58% and
# multi-fragment rate 50% vs 100%. Only endpoints where PASS 1 separates from PASS 2 are
# attributable to the model.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY=~/.conda/envs/atomica-interface/bin/python
mkdir -p logs results/step2/arm_d_generation results/step2/arm_d_control_random

# Amendment 1 (ANALYSIS_PLAN_ARMD.md, 2026-08-19): cluster-stratified sampling.
# The analysis weights clusters EQUALLY, but the X-ray stratum is severely skewed -- C07
# holds 50 of 127 targets, C03 holds 32, and 11 clusters hold exactly one. Sequential
# generation needs 117 of 133 targets just to touch all 21 clusters. Capping at 3
# targets/cluster covers 21/21 clusters in 42 targets and costs 0.11 pp of MDE
# (4.59 -> 4.70), measured by subsampling the published Arm C vs Arm A contrast.
# The saving lets control S1 run at N=100 instead of N=25, improving its MDE from
# 6.77 pp to 4.70 pp. Endpoints, thresholds and decision rules are unchanged.
# Set CLUSTER_CAP=0 to fall back to the full 133-target cohort.
CLUSTER_CAP="${CLUSTER_CAP:-3}"
N_VALID="${N_VALID:-100}"
N_VALID_CONTROL="${N_VALID_CONTROL:-100}"
BATCH="${BATCH:-20}"

run_pass () {
    local mode="$1" outdir="$2" log="$3" nvalid="$4"
    echo "=========================================================="
    echo "Arm D pass: seed-mode=${mode}  ->  ${outdir}"
    echo "=========================================================="
    $PY -u scripts/generate_arm_d.py \
        --cohort external_zn \
        --targets data/external_zn_test_clean.pt \
        --struct-dir data/external_pdbs \
        --protein-donors data/protein_donors.json \
        --outdir "$outdir" \
        --checkpoint checkpoints/arm_c_best.ckpt \
        --donor-element O \
        --donor-dist 2.05 \
        --resamplings 1 \
        --seed-mode "$mode" \
        --cluster-cap "$CLUSTER_CAP" \
        --n-valid "$nvalid" \
        --batch-size "$BATCH" \
        --num-shards 1 \
        --shard 0 > "$log" 2>&1

    echo "Scoring ${mode} pass at N=${nvalid} (seed-excluded primary endpoint)..."
    $PY scripts/score_arm_d.py \
        --manifest "$outdir/generation_manifest_shard0.jsonl" \
        --sdf-dir "$outdir/sdf" \
        --protein-donors data/protein_donors.json \
        --source "generated_arm_d_${mode}" \
        --out "$outdir/checker_results.jsonl"
}

run_pass open   results/step2/arm_d_generation     logs/gen_arm_d.log         "$N_VALID"
run_pass random results/step2/arm_d_control_random logs/gen_arm_d_control.log "$N_VALID_CONTROL"

echo "Arm D + control S1 complete."
echo "Report only endpoints where the open pass separates from the random control."
