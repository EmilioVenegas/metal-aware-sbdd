#!/bin/bash
set -e

echo "Starting Arm C Generation..."
~/.conda/envs/atomica-interface/bin/python -u scripts/generate_step1.py \
    --cohort external_zn \
    --targets data/external_zn_test_clean.pt \
    --struct-dir data/external_pdbs \
    --outdir results/step2/arm_c_generation \
    --checkpoint checkpoints/arm_c_best.ckpt \
    --n-valid 100 \
    --batch-size 20 \
    --num-shards 1 \
    --shard 0 > logs/gen_arm_c.log 2>&1

echo "Arm C Generation complete. Running checker..."
~/.conda/envs/atomica-interface/bin/python scripts/coordination_checker.py \
    --targets data/external_zn_test_clean.pt \
    --sdf-dir results/step2/arm_c_generation/sdf \
    --source generated_arm_c \
    --out results/step2/arm_c_generation/checker_results.jsonl \
    --protein-donors data/protein_donors.json

echo "Checker complete."
