#!/usr/bin/env python
"""Master runner to regenerate all figures for README.md.

Every number plotted here is recomputed directly from committed analysis artifacts.
Individual figures can also be generated and tweaked independently:
    python docs/figures/fig1_coordination_site.py
    python docs/figures/fig2_native_ligands.py
    python docs/figures/fig3_valid_coordination.py
    python docs/figures/fig4_distance_distribution.py
    python docs/figures/fig5_coordination_spheres.py

Usage:
    python docs/figures/make_readme_figures.py [--outdir docs/figures] [--figure all|1|2|3|4|5]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure local directory is on path when running from any cwd
FIGURES_DIR = Path(__file__).resolve().parent
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

from common import REPO, load_cohort, load_jsonl
import fig1_coordination_site
import fig2_native_ligands
import fig3_valid_coordination
import fig4_distance_distribution
import fig5_coordination_spheres


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--figure", choices=["all", "1", "2", "3", "4", "5"], default="all",
                    help="Which figure to generate (default: all)")
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    targets, xray, clusters = load_cohort()
    arm_a = load_jsonl(REPO / "results/step1/checker/generated.jsonl")
    arm_b = load_jsonl(REPO / "results/step2/arm_b_generation/checker_results.jsonl")
    arm_c = load_jsonl(REPO / "results/step2/arm_c_generation/checker_results.jsonl")
    arm_d = load_jsonl(REPO / "results/step2/arm_d_generation/checker_results.jsonl")
    native = load_jsonl(REPO / "results/step1/checker/native_c1.jsonl")
    pdb_to_cluster = {p: f"C{i+1:02d}" for i, m in enumerate(clusters) for p in m}

    if args.figure in ("all", "1"):
        fig1_coordination_site.make_figure(outdir, targets)
    if args.figure in ("all", "2"):
        fig2_native_ligands.make_figure(outdir, targets, xray, clusters)
    if args.figure in ("all", "3"):
        fig3_valid_coordination.make_figure(outdir, arm_a, arm_b, arm_c, arm_d, native, xray, pdb_to_cluster)
    if args.figure in ("all", "4"):
        fig4_distance_distribution.make_figure(outdir, arm_a, arm_b, arm_c, arm_d, native, xray)
    if args.figure in ("all", "5"):
        fig5_coordination_spheres.make_figure(outdir, arm_a, arm_b, arm_c, arm_d, native, xray)
if __name__ == "__main__":
    main()
