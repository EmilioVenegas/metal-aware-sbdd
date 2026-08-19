#!/usr/bin/env python
"""Figure 3: Valid coordination rate against deleted catalytic Zn²⁺ across arms.

Usage:
    python docs/figures/fig3_valid_coordination.py [--outdir docs/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import (
    REPO, ACADEMIC_RC, ARM_COLORS, load_cohort, load_jsonl,
    cluster_bootstrap_ci, rate, per_target_rate, smarts_matches
)


def make_figure(outdir: Path, arm_a: list[dict], arm_b: list[dict], native: list[dict],
                xray: set[str], pdb_to_cluster: dict[str, str]) -> Path:
    match = smarts_matches(REPO / "results/step1/generation/sdf", xray)
    kept = sum(1 for r in arm_a if not r.get("unreadable") and r["pdb_id"] in xray
               and match.get((r["pdb_id"], r["mol_index"]), False))
    total = sum(1 for r in arm_a if not r.get("unreadable") and r["pdb_id"] in xray)
    retained = 100.0 * kept / total
    smarts_recs = [r for r in arm_a
                   if match.get((r["pdb_id"], r["mol_index"]), False)]

    F = "has_valid_coordination"
    bars = [
        ("Native ligands\n(C1 ceiling)", rate(native, F, xray),
         per_target_rate(native, F, xray), ARM_COLORS["native"], False),
        (f"Arm A + SMARTS filter\n(post-hoc, keeps {retained:.0f}% of molecules)",
         rate(smarts_recs, F, xray),
         per_target_rate(arm_a, F, xray, subset=match), ARM_COLORS["smarts"], True),
        ("Arm A\nbase DiffSBDD", rate(arm_a, F, xray),
         per_target_rate(arm_a, F, xray), ARM_COLORS["arm_a"], False),
        ("Arm B\nfine-tuned, metal-blind", rate(arm_b, F, xray),
         per_target_rate(arm_b, F, xray), ARM_COLORS["arm_b"], False),
    ]

    with plt.rc_context(ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        ypos = np.arange(len(bars))[::-1]
        for y, (_, pooled, pertgt, color, is_subset) in zip(ypos, bars):
            by_cluster: dict[str, list[float]] = {}
            for pdb, v in pertgt.items():
                by_cluster.setdefault(pdb_to_cluster[pdb], []).append(v)
            cmean = float(np.mean([np.mean(v) for v in by_cluster.values()]))
            lo, hi = cluster_bootstrap_ci(pertgt, pdb_to_cluster)
            ax.barh(y, cmean, height=0.52, color=color, alpha=0.88,
                    edgecolor="#222222", linewidth=0.7, zorder=2)
            ax.errorbar(cmean, y, xerr=[[max(cmean - lo, 0)], [max(hi - cmean, 0)]],
                        fmt="none", ecolor="#1a1a1a", elinewidth=1.1, capsize=3.5,
                        capthick=1.1, zorder=3)
            ax.plot([pooled], [y], marker="D", ms=5.4, mfc="white", mec="#1a1a1a",
                    mew=1.2, ls="none", zorder=4)
            # Structured annotation: cluster mean % [CI] and pooled %
            label_x = max(hi, cmean, pooled) + 2.5
            ax.text(label_x, y,
                    f"{cmean:.1f}%  [{lo:.1f}, {hi:.1f}%]\n◆ pooled {pooled:.2f}%",
                    fontsize=8.8, color="#1f2937", va="center", linespacing=1.35, zorder=3)

        ax.set_yticks(ypos)
        ax.set_yticklabels([b[0] for b in bars], fontsize=9.2)
        ax.set_xlim(0, 132)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.grid(True, color="#e5e7eb", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.set_xlabel("Molecules with ≥1 valid metal coordination (%)", fontsize=9.5)

        # In-plot minimal key
        key_elements = [
            Line2D([], [], color="#444444", lw=5, alpha=0.8, label="Cluster-mean (95% CI)"),
            Line2D([], [], marker="D", color="none", mfc="white", mec="#1a1a1a", mew=1.2,
                   ms=5.2, label="Pooled rate"),
        ]
        ax.legend(handles=key_elements, loc="lower right", frameon=True, facecolor="#fbfbfb",
                  edgecolor="#dcdcdc", framealpha=0.94, fontsize=8.2, borderaxespad=0.8,
                  handletextpad=0.5, labelspacing=0.35)

        fig.text(0.5, 0.97, "Valid coordination against the deleted catalytic Zn²⁺",
                 ha="center", va="top", fontsize=11.5, weight="bold")
        fig.text(0.5, 0.91, "primary X-ray cohort: 21 sequence clusters, 127 targets, "
                             "100 valid molecules per target", ha="center", va="top",
                 fontsize=8.6, color="#555555")

        fig.subplots_adjust(left=0.34, right=0.98, top=0.85, bottom=0.14)
        out = outdir / "valid_coordination_by_arm.png"
        fig.savefig(out)
        plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    _, xray, clusters = load_cohort()
    arm_a = load_jsonl(REPO / "results/step1/checker/generated.jsonl")
    arm_b = load_jsonl(REPO / "results/step2/arm_b_generation/checker_results.jsonl")
    native = load_jsonl(REPO / "results/step1/checker/native_c1.jsonl")
    pdb_to_cluster = {p: f"C{i+1:02d}" for i, m in enumerate(clusters) for p in m}

    make_figure(outdir, arm_a, arm_b, native, xray, pdb_to_cluster)


if __name__ == "__main__":
    main()
