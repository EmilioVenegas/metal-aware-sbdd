#!/usr/bin/env python
"""Figure 4: Where generated ligand atoms land relative to deleted catalytic metal.

Usage:
    python docs/figures/fig4_distance_distribution.py [--outdir docs/figures] [--kde-sigma 0.10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from common import (
    REPO, ACADEMIC_RC, ARM_COLORS, VALID_WINDOW, V1_CLASH, SHELL,
    load_cohort, load_jsonl
)


def make_figure(outdir: Path, arm_a: list[dict], arm_b: list[dict], native: list[dict],
                xray: set[str], kde_sigma: float = 0.10) -> Path:
    series = [
        ("Native ligands", native, ARM_COLORS["native"]),
        ("Arm A — base DiffSBDD", arm_a, ARM_COLORS["arm_a"]),
        ("Arm B — fine-tuned, metal-blind", arm_b, ARM_COLORS["arm_b"]),
    ]
    bins = np.arange(0.8, 8.01, 0.2)
    width = bins[1] - bins[0]
    grid = np.linspace(bins[0], bins[-1], 800)

    with plt.rc_context(ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        peak = 0.0
        for label, recs, color in series:
            d = np.array([r["min_dist_to_metal"] for r in recs
                          if not r.get("unreadable") and r["pdb_id"] in xray])
            beyond = 100.0 * float(np.mean(d > bins[-1]))
            w = np.ones_like(d) * 100.0 / len(d)
            ax.hist(d, bins=bins, weights=w, histtype="step", lw=0.7, color=color,
                    alpha=0.40, zorder=2)
            kde = gaussian_kde(d, bw_method=kde_sigma / float(np.std(d)))
            dens = kde(grid) * width * 100.0
            ax.fill_between(grid, dens, color=color, alpha=0.10, lw=0, zorder=1)
            ax.plot(grid, dens, color=color, lw=1.7, zorder=3,
                    label=f"{label}  (n={len(d):,}; {beyond:.0f}% beyond {bins[-1]:.0f} Å)")
            peak = max(peak, float(dens.max()))

        top = peak * 1.26
        ax.set_ylim(0, top)
        ax.set_xlim(bins[0] - 0.1, bins[-1] + 0.1)

        ax.axvspan(*VALID_WINDOW, color="#1a9850", alpha=0.10, zorder=0)
        ax.axvline(V1_CLASH, color="#b2182b", ls="--", lw=1.0, zorder=1)
        ax.axvline(SHELL, color="#666666", ls=":", lw=1.0, zorder=1)
        ax.text(np.mean(VALID_WINDOW) + 0.12, peak * 1.10, "valid Zn–N/O\n1.90–2.35 Å",
                ha="center", va="bottom", fontsize=8.5, color="#1a7d3c")
        ax.text(V1_CLASH - 0.07, top * 0.34, "clash < 1.70 Å", rotation=90, ha="right",
                va="bottom", fontsize=8.5, color="#b2182b")
        ax.text(SHELL + 0.09, top * 0.34, "first shell 2.70 Å", rotation=90, ha="left",
                va="bottom", fontsize=8.5, color="#555555")

        ax.set_xlabel("distance from Zn²⁺ to the nearest ligand heavy atom (Å)", fontsize=9.5)
        ax.set_ylabel("molecules (%)", fontsize=9.5)
        fig.text(0.5, 0.97, "Where generated ligand atoms land relative to the deleted metal",
                 ha="center", va="top", fontsize=11.5, weight="bold")
        fig.text(0.5, 0.91, "primary X-ray cohort: 12,700 molecules per generative arm",
                 ha="center", va="top", fontsize=8.6, color="#555555")
        ax.legend(frameon=False, loc="upper right", borderaxespad=0.6)
        fig.subplots_adjust(left=0.085, right=0.975, top=0.86, bottom=0.14)
        out = outdir / "zn_distance_distribution.png"
        fig.savefig(out)
        plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--kde-sigma", type=float, default=0.10)
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    _, xray, _ = load_cohort()
    arm_a = load_jsonl(REPO / "results/step1/checker/generated.jsonl")
    arm_b = load_jsonl(REPO / "results/step2/arm_b_generation/checker_results.jsonl")
    native = load_jsonl(REPO / "results/step1/checker/native_c1.jsonl")

    make_figure(outdir, arm_a, arm_b, native, xray, kde_sigma=args.kde_sigma)


if __name__ == "__main__":
    main()
