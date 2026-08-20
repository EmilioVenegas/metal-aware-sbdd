#!/usr/bin/env python
"""Figure 5: Cross-section through the catalytic Zn²⁺ coordination sphere with aligned bilateral radial profiles.

Usage:
    python docs/figures/fig5_coordination_spheres.py [--outdir docs/figures] [--n-show 600] [--rmax 4.6] [--seed 7]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from common import (
    REPO, ACADEMIC_RC, ARM_COLORS, VALID_WINDOW, V1_CLASH, SHELL, ZN_IONIC_RADIUS,
    load_cohort, load_jsonl
)


def make_figure(outdir: Path, arm_a: list[dict], arm_b: list[dict], arm_c: list[dict],
                arm_d: list[dict], native: list[dict],
                xray: set[str], n_show: int = 600, rmax: float = 4.6, seed: int = 7) -> Path:
    """Cross-section through coordination sphere (top) and aligned bilateral density profile (bottom)."""
    panels = [
        ("Native ligands", native, ARM_COLORS["native"]),
        ("Arm C: metal-aware", arm_c, ARM_COLORS["arm_c"]),
        ("Arm D: inference seed", arm_d, ARM_COLORS["arm_d"]),
        ("Arm A: status quo", arm_a, ARM_COLORS["arm_a"]),
        ("Arm B: metal-blind", arm_b, ARM_COLORS["arm_b"]),
    ]
    rng = np.random.default_rng(seed)

    grid_sym = np.linspace(-rmax, rmax, 600)
    max_dens_sym = 0.0
    data_dict = {}

    for label, recs, color in panels:
        d = np.array([r["min_dist_to_metal"] for r in recs
                      if not r.get("unreadable") and r["pdb_id"] in xray])
        kde = gaussian_kde(d, bw_method=0.10 / float(np.std(d)))
        dens_sym = kde(np.abs(grid_sym)) * 0.2 * 100.0
        data_dict[label] = (d, dens_sym)
        max_dens_sym = max(max_dens_sym, float(dens_sym.max()))

    with plt.rc_context(ACADEMIC_RC):
        fig = plt.figure(figsize=(15.8, 5.8))
        gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.44], hspace=0.28, wspace=0.15,
                               left=0.04, right=0.985, top=0.81, bottom=0.09)
        top_y = max_dens_sym * 1.15

        for i, (label, recs, color) in enumerate(panels):
            ax_top = fig.add_subplot(gs[0, i])
            ax_bot = fig.add_subplot(gs[1, i])
            d, dens_sym = data_dict[label]
            pct_valid = 100.0 * np.mean((d >= VALID_WINDOW[0]) & (d <= VALID_WINDOW[1]))
            pct_clash = 100.0 * np.mean(d < V1_CLASH)
            pct_out = 100.0 * np.mean(d > rmax)

            # Top panel: 2D circular cross-section
            ax_top.add_patch(plt.Circle((0, 0), SHELL, facecolor="#f4f4f4", edgecolor="none", zorder=0))
            ax_top.add_patch(plt.Circle((0, 0), VALID_WINDOW[1], facecolor="#1a9850", alpha=0.16, edgecolor="none", zorder=0))
            ax_top.add_patch(plt.Circle((0, 0), VALID_WINDOW[0], facecolor="white", edgecolor="none", zorder=0))
            ax_top.add_patch(plt.Circle((0, 0), V1_CLASH, facecolor="#b2182b", alpha=0.14, edgecolor="none", zorder=0))
            for r_ring, ls, c in [(V1_CLASH, "--", "#b2182b"), (VALID_WINDOW[0], "-", "#1a7d3c"),
                                  (VALID_WINDOW[1], "-", "#1a7d3c"), (SHELL, ":", "#666666")]:
                ax_top.add_patch(plt.Circle((0, 0), r_ring, fill=False, ls=ls, lw=0.8, edgecolor=c, zorder=2))

            sub = d if len(d) <= n_show else rng.choice(d, n_show, replace=False)
            theta = rng.uniform(0, 2 * np.pi, len(sub))
            rad = np.clip(sub, 0, rmax + 0.12)
            ax_top.scatter(rad * np.cos(theta), rad * np.sin(theta), s=7, color=color,
                           alpha=0.45, linewidths=0, zorder=3)
            ax_top.add_patch(plt.Circle((0, 0), ZN_IONIC_RADIUS, facecolor="#7b3294",
                                        edgecolor="black", lw=0.7, zorder=4))
            ax_top.text(0, 0, "Zn", color="white", ha="center", va="center", fontsize=7.5,
                        weight="bold", zorder=5)

            ax_top.set_xlim(-rmax - 0.45, rmax + 0.45)
            ax_top.set_ylim(-rmax - 0.45, rmax + 1.05)
            ax_top.set_aspect("equal")
            ax_top.set_axis_off()
            ax_top.set_title(f"{label}\n1.90–2.35 Å: {pct_valid:.1f}%   ·   < 1.70 Å: {pct_clash:.1f}%",
                             fontsize=8.2, pad=5)
            ax_top.text(0, -rmax - 0.30, f"{pct_out:.0f}% beyond {rmax:.1f} Å",
                        ha="center", va="top", fontsize=7.5, color="#666666")

            # Bottom panel: Aligned bilateral mirrored profile spanning [-rmax, +rmax]
            ax_bot.axvspan(-VALID_WINDOW[1], -VALID_WINDOW[0], color="#1a9850", alpha=0.12, zorder=0)
            ax_bot.axvspan(VALID_WINDOW[0], VALID_WINDOW[1], color="#1a9850", alpha=0.12, zorder=0)
            ax_bot.axvspan(-V1_CLASH, V1_CLASH, color="#b2182b", alpha=0.08, zorder=0)
            ax_bot.axvline(-V1_CLASH, color="#b2182b", ls="--", lw=0.8, zorder=1)
            ax_bot.axvline(V1_CLASH, color="#b2182b", ls="--", lw=0.8, zorder=1)
            ax_bot.axvline(-SHELL, color="#666666", ls=":", lw=0.8, zorder=1)
            ax_bot.axvline(SHELL, color="#666666", ls=":", lw=0.8, zorder=1)
            ax_bot.axvline(0, color="#7b3294", ls="-", lw=1.0, alpha=0.6, zorder=1)

            ax_bot.fill_between(grid_sym, dens_sym, color=color, alpha=0.15, zorder=2)
            ax_bot.plot(grid_sym, dens_sym, color=color, lw=1.5, zorder=3)

            ax_bot.set_xlim(-rmax, rmax)
            ax_bot.set_ylim(0, top_y)
            ax_bot.set_xticks([-4.6, -2.35, 0, 2.35, 4.6])
            ax_bot.set_xticklabels(["4.6", "2.35", "Zn", "2.35", "4.6"], fontsize=7.8)
            ax_bot.set_xlabel("radial cross-section (Å)", fontsize=8.2, labelpad=2)
            if i == 0:
                ax_bot.set_ylabel("density (%)", fontsize=8.2)
            else:
                ax_bot.set_ylabel("")
                ax_bot.tick_params(labelleft=False)

        key = [Line2D([], [], color="#666666", ls=":", lw=1.1, label="2.70 Å first shell"),
               Line2D([], [], color="#1a7d3c", ls="-", lw=1.1, label="1.90–2.35 Å valid"),
               Line2D([], [], color="#b2182b", ls="--", lw=1.1, label="1.70 Å clash"),
               Line2D([], [], color="#7b3294", marker="o", ls="none", ms=4.5, label="Zn²⁺ (0.74 Å)")]
        leg = fig.axes[0].legend(
            handles=key, loc="upper left", frameon=True, fancybox=True,
            facecolor="#f6f6f8", edgecolor="#d5d5db", framealpha=0.96,
            fontsize=7.2, handlelength=1.4, handletextpad=0.5, labelspacing=0.38,
            borderaxespad=0.3, borderpad=0.45
        )
        leg.get_frame().set_linewidth(0.7)

        fig.text(0.5, 0.97, "Cross-section through the catalytic Zn²⁺ coordination sphere & radial profiles",
                 ha="center", va="top", fontsize=11.2, weight="bold")
        fig.text(0.5, 0.915, "Top: 600 sampled molecules plotted radially; Bottom: mirrored full cohort radial density",
                 ha="center", va="top", fontsize=8.4, color="#555555")

        out = outdir / "zn_coordination_spheres.png"
        fig.savefig(out)
        plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--n-show", type=int, default=600)
    ap.add_argument("--rmax", type=float, default=4.6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    targets, xray, _ = load_cohort()
    arm_a = load_jsonl(REPO / "results/step1/checker/generated.jsonl")
    arm_b = load_jsonl(REPO / "results/step2/arm_b_generation/checker_results.jsonl")
    arm_c = load_jsonl(REPO / "results/step2/arm_c_generation/checker_results.jsonl")
    arm_d = load_jsonl(REPO / "results/step2/arm_d_generation/checker_results.jsonl")
    native = load_jsonl(REPO / "results/step1/checker/native_c1.jsonl")

    make_figure(outdir, arm_a, arm_b, arm_c, arm_d, native, xray, n_show=args.n_show, rmax=args.rmax, seed=args.seed)
if __name__ == "__main__":
    main()
