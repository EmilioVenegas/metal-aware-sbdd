#!/usr/bin/env python
"""Figure 1: Catalytic zinc site 3D coordination sphere (PDB 9ZSN).

Renders a publication-grade molecular visualization using PyMOL raytracing
and composites academic typography, annotations, and legends.

Usage:
    python docs/figures/fig1_coordination_site.py [--outdir docs/figures] [--pdb-id 9ZSN]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from PIL import Image

from common import REPO, ACADEMIC_RC, ELEMENT_COLOR, load_cohort


def render_pymol_scene(pdb_id: str, donors: dict, temp_png: Path) -> None:
    """Renders the coordination site via headless PyMOL raytracing."""
    import pymol
    from pymol import cmd

    cif_path = REPO / f"data/external_pdbs/{pdb_id}.cif"
    sdf_path = REPO / f"data/native_ligands/{pdb_id}.sdf"

    pymol.finish_launching(["pymol", "-cq"])
    cmd.reinitialize()

    cmd.load(str(cif_path), "prot")
    cmd.load(str(sdf_path), "lig")
    cmd.hide("everything")

    # Select coordination components
    res_nums = "+".join(str(d["residue"].split("_")[2]) for d in donors["protein_donors"])
    cmd.select("zn_site", "chain A and resn ZN")
    cmd.select("coord_res", f"chain A and resi {res_nums} and (sidechain or name CA)")
    cmd.select("lig_near", "lig within 6.5 of zn_site")
    cmd.select("bb_cartoon", f"byres (chain A and resi {res_nums} expand 3)")

    # Secondary structure backbone cartoon + Ball and stick sidechains
    cmd.show("cartoon", "bb_cartoon")
    cmd.set("cartoon_color", "gray85")
    cmd.set("cartoon_transparency", 0.25)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_flat_sheets", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_side_chain_helper", 1)

    cmd.show("sticks", "coord_res or lig_near")
    cmd.show("spheres", "zn_site")
    cmd.set("sphere_scale", 0.54, "zn_site")
    cmd.set("stick_ball", 1)
    cmd.set("stick_ball_ratio", 1.85)
    cmd.set("stick_radius", 0.11)

    # Color palette
    cmd.set_color("prot_carbon", [0.82, 0.84, 0.86])
    cmd.set_color("lig_carbon", [0.08, 0.58, 0.53])
    cmd.set_color("zn_purple", [0.48, 0.20, 0.58])
    cmd.set_color("coord_purple", [0.55, 0.25, 0.65])
    cmd.set_color("coord_green", [0.10, 0.60, 0.31])

    cmd.color("prot_carbon", "coord_res and elem C")
    cmd.color("lig_carbon", "lig_near and elem C")
    cmd.color("zn_purple", "zn_site")

    # CPK heteroatoms
    cmd.color("tv_blue", "(coord_res or lig_near) and elem N")
    cmd.color("firebrick", "(coord_res or lig_near) and elem O")
    cmd.color("brightorange", "(coord_res or lig_near) and elem P")

    # Distance measurements (dashed coordination bonds)
    for i, d in enumerate(donors["protein_donors"]):
        _, resn, resi = d["residue"].split("_")
        atom_name = d["atom"]
        dist_name = f"dist_prot_{i}"
        cmd.distance(dist_name, "zn_site", f"chain A and resi {resi} and name {atom_name}")
        cmd.set("dash_color", "coord_purple", dist_name)
        cmd.set("dash_gap", 0.22, dist_name)
        cmd.set("dash_radius", 0.055, dist_name)

    cmd.distance("dist_lig", "zn_site", "lig_near and elem O within 2.3 of zn_site")
    cmd.set("dash_color", "coord_green", "dist_lig")
    cmd.set("dash_gap", 0.22, "dist_lig")
    cmd.set("dash_radius", 0.055, "dist_lig")

    cmd.set("dash_width", 2.2)
    cmd.set("dash_round_ends", 1)
    cmd.hide("labels")
    cmd.set("ray_trace_fog", 0)
    cmd.set("ambient", 0.60)      # High diffuse ambient light
    cmd.set("direct", 0.65)       # Soft directional key light
    cmd.set("specular", 0.04)     # Non-metallic matte finish (minimal specular highlight)
    cmd.set("shininess", 10)
    cmd.set("reflect", 0.45)
    cmd.set("antialias", 2)
    cmd.set("ray_opaque_background", 0)

    # Orientation (optimized non-occluded view of tetrahedral coordination)
    cmd.center("zn_site")
    cmd.orient("zn_site or coord_res or lig_near")
    cmd.zoom("zn_site or coord_res or lig_near", buffer=1.5)
    cmd.set_view((
        0.961649299,   -0.153367192,   -0.227400824,
        -0.088696443,    0.610638738,   -0.786925435,
        0.259548813,    0.776917875,    0.573616385,
        0.000000000,    0.000000000,  -52.616970062,
       -16.350713730,   16.390314102,   -5.329314232,
        43.636070251,   61.597869873,  -20.000000000
    ))

    cmd.ray(2000, 1600)
    cmd.png(str(temp_png), dpi=300)


def make_figure(outdir: Path, targets: dict, pdb_id: str = "9ZSN") -> Path:
    donors = json.loads((REPO / "data/protein_donors.json").read_text())[pdb_id]
    tgt = targets[pdb_id]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_png = Path(tmp.name)

    try:
        render_pymol_scene(pdb_id, donors, temp_png)
        rendered_img = Image.open(temp_png)
    finally:
        if temp_png.exists():
            os.remove(temp_png)

    with plt.rc_context(ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
        ax.imshow(rendered_img)
        ax.axis("off")

        # Visual Callout Badges for Coordination Shell
        # His48 (Bottom Left)
        ax.text(220, 1080, "His48 (Nε2)\n2.10 Å", fontsize=8.8, weight="bold",
                ha="center", va="center", color="#4a1f5c",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="#f5f3ff", edgecolor="#c4b5fd", alpha=0.95, lw=0.8))

        # His50 (Upper Left)
        ax.text(480, 580, "His50 (Nε2)\n2.09 Å", fontsize=8.8, weight="bold",
                ha="center", va="center", color="#4a1f5c",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="#f5f3ff", edgecolor="#c4b5fd", alpha=0.95, lw=0.8))

        # Glu247 (Bottom Right)
        ax.text(1280, 1180, "Glu247 (Oε2)\n2.05 Å", fontsize=8.8, weight="bold",
                ha="center", va="center", color="#4a1f5c",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="#f5f3ff", edgecolor="#c4b5fd", alpha=0.95, lw=0.8))

        # Ligand Donor (Top Right)
        ax.text(1480, 450, "Native ligand (O)\n2.03 Å", fontsize=9.2, weight="bold",
                ha="center", va="center", color="#065f46",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#ecfdf5", edgecolor="#6ee7b7", alpha=0.95, lw=0.9))
        ax.text(995, 770, "Zn²⁺", fontsize=11.5, weight="bold", color="#ffffff",
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#4c1d95")])

        # Header and subtitle
        fig.text(0.5, 0.965, f"Catalytic zinc site: PDB {pdb_id} ({tgt['resolution']:.2f} Å), ligand {tgt['ligand_resname']}",
                 ha="center", va="top", fontsize=11.5, weight="bold", color="#111827")
        fig.text(0.5, 0.915, f"{len(donors['protein_donors'])} protein sidechain donors and the ligand donor share one coordination sphere.\n"
                             "The metal is deleted from the pocket during model preprocessing.",
                 ha="center", va="top", fontsize=9.0, color="#374151")

        # Academic legend
        handles = [
            Line2D([], [], color="#7b3294", ls="--", lw=1.8, label="Protein coordination (His48, His50, Glu247)"),
            Line2D([], [], color="#10b981", ls="--", lw=2.0, label="Native ligand coordination (O)"),
            Line2D([], [], marker="o", color="w", markerfacecolor="#d1d5db", markeredgecolor="#6b7280", markersize=7, label="Protein C"),
            Line2D([], [], marker="o", color="w", markerfacecolor="#0d9488", markeredgecolor="#0f766e", markersize=7, label="Ligand C"),
            Line2D([], [], marker="o", color="w", markerfacecolor=ELEMENT_COLOR["N"], markersize=7, label="N"),
            Line2D([], [], marker="o", color="w", markerfacecolor=ELEMENT_COLOR["O"], markersize=7, label="O"),
            Line2D([], [], marker="o", color="w", markerfacecolor=ELEMENT_COLOR["P"], markersize=7, label="P"),
            Line2D([], [], marker="o", color="w", markerfacecolor=ELEMENT_COLOR["ZN"], markersize=9, label="Zn²⁺"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.0, frameon=False,
                   bbox_to_anchor=(0.5, 0.01))

        plt.subplots_adjust(top=0.88, bottom=0.10, left=0.02, right=0.98)
        out = outdir / "coordination_site.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--pdb-id", default="9ZSN")
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    targets, _, _ = load_cohort()
    make_figure(outdir, targets, pdb_id=args.pdb_id)


if __name__ == "__main__":
    main()
