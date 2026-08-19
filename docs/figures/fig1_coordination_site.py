#!/usr/bin/env python
"""Figure 1: Catalytic zinc site 3D coordination sphere (PDB 9ZSN).

Usage:
    python docs/figures/fig1_coordination_site.py [--outdir docs/figures] [--pdb-id 9ZSN]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from rdkit import Chem

from common import REPO, ELEMENT_COLOR, load_cohort


def make_figure(outdir: Path, targets: dict, pdb_id: str = "9ZSN") -> Path:
    donors = json.loads((REPO / "data/protein_donors.json").read_text())[pdb_id]
    zn = np.array(donors["zn"], dtype=float)
    mol = Chem.SDMolSupplier(str(REPO / f"data/native_ligands/{pdb_id}.sdf"),
                             sanitize=False, removeHs=True)[0]
    conf = mol.GetConformer()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    allpos = {i: np.array(list(conf.GetAtomPosition(i))) for i in heavy}
    # keep the coordination-sphere neighbourhood, not the whole ligand
    near = {i for i in heavy if np.linalg.norm(allpos[i] - zn) < 6.5}
    idx = sorted(near)
    remap = {old: new for new, old in enumerate(idx)}
    pos = np.array([allpos[i] for i in idx])
    els = [mol.GetAtomWithIdx(i).GetSymbol() for i in idx]
    bonds = [(remap[b.GetBeginAtomIdx()], remap[b.GetEndAtomIdx()]) for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in near and b.GetEndAtomIdx() in near]

    fig = plt.figure(figsize=(6.8, 5.3))
    ax = fig.add_subplot(111, projection="3d")

    for i, j in bonds:
        ax.plot(*zip(pos[i], pos[j]), color="#777777", lw=2.4, zorder=1)
    for xyz, el in zip(pos, els):
        ax.scatter(*xyz, s=90, color=ELEMENT_COLOR.get(el, "#4d4d4d"),
                   edgecolors="white", linewidths=0.7, depthshade=False, zorder=2)

    ax.scatter(*zn, s=900, color=ELEMENT_COLOR["ZN"], edgecolors="black", linewidths=1.2,
               depthshade=False, zorder=3)

    for d in donors["protein_donors"]:
        xyz = np.array(d["xyz"], dtype=float)
        ax.plot(*zip(zn, xyz), color="#7b3294", ls=":", lw=2.2, zorder=2)
        ax.scatter(*xyz, s=170, color=ELEMENT_COLOR.get(d["element"], "#4d4d4d"),
                   edgecolors="black", linewidths=0.9, depthshade=False, zorder=3)
        _, resn, resi = d["residue"].split("_")
        out_dir_v = (xyz - zn) / np.linalg.norm(xyz - zn)
        lab = xyz + out_dir_v * 2.0
        ax.text(*lab, f"{resn}{resi}\n{d['distance']:.2f} Å", fontsize=9.5, ha="center",
                va="center", color="#4a1f5c")

    # closest ligand heavy atom: the coordination the generative models must reproduce
    dist = np.linalg.norm(pos - zn, axis=1)
    k = int(np.argmin(dist))
    ax.plot(*zip(zn, pos[k]), color="#1a9850", ls="--", lw=3.0, zorder=2)
    lig_lab = zn + (pos[k] - zn) * 2.05
    ax.text(*lig_lab, f"ligand {els[k]}\n{dist[k]:.2f} Å", fontsize=10.5, color="#12703a",
            weight="bold", ha="center", va="center")

    # View down the normal of the protein-donor plane, so the donors never project on
    # top of the metal. Deterministic — no hand-tuned camera angle.
    dvec = np.array([d["xyz"] for d in donors["protein_donors"]], dtype=float) - zn
    _, _, vt = np.linalg.svd(dvec - dvec.mean(axis=0), full_matrices=True)
    nrm = vt[2] / np.linalg.norm(vt[2])
    elev = float(np.degrees(np.arcsin(np.clip(nrm[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(nrm[1], nrm[0])))

    r = 4.2
    centre = (zn + pos.mean(axis=0)) / 2.0
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_zlim(centre[2] - r, centre[2] + r)
    ax.set_box_aspect((1, 1, 1))
    # tilt off the donor-plane normal so the ligand is not foreshortened onto the metal
    ax.view_init(elev=elev + 28.0, azim=azim + 12.0)
    ax.set_axis_off()
    ax.set_position([-0.10, 0.02, 1.20, 0.88])

    tgt = targets[pdb_id]
    fig.text(0.5, 0.975, f"Catalytic zinc site — PDB {pdb_id} ({tgt['resolution']:.2f} Å), "
                         f"ligand {tgt['ligand_resname']}", ha="center", va="top",
             fontsize=12.5, weight="bold")
    fig.text(0.5, 0.925, f"{len(donors['protein_donors'])} protein sidechain donors and the "
                         f"ligand donor share one coordination sphere —\nthe metal is deleted "
                         f"from the pocket every one of these models is conditioned on",
             ha="center", va="top", fontsize=9.8, color="#333333")
    handles = [Line2D([], [], ls=":", color="#7b3294", lw=2, label="protein donor"),
               Line2D([], [], ls="--", color="#1a9850", lw=2, label="native ligand donor"),
               Line2D([], [], marker="o", ls="", color=ELEMENT_COLOR["N"], label="N"),
               Line2D([], [], marker="o", ls="", color=ELEMENT_COLOR["O"], label="O"),
               Line2D([], [], marker="o", ls="", color=ELEMENT_COLOR["C"], label="C")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.canvas.draw()
    zx, zy, _ = proj3d.proj_transform(*zn, ax.get_proj())
    ax.annotate("Zn²⁺", xy=(zx, zy), xycoords="data", xytext=(-46, 10),
                textcoords="offset points", fontsize=13, weight="bold",
                color=ELEMENT_COLOR["ZN"])

    out = outdir / "coordination_site.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.12)
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
