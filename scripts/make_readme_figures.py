#!/usr/bin/env python
"""Figures for README.md, regenerated from committed analysis artifacts only.

Every number plotted here is recomputed from the checker JSONL outputs, not copied from a
markdown table, so the figures cannot drift from the results. Rates are reported on the
pre-registered primary X-ray cohort (m=21 clusters, n=127 targets, 12,700 molecules), which
is the denominator behind every headline number in results/step1/STEP1_RESULTS.md.

Usage:
    python scripts/make_readme_figures.py [--outdir docs/figures]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent

VALID_WINDOW = (1.90, 2.35)   # Zn-N / Zn-O accepted coordination distance (checker constant)
V1_CLASH = 1.70
SHELL = 2.70

ELEMENT_COLOR = {"C": "#4d4d4d", "N": "#2166ac", "O": "#b2182b", "S": "#d6a419",
                 "P": "#e08214", "F": "#5aae61", "Cl": "#1b7837", "Br": "#8c510a",
                 "I": "#762a83", "ZN": "#7b3294"}

ZBG_SMARTS = {
    "carboxylate": "[CX3](=O)[OX1H0-,OX2H1]",
    "hydroxamate": "[NX3H1,NX3H0]([OX2H1,OX2H0])C(=O)",
    "thiol": "[SX2H1,SX1H0-]",
    "imidazole": "c1ncnc1",
    "sulfonamide": "[NX3H2,NX3H1][SX4](=O)(=O)",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def load_cohort():
    blob = torch.load(REPO / "data/external_zn_test_clean.pt", map_location="cpu",
                      weights_only=False)
    targets = {t["pdb_id"]: t for t in blob["targets"]}
    xray = {p for p, t in targets.items() if t.get("method", "X-ray") == "X-ray"}
    return targets, xray, blob["clusters"]


def rate(records, field, keep) -> float:
    vals = [bool(r.get(field)) for r in records
            if not r.get("unreadable") and r["pdb_id"] in keep]
    return 100.0 * float(np.mean(vals))


# --- Figure 1: one real catalytic zinc site -------------------------------------------

def fig_coordination_site(outdir: Path, targets, pdb_id: str = "9ZSN"):
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
    # 3-D text z-ordering against scatter markers is unreliable, so the metal label is
    # placed in display space from the projected position of the ion.
    fig.canvas.draw()
    zx, zy, _ = proj3d.proj_transform(*zn, ax.get_proj())
    ax.annotate("Zn²⁺", xy=(zx, zy), xycoords="data", xytext=(-46, 10),
                textcoords="offset points", fontsize=13, weight="bold",
                color=ELEMENT_COLOR["ZN"])

    out = outdir / "coordination_site.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out}")


# --- Figure 2: native zinc-binding ligands from the benchmark -------------------------

def _parse_dist(label: str) -> float:
    """'O4 (1.92A)' -> 1.92; unparseable labels sort last."""
    m = re.search(r"\(([\d.]+)\s*A\)", label)
    return float(m.group(1)) if m else float("inf")


def _fmt_coord(label: str) -> str:
    """RDKit's grid legend renderer drops non-ASCII glyphs, so 'A' stays ASCII here."""
    return re.sub(r"\(([\d.]+)\s*A\)", r"\1 A", label)


def fig_native_ligands(outdir: Path, targets, xray, clusters, n: int = 8):
    """One ligand per sequence cluster, greedily spread across zinc-binding-group classes."""
    patterns = {k: Chem.MolFromSmarts(v) for k, v in ZBG_SMARTS.items()}
    candidates = []
    for members in clusters:
        for pdb_id in members:
            if pdb_id not in xray:
                continue
            t = targets[pdb_id]
            sdf = REPO / f"data/native_ligands/{pdb_id}.sdf"
            if not sdf.exists():
                continue
            mol = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True)[0]
            if mol is None:
                continue
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            if not (10 <= mol.GetNumHeavyAtoms() <= 45):
                continue
            zbg = [name for name, p in patterns.items()
                   if p is not None and mol.HasSubstructMatch(p)]
            if not zbg:
                continue
            candidates.append((pdb_id, t, mol, zbg[0]))
            break   # one representative per cluster

    # greedy: cover as many distinct ZBG classes as possible before repeating one
    picked, used_zbg = [], set()
    for want_new in (True, False):
        for pdb_id, t, mol, zbg in candidates:
            if len(picked) == n:
                break
            if any(p[0] == pdb_id for p in picked):
                continue
            if want_new and zbg in used_zbg:
                continue
            flat = Chem.Mol(mol)
            flat.RemoveAllConformers()
            AllChem.Compute2DCoords(flat)
            coords = t.get("coordinating_ligand_atoms") or []
            best = min(coords, key=_parse_dist, default=None)
            coord = best if best else f"{t['min_zn_ligand_dist']:.2f}A"
            name = t.get("protein_name", "")
            if len(name) > 34:
                name = name[:32].rsplit(" ", 1)[0] + "…"
            picked.append((pdb_id, flat,
                           f"{pdb_id} · {t['ligand_resname']} · {zbg}\n"
                           f"Zn-{_fmt_coord(coord)}\n{name}"))
            used_zbg.add(zbg)

    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.legendFontSize = 20
    opts.legendFraction = 0.28
    img = Draw.MolsToGridImage([m for _, m, _ in picked],
                               legends=[l for _, _, l in picked],
                               molsPerRow=4, subImgSize=(360, 340), useSVG=False,
                               drawOptions=opts)
    out = outdir / "native_zbg_ligands.png"
    data = img.data if hasattr(img, "data") else img
    if isinstance(data, bytes):
        out.write_bytes(data)
    else:
        img.save(out)
    print(f"wrote {out} ({len(picked)} ligands, "
          f"{len({z for z in used_zbg})} ZBG classes)")


# --- Figure 3: valid-coordination rate by arm -----------------------------------------

def smarts_filtered(records, sdf_dir: Path, keep):
    patterns = [p for p in (Chem.MolFromSmarts(s) for s in ZBG_SMARTS.values())
                if p is not None]
    match = {}
    for pdb_id in sorted(keep):
        f = sdf_dir / f"{pdb_id}.sdf"
        if not f.exists():
            continue
        for k, mol in enumerate(Chem.SDMolSupplier(str(f), sanitize=True)):
            match[(pdb_id, k)] = mol is not None and any(mol.HasSubstructMatch(p)
                                                         for p in patterns)
    sel = [r for r in records if not r.get("unreadable") and r["pdb_id"] in keep
           and match.get((r["pdb_id"], r["mol_index"]), False)]
    total = len([r for r in records if not r.get("unreadable") and r["pdb_id"] in keep])
    return (100.0 * float(np.mean([bool(r["has_valid_coordination"]) for r in sel])),
            100.0 * len(sel) / total)


def fig_valid_coordination(outdir: Path, arm_a, arm_b, native, xray):
    smarts_rate, retained = smarts_filtered(arm_a, REPO / "results/step1/generation/sdf", xray)
    bars = [
        ("Native ligands\n(ceiling, C1)", rate(native, "has_valid_coordination", xray), "#1a9850"),
        (f"Arm A + SMARTS filter\n({retained:.0f}% of molecules kept)", smarts_rate, "#8073ac"),
        ("Arm A\nbase DiffSBDD", rate(arm_a, "has_valid_coordination", xray), "#d6604d"),
        ("Arm B\nfine-tuned, metal-blind", rate(arm_b, "has_valid_coordination", xray), "#b2182b"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ypos = np.arange(len(bars))[::-1]
    ax.barh(ypos, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
    for y, (_, v, _) in zip(ypos, bars):
        ax.text(v + 1.2, y, f"{v:.2f}%", va="center", fontsize=11, weight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bars], fontsize=10)
    ax.set_xlim(0, 92)
    ax.set_xlabel("molecules with ≥1 valid metal coordination (%)")
    ax.set_title("Valid coordination rate — primary X-ray cohort\n"
                 "m=21 clusters, n=127 targets, 100 valid molecules per target", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = outdir / "valid_coordination_by_arm.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --- Figure 4: where the ligand atoms actually land -----------------------------------

def fig_distance_distribution(outdir: Path, arm_a, arm_b, native, xray):
    series = [("Native ligands", native, "#1a9850"),
              ("Arm A — base DiffSBDD", arm_a, "#d6604d"),
              ("Arm B — fine-tuned, metal-blind", arm_b, "#2166ac")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bins = np.arange(0.8, 8.01, 0.2)
    for label, recs, color in series:
        d = np.array([r["min_dist_to_metal"] for r in recs
                      if not r.get("unreadable") and r["pdb_id"] in xray])
        w = np.ones_like(d) * 100.0 / len(d)
        ax.hist(np.clip(d, bins[0], bins[-1]), bins=bins, weights=w, histtype="step",
                lw=2.0, color=color, label=f"{label}  (n={len(d):,})")
    ax.axvspan(*VALID_WINDOW, color="#1a9850", alpha=0.12, zorder=0)
    ax.axvline(V1_CLASH, color="#b2182b", ls="--", lw=1.4)
    ax.axvline(SHELL, color="#666666", ls=":", lw=1.4)
    ax.text(np.mean(VALID_WINDOW), ax.get_ylim()[1] * 0.94, "valid\nZn–N/O", ha="center",
            fontsize=9, color="#1a7d3c")
    ax.text(V1_CLASH - 0.06, ax.get_ylim()[1] * 0.94, "clash\n<1.70 Å", ha="right",
            fontsize=9, color="#b2182b")
    ax.text(SHELL + 0.08, ax.get_ylim()[1] * 0.94, "shell 2.70 Å", ha="left", fontsize=9,
            color="#666666")
    ax.text(bins[-1], -ax.get_ylim()[1] * 0.135, "≥8\n(clipped)", ha="center", fontsize=8,
            color="#666666")
    ax.set_xlabel("distance from Zn²⁺ to the nearest ligand heavy atom (Å)")
    ax.set_ylabel("molecules (%)")
    ax.set_title("Where generated ligand atoms land relative to the deleted metal\n"
                 "primary X-ray cohort, 12,700 molecules per arm", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = outdir / "zn_distance_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    targets, xray, clusters = load_cohort()
    arm_a = load_jsonl(REPO / "results/step1/checker/generated.jsonl")
    arm_b = load_jsonl(REPO / "results/step2/arm_b_generation/checker_results.jsonl")
    native = load_jsonl(REPO / "results/step1/checker/native_c1.jsonl")

    fig_coordination_site(outdir, targets)
    fig_native_ligands(outdir, targets, xray, clusters)
    fig_valid_coordination(outdir, arm_a, arm_b, native, xray)
    fig_distance_distribution(outdir, arm_a, arm_b, native, xray)


if __name__ == "__main__":
    main()
