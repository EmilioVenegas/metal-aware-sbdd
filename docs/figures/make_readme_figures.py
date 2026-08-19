#!/usr/bin/env python
"""Figures for README.md, regenerated from committed analysis artifacts only.

Every number plotted here is recomputed from the checker JSONL outputs, not copied from a
markdown table, so the figures cannot drift from the results. Rates are reported on the
pre-registered primary X-ray cohort (m=21 clusters, n=127 targets, 12,700 molecules), which
is the denominator behind every headline number in results/step1/STEP1_RESULTS.md.

Usage:
    python docs/figures/make_readme_figures.py [--outdir docs/figures]
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
from scipy.stats import gaussian_kde
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]

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

# Shared publication styling. Serif text, hairline spines, no chartjunk.
ACADEMIC_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
}


def cluster_bootstrap_ci(per_target: dict[str, float], pdb_to_cluster: dict[str, str],
                         n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Resample sequence clusters with replacement; percentile CI on the cluster-mean rate.

    Clusters, not targets, are the independent unit — the cohort deliberately contains
    families of near-identical structures (one 50-member cluster, one 32-member).
    """
    by_cluster: dict[str, list[float]] = {}
    for pdb, val in per_target.items():
        by_cluster.setdefault(pdb_to_cluster[pdb], []).append(val)
    means = np.array([np.mean(v) for v in by_cluster.values()])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(n_boot, len(means)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


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

def smarts_matches(sdf_dir: Path, keep) -> dict[tuple[str, int], bool]:
    patterns = [p for p in (Chem.MolFromSmarts(s) for s in ZBG_SMARTS.values())
                if p is not None]
    match: dict[tuple[str, int], bool] = {}
    for pdb_id in sorted(keep):
        f = sdf_dir / f"{pdb_id}.sdf"
        if not f.exists():
            continue
        for k, mol in enumerate(Chem.SDMolSupplier(str(f), sanitize=True)):
            match[(pdb_id, k)] = mol is not None and any(mol.HasSubstructMatch(p)
                                                         for p in patterns)
    return match


def per_target_rate(records, field, keep, subset=None) -> dict[str, float]:
    acc: dict[str, list[bool]] = {}
    for r in records:
        if r.get("unreadable") or r["pdb_id"] not in keep:
            continue
        if subset is not None and not subset.get((r["pdb_id"], r["mol_index"]), False):
            continue
        acc.setdefault(r["pdb_id"], []).append(bool(r.get(field)))
    return {p: 100.0 * float(np.mean(v)) for p, v in acc.items()}


def fig_valid_coordination(outdir: Path, arm_a, arm_b, native, xray, pdb_to_cluster):
    match = smarts_matches(REPO / "results/step1/generation/sdf", xray)
    kept = sum(1 for r in arm_a if not r.get("unreadable") and r["pdb_id"] in xray
               and match.get((r["pdb_id"], r["mol_index"]), False))
    total = sum(1 for r in arm_a if not r.get("unreadable") and r["pdb_id"] in xray)
    retained = 100.0 * kept / total
    smarts_recs = [r for r in arm_a
                   if match.get((r["pdb_id"], r["mol_index"]), False)]
    smarts_yield = 100.0 * sum(1 for r in smarts_recs if r["pdb_id"] in xray
                               and r.get("has_valid_coordination")) / total

    F = "has_valid_coordination"
    bars = [
        ("Native ligands\n(C1 ceiling)", rate(native, F, xray),
         per_target_rate(native, F, xray), "#3d6b52", False),
        (f"Arm A + SMARTS filter\n(post-hoc, keeps {retained:.0f}% of molecules)",
         rate(smarts_recs, F, xray),
         per_target_rate(arm_a, F, xray, subset=match), "#6f6193", True),
        ("Arm A\nbase DiffSBDD", rate(arm_a, F, xray),
         per_target_rate(arm_a, F, xray), "#b5623f", False),
        ("Arm B\nfine-tuned, metal-blind", rate(arm_b, F, xray),
         per_target_rate(arm_b, F, xray), "#8f2d38", False),
    ]

    with plt.rc_context(ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        ypos = np.arange(len(bars))[::-1]
        for y, (_, pooled, pertgt, color, is_subset) in zip(ypos, bars):
            # The bar and its interval must be the SAME estimand. The bootstrap resamples
            # clusters, so the bar is the cluster-mean rate; the pooled molecule-level rate
            # (the headline number, dominated by the two 50- and 32-target families) is
            # overlaid as a separate marker rather than silently given someone else's CI.
            by_cluster: dict[str, list[float]] = {}
            for pdb, v in pertgt.items():
                by_cluster.setdefault(pdb_to_cluster[pdb], []).append(v)
            cmean = float(np.mean([np.mean(v) for v in by_cluster.values()]))
            lo, hi = cluster_bootstrap_ci(pertgt, pdb_to_cluster)
            face = "white" if is_subset else color
            ax.barh(y, cmean, height=0.52, color=face, alpha=1.0 if is_subset else 0.92,
                    edgecolor=color, linewidth=1.4 if is_subset else 0.8, zorder=2)
            ax.errorbar(cmean, y, xerr=[[max(cmean - lo, 0)], [max(hi - cmean, 0)]],
                        fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3,
                        capthick=0.9, zorder=3)
            ax.plot([pooled], [y], marker="D", ms=4.2, mfc="white", mec="#222222",
                    mew=0.9, ls="none", zorder=4)
            ax.text(max(hi, cmean, pooled) + 2.4, y,
                    f"{cmean:.1f}  [{lo:.1f}, {hi:.1f}]\n◇ pooled {pooled:.2f}",
                    va="center", fontsize=8.6, linespacing=1.45, zorder=3)

        ax.set_yticks(ypos)
        ax.set_yticklabels([b[0] for b in bars])
        ax.set_xlim(0, 138)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.xaxis.grid(True, color="#d5d5d5", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.set_xlabel("molecules with ≥1 valid metal coordination (%)")
        fig.text(0.5, 0.965, "Valid coordination against the deleted catalytic Zn²⁺",
                 ha="center", va="top", fontsize=11.5)
        fig.text(0.5, 0.905, "primary X-ray cohort: 21 sequence clusters, 127 targets, "
                             "100 valid molecules per target", ha="center", va="top",
                 fontsize=8.6, color="#444444")
        fig.text(0.012, 0.018,
                 "Bars, cluster-mean rate: each of the 21 sequence clusters weighted equally.\n"
                 "Whiskers, 95% percentile CI from 10,000 bootstrap resamples of clusters.\n"
                 "◇, pooled molecule-level rate. It sits below the cluster mean because the\n"
                 "two largest families (50 and 32 targets) score below average.\n"
                 "Open bar is a post-hoc filtered subset, not an equal-N arm: it discards "
                 f"{100 - retained:.0f}%\nof molecules, so its yield per molecule generated is "
                 f"{smarts_yield:.1f}%.",
                 fontsize=7.8, color="#444444", va="bottom")
        fig.subplots_adjust(left=0.30, right=0.995, top=0.855, bottom=0.345)
        out = outdir / "valid_coordination_by_arm.png"
        fig.savefig(out)
        plt.close(fig)
    print(f"wrote {out}")


# --- Figure 5: the coordination sphere itself, in cross-section -----------------------

ZN_IONIC_RADIUS = 0.74   # Shannon ionic radius, Zn(II), four-coordinate


def fig_coordination_spheres(outdir: Path, arm_a, arm_b, native, xray,
                             n_show: int = 600, rmax: float = 4.6, seed: int = 7):
    """Cross-section through the coordination sphere: one dot per molecule, at its true
    radial distance from the metal and a random bearing. Same data as the histogram —
    read radially instead of along an axis, so the shells are visible as shells."""
    panels = [("Native ligands", native, "#1a9850"),
              ("Arm A — base DiffSBDD", arm_a, "#d6604d"),
              ("Arm B — fine-tuned, metal-blind", arm_b, "#2166ac")]
    rng = np.random.default_rng(seed)

    with plt.rc_context(ACADEMIC_RC):
        fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.9))
        for ax, (label, recs, color) in zip(axes, panels):
            d = np.array([r["min_dist_to_metal"] for r in recs
                          if not r.get("unreadable") and r["pdb_id"] in xray])
            pct_valid = 100.0 * np.mean((d >= VALID_WINDOW[0]) & (d <= VALID_WINDOW[1]))
            pct_clash = 100.0 * np.mean(d < V1_CLASH)
            pct_out = 100.0 * np.mean(d > rmax)

            # shells
            ax.add_patch(plt.Circle((0, 0), SHELL, facecolor="#f2f2f2", edgecolor="none",
                                    zorder=0))
            ax.add_patch(plt.Circle((0, 0), VALID_WINDOW[1], facecolor="#1a9850",
                                    alpha=0.16, edgecolor="none", zorder=0))
            ax.add_patch(plt.Circle((0, 0), VALID_WINDOW[0], facecolor="white",
                                    edgecolor="none", zorder=0))
            ax.add_patch(plt.Circle((0, 0), V1_CLASH, facecolor="#b2182b", alpha=0.14,
                                    edgecolor="none", zorder=0))
            for r_ring, ls, c in [(V1_CLASH, "--", "#b2182b"), (VALID_WINDOW[0], "-", "#1a7d3c"),
                                  (VALID_WINDOW[1], "-", "#1a7d3c"), (SHELL, ":", "#666666")]:
                ax.add_patch(plt.Circle((0, 0), r_ring, fill=False, ls=ls, lw=0.8,
                                        edgecolor=c, zorder=2))

            sub = d if len(d) <= n_show else rng.choice(d, n_show, replace=False)
            theta = rng.uniform(0, 2 * np.pi, len(sub))
            rad = np.clip(sub, 0, rmax + 0.12)
            ax.scatter(rad * np.cos(theta), rad * np.sin(theta), s=7, color=color,
                       alpha=0.45, linewidths=0, zorder=3)

            ax.add_patch(plt.Circle((0, 0), ZN_IONIC_RADIUS, facecolor="#7b3294",
                                    edgecolor="black", lw=0.7, zorder=4))
            ax.text(0, 0, "Zn", color="white", ha="center", va="center", fontsize=7.5,
                    weight="bold", zorder=5)

            ax.set_xlim(-rmax - 0.35, rmax + 0.35)
            ax.set_ylim(-rmax - 0.35, rmax + 0.9)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(f"{label}\n1.90–2.35 Å: {pct_valid:.1f}%   ·   "
                         f"< 1.70 Å: {pct_clash:.1f}%", fontsize=8.8, pad=5)
            ax.text(0, -rmax - 0.30, f"{pct_out:.0f}% beyond {rmax:.1f} Å (plotted at the rim)",
                    ha="center", va="top", fontsize=7.5, color="#666666")

        # one shared ring legend, on the first panel
        ax0 = axes[0]
        # A corner key, not leader lines: arrows to concentric rings cross each other and
        # their own labels for any placement that fits inside an equal-aspect panel.
        key = [Line2D([], [], color="#666666", ls=":", lw=1.0, label="2.70 Å first shell"),
               Line2D([], [], color="#1a7d3c", ls="-", lw=1.0, label="1.90–2.35 Å valid"),
               Line2D([], [], color="#b2182b", ls="--", lw=1.0, label="1.70 Å clash"),
               Line2D([], [], color="#7b3294", marker="o", ls="none", ms=5,
                      label="Zn²⁺, 0.74 Å ionic radius")]
        ax0.legend(handles=key, loc="upper left", frameon=False, fontsize=7.2,
                   handlelength=1.7, handletextpad=0.6, labelspacing=0.42,
                   borderaxespad=0.0, borderpad=0.0)

        fig.text(0.5, 0.975, "Cross-section through the catalytic Zn²⁺ coordination sphere",
                 ha="center", va="top", fontsize=11.5)
        fig.text(0.5, 0.925, "one dot per molecule, at the true distance from the metal to "
                             "its nearest heavy atom and a random bearing; 600 molecules "
                             "sampled per arm",
                 ha="center", va="top", fontsize=8.6, color="#444444")
        fig.text(0.012, 0.015,
                 "Zn(II) ionic radius is the Shannon four-coordinate value. Radii are "
                 "nearest-heavy-atom distances, so the percentages above are geometric only:\n"
                 "the valid-coordination endpoint additionally requires the contacting atom to "
                 "be an N, O or S donor.",
                 fontsize=7.8, color="#444444", va="bottom")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.795, bottom=0.155, wspace=0.05)
        out = outdir / "zn_coordination_spheres.png"
        fig.savefig(out)
        plt.close(fig)
    print(f"wrote {out}")

# --- Figure 4: where the ligand atoms actually land -----------------------------------

def fig_distance_distribution(outdir: Path, arm_a, arm_b, native, xray,
                              kde_sigma: float = 0.10):
    """Binned counts drawn faintly, with a smoothed continuous density on top.

    The KDE bandwidth is fixed at an absolute 0.10 Å rather than a rule-of-thumb multiple
    of each series' standard deviation: Arm B's distribution is ~7x wider than the natives',
    so a relative bandwidth would smooth the three curves by wildly different amounts and
    make the comparison an artefact of the smoother.
    """
    series = [("Native ligands", native, "#1a9850"),
              ("Arm A — base DiffSBDD", arm_a, "#d6604d"),
              ("Arm B — fine-tuned, metal-blind", arm_b, "#2166ac")]
    bins = np.arange(0.8, 8.01, 0.2)
    width = bins[1] - bins[0]
    grid = np.linspace(bins[0], bins[-1], 800)

    with plt.rc_context(ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        peak = 0.0
        for label, recs, color in series:
            d = np.array([r["min_dist_to_metal"] for r in recs
                          if not r.get("unreadable") and r["pdb_id"] in xray])
            beyond = 100.0 * float(np.mean(d > bins[-1]))
            w = np.ones_like(d) * 100.0 / len(d)
            # no clipping: a pile-up bar in the last bin would be an artefact, and the
            # out-of-range fraction is stated in the legend instead
            ax.hist(d, bins=bins, weights=w, histtype="step", lw=0.7, color=color,
                    alpha=0.40, zorder=2)
            # density on the same axis as the histogram: % of molecules per bin width
            kde = gaussian_kde(d, bw_method=kde_sigma / float(np.std(d)))
            dens = kde(grid) * width * 100.0
            ax.fill_between(grid, dens, color=color, alpha=0.10, lw=0, zorder=1)
            ax.plot(grid, dens, color=color, lw=1.7, zorder=3,
                    label=f"{label}  (n={len(d):,}; {beyond:.0f}% beyond {bins[-1]:.0f} Å)")
            peak = max(peak, float(dens.max()))

        # Headroom above the native peak for the band annotation; the two threshold
        # lines are labelled vertically alongside themselves so nothing collides.
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

        ax.set_xlabel("distance from Zn²⁺ to the nearest ligand heavy atom (Å)")
        ax.set_ylabel("molecules (%)")
        fig.text(0.5, 0.965, "Where generated ligand atoms land relative to the deleted metal",
                 ha="center", va="top", fontsize=11.5)
        fig.text(0.5, 0.905, "primary X-ray cohort: 12,700 molecules per generative arm",
                 ha="center", va="top", fontsize=8.6, color="#444444")
        ax.legend(frameon=False, loc="upper right", borderaxespad=0.6)
        fig.text(0.012, 0.018,
                 "Heavy lines, Gaussian kernel density (σ = 0.10 Å, identical for all three "
                 "series) scaled to % of molecules per 0.2 Å bin;\nfaint steps, the raw binned "
                 "counts. Smoothing leaks a little density across the 1.70 Å boundary that the "
                 "raw bins do not have.",
                 fontsize=7.8, color="#444444", va="bottom")
        fig.subplots_adjust(left=0.085, right=0.975, top=0.855, bottom=0.275)
        out = outdir / "zn_distance_distribution.png"
        fig.savefig(out)
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
    pdb_to_cluster = {p: f"C{i+1:02d}" for i, m in enumerate(clusters) for p in m}

    fig_coordination_site(outdir, targets)
    fig_native_ligands(outdir, targets, xray, clusters)
    fig_valid_coordination(outdir, arm_a, arm_b, native, xray, pdb_to_cluster)
    fig_distance_distribution(outdir, arm_a, arm_b, native, xray)
    fig_coordination_spheres(outdir, arm_a, arm_b, native, xray)


if __name__ == "__main__":
    main()
