#!/usr/bin/env python
"""Figure 2: Native zinc-binding ligands spanning multiple ZBG classes.

Usage:
    python docs/figures/fig2_native_ligands.py [--outdir docs/figures] [--n 8]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from common import REPO, ZBG_SMARTS, load_cohort


def _parse_dist(label: str) -> float:
    """'O4 (1.92A)' -> 1.92; unparseable labels sort last."""
    m = re.search(r"\(([\d.]+)\s*A\)", label)
    return float(m.group(1)) if m else float("inf")


def _fmt_coord(label: str) -> str:
    """RDKit's grid legend renderer drops non-ASCII glyphs, so 'A' stays ASCII here."""
    return re.sub(r"\(([\d.]+)\s*A\)", r"\1 A", label)


def make_figure(outdir: Path, targets: dict, xray: set[str], clusters: list[list[str]],
                n: int = 8) -> Path:
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
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    targets, xray, clusters = load_cohort()
    make_figure(outdir, targets, xray, clusters, n=args.n)


if __name__ == "__main__":
    main()
