#!/usr/bin/env python
"""Extract the protein-contributed donor atoms of each catalytic metal site.

The coordination sphere of a catalytic metal is shared: 2-3 donors come from protein
sidechains, the rest from the ligand. Any geometry statement about the ligand alone is
meaningless without them - a ligand donating one oxygen into a 3-His site reads as CN=1
when the true sphere is tetrahedral.

Emits coordinates, not just distances, so the angular geometry can be computed over the
combined sphere.
"""
import glob, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DONOR_ELEMENTS = {"N", "O", "S"}
SHELL = 2.80  # matches the catalytic-site definition used to build the cohort

# Sidechain donor atoms only; backbone N/O rarely coordinate catalytic metals and
# including them would inflate the coordination number.
BACKBONE = {"N", "CA", "C", "O", "OXT"}

STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}


def donors_for(struct_path: Path, zn: np.ndarray):
    import gemmi
    st = gemmi.read_structure(str(struct_path))
    st.setup_entities()
    out = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name not in STANDARD_AA:
                    continue
                for atom in res:
                    if atom.element.name.upper() not in DONOR_ELEMENTS:
                        continue
                    if atom.name in BACKBONE:
                        continue
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    d = float(np.linalg.norm(p - zn))
                    if d <= SHELL:
                        out.append({
                            "residue": f"{chain.name}_{res.name}_{res.seqid.num}",
                            "atom": atom.name, "element": atom.element.name.upper(),
                            "xyz": p.tolist(), "distance": round(d, 3),
                        })
        break  # first model only
    return out


def main():
    blob = torch.load(REPO / "data/external_zn_test_clean.pt",
                      map_location="cpu", weights_only=False)
    records = blob["targets"] if isinstance(blob, dict) else blob

    out, missing, counts = {}, [], []
    for r in records:
        pid = r["pdb_id"]
        hits = glob.glob(str(REPO / f"data/external_pdbs/{pid}.*"))
        if not hits:
            missing.append(pid)
            continue
        zn = np.array(r["zn_coord"], dtype=float)
        d = donors_for(Path(hits[0]), zn)
        out[pid] = {"zn": zn.tolist(), "protein_donors": d}
        counts.append(len(d))

    path = REPO / "data/protein_donors.json"
    path.write_text(json.dumps(out, indent=1))
    c = np.array(counts)
    print(f"{len(out)} targets -> {path}")
    print(f"protein donors per site: median {np.median(c):.0f}  "
          f"mean {c.mean():.2f}  range {c.min()}-{c.max()}")
    print(f"sites with <2 donors (cohort required >=2): "
          f"{int((c < 2).sum())}")
    if missing:
        print("missing structures:", missing)


if __name__ == "__main__":
    main()
