#!/usr/bin/env python
"""Metal coordination validity checker.

Implements results/step1/ANALYSIS_PLAN.md section 3 exactly. Every threshold below is
pre-registered; none may be changed after seeing results.

No existing tool measures this. PoseBusters checks ligand conformation, UFF strain and
clashes; GenBench3D checks conformer quality. Neither checks coordination number,
metal-donor distance by element pair, or angular geometry.

Emits one record per molecule AND one per metal-contacting atom, so every secondary
endpoint in the plan is computable without re-running the geometry.
"""
from __future__ import annotations

import argparse, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# --- Pre-registered constants (ANALYSIS_PLAN.md section 3) -------------------

DONORS = {"N", "O", "S"}

# (metal, donor) -> (min, max) accepted coordination distance, Angstrom
COORD_RANGES = {
    ("ZN", "N"): (1.90, 2.35), ("ZN", "O"): (1.85, 2.30), ("ZN", "S"): (2.15, 2.50),
    ("MG", "O"): (1.95, 2.25), ("MG", "N"): (1.95, 2.25),
    ("MN", "N"): (2.00, 2.35), ("MN", "O"): (2.00, 2.35),
    ("FE", "N"): (1.95, 2.30), ("FE", "O"): (1.95, 2.30),
    ("CA", "O"): (2.25, 2.65), ("CA", "N"): (2.25, 2.65),
    ("CU", "N"): (1.90, 2.30), ("CU", "O"): (1.90, 2.30),
}

V1_CLASH = 1.70   # any heavy atom closer than this to the metal centre
SHELL = 2.70      # first-coordination-shell radius

# Ideal angles by coordination number, for RMS angular deviation
IDEAL_ANGLES = {4: [109.47], 5: [90.0, 120.0], 6: [90.0]}


def _ranges_for(metal: str, elem: str):
    return COORD_RANGES.get((metal.upper(), elem.upper()))


def check_molecule(coords: np.ndarray, elements: list[str], metal_xyz: np.ndarray,
                   metal: str = "ZN", protein_donors: np.ndarray | None = None) -> dict:
    """Geometry of one molecule against one metal centre.

    coords/elements must be HEAVY ATOMS ONLY, in the same frame as metal_xyz.

    protein_donors: (M,3) coordinates of the protein sidechain donors already
    coordinating this metal. The coordination sphere of a catalytic metal is SHARED -
    typically 2-3 protein donors plus whatever the ligand contributes. Coordination
    number and angular geometry are only meaningful over the combined sphere: a ligand
    donating one oxygen into a 3-His site is not "CN=1", it completes a tetrahedron.
    Omitting them reports ligand-only geometry, which is not a chemical quantity.
    """
    d = np.linalg.norm(coords - metal_xyz, axis=1)

    contacts = []
    for i, (dist, el) in enumerate(zip(d, elements)):
        if dist >= SHELL:
            continue
        el = el.upper()
        is_donor = el in DONORS
        rng = _ranges_for(metal, el) if is_donor else None
        in_range = bool(rng and rng[0] <= dist <= rng[1])
        contacts.append({
            "atom_index": i, "element": el, "distance": round(float(dist), 3),
            "is_donor": is_donor, "in_range": in_range,
            "v1_clash": bool(dist < V1_CLASH),
            # V2: a NON-donor occupying the shell
            "v2_shell": bool(not is_donor),
            # V3: a donor in the shell but at a malformed distance
            "v3_malformed": bool(is_donor and not in_range and dist >= V1_CLASH),
        })

    v1 = any(c["v1_clash"] for c in contacts)
    v2 = any(c["v2_shell"] for c in contacts)
    v3 = any(c["v3_malformed"] for c in contacts)
    valid_coord = [c for c in contacts if c["in_range"] and not c["v1_clash"]]

    # Coordination geometry over the COMBINED sphere: protein donors + ligand donors.
    cn_ligand = len(valid_coord)
    vecs = [coords[c["atom_index"]] - metal_xyz for c in valid_coord]
    n_protein = 0
    if protein_donors is not None and len(protein_donors):
        n_protein = len(protein_donors)
        vecs = [np.asarray(p, float) - metal_xyz for p in protein_donors] + vecs
    cn_total = n_protein + cn_ligand

    rms_dev = None
    if cn_total >= 2 and cn_total in IDEAL_ANGLES:
        angles = []
        for a, b in combinations(vecs, 2):
            cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        ideals = IDEAL_ANGLES[cn_total]
        devs = [min(abs(ang - ideal) for ideal in ideals) for ang in angles]
        rms_dev = round(float(np.sqrt(np.mean(np.square(devs)))), 2)
    cn = cn_ligand  # backwards-compatible field name

    return {
        # PRIMARY ENDPOINT: >=1 violation of V1 or V2
        "primary_violation": bool(v1 or v2),
        "v1_clash": v1, "v2_shell_occupancy": v2, "v3_malformed": v3,
        "has_valid_coordination": cn_ligand > 0,
        "n_valid_coordination": cn_ligand,
        "n_protein_donors": n_protein,
        "coordination_number_total": cn_total,
        "min_dist_to_metal": round(float(d.min()), 3),
        "nearest_element": elements[int(d.argmin())].upper(),
        "n_shell_contacts": len(contacts),
        "coordination_rms_angle_dev": rms_dev,
        "contacts": contacts,
    }


def heavy_atoms(mol: Chem.Mol):
    conf = mol.GetConformer()
    pos, els = [], []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() <= 1:
            continue
        pos.append(list(conf.GetAtomPosition(a.GetIdx())))
        els.append(a.GetSymbol())
    return np.array(pos, dtype=float), els


def run_sdf(sdf: Path, metal_xyz: np.ndarray, metal: str, source: str, pdb_id: str,
            protein_donors=None):
    out = []
    for k, mol in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)):
        if mol is None or mol.GetNumConformers() == 0:
            out.append({"pdb_id": pdb_id, "source": source, "mol_index": k,
                        "unreadable": True})
            continue
        pos, els = heavy_atoms(mol)
        if len(pos) == 0:
            out.append({"pdb_id": pdb_id, "source": source, "mol_index": k,
                        "unreadable": True})
            continue
        rec = check_molecule(pos, els, metal_xyz, metal, protein_donors)
        rec.update({"pdb_id": pdb_id, "source": source, "mol_index": k,
                    "n_heavy_atoms": len(pos), "unreadable": False})
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--sdf-dir", required=True, help="one <PDB>.sdf per target")
    ap.add_argument("--source", default="generated",
                    help="label: generated | native | decoy")
    ap.add_argument("--out", required=True, help="JSONL, one record per molecule")
    ap.add_argument("--metal", default="ZN")
    ap.add_argument("--protein-donors", default=str(Path(__file__).resolve().parent.parent
                                                    / "data/protein_donors.json"),
                    help="JSON from extract_protein_donors.py; required for meaningful "
                         "coordination geometry")
    args = ap.parse_args()

    import torch
    blob = torch.load(args.targets, map_location="cpu", weights_only=False)
    records = blob["targets"] if isinstance(blob, dict) and "targets" in blob else blob
    coords = {r["pdb_id"]: np.array(r["zn_coord"], dtype=float) for r in records}

    pdon = {}
    pd_path = Path(args.protein_donors)
    if pd_path.exists():
        raw = json.loads(pd_path.read_text())
        pdon = {k: np.array([d["xyz"] for d in v["protein_donors"]], dtype=float)
                for k, v in raw.items()}
    else:
        print(f"WARNING: {pd_path} missing - coordination geometry will be ligand-only "
              f"and is NOT a chemical quantity", file=sys.stderr)

    sdf_dir, n_mol, n_tgt = Path(args.sdf_dir), 0, 0
    with open(args.out, "w") as fh:
        for pdb_id, xyz in sorted(coords.items()):
            sdf = sdf_dir / f"{pdb_id}.sdf"
            if not sdf.exists():
                continue
            for rec in run_sdf(sdf, xyz, args.metal, args.source, pdb_id,
                               pdon.get(pdb_id)):
                fh.write(json.dumps(rec) + "\n")
                n_mol += 1
            n_tgt += 1
    print(f"{n_tgt} targets, {n_mol} molecules -> {args.out}")


if __name__ == "__main__":
    main()
