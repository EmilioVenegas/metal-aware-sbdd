#!/usr/bin/env python
"""Arm D scorer: seed-excluded endpoints, seed incorporation, connectivity.

Arm D pins a donor seed at 2.05 A from the catalytic metal. Because
COORD_RANGES[("ZN","O")] = (1.85, 2.30), that seed satisfies the valid-coordination
criterion BY CONSTRUCTION -- a one-atom "molecule" scores 100.00% valid coordination and
13.22 deg angular RMSD across the 133-target cohort, beating the 18.04 deg native ceiling
with no model involved. As-scored coordination and angular endpoints are therefore
uninformative for this arm and are retired.

This script computes the pre-registered replacements
(results/step2/ANALYSIS_PLAN_ARMD.md section 4b):

  PRIMARY    seed-excluded valid coordination -- the seed atom is removed and every
             endpoint recomputed over model-generated atoms only, so the seed acts as
             conditioning rather than as measurement. Directly comparable to Arm C.
  CO-PRIMARY as-scored V1 hard clash (<1.70 A) -- the seed cannot clash at 2.05 A, so
             every clash is a secondary atom the model placed. Not saturated.
  SECONDARY  seed incorporation into a recognised zinc-binding group (SMARTS reused
             verbatim from run_smarts_baseline.py), and fragment connectivity.

Emits one JSONL record per molecule, carrying both as-scored and seed-excluded fields so
the saturation can be shown side by side rather than asserted.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from coordination_checker import check_molecule, heavy_atoms
from run_smarts_baseline import ZBG_SMARTS

RDLogger.DisableLog("rdApp.*")

# Seed is written at an exact coordinate and inpainting holds it fixed; anything beyond
# this is a different atom, not the seed drifting.
SEED_TOL = 0.15


def _drop_atom(pos, els, bonds, idx):
    """Re-index a molecule's heavy-atom arrays with atom `idx` removed."""
    keep = [i for i in range(len(pos)) if i != idx]
    remap = {old: new for new, old in enumerate(keep)}
    new_bonds = [(remap[a], remap[b]) for a, b in bonds
                 if a in remap and b in remap]
    return pos[keep], [els[i] for i in keep], new_bonds


def _seed_zbg(mol, seed_heavy_idx, zbg_patterns):
    """Name of the zinc-binding group the seed atom belongs to, or None.

    Works on a sanitized copy: the SDF is read with sanitize=False to preserve the
    pre-registered geometry path, which leaves implicit valences uncomputed and makes
    H-bearing SMARTS ([OX2H1], [NX3H1], [SX2H1]) raise a pre-condition violation.
    Returns None when sanitization fails -- an unsanitizable molecule has no
    well-defined functional group.
    """
    probe = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(probe)
    except Exception:
        return None
    # heavy-atom index space (heavy_atoms) -> full-molecule index space (SMARTS matches)
    heavy_to_full = [a.GetIdx() for a in probe.GetAtoms() if a.GetAtomicNum() > 1]
    if seed_heavy_idx >= len(heavy_to_full):
        return None
    seed_full = heavy_to_full[seed_heavy_idx]
    for name, patt in zbg_patterns.items():
        if patt is None:
            continue
        if any(seed_full in m for m in probe.GetSubstructMatches(patt)):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="generation_manifest_shard*.jsonl from generate_arm_d.py "
                         "(supplies seed_xyz per target)")
    ap.add_argument("--sdf-dir", required=True)
    ap.add_argument("--protein-donors", default=str(REPO / "data/protein_donors.json"))
    ap.add_argument("--source", default="generated_arm_d")
    ap.add_argument("--metal", default="ZN")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    donors_data = json.load(open(args.protein_donors))
    zbg = {n: Chem.MolFromSmarts(s) for n, s in ZBG_SMARTS.items()}

    seeds = {}
    for line in Path(args.manifest).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("seed_xyz"):
            seeds[rec["pdb_id"]] = rec

    sdf_dir = Path(args.sdf_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_mol = n_tgt = 0
    agg = {"as_valid": 0, "ex_valid": 0, "as_clash": 0, "ex_scored": 0,
           "seed_found": 0, "zbg": 0, "multifrag": 0}

    with open(out_path, "w") as fh:
        for pdb_id, mrec in sorted(seeds.items()):
            sdf = sdf_dir / f"{pdb_id}.sdf"
            if not sdf.exists():
                continue
            info = donors_data.get(pdb_id, {})
            metal_xyz = np.array(info.get("zn", mrec.get("zn_coord")), float)
            pdon = info.get("protein_donors", [])
            pxyz = np.array([d["xyz"] for d in pdon], float) if pdon else None
            seed_xyz = np.array(mrec["seed_xyz"], float)
            n_tgt += 1

            for k, mol in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=False,
                                                       removeHs=False)):
                if mol is None or mol.GetNumConformers() == 0:
                    fh.write(json.dumps({"pdb_id": pdb_id, "source": args.source,
                                         "mol_index": k, "unreadable": True}) + "\n")
                    continue
                pos, els, bonds = heavy_atoms(mol)
                if len(pos) == 0:
                    fh.write(json.dumps({"pdb_id": pdb_id, "source": args.source,
                                         "mol_index": k, "unreadable": True}) + "\n")
                    continue
                n_mol += 1

                as_scored = check_molecule(pos, els, metal_xyz, args.metal, pxyz,
                                           bonds=bonds)

                # Locate the fixed seed atom.
                d_seed = np.linalg.norm(pos - seed_xyz, axis=1)
                si = int(d_seed.argmin())
                seed_found = bool(d_seed[si] < SEED_TOL)

                rec = {
                    "pdb_id": pdb_id, "source": args.source, "mol_index": k,
                    "unreadable": False, "n_heavy_atoms": len(pos),
                    "seed_mode": mrec.get("seed_mode"),
                    "seed_found": seed_found,
                    "seed_offset": round(float(d_seed[si]), 4),
                    # as-scored: RETIRED as an Arm D endpoint, retained for transparency
                    "as_scored_has_valid_coordination": as_scored["has_valid_coordination"],
                    "as_scored_angular_rms": as_scored["coordination_rms_angle_dev"],
                    # CO-PRIMARY: seed sits at 2.05 A, so any clash is a secondary atom
                    "v1_clash": as_scored["v1_clash"],
                    "min_dist_to_metal": as_scored["min_dist_to_metal"],
                    "n_fragments": len(Chem.GetMolFrags(mol)),
                }
                agg["as_valid"] += bool(as_scored["has_valid_coordination"])
                agg["as_clash"] += bool(as_scored["v1_clash"])
                agg["multifrag"] += rec["n_fragments"] > 1
                agg["seed_found"] += seed_found

                if seed_found:
                    p2, e2, b2 = _drop_atom(pos, els, bonds, si)
                    ex = (check_molecule(p2, e2, metal_xyz, args.metal, pxyz, bonds=b2)
                          if len(p2) else None)
                    # PRIMARY ENDPOINT
                    rec["seed_excluded_has_valid_coordination"] = (
                        ex["has_valid_coordination"] if ex else False)
                    rec["seed_excluded_n_valid_coordination"] = (
                        ex["n_valid_coordination"] if ex else 0)
                    rec["seed_excluded_primary_violation"] = (
                        ex["primary_violation"] if ex else False)
                    rec["seed_excluded_angular_rms"] = (
                        ex["coordination_rms_angle_dev"] if ex else None)
                    agg["ex_scored"] += 1
                    agg["ex_valid"] += bool(rec["seed_excluded_has_valid_coordination"])

                    # Is the seed built into a real zinc-binding group?
                    # Two hazards handled explicitly:
                    #  1. The supplier reads with sanitize=False to match the
                    #     pre-registered geometry path (coordination_checker.py:171), so
                    #     implicit valences are uncomputed and H-bearing SMARTS such as
                    #     [OX2H1] raise a pre-condition violation. Sanitize a COPY.
                    #  2. `si` indexes the heavy-atom-only arrays from heavy_atoms();
                    #     GetSubstructMatches returns full-molecule indices. Map between
                    #     them rather than assuming they coincide.
                    rec["seed_zbg"] = _seed_zbg(mol, si, zbg)
                    agg["zbg"] += rec["seed_zbg"] is not None

                fh.write(json.dumps(rec) + "\n")

    def pct(a, b):
        return f"{100.0*a/b:.2f}%" if b else "n/a"

    print(f"targets {n_tgt}, molecules {n_mol} -> {out_path}")
    print()
    print("  RETIRED (seed-determined, floor is 100.00% / 13.22 deg):")
    print(f"    as-scored valid coordination : {pct(agg['as_valid'], n_mol)}")
    print()
    print("  PRIMARY (model-generated atoms only; Arm C reference 24.05%):")
    print(f"    seed-excluded valid coord    : {pct(agg['ex_valid'], agg['ex_scored'])}"
          f"   (n={agg['ex_scored']})")
    print("  CO-PRIMARY (Arm C reference 11.59%):")
    print(f"    V1 hard clash                : {pct(agg['as_clash'], n_mol)}")
    print("  SECONDARY:")
    print(f"    seed survived sanitization   : {pct(agg['seed_found'], n_mol)}")
    print(f"    seed built into a ZBG        : {pct(agg['zbg'], agg['ex_scored'])}")
    print(f"    multi-fragment molecules     : {pct(agg['multifrag'], n_mol)}")


if __name__ == "__main__":
    main()
