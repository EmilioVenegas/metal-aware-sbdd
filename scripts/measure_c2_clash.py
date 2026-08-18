#!/usr/bin/env python
"""C2 control: protein-atom clash rate paired within molecule.

Implements results/step1/ANALYSIS_PLAN.md section 5 (C2 control).
For each generated molecule, measures clash and shell occupancy against ordinary pocket
protein heavy atoms using the identical distance cutoffs (1.70 A clash, 2.70 A shell).

Emits JSONL with paired per-molecule measurements.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

cif_parser = MMCIFParser(QUIET=True)
pdb_parser = PDBParser(QUIET=True)


def get_pocket_heavy_atoms(pdb_id: str, ref_ligand_str: str, struct_dir: str = "data/external_pdbs",
                           dist_cutoff: float = 8.0) -> np.ndarray:
    cif_p = Path(struct_dir) / f"{pdb_id}.cif"
    pdb_p = Path(struct_dir) / f"{pdb_id}.pdb"
    if cif_p.exists():
        structure = cif_parser.get_structure(pdb_id, str(cif_p))
    elif pdb_p.exists():
        structure = pdb_parser.get_structure(pdb_id, str(pdb_p))
    else:
        raise FileNotFoundError(f"Structure for {pdb_id} not found in {struct_dir}")

    model = next(structure.get_models())
    chain_id, resi_str = ref_ligand_str.split(":")
    resi = int(resi_str)

    ref_res = None
    for res in model[chain_id].get_residues():
        if res.id[1] == resi:
            ref_res = res
            break
    if ref_res is None:
        raise ValueError(f"Ref ligand {ref_ligand_str} not found in {pdb_id}")

    lig_coords = np.array([a.get_coord() for a in ref_res.get_atoms() if a.element != "H"])

    pocket_atoms = []
    for res in model.get_residues():
        if res.id[1] == resi and res.get_parent().id == chain_id:
            continue
        if is_aa(res.get_resname(), standard=True):
            res_coords = np.array([a.get_coord() for a in res.get_atoms() if a.element != "H"])
            if len(res_coords) == 0:
                continue
            dists = np.linalg.norm(res_coords[:, None, :] - lig_coords[None, :, :], axis=2)
            if dists.min() < dist_cutoff:
                for a in res.get_atoms():
                    if a.element != "H":
                        pocket_atoms.append(a.get_coord().copy())
    return np.array(pocket_atoms, dtype=float)


def measure_target_c2(sdf_path: Path, zn_coord: np.ndarray, pocket_atoms: np.ndarray,
                      pdb_id: str) -> list[dict]:
    results = []
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for mol_idx, mol in enumerate(suppl):
        if mol is None or mol.GetNumConformers() == 0:
            results.append({
                "pdb_id": pdb_id, "mol_index": mol_idx, "unreadable": True
            })
            continue

        conf = mol.GetConformer()
        lig_pos = []
        for a in mol.GetAtoms():
            if a.GetAtomicNum() > 1:
                pos = conf.GetAtomPosition(a.GetIdx())
                lig_pos.append([pos.x, pos.y, pos.z])

        if not lig_pos:
            results.append({
                "pdb_id": pdb_id, "mol_index": mol_idx, "unreadable": True
            })
            continue

        lig_coords = np.array(lig_pos, dtype=float)

        # Distance to metal center
        d_zn = np.linalg.norm(lig_coords - zn_coord, axis=1)
        min_dist_zn = float(d_zn.min())
        metal_clash_v1 = bool(min_dist_zn < 1.70)
        metal_shell_2_7 = bool(min_dist_zn < 2.70)

        # Distance to pocket protein atoms
        # pairwise dists: (N_lig, N_prot)
        d_prot = np.linalg.norm(lig_coords[:, None, :] - pocket_atoms[None, :, :], axis=2)
        min_dist_per_prot_atom = d_prot.min(axis=0)  # (N_prot,)
        min_dist_prot = float(min_dist_per_prot_atom.min())

        n_prot = len(pocket_atoms)
        n_prot_clash_1_7 = int(np.sum(min_dist_per_prot_atom < 1.70))
        n_prot_shell_2_7 = int(np.sum(min_dist_per_prot_atom < 2.70))

        prot_clash_rate_1_7 = float(n_prot_clash_1_7 / n_prot) if n_prot > 0 else 0.0
        prot_shell_rate_2_7 = float(n_prot_shell_2_7 / n_prot) if n_prot > 0 else 0.0
        mol_has_prot_clash_1_7 = bool(n_prot_clash_1_7 > 0)
        mol_has_prot_shell_2_7 = bool(n_prot_shell_2_7 > 0)

        results.append({
            "pdb_id": pdb_id,
            "mol_index": mol_idx,
            "unreadable": False,
            "n_heavy_atoms": len(lig_coords),
            "n_pocket_protein_atoms": n_prot,
            "min_dist_to_zn": round(min_dist_zn, 3),
            "metal_clash_v1": metal_clash_v1,
            "metal_shell_2_7": metal_shell_2_7,
            "min_dist_to_protein": round(min_dist_prot, 3),
            "mol_has_protein_clash_1_7": mol_has_prot_clash_1_7,
            "protein_atoms_clashed_1_7": n_prot_clash_1_7,
            "protein_clash_rate_1_7": round(prot_clash_rate_1_7, 5),
            "mol_has_protein_shell_2_7": mol_has_prot_shell_2_7,
            "protein_atoms_in_shell_2_7": n_prot_shell_2_7,
            "protein_shell_rate_2_7": round(prot_shell_rate_2_7, 5),
            # Paired difference (metal indicator - mean protein atom clash probability)
            "paired_clash_diff_1_7": round(float(metal_clash_v1) - prot_clash_rate_1_7, 5),
            "paired_shell_diff_2_7": round(float(metal_shell_2_7) - prot_shell_rate_2_7, 5),
        })

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="data/external_zn_test_clean.pt")
    ap.add_argument("--sdf-dir", default="results/step1/generation/sdf")
    ap.add_argument("--struct-dir", default="data/external_pdbs")
    ap.add_argument("--out", default="results/step1/checker/c2_protein_clash.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.targets, map_location="cpu", weights_only=False)
    records = blob["targets"] if isinstance(blob, dict) and "targets" in blob else blob

    sdf_dir = Path(args.sdf_dir)
    n_tgt, n_mol = 0, 0
    with open(out_path, "w") as out_f:
        for r in records:
            pdb_id = r["pdb_id"]
            sdf_p = sdf_dir / f"{pdb_id}.sdf"
            if not sdf_p.exists():
                continue

            chain, _, resi = r["ligand_id"].split("_")
            ref_ligand = f"{chain}:{resi}"
            zn_coord = np.array(r["zn_coord"], dtype=float)

            try:
                pocket_atoms = get_pocket_heavy_atoms(pdb_id, ref_ligand, args.struct_dir)
            except Exception as e:
                print(f"Error getting pocket for {pdb_id}: {e}", file=sys.stderr)
                continue

            res_list = measure_target_c2(sdf_p, zn_coord, pocket_atoms, pdb_id)
            for item in res_list:
                out_f.write(json.dumps(item) + "\n")
                n_mol += 1
            n_tgt += 1

    print(f"C2 measured: {n_tgt} targets, {n_mol} molecules -> {args.out}")


if __name__ == "__main__":
    main()
