"""Build Arm C's training dataset: real metal-retained pocket/ligand pairs.

This did not exist before this script. What "GATE_CHECKS.md" / results/step0/GATE_CHECKS.md
call the "2,886 usable complexes" scale check only counted, against the OLD metal-blind LMDB
(the same one Arm B trains on), how many of its already-processed entries correspond to a
Zn-containing receptor with the ligand pose sitting near that Zn. It never re-ran preprocessing
with metals retained — verified by reading scripts/analyze_scale_and_leakage.py directly: the
ligand positions it uses come from `crossdocked_pocket10_processed.lmdb` (predecessor project,
metal-blind), and the freshly-fetched data/crossdocked_receptors/*.pdb files are used only to
check for Zn presence, not as input to any (re-)processing step. No raw CrossDocked ligand SDFs
exist on this machine to re-run the original process_crossdock.py pipeline (checked: no
crossdocked_pocket10 directory tree or *_lig_*.sdf files anywhere under the predecessor repo).

This script sidesteps that by using each fetched receptor PDB's own co-crystallized native
ligand (not CrossDocked's cross-docked pose augmentation) as the training example, reusing the
now-patched process_crossdock.process_ligand_and_pocket for pocket construction — the exact
same code path Arm C training and inference will use, so training-time and inference-time
pocket definitions are guaranteed identical by construction (the "one pocket definition" property
Step 1's own G1 gate required).

Ligand extraction here does not fetch CCD bond orders (unlike scripts/extract_native_ligands.py,
which does, for the eval cohort). Bonds are not used anywhere in diffusion training — verified:
process_ligand_and_pocket only reads atom.GetSymbol() and the conformer's atom positions from
the ligand SDF — so a bond-free synthetic SDF (positions + elements only) is sufficient here and
avoids ~1,500 RCSB CCD lookups this step doesn't need.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Geometry import Point3D

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "DiffSBDD"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import process_crossdock as pc  # noqa: E402
from constants import dataset_params  # noqa: E402
from analyze_scale_and_leakage import inspect_zinc_in_pdb  # noqa: E402

RECEPTORS_DIR = REPO_ROOT / "data" / "crossdocked_receptors"
LIGAND_SDF_DIR = REPO_ROOT / "data" / "arm_c_native_ligands"
OUT_TRAIN = REPO_ROOT / "data" / "arm_c_train.pt"
OUT_VAL = REPO_ROOT / "data" / "arm_c_val.pt"
DIST_CUTOFF = 8.0
MIN_HEAVY_ATOMS = 5
MAX_LIG_TO_ZN_DIST = 8.0

# Common crystallization additives / cryoprotectants / buffer components / monatomic ions —
# excluded so the "ligand" we train on is a real small-molecule binder, not a buffer artifact.
# Deliberately not Step-1-grade curation (no drug-likeness/QED filter, no resolution filter) —
# this is training data, not a benchmark; it needs to be reasonable, not publication-curated.
EXCLUDE_RESNAMES = {
    "HOH", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "ACT", "TRS", "IMD", "CIT", "MPD",
    "DMS", "FMT", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO", "BR",
    "IOD", "ACY", "BME", "MES", "HEPES", "EPE", "TAM", "1PE", "P6G", "PGE", "BOG", "OGA",
    "UNX", "UNL", "NO3", "SCN", "AZI", "CD", "HG", "PB", "SR", "BA", "CS", "RB",
}


def find_native_ligand(structure, catalytic_zn_coords):
    """Best candidate: non-excluded HETATM group, >=MIN_HEAVY_ATOMS heavy atoms, minimum
    distance to any catalytic Zn <= MAX_LIG_TO_ZN_DIST. Returns (residue, min_dist) or None.
    """
    best = None
    for chain in structure[0]:
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            hetflag = residue.id[0]
            if hetflag == " ":  # standard polymer residue, not a ligand
                continue
            if resname in EXCLUDE_RESNAMES:
                continue
            heavy_atoms = [a for a in residue.get_atoms()
                          if a.element not in ("H", "D") and a.get_altloc().strip() in ("", "A")]
            if len(heavy_atoms) < MIN_HEAVY_ATOMS:
                continue
            coords = np.array([a.get_coord() for a in heavy_atoms])
            dmin = min(np.linalg.norm(coords - zc, axis=-1).min() for zc in catalytic_zn_coords)
            if dmin > MAX_LIG_TO_ZN_DIST:
                continue
            if best is None or dmin < best[1]:
                best = (residue, dmin)
    return best


def write_synthetic_ligand_sdf(residue, out_path):
    """Positions + elements only, no bonds — sufficient for process_ligand_and_pocket (see
    module docstring). Deduplicates alt-locs, preferring blank/'A'.
    """
    heavy_atoms, seen = [], set()
    for a in residue.get_atoms():
        if a.element in ("H", "D"):
            continue
        name = a.get_name().strip()
        if name in seen:
            continue
        if a.get_altloc().strip() in ("", "A"):
            seen.add(name)
            heavy_atoms.append(a)
    for a in residue.get_atoms():
        if a.element in ("H", "D"):
            continue
        name = a.get_name().strip()
        if name not in seen:
            seen.add(name)
            heavy_atoms.append(a)

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(heavy_atoms))
    for i, a in enumerate(heavy_atoms):
        elem = a.element.strip().capitalize()
        rd_atom = Chem.Atom(elem)
        mol.AddAtom(rd_atom)
        c = a.get_coord()
        conf.SetAtomPosition(i, Point3D(float(c[0]), float(c[1]), float(c[2])))
    mol.AddConformer(conf)
    final_mol = mol.GetMol()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.write(final_mol)
    writer.close()
    return len(heavy_atoms)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N receptor PDBs (smoke-testing).")
    args = ap.parse_args()

    eval_pdbs = set(t["pdb_id"].lower() for t in
                    torch.load(REPO_ROOT / "data" / "external_zn_test_clean.pt",
                               map_location="cpu")["targets"])

    pdb_files = sorted(RECEPTORS_DIR.glob("*.pdb"))
    if args.limit:
        pdb_files = pdb_files[:args.limit]
    print(f"Scanning {len(pdb_files)} receptor PDBs from {RECEPTORS_DIR}")

    pc.amino_acid_dict = dataset_params["crossdock_metal"]["aa_encoder"]
    lig_atom_dict = dataset_params["crossdock_metal"]["atom_encoder"]

    parser = PDBParser(QUIET=True)
    examples = []
    skip_counts = {"leaked": 0, "no_catalytic_zn": 0, "no_ligand": 0,
                   "processing_error": 0, "no_pocket_atoms": 0}

    for i, pdb_path in enumerate(pdb_files):
        pdb_id = pdb_path.stem.lower()
        if pdb_id in eval_pdbs:
            skip_counts["leaked"] += 1
            continue

        zns = inspect_zinc_in_pdb(str(pdb_path))
        catalytic = [z for z in zns if z["is_catalytic"]]
        if not catalytic:
            skip_counts["no_catalytic_zn"] += 1
            continue

        try:
            structure = parser.get_structure(pdb_id, str(pdb_path))
        except Exception:
            skip_counts["processing_error"] += 1
            continue

        found = find_native_ligand(structure, [z["coord"] for z in catalytic])
        if found is None:
            skip_counts["no_ligand"] += 1
            continue
        residue, dmin = found

        sdf_path = LIGAND_SDF_DIR / f"{pdb_id}.sdf"
        try:
            n_heavy = write_synthetic_ligand_sdf(residue, sdf_path)
            ligand_data, pocket_data = pc.process_ligand_and_pocket(
                str(pdb_path), str(sdf_path), lig_atom_dict, DIST_CUTOFF, ca_only=False)
        except Exception as e:
            skip_counts["processing_error"] += 1
            if skip_counts["processing_error"] <= 5:
                print(f"  [{pdb_id}] processing error: {e}")
            continue

        if pocket_data["pocket_coords"].shape[0] == 0:
            skip_counts["no_pocket_atoms"] += 1
            continue

        examples.append({
            "pdb_id": pdb_id,
            "ligand_pos": torch.tensor(ligand_data["lig_coords"], dtype=torch.float32),
            "ligand_one_hot": torch.tensor(ligand_data["lig_one_hot"], dtype=torch.float32),
            "protein_pos": torch.tensor(pocket_data["pocket_coords"], dtype=torch.float32),
            "protein_one_hot": torch.tensor(pocket_data["pocket_one_hot"], dtype=torch.float32),
            "min_lig_zn_dist": dmin,
            "n_ligand_heavy_atoms": n_heavy,
        })

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(pdb_files)} scanned, {len(examples)} usable so far")

    print(f"\nUsable examples: {len(examples)}")
    print(f"Skipped: {skip_counts}")

    # Sanity check the entire point of this dataset: metal columns must actually fire.
    metal_cols = slice(10, 16)
    n_with_metal_in_pocket = sum(
        1 for e in examples if e["protein_one_hot"][:, metal_cols].sum().item() > 0)
    print(f"Examples with >=1 retained metal atom in the encoded pocket: "
          f"{n_with_metal_in_pocket}/{len(examples)}")
    assert n_with_metal_in_pocket > 0, (
        "Zero examples have a metal atom in the pocket encoding — the whole point of this "
        "dataset failed silently. Do not proceed to training."
    )

    # Deterministic 90/10 split by PDB ID (sorted, not shuffled — reproducible without a seed).
    examples.sort(key=lambda e: e["pdb_id"])
    n_val = max(1, int(0.1 * len(examples)))
    val_examples = examples[::10][:n_val] if n_val else []
    val_ids = set(e["pdb_id"] for e in val_examples)
    train_examples = [e for e in examples if e["pdb_id"] not in val_ids]

    torch.save({"examples": train_examples, "provenance": {
        "source": "native ligands from data/crossdocked_receptors, RCSB-fetched",
        "dist_cutoff": DIST_CUTOFF,
        "pocket_vocab": "crossdock_metal",
        "skip_counts": skip_counts,
    }}, OUT_TRAIN)
    torch.save({"examples": val_examples}, OUT_VAL)
    print(f"\nSaved {len(train_examples)} train / {len(val_examples)} val examples to "
          f"{OUT_TRAIN} / {OUT_VAL}")


if __name__ == "__main__":
    main()
