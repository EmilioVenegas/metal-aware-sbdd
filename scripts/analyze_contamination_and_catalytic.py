"""Analyze test-set contamination and classify catalytic vs incidental metals.

Task 2: Quantify how many of the 148 metalloprotein test targets were in DiffSBDD's original training set.
Task 3: Classify metal sites as catalytic (>=2 protein sidechain donors, in pocket) vs incidental (<2 sidechain donors).
"""

import json
import os
import sys
import urllib.request
import io
import warnings
from collections import defaultdict, Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

METALS = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}

# Amino acid sidechain donor definitions:
# Standard residues: HIS (NE2, ND1), ASP (OD1, OD2), GLU (OE1, OE2), CYS (SG),
# SER (OG), THR (OG1), TYR (OH), ASN (OD1, ND2), GLN (OE1, NE2), LYS (NZ), MET (SD),
# and backbone atoms (O, N).
SIDECHAIN_DONOR_ATOMS = {
    'HIS': {'NE2', 'ND1', 'CG', 'CD2', 'CE1'},
    'ASP': {'OD1', 'OD2', 'CG'},
    'GLU': {'OE1', 'OE2', 'CD'},
    'CYS': {'SG', 'CB'},
    'SER': {'OG', 'CB'},
    'THR': {'OG1', 'CB'},
    'TYR': {'OH', 'CZ'},
    'ASN': {'OD1', 'ND2', 'CG'},
    'GLN': {'OE1', 'NE2', 'CD'},
    'LYS': {'NZ', 'CE'},
    'MET': {'SD', 'CG', 'CE'},
    'TRP': {'NE1', 'CD1', 'CZ2'},
}

def analyze_target_pdb(pdb_id):
    """Fetch CIF from RCSB and analyze metal coordination in the binding site."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        req = urllib.request.urlopen(url, timeout=15)
        content = req.read().decode('utf-8')
    except Exception as e:
        return pdb_id, None, f"Fetch error: {e}"

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, io.StringIO(content))
    except Exception as e:
        return pdb_id, None, f"Parse error: {e}"

    metal_records = [] # (metal_resname, metal_coord, metal_id)
    ligand_atoms = []
    protein_atoms = [] # (resname, res_id, atom_name, coord)

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                res_id = residue.id
                het_flag = res_id[0]

                if het_flag != ' ' and het_flag != 'W':
                    if resname in METALS:
                        for atom in residue:
                            metal_records.append({
                                'metal': resname,
                                'coord': atom.coord,
                                'residue_id': f"{chain.id}_{resname}_{res_id[1]}",
                            })
                    else:
                        # Ligand atom
                        for atom in residue:
                            ligand_atoms.append(atom.coord)
                elif het_flag == ' ':
                    # Standard protein residue
                    for atom in residue:
                        protein_atoms.append({
                            'resname': resname,
                            'res_full_id': f"{chain.id}_{resname}_{res_id[1]}",
                            'atom_name': atom.name.strip().upper(),
                            'coord': atom.coord,
                        })

    if not metal_records:
        return pdb_id, [], "No metals found"

    ligand_coords = np.array(ligand_atoms) if ligand_atoms else np.empty((0, 3))
    
    metal_classifications = []
    for m in metal_records:
        m_coord = m['coord']
        metal_elem = m['metal']

        # 1. Distance to ligand
        if len(ligand_coords) > 0:
            lig_dists = np.linalg.norm(ligand_coords - m_coord, axis=1)
            min_lig_dist = float(lig_dists.min())
        else:
            min_lig_dist = 999.0

        # Only consider metals in the binding site (< 6.0 A from ligand)
        if min_lig_dist > 6.0:
            continue

        # 2. Find coordinating protein sidechain residues (distance <= 2.8 A)
        coordinating_sidechain_residues = set()
        coordinating_atoms = []
        for p in protein_atoms:
            p_dist = np.linalg.norm(p['coord'] - m_coord)
            if p_dist <= 2.8:
                rname = p['resname']
                aname = p['atom_name']
                is_sidechain = (rname in SIDECHAIN_DONOR_ATOMS and aname in SIDECHAIN_DONOR_ATOMS[rname])
                if is_sidechain:
                    coordinating_sidechain_residues.add(p['res_full_id'])
                    coordinating_atoms.append(f"{p['res_full_id']}:{aname} ({p_dist:.2f}A)")

        num_sidechain_donors = len(coordinating_sidechain_residues)
        is_catalytic = (num_sidechain_donors >= 2)

        metal_classifications.append({
            'metal': metal_elem,
            'residue_id': m['residue_id'],
            'min_lig_dist': min_lig_dist,
            'num_sidechain_donors': num_sidechain_donors,
            'coordinating_residues': list(coordinating_sidechain_residues),
            'coordinating_atoms': coordinating_atoms,
            'is_catalytic': is_catalytic,
            'classification': 'catalytic' if is_catalytic else 'incidental'
        })

    return pdb_id, metal_classifications, "OK"


if __name__ == '__main__':
    # Step 2: Test-set contamination
    split_path = 'data/metal_target_split.pt'
    orig_split_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/crossdocked_split.pt'
    manifest_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/lmdb_index_manifest.json'

    with open(manifest_path) as f:
        manifest = json.load(f)

    new_split = torch.load(split_path)
    orig_split = torch.load(orig_split_path)

    orig_train_targets = {manifest[idx].split('/')[0] for idx in orig_split['train'] if manifest[idx]}
    orig_holdout_targets = {manifest[idx].split('/')[0] for idx in set(orig_split['val']) | set(orig_split['test']) if manifest[idx]}

    test_targets = new_split['test_targets']
    target_primary_metal = new_split['target_primary_metal']

    print(f"Total test targets in new split: {len(test_targets)}")
    metallo_test_targets = [t for t in test_targets if target_primary_metal.get(t, 'NO_METAL') != 'NO_METAL']
    print(f"Metalloprotein test targets: {len(metallo_test_targets)}")

    clean_by_metal = defaultdict(list)
    contam_by_metal = defaultdict(list)

    for t in metallo_test_targets:
        m = target_primary_metal[t]
        if t in orig_train_targets:
            contam_by_metal[m].append(t)
        else:
            clean_by_metal[m].append(t)

    print("\n" + "=" * 70)
    print("TEST-SET CONTAMINATION BY METAL (against base checkpoint training set):")
    print(f"{'Metal':10s} | {'Clean (Unseen)':>15s} | {'Contaminated (Seen)':>20s} | {'Total Test':>10s} | {'Clean %':>8s}")
    print("-" * 70)
    all_metals = ["ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU"]
    total_clean = total_contam = 0
    for m in all_metals:
        c = len(clean_by_metal[m])
        cnt = len(contam_by_metal[m])
        tot = c + cnt
        pct = (c / tot * 100) if tot > 0 else 0
        print(f"{m:10s} | {c:15d} | {cnt:20d} | {tot:10d} | {pct:7.1f}%")
        total_clean += c
        total_contam += cnt

    print("-" * 70)
    print(f"{'TOTAL':10s} | {total_clean:15d} | {total_contam:20d} | {total_clean+total_contam:10d} | {total_clean/(total_clean+total_contam)*100:7.1f}%")
    print("=" * 70)
