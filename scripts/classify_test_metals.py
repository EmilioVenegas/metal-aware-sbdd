"""Classify metals in the 148 test targets as catalytic vs incidental.

A metal site is classified as:
- Catalytic: coordinated by >= 2 protein sidechain donors (distance <= 2.8 A) within binding site (< 5.0 A from ligand).
- Incidental: < 2 protein sidechain donors (surface bound, solvent/crystallisation additive).
"""

import io
import json
import urllib.request
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

METALS = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}

SIDECHAIN_DONORS = {
    'HIS': {'NE2', 'ND1'},
    'ASP': {'OD1', 'OD2'},
    'GLU': {'OE1', 'OE2'},
    'CYS': {'SG'},
    'SER': {'OG'},
    'THR': {'OG1'},
    'TYR': {'OH'},
    'ASN': {'OD1', 'ND2'},
    'GLN': {'OE1', 'NE2'},
    'LYS': {'NZ'},
    'MET': {'SD'},
    'TRP': {'NE1'},
}

def analyze_structure(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        req = urllib.request.urlopen(url, timeout=12)
        content = req.read().decode('utf-8')
    except Exception as e:
        return pdb_id, None

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, io.StringIO(content))
    except Exception:
        return pdb_id, None

    metals_found = []
    ligand_atoms = []
    protein_sidechain_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                het_flag = residue.id[0]

                if het_flag != ' ' and het_flag != 'W':
                    if resname in METALS:
                        for atom in residue:
                            metals_found.append({
                                'metal': resname,
                                'coord': atom.coord,
                                'res_id': f"{chain.id}_{resname}_{residue.id[1]}"
                            })
                    else:
                        for atom in residue:
                            ligand_atoms.append(atom.coord)
                elif het_flag == ' ':
                    if resname in SIDECHAIN_DONORS:
                        valid_names = SIDECHAIN_DONORS[resname]
                        for atom in residue:
                            aname = atom.name.strip().upper()
                            if aname in valid_names:
                                protein_sidechain_atoms.append({
                                    'res_id': f"{chain.id}_{resname}_{residue.id[1]}",
                                    'coord': atom.coord
                                })

    if not metals_found:
        return pdb_id, []

    lig_coords = np.array(ligand_atoms) if ligand_atoms else np.empty((0, 3))
    results = []

    for m in metals_found:
        m_coord = m['coord']
        if len(lig_coords) > 0:
            lig_dist = float(np.linalg.norm(lig_coords - m_coord, axis=1).min())
        else:
            lig_dist = 999.0

        # In pocket cutoff: within 5.5 A of ligand
        if lig_dist > 5.5:
            continue

        coordinating_res = set()
        for p in protein_sidechain_atoms:
            d = float(np.linalg.norm(p['coord'] - m_coord))
            if d <= 2.8:
                coordinating_res.add(p['res_id'])

        num_sc = len(coordinating_res)
        is_catalytic = (num_sc >= 2)

        results.append({
            'pdb_id': pdb_id,
            'metal': m['metal'],
            'res_id': m['res_id'],
            'lig_dist': lig_dist,
            'num_sidechain_donors': num_sc,
            'is_catalytic': is_catalytic,
            'category': 'catalytic' if is_catalytic else 'incidental'
        })

    return pdb_id, results


def main():
    split = torch.load('data/metal_target_split.pt')
    manifest_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/lmdb_index_manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_targets = split['test_targets']
    target_primary_metal = split['target_primary_metal']
    metallo_test_targets = [t for t in test_targets if target_primary_metal.get(t, 'NO_METAL') != 'NO_METAL']

    # Find representative PDBs for each test target
    target_to_pdbs = defaultdict(set)
    for idx in split['test_indices']:
        name = manifest[idx]
        t = name.split('/')[0]
        if t in metallo_test_targets:
            parts = name.split('/')[-1].split('_')
            if len(parts) >= 2:
                target_to_pdbs[t].add(parts[0].lower())

    all_test_pdbs = sorted(list({p for pdbs in target_to_pdbs.values() for p in pdbs}))
    print(f"Analyzing {len(all_test_pdbs)} unique PDBs across {len(metallo_test_targets)} metalloprotein test targets...")

    with Pool(16) as pool:
        results = pool.map(analyze_structure, all_test_pdbs)

    pdb_results = {pid: res for pid, res in results if res is not None}

    # Classify each target
    target_classifications = {}
    metal_cat_counts = defaultdict(lambda: {'catalytic': 0, 'incidental': 0, 'no_pocket_metal': 0})

    for t in metallo_test_targets:
        primary_m = target_primary_metal[t]
        pdbs = target_to_pdbs[t]
        
        # Aggregate metal observations across PDBs for this target
        all_obs = []
        for p in pdbs:
            for obs in pdb_results.get(p, []):
                if obs['metal'] == primary_m:
                    all_obs.append(obs)

        if not all_obs:
            # Check any metal
            for p in pdbs:
                all_obs.extend(pdb_results.get(p, []))

        if not all_obs:
            status = 'incidental' # conservatively incidental if not detected in CIF
            metal_cat_counts[primary_m]['no_pocket_metal'] += 1
        else:
            # If any observation has >= 2 sidechain donors -> catalytic
            max_donors = max(obs['num_sidechain_donors'] for obs in all_obs)
            if max_donors >= 2:
                status = 'catalytic'
                metal_cat_counts[primary_m]['catalytic'] += 1
            else:
                status = 'incidental'
                metal_cat_counts[primary_m]['incidental'] += 1

        target_classifications[t] = {
            'target': t,
            'primary_metal': primary_m,
            'classification': status,
            'num_pdbs': len(pdbs),
            'observations': all_obs
        }

    with open('data/test_target_metal_classification.json', 'w') as f:
        json.dump(target_classifications, f, indent=2)

    print("\n" + "=" * 70)
    print("METALLOPROTEIN TEST TARGETS: CATALYTIC VS INCIDENTAL CLASSIFICATION")
    print(f"{'Metal':8s} | {'Catalytic (>=2 SC)':>18s} | {'Incidental (<2 SC)':>18s} | {'Total Targets':>14s} | {'Catalytic %':>11s}")
    print("-" * 70)
    tot_cat = tot_inc = tot_all = 0
    for m in ["ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU"]:
        c = metal_cat_counts[m]['catalytic']
        inc = metal_cat_counts[m]['incidental'] + metal_cat_counts[m]['no_pocket_metal']
        tot = c + inc
        pct = (c / tot * 100) if tot > 0 else 0
        print(f"{m:8s} | {c:18d} | {inc:18d} | {tot:14d} | {pct:10.1f}%")
        tot_cat += c
        tot_inc += inc
        tot_all += tot

    print("-" * 70)
    print(f"{'TOTAL':8s} | {tot_cat:18d} | {tot_inc:18d} | {tot_all:14d} | {tot_cat/tot_all*100:10.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
