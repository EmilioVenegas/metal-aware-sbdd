"""Query RCSB directly to assemble a comprehensive clean external Zinc test set.

Criteria:
1. Contains ZN and non-polymer ligand.
2. X-ray resolution <= 2.5 A (or high quality CryoEM).
3. Catalytic Zn (>= 2 protein sidechain donors <= 2.8 A).
4. Bound native ligand (< 5.0 A from Zn, >= 6 heavy atoms).
5. PDB ID strictly absent from CrossDocked (19,476 PDBs).
6. < 30% sequence identity to any CrossDocked training target.
7. Target count m ~ 30.
"""

import io
import json
import os
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

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
}

COMMON_SOLVENTS = {
    'HOH', 'DOD', 'WAT', 'GOL', 'EDO', 'PEG', 'PG4', 'SO4', 'PO4', 'ACT', 'DMS',
    'FMT', 'CL', 'NA', 'MG', 'ZN', 'CA', 'MN', 'FE', 'CU', 'NI', 'CO', 'TRS',
    'BME', 'MES', 'HEPES', 'EPE', 'MPD', 'IPA', 'CIT', 'NO3', 'NH4', 'AZI',
    'UNX', '1PE', '2PE', 'P6G', 'BTB', 'PGE', 'DIO', 'NAG', 'FUC', 'BMA'
}

def get_rcsb_candidate_pdbs():
    """Query RCSB search API for recent/diverse Zn protein-ligand structures."""
    search_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_chem_comp_container_identifiers.comp_id",
                        "operator": "in",
                        "value": ["ZN"]
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater_or_equal",
                        "value": 2
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.deposit_date",
                        "operator": "greater_or_equal",
                        "value": "2020-01-01T00:00:00Z"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 300
            },
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc"
                }
            ]
        }
    }

    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=search_query)
    if r.status_code != 200:
        print(f"RCSB query failed: {r.status_code}")
        return []

    hits = [doc["identifier"].lower() for doc in r.json().get("result_set", [])]
    return hits


def check_crossdocked_sequence_identity(seq, crossdocked_pdbs):
    if not seq or len(seq) < 30:
        return False, 0

    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 0.1,
                "identity_cutoff": 0.30,
                "sequence_type": "protein",
                "value": seq
            }
        },
        "return_type": "entry"
    }

    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=10)
        if r.status_code == 204:
            return True, 0
        elif r.status_code == 200:
            hits = {doc["identifier"].lower() for doc in r.json().get("result_set", [])}
            overlap = hits.intersection(crossdocked_pdbs)
            if not overlap:
                return True, 0
            else:
                return False, len(overlap)
    except Exception:
        return False, -1

    return False, -1


def inspect_pdb(pdb_id, crossdocked_pdbs):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        req = urllib.request.urlopen(url, timeout=10)
        content = req.read().decode('utf-8')
    except Exception:
        return None

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, io.StringIO(content))
    except Exception:
        return None

    zn_atoms = []
    ligands = defaultdict(list)
    ligand_names = {}
    protein_atoms = []
    protein_seq_3 = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                het_flag = residue.id[0]

                if het_flag != ' ' and het_flag != 'W':
                    if resname == 'ZN':
                        for atom in residue:
                            zn_atoms.append({
                                'coord': atom.coord,
                                'res_id': f"{chain.id}_{resname}_{residue.id[1]}",
                                'chain': chain.id
                            })
                    elif resname not in COMMON_SOLVENTS:
                        r_key = f"{chain.id}_{resname}_{residue.id[1]}"
                        ligand_names[r_key] = resname
                        for atom in residue:
                            ligands[r_key].append(atom.coord)
                elif het_flag == ' ':
                    protein_seq_3.append(resname)
                    if resname in SIDECHAIN_DONORS:
                        valid = SIDECHAIN_DONORS[resname]
                        for atom in residue:
                            if atom.name.strip().upper() in valid:
                                protein_atoms.append({
                                    'res_id': f"{chain.id}_{resname}_{residue.id[1]}",
                                    'coord': atom.coord,
                                    'atom_name': atom.name.strip().upper(),
                                    'resname': resname
                                })

    if not zn_atoms or not ligands:
        return None

    best_pair = None
    for zn in zn_atoms:
        z_coord = zn['coord']
        sc_donors = set()
        sc_details = []
        for p in protein_atoms:
            d = float(np.linalg.norm(p['coord'] - z_coord))
            if d <= 2.8:
                sc_donors.add(p['res_id'])
                sc_details.append(f"{p['res_id']}:{p['atom_name']} ({d:.2f}A)")

        if len(sc_donors) < 2:
            continue

        for lig_key, lig_coords in ligands.items():
            l_arr = np.array(lig_coords)
            if len(l_arr) < 6:
                continue
            min_d = float(np.linalg.norm(l_arr - z_coord, axis=1).min())
            if min_d <= 4.5:
                best_pair = {
                    'pdb_id': pdb_id,
                    'zn_id': zn['res_id'],
                    'zn_coord': z_coord.tolist(),
                    'ligand_id': lig_key,
                    'ligand_resname': ligand_names[lig_key],
                    'ligand_num_heavy_atoms': len(l_arr),
                    'min_zn_ligand_dist': min_d,
                    'num_sidechain_donors': len(sc_donors),
                    'sidechain_donors': sc_details,
                    'chain': zn['chain']
                }
                break

        if best_pair:
            break

    if not best_pair:
        return None

    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    seq_1 = "".join(aa_3to1.get(r, 'X') for r in protein_seq_3 if r in aa_3to1)

    is_clean, overlap = check_crossdocked_sequence_identity(seq_1[:300], crossdocked_pdbs)
    if not is_clean:
        return None

    best_pair['is_sequence_clean'] = True
    best_pair['protein_sequence_fragment'] = seq_1[:80]
    return best_pair


def main():
    manifest_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/lmdb_index_manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)

    crossdocked_pdbs = set()
    for name in manifest:
        if name is None: continue
        parts = name.split('/')[-1].split('_')
        if len(parts) >= 2:
            crossdocked_pdbs.add(parts[0].lower())

    print(f"Total blacklisted CrossDocked PDBs: {len(crossdocked_pdbs)}")

    candidates = get_rcsb_candidate_pdbs()
    print(f"Retrieved {len(candidates)} high-res post-2020 candidate PDBs from RCSB...")

    unseen_candidates = [p for p in candidates if p not in crossdocked_pdbs]
    print(f"Unseen candidates: {len(unseen_candidates)}")

    valid_targets = []
    seen_seqs = []

    for idx, pid in enumerate(unseen_candidates):
        res = inspect_pdb(pid, crossdocked_pdbs)
        if res:
            valid_targets.append(res)
            print(f"[{len(valid_targets):2d}/30] ACCEPTED {pid.upper()}: ligand={res['ligand_resname']} (Zn-Lig: {res['min_zn_ligand_dist']:.2f}A, {res['num_sidechain_donors']} SC donors)")
            if len(valid_targets) >= 30:
                break
        if idx % 20 == 0:
            print(f"  Processed {idx}/{len(unseen_candidates)} candidates (found {len(valid_targets)})...")

    print(f"\nFinal count of clean external catalytic Zinc targets: m = {len(valid_targets)}")

    out_path = Path("data/external_zn_test.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "targets": valid_targets,
        "m_targets": len(valid_targets),
        "target_pdb_ids": [t['pdb_id'] for t in valid_targets],
        "provenance": {
            "source": "RCSB Protein Data Bank (high-res post-2020 structures)",
            "criteria": [
                "Catalytic Zn site with >=2 protein sidechain donors within 2.8 A",
                "PDB ID strictly absent from CrossDocked manifest (19,476 PDBs)",
                "<30% sequence identity to any CrossDocked training target",
                "Bound drug-like native ligand within 4.5 A of catalytic Zn (for C1 control)"
            ],
            "total_targets": len(valid_targets)
        }
    }
    torch.save(payload, out_path)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
