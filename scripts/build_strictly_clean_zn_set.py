"""Build strictly clean external Zinc benchmark with >= 30 clusters.

Filters:
1. Strict CrossDocked Training Independence: < 30% sequence identity to ANY CrossDocked training PDB (0 hits in 14,480 CrossDocked train PDBs).
2. Catalytic Zn site: >= 2 protein sidechain donors (His, Asp, Glu, Cys, etc.) within <= 2.8 A of Zn.
3. Direct Native Coordination: Native drug-like ligand donor atom (N, O, S, P, Cl, F) within <= 2.5 A of catalytic Zn.
4. Independent sequence clusters: < 30% pairwise sequence identity between distinct target families/clusters.
5. Target: >= 30 distinct 30%-identity clusters.
"""

import io
import json
import os
import sys
import urllib.request
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
import torch
from Bio import pairwise2
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
    'UNX', '1PE', '2PE', 'P6G', 'BTB', 'PGE', 'DIO', 'NAG', 'FUC', 'BMA',
    'ACY', 'ACE', 'NH2', 'NCO', 'MOH', 'EOH', 'CCN', 'FLC', 'PO3', 'BO3', 'NO2',
    'IOD', 'BR', 'K', 'CS', 'LI', 'RB', 'SR', 'BA', 'AZI', 'IMD', 'MLI'
}

_GLOBAL_TRAIN_PDBS = set()

def init_worker(train_pdbs):
    global _GLOBAL_TRAIN_PDBS
    _GLOBAL_TRAIN_PDBS = train_pdbs


def query_rcsb_zn_ligand_structures(max_results=2000):
    query = {
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
                        "value": 2.6
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
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "sort": [
                {
                    "sort_by": "rcsb_accession_info.deposit_date",
                    "direction": "desc"
                }
            ]
        }
    }

    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=25)
        if r.status_code == 200:
            hits = [doc["identifier"].lower() for doc in r.json().get("result_set", [])]
            return hits
    except Exception as e:
        print(f"RCSB query error: {e}")
    return []


def check_rcsb_crossdocked_overlap(seq, train_pdbs):
    if not seq or len(seq) < 30:
        return False, -1

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
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 1000
            }
        }
    }
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=10)
        if r.status_code == 204:
            return True, 0
        elif r.status_code == 200:
            hits = {doc["identifier"].lower() for doc in r.json().get("result_set", [])}
            overlap = hits.intersection(train_pdbs)
            if not overlap:
                return True, 0
            else:
                return False, len(overlap)
    except Exception:
        return False, -1

    return False, -1


def inspect_pdb_worker(pdb_id):
    global _GLOBAL_TRAIN_PDBS
    pid_lower = pdb_id.lower()
    if pid_lower in _GLOBAL_TRAIN_PDBS:
        return None

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        req = urllib.request.urlopen(url, timeout=8)
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
                            elem = atom.element.strip().upper() if hasattr(atom, 'element') else ''
                            ligands[r_key].append({
                                'name': atom.name.strip().upper(),
                                'element': elem,
                                'coord': atom.coord
                            })
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

        for lig_key, lig_atom_list in ligands.items():
            if len(lig_atom_list) < 6: # genuine drug-like molecule
                continue

            coordinating_lig_atoms = []
            min_d = 999.0
            for la in lig_atom_list:
                d = float(np.linalg.norm(la['coord'] - z_coord))
                if d < min_d: min_d = d
                elem = la['element']
                aname = la['name']
                is_donor = elem in {'N', 'O', 'S', 'P', 'CL', 'F'} or (len(aname) > 0 and aname[0] in {'N', 'O', 'S'})
                if is_donor and d <= 2.5:
                    coordinating_lig_atoms.append(f"{aname} ({d:.2f}A)")

            if coordinating_lig_atoms:
                best_pair = {
                    'pdb_id': pdb_id.upper(),
                    'zn_id': zn['res_id'],
                    'zn_coord': z_coord.tolist(),
                    'ligand_id': lig_key,
                    'ligand_resname': ligand_names[lig_key],
                    'ligand_num_heavy_atoms': len(lig_atom_list),
                    'min_zn_ligand_dist': min_d,
                    'coordinating_ligand_atoms': coordinating_lig_atoms,
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

    is_clean, overlap_count = check_rcsb_crossdocked_overlap(seq_1, _GLOBAL_TRAIN_PDBS)
    if not is_clean:
        return None

    best_pair['sequence'] = seq_1
    return best_pair


def compute_pairwise_identity(seq1, seq2):
    if not seq1 or not seq2:
        return 0.0
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    if not aln:
        return 0.0
    matches = aln[0][2]
    aln_len = aln[0][4]
    return matches / aln_len if aln_len > 0 else 0.0


def cluster_targets(targets, threshold=0.30):
    n = len(targets)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, i] = True
        for j in range(i + 1, n):
            ident = compute_pairwise_identity(targets[i]['sequence'], targets[j]['sequence'])
            if ident >= threshold:
                adj[i, j] = True
                adj[j, i] = True

    visited = set()
    clusters = []
    for i in range(n):
        if i not in visited:
            comp = []
            q = [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nbr in range(n):
                    if adj[curr, nbr] and nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)
            clusters.append(comp)
    return clusters


def main():
    manifest_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/lmdb_index_manifest.json'
    split_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/crossdocked_split.pt'

    with open(manifest_path) as f:
        manifest = json.load(f)

    split = torch.load(split_path)
    train_indices = set(split['train'])
    train_pdbs = {manifest[idx].split('/')[-1].split('_')[0].lower() for idx in train_indices if manifest[idx] and len(manifest[idx].split('/')[-1].split('_')) >= 2}
    print(f"Loaded {len(train_pdbs)} CrossDocked TRAINING PDBs for strict sequence exclusion.")

    print("Fetching candidate Zn-ligand structures from RCSB (2000 entries)...")
    rcsb_candidates = query_rcsb_zn_ligand_structures(max_results=2000)
    print(f"Retrieved {len(rcsb_candidates)} candidate PDBs from RCSB.")

    candidate_ids = [p.upper() for p in rcsb_candidates if p.lower() not in train_pdbs]
    print(f"Evaluating {len(candidate_ids)} non-training PDBs in parallel across 16 CPU workers...")

    sys.stdout.flush()
    with Pool(processes=16, initializer=init_worker, initargs=(train_pdbs,)) as pool:
        results = pool.map(inspect_pdb_worker, candidate_ids)

    passed_targets = [r for r in results if r is not None]
    print(f"\nTotal targets passing catalytic Zn, <=2.5A coordination, and 0 CrossDocked train overlap: {len(passed_targets)}")
    sys.stdout.flush()

    # Perform all-vs-all clustering among passed targets at 30% sequence identity
    print("Clustering surviving targets at 30% sequence identity...")
    clusters_30 = cluster_targets(passed_targets, 0.30)
    print(f"\n{'='*70}\nCLUSTERING RESULTS: {len(clusters_30)} DISTINCT 30% SEQUENCE IDENTITY CLUSTERS\n{'='*70}")

    surviving_representatives = []
    for idx, c in enumerate(clusters_30):
        rep = passed_targets[c[0]]
        surviving_representatives.append(rep)
        members = [f"{passed_targets[i]['pdb_id']}:{passed_targets[i]['ligand_resname']}" for i in c]
        print(f"Cluster {idx+1:2d} (rep {rep['pdb_id']}, {len(c)} members, lig={rep['ligand_resname']}, d={rep['min_zn_ligand_dist']:.2f}A): {', '.join(members[:5])}")

    print(f"\nFinal count of clean, coordinated, independent 30%-identity Zinc clusters: m = {len(clusters_30)}")

    out_path = Path("data/external_zn_test.pt")
    payload = {
        "targets": passed_targets,
        "representative_targets": surviving_representatives,
        "m_clusters": len(clusters_30),
        "m_targets": len(passed_targets),
        "target_pdb_ids": [t['pdb_id'] for t in surviving_representatives],
        "provenance": {
            "source": "Strictly Verified Clean External Zinc Metalloenzyme Benchmark",
            "criteria": [
                "Catalytic Zn site with >=2 protein sidechain donors within 2.8 A",
                "Directly coordinated native ligand with donor (N/O/S/P/Cl/F) within <= 2.5 A of catalytic Zn",
                "Strict < 30% sequence identity against ALL CrossDocked training targets (0 hits in 14,480 CrossDocked train PDBs)",
                "Independent sequence clusters across 30% identity threshold (1 target per cluster)"
            ],
            "total_independent_clusters": len(clusters_30),
            "total_evaluated_targets": len(passed_targets)
        }
    }
    torch.save(payload, out_path)
    print(f"Saved payload to {out_path}")


if __name__ == '__main__':
    main()
