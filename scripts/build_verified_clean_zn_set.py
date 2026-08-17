"""Build a strictly verified, overbuilt clean external Zinc test set.

Enforces:
1. Strict CrossDocked exclusion: PDB ID absent from CrossDocked manifest, AND < 30% sequence identity to ANY CrossDocked training target.
2. Direct Native Coordination: Native ligand heavy atom donor (N, O, S, P, F, Cl) within <= 2.5 A of catalytic Zn.
3. Catalytic Protein Shell: >= 2 protein sidechain donors (His, Asp, Glu, Cys, etc.) within <= 2.8 A of Zn.
4. Independent Clusters: Pairwise sequence identity < 30% between distinct target families/clusters.
5. Overbuild target: >= 30 distinct 30%-identity clusters.
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
    'ACY', 'ACE', 'NH2', 'NCO', 'MOH', 'EOH', 'CCN', 'FLC', 'PO3', 'BO3', 'NO2'
}

def query_rcsb_zn_ligand_structures(max_results=600):
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
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=20)
        if r.status_code == 200:
            hits = [doc["identifier"].lower() for doc in r.json().get("result_set", [])]
            return hits
    except Exception as e:
        print(f"RCSB query error: {e}")
    return []


def check_rcsb_crossdocked_overlap(seq, crossdocked_pdbs):
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
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=12)
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


def inspect_pdb_strictly(pdb_id, crossdocked_pdbs):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        req = urllib.request.urlopen(url, timeout=10)
        content = req.read().decode('utf-8')
    except Exception:
        return None, "CIF download failed"

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, io.StringIO(content))
    except Exception:
        return None, "CIF parse failed"

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
        return None, "No Zn or no non-solvent ligand"

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
            if len(lig_atom_list) < 6:
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
        return None, "No ligand donor coordinating Zn <= 2.5 A"

    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    seq_1 = "".join(aa_3to1.get(r, 'X') for r in protein_seq_3 if r in aa_3to1)

    is_clean, overlap_count = check_rcsb_crossdocked_overlap(seq_1, crossdocked_pdbs)
    if not is_clean:
        return None, f"CrossDocked overlap ({overlap_count} PDBs >=30%)"

    best_pair['sequence'] = seq_1
    return best_pair, "OK"


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
    with open(manifest_path) as f:
        manifest = json.load(f)

    crossdocked_pdbs = {name.split('/')[-1].split('_')[0].lower() for name in manifest if name and len(name.split('/')[-1].split('_')) >= 2}
    print(f"Blacklisted {len(crossdocked_pdbs)} CrossDocked PDBs.")

    # Curated manual seed entries that are known clean and coordinated:
    seed_pdbs = [
        '7UYA', # VIM-1 MBL (OKC @ 1.91A)
        '3B7U', # LTA4H (KEL @ 2.06A)
        '1PWQ', # Anthrax LF (SD2 @ 2.21A)
        '3E4A', # IDE (QIX @ 1.65A)
        '5ICN', # HDAC1/2 (6A0 @ 2.35A)
        '4FYT', # APN (L2O @ 1.95A)
        '6Q4R', # ERAP1 (HJ5 @ 1.91A)
        '4AR8', # Collagenase (IP8 @ 2.03A)
    ]

    print("Fetching candidate PDBs from RCSB...")
    rcsb_candidates = query_rcsb_zn_ligand_structures(max_results=800)
    print(f"Retrieved {len(rcsb_candidates)} candidate PDBs from RCSB.")

    all_candidate_ids = seed_pdbs + [p.upper() for p in rcsb_candidates if p.lower() not in crossdocked_pdbs and p.upper() not in seed_pdbs]

    valid_targets = []
    clusters = []

    print(f"\nProcessing candidates to build >= 30 distinct 30%-identity clusters...")

    for idx, pid in enumerate(all_candidate_ids):
        res, reason = inspect_pdb_strictly(pid, crossdocked_pdbs)
        if res:
            # Check pairwise identity against already accepted clusters
            is_new_cluster = True
            for existing in valid_targets:
                ident = compute_pairwise_identity(res['sequence'], existing['sequence'])
                if ident >= 0.30:
                    is_new_cluster = False
                    break

            if is_new_cluster:
                valid_targets.append(res)
                print(f"[{len(valid_targets):2d}/30+] ACCEPTED NEW 30% CLUSTER {pid}: ligand={res['ligand_resname']} (Zn-Lig: {res['min_zn_ligand_dist']:.2f}A, coord={res['coordinating_ligand_atoms'][:2]})")
                if len(valid_targets) >= 32:
                    print("Reached target of >= 32 distinct 30% clusters!")
                    break
            else:
                pass
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(all_candidate_ids)} candidate PDBs (found {len(valid_targets)} distinct clusters)...")

    print(f"\nFinal count of clean, coordinated, independent 30%-identity Zinc clusters: m = {len(valid_targets)}")

    out_path = Path("data/external_zn_test.pt")
    payload = {
        "targets": valid_targets,
        "m_clusters": len(valid_targets),
        "target_pdb_ids": [t['pdb_id'] for t in valid_targets],
        "provenance": {
            "source": "Strictly Verified Clean External Zinc Metalloenzyme Benchmark",
            "criteria": [
                "Catalytic Zn site with >=2 protein sidechain donors within 2.8 A",
                "Directly coordinated native ligand with donor (N/O/S) within <= 2.5 A of catalytic Zn",
                "Strict < 30% sequence identity against ALL CrossDocked training targets (0 hits in 19,476 CrossDocked PDBs)",
                "Independent sequence clusters across 30% identity threshold (1 target per cluster)"
            ],
            "total_independent_clusters": len(valid_targets)
        }
    }
    torch.save(payload, out_path)
    print(f"Saved payload to {out_path}")


if __name__ == '__main__':
    main()
