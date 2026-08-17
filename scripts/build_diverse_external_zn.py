"""Build and audit a diverse external Zinc test set spanning distinct metalloenzyme families.

Target: >= 25 distinct protein clusters at 30% sequence identity.
"""

import io
import json
import os
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    'UNX', '1PE', '2PE', 'P6G', 'BTB', 'PGE', 'DIO', 'NAG', 'FUC', 'BMA'
}

# 28 distinct candidate families
FAMILY_CANDIDATES = [
    ('VIM-1 MBL', ['7UYA']),
    ('NDM-1 MBL', ['5ZGP']),
    ('IMP-1 MBL', ['5YPK']),
    ('HDAC1/2', ['5ICN']),
    ('HDAC8', ['1T64']),
    ('HDAH (Bacterial)', ['1W22']),
    ('MMP-2', ['1HOV']),
    ('MMP-8', ['1BZS']),
    ('Anthrax Lethal Factor', ['1PWQ']),
    ('Thermolysin', ['4TMN']),
    ('Neprilysin (NEP)', ['1R1J']),
    ('Peptide Deformylase', ['1LRU']),
    ('Glyoxalase I', ['3VW9']),
    ('Aminopeptidase N', ['4FYT']),
    ('ERAP1', ['6Q4R']),
    ('Alcohol Dehydrogenase', ['1D1S']),
    ('Farnesyltransferase', ['1JCQ']),
    ('Geranylgeranyltransferase', ['1S63']),
    ('Astacin / Meprin', ['4G1P']),
    ('Nuclease S1/P1', ['7QTA']),
    ('PSMA / GCPII', ['2PVW']),
    ('Carbonic Anhydrase', ['6G3V']),
    ('TACE / ADAM17', ['2I47']),
    ('ACE (Angiotensin Converting Enzyme)', ['1O86']),
    ('Carboxypeptidase A (CPA)', ['1CBX']),
    ('LpxC Deacetylase', ['3P3G']),
    ('Botulinum Neurotoxin', ['2IMA']),
    ('Insulin-Degrading Enzyme (IDE)', ['3E4A']),
]

def fetch_cif_and_inspect(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        req = urllib.request.urlopen(url, timeout=12)
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
                    'pdb_id': pdb_id.upper(),
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


def cluster_targets(targets, threshold):
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
    print("Selecting 1 representative per zinc metalloenzyme family...")
    selected_targets = []

    for fam_name, candidate_pdbs in FAMILY_CANDIDATES:
        accepted = None
        for pid in candidate_pdbs:
            res = fetch_cif_and_inspect(pid)
            if res:
                res['family_name'] = fam_name
                accepted = res
                print(f"  [ACCEPTED] {fam_name:35s} -> {pid}: ligand={res['ligand_resname']} (Zn-Lig: {res['min_zn_ligand_dist']:.2f}A, {res['num_sidechain_donors']} SC donors)")
                break
        if accepted:
            selected_targets.append(accepted)
        else:
            print(f"  [FAILED]   {fam_name:35s} -> None of {candidate_pdbs} met criteria")

    print(f"\nTotal families successfully selected: {len(selected_targets)}")

    # 1. Cluster at 90%
    c90 = cluster_targets(selected_targets, 0.90)
    print(f"\n{'='*70}\nCLUSTERING AT 90% SEQUENCE IDENTITY: {len(c90)} CLUSTERS\n{'='*70}")
    for idx, c in enumerate(c90):
        members = [f"{selected_targets[i]['pdb_id']} ({selected_targets[i]['family_name']})" for i in c]
        print(f"Cluster {idx+1:2d} ({len(c)} members): {', '.join(members)}")

    # 2. Cluster at 30%
    c30 = cluster_targets(selected_targets, 0.30)
    print(f"\n{'='*70}\nCLUSTERING AT 30% SEQUENCE IDENTITY: {len(c30)} CLUSTERS\n{'='*70}")
    for idx, c in enumerate(c30):
        members = [f"{selected_targets[i]['pdb_id']} ({selected_targets[i]['family_name']})" for i in c]
        print(f"Cluster {idx+1:2d} ({len(c)} members): {', '.join(members)}")

    # Save to data/external_zn_test.pt
    out_path = Path("data/external_zn_test.pt")
    payload = {
        "targets": selected_targets,
        "m_targets": len(selected_targets),
        "n_clusters_90": len(c90),
        "n_clusters_30": len(c30),
        "target_pdb_ids": [t['pdb_id'] for t in selected_targets],
        "provenance": {
            "source": "Curated Zinc Metalloenzyme Families (1 structure per family/target)",
            "criteria": [
                "Catalytic Zn site with >=2 protein sidechain donors within 2.8 A",
                "Bound drug-like native ligand within 4.5 A of catalytic Zn (for C1 control)",
                "Distinct protein families across 30% sequence identity threshold"
            ],
            "total_targets": len(selected_targets),
            "clusters_30pct": len(c30),
            "clusters_90pct": len(c90)
        }
    }
    torch.save(payload, out_path)
    print(f"\nSaved curated payload to {out_path}")


if __name__ == '__main__':
    main()
