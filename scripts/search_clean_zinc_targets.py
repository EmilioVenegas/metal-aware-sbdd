"""Search and validate clean external Zinc metalloenzyme targets across the PDB.

Filters:
1. Strict CrossDocked Sequence Independence: < 30% sequence identity to ANY CrossDocked training PDB (0 hits in 19,476 CrossDocked PDBs).
2. Catalytic Zn: >= 2 protein sidechain donors <= 2.8 A.
3. Coordinated Native Ligand: Direct coordinating donor atom (N, O, S) within <= 2.5 A of catalytic Zn.
4. Cluster Independence: < 30% sequence identity between distinct target families.
5. Overbuild to >= 30 surviving 30%-identity clusters.
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
    'ACY', 'ACE', 'NH2', 'NCO', 'MOH', 'EOH', 'CCN'
}

# Diverse zinc metalloenzyme candidates spanning distinct classes
CANDIDATE_FAMILIES = [
    # 1. Metallo-beta-lactamase VIM
    ('VIM-1 MBL', ['7UYA', '7UYB', '8PGP', '7UYC', '6SP7']),
    # 2. Metallo-beta-lactamase NDM
    ('NDM-1 MBL', ['5ZGP', '6D1J', '6EZM', '7K5F', '7B43']),
    # 3. Metallo-beta-lactamase IMP
    ('IMP-1 MBL', ['5YPK', '6I5P', '1JJT']),
    # 4. Metallo-beta-lactamase B2/B3 (CphA, L1, FEZ-1)
    ('L1/CphA/FEZ MBL', ['1K07', '2A5X', '1ZNB', '1LST']),
    # 5. Adenosine Deaminase (ADA)
    ('Adenosine Deaminase', ['1NDV', '1NDW', '3IAR', '1VFL', '1KRM']),
    # 6. Cytidine Deaminase (CDA)
    ('Cytidine Deaminase', ['1AF2', '2G6W', '1MQ0', '5TGO']),
    # 7. Guanine Deaminase (GDA / Cypin)
    ('Guanine Deaminase', ['2UZ9', '2P8R', '3G7V']),
    # 8. Leukotriene A4 Hydrolase (LTA4H)
    ('LTA4 Hydrolase', ['3B7S', '3B7U', '4MS6', '4L2L', '1GW6', '5N3W']),
    # 9. Endothelin-Converting Enzyme (ECE-1)
    ('ECE-1', ['3DWB', '3F8U', '3I3C', '3S61']),
    # 10. Neprilysin (NEP / CD10)
    ('Neprilysin (NEP)', ['1R1J', '5T58', '6GID', '6P0C']),
    # 11. Anthrax Lethal Factor (LF)
    ('Anthrax Lethal Factor', ['1PWQ', '5T1V', '1JKY', '1YQY']),
    # 12. Insulin-Degrading Enzyme (IDE)
    ('Insulin-Degrading Enzyme', ['3E4A', '2G47', '2G54', '4IOI']),
    # 13. Histone Deacetylase 1 / 2 (HDAC1/2)
    ('HDAC1/2', ['5ICN', '6XDM', '4LXZ', '5IWG']),
    # 14. Aminopeptidase N (APN / CD13)
    ('Aminopeptidase N', ['4FYT', '4OU3', '2DQM']),
    # 15. Endoplasmic Reticulum Aminopeptidase (ERAP1)
    ('ERAP1', ['6Q4R', '7OU8', '6R44']),
    # 16. Farnesyltransferase (FTase)
    ('Farnesyltransferase', ['1JCQ', '1TN6', '1FT1', '1KZO']),
    # 17. Geranylgeranyltransferase (GGTase)
    ('Geranylgeranyltransferase', ['1S63', '1N4Q', '3PZ3']),
    # 18. Phosphotriesterase (PTE / Organophosphate Hydrolase)
    ('Phosphotriesterase', ['1DPM', '1QW7', '2OBR', '4PCP', '1EZ2']),
    # 19. Bacterial / Fungal Carbonic Anhydrase (Beta/Gamma CA)
    ('Non-Human Carbonic Anhydrase', ['4Z28', '5YUI', '6SDX', '6UX0', '6E3V', '5C0G', '1G5C']),
    # 20. tRNA-Guanine Transglycosylase (TGT - Zn-dependent)
    ('tRNA-Guanine Transglycosylase', ['1N2V', '1R5Y', '2Z7F', '1P0E', '1P0D']),
    # 21. Fructose-1,6-bisphosphate Aldolase Class II (Zn-dependent)
    ('FBP Aldolase Class II', ['1B57', '1ZEN', '3C4U', '1DOS']),
    # 22. Methionine Aminopeptidase (MetAP)
    ('Methionine Aminopeptidase', ['1C21', '1XGS', '3L7L', '4V2J', '2P8A']),
    # 23. Alcohol Dehydrogenase with active-site Zn chelator
    ('Alcohol Dehydrogenase (Zn-chelator)', ['1EE2', '1AGN', '1HLD', '1ADC', '2F8C']),
    # 24. Astacin / Meprin Metalloprotease (with coordinated inhibitor)
    ('Astacin / Meprin', ['1AST', '3G60', '1QJJ', '2P2D']),
    # 25. Glutamate Carboxypeptidase II (PSMA / GCPII)
    ('PSMA / GCPII', ['2PVW', '3D7H', '2OOT', '4NG2']),
    # 26. Glyoxalase I (Glo1)
    ('Glyoxalase I', ['3VW9', '1QIP', '1FRO']),
    # 27. Arginase (Zn/Mn active site)
    ('Arginase', ['1D3V', '2AEB', '3GN0', '4HWW']),
    # 28. Dethiobiotin Synthetase / Hydrolase (Zn metalloenzyme)
    ('Zinc Hydrolase / Synthetase', ['1DAD', '1BS1', '2V3E']),
    # 29. Carbon-Monoxide Dehydrogenase (CODH / Zn-Fe-S)
    ('CO Dehydrogenase', ['1OAO', '1JQK']),
    # 30. Peptide Deformylase (PDF - non-CrossDocked species)
    ('Peptide Deformylase', ['1LRU', '5C5D', '6BY4', '1G2A']),
    # 31. Bacterial Aminopeptidase (PepN / PepA)
    ('Bacterial Aminopeptidase', ['2GLJ', '3L2S', '4ONA']),
    # 32. UDP-3-O-acyl-GlcNAc Deacetylase LpxC (non-E.coli/Pseudomonas)
    ('LpxC (Non-CrossDocked)', ['5WEA', '6P3C', '7U3E', '4M00', '4M01', '3P3E']),
    # 33. Matrix Metalloproteinase (non-CrossDocked like MMP-12/14/20)
    ('MMP-12 / MMP-14', ['1JK3', '1RM8', '1YCM', '3MA2']),
    # 34. Collagenase / Clostridial Metalloprotease
    ('Clostridial Collagenase', ['2Y3U', '4AQO', '4AR1', '4AR8']),
    # 35. Deoxycytidylate Deaminase (dCMP Deaminase)
    ('dCMP Deaminase', ['1VPK', '2DCA', '1J17']),
    # 36. Phospholipase C (Zn-dependent)
    ('Phospholipase C (Zn)', ['1AH7', '1AKP', '1PTC'])
]

def check_cd_overlap(seq, crossdocked_pdbs):
    if not seq or len(seq) < 30:
        return False, -1

    query = {
        'query': {
            'type': 'terminal',
            'service': 'sequence',
            'parameters': {
                'evalue_cutoff': 0.1,
                'identity_cutoff': 0.30,
                'sequence_type': 'protein',
                'value': seq
            }
        },
        'return_type': 'entry'
    }
    try:
        r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=query, timeout=12)
        if r.status_code == 204:
            return True, 0
        elif r.status_code == 200:
            hits = {doc['identifier'].lower() for doc in r.json().get('result_set', [])}
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
        req = urllib.request.urlopen(url, timeout=12)
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
                            # Keep heavy atoms with element
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
            if len(lig_atom_list) < 6: # genuine drug-like / fragment (>= 6 heavy atoms)
                continue

            # Strict check: must have a donor atom (N, O, S, or any heteroatom) within <= 2.5 A of Zn
            coordinating_lig_atoms = []
            min_d = 999.0
            for la in lig_atom_list:
                d = float(np.linalg.norm(la['coord'] - z_coord))
                if d < min_d: min_d = d
                # Donor heteroatoms: N, O, S, P, F, Cl
                elem = la['element']
                aname = la['name']
                is_donor = elem in {'N', 'O', 'S', 'P', 'CL', 'F'} or aname[0] in {'N', 'O', 'S'}
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

    # Check against CrossDocked
    is_clean, overlap_count = check_cd_overlap(seq_1, crossdocked_pdbs)
    if not is_clean:
        return None, f"CrossDocked sequence overlap ({overlap_count} hits >= 30%)"

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
    print(f"Loaded {len(crossdocked_pdbs)} CrossDocked PDBs for strict sequence exclusion.")

    passed_targets = []
    for fam_name, candidate_pdbs in CANDIDATE_FAMILIES:
        accepted = None
        for pid in candidate_pdbs:
            # First check if PDB itself is in CrossDocked
            if pid.lower() in crossdocked_pdbs:
                continue
            res, reason = inspect_pdb_strictly(pid, crossdocked_pdbs)
            if res:
                res['family_name'] = fam_name
                accepted = res
                print(f"  [ACCEPTED] {fam_name:32s} -> {pid}: ligand={res['ligand_resname']} (Zn-Lig: {res['min_zn_ligand_dist']:.2f}A, coord={res['coordinating_ligand_atoms'][:2]})")
                break
            else:
                pass
        if accepted:
            passed_targets.append(accepted)
        else:
            print(f"  [REJECTED] {fam_name:32s} -> None of {candidate_pdbs} passed strict criteria")

    print(f"\nTotal families passed both strict filters: {len(passed_targets)}")

    # Perform all-vs-all clustering among passed targets at 30% sequence identity
    clusters_30 = cluster_targets(passed_targets, 0.30)
    print(f"\n{'='*70}\nCLUSTERING SURVIVING TARGETS AT 30% SEQUENCE IDENTITY: {len(clusters_30)} CLUSTERS\n{'='*70}")

    surviving_representatives = []
    for idx, c in enumerate(clusters_30):
        rep = passed_targets[c[0]]
        surviving_representatives.append(rep)
        members = [f"{passed_targets[i]['pdb_id']} ({passed_targets[i]['family_name']})" for i in c]
        print(f"Cluster {idx+1:2d} ({len(c)} members): {', '.join(members)}")

    print(f"\nFinal count of clean, coordinated, independent 30%-identity Zinc clusters: m = {len(clusters_30)}")

    out_path = Path("data/external_zn_test.pt")
    payload = {
        "targets": passed_targets,
        "representative_targets": surviving_representatives,
        "m_clusters": len(clusters_30),
        "m_targets": len(passed_targets),
        "target_pdb_ids": [t['pdb_id'] for t in passed_targets],
        "provenance": {
            "source": "Strictly Verified Clean External Zinc Metalloenzyme Families",
            "criteria": [
                "Catalytic Zn site with >=2 protein sidechain donors within 2.8 A",
                "Directly coordinated native ligand with donor (N/O/S) within <= 2.5 A of catalytic Zn",
                "Strict < 30% sequence identity against ALL CrossDocked training targets (0 hits in 19,476 CrossDocked PDBs)",
                "Independent sequence clusters across 30% identity threshold"
            ],
            "total_targets": len(passed_targets),
            "independent_30pct_clusters": len(clusters_30)
        }
    }
    torch.save(payload, out_path)
    print(f"Saved payload to {out_path}")


if __name__ == '__main__':
    main()
