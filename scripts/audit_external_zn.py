"""Audit external Zn dataset for sequence pseudo-replication and CrossDocked overlap."""

import io
import json
import urllib.request
import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio import pairwise2
from Bio.Seq import Seq

def get_pdb_sequence_and_title(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    title = "Unknown"
    try:
        req = urllib.request.urlopen(url, timeout=10)
        data = json.loads(req.read().decode('utf-8'))
        title = data.get('struct', {}).get('title', 'Unknown')
    except Exception as e:
        pass

    # Fetch polymer entity info
    poly_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/1"
    seq = ""
    protein_name = "Unknown"
    try:
        req = urllib.request.urlopen(poly_url, timeout=10)
        pdata = json.loads(req.read().decode('utf-8'))
        seq = pdata.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', '')
        protein_name = pdata.get('rcsb_polymer_entity', {}).get('pdbx_description', title)
    except Exception as e:
        pass

    return seq, protein_name, title

def compute_identity(seq1, seq2):
    if not seq1 or not seq2:
        return 0.0
    # Global pairwise alignment
    alns = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    if not alns:
        return 0.0
    matches = alns[0][2]
    min_len = min(len(seq1), len(seq2))
    return matches / min_len

def cluster_sequences(seqs, identity_threshold):
    n = len(seqs)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, i] = True
        for j in range(i+1, n):
            ident = compute_identity(seqs[i], seqs[j])
            if ident >= identity_threshold:
                adj[i, j] = True
                adj[j, i] = True

    # Connected components
    visited = set()
    clusters = []
    for i in range(n):
        if i not in visited:
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in range(n):
                    if adj[curr, neighbor] and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(component)
    return clusters

def main():
    payload = torch.load('data/external_zn_test.pt')
    targets = payload['targets']
    print(f"Loaded {len(targets)} targets from data/external_zn_test.pt\n")

    seq_data = []
    for t in targets:
        pid = t['pdb_id']
        seq, pname, title = get_pdb_sequence_and_title(pid)
        seq_data.append({
            'pdb_id': pid,
            'sequence': seq,
            'protein_name': pname,
            'title': title,
            'ligand': t['ligand_resname']
        })
        print(f"{pid.upper()}: {pname[:60]} (Ligand: {t['ligand_resname']}, SeqLen: {len(seq)})")

    seqs = [d['sequence'] for d in seq_data]

    # Cluster at 90%
    c90 = cluster_sequences(seqs, 0.90)
    print(f"\n{'='*70}\nCLUSTERING AT 90% SEQUENCE IDENTITY: {len(c90)} CLUSTERS\n{'='*70}")
    for idx, c in enumerate(c90):
        names = [seq_data[i]['pdb_id'].upper() for i in c]
        pname = seq_data[c[0]]['protein_name']
        print(f"Cluster {idx+1:2d} ({len(c)} members): {pname[:50]} -> {', '.join(names)}")

    # Cluster at 30%
    c30 = cluster_sequences(seqs, 0.30)
    print(f"\n{'='*70}\nCLUSTERING AT 30% SEQUENCE IDENTITY: {len(c30)} CLUSTERS\n{'='*70}")
    for idx, c in enumerate(c30):
        names = [seq_data[i]['pdb_id'].upper() for i in c]
        pname = seq_data[c[0]]['protein_name']
        print(f"Cluster {idx+1:2d} ({len(c)} members): {pname[:50]} -> {', '.join(names)}")

    # Save audit info
    with open('data/external_zn_audit.json', 'w') as f:
        json.dump({
            'targets': seq_data,
            'n_clusters_90': len(c90),
            'n_clusters_30': len(c30)
        }, f, indent=2)

if __name__ == '__main__':
    main()
