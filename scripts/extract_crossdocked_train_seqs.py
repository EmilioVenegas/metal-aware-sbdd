"""Extract all protein sequences from CrossDocked training targets."""

import io
import json
import pickle
import lmdb
import torch
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict

def main():
    db_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/crossdocked_pocket10_processed.lmdb'
    manifest_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/lmdb_index_manifest.json'
    split_path = '/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/crossdocked_split.pt'

    with open(manifest_path) as f:
        manifest = json.load(f)

    split = torch.load(split_path)
    train_indices = set(split['train'])

    train_targets = defaultdict(list)
    for idx in train_indices:
        name = manifest[idx]
        if name is None: continue
        t = name.split('/')[0]
        train_targets[t].append(idx)

    print(f"Found {len(train_targets)} distinct training target directories in CrossDocked.")

    # We will also load UniProt/PDB sequences for CrossDocked targets by query
    # or from representative receptor PDBs
    train_clusters = set()
    for t in train_targets:
        parts = t.split('_')
        if len(parts) >= 2:
            train_clusters.add(f"{parts[0]}_{parts[1]}")

    print(f"Total distinct protein clusters (Gene_Organism) in CrossDocked train: {len(train_clusters)}")

    with open('data/crossdocked_train_clusters.json', 'w') as f:
        json.dump({
            'train_targets': list(train_targets.keys()),
            'train_clusters': sorted(list(train_clusters))
        }, f, indent=2)

if __name__ == '__main__':
    main()
