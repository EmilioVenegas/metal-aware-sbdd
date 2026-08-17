"""Build a metal-enriched, target-disjoint split of CrossDocked.

Adaptation of build_holdout_split.py for metal-aware generative SBDD.

Requirements:
1. Zero sequence-identity / protein overlap between train, val, and test.
   Targets are clustered by protein identifier (gene + organism prefix).
   All targets/complexes in a cluster are assigned to exactly one partition.
2. Maximise metalloprotein test target count while maintaining large training data.
3. Maintain metal-type diversity across train, val, and test (stratified by metal element:
   ZN, MG, CA, MN, FE, NI, CO, CU, and NO_METAL).
4. Deterministic partition without unseeded RNG.
5. Emits index lists and per-target capped expanded lists.
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

import torch


def get_cluster_id(target_name):
    """Cluster targets by protein identity (gene_organism prefix)."""
    parts = str(target_name).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(target_name)


def load_pdb_metals(cache_path, metals_map_path):
    """Load metal annotations for PDB IDs."""
    with open(metals_map_path) as f:
        pdb_metals = json.load(f)
    return pdb_metals


def main():
    parser = argparse.ArgumentParser(description="Build metal-enriched target-disjoint split")
    parser.add_argument("--manifest", default="data/lmdb_index_manifest.json",
                        help="Path to manifest JSON (or ATOMICA manifest)")
    parser.add_argument("--metals_map", default="data/pdb_metals_map.json",
                        help="Path to PDB metals mapping JSON")
    parser.add_argument("--output", default="data/metal_target_split.pt",
                        help="Output path for split dict")
    parser.add_argument("--max_per_target", type=int, default=30,
                        help="Cap on complexes per target in expanded lists")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    args = parser.parse_args()

    # Fallback paths if running from repo root
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        alt = Path("/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/lmdb_index_manifest.json")
        if alt.exists():
            manifest_path = alt

    metals_map_path = Path(args.metals_map)
    if not metals_map_path.exists():
        raise FileNotFoundError(f"Metals map not found: {metals_map_path}")

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loading metals map from {metals_map_path}...")
    with open(metals_map_path) as f:
        pdb_metals = json.load(f)

    # 1. Map each manifest entry to target, cluster, and metal
    target_to_indices = defaultdict(list)
    target_metal_counts = defaultdict(Counter)

    for idx, name in enumerate(manifest):
        if name is None:
            continue
        target = name.split("/")[0]
        target_to_indices[target].append(idx)

        parts = name.split("/")[-1].split("_")
        if len(parts) >= 2:
            pdb = parts[0].lower()
            if pdb in pdb_metals:
                for m in pdb_metals[pdb]:
                    target_metal_counts[target][m] += 1

    # 2. Cluster targets by protein ID (gene + organism)
    cluster_targets = defaultdict(list)
    cluster_complexes = defaultdict(int)
    cluster_metal_counts = defaultdict(Counter)

    for target, indices in target_to_indices.items():
        cid = get_cluster_id(target)
        cluster_targets[cid].append(target)
        cluster_complexes[cid] += len(indices)
        for m, count in target_metal_counts[target].items():
            cluster_metal_counts[cid][m] += count

    # Determine primary metal for each cluster
    cluster_primary_metal = {}
    for cid in cluster_targets:
        mcounts = cluster_metal_counts[cid]
        if mcounts:
            cluster_primary_metal[cid] = mcounts.most_common(1)[0][0]
        else:
            cluster_primary_metal[cid] = "NO_METAL"

    # Group clusters by primary metal
    metal_to_clusters = defaultdict(list)
    for cid, pmetal in cluster_primary_metal.items():
        metal_to_clusters[pmetal].append(cid)

    # 3. Stratified Deterministic Partition
    # For each metal group, sort clusters deterministically by:
    # (-num_complexes, cluster_id)
    # Then assign in 10-slot pattern: [train x 7, val x 1, test x 2] -> 70% / 10% / 20%
    train_clusters, val_clusters, test_clusters = [], [], []

    # Standard 10-slot pattern: 0..6 -> train, 7 -> val, 8..9 -> test
    slot_map = {
        0: 'train', 1: 'train', 2: 'train', 3: 'train', 4: 'train', 5: 'train', 6: 'train',
        7: 'val',
        8: 'test', 9: 'test'
    }

    # For small categories (<5 clusters), ensure at least 1 in test and 1 in train if >=2
    for metal, cids in sorted(metal_to_clusters.items()):
        ordered_cids = sorted(cids, key=lambda c: (-cluster_complexes[c], c))
        n = len(ordered_cids)

        if n == 1:
            # Single cluster: assign to train
            train_clusters.append(ordered_cids[0])
        elif n == 2:
            train_clusters.append(ordered_cids[0])
            test_clusters.append(ordered_cids[1])
        elif n == 3:
            train_clusters.append(ordered_cids[0])
            train_clusters.append(ordered_cids[1])
            test_clusters.append(ordered_cids[2])
        elif n == 4:
            train_clusters.append(ordered_cids[0])
            train_clusters.append(ordered_cids[1])
            val_clusters.append(ordered_cids[2])
            test_clusters.append(ordered_cids[3])
        elif n == 5:
            train_clusters.append(ordered_cids[0])
            train_clusters.append(ordered_cids[1])
            train_clusters.append(ordered_cids[2])
            val_clusters.append(ordered_cids[3])
            test_clusters.append(ordered_cids[4])
        else:
            # Use deterministic slot mapping
            for idx, cid in enumerate(ordered_cids):
                slot = idx % 10
                dest = slot_map[slot]
                if dest == 'train':
                    train_clusters.append(cid)
                elif dest == 'val':
                    val_clusters.append(cid)
                else:
                    test_clusters.append(cid)

    # 4. Map clusters back to targets and indices
    def gather_split(clusters):
        targets = []
        for c in sorted(clusters):
            targets.extend(cluster_targets[c])
        targets = sorted(targets)
        
        indices = []
        for t in targets:
            indices.extend(target_to_indices[t])
        indices = sorted(indices)

        # Expanded capped indices
        indices_expanded = []
        for t in targets:
            t_inds = sorted(target_to_indices[t])
            indices_expanded.extend(t_inds[:args.max_per_target])
        indices_expanded = sorted(indices_expanded)

        return targets, indices, indices_expanded

    train_targets, train_indices, train_expanded = gather_split(train_clusters)
    val_targets, val_indices, val_expanded = gather_split(val_clusters)
    test_targets, test_indices, test_expanded = gather_split(test_clusters)

    # 5. Assertions for integrity
    train_c_set, val_c_set, test_c_set = set(train_clusters), set(val_clusters), set(test_clusters)
    train_t_set, val_t_set, test_t_set = set(train_targets), set(val_targets), set(test_targets)
    train_i_set, val_i_set, test_i_set = set(train_indices), set(val_indices), set(test_indices)

    assert not (train_c_set & val_c_set), "Train and Val clusters overlap!"
    assert not (train_c_set & test_c_set), "Train and Test clusters overlap!"
    assert not (val_c_set & test_c_set), "Val and Test clusters overlap!"

    assert not (train_t_set & val_t_set), "Train and Val targets overlap!"
    assert not (train_t_set & test_t_set), "Train and Test targets overlap!"
    assert not (val_t_set & test_t_set), "Val and Test targets overlap!"

    assert not (train_i_set & val_i_set), "Train and Val indices overlap!"
    assert not (train_i_set & test_i_set), "Train and Test indices overlap!"
    assert not (val_i_set & test_i_set), "Val and Test indices overlap!"

    total_complexes = len(train_indices) + len(val_indices) + len(test_indices)
    assert total_complexes == sum(len(v) for v in target_to_indices.values()), "Total complex count mismatch!"

    # 6. Compute metal breakdown statistics
    def get_metal_breakdown(targets, indices):
        target_counts = Counter()
        complex_counts = Counter()
        for t in targets:
            m = target_metal_counts[t].most_common(1)
            primary_m = m[0][0] if m else "NO_METAL"
            target_counts[primary_m] += 1
        for idx in indices:
            name = manifest[idx]
            parts = name.split("/")[-1].split("_")
            if len(parts) >= 2:
                pdb = parts[0].lower()
                if pdb in pdb_metals:
                    for m in pdb_metals[pdb]:
                        complex_counts[m] += 1
                else:
                    complex_counts["NO_METAL"] += 1
        return target_counts, complex_counts

    train_m_t, train_m_c = get_metal_breakdown(train_targets, train_indices)
    val_m_t, val_m_c = get_metal_breakdown(val_targets, val_indices)
    test_m_t, test_m_c = get_metal_breakdown(test_targets, test_indices)

    # 7. Save Payload
    payload = {
        "train_clusters": sorted(train_clusters),
        "val_clusters": sorted(val_clusters),
        "test_clusters": sorted(test_clusters),
        "train_targets": train_targets,
        "val_targets": val_targets,
        "test_targets": test_targets,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_indices_expanded": train_expanded,
        "val_indices_expanded": val_expanded,
        "test_indices_expanded": test_expanded,
        "train_filenames": [manifest[i] for i in train_indices],
        "val_filenames": [manifest[i] for i in val_indices],
        "test_filenames": [manifest[i] for i in test_indices],
        "target_primary_metal": {t: (target_metal_counts[t].most_common(1)[0][0] if target_metal_counts[t] else "NO_METAL") for t in target_to_indices},
        "provenance": {
            "manifest": str(manifest_path),
            "metals_map": str(metals_map_path),
            "strategy": "Stratified deterministic partition by primary metal element, clustered by gene_organism prefix (zero sequence/protein overlap)",
            "max_per_target": args.max_per_target,
            "counts": {
                "train_targets": len(train_targets),
                "val_targets": len(val_targets),
                "test_targets": len(test_targets),
                "train_complexes": len(train_indices),
                "val_complexes": len(val_indices),
                "test_complexes": len(test_indices),
            }
        }
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nWrote split to {out_path}")

    # 8. Print Summary Tables
    all_metals = ["ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU", "NO_METAL"]
    print("\n" + "=" * 80)
    print("TARGET COUNTS BY PRIMARY METAL:")
    print(f"{'Metal':10s} | {'Train':>8s} | {'Val':>8s} | {'Test':>8s} | {'Total':>8s} | {'Test %':>8s}")
    print("-" * 80)
    total_tr_t = total_va_t = total_te_t = 0
    metallo_tr_t = metallo_va_t = metallo_te_t = 0
    for m in all_metals:
        tr = train_m_t[m]
        va = val_m_t[m]
        te = test_m_t[m]
        tot = tr + va + te
        pct = (te / tot * 100) if tot > 0 else 0
        print(f"{m:10s} | {tr:8d} | {va:8d} | {te:8d} | {tot:8d} | {pct:7.1f}%")
        total_tr_t += tr
        total_va_t += va
        total_te_t += te
        if m != "NO_METAL":
            metallo_tr_t += tr
            metallo_va_t += va
            metallo_te_t += te

    print("-" * 80)
    print(f"{'METALLO':10s} | {metallo_tr_t:8d} | {metallo_va_t:8d} | {metallo_te_t:8d} | {metallo_tr_t+metallo_va_t+metallo_te_t:8d} | {metallo_te_t/(metallo_tr_t+metallo_va_t+metallo_te_t)*100:7.1f}%")
    print(f"{'TOTAL':10s} | {total_tr_t:8d} | {total_va_t:8d} | {total_te_t:8d} | {total_tr_t+total_va_t+total_te_t:8d} | {total_te_t/(total_tr_t+total_va_t+total_te_t)*100:7.1f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("COMPLEX COUNTS BY METAL:")
    print(f"{'Metal':10s} | {'Train':>10s} | {'Val':>10s} | {'Test':>10s} | {'Total':>10s}")
    print("-" * 80)
    for m in ["ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU"]:
        tr = train_m_c[m]
        va = val_m_c[m]
        te = test_m_c[m]
        tot = tr + va + te
        print(f"{m:10s} | {tr:10d} | {va:10d} | {te:10d} | {tot:10d}")
    print("-" * 80)
    print(f"{'ALL COMPLEX':10s} | {len(train_indices):10d} | {len(val_indices):10d} | {len(test_indices):10d} | {total_complexes:10d}")
    print(f"{'EXPANDED':10s} | {len(train_expanded):10d} | {len(val_expanded):10d} | {len(test_expanded):10d} | {len(train_expanded)+len(val_expanded)+len(test_expanded):10d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
