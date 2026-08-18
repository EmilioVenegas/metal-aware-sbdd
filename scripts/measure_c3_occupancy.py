import sys
import os
import json
import numpy as np
from pathlib import Path
from rdkit import Chem

def main():
    decoy_file = Path("data/c3_decoys.json")
    if not decoy_file.exists():
        print("data/c3_decoys.json not found. Run step 1 first.")
        sys.exit(1)
        
    with open(decoy_file, 'r') as f:
        data = json.load(f)
        
    out_dir = Path("results/step1/checker")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "c3_occupancy.jsonl"
    
    gen_dir = Path("results/step1/generation/sdf")
    
    skipped = []
    processed = 0
    
    with open(out_file, 'w') as out_f:
        for pdb_id, info in data.items():
            sdf_path = gen_dir / f"{pdb_id}.sdf"
            if not sdf_path.exists():
                skipped.append(pdb_id)
                continue
                
            zn_coord = np.array(info['zn'])
            decoys = [np.array(d['xyz']) for d in info['decoys']]
            n_decoys = len(decoys)
            
            suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
            
            for mol_idx, mol in enumerate(suppl):
                if mol is None:
                    continue
                    
                coords = []
                conf = mol.GetConformer()
                for i, atom in enumerate(mol.GetAtoms()):
                    if atom.GetSymbol() != 'H':
                        pos = conf.GetAtomPosition(i)
                        coords.append([pos.x, pos.y, pos.z])
                        
                if not coords:
                    continue
                coords = np.array(coords)
                
                # Check metal
                dists_to_zn = np.linalg.norm(coords - zn_coord, axis=1)
                metal_occ = bool(np.any(dists_to_zn <= 2.70))
                
                # Check decoys
                decoy_occ = []
                for decoy in decoys:
                    dists_to_decoy = np.linalg.norm(coords - decoy, axis=1)
                    decoy_occ.append(bool(np.any(dists_to_decoy <= 2.70)))
                    
                mean_decoy_occ = float(np.mean(decoy_occ)) if decoy_occ else None
                paired_diff = float(metal_occ) - mean_decoy_occ if mean_decoy_occ is not None else None

                record = {
                    "pdb_id": pdb_id,
                    "mol_index": mol_idx,
                    "metal_occupied": metal_occ,
                    "decoy_occupied": decoy_occ,
                    "mean_decoy_occupied": round(mean_decoy_occ, 5) if mean_decoy_occ is not None else None,
                    "paired_occupancy_diff": round(paired_diff, 5) if paired_diff is not None else None,
                    "n_decoys": n_decoys
                }
                out_f.write(json.dumps(record) + "\n")
                
            processed += 1
            
    print(f"Processed {processed} targets.")
    if skipped:
        print(f"Skipped {len(skipped)} targets (SDF not found): {', '.join(skipped)}")
        
if __name__ == '__main__':
    main()
