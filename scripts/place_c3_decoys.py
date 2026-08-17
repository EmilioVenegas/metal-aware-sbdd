import sys
import os
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial.distance import cdist

def main():
    print("Loading targets...")
    data = torch.load('data/external_zn_test_clean.pt')
    targets = data['targets']
    
    out_dict = {}
    
    for i, target in enumerate(targets):
        pdb_id = target['pdb_id']
        zn_coord = np.array(target['zn_coord'])
        
        print(f"[{i+1}/{len(targets)}] Processing {pdb_id}...")
        
        sdf_path = Path(f"data/native_ligands/{pdb_id.upper()}.sdf")
        if not sdf_path.exists():
            print(f"  Missing SDF for {pdb_id}")
            continue
            
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        mol = suppl[0]
        lig_coords = []
        conf = mol.GetConformer()
        for i_atom, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                pos = conf.GetAtomPosition(i_atom)
                lig_coords.append([pos.x, pos.y, pos.z])
        lig_coords = np.array(lig_coords)
        
        cif_path = Path(f"data/external_pdbs/{pdb_id.upper()}.cif")
        pdb_path = Path(f"data/external_pdbs/{pdb_id.upper()}.pdb")
        
        if cif_path.exists():
            parser = MMCIFParser(QUIET=True)
            struct = parser.get_structure(pdb_id, str(cif_path))
        elif pdb_path.exists():
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(pdb_id, str(pdb_path))
        else:
            print(f"  Missing PDB/CIF for {pdb_id}")
            continue
            
        all_coords = []
        pocket_coords = []
        
        for model in struct:
            for chain in model:
                for res in chain:
                    rname = res.get_resname().strip().upper()
                    het = res.id[0]
                    if het == ' ' and is_aa(rname, standard=True):
                        res_coords = []
                        for atom in res:
                            if atom.element != 'H':
                                res_coords.append(atom.coord.copy())
                        if not res_coords:
                            continue
                        res_coords = np.array(res_coords)
                        all_coords.append(res_coords)
                        
                        dists = cdist(res_coords, lig_coords)
                        if np.min(dists) <= 8.0:
                            pocket_coords.append(res_coords)
                            
        if not all_coords:
            print(f"  No valid protein heavy atoms found for {pdb_id}")
            continue
            
        all_coords = np.vstack(all_coords)
        
        if not pocket_coords:
            print(f"  No pocket residues found for {pdb_id}")
            continue
            
        pocket_coords = np.vstack(pocket_coords)
        
        pocket_centroid = np.mean(pocket_coords, axis=0)
        zn_burial = int(np.sum(np.linalg.norm(all_coords - zn_coord, axis=1) <= 8.0))
        zn_dist_to_centroid = float(np.linalg.norm(zn_coord - pocket_centroid))
        
        # Search deterministic
        seed = int(hashlib.sha256(pdb_id.encode()).hexdigest()[:8], 16) % (2**31)
        np.random.seed(seed)
        
        min_b = np.min(pocket_coords, axis=0)
        max_b = np.max(pocket_coords, axis=0)
        
        decoys = []
        batch_size = 100000
        max_tries = 20
        
        for _ in range(max_tries):
            if len(decoys) == 5:
                break
            points = np.random.uniform(min_b, max_b, size=(batch_size, 3))
            
            # Constraint b
            dists_to_centroid = np.linalg.norm(points - pocket_centroid, axis=1)
            mask_b = np.abs(dists_to_centroid - zn_dist_to_centroid) <= 0.5
            
            # Constraint d
            dists_to_zn = np.linalg.norm(points - zn_coord, axis=1)
            mask_d1 = dists_to_zn >= 4.0
            
            mask_b_d1 = mask_b & mask_d1
            if not np.any(mask_b_d1):
                continue
                
            valid_points = points[mask_b_d1]
            valid_idx = np.where(mask_b_d1)[0]
            
            # Constraint c
            dists_to_prot = cdist(valid_points, all_coords)
            mask_c = np.min(dists_to_prot, axis=1) >= 2.0
            
            if not np.any(mask_c):
                continue
                
            valid_points = valid_points[mask_c]
            dists_to_prot = dists_to_prot[mask_c]
            valid_idx = valid_idx[mask_c]
            
            dist_cent = dists_to_centroid[valid_idx]
            dist_zn = dists_to_zn[valid_idx]
            
            # Constraint a
            burials = np.sum(dists_to_prot <= 8.0, axis=1)
            mask_a = np.abs(burials - zn_burial) <= 0.15 * zn_burial
            
            if not np.any(mask_a):
                continue
                
            valid_points = valid_points[mask_a]
            burials = burials[mask_a]
            dist_cent = dist_cent[mask_a]
            dist_zn = dist_zn[mask_a]
            
            # Additional constraint d (dist to other decoys)
            for idx in range(len(valid_points)):
                if len(decoys) == 5:
                    break
                pt = valid_points[idx]
                
                valid_decoy_dist = True
                for d in decoys:
                    if np.linalg.norm(pt - np.array(d['xyz'])) < 2.0:
                        valid_decoy_dist = False
                        break
                        
                if not valid_decoy_dist:
                    continue
                    
                decoys.append({
                    "xyz": pt.tolist(),
                    "burial": int(burials[idx]),
                    "dist_to_centroid": float(dist_cent[idx]),
                    "dist_to_zn": float(dist_zn[idx])
                })
                
        out_dict[pdb_id] = {
            "zn": zn_coord.tolist(),
            "burial_zn": zn_burial,
            "decoys": decoys,
            "n_requested": 5,
            "n_placed": len(decoys)
        }
        
        if len(decoys) < 5:
            print(f"  SHORTFALL: placed {len(decoys)}/5 decoys.")
            
    print("Done generating decoys. Writing to output...")
    with open('data/c3_decoys.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
    print("Saved to data/c3_decoys.json")

if __name__ == '__main__':
    main()
