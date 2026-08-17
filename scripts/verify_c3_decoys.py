import json
import numpy as np

def verify():
    with open('data/c3_decoys.json', 'r') as f:
        data = json.load(f)
        
    print(f"Total targets: {len(data)}")
    
    shortfalls = []
    dist_to_zn_all = []
    
    print("\n--- Per-target verification ---")
    print(f"{'PDB':<6} | {'Zn Burial':<10} | {'Mean Decoy':<12} | {'Max Dev':<8} | {'N Placed':<8} | {'Status'}")
    print("-" * 65)
    
    all_meet_rule = True
    
    for pdb, info in data.items():
        zn_b = info['burial_zn']
        decoys = info['decoys']
        n_placed = info['n_placed']
        
        if n_placed < 5:
            shortfalls.append(pdb)
            
        if decoys:
            decoy_burials = [d['burial'] for d in decoys]
            mean_b = np.mean(decoy_burials)
            max_dev = max([abs(b - zn_b) for b in decoy_burials])
            
            # Rule check
            limit = 0.15 * zn_b
            if max_dev > limit:
                all_meet_rule = False
                status = "FAIL"
            else:
                status = "PASS"
                
            dist_to_zn_all.extend([d['dist_to_zn'] for d in decoys])
            
            print(f"{pdb:<6} | {zn_b:<10} | {mean_b:<12.1f} | {max_dev:<8} | {n_placed:<8} | {status}")
        else:
            print(f"{pdb:<6} | {zn_b:<10} | {'-':<12} | {'-':<8} | {n_placed:<8} | {'-'}")
            
    print("\n--- Shortfall Report ---")
    if shortfalls:
        print(f"{len(shortfalls)} targets got fewer than 5 decoys: {', '.join(shortfalls)}")
    else:
        print("0 targets got fewer than 5 decoys.")
        
    print("\n--- Rule Check ---")
    print(f"All decoys meet +/- 15% rule: {all_meet_rule}")
    
    print("\n--- Sanity Check ---")
    if dist_to_zn_all:
        min_dist = min(dist_to_zn_all)
        print(f"Min dist_to_zn: {min_dist:.2f} A (Should be >= 4.0)")
        print(f"All >= 4.0 A: {min_dist >= 4.0}")
    else:
        print("No decoys found.")
        
if __name__ == '__main__':
    verify()
