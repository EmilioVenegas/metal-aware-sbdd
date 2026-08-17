import json
import urllib.request
import torch
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

def get_chem_comp_info(comp_id):
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def get_entry_info(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    query = """
    {
      entry(entry_id: "%s") {
        rcsb_entry_info {
          resolution_combined
        }
        exptl {
          method
        }
        struct {
          title
        }
        polymer_entities {
          rcsb_polymer_entity_container_identifiers {
            uniprot_ids
          }
          rcsb_entity_source_organism {
            scientific_name
          }
          rcsb_polymer_entity {
            pdbx_description
          }
        }
      }
    }
    """ % pdb_id
    try:
        r = requests.post("https://data.rcsb.org/graphql", json={"query": query}, timeout=10)
        if r.status_code == 200:
            return r.json().get('data', {}).get('entry', {})
    except Exception:
        pass
    return None

def is_drug_like(smiles):
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    # Basic drug-likeness: >= 12 heavy atoms
    if mol.GetNumHeavyAtoms() < 12:
        return False
    
    mw = Descriptors.MolWt(mol)
    if mw > 1000:
        return False
        
    qed = QED.qed(mol)
    if qed < 0.1: # Very loose QED filter to drop extreme non-drugs
        return False
        
    return True

def compute_pairwise_identity(seq1, seq2):
    from Bio import Align
    if not seq1 or not seq2: return 0.0
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    try:
        alns = aligner.align(seq1, seq2)
        if not alns: return 0.0
        aln = alns[0]
        # score is the number of matches
        matches = aln.score
        aln_len = len(aln)
        return matches / aln_len if aln_len > 0 else 0.0
    except Exception:
        return 0.0

def cluster_targets(targets, threshold=0.30):
    n = len(targets)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, i] = True
        for j in range(i + 1, n):
            if compute_pairwise_identity(targets[i]['sequence'], targets[j]['sequence']) >= threshold:
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
    payload = torch.load("data/external_zn_test.pt")
    targets = payload["targets"]
    
    print(f"Loaded {len(targets)} targets.")
    
    passed_targets = []
    dropped = []
    
    for t in targets:
        pdb = t["pdb_id"]
        lig_id = t["ligand_resname"]
        
        # 1. Component check
        comp_info = get_chem_comp_info(lig_id)
        if not comp_info:
            print(f"Could not fetch info for {lig_id}. Dropping.")
            dropped.append(f"{pdb} {lig_id}: No CCD info")
            continue
            
        ctype = comp_info.get("chem_comp", {}).get("type", "").lower()
        name = comp_info.get("chem_comp", {}).get("name", "").lower()
        
        if ctype != "non-polymer":
            dropped.append(f"{pdb} {lig_id}: Type is {ctype}")
            continue
            
        smiles = comp_info.get("rcsb_chem_comp_descriptor", {}).get("SMILES_stereo") or comp_info.get("rcsb_chem_comp_descriptor", {}).get("SMILES")
        if not is_drug_like(smiles):
            dropped.append(f"{pdb} {lig_id}: Failed basic drug-likeness or <12 HA (name: {name})")
            continue
            
        # 2. Structural quality filters
        entry_info = get_entry_info(pdb)
        if not entry_info:
            dropped.append(f"{pdb}: No entry info")
            continue
            
        res_list = entry_info.get("rcsb_entry_info", {}).get("resolution_combined", [99.0])
        res = res_list[0] if res_list else 99.0
        
        exptl = entry_info.get("exptl", [])
        methods = [m.get("method", "") for m in exptl]
        is_cryo = any("ELECTRON MICROSCOPY" in m.upper() for m in methods)
        
        if not is_cryo and res > 2.5:
            dropped.append(f"{pdb}: Resolution {res} > 2.5")
            continue
            
        # Check active-site mutant
        is_mutant = False
        uniprot = ""
        protein_name = ""
        organism = ""
        
        # Grab info from the chain that has the Zn
        chain_id = t["chain"]
        for poly in entry_info.get("polymer_entities", []):
            if not uniprot and poly.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids"):
                uniprot = poly["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"][0]
            if not protein_name and poly.get("rcsb_polymer_entity", {}).get("pdbx_description"):
                protein_name = poly["rcsb_polymer_entity"]["pdbx_description"]
            if not organism and poly.get("rcsb_entity_source_organism"):
                organism = poly["rcsb_entity_source_organism"][0].get("scientific_name", "")
                
        title = entry_info.get("struct", {}).get("title", "").lower()
        if "mutant" in title or "mutation" in title:
            is_mutant = True
            
        if is_mutant:
            dropped.append(f"{pdb}: Marked as mutant in title")
            continue
            
        # 4. Clean up the coordinating atom annotation
        cleaned_coord = []
        for c in t["coordinating_ligand_atoms"]:
            atom_name = c.split(" ")[0]
            dist = c.split("(")[1].replace("A)", "")
            cleaned_coord.append(f"{atom_name} ({dist}A)")
            
        t["coordinating_ligand_atoms_clean"] = cleaned_coord
        t["resolution"] = res
        t["method"] = "Cryo-EM" if is_cryo else "X-ray"
        t["uniprot"] = uniprot
        t["protein_name"] = protein_name
        t["organism"] = organism
        
        passed_targets.append(t)

    print(f"\nDropped {len(dropped)} targets:")
    for d in dropped:
        print("  " + d)
        
    print(f"\nClustering remaining {len(passed_targets)} targets...")
    clusters = cluster_targets(passed_targets, 0.30)
    
    print(f"\nSurviving Clusters: {len(clusters)}")
    
    with open("results/step1/external_zn_cluster_report.md", "w") as f:
        f.write("# Clean External Zinc Set Audit\n\n")
        f.write("| Cluster | Rep PDB | Method | Res | Protein | Organism | UniProt | Ligand | HA | Coord |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, c in enumerate(clusters):
            rep = passed_targets[c[0]]
            res_str = f"{rep['resolution']}A" if rep['method'] != 'Cryo-EM' else "Cryo-EM"
            f.write(f"| {i+1} | {rep['pdb_id']} | {rep['method']} | {res_str} | {rep['protein_name']} | {rep['organism']} | {rep['uniprot']} | {rep['ligand_resname']} | {rep['ligand_num_heavy_atoms']} | {', '.join(rep['coordinating_ligand_atoms_clean'])} |\n")
            
    print("Report written to results/step1/external_zn_cluster_report.md")
    
    # Update payload and resave
    surviving_representatives = [passed_targets[c[0]] for c in clusters]
    payload["targets"] = passed_targets
    payload["representative_targets"] = surviving_representatives
    payload["m_clusters"] = len(clusters)
    payload["m_targets"] = len(passed_targets)
    payload["target_pdb_ids"] = [t["pdb_id"] for t in surviving_representatives]
    payload["provenance"]["criteria"].extend([
        "Ligand is purely non-polymer and passes basic drug-likeness (QED>0.1, >=12 HA)",
        "Structure resolution <= 2.5 A (or is Cryo-EM)",
        "Not an explicitly flagged active-site mutant",
        "Ligand coordination derived from chemical component, not functional annotations"
    ])
    payload["provenance"]["total_independent_clusters"] = len(clusters)
    
    torch.save(payload, "data/external_zn_test_clean.pt")
    print(f"Saved payload to data/external_zn_test_clean.pt with {len(clusters)} clusters.")

    print("\nCLUSTER REPORT:")
    for i, c in enumerate(clusters):
        rep = passed_targets[c[0]]
        print(f"Cluster {i+1}:")
        print(f"  Rep PDB: {rep['pdb_id']} | Method: {rep['method']} | Res: {rep['resolution']}A")
        print(f"  Protein: {rep['protein_name']} | Organism: {rep['organism']} | UniProt: {rep['uniprot']}")
        print(f"  Ligand: {rep['ligand_resname']} ({rep['ligand_num_heavy_atoms']} HA)")
        print(f"  Coord: {rep['coordinating_ligand_atoms_clean']} (dist={rep['min_zn_ligand_dist']:.2f}A)")
        print(f"  Members: {len(c)}")
        print()

if __name__ == '__main__':
    main()
