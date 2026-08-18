#!/usr/bin/env python
"""
Independent Leakage Verification Script

Per docs/step2.md: "re-run the leakage check independently before it becomes load-bearing for a headline claim."
Checks for PDB ID overlap between the Arm C training set and the external Zn test set.
"""
import torch

def verify_leakage():
    # Load test set
    test_data = torch.load("data/external_zn_test_clean.pt", map_location="cpu", weights_only=False)
    test_pdbs = {t["pdb_id"].lower() for t in test_data["targets"]}
    print(f"Loaded {len(test_pdbs)} test PDB IDs from external Zn cohort.")

    # Load Arm C train set
    arm_c_train = torch.load("data/arm_c_train.pt", map_location="cpu", weights_only=False)
    if isinstance(arm_c_train, dict) and "examples" in arm_c_train:
        train_pdbs = {t["pdb_id"].lower() for t in arm_c_train["examples"]}
    else:
        train_pdbs = {t["pdb_id"].lower() for t in arm_c_train}
    print(f"Loaded {len(train_pdbs)} train PDB IDs from Arm C training set.")

    # Load Arm C val set
    arm_c_val = torch.load("data/arm_c_val.pt", map_location="cpu", weights_only=False)
    if isinstance(arm_c_val, dict) and "examples" in arm_c_val:
        val_pdbs = {t["pdb_id"].lower() for t in arm_c_val["examples"]}
    else:
        val_pdbs = {t["pdb_id"].lower() for t in arm_c_val}
    print(f"Loaded {len(val_pdbs)} val PDB IDs from Arm C validation set.")

    train_overlap = test_pdbs.intersection(train_pdbs)
    val_overlap = test_pdbs.intersection(val_pdbs)

    if not train_overlap and not val_overlap:
        print("\nSUCCESS: 0 PDB ID leakage detected between Arm C sets and Test set.")
    else:
        print("\nCRITICAL FAILURE: Leakage detected!")
        if train_overlap:
            print(f"Train overlap ({len(train_overlap)} targets): {train_overlap}")
        if val_overlap:
            print(f"Val overlap ({len(val_overlap)} targets): {val_overlap}")
        exit(1)

    # Let's also check UniProt overlap if available
    try:
        test_uniprots = {t.get("uniprot") for t in test_data["targets"] if t.get("uniprot")}
        train_uniprots = {t.get("uniprot") for t in (arm_c_train["targets"] if isinstance(arm_c_train, dict) else arm_c_train) if t.get("uniprot")}
        
        if test_uniprots and train_uniprots:
            uniprot_overlap = test_uniprots.intersection(train_uniprots)
            if not uniprot_overlap:
                print("SUCCESS: 0 UniProt ID leakage detected.")
            else:
                print(f"WARNING: UniProt overlap detected ({len(uniprot_overlap)} targets): {uniprot_overlap}")
    except Exception as e:
        print(f"Skipping UniProt overlap check due to data structure: {e}")

if __name__ == "__main__":
    verify_leakage()
