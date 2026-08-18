#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Standard Zinc-Binding Groups (ZBGs)
ZBG_SMARTS = {
    "carboxylate": "[CX3](=O)[OX1H0-,OX2H1]",
    "hydroxamate": "[NX3H1,NX3H0]([OX2H1,OX2H0])C(=O)",
    "thiol": "[SX2H1,SX1H0-]",
    "imidazole": "c1ncnc1", # general imidazole
    "sulfonamide": "[NX3H2,NX3H1][SX4](=O)(=O)"
}

def main():
    parser = argparse.ArgumentParser(description="Filter Arm A generation using SMARTS for known Zinc-Binding Groups")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing Arm A SDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save filtered SDF files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile SMARTS
    zbg_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in ZBG_SMARTS.items()}

    total_mols = 0
    total_retained = 0

    for sdf_file in input_dir.glob("*.sdf"):
        supplier = Chem.SDMolSupplier(str(sdf_file), sanitize=True)
        retained_mols = []
        for mol in supplier:
            if mol is None:
                continue
            total_mols += 1
            
            # Check if mol matches ANY ZBG
            matched = False
            for name, pattern in zbg_patterns.items():
                if pattern is not None and mol.HasSubstructMatch(pattern):
                    matched = True
                    break
            
            if matched:
                retained_mols.append(mol)
                total_retained += 1
        
        if retained_mols:
            w = Chem.SDWriter(str(output_dir / sdf_file.name))
            for m in retained_mols:
                w.write(m)
            w.close()

    print(f"SMARTS Filtering Complete.")
    print(f"Total valid generated molecules evaluated: {total_mols}")
    print(f"Molecules retained containing a ZBG: {total_retained} ({(total_retained/total_mols)*100:.1f}%)")

if __name__ == "__main__":
    main()
