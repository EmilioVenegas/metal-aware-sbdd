#!/usr/bin/env python
"""Extract native ligands to SDF for external Zn test targets.

For each of the 133 targets in data/external_zn_test_clean.pt:
- Extracts the exact ligand instance specified by ligand_id (<chain>_<resname>_<resnum>).
- Preserves exact deposited coordinate frame (no translation/rotation/centering).
- Assigns bond orders and connectivity from the PDB Chemical Component Dictionary (CCD).
- Writes heavy atom SDF to data/native_ligands/<PDB_ID>.sdf.
- Verifies heavy-atom to Zn coordinate distance against recorded min_zn_ligand_dist.
"""

import os
import sys
import urllib.request
import numpy as np
import torch
import gemmi
from rdkit import Chem
from rdkit.Geometry import Point3D


BOND_ORDER_MAP = {
    'SING': Chem.BondType.SINGLE,
    'DOUB': Chem.BondType.DOUBLE,
    'TRIP': Chem.BondType.TRIPLE,
    'QUAD': Chem.BondType.QUADRUPLE,
    'AROM': Chem.BondType.AROMATIC,
    'DELO': Chem.BondType.AROMATIC,
}


def get_ccd_info(resname: str, cif_path: str = None, ccd_cache_dir: str = "data/ccd_cache"):
    """Fetch bond orders and connectivity from mmCIF chem_comp categories or RCSB CCD."""
    bonds = []
    atoms_info = {}

    def parse_block(b):
        tb = b.find_mmcif_category('_chem_comp_bond')
        if len(tb) > 0:
            for row in tb:
                if row[0].upper() == resname.upper():
                    bonds.append((row[1], row[2], row[3], row[4] if len(row) > 4 else 'N'))
        ta = b.find_mmcif_category('_chem_comp_atom')
        if len(ta) > 0:
            for row in ta:
                if row[0].upper() == resname.upper():
                    aname = row[1]
                    elem = row[2]
                    atoms_info[aname] = {'elem': elem}

    # 1. Try embedded in the target structure's CIF file if available
    if cif_path and os.path.exists(cif_path) and cif_path.endswith('.cif'):
        try:
            doc = gemmi.cif.read_file(cif_path)
            if doc:
                parse_block(doc[0])
                if bonds:
                    return bonds, atoms_info
        except Exception:
            pass

    # 2. Check local CCD cache
    os.makedirs(ccd_cache_dir, exist_ok=True)
    cache_file = os.path.join(ccd_cache_dir, f"{resname.upper()}.cif")
    if not os.path.exists(cache_file):
        url = f"https://files.rcsb.org/ligands/view/{resname.upper()}.cif"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            content = urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
            with open(cache_file, 'w') as f:
                f.write(content)
        except Exception as e:
            # Return empty if CCD cannot be fetched
            return [], {}

    try:
        doc = gemmi.cif.read_file(cache_file)
        if doc:
            parse_block(doc[0])
    except Exception:
        pass

    return bonds, atoms_info


def parse_ligand_id(ligand_id: str):
    """Parse ligand_id format: <chain>_<resname>_<resnum>."""
    parts = ligand_id.split('_')
    chain_id = parts[0]
    resnum = parts[-1]
    resname = '_'.join(parts[1:-1])
    return chain_id, resname, resnum


def extract_ligand(target_record: dict, pdbs_dir: str = "data/external_pdbs", out_dir: str = "data/native_ligands"):
    """Extract native ligand for target record to SDF and compute verification distance."""
    pdb_id = target_record['pdb_id']
    ligand_id = target_record['ligand_id']
    target_resname = target_record['ligand_resname']
    zn_coord = np.array(target_record['zn_coord'], dtype=float)
    rec_min_dist = float(target_record['min_zn_ligand_dist'])

    chain_id, resname, resnum = parse_ligand_id(ligand_id)

    pdb_path = os.path.join(pdbs_dir, f"{pdb_id}.pdb")
    cif_path = os.path.join(pdbs_dir, f"{pdb_id}.cif")
    path = pdb_path if os.path.exists(pdb_path) else cif_path

    if not os.path.exists(path):
        return {
            'pdb_id': pdb_id,
            'ligand_id': ligand_id,
            'status': 'FAILED',
            'error': f'Structure file not found: {pdb_path} or {cif_path}',
            'n_heavy_atoms': 0,
            'computed_min_dist': float('nan'),
            'recorded_min_dist': rec_min_dist,
            'delta': float('nan'),
            'flagged': True
        }

    try:
        st = gemmi.read_structure(path)
    except Exception as e:
        return {
            'pdb_id': pdb_id,
            'ligand_id': ligand_id,
            'status': 'FAILED',
            'error': f'Failed to parse structure: {e}',
            'n_heavy_atoms': 0,
            'computed_min_dist': float('nan'),
            'recorded_min_dist': rec_min_dist,
            'delta': float('nan'),
            'flagged': True
        }

    # Find the matching residue instance: chain, resname, and resnum
    target_res = None
    for model in st:
        for chain in model:
            if chain.name == chain_id:
                for res in chain:
                    if res.name == resname and str(res.seqid.num) == str(resnum):
                        target_res = res
                        break
                if target_res:
                    break
        if target_res:
            break

    if target_res is None:
        return {
            'pdb_id': pdb_id,
            'ligand_id': ligand_id,
            'status': 'FAILED',
            'error': f'Residue {ligand_id} not found in structure',
            'n_heavy_atoms': 0,
            'computed_min_dist': float('nan'),
            'recorded_min_dist': rec_min_dist,
            'delta': float('nan'),
            'flagged': True
        }

    # Extract heavy atoms, handling alternate conformations (prefer 'A' or first seen)
    res_atoms = []
    seen_names = set()
    for a in target_res:
        elem = a.element.name.strip().capitalize()
        if elem in ('H', 'D'):
            continue
        aname = a.name.strip()
        if aname in seen_names:
            continue
        if a.altloc in ('', 'A', '1', '\x00'):
            seen_names.add(aname)
            res_atoms.append(a)

    for a in target_res:
        elem = a.element.name.strip().capitalize()
        if elem in ('H', 'D'):
            continue
        aname = a.name.strip()
        if aname not in seen_names:
            seen_names.add(aname)
            res_atoms.append(a)

    if not res_atoms:
        return {
            'pdb_id': pdb_id,
            'ligand_id': ligand_id,
            'status': 'FAILED',
            'error': 'No heavy atoms found in residue',
            'n_heavy_atoms': 0,
            'computed_min_dist': float('nan'),
            'recorded_min_dist': rec_min_dist,
            'delta': float('nan'),
            'flagged': True
        }

    # Fetch CCD bond definitions
    ccd_bonds, _ = get_ccd_info(resname, path)

    # Build RDKit Mol preserving exact coordinates
    mol = Chem.RWMol()
    atom_idx_map = {}
    conf = Chem.Conformer()

    for a in res_atoms:
        aname = a.name.strip()
        elem = a.element.name.strip().capitalize()
        rd_atom = Chem.Atom(elem)
        rd_atom.SetProp('name', aname)
        idx = mol.AddAtom(rd_atom)
        atom_idx_map[aname] = idx
        conf.SetAtomPosition(idx, Point3D(float(a.pos.x), float(a.pos.y), float(a.pos.z)))

    mol.AddConformer(conf)

    # Add CCD bonds
    bonds_added = 0
    for b in ccd_bonds:
        a1, a2, order, arom = b
        if a1 in atom_idx_map and a2 in atom_idx_map:
            i1 = atom_idx_map[a1]
            i2 = atom_idx_map[a2]
            if mol.GetBondBetweenAtoms(i1, i2) is None:
                btype = BOND_ORDER_MAP.get(order, Chem.BondType.SINGLE)
                mol.AddBond(i1, i2, btype)
                bonds_added += 1

    final_mol = mol.GetMol()
    final_mol.SetProp('_Name', f"{pdb_id}_{ligand_id}")

    # Write SDF
    os.makedirs(out_dir, exist_ok=True)
    out_sdf = os.path.join(out_dir, f"{pdb_id}.sdf")
    writer = Chem.SDWriter(out_sdf)
    writer.write(final_mol)
    writer.close()

    # Verify produced SDF by reading it back
    suppl = Chem.SDMolSupplier(out_sdf, removeHs=False, sanitize=False)
    read_mol = suppl[0]
    read_conf = read_mol.GetConformer()

    heavy_coords = []
    for idx, atom in enumerate(read_mol.GetAtoms()):
        if atom.GetSymbol() not in ('H', 'D'):
            pos = read_conf.GetAtomPosition(idx)
            heavy_coords.append(np.array([pos.x, pos.y, pos.z]))

    n_heavy = len(heavy_coords)
    dists = [np.linalg.norm(c - zn_coord) for c in heavy_coords]
    computed_min_dist = min(dists)
    delta = abs(computed_min_dist - rec_min_dist)
    flagged = delta > 0.1

    return {
        'pdb_id': pdb_id,
        'ligand_id': ligand_id,
        'status': 'SUCCESS',
        'n_heavy_atoms': n_heavy,
        'computed_min_dist': computed_min_dist,
        'recorded_min_dist': rec_min_dist,
        'delta': delta,
        'flagged': flagged,
        'bonds_count': bonds_added,
        'ccd_found': len(ccd_bonds) > 0
    }


def main():
    data_path = "data/external_zn_test_clean.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    payload = torch.load(data_path, map_location='cpu')
    targets = payload["targets"]
    print(f"Loaded {len(targets)} targets from {data_path}")

    results = []
    for target in targets:
        res = extract_ligand(target)
        results.append(res)

    print("\n" + "=" * 80)
    print(f"{'pdb_id':<8} | {'n_heavy':<7} | {'computed_min':<12} | {'recorded_min':<12} | {'delta':<10} | {'flag'}")
    print("-" * 80)

    for r in results:
        flag_str = "FLAG (>0.1A)" if r['flagged'] else "OK"
        print(f"{r['pdb_id']:<8} | {r['n_heavy_atoms']:<7} | {r['computed_min_dist']:<12.3f} | {r['recorded_min_dist']:<12.3f} | {r['delta']:<10.4f} | {flag_str}")

    print("=" * 80)

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    flagged_count = sum(1 for r in results if r['flagged'])
    print(f"\nSummary:")
    print(f"  Total targets processed: {len(results)}")
    print(f"  Extraction successful:   {success_count}/{len(results)}")
    print(f"  Flagged (|delta| > 0.1A): {flagged_count}/{len(results)}")

    if flagged_count > 0:
        print(f"\nFlagged targets ({flagged_count}):")
        for r in results:
            if r['flagged']:
                print(f"  {r['pdb_id']}: n_heavy={r['n_heavy_atoms']}, computed={r['computed_min_dist']:.3f}, recorded={r['recorded_min_dist']:.3f}, delta={r['delta']:.4f}")


if __name__ == "__main__":
    main()
