"""G2: Validate coordinate frame — median Zn-donor distance for native ligands.

Gate: median must fall in 1.9–2.3 Å.
Uses the representative target from each X-ray cluster in external_zn_test_clean.pt.
Downloads CIF fresh (not relying on cached coordinates from build step) and
re-measures Zn-to-coordinating-atom distances.

Reports:
  - Per-cluster distances
  - Median and percentiles over all coordinating atoms
  - PASS / FAIL gate result
"""

import io
import sys
import urllib.request
import warnings

import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

PARSER = MMCIFParser(QUIET=True)

def download_cif(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    req = urllib.request.urlopen(url, timeout=15)
    return req.read().decode('utf-8')


def measure_distances(pdb_id, ligand_resname, coordinating_atoms_clean):
    """Re-download CIF and measure Zn-to-donor distances.

    Multi-Zn / multi-ligand fix: enumerate all (Zn, ligand-residue) pairs and
    select the pair whose minimum donor distance is smallest — that is the
    catalytic pair. Only report distances from that pair.
    """
    try:
        content = download_cif(pdb_id)
    except Exception as e:
        return None, f"download failed: {e}"

    try:
        structure = PARSER.get_structure(pdb_id, io.StringIO(content))
    except Exception as e:
        return None, f"parse failed: {e}"

    # Expected coordinating-atom names, e.g. ["O2", "O4", "O5"]
    expected_atoms = {c.split(" ")[0].upper() for c in coordinating_atoms_clean}

    # Collect all Zn positions
    zn_positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if (residue.id[0] not in (' ', 'W')
                        and residue.get_resname().strip().upper() == 'ZN'):
                    for atom in residue:
                        zn_positions.append(atom.coord.copy())

    if not zn_positions:
        return None, "no ZN found in structure"

    # Collect all ligand residue instances matching the resname,
    # keeping only atoms whose names are in the expected set.
    # Store as list of {atom_name: coord} dicts.
    lig_instances = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if (residue.id[0] not in (' ', 'W')
                        and residue.get_resname().strip().upper() == ligand_resname.upper()):
                    donors = {}
                    for atom in residue:
                        aname = atom.name.strip().upper()
                        if aname in expected_atoms:
                            donors[aname] = atom.coord.copy()
                    if donors:
                        lig_instances.append(donors)

    if not lig_instances:
        return None, f"no donor atoms found in any {ligand_resname} residue (expected {expected_atoms})"

    # Select the (Zn, ligand-residue) pair with the smallest minimum donor distance.
    best_dists = None
    best_min = float('inf')
    for zn_c in zn_positions:
        for lig in lig_instances:
            dists = [float(np.linalg.norm(zn_c - dc)) for dc in lig.values()]
            mn = min(dists)
            if mn < best_min:
                best_min = mn
                best_dists = dists

    if best_dists is None:
        return None, "no valid (Zn, ligand) pair found"

    return sorted(best_dists), None


def main():
    payload = torch.load("data/external_zn_test_clean.pt")
    reps = payload["representative_targets"]

    # Use X-ray clusters only for G2
    xray_reps = [t for t in reps if t.get("method", "X-ray") == "X-ray"]
    print(f"G2: validating {len(xray_reps)} X-ray representative targets")
    print()

    all_distances = []
    results = []

    for t in xray_reps:
        pdb = t["pdb_id"]
        lig = t["ligand_resname"]
        coord_labels = t.get("coordinating_ligand_atoms_clean", t.get("coordinating_ligand_atoms", []))
        stored_dists = []
        for c in coord_labels:
            try:
                d = float(c.split("(")[1].replace("A)", ""))
                stored_dists.append(d)
            except Exception:
                pass

        dists, err = measure_distances(pdb, lig, coord_labels)
        if err:
            print(f"  {pdb} {lig}: ERROR — {err}")
            results.append({"pdb": pdb, "lig": lig, "error": err})
            continue

        min_d = min(dists) if dists else float('nan')
        all_distances.extend(dists)
        results.append({"pdb": pdb, "lig": lig, "distances": dists, "min": min_d})
        stored_str = ", ".join(f"{d:.2f}" for d in stored_dists)
        remeas_str = ", ".join(f"{d:.2f}" for d in sorted(dists))
        flag = " <-- CHECK" if min_d < 1.75 or min_d > 2.55 else ""
        print(f"  {pdb} {lig}: stored=[{stored_str}] remeasured=[{remeas_str}]{flag}")

    print()
    if not all_distances:
        print("G2: FAIL — no distances measured")
        sys.exit(1)

    arr = np.array(all_distances)
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])
    print(f"  N coordinating atoms measured: {len(arr)}")
    print(f"  5th percentile:  {p5:.3f} Å")
    print(f"  Median:          {p50:.3f} Å")
    print(f"  95th percentile: {p95:.3f} Å")
    print()

    gate_low, gate_high = 1.9, 2.3
    if gate_low <= p50 <= gate_high:
        print(f"G2: PASS — median {p50:.3f} Å is within [{gate_low}, {gate_high}] Å")
    else:
        print(f"G2: FAIL — median {p50:.3f} Å is OUTSIDE [{gate_low}, {gate_high}] Å")
        print("    Coordinate frame is wrong. Step 1 must not proceed.")
        sys.exit(1)

    # Count outlier donors (outside 1.75–2.55 for all metal-donor pairs)
    outliers = arr[(arr < 1.75) | (arr > 2.55)]
    print(f"  Outlier donors (<1.75 or >2.55 Å): {len(outliers)}/{len(arr)} ({100*len(outliers)/len(arr):.1f}%)")


if __name__ == "__main__":
    main()
