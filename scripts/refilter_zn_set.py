"""Refilter external_zn_test.pt with strict positive criteria.

Fixes applied over the initial build_overbuilt_clean_zn_set.py output:
  1. Explicit CCD code blacklist for nucleotides, cofactors, and metabolites
     (AMP/GTP/C5P typed 'non-polymer' but are substrates, not drug-like inhibitors)
  2. min_zn_ligand_dist corrected to minimum over coordinating-donor atoms only,
     not all ligand atoms (was returning covalent-bond distances ~1.48 A)
  3. Same-UniProt cluster merging: clusters sharing a UniProt ID are collapsed
     before the final count (ACE P12821 appeared as three separate clusters)
  4. Cryo-EM entries flagged separately; not counted in primary (X-ray) total
"""

import json
import torch
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# ── CCD blacklist: metabolites / cofactors / nucleotides ──────────────────────
# These pass 'non-polymer' type but are not drug-like SBDD ligands.
METABOLITE_BLACKLIST = {
    # Adenine nucleotides
    'AMP', 'ADP', 'ATP', 'ANP', 'APC', 'APN', 'AP5', 'AP4',
    # Guanine nucleotides (all aliases)
    'GMP', 'GDP', 'GTP', 'GNP', 'GMX', '5GP',   # 5GP = GMP by another code
    # Cytidine nucleotides
    'CMP', 'CDP', 'CTP', 'C5P',
    # Uridine nucleotides
    'UMP', 'UDP', 'UTP', 'URI',
    # Thymidine nucleotides
    'TMP', 'TDP', 'TTP',
    # Nicotinamides
    'NAD', 'NAI', 'NDP', 'NAP', 'NMN', 'NHD',
    # Flavins
    'FAD', 'FMN', 'FAO',
    # CoA / acyl-CoA
    'COA', 'ACO', 'MLC', 'PAF',
    # PAP-sulfate and related cofactor analogues
    'PPS', 'PAP',
    # SAM / SAH
    'SAM', 'SAH', 'MSM',
    # PLP / pyridoxal
    'PLP', 'PMP', 'PYR',
    # Heme
    'HEM', 'HEA', 'HEB', 'HEC',
    # Lipid metabolites / substrates that are not drug-like
    'NKP',    # lysophospholipid (ENPP2 substrate)
    # Simple phosphates / sugars that slipped through
    'GLC', 'FRU', 'GAL', 'MAN',
}

# Minimum allowed coordination distance.
# Values <= this threshold indicate a covalent adduct, not ion coordination.
MIN_COORD_DIST_CUTOFF = 1.75  # Ångström


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
    if mol.GetNumHeavyAtoms() < 12:
        return False
    mw = Descriptors.MolWt(mol)
    if mw > 1000:
        return False
    qed = QED.qed(mol)
    if qed < 0.1:
        return False
    return True


def compute_pairwise_identity(seq1, seq2):
    from Bio import Align
    if not seq1 or not seq2:
        return 0.0
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    try:
        alns = aligner.align(seq1, seq2)
        if not alns:
            return 0.0
        aln = alns[0]
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


def merge_same_uniprot_clusters(clusters, targets):
    """Collapse clusters that share a UniProt ID into one cluster.

    Rationale: ACE (P12821) appeared as 3 clusters because its soluble and
    membrane forms differ in sequence length but are the same drug target.
    Reporting them as independent would overcount the effective sample size.
    We keep all members but count the merged group as one cluster.
    """
    uniprot_to_cluster = {}  # uniprot -> cluster index in merged list
    merged = []

    for c in clusters:
        rep = targets[c[0]]
        uid = rep.get('uniprot', '')
        if uid and uid in uniprot_to_cluster:
            # Append all members to the existing cluster
            merged[uniprot_to_cluster[uid]].extend(c)
        else:
            idx = len(merged)
            merged.append(list(c))
            if uid:
                uniprot_to_cluster[uid] = idx

    return merged


def main():
    payload = torch.load("data/external_zn_test.pt")
    targets_raw = payload["targets"]
    print(f"Loaded {len(targets_raw)} targets from data/external_zn_test.pt")

    passed_targets = []
    dropped = []

    for t in targets_raw:
        pdb = t["pdb_id"]
        lig_id = t["ligand_resname"]

        # ── Filter 1: CCD metabolite blacklist ────────────────────────────────
        if lig_id.upper() in METABOLITE_BLACKLIST:
            dropped.append(f"{pdb} {lig_id}: In metabolite/nucleotide blacklist")
            continue

        # ── Filter 2: CCD type must be non-polymer ────────────────────────────
        comp_info = get_chem_comp_info(lig_id)
        if not comp_info:
            dropped.append(f"{pdb} {lig_id}: No CCD info")
            continue

        ctype = comp_info.get("chem_comp", {}).get("type", "").lower()
        name = comp_info.get("chem_comp", {}).get("name", "").lower()

        if ctype != "non-polymer":
            dropped.append(f"{pdb} {lig_id}: CCD type={ctype}")
            continue

        # ── Filter 3: RDKit drug-likeness (>=12 HA, QED>0.1, MW<=1000) ───────
        smiles = (comp_info.get("rcsb_chem_comp_descriptor", {}).get("SMILES_stereo")
                  or comp_info.get("rcsb_chem_comp_descriptor", {}).get("SMILES"))
        if not is_drug_like(smiles):
            dropped.append(f"{pdb} {lig_id}: Failed drug-likeness (name: {name})")
            continue

        # ── Filter 4: Structural quality ──────────────────────────────────────
        entry_info = get_entry_info(pdb)
        if not entry_info:
            dropped.append(f"{pdb}: No entry info from GraphQL")
            continue

        res_list = entry_info.get("rcsb_entry_info", {}).get("resolution_combined", [99.0])
        res = res_list[0] if res_list else 99.0

        exptl = entry_info.get("exptl", [])
        methods = [m.get("method", "") for m in exptl]
        is_cryo = any("ELECTRON MICROSCOPY" in m.upper() for m in methods)

        if not is_cryo and res > 2.5:
            dropped.append(f"{pdb}: Resolution {res:.2f} > 2.5 (X-ray)")
            continue

        # ── Filter 5: Not an active-site mutant ───────────────────────────────
        title = entry_info.get("struct", {}).get("title", "").lower()
        if "mutant" in title or "mutation" in title:
            dropped.append(f"{pdb}: Title contains 'mutant/mutation'")
            continue

        # ── Collect metadata ──────────────────────────────────────────────────
        uniprot = ""
        protein_name = ""
        organism = ""
        for poly in entry_info.get("polymer_entities", []):
            if not uniprot:
                uid_list = poly.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids")
                if uid_list:
                    uniprot = uid_list[0]
            if not protein_name:
                desc = poly.get("rcsb_polymer_entity", {}).get("pdbx_description")
                if desc:
                    protein_name = desc
            if not organism:
                org_list = poly.get("rcsb_entity_source_organism")
                if org_list:
                    organism = org_list[0].get("scientific_name", "")

        # ── Fix 1: min_zn_ligand_dist = min over COORDINATING donor atoms only ──
        # The original field tracked min over ALL atoms and could return 1.48 A
        # for a covalent bond atom that happens to be closer than the donor.
        coord_dists = []
        for c in t["coordinating_ligand_atoms"]:
            try:
                dist = float(c.split("(")[1].replace("A)", ""))
                coord_dists.append(dist)
            except Exception:
                pass
        min_coord_dist = min(coord_dists) if coord_dists else t["min_zn_ligand_dist"]

        # ── Filter 6: Reject covalent adducts (min coord dist <= 1.75 A) ─────
        if min_coord_dist <= MIN_COORD_DIST_CUTOFF:
            dropped.append(
                f"{pdb} {lig_id}: min coord dist {min_coord_dist:.2f} A <= {MIN_COORD_DIST_CUTOFF} A (covalent adduct)"
            )
            continue

        # Clean coord labels: "O3 (1.81A)" stays as-is (atom name + distance)
        cleaned_coord = []
        for c in t["coordinating_ligand_atoms"]:
            atom_name = c.split(" ")[0]
            dist_str = c.split("(")[1].replace("A)", "") if "(" in c else "?"
            cleaned_coord.append(f"{atom_name} ({dist_str}A)")

        t["coordinating_ligand_atoms_clean"] = cleaned_coord
        t["min_coord_dist"] = min_coord_dist  # corrected field
        t["resolution"] = res
        t["method"] = "Cryo-EM" if is_cryo else "X-ray"
        t["uniprot"] = uniprot
        t["protein_name"] = protein_name
        t["organism"] = organism

        passed_targets.append(t)

    print(f"\nDropped {len(dropped)} targets:")
    for d in dropped:
        print("  " + d)

    print(f"\nClustering {len(passed_targets)} surviving targets at 30% sequence identity...")
    clusters_seq = cluster_targets(passed_targets, 0.30)
    print(f"  Sequence clusters: {len(clusters_seq)}")

    # ── Fix 2: Merge clusters sharing a UniProt ID ────────────────────────────
    clusters = merge_same_uniprot_clusters(clusters_seq, passed_targets)
    n_merged = len(clusters_seq) - len(clusters)
    print(f"  After same-UniProt merge: {len(clusters)} clusters ({n_merged} merged)")

    # Separate X-ray vs cryo-EM clusters
    xray_clusters = [c for c in clusters
                     if passed_targets[c[0]]['method'] == 'X-ray']
    cryo_clusters = [c for c in clusters
                     if passed_targets[c[0]]['method'] == 'Cryo-EM']

    print(f"\n  X-ray clusters (primary):   {len(xray_clusters)}")
    print(f"  Cryo-EM clusters (flagged): {len(cryo_clusters)}")

    # ── Write cluster report ──────────────────────────────────────────────────
    import os
    os.makedirs("results/step1", exist_ok=True)
    with open("results/step1/external_zn_cluster_report.md", "w") as f:
        f.write("# Clean External Zinc Set — Cluster Audit\n\n")
        f.write(f"- Input targets: {len(targets_raw)}\n")
        f.write(f"- After all filters: {len(passed_targets)}\n")
        f.write(f"- Sequence clusters (30% identity): {len(clusters_seq)}\n")
        f.write(f"- After same-UniProt merge: {len(clusters)}\n")
        f.write(f"  - X-ray (primary): {len(xray_clusters)}\n")
        f.write(f"  - Cryo-EM (flagged, not primary): {len(cryo_clusters)}\n\n")
        f.write("## Primary X-ray Clusters\n\n")
        f.write("| # | Rep PDB | Res (Å) | Protein | UniProt | Organism | Ligand | HA | Coord (Å) | Members |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, c in enumerate(xray_clusters):
            rep = passed_targets[c[0]]
            f.write(
                f"| {i+1} | {rep['pdb_id']} | {rep['resolution']:.2f} "
                f"| {rep['protein_name']} | {rep['uniprot']} | {rep['organism']} "
                f"| {rep['ligand_resname']} | {rep['ligand_num_heavy_atoms']} "
                f"| {', '.join(rep['coordinating_ligand_atoms_clean'])} | {len(c)} |\n"
            )
        f.write("\n## Cryo-EM Clusters (flagged, not counted in primary m)\n\n")
        f.write("| # | Rep PDB | Res (Å) | Protein | UniProt | Organism | Ligand | HA | Coord (Å) | Members |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, c in enumerate(cryo_clusters):
            rep = passed_targets[c[0]]
            f.write(
                f"| {i+1} | {rep['pdb_id']} | {rep['resolution']:.2f} "
                f"| {rep['protein_name']} | {rep['uniprot']} | {rep['organism']} "
                f"| {rep['ligand_resname']} | {rep['ligand_num_heavy_atoms']} "
                f"| {', '.join(rep['coordinating_ligand_atoms_clean'])} | {len(c)} |\n"
            )

    print("Report written to results/step1/external_zn_cluster_report.md")

    # ── Save updated payload ──────────────────────────────────────────────────
    surviving_representatives = [passed_targets[c[0]] for c in clusters]
    payload["targets"] = passed_targets
    payload["representative_targets"] = surviving_representatives
    payload["clusters"] = [[passed_targets[i]['pdb_id'] for i in c] for c in clusters]
    payload["m_clusters"] = len(clusters)
    payload["m_clusters_xray"] = len(xray_clusters)
    payload["m_clusters_cryo"] = len(cryo_clusters)
    payload["m_targets"] = len(passed_targets)
    payload["target_pdb_ids"] = [t['pdb_id'] for t in surviving_representatives]
    payload["provenance"]["criteria"] = [
        "Catalytic Zn site: >=2 protein sidechain donors within 2.8 A",
        "Ligand: non-polymer CCD type, not in metabolite/nucleotide blacklist",
        "Drug-likeness: >=12 heavy atoms, QED>0.1, MW<=1000 (RDKit)",
        "Coordination: >=1 donor atom (N/O/S/P) within 2.5 A of Zn",
        "min_coord_dist = minimum over coordinating-donor atoms (not all atoms)",
        "Structure: X-ray resolution <=2.5 A, or Cryo-EM (flagged separately)",
        "Not an active-site mutant (title keyword screen)",
        "<30% sequence identity to any CrossDocked training target (RCSB search)",
        "Sequence clusters merged by UniProt ID to avoid counting isoforms twice",
    ]
    payload["provenance"]["total_independent_clusters"] = len(clusters)
    payload["provenance"]["total_independent_clusters_xray"] = len(xray_clusters)

    torch.save(payload, "data/external_zn_test_clean.pt")
    print(f"Saved data/external_zn_test_clean.pt")
    print(f"\nFINAL: m_clusters={len(clusters)}  m_xray={len(xray_clusters)}  m_cryo={len(cryo_clusters)}  n_targets={len(passed_targets)}")


if __name__ == '__main__':
    main()
