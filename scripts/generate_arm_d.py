#!/usr/bin/env python
"""Step 2 Arm D generation: Inference-time coordination-constrained generation.

Arm D tests whether inference-time geometric conditioning (masked in-painting of an
ideal catalytic donor seed) on top of the metal-aware model (Arm C) resolves the
distance calibration dilemma without retraining.

Implements:
  - Open coordination vector calculation from catalytic protein sidechain donors
  - Seed donor atom placement at ideal distance (2.05 A for Zn-O/N)
  - Masked in-painting via DiffSBDD's RePaint algorithm (model.ddpm.inpaint)
  - Full validity accounting, per-target attempt caps, deterministic seeding
  - Multi-shard resume support matching Step 1 / Arm C generation harness
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_scatter import scatter_mean

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "DiffSBDD"))
sys.path.insert(0, str(REPO / "scripts"))

import utils
from constants import FLOAT_TYPE
from analysis.molecule_builder import build_molecule, process_molecule
from lora import apply_lora
from lightning_modules import LigandPocketDDPM
from generate_step1 import target_seed, structure_path

RDLogger.DisableLog("rdApp.*")


def compute_open_coordination_seed(zn_coord: np.ndarray, protein_donors: list[dict],
                                   donor_dist: float = 2.05) -> np.ndarray:
    """Calculate ideal donor seed position along open coordination sphere."""
    zn = np.array(zn_coord, dtype=float)
    if not protein_donors:
        # Fallback if no protein donors: default offset along +x
        return zn + np.array([donor_dist, 0.0, 0.0])

    vectors = []
    for d in protein_donors:
        xyz = np.array(d["xyz"], dtype=float)
        v = xyz - zn
        norm = np.linalg.norm(v)
        if norm > 1e-4:
            vectors.append(v / norm)

    if vectors:
        v_sum = np.sum(vectors, axis=0)
        norm_sum = np.linalg.norm(v_sum)
        if norm_sum > 0.1:
            u_open = - v_sum / norm_sum
        else:
            u_open = np.array([1.0, 0.0, 0.0])
    else:
        u_open = np.array([1.0, 0.0, 0.0])

    return zn + donor_dist * u_open


def compute_random_vector_seed(zn_coord: np.ndarray, donor_dist: float = 2.05,
                               rng: np.random.Generator | None = None) -> np.ndarray:
    """Control S1: donor seed at the SAME distance, in a uniformly random direction.

    The pre-registered control that could detect the positive. The checker's
    valid-coordination endpoint is satisfied by any donor inside COORD_RANGES, so a
    random direction scores identically to the open coordination vector (verified: both
    100% on the 9ZSN pilot). Endpoints where Arm D does NOT separate from this control
    are measuring seed-placement arithmetic, not the model. See
    results/step2/ANALYSIS_PLAN_ARMD.md section 5.
    """
    rng = np.random.default_rng() if rng is None else rng
    zn = np.array(zn_coord, dtype=float)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return zn + donor_dist * v


def inpaint_target_batch(model: LigandPocketDDPM, residues, seed_coord: np.ndarray,
                         donor_element: str, n_samples: int, resamplings: int = 1):
    """Generate a batch of inpainted ligands given pocket residues and fixed seed."""
    pocket = model.prepare_pocket(residues, repeats=n_samples)
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    n_fixed = 1
    seed_tensor = torch.tensor(seed_coord, dtype=FLOAT_TYPE, device=model.device).unsqueeze(0)
    seed_one_hot = torch.zeros((1, model.atom_nf), dtype=FLOAT_TYPE, device=model.device)
    element_idx = model.dataset_info['atom_encoder'][donor_element]
    seed_one_hot[0, element_idx] = 1.0

    num_nodes_lig = model.ddpm.size_distribution.sample_conditional(n1=None, n2=pocket['size'])
    num_nodes_lig = torch.clamp(num_nodes_lig, min=n_fixed + 8)

    ligand_mask = utils.num_nodes_to_batch_mask(len(num_nodes_lig), num_nodes_lig, model.device)
    ligand = {
        'x': torch.zeros((len(ligand_mask), model.x_dims), device=model.device, dtype=FLOAT_TYPE),
        'one_hot': torch.zeros((len(ligand_mask), model.atom_nf), device=model.device, dtype=FLOAT_TYPE),
        'size': num_nodes_lig,
        'mask': ligand_mask
    }

    lig_fixed = torch.zeros_like(ligand_mask)
    for i in range(n_samples):
        sele = (ligand_mask == i)
        x_new = ligand['x'][sele]
        x_new[:n_fixed] = seed_tensor
        ligand['x'][sele] = x_new

        h_new = ligand['one_hot'][sele]
        h_new[:n_fixed] = seed_one_hot
        ligand['one_hot'][sele] = h_new

        fixed_new = lig_fixed[sele]
        fixed_new[:n_fixed] = 1
        lig_fixed[sele] = fixed_new

    xh_lig, xh_pocket, lig_mask_out, pocket_mask_out = model.ddpm.inpaint(
        ligand, pocket, lig_fixed, center='ligand', resamplings=resamplings, timesteps=None
    )

    pocket_com_after = scatter_mean(xh_pocket[:, :model.x_dims], pocket_mask_out, dim=0)
    xh_pocket[:, :model.x_dims] += (pocket_com_before - pocket_com_after)[pocket_mask_out]
    xh_lig[:, :model.x_dims] += (pocket_com_before - pocket_com_after)[lig_mask_out]

    x = xh_lig[:, :model.x_dims].detach().cpu()
    atom_type = xh_lig[:, model.x_dims:].argmax(1).detach().cpu()
    lig_mask_cpu = lig_mask_out.cpu()

    molecules = []
    for mol_pc in zip(utils.batch_to_list(x, lig_mask_cpu), utils.batch_to_list(atom_type, lig_mask_cpu)):
        mol = build_molecule(*mol_pc, model.dataset_info, add_coords=True)
        mol = process_molecule(mol, add_hydrogens=False, sanitize=True, relax_iter=0, largest_frag=False)
        if mol is not None:
            molecules.append(mol)

    return molecules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["external_zn", "crossdocked_zn"])
    ap.add_argument("--targets", required=True, help="torch .pt with target records")
    ap.add_argument("--struct-dir", required=True)
    ap.add_argument("--protein-donors", default=str(REPO / "data/protein_donors.json"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints/arm_c_best.ckpt"))
    ap.add_argument("--donor-element", default="O", choices=["O", "N", "S"])
    ap.add_argument("--donor-dist", type=float, default=2.05)
    ap.add_argument("--resamplings", type=int, default=1)
    ap.add_argument("--seed-mode", default="open", choices=["open", "random"],
                    help="open = ideal open-coordination vector (Arm D); "
                         "random = control S1, same distance, random direction")
    ap.add_argument("--cluster-cap", type=int, default=0,
                    help="max targets per sequence cluster (0 = all). Cluster-stratified "
                         "subsample; cluster is the resampling unit, so this preserves m "
                         "while cutting cost. See ANALYSIS_PLAN_ARMD.md section 5.")
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sdf_dir = outdir / "sdf"
    sdf_dir.mkdir(exist_ok=True)
    manifest_path = outdir / f"generation_manifest_shard{args.shard}.jsonl"

    done = set()
    for mf in sorted(outdir.glob("generation_manifest_shard*.jsonl")):
        for line in mf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") in ("complete", "under_cap"):
                    done.add(rec["pdb_id"])
            except Exception:
                continue
    if done:
        print(f"resuming: {len(done)} targets already complete", flush=True)

    with open(args.protein_donors) as f:
        donors_data = json.load(f)

    blob = torch.load(args.targets, map_location="cpu", weights_only=False)
    records = blob["targets"] if isinstance(blob, dict) and "targets" in blob else blob

    if args.cluster_cap:
        # Cluster-stratified subsample. The registered analysis weights every cluster
        # EQUALLY (cluster is the resampling unit), but the cohort is severely skewed:
        # C07 holds 50 of 127 X-ray targets and C03 holds 32, while 11 clusters hold one.
        # Deep sampling inside the two mega-clusters therefore buys almost no inferential
        # precision. Measured on the published Arm C vs Arm A contrast, capping at 3
        # targets/cluster moves the MDE from 4.59 pp to 4.70 pp while cutting GPU cost
        # ~3.5x. Selection is by cohort-file order within each cluster -- deterministic,
        # identical to the order generation already walks, and independent of any result,
        # so it also reuses whatever a sequential run already completed.
        clusters = blob["clusters"]
        pdb_to_cluster = {p: f"C{i+1:02d}"
                          for i, mem in enumerate(clusters) for p in mem}
        seen, capped = {}, []
        for r in records:
            cid = pdb_to_cluster.get(r["pdb_id"])
            if cid is None:
                continue
            if seen.get(cid, 0) < args.cluster_cap:
                seen[cid] = seen.get(cid, 0) + 1
                capped.append(r)
        records = capped
        print(f"cluster-cap {args.cluster_cap}: {len(records)} targets "
              f"across {len(seen)} clusters", flush=True)

    if args.num_shards > 1:
        records = [r for j, r in enumerate(records) if j % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(records)} targets", flush=True)

    t0 = time.time()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LigandPocketDDPM(**dict(ck["hyper_parameters"]))
    if any("lora_A" in k for k in ck["state_dict"]):
        apply_lora(model)
    model.load_state_dict(ck["state_dict"])
    model = model.cuda().eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    struct_dir = Path(args.struct_dir)
    for i, rec in enumerate(records):
        pdb_id = rec["pdb_id"]
        if pdb_id in done:
            continue

        chain, _, resi = rec["ligand_id"].split("_")
        ref_ligand = f"{chain}:{resi}"

        seed = target_seed(pdb_id)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))

        t_start = time.time()
        valid, attempts, failures = [], 0, []
        # Initialized before the try: an exception raised before these are assigned must
        # still produce an error row in the manifest, not a NameError that kills the run.
        zn_xyz, seed_coord = None, None
        try:
            path = structure_path(pdb_id, struct_dir)
            if str(path).endswith(".cif"):
                from Bio.PDB.MMCIFParser import MMCIFParser
                pdb_struct = MMCIFParser(QUIET=True).get_structure("", str(path))[0]
            else:
                from Bio.PDB import PDBParser
                pdb_struct = PDBParser(QUIET=True).get_structure("", str(path))[0]

            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

            zn_info = donors_data.get(pdb_id, {})
            zn_xyz = zn_info.get("zn", rec.get("zn_coord"))
            p_donors = zn_info.get("protein_donors", [])
            if args.seed_mode == "random":
                seed_coord = compute_random_vector_seed(
                    zn_xyz, donor_dist=args.donor_dist,
                    rng=np.random.default_rng(seed))
            else:
                seed_coord = compute_open_coordination_seed(
                    zn_xyz, p_donors, donor_dist=args.donor_dist)

            while len(valid) < args.n_valid and attempts < args.max_attempts:
                want = min(args.batch_size, args.max_attempts - attempts)
                mols = inpaint_target_batch(
                    model, residues, seed_coord, donor_element=args.donor_element,
                    n_samples=want, resamplings=args.resamplings
                )
                attempts += want
                for m in mols:
                    if m is None:
                        failures.append("build_or_sanitize_failed")
                    else:
                        valid.append(m)

            status = "complete" if len(valid) >= args.n_valid else "under_cap"
            err = None
        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"

        if valid:
            w = Chem.SDWriter(str(sdf_dir / f"{pdb_id}.sdf"))
            for m in valid[: args.n_valid]:
                w.write(m)
            w.close()

        rate = len(valid) / attempts if attempts else 0.0
        entry = {
            "pdb_id": pdb_id, "cohort": args.cohort, "status": status,
            "seed": seed, "ref_ligand": ref_ligand,
            "donor_element": args.donor_element,
            "donor_dist": args.donor_dist,
            "resamplings": args.resamplings,
            "attempts": attempts, "n_valid": len(valid),
            "validity_rate": round(rate, 4),
            "n_written": min(len(valid), args.n_valid),
            "seed_mode": args.seed_mode,
            "seed_xyz": list(map(float, seed_coord)) if seed_coord is not None else None,
            "zn_coord": list(map(float, zn_xyz)) if zn_xyz is not None else None,
            "elapsed_s": round(time.time() - t_start, 1),
            "error": err,
            "n_invalid": len(failures),
        }
        with open(manifest_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{i+1}/{len(records)}] {pdb_id} {status} "
              f"valid={len(valid)}/{attempts} rate={rate:.2f} "
              f"{entry['elapsed_s']}s", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
