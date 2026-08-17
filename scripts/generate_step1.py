#!/usr/bin/env python
"""Step 1 generation: N valid molecules per target, with full validity accounting.

Implements Amendment 4 of results/step1/ANALYSIS_PLAN.md:
  - N = 100 *valid* molecules per target (upstream DiffSBDD validity, unmodified)
  - per-target attempts / valid / validity-rate recorded
  - hard cap on attempts; short targets flagged, never silently dropped
  - deterministic per-target seed

Loads the model once and loops over targets, so the ~20 s checkpoint load is paid
once rather than per target.

Resumable: targets with a completed record in the manifest are skipped.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "DiffSBDD"))
RDLogger.DisableLog("rdApp.*")


def target_seed(pdb_id: str) -> int:
    return int(hashlib.sha256(pdb_id.encode()).hexdigest()[:8], 16) % 2**31


def structure_path(pdb_id: str, struct_dir: Path) -> Path:
    for ext in (".pdb", ".cif"):
        p = struct_dir / f"{pdb_id}{ext}"
        # guard against saved HTML error pages masquerading as structures
        if p.exists() and p.stat().st_size > 5000 and not p.read_bytes()[:1] == b"<":
            return p
    raise FileNotFoundError(f"no structure for {pdb_id} in {struct_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["external_zn", "crossdocked_zn"])
    ap.add_argument("--targets", required=True, help="torch .pt with target records")
    ap.add_argument("--struct-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints/crossdocked_fullatom_cond.ckpt"))
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sdf_dir = outdir / "sdf"; sdf_dir.mkdir(exist_ok=True)
    manifest_path = outdir / f"generation_manifest_shard{args.shard}.jsonl"

    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["pdb_id"])
        print(f"resuming: {len(done)} targets already complete", flush=True)

    blob = torch.load(args.targets, map_location="cpu", weights_only=False)
    records = blob["targets"] if isinstance(blob, dict) and "targets" in blob else blob
    if args.num_shards > 1:
        records = [r for j, r in enumerate(records) if j % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(records)} targets", flush=True)

    from lightning_modules import LigandPocketDDPM
    t0 = time.time()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LigandPocketDDPM(**dict(ck["hyper_parameters"]))
    model.load_state_dict(ck["state_dict"])
    model = model.cuda().eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    struct_dir = Path(args.struct_dir)
    for i, rec in enumerate(records):
        pdb_id = rec["pdb_id"]
        if pdb_id in done:
            continue

        # ligand_id looks like "A_A1C39_303" -> ref_ligand "A:303"
        chain, _, resi = rec["ligand_id"].split("_")
        ref_ligand = f"{chain}:{resi}"

        seed = target_seed(pdb_id)
        torch.manual_seed(seed)

        t_start = time.time()
        valid, attempts, failures = [], 0, []
        try:
            path = structure_path(pdb_id, struct_dir)
            while len(valid) < args.n_valid and attempts < args.max_attempts:
                want = min(args.batch_size, args.max_attempts - attempts)
                mols = model.generate_ligands(
                    str(path), n_samples=want, ref_ligand=ref_ligand,
                    num_nodes_lig=None, sanitize=True,
                )
                attempts += want
                for m in mols:
                    if m is None:
                        failures.append("build_or_sanitize_failed")
                    else:
                        valid.append(m)
            status = "complete" if len(valid) >= args.n_valid else "under_cap"
            err = None
        except Exception as e:  # noqa: BLE001 - record, never abort the run
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
            "attempts": attempts, "n_valid": len(valid),
            "validity_rate": round(rate, 4),
            "n_written": min(len(valid), args.n_valid),
            "zn_coord": rec.get("zn_coord"),
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
