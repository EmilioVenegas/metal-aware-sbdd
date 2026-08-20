#!/usr/bin/env python
"""Step 3 cross-model generation: TargetDiff and Pocket2Mol on the external Zn cohort.

Measures whether metal-blindness is field-wide or DiffSBDD-specific, using released
checkpoints and inference only. See docs/step3.md.

DESIGN NOTE -- deliberately different from Step 2. Arms A-D shared one pocket definition
so the intervention was not confounded with pocket radius. Here every model uses its OWN
published preprocessing (TargetDiff: 10 A ligand-centred pocket; Pocket2Mol: 23 A
bounding box), because the object of measurement is what the published pipeline actually
does to a metalloprotein. Substituting a foreign pocket definition would measure a system
nobody runs. Consequence: cross-model numbers are independent measurements on shared
targets, NOT paired contrasts against Arm A, and each model is compared to the native
ceiling rather than to the other models.

Each model is invoked through its own interpreter, mirroring the pattern
run_arm_c_analysis.py already uses to reach statsmodels in the `ifp` env. Nothing is
imported across environments.

SAMPLER CONTRACTS -- read from upstream source 2026-08-20, not assumed. Both samplers take
their checkpoint path and their seed from a YAML config, NOT from the command line, so this
script writes one config per target and keeps it as provenance:

  targetdiff  scripts/sample_for_pocket.py -- `config` is POSITIONAL; flags are --pdb_path,
              --result_path, --num_samples, --batch_size, --device. Reads
              config.model.checkpoint and seeds with misc.seed_all(config.sample.seed).
              Expects an ALREADY-CLIPPED pocket PDB: pdb_to_pocket_data calls
              PDBProtein(pdb_path) directly and performs no clipping of its own.
              Writes <result_path>/sdf/<idx:03d>.sdf, one molecule per file, skipping
              molecules that failed reconstruction, plus sample.pt and sample.yml.
  pocket2mol  sample_for_pdb.py -- flags are --pdb_path, --center "x,y,z", --bbox_size,
              --config, --outdir, --device. There is NO --num_samples flag: the sampler
              runs until len(pool.finished) >= config.sample.num_samples or
              config.sample.max_steps is exceeded. It crops the pocket itself, per atom,
              against --center/--bbox_size. Writes
              <outdir>/<cfgname>_<pdbname>_<timestamp>/SDF/<i>.sdf plus samples_*.pt.

Sampling hyperparameters in the generated configs are copied verbatim from each model's
published config (targetdiff configs/sampling.yml, Pocket2Mol configs/sample_for_pdb.yml);
only the checkpoint path, the seed and the sample count are substituted. --batch-size is a
pure VRAM knob for TargetDiff (samples are independent) and is the one value allowed to
differ from upstream, because upstream's default of 100 does not fit 8 GB.

Output is written in the exact layout scripts/coordination_checker.py already consumes:
  <outdir>/sdf/<PDB>.sdf                        one multi-molecule SDF per target
  <outdir>/generation_manifest_shard0.jsonl     one record per target
so scoring needs no new machinery.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from generate_step1 import target_seed, structure_path

RDLogger.DisableLog("rdApp.*")

# Each model's native pocket convention and sampler contract, taken from its own source.
MODELS: Dict[str, Dict[str, Any]] = {
    "targetdiff": {
        # scripts/sample_for_pocket.py does NOT clip; the pocket PDB must arrive clipped.
        # 10 A around the reference ligand is scripts/data_preparation/extract_pockets.py.
        "pocket_mode": "ligand_radius",
        "radius": 10.0,
        "repo_env": "targetdiff",
        "sampler": "scripts/sample_for_pocket.py",
        "ckpt": "checkpoints/crossmodel/targetdiff/pretrained_diffusion.pt",
    },
    "pocket2mol": {
        # sample_for_pdb.py takes a bounding-box centre and side length (default 23 A) and
        # applies the box itself; we pre-clip only to normalise mmCIF -> PDB and drop H,
        # which reproduces rather than alters its per-atom filter.
        "pocket_mode": "box",
        "box_size": 23.0,
        "repo_env": "pocket2mol",
        "sampler": "sample_for_pdb.py",
        "ckpt": "checkpoints/crossmodel/pocket2mol/pretrained_Pocket2Mol.pt",
    },
}


def capped_targets(blob, cap: int) -> List[Dict[str, Any]]:
    """Amendment 1 cluster-stratified selection, shared with Arm D so all arms
    (B/C/D and Step 3) are scored on one target set."""
    records = blob["targets"]
    if not cap:
        return records
    clusters = blob["clusters"]
    p2c = {p: f"C{i+1:02d}" for i, mem in enumerate(clusters) for p in mem}
    seen: Dict[str, int] = {}
    out = []
    for r in records:
        c = p2c.get(r["pdb_id"])
        if c is None:
            continue
        if seen.get(c, 0) < cap:
            seen[c] = seen.get(c, 0) + 1
            out.append(r)
    return out


def write_pocket_pdb(struct_path: Path, ref_ligand: str, out_pdb: Path,
                     mode: str, radius: float = 10.0) -> Dict[str, Any]:
    """Extract a pocket PDB in the requested convention.

    Returns the ligand centroid, which Pocket2Mol needs as its box centre.

    IMPORTANT: this writes ATOM records for standard residues only, reproducing each
    upstream pipeline's own behaviour. That is the point of the experiment -- we are
    measuring the published preprocessing, not fixing it. The metal is deliberately NOT
    retained here; the checker supplies the true metal position independently from
    data/protein_donors.json.
    """
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa

    parser = MMCIFParser(QUIET=True) if str(struct_path).endswith(".cif") \
        else PDBParser(QUIET=True)
    model = parser.get_structure("", str(struct_path))[0]

    chain_id, resi = ref_ligand.split(":")
    lig_atoms = None
    for res in model[chain_id]:
        if str(res.id[1]) == str(resi) and not is_aa(res.get_resname(), standard=True):
            lig_atoms = [a for a in res if a.element != "H"]
            break
    if lig_atoms is None:
        raise ValueError(f"reference ligand {ref_ligand} not found in {struct_path}")
    lig_xyz = np.array([a.get_coord() for a in lig_atoms], dtype=float)
    centroid = lig_xyz.mean(axis=0)

    keep = set()
    for chain in model:
        for res in chain:
            if not is_aa(res.get_resname(), standard=True):
                continue
            coords = np.array([a.get_coord() for a in res if a.element != "H"], dtype=float)
            if len(coords) == 0:
                continue
            if mode == "ligand_radius":
                d = np.linalg.norm(coords[:, None, :] - lig_xyz[None, :, :], axis=-1)
                if d.min() < radius:
                    keep.add((chain.id, res.id))
            else:  # box
                half = radius / 2.0
                if np.any(np.all(np.abs(coords - centroid) <= half, axis=1)):
                    keep.add((chain.id, res.id))

    # NB: PDBIO.set_structure() COPIES and re-parents a Model, which rewrites every
    # get_full_id() (the structure id changes), so full_id can never be used to match
    # residues across the save. Chain id + residue id survive the copy.
    class _Sel(Select):
        def accept_residue(self, residue):
            return (residue.get_parent().id, residue.id) in keep

        def accept_atom(self, atom):
            return atom.element != "H"

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(str(out_pdb), select=_Sel())
    return {"centroid": centroid.tolist(), "n_residues": len(keep)}


def write_sample_config(model: str, ckpt: Path, seed: int, n_samples: int,
                        out_yml: Path) -> Path:
    """Both samplers read the checkpoint and the RNG seed from YAML, not from argv.

    Every other value is copied verbatim from the model's published sampling config, so
    the only substitutions are checkpoint path, seed and sample count. Written as plain
    text (YAML is not imported) and kept on disk as run provenance.
    """
    out_yml.parent.mkdir(parents=True, exist_ok=True)
    if model == "targetdiff":
        # targetdiff configs/sampling.yml
        text = (
            "model:\n"
            f"  checkpoint: {ckpt}\n"
            "\n"
            "sample:\n"
            f"  seed: {seed}\n"
            f"  num_samples: {n_samples}\n"
            "  num_steps: 1000\n"
            "  pos_only: False\n"
            "  center_pos_mode: protein\n"
            "  sample_num_atoms: prior\n"
        )
    else:
        # Pocket2Mol configs/sample_for_pdb.yml
        text = (
            "model:\n"
            f"  checkpoint: {ckpt}\n"
            "\n"
            "sample:\n"
            f"  seed: {seed}\n"
            f"  num_samples: {n_samples}\n"
            "  beam_size: 300\n"
            "  max_steps: 50\n"
            "  threshold:\n"
            "    focal_threshold: 0.5\n"
            "    pos_threshold: 0.25\n"
            "    element_threshold: 0.3\n"
            "    hasatom_threshold: 0.6\n"
            "    bond_threshold: 0.4\n"
        )
    out_yml.write_text(text)
    return out_yml


def preflight(model: str, repo_root: Path, python_bin: Path) -> None:
    """Fail loudly and specifically rather than producing empty output."""
    spec = MODELS[model]
    problems = []
    if not repo_root.exists():
        problems.append(
            f"repo not found: {repo_root}\n"
            f"    git clone https://github.com/"
            f"{'guanjq/targetdiff' if model == 'targetdiff' else 'pengxingang/Pocket2Mol'}"
            f" {repo_root}")
    elif not (repo_root / spec["sampler"]).exists():
        problems.append(f"sampler missing: {repo_root / spec['sampler']}")
    if not python_bin.exists():
        problems.append(
            f"interpreter not found: {python_bin}\n"
            f"    create the '{spec['repo_env']}' env per docs/step3.md section 5")
    if not (REPO / spec["ckpt"]).exists():
        problems.append(f"checkpoint missing: {REPO / spec['ckpt']}")
    if problems:
        raise SystemExit(f"[{model}] preflight failed:\n  - " + "\n  - ".join(problems))


def sample_target(model: str, repo_root: Path, python_bin: Path, pocket_pdb: Path,
                  centroid: List[float], cfg: Path, workdir: Path, device: str,
                  batch_size: int, n_samples: int,
                  bbox_size: float) -> tuple[List[Chem.Mol], str]:
    """Invoke the model's own sampler in its own interpreter; collect SDF output."""
    spec = MODELS[model]
    workdir.mkdir(parents=True, exist_ok=True)

    if model == "targetdiff":
        # `config` is positional; --num_samples overrides config.sample.num_samples.
        cmd = [str(python_bin), spec["sampler"], str(cfg.resolve()),
               "--pdb_path", str(pocket_pdb.resolve()),
               "--result_path", str(workdir.resolve()),
               "--num_samples", str(n_samples),
               "--batch_size", str(batch_size),
               "--device", device]
    else:
        # No --num_samples: the count lives in the config. --center is parsed as
        # "x,y,z" by a lambda, so no whitespace and no brackets.
        cx, cy, cz = centroid
        cmd = [str(python_bin), spec["sampler"],
               "--pdb_path", str(pocket_pdb.resolve()),
               "--center", f"{cx:.3f},{cy:.3f},{cz:.3f}",
               "--bbox_size", f"{bbox_size}",
               "--config", str(cfg.resolve()),
               "--outdir", str(workdir.resolve()),
               "--device", device]

    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"{model} sampler exited {proc.returncode}:\n{tail}")

    mols: List[Chem.Mol] = []
    for sdf in sorted(workdir.rglob("*.sdf")):
        for m in Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True):
            if m is not None and m.GetNumConformers() > 0:
                mols.append(m)
    return mols, " ".join(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--repo-root", required=True,
                    help="clone of the model's upstream repository")
    ap.add_argument("--python-bin", required=True,
                    help="interpreter of that model's conda env")
    ap.add_argument("--targets", default="data/external_zn_test_clean.pt")
    ap.add_argument("--struct-dir", default="data/external_pdbs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cluster-cap", type=int, default=3,
                    help="Amendment 1 design: 3 targets/cluster (0 = full cohort)")
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=20,
                    help="TargetDiff VRAM knob; samples are independent so this does not "
                         "change the sampled distribution (upstream default 100 OOMs on 8 GB)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default=None,
                    help="comma-separated PDB ids; smoke-test a subset of the cohort")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser()
    preflight(args.model, repo_root, python_bin)

    spec = MODELS[args.model]
    ckpt = (REPO / spec["ckpt"]).resolve()
    device = args.device if args.model == "targetdiff" else args.device.split(":")[0]
    bbox = float(spec.get("box_size", 0.0))

    outdir = Path(args.outdir)
    sdf_dir = outdir / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    pocket_dir = outdir / "pockets"
    cfg_dir = outdir / "configs"
    manifest = outdir / "generation_manifest_shard0.jsonl"

    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") in ("complete", "under_cap"):
                    done.add(r["pdb_id"])
        if done:
            print(f"resuming: {len(done)} targets already complete", flush=True)

    blob = torch.load(args.targets, map_location="cpu", weights_only=False)
    records = capped_targets(blob, args.cluster_cap)
    if args.only:
        wanted = {p.strip() for p in args.only.split(",") if p.strip()}
        records = [r for r in records if r["pdb_id"] in wanted]
        missing = wanted - {r["pdb_id"] for r in records}
        if missing:
            raise SystemExit(f"--only ids not in cohort: {sorted(missing)}")
    print(f"{args.model}: {len(records)} targets "
          f"(cluster-cap {args.cluster_cap or 'off'})", flush=True)

    struct_dir = Path(args.struct_dir)
    for i, rec in enumerate(records):
        pdb_id = rec["pdb_id"]
        if pdb_id in done:
            continue
        chain, _, resi = rec["ligand_id"].split("_")
        ref_ligand = f"{chain}:{resi}"
        seed = target_seed(pdb_id)
        t0 = time.time()

        valid: List[Chem.Mol] = []
        attempts = 0
        status, err, pinfo, cmdline = "complete", None, {}, None
        try:
            path = structure_path(pdb_id, struct_dir)
            pocket_pdb = pocket_dir / f"{pdb_id}_pocket.pdb"
            pinfo = write_pocket_pdb(
                Path(path), ref_ligand, pocket_pdb,
                mode=spec["pocket_mode"],
                radius=spec.get("radius", spec.get("box_size", 10.0)))

            with tempfile.TemporaryDirectory() as td:
                while len(valid) < args.n_valid and attempts < args.max_attempts:
                    want = min(args.n_valid - len(valid), args.max_attempts - attempts)
                    cfg = write_sample_config(
                        args.model, ckpt, seed + attempts, want,
                        cfg_dir / f"{pdb_id}_a{attempts}.yml")
                    got, cmdline = sample_target(
                        args.model, repo_root, python_bin, pocket_pdb,
                        pinfo["centroid"], cfg, Path(td) / f"a{attempts}",
                        device, args.batch_size, want, bbox)
                    attempts += want
                    valid.extend(got)
                    if not got:
                        break  # sampler yielding nothing: record, do not spin
            if len(valid) < args.n_valid:
                status = "under_cap"
        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"

        if valid:
            w = Chem.SDWriter(str(sdf_dir / f"{pdb_id}.sdf"))
            for m in valid[: args.n_valid]:
                w.write(m)
            w.close()

        entry = {
            "pdb_id": pdb_id, "model": args.model, "status": status,
            "seed": seed, "ref_ligand": ref_ligand,
            "pocket_mode": spec["pocket_mode"],
            "pocket_residues": pinfo.get("n_residues"),
            "attempts": attempts, "n_valid": len(valid),
            "validity_rate": round(len(valid) / attempts, 4) if attempts else 0.0,
            "n_written": min(len(valid), args.n_valid),
            "zn_coord": rec.get("zn_coord"),
            "elapsed_s": round(time.time() - t0, 1),
            "cmd": cmdline,
            "error": err,
        }
        with open(manifest, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{i+1}/{len(records)}] {pdb_id} {status} "
              f"valid={len(valid)}/{attempts} {entry['elapsed_s']}s"
              + (f" ERR {err}" if err else ""), flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
