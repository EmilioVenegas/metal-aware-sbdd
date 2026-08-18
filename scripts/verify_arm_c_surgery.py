"""Step 2 Arm C — mandatory numerical-identity check on the surgered checkpoint.

Confirms that on a metal-free pocket, checkpoints/arm_c_surgered_init.ckpt produces the same
ligand-side dynamics output as the unmodified base checkpoint. This is the check docs/step2.md
§4 flags as "cheap and decisive — must actually be run, not just planned."

Calls EGNNDynamics.forward directly (bypassing the stochastic diffusion noising/sampling code)
so the comparison isolates exactly what the surgery could have broken: the residue_encoder MLP
that turns pocket one-hot vectors into embeddings. residue_decoder is deliberately not part of
this check — verified separately (see the script's closing note) that ConditionalDDPM discards
its output, so it cannot affect this checkpoint's behavior regardless of its weights.

Runs on CPU, deliberately, to avoid contending with Arm B training on the single GPU.
"""

import sys
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DiffSBDD"))
from lightning_modules import LigandPocketDDPM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CKPT = REPO_ROOT / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
SURGERED_CKPT = REPO_ROOT / "checkpoints" / "arm_c_surgered_init.ckpt"
TOL = 1e-5


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LigandPocketDDPM(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def metal_free_residues(pdb_path, max_residues=25):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    residues = []
    for residue in structure[0].get_residues():
        if is_aa(residue.get_resname(), standard=True):
            residues.append(residue)
        if len(residues) >= max_residues:
            break
    if len(residues) < 5:
        raise RuntimeError(f"Too few standard-AA residues found in {pdb_path}")
    return residues


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    print(f"Loading base checkpoint: {BASE_CKPT}")
    base_model = load_model(BASE_CKPT, device)
    print(f"Loading surgered checkpoint: {SURGERED_CKPT}")
    surgered_model = load_model(SURGERED_CKPT, device)

    assert base_model.aa_nf == 10, f"expected base aa_nf=10, got {base_model.aa_nf}"
    assert surgered_model.aa_nf == 16, f"expected surgered aa_nf=16, got {surgered_model.aa_nf}"

    ext_pdb_dir = REPO_ROOT / "data" / "external_pdbs"
    pdb_files = sorted(ext_pdb_dir.glob("*.pdb"))
    assert pdb_files, f"no PDB files found under {ext_pdb_dir}"
    residues = metal_free_residues(pdb_files[0])
    print(f"Using {len(residues)} metal-free standard-AA residues from {pdb_files[0].name}")

    # Real pocket, encoded twice: once through each model's own pocket_type_encoder.
    # base uses dataset_params['crossdock']['atom_encoder'] (10-wide, no metal classes);
    # surgered uses dataset_params['crossdock_metal']['aa_encoder'] (16-wide, metal classes
    # appended after the same first 10). Since these residues are all standard amino acids,
    # every atom hits the shared first-10 classes in both — the two encodings should describe
    # the exact same physical pocket, differing only in how many trailing zero columns the
    # one-hot vectors carry.
    base_pocket = base_model.prepare_pocket(residues, repeats=1)
    surgered_pocket = surgered_model.prepare_pocket(residues, repeats=1)

    assert base_pocket["one_hot"].shape[1] == 10
    assert surgered_pocket["one_hot"].shape[1] == 16
    assert torch.equal(base_pocket["one_hot"][:, :10].float(),
                        surgered_pocket["one_hot"][:, :10].float()), \
        "pocket atom typing disagrees between base and surgered encoders on the shared classes"
    assert surgered_pocket["one_hot"][:, 10:].sum() == 0, \
        "metal-free pocket should have all-zero metal columns in the surgered encoding"
    assert torch.equal(base_pocket["x"], surgered_pocket["x"])
    print("Pocket encodings agree on the shared 10 classes; metal columns are zero. Good.")

    # Synthetic ligand: identical for both models, shape/content irrelevant beyond being valid.
    n_lig = 12
    lig_x = torch.randn(n_lig, 3)
    lig_types = torch.randint(0, 10, (n_lig,))
    lig_one_hot = torch.nn.functional.one_hot(lig_types, num_classes=10).float()
    lig_mask = torch.zeros(n_lig, dtype=torch.long)
    pocket_mask = torch.zeros(base_pocket["x"].shape[0], dtype=torch.long)
    t = torch.zeros(1)

    xh_atoms = torch.cat([lig_x, lig_one_hot], dim=1)
    xh_res_base = torch.cat([base_pocket["x"], base_pocket["one_hot"].float()], dim=1)
    xh_res_surgered = torch.cat([surgered_pocket["x"], surgered_pocket["one_hot"].float()], dim=1)

    with torch.no_grad():
        out_lig_base, _ = base_model.ddpm.dynamics(
            xh_atoms, xh_res_base, t, lig_mask, pocket_mask)
        out_lig_surgered, _ = surgered_model.ddpm.dynamics(
            xh_atoms, xh_res_surgered, t, lig_mask, pocket_mask)

    max_abs_diff = (out_lig_base - out_lig_surgered).abs().max().item()
    print(f"\nLigand-side dynamics output — max abs diff (base vs surgered): {max_abs_diff:.3e}")
    print(f"Tolerance: {TOL:.1e}")

    if max_abs_diff < TOL:
        print("PASS — surgered checkpoint is numerically identical to the base checkpoint "
              "on a metal-free pocket.")
    else:
        print("FAIL — surgery changed model behavior on a metal-free pocket. "
              "Do not proceed to Arm C training until this is understood.")
        sys.exit(1)


if __name__ == "__main__":
    main()
