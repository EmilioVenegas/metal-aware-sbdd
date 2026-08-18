"""Step 2 Arm C — first-layer surgery: expand the pocket vocabulary in the base checkpoint.

Builds a fresh LigandPocketDDPM with dataset='crossdock_metal' (pocket vocabulary widened from
10 to 16 element types — see DiffSBDD/constants.py and docs/step2.md §4), then copies over every
parameter from the base checkpoint that is unaffected by the vocabulary change. Parameters whose
shape changed (residue_encoder's two Linear layers — see docs/step2.md's note on why it is two
layers, not one) get their overlapping [old_rows, old_cols] block copied and the new rows/columns
left at the fresh model's own initialization.

Runs on CPU deliberately: this only needs a couple of full-model constructions and a state_dict
copy, no training, and Arm B is currently using the single GPU on this machine.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DiffSBDD"))
from lightning_modules import LigandPocketDDPM  # noqa: E402
from constants import dataset_params  # noqa: E402


def surgery_report(old_sd, new_sd):
    """Classify every parameter name in the new model relative to the old checkpoint."""
    identical, resized, new_only = [], [], []
    for name, new_t in new_sd.items():
        if name not in old_sd:
            new_only.append(name)
            continue
        old_t = old_sd[name]
        if tuple(old_t.shape) == tuple(new_t.shape):
            identical.append(name)
        else:
            resized.append((name, tuple(old_t.shape), tuple(new_t.shape)))
    missing_in_new = [n for n in old_sd if n not in new_sd]
    return identical, resized, new_only, missing_in_new


def copy_overlap(old_t: torch.Tensor, new_t: torch.Tensor) -> torch.Tensor:
    """Copy the overlapping top-left block of old_t into a clone of new_t, elementwise."""
    out = new_t.clone()
    if old_t.dim() == 1:
        n = min(old_t.shape[0], new_t.shape[0])
        out[:n] = old_t[:n]
    elif old_t.dim() == 2:
        r = min(old_t.shape[0], new_t.shape[0])
        c = min(old_t.shape[1], new_t.shape[1])
        out[:r, :c] = old_t[:r, :c]
    else:
        raise ValueError(f"Unexpected param rank {old_t.dim()} for shape {old_t.shape}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", default="checkpoints/crossdocked_fullatom_cond.ckpt")
    ap.add_argument("--out_checkpoint", default="checkpoints/arm_c_surgered_init.ckpt")
    ap.add_argument("--new_dataset", default="crossdock_metal")
    args = ap.parse_args()

    assert args.new_dataset in dataset_params, (
        f"{args.new_dataset} not found in DiffSBDD/constants.py dataset_params — "
        "check the vocabulary-expansion edit landed before running this script."
    )

    device = torch.device("cpu")
    print(f"Loading base checkpoint: {args.base_checkpoint}")
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    old_hparams = dict(ckpt["hyper_parameters"])
    old_sd = ckpt["state_dict"]

    print(f"Base dataset={old_hparams['dataset']!r} "
          f"pocket_representation={old_hparams['pocket_representation']!r}")
    old_aa_nf = len(dataset_params[old_hparams["dataset"]][
        "aa_encoder" if old_hparams["pocket_representation"] == "CA" else "atom_encoder"
    ])
    print(f"Old pocket vocabulary size: {old_aa_nf}")

    new_hparams = copy.deepcopy(old_hparams)
    new_hparams["dataset"] = args.new_dataset
    new_aa_nf = len(dataset_params[args.new_dataset]["aa_encoder"])
    print(f"New pocket vocabulary size: {new_aa_nf} "
          f"({new_aa_nf - old_aa_nf} new metal classes)")

    print("Constructing fresh model at the new vocabulary size (CPU, random init)...")
    new_model = LigandPocketDDPM(**new_hparams)
    new_sd = new_model.state_dict()

    identical, resized, new_only, missing_in_new = surgery_report(old_sd, new_sd)
    print(f"\nParameter census: {len(identical)} identical-shape (direct copy), "
          f"{len(resized)} resized (partial copy), {len(new_only)} new-only (kept fresh init), "
          f"{len(missing_in_new)} in old checkpoint but absent from new model.")

    if missing_in_new:
        print("WARNING — present in base checkpoint but not in the new model (dropped):")
        for n in missing_in_new:
            print(f"    {n}")

    print("\nResized parameters (old shape -> new shape):")
    for name, old_shape, new_shape in resized:
        print(f"    {name}: {old_shape} -> {new_shape}")

    merged_sd = dict(new_sd)
    for name in identical:
        merged_sd[name] = old_sd[name].clone()
    for name, _, _ in resized:
        merged_sd[name] = copy_overlap(old_sd[name], new_sd[name])
    # new_only entries: left as new_sd's own fresh initialization, already in merged_sd.

    new_model.load_state_dict(merged_sd, strict=True)

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": new_model.state_dict(),
        "hyper_parameters": new_hparams,
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "pytorch-lightning_version": ckpt.get("pytorch-lightning_version", "2.3.3"),
        "surgery_provenance": {
            "base_checkpoint": str(args.base_checkpoint),
            "old_dataset": old_hparams["dataset"],
            "new_dataset": args.new_dataset,
            "old_pocket_vocab_size": old_aa_nf,
            "new_pocket_vocab_size": new_aa_nf,
            "resized_params": [n for n, _, _ in resized],
        },
    }, out_path)
    print(f"\nSaved surgered checkpoint: {out_path}")


if __name__ == "__main__":
    main()
