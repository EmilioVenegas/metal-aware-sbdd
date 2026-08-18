"""Execute Arm B Full Fine-Tuning for Metal Coordination Ablation Study.

Fine-tunes the base DiffSBDD checkpoint on the metalloprotein training partition
using the existing metal-blind pocket representation.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure DiffSBDD is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DiffSBDD")))
from lightning_modules import LigandPocketDDPM


class MetalloproteinLMDBDataset(Dataset):
    """Loads pocket-ligand complexes directly from processed LMDB."""

    def __init__(self, lmdb_path: str, indices: list, center: bool = True):
        self.lmdb_path = str(lmdb_path)
        self.indices = list(indices)
        self.center = center
        self.env = None
        self.z_to_idx = {6: 0, 7: 1, 8: 2, 16: 3, 5: 4, 35: 5, 17: 6, 15: 7, 53: 8, 9: 9}

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                max_readers=32,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._init_env()
        idx = self.indices[i]
        with self.env.begin() as txn:
            v = txn.get(str(idx).encode("ascii"))
            if v is None:
                raise KeyError(f"LMDB index {idx} not found")
            rec = pickle.loads(v)

        lig_pos = rec["ligand_pos"].float()
        lig_z = rec["ligand_element"].tolist()
        lig_one_hot = torch.zeros(len(lig_z), 10)
        for idx_a, z in enumerate(lig_z):
            if z in self.z_to_idx:
                lig_one_hot[idx_a, self.z_to_idx[z]] = 1.0

        prot_pos = rec["protein_pos"].float()
        prot_z = rec["protein_element"].tolist()
        prot_one_hot = torch.zeros(len(prot_z), 10)
        for idx_a, z in enumerate(prot_z):
            if z in self.z_to_idx:
                prot_one_hot[idx_a, self.z_to_idx[z]] = 1.0

        if self.center:
            mean = (lig_pos.sum(0) + prot_pos.sum(0)) / (len(lig_pos) + len(prot_pos))
            lig_pos = lig_pos - mean
            prot_pos = prot_pos - mean

        return {
            "name": f"complex_{idx}",
            "lig_coords": lig_pos,
            "lig_one_hot": lig_one_hot,
            "num_lig_atoms": len(lig_pos),
            "pocket_coords": prot_pos,
            "pocket_one_hot": prot_one_hot,
            "num_pocket_nodes": len(prot_pos)
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "names": [s["name"] for s in batch],
            "num_lig_atoms": torch.tensor([s["num_lig_atoms"] for s in batch]),
            "num_pocket_nodes": torch.tensor([s["num_pocket_nodes"] for s in batch]),
            "lig_mask": torch.cat([i * torch.ones(s["num_lig_atoms"], dtype=torch.long) for i, s in enumerate(batch)]),
            "pocket_mask": torch.cat([i * torch.ones(s["num_pocket_nodes"], dtype=torch.long) for i, s in enumerate(batch)]),
            "lig_coords": torch.cat([s["lig_coords"] for s in batch]),
            "lig_one_hot": torch.cat([s["lig_one_hot"] for s in batch]),
            "pocket_coords": torch.cat([s["pocket_coords"] for s in batch]),
            "pocket_one_hot": torch.cat([s["pocket_one_hot"] for s in batch]),
        }


def save_diffsbdd_checkpoint(model, path, epoch, global_step):
    """Save checkpoint in PyTorch Lightning format compatible with LigandPocketDDPM."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "hyper_parameters": model.hparams,
        "epoch": epoch,
        "global_step": global_step,
        "pytorch-lightning_version": "2.3.3"
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path} (Epoch {epoch}, Step {global_step})")


def evaluate_validation(model, val_loader, device, max_batches=None):
    """Compute average validation loss across validation dataset."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            nll, info = model(batch)
            loss = nll.mean(0)
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Arm B: Full Fine-Tune on Metalloprotein Data")
    parser.add_argument("--base_checkpoint", default="checkpoints/crossdocked_fullatom_cond.ckpt")
    parser.add_argument("--lmdb_path", default="/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/data/crossdocked_pocket10_processed.lmdb")
    parser.add_argument("--split_path", default="data/metal_target_split.pt")
    parser.add_argument("--metals_map", default="data/pdb_metals_map.json")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="results/step2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_pretrain", action="store_true", default=True,
                        help="Strictly exclude all PDB IDs seen in base model pretraining (default: True)")
    parser.add_argument("--pretrain_train_split", default="DiffSBDD/data/timesplit_no_lig_or_rec_overlap_train")
    parser.add_argument("--pretrain_val_split", default="DiffSBDD/data/timesplit_no_lig_or_rec_overlap_val")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for dry-run/testing")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples for dry-run/testing")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load pretraining PDBs if excluding
    excluded_pretrain_pdbs = set()
    if args.exclude_pretrain:
        for split_file in [args.pretrain_train_split, args.pretrain_val_split]:
            if os.path.exists(split_file):
                with open(split_file) as f:
                    for line in f:
                        l = line.strip().lower()
                        if l:
                            excluded_pretrain_pdbs.add(l)
        print(f"Loaded {len(excluded_pretrain_pdbs)} PDB IDs to exclude from base pretraining splits")

    # Load split and identify metalloprotein indices
    print(f"Loading split from {args.split_path}...")
    split = torch.load(args.split_path, map_location="cpu")
    with open(split["provenance"]["manifest"]) as f:
        manifest = json.load(f)
    with open(args.metals_map) as f:
        pdb_metals = json.load(f)

    train_metallo_indices = []
    n_train_seen_excluded = 0
    for idx in split["train_indices"]:
        fn = manifest[idx]
        parts = fn.split("/")[-1].split("_")
        if len(parts) >= 2:
            rec_pdb = parts[0].lower()
            if rec_pdb in pdb_metals:
                if args.exclude_pretrain and rec_pdb in excluded_pretrain_pdbs:
                    n_train_seen_excluded += 1
                else:
                    train_metallo_indices.append(idx)

    val_metallo_indices = []
    n_val_seen_excluded = 0
    for idx in split["val_indices"]:
        fn = manifest[idx]
        parts = fn.split("/")[-1].split("_")
        if len(parts) >= 2:
            rec_pdb = parts[0].lower()
            if rec_pdb in pdb_metals:
                if args.exclude_pretrain and rec_pdb in excluded_pretrain_pdbs:
                    n_val_seen_excluded += 1
                else:
                    val_metallo_indices.append(idx)

    if args.max_train_samples is not None:
        train_metallo_indices = train_metallo_indices[:args.max_train_samples]
    if args.max_val_samples is not None:
        val_metallo_indices = val_metallo_indices[:args.max_val_samples]

    print(f"Found {len(train_metallo_indices)} strictly novel training metalloprotein complexes (excluded {n_train_seen_excluded} pretraining-seen)")
    print(f"Found {len(val_metallo_indices)} strictly novel validation metalloprotein complexes (excluded {n_val_seen_excluded} pretraining-seen)")

    train_ds = MetalloproteinLMDBDataset(args.lmdb_path, train_metallo_indices, center=True)
    val_ds = MetalloproteinLMDBDataset(args.lmdb_path, val_metallo_indices, center=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_ds.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_ds.collate_fn,
        pin_memory=True
    )

    # Load base model
    print(f"Loading base checkpoint from {args.base_checkpoint}...")
    model = LigandPocketDDPM.load_from_checkpoint(args.base_checkpoint, map_location=device)
    model.to(device)
    model.train()

    # Full fine-tune optimizer (all parameters trainable)
    optimizer = torch.optim.AdamW(model.ddpm.parameters(), lr=args.lr, amsgrad=True, weight_decay=1.0e-12)

    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    effective_batch = args.batch_size * args.accum_steps
    print(f"Starting Arm B Fine-Tuning:")
    print(f"  Total examples per epoch: {len(train_metallo_indices)}")
    print(f"  Micro-batch size: {args.batch_size}, Accumulation: {args.accum_steps} -> Effective batch: {effective_batch}")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Total examples seen across run: {len(train_metallo_indices) * args.epochs}")

    # Initial validation loss
    print("Evaluating initial validation loss on metalloprotein val set...")
    val_loss_init = evaluate_validation(model, val_loader, device)
    print(f"Initial Metalloprotein Val Loss: {val_loss_init:.4f}")

    training_log = {
        "arm": "Arm B (Full Fine-Tune, Metal-Blind Representation)",
        "base_checkpoint": args.base_checkpoint,
        "effective_batch_size": effective_batch,
        "lr": args.lr,
        "train_examples_count": len(train_metallo_indices),
        "val_examples_count": len(val_metallo_indices),
        "val_loss_initial": val_loss_init,
        "epochs": [],
        "intermediate_checkpoints": []
    }

    best_val_loss = val_loss_init
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch tensors to device if needed (handled in get_ligand_and_pocket, but coords/one-hot need device)
            nll, info = model(batch)
            loss = nll.mean(0) / args.accum_steps
            loss.backward()

            running_loss += (loss.item() * args.accum_steps)
            num_batches += 1

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.ddpm.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % (args.accum_steps * 10) == 0:
                cur_avg_loss = running_loss / num_batches
                pbar.set_postfix({
                    "loss": f"{cur_avg_loss:.4f}",
                    "step": global_step,
                    "vram_mib": f"{torch.cuda.max_memory_allocated() / (1024**2):.0f}"
                })

        epoch_train_loss = running_loss / max(num_batches, 1)
        epoch_val_loss = evaluate_validation(model, val_loader, device)
        epoch_duration = time.time() - epoch_start

        print(f"\n[Epoch {epoch}/{args.epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Duration: {epoch_duration:.1f}s")

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "duration_sec": epoch_duration,
            "examples_seen": epoch * len(train_metallo_indices),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2)
        }
        training_log["epochs"].append(epoch_record)

        # Save intermediate checkpoint for every epoch
        ckpt_path = os.path.join(args.output_dir, f"arm_b_epoch{epoch}.ckpt")
        save_diffsbdd_checkpoint(model, ckpt_path, epoch, global_step)
        training_log["intermediate_checkpoints"].append({"epoch": epoch, "path": ckpt_path})

        # 4. Best validation loss checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(args.output_dir, "arm_b_best.ckpt")
            save_diffsbdd_checkpoint(model, ckpt_path, epoch, global_step)
            training_log["best_checkpoint"] = {"epoch": epoch, "val_loss": best_val_loss, "path": ckpt_path}

    total_duration = time.time() - start_time
    training_log["total_duration_sec"] = total_duration
    print(f"\nArm B Training Complete! Total time: {total_duration/60:.2f} minutes")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "arm_b_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
