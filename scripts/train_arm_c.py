"""Execute Arm C LoRA fine-tune for Metal Coordination Ablation Study.

Starts from checkpoints/arm_c_surgered_init.ckpt (base checkpoint + expanded, metal-aware
pocket vocabulary — see scripts/build_arm_c_surgery.py, numerically verified identical to the
base checkpoint on metal-free pockets by scripts/verify_arm_c_surgery.py). Trains with LoRA
adapters on two EGNN layers plus the fully-trainable resized residue_encoder/residue_decoder
(scripts/lora.py) — gradient flow through both paths verified by
scripts/verify_arm_c_gradient_flow.py before this script was written.

Data: scripts/build_arm_c_dataset.py's real, metal-retained native-ligand pockets
(data/arm_c_train.pt / data/arm_c_val.pt) — not the old metal-blind LMDB Arm B uses.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DiffSBDD")))
from lightning_modules import LigandPocketDDPM  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from lora import apply_lora, mark_new_vocab_trainable, trainable_parameter_report  # noqa: E402


class ArmCDataset(Dataset):
    """Wraps the list-of-dicts produced by scripts/build_arm_c_dataset.py."""

    def __init__(self, pt_path: str, center: bool = True):
        payload = torch.load(pt_path, map_location="cpu")
        self.examples = payload["examples"]
        self.center = center

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        lig_pos, prot_pos = e["ligand_pos"].clone(), e["protein_pos"].clone()
        if self.center:
            mean = (lig_pos.sum(0) + prot_pos.sum(0)) / (len(lig_pos) + len(prot_pos))
            lig_pos = lig_pos - mean
            prot_pos = prot_pos - mean
        return {
            "name": e["pdb_id"],
            "lig_coords": lig_pos,
            "lig_one_hot": e["ligand_one_hot"],
            "num_lig_atoms": len(lig_pos),
            "pocket_coords": prot_pos,
            "pocket_one_hot": e["protein_one_hot"],
            "num_pocket_nodes": len(prot_pos),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "names": [s["name"] for s in batch],
            "num_lig_atoms": torch.tensor([s["num_lig_atoms"] for s in batch]),
            "num_pocket_nodes": torch.tensor([s["num_pocket_nodes"] for s in batch]),
            "lig_mask": torch.cat([i * torch.ones(s["num_lig_atoms"], dtype=torch.long)
                                   for i, s in enumerate(batch)]),
            "pocket_mask": torch.cat([i * torch.ones(s["num_pocket_nodes"], dtype=torch.long)
                                      for i, s in enumerate(batch)]),
            "lig_coords": torch.cat([s["lig_coords"] for s in batch]),
            "lig_one_hot": torch.cat([s["lig_one_hot"] for s in batch]),
            "pocket_coords": torch.cat([s["pocket_coords"] for s in batch]),
            "pocket_one_hot": torch.cat([s["pocket_one_hot"] for s in batch]),
        }


def save_diffsbdd_checkpoint(model, path, epoch, global_step):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": model.hparams,
        "epoch": epoch,
        "global_step": global_step,
        "pytorch-lightning_version": "2.3.3",
    }, path)
    print(f"Saved checkpoint: {path} (Epoch {epoch}, Step {global_step})")


def evaluate_validation(model, val_loader, device, max_batches=None):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            nll, info = model(batch)
            total_loss += nll.mean(0).item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Arm C: LoRA fine-tune on metal-retained pockets")
    parser.add_argument("--surgered_checkpoint", default="checkpoints/arm_c_surgered_init.ckpt")
    parser.add_argument("--train_path", default="data/arm_c_train.pt")
    parser.add_argument("--val_path", default="data/arm_c_val.pt")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="results/step2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3,
                        help="Higher than Arm B's full-fine-tune LR: only ~0.7%% of params "
                             "are trainable here (LoRA + resized vocab layers), the usual "
                             "reason LoRA setups use a larger LR than full fine-tuning.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="More epochs than Arm B: the dataset is far smaller (~1-1.5k vs "
                             "24k examples) and only a small trainable fraction needs to move.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_ds = ArmCDataset(args.train_path)
    val_ds = ArmCDataset(args.val_path)
    print(f"Train examples: {len(train_ds)}  Val examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=train_ds.collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=val_ds.collate_fn,
                            pin_memory=True)

    print(f"Loading surgered checkpoint from {args.surgered_checkpoint}...")
    ckpt = torch.load(args.surgered_checkpoint, map_location=device, weights_only=False)
    model = LigandPocketDDPM(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.train()

    print("Applying LoRA and unfreezing resized vocabulary layers...")
    lora_layers = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    unfrozen_vocab = mark_new_vocab_trainable(model)
    report = trainable_parameter_report(model)
    print(f"LoRA targets: {[n for n, _ in lora_layers]}")
    print(f"Unfrozen vocab params: {unfrozen_vocab}")
    print(f"Trainable: {report['n_trainable_params']:,} / {report['n_total_params']:,} "
          f"({100 * report['fraction_trainable']:.3f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, amsgrad=True, weight_decay=1.0e-12)

    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    effective_batch = args.batch_size * args.accum_steps
    print(f"Starting Arm C LoRA fine-tune: {len(train_ds)} examples/epoch, "
          f"effective batch {effective_batch}, {args.epochs} epochs, "
          f"~{total_steps} optimization steps")

    print("Evaluating initial validation loss...")
    val_loss_init = evaluate_validation(model, val_loader, device)
    print(f"Initial Val Loss: {val_loss_init:.4f}")

    training_log = {
        "arm": "Arm C (LoRA Fine-Tune, Metal-Aware Pocket Representation)",
        "surgered_checkpoint": args.surgered_checkpoint,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_targets": [n for n, _ in lora_layers],
        "unfrozen_vocab_params": unfrozen_vocab,
        "n_trainable_params": report["n_trainable_params"],
        "n_total_params": report["n_total_params"],
        "effective_batch_size": effective_batch, "lr": args.lr,
        "train_examples_count": len(train_ds), "val_examples_count": len(val_ds),
        "val_loss_initial": val_loss_init,
        "epochs": [], "intermediate_checkpoints": [],
    }

    best_val_loss = val_loss_init
    global_step = 0
    checkpoint_epochs = {max(1, args.epochs // 4), max(1, args.epochs // 2), args.epochs}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss, num_batches = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            nll, info = model(batch)
            loss = nll.mean(0) / args.accum_steps
            loss.backward()
            running_loss += loss.item() * args.accum_steps
            num_batches += 1

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % max(1, args.accum_steps * 10) == 0:
                pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}",
                                  "step": global_step})

        epoch_train_loss = running_loss / max(num_batches, 1)
        epoch_val_loss = evaluate_validation(model, val_loader, device)
        epoch_duration = time.time() - epoch_start
        print(f"\n[Epoch {epoch}/{args.epochs}] Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Duration: {epoch_duration:.1f}s")

        training_log["epochs"].append({
            "epoch": epoch, "global_step": global_step,
            "train_loss": epoch_train_loss, "val_loss": epoch_val_loss,
            "duration_sec": epoch_duration, "examples_seen": epoch * len(train_ds),
            "peak_vram_mib": (torch.cuda.max_memory_allocated() / (1024 ** 2)
                              if device.type == "cuda" else None),
        })

        if epoch in checkpoint_epochs:
            ckpt_path = os.path.join(args.output_dir, f"arm_c_epoch{epoch}.ckpt")
            save_diffsbdd_checkpoint(model, ckpt_path, epoch, global_step)
            training_log["intermediate_checkpoints"].append({"epoch": epoch, "path": ckpt_path})

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(args.output_dir, "arm_c_best.ckpt")
            save_diffsbdd_checkpoint(model, ckpt_path, epoch, global_step)
            training_log["best_checkpoint"] = {"epoch": epoch, "val_loss": best_val_loss,
                                               "path": ckpt_path}

    training_log["total_duration_sec"] = time.time() - start_time
    print(f"\nArm C Training Complete! Total time: "
          f"{training_log['total_duration_sec'] / 60:.2f} minutes")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "arm_c_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
