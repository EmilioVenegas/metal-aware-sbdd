"""Step 2 Arm C — mandatory gradient-flow check on the LoRA + surgered-vocabulary setup.

Confirms, with one real forward+backward pass, that:
1. LoRA adapter parameters (lora_A, lora_B) receive nonzero gradient.
2. The resized residue_encoder / residue_decoder parameters — which LoRA does NOT and cannot
   adapt, since they are new parameters rather than existing weight matrices — receive nonzero
   gradient because they were explicitly unfrozen by mark_new_vocab_trainable().
3. An arbitrary frozen base-model parameter (outside both of the above) receives NO gradient
   (.grad is None), proving the freezing in apply_lora() actually took effect.

This is the check docs/step2.md §4 calls out as the fix for "LoRA cannot train new vocabulary
rows" — silently training arm C with the new columns frozen would make it degenerate into Arm B
while appearing to run as Arm C. Runs on CPU (Arm B is using the single GPU on this machine);
one forward+backward pass on a handful of real complexes does not need a GPU to be decisive.
"""

import pickle
import sys
from pathlib import Path

import lmdb
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DiffSBDD"))
from lightning_modules import LigandPocketDDPM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import apply_lora, mark_new_vocab_trainable, trainable_parameter_report  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SURGERED_CKPT = REPO_ROOT / "checkpoints" / "arm_c_surgered_init.ckpt"
LMDB_PATH = ("/home/emilio/Documents/atomica-diff-antibiotic/"
             "ATOMICA-Diffusion-Antibiotic-design/data/crossdocked_pocket10_processed.lmdb")
Z_TO_IDX = {6: 0, 7: 1, 8: 2, 16: 3, 5: 4, 35: 5, 17: 6, 15: 7, 53: 8, 9: 9}


def load_one_batch(n_examples=2, pocket_nf=16):
    """Pull a couple of real complexes from the same LMDB Arm B trains on (read-only, safe to
    share concurrently with the running Arm B process — LMDB is designed for concurrent
    readers and Arm B's own loader opens with lock=False). Pocket one-hot is padded from the
    stored 10-wide encoding to pocket_nf columns (metal columns zero) — sufficient for a
    gradient-flow check, which does not require real metal-containing data.
    """
    env = lmdb.open(LMDB_PATH, subdir=False, readonly=True, lock=False, readahead=False)
    examples = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (k, v) in enumerate(cursor):
            if i >= n_examples:
                break
            examples.append(pickle.loads(v))

    def one_hot(elements, width):
        oh = torch.zeros(len(elements), width)
        for idx, z in enumerate(elements):
            if int(z) in Z_TO_IDX:
                oh[idx, Z_TO_IDX[int(z)]] = 1.0
        return oh

    lig_coords_list = [e["ligand_pos"].float() for e in examples]
    lig_oh_list = [one_hot(e["ligand_element"].tolist(), 10) for e in examples]
    prot_coords_list = [e["protein_pos"].float() for e in examples]
    prot_oh_list = [one_hot(e["protein_element"].tolist(), pocket_nf) for e in examples]

    lig_mask = torch.cat([i * torch.ones(len(c), dtype=torch.long)
                          for i, c in enumerate(lig_coords_list)])
    pocket_mask = torch.cat([i * torch.ones(len(c), dtype=torch.long)
                             for i, c in enumerate(prot_coords_list)])

    return {
        "num_lig_atoms": torch.tensor([len(c) for c in lig_coords_list]),
        "num_pocket_nodes": torch.tensor([len(c) for c in prot_coords_list]),
        "lig_mask": lig_mask,
        "pocket_mask": pocket_mask,
        "lig_coords": torch.cat(lig_coords_list),
        "lig_one_hot": torch.cat(lig_oh_list),
        "pocket_coords": torch.cat(prot_coords_list),
        "pocket_one_hot": torch.cat(prot_oh_list),
    }


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    print(f"Loading surgered checkpoint: {SURGERED_CKPT}")
    ckpt = torch.load(SURGERED_CKPT, map_location=device, weights_only=False)
    model = LigandPocketDDPM(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.train()

    print("Applying LoRA + unfreezing resized vocabulary layers...")
    lora_layers = apply_lora(model)
    unfrozen_vocab_params = mark_new_vocab_trainable(model)
    report = trainable_parameter_report(model)
    print(f"LoRA targets: {[n for n, _ in lora_layers]}")
    print(f"Unfrozen vocabulary params: {unfrozen_vocab_params}")
    print(f"Trainable: {report['n_trainable_params']:,} / {report['n_total_params']:,} "
          f"({100 * report['fraction_trainable']:.3f}%)")

    print("\nLoading one real batch...")
    batch = load_one_batch(n_examples=2, pocket_nf=model.aa_nf)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def step():
        optimizer.zero_grad()
        nll, info = model(batch)
        loss = nll.mean(0)
        loss.backward()
        return loss.item()

    # Step 1: with lora_B zero-initialized (standard LoRA init), d(loss)/d(lora_A) is
    # *exactly* zero by construction here — it flows through B, which is zero — regardless of
    # whether the adapter is correctly wired. Only lora_B's gradient is diagnostic on step 1.
    loss1 = step()
    print(f"Step 1 loss: {loss1:.4f}")
    step1_grads = {name: (layer.lora_A.grad.norm().item() if layer.lora_A.grad is not None else None,
                          layer.lora_B.grad.norm().item() if layer.lora_B.grad is not None else None)
                  for name, layer in lora_layers}
    optimizer.step()
    # Step 2: B has now moved off zero, so lora_A should receive a real gradient too. This is
    # what actually distinguishes "wired correctly" from "silently disconnected".
    loss2 = step()
    print(f"Step 2 loss (after one optimizer step): {loss2:.4f}")

    print("\n--- Gradient-flow report ---")
    ok = True

    for name, layer in lora_layers:
        a1, b1 = step1_grads[name]
        a2 = layer.lora_A.grad.norm().item() if layer.lora_A.grad is not None else None
        b2 = layer.lora_B.grad.norm().item() if layer.lora_B.grad is not None else None
        print(f"[LoRA]   {name}")
        print(f"    step 1: |grad A|={a1} (expected exactly 0.0 — B is zero-init), |grad B|={b1}")
        print(f"    step 2: |grad A|={a2} (expected nonzero now — B moved), |grad B|={b2}")
        if b1 is None or b1 == 0.0:
            print(f"  FAIL — lora_B on {name} got no gradient on step 1")
            ok = False
        if a2 is None or a2 == 0.0:
            print(f"  FAIL — lora_A on {name} still has no gradient on step 2 — adapter is "
                  f"disconnected from the loss")
            ok = False

    # residue_encoder is on the loss path (its output feeds the EGNN that produces the ligand
    # denoising target); residue_decoder is not (ConditionalDDPM discards the dynamics module's
    # pocket-side output — verified in equivariant_diffusion/conditional_model.py, `_` on the
    # second return value). Both are marked trainable for architectural consistency, but only
    # residue_encoder should actually receive gradient — a None grad on residue_decoder is the
    # expected, correct outcome here, not a failure.
    checked_encoder = 0
    for name, p in model.named_parameters():
        if "residue_encoder" in name:
            g = p.grad
            norm = g.norm().item() if g is not None else None
            status = "OK" if (norm is not None and norm > 0) else "FAIL"
            print(f"[Vocab/on-path]  {name}: |grad|={norm}  [{status}]")
            checked_encoder += 1
            if status == "FAIL":
                ok = False
        elif "residue_decoder" in name:
            g = p.grad
            print(f"[Vocab/off-path] {name}: |grad|="
                  f"{'None (expected — see comment above)' if g is None else g.norm().item()}")
    assert checked_encoder > 0, "no residue_encoder parameters were checked — something is wrong upstream"

    # A frozen, non-LoRA, non-vocab parameter deep in the backbone should have received no grad.
    frozen_probe_name = "ddpm.dynamics.egnn.e_block_0.gcl_0.node_mlp.2.weight"
    frozen_probe = dict(model.named_parameters())[frozen_probe_name]
    print(f"[Frozen] {frozen_probe_name}: requires_grad={frozen_probe.requires_grad}, "
          f"grad={'None' if frozen_probe.grad is None else frozen_probe.grad.norm().item()}")
    if frozen_probe.requires_grad or frozen_probe.grad is not None:
        print("  FAIL — a parameter that should be frozen is trainable or received a gradient")
        ok = False

    print("\n" + ("PASS — LoRA adapters and vocabulary columns both receive gradient; "
                   "the rest of the base model is correctly frozen." if ok else
                   "FAIL — see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
