"""Minimal hand-rolled LoRA for the DiffSBDD EGNN.

No LoRA implementation exists anywhere in this codebase or the vendored DiffSBDD/ tree
(verified by grep before writing this — see docs/step2.md §4). `peft` is not in
DiffSBDD/environment.yaml and its version compatibility with the pinned
pytorch==2.0.1 / pytorch-lightning==1.8.4 was not checked, so this hand-rolls the small
piece actually needed rather than taking on a new dependency for it.

Target layers, by design (see docs/step2.md §4, "2-4 layers... not every linear in the
network"): the *node_mlp* final Linear in the last two EquivariantBlocks
(egnn.e_block_3.gcl_0.node_mlp.2, egnn.e_block_4.gcl_0.node_mlp.2). These are plain invariant
feature MLPs operating on scalar node features h, not the coordinate-update pathway
(gcl_equiv) — deliberately left alone so LoRA can't perturb the model's equivariance
guarantees, which live entirely in the coordinate-update math.

Two independent gotchas this module exists to satisfy correctly (docs/step2.md §4):
1. LoRA adapts *existing* weight matrices; it cannot train genuinely new parameters. The
   residue_encoder/residue_decoder layers that Arm C's vocabulary surgery resized are NOT
   LoRA targets here — they are left fully trainable instead (see mark_new_vocab_trainable).
2. Freezing must be explicit and total: apply_lora() freezes every parameter in the model
   first, then re-enables exactly the LoRA adapter parameters plus whatever the caller marks
   separately (e.g. via mark_new_vocab_trainable). Nothing is trainable by omission.
"""

import math

import torch
import torch.nn as nn


DEFAULT_LORA_TARGETS = [
    "ddpm.dynamics.egnn.e_block_3.gcl_0.node_mlp.2",
    "ddpm.dynamics.egnn.e_block_4.gcl_0.node_mlp.2",
]


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank additive update.

    forward(x) = base(x) + scaling * (x @ A^T) @ B^T
    A: (r, in_features), initialized Kaiming-uniform (standard LoRA init).
    B: (out_features, r), initialized to zero, so the wrapped layer is an exact identity
       (behaves as the frozen base alone) until any training happens — the LoRA-equivalent of
       the numerical-identity property already verified for the vocabulary surgery itself.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def _get_submodule(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(model: nn.Module, target_names=None, r: int = 8, alpha: int = 16):
    """Freeze the whole model, then inject LoRA adapters at target_names.

    Returns the list of (qualified_name, LoRALinear) actually created, for the caller's
    optimizer parameter group and for the gradient-flow check.
    """
    if target_names is None:
        target_names = DEFAULT_LORA_TARGETS

    for p in model.parameters():
        p.requires_grad = False

    created = []
    for name in target_names:
        parent, attr = _get_submodule(model, name)
        base_linear = getattr(parent, attr)
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear (got {type(base_linear)})")
        lora_layer = LoRALinear(base_linear, r=r, alpha=alpha)
        setattr(parent, attr, lora_layer)
        created.append((name, lora_layer))
    return created


def mark_new_vocab_trainable(model: nn.Module):
    """Unfreeze residue_encoder/residue_decoder in full (small MLPs, resized by the vocabulary
    surgery — see docs/step2.md §4). Not LoRA targets: LoRA adapts existing weight matrices,
    and the new pocket-vocabulary rows/columns in these layers are not existing parameters.
    """
    made_trainable = []
    for name, p in model.named_parameters():
        if "residue_encoder" in name or "residue_decoder" in name:
            p.requires_grad = True
            made_trainable.append(name)
    return made_trainable


def trainable_parameter_report(model: nn.Module):
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append(name)
    n_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_names": trainable,
        "frozen_names": frozen,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
        "fraction_trainable": n_trainable / n_total,
    }
