# Modifications to vendored DiffSBDD

`DiffSBDD/` is vendored from [arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD)
at commit **`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`** (2025-06-25, "conda environment").

This file records every deviation from that commit, so the boundary between upstream work and
this project's contribution stays inspectable. Regenerate the comparison at any time:

```bash
git clone https://github.com/arneschneuing/DiffSBDD /tmp/upstream-diffsbdd
cd /tmp/upstream-diffsbdd && git checkout 5d0d38d
diff -rq /tmp/upstream-diffsbdd ~/Documents/metal-aware-sbdd/DiffSBDD
```

**Why a fresh vendor rather than reusing the predecessor's copy.** The DiffSBDD in
`~/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/` is a fork with
~850 changed lines, including 239 in `equivariant_diffusion/dynamics.py` and 97 in
`equivariant_diffusion/conditional_model.py` that thread an ATOMICA embedding through every
step of the reverse process. Arm A must be unmodified upstream sampling, and the Step 2 diff
must read as this project's contribution alone rather than as a delta on a dead research
direction.

## Changes — compatibility only, 2 lines total

| file | change | why |
|---|---|---|
| `lightning_modules.py` | `three_to_one` import wrapped in try/except | BioPython ≥ 1.80 moved it to `protein_letters_3to1`; upstream import raises ImportError |
| `analysis/molecule_builder.py` | `import openbabel` → `from openbabel import openbabel` | `openbabel-wheel` packaging |

Neither touches sampling, the diffusion process, the model, or molecule validity.

`img/` and `colab/` were dropped as unused.

## Deliberately NOT ported

**`analysis/molecule_builder.py` sanitization changes.** The predecessor fork also added
`Chem.SanitizeMol` fallbacks, `UpdatePropertyCache(strict=False)`, and `return None` paths in
`build_molecule`. **These change which generated molecules count as valid**, and Step 1's
primary endpoint is measured over N = 100 *valid* molecules per target. Importing them would
mean arm A's validity rate is this project's, not upstream's — silently changing the
denominator of the headline measurement.

If a validity-related crash appears during generation, fix it by *recording* the failure, not
by making borderline molecules pass. Any change to validity semantics must be a pre-registered
amendment.

**`analysis/SA_Score/sascorer.py` Morgan-fingerprint shim.** The fork added a
`GetMorganGenerator` fallback, but never wired it into the call site — it is dead code there.
`sascorer.calculateScore` was verified working unmodified under this environment's RDKit
(acetaminophen → SA 1.407), so the file is byte-identical to upstream.

## Checkpoint provenance

`checkpoints/crossdocked_fullatom_cond.ckpt`

- SHA256 `07f86764bf569aafbc40a9c15fc02de8e2550437dd0f17f657eab3abe66c372c`
- epoch 999, global_step 1,562,000
- `pocket_representation: full-atom`, `dataset: crossdock`

**Verified pristine upstream.** Loaded into the clean upstream `LigandPocketDDPM` with
`strict=False`: **0 missing keys, 0 unexpected keys**, and zero parameter names matching
`atomica|cross_att|adapter|lora`. Arm A's baseline is upstream's released model, not a
fork-trained one.

## Addendum — mmCIF parser selection

| file | change | why |
|---|---|---|
| `lightning_modules.py` (line ~791) | select `MMCIFParser` when the input path ends in `.cif` | 26 of 133 external Zn targets have no legacy PDB file; 25 carry 5-character CCD ligand codes that legacy PDB format cannot represent |

I/O only. Parser choice does not affect residue selection, pocket geometry, atom typing, or
sampling. Total local deviation from upstream is now **3 lines across 2 files**.
