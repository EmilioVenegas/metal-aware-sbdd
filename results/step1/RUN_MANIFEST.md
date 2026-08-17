# Step 1 run manifest

Provenance for the Step 1 generation run. Written before generation.

## Code

| item | value |
|---|---|
| DiffSBDD upstream | `arneschneuing/DiffSBDD` @ `5d0d38d16c8932a0339fd2ce3f67ade98bbdff27` (2025-06-25) |
| local deviations | 3 lines, I/O and import compatibility only — see `MODIFICATIONS.md` |
| environment | conda `atomica-interface` (torch 2.0.1, CUDA 11.8) |
| GPU | NVIDIA RTX 4060 Laptop, 8188 MiB |

## Checkpoint

| item | value |
|---|---|
| file | `checkpoints/crossdocked_fullatom_cond.ckpt` |
| SHA256 | `07f86764bf569aafbc40a9c15fc02de8e2550437dd0f17f657eab3abe66c372c` |
| epoch / step | 999 / 1,562,000 |
| pocket representation | full-atom |
| pristine upstream | **yes** — 0 missing, 0 unexpected keys under clean upstream `LigandPocketDDPM`; no `atomica\|cross_att\|adapter\|lora` parameter names |

## Pocket construction — G1 satisfied, and a second finding

Generation uses `generate_ligands.py`, whose pocket comes from
`DiffSBDD/utils.py:get_pocket_from_ligand`:

```python
def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):
    ...
    if is_aa(residue.get_resname(), standard=True) \
            and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff:
```

- `dist_cutoff=8.0` matches the G1-registered definition.
- **The same `is_aa(..., standard=True)` filter runs at inference time.**

This is an independent instance of the mechanism, in a different code path from
`process_crossdock.py`. Even when a user supplies a fresh metalloprotein structure directly at
inference, the pocket handed to the model is metal-free. The bug is not only in the
distributed training data — it is in the inference API. Add this to the plan's mechanism
table.

## G2 — coordinate frame

**No transform is required.** Verified on 9ZSN: the Zn coordinate recorded in
`external_zn_test_clean.pt` is identical to the deposited mmCIF coordinate, pocket atoms are
in the deposited frame, and generated molecules are emitted in that same frame (DiffSBDD
centres on pocket centre of mass internally and translates back before writing).

Validation, 9ZSN: raw mmCIF contains 2 Zn atoms at the recorded coordinate; the constructed
pocket has 39 residues / 311 atoms with elements `{C, N, O}` only; **no Zn**; nearest pocket
atom to the Zn is 2.050 Å (a coordinating sidechain donor).

## mmCIF compatibility

26 of 133 external targets have no legacy PDB file (recent depositions; 25 carry 5-character
CCD ligand codes that legacy PDB cannot represent). `lightning_modules.py` selects
`MMCIFParser` for `.cif` inputs. Parser choice does not affect pocket construction, geometry,
or sampling — recorded in `MODIFICATIONS.md`.

## Sampling command

```bash
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
  --pdbfile <structure>.{pdb,cif} \
  --ref_ligand <chain>:<resi> \
  --outfile <target>.sdf \
  --n_samples <batch> --batch_size 20
```

Seeds: per target, `seed = int(sha256(pdb_id).hexdigest()[:8], 16) % 2**31`, recorded per
target in the run log.

## Pilot observation — 9ZSN, n=5, NOT a result

| mol | atoms | min dist to Zn | nearest element | V1 (<1.7 Å) | V2 (non-donor <2.7 Å) |
|---|---|---|---|---|---|
| 0 | 26 | 3.21 Å | C | no | no |
| 1 | 21 | 1.96 Å | **O** | no | — donor, in range |
| 2 | 17 | 2.67 Å | C | no | **yes** |
| 3 | 17 | 2.03 Å | **O** | no | — donor, in range |
| 4 | 18 | 2.30 Å | **O** | no | — donor, in range |

Three of five placed an **oxygen** at 1.96 / 2.03 / 2.30 Å — inside the registered Zn–O
coordination window (1.85–2.30 Å). Zero hard clashes.

**This is n=5 on one target and is not evidence of anything.** It is recorded here, before the
full run, because it points at a mechanism the pre-registered predictions may have got wrong:
the model never sees the metal, but it learned the *ligand density* of training complexes
whose ligands did coordinate metals. Reproducing where ligand atoms sit can therefore
reproduce coordination distances without any representation of the metal.

If that holds at scale, the registered prediction (>30% V1/V2 violation) fails while a
different and more interesting claim survives: **the model gets coordination distance
approximately right by copying ligand density, while having no notion of coordination
chemistry** — wrong donor identity, wrong coordination number, wrong geometry. Section 4's
secondary endpoints (valid-coordination rate, V3 rate, coordination number and angular
distributions) already measure exactly this, so the design survives the surprise.

**No thresholds, endpoints or predictions are changed on the basis of this pilot.** It is
logged as a pre-registered expectation of where the surprise may land.
