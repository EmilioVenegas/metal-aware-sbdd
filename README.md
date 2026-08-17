# metal-aware-sbdd

Pocket-conditioned 3D generative models for structure-based drug design **delete metal ions
from the protein pocket during preprocessing**. They generate ligands into metalloprotein
sites as though the catalytic metal were empty space.

Verified 2026-08-16 across three codebases:

| model | mechanism | metals in pocket? |
|---|---|---|
| DiffSBDD | `is_aa(resname, standard=True)`, `process_crossdock.py:54` | no |
| TargetDiff | `if line[0:6].strip() == 'ATOM':`, `utils/data.py` | no |
| Pocket2Mol | `if line[0:6].strip() == 'ATOM':`, `utils/protein_ligand.py` | no |

Metal ions are HETATM records (`ZN`, `MG`, `FE`, …). Every one of these filters excludes them.

## The work

1. **A metal coordination validity checker** — coordination number, metal–donor distances by
   element pair, angular geometry, denticity. PoseBusters and GenBench3D check none of these.
2. **The failure measurement**, with a non-metallo control to establish that the failure is
   specific to metals rather than general geometric sloppiness.
3. **The fix**, as a four-arm ablation separating "undertrained on metalloproteins" from "the
   representation is the bug".

## Status

**Step 1 generation running (launched 2026-08-16 ~22:30, ETA ~08:00).
Read `docs/RUNBOOK.md` first.**

Step 0 complete. Step 1 pre-registered, gates G1 and G2 passed, clean upstream DiffSBDD
vendored and the checkpoint verified pristine. Generating 100 valid molecules for each of 133
external catalytic-Zn targets. The coordination checker is not yet written — that is the next
task, and it needs no GPU.

## Layout

```
docs/plan.md    the research plan; read this first
docs/step0.md   the current work, as an executable checklist
scripts/        analysis and experiment code
results/        one directory per experiment, each with a README and an ANALYSIS_PLAN
AGENTS.md       instructions for coding agents; CLAUDE.md points here
```

## Constraints

One 8 GB consumer GPU, 32 CPU cores, one person, no wet lab. See `AGENTS.md`.

## Predecessor

Chosen after two other directions were investigated and dropped. The record of that — what
was ruled out and why — is in
`~/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design/docs/`:
`oracle-steering-plan.md`, `neighborhood-scout.md`, `metal-aware-plan.md`.
