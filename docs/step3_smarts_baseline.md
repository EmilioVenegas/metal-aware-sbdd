# Step 3 De-Risking: The SMARTS-Baseline "Kill Check"

**Date:** 2026-08-17
**Status:** EXECUTED

## Rationale
As pre-registered in `docs/plan.md` and `docs/step2.md`, the most potent critique of the Arm C representation fix is that explicit geometry modeling is unnecessary, and that a cheap post-hoc SMARTS filter for known Zinc-Binding Groups (ZBGs) applied to the base model's (Arm A) outputs would recover just as much valid coordination.

If this filter matches our predicted Arm C >35% valid-coordination rate, the geometric bottleneck hypothesis is irrelevant for practical drug design.

## Methodology
- **Input:** 13,300 total generated molecules from Arm A (`results/step1/generation/sdf`) covering the 133 external Zn targets.
- **Filter:** RDKit SMARTS matching for:
  - Carboxylates: `[CX3](=O)[OX1H0-,OX2H1]`
  - Hydroxamates: `[NX3H1,NX3H0]([OX2H1,OX2H0])C(=O)`
  - Thiols: `[SX2H1,SX1H0-]`
  - Imidazoles: `c1ncnc1`
  - Sulfonamides: `[NX3H2,NX3H1][SX4](=O)(=O)`
- **Evaluation:** The surviving filtered subset was scored through the official `results/step1/checker/coordination_checker.py` geometry harness.

## Results
- **Initial Generated Molecules:** 13,300
- **Molecules Surviving SMARTS Filter:** 5,116 (38.5% yield)
- **Valid-Coordination Rate of Filtered Subset:** **24.96%** (1,277 / 5,116)
- **Absolute Yield Rate:** 9.6% (1,277 / 13,300)

## Conclusion
The post-hoc SMARTS kill check **fails to recover adequate geometry**. 

While filtering enriches the valid-coordination rate slightly (from Arm A's raw 19.98% up to 24.96% for the filtered subset), it falls drastically short of the >35% threshold we predict for the representation fix (Arm C), and nowhere near the 77.17% native ceiling. 

This confirms that simply forcing the presence of a chelating functional group does *not* mean the generative model placed it in a valid spatial orientation relative to the metal. The representation geometry fix remains strictly necessary.
