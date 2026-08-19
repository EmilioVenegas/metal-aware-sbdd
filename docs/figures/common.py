#!/usr/bin/env python
"""Shared constants, academic styling, and data loaders for README figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]

# Coordination checker distance thresholds (Å)
VALID_WINDOW = (1.90, 2.35)   # Zn-N / Zn-O accepted coordination distance
V1_CLASH = 1.70               # V1 hard clash boundary
SHELL = 2.70                  # First coordination shell outer boundary
ZN_IONIC_RADIUS = 0.74        # Shannon ionic radius, Zn(II), four-coordinate

ELEMENT_COLOR = {
    "C": "#4d4d4d", "N": "#2166ac", "O": "#b2182b", "S": "#d6a419",
    "P": "#e08214", "F": "#5aae61", "Cl": "#1b7837", "Br": "#8c510a",
    "I": "#762a83", "ZN": "#7b3294",
}

# Neutral / publication series colors across arms
ARM_COLORS = {
    "native": "#1a9850",
    "smarts": "#7570b3",
    "arm_a": "#d6604d",
    "arm_b": "#2166ac",
}

ZBG_SMARTS = {
    "carboxylate": "[CX3](=O)[OX1H0-,OX2H1]",
    "hydroxamate": "[NX3H1,NX3H0]([OX2H1,OX2H0])C(=O)",
    "thiol": "[SX2H1,SX1H0-]",
    "imidazole": "c1ncnc1",
    "sulfonamide": "[NX3H2,NX3H1][SX4](=O)(=O)",
}

# Shared publication styling. Serif text, hairline spines, no chartjunk.
ACADEMIC_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
}


def load_cohort():
    """Loads target metadata, X-ray target set, and 30% sequence clusters."""
    blob = torch.load(REPO / "data/external_zn_test_clean.pt", map_location="cpu",
                      weights_only=False)
    targets = {t["pdb_id"]: t for t in blob["targets"]}
    xray = {p for p, t in targets.items() if t.get("method", "X-ray") == "X-ray"}
    return targets, xray, blob["clusters"]


def load_jsonl(path: Path) -> list[dict]:
    """Reads lines from a JSONL file."""
    return [json.loads(line) for line in path.open() if line.strip()]


def cluster_bootstrap_ci(per_target: dict[str, float], pdb_to_cluster: dict[str, str],
                         n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Resample sequence clusters with replacement; percentile CI on the cluster-mean rate."""
    by_cluster: dict[str, list[float]] = {}
    for pdb, val in per_target.items():
        by_cluster.setdefault(pdb_to_cluster[pdb], []).append(val)
    means = np.array([np.mean(v) for v in by_cluster.values()])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(n_boot, len(means)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rate(records: list[dict], field: str, keep: set[str]) -> float:
    """Calculates percentage of valid records matching field."""
    vals = [bool(r.get(field)) for r in records
            if not r.get("unreadable") and r["pdb_id"] in keep]
    return 100.0 * float(np.mean(vals))


def per_target_rate(records: list[dict], field: str, keep: set[str],
                    subset: dict[tuple[str, int], bool] | None = None) -> dict[str, float]:
    """Calculates per-target rates for target-level clustering analysis."""
    acc: dict[str, list[bool]] = {}
    for r in records:
        if r.get("unreadable") or r["pdb_id"] not in keep:
            continue
        if subset is not None and not subset.get((r["pdb_id"], r["mol_index"]), False):
            continue
        acc.setdefault(r["pdb_id"], []).append(bool(r.get(field)))
    return {p: 100.0 * float(np.mean(v)) for p, v in acc.items()}


def smarts_matches(sdf_dir: Path, keep: set[str]) -> dict[tuple[str, int], bool]:
    """Identifies generated molecules matching any registered ZBG SMARTS pattern."""
    patterns = [p for p in (Chem.MolFromSmarts(s) for s in ZBG_SMARTS.values())
                if p is not None]
    match: dict[tuple[str, int], bool] = {}
    for pdb_id in sorted(keep):
        f = sdf_dir / f"{pdb_id}.sdf"
        if not f.exists():
            continue
        for k, mol in enumerate(Chem.SDMolSupplier(str(f), sanitize=True)):
            match[(pdb_id, k)] = mol is not None and any(mol.HasSubstructMatch(p)
                                                         for p in patterns)
    return match
