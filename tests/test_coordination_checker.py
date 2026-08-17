"""Geometry unit tests for the coordination checker.

Validates the pre-registered thresholds against known geometry, independently of any
generated data. Run before trusting any measurement the checker produces.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from coordination_checker import check_molecule  # noqa: E402

Z = np.zeros(3)


def chk(coords, els, metal="ZN"):
    return check_molecule(np.array(coords, float), els, Z, metal)


def test_hard_clash_is_v1():
    r = chk([[1.2, 0, 0]], ["C"])
    assert r["v1_clash"] and r["primary_violation"]


def test_nondonor_in_shell_is_v2():
    r = chk([[2.4, 0, 0]], ["C"])
    assert not r["v1_clash"] and r["v2_shell_occupancy"] and r["primary_violation"]


def test_donor_in_range_is_valid_not_violation():
    r = chk([[2.05, 0, 0]], ["O"])
    assert r["has_valid_coordination"] and not r["primary_violation"] and not r["v3_malformed"]


def test_donor_out_of_range_is_v3_only():
    """A donor at a malformed distance is NOT a primary violation - by design.

    The primary endpoint asks whether the model occupies the metal's excluded volume,
    not whether it coordinates well. V3 is a secondary endpoint.
    """
    r = chk([[2.55, 0, 0]], ["O"])
    assert r["v3_malformed"] and not r["primary_violation"] and not r["has_valid_coordination"]


def test_distant_atom_is_clean():
    r = chk([[5.0, 0, 0]], ["C"])
    assert not r["primary_violation"] and r["n_shell_contacts"] == 0


def test_ideal_tetrahedral_geometry():
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    v = v / np.linalg.norm(v, axis=1)[:, None] * 2.0
    r = check_molecule(v, ["O"] * 4, Z, "ZN")
    assert r["n_valid_coordination"] == 4
    assert r["coordination_rms_angle_dev"] < 1.0


def test_per_element_ranges_are_distinct():
    """S at 2.4 A is valid for Zn-S (2.15-2.50) but would be out of range for Zn-O."""
    assert chk([[2.4, 0, 0]], ["S"])["has_valid_coordination"]
    assert not chk([[2.4, 0, 0]], ["O"])["has_valid_coordination"]


def test_metal_identity_changes_ranges():
    """Ca-O (2.25-2.65) accepts 2.5 A where Zn-O (1.85-2.30) does not."""
    assert chk([[2.5, 0, 0]], ["O"], metal="CA")["has_valid_coordination"]
    assert not chk([[2.5, 0, 0]], ["O"], metal="ZN")["has_valid_coordination"]


# --- combined coordination sphere (protein donors + ligand donors) ----------

def test_ligand_completes_tetrahedron_with_protein_donors():
    """A ligand donating ONE oxygen into a 3-His site is not CN=1 - it completes a
    tetrahedron. This is why protein donors must be passed in."""
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    tet = tet / np.linalg.norm(tet, axis=1)[:, None] * 2.0
    protein = tet[:3]              # 3 His nitrogens
    ligand = tet[3:4]              # 1 ligand oxygen
    r = check_molecule(ligand, ["O"], Z, "ZN", protein_donors=protein)
    assert r["n_valid_coordination"] == 1      # ligand contributes one
    assert r["n_protein_donors"] == 3
    assert r["coordination_number_total"] == 4
    assert r["coordination_rms_angle_dev"] < 1.0   # ideal tetrahedron


def test_without_protein_donors_geometry_is_undefined():
    """Same ligand, no protein donors supplied: CN=1, no angle computable."""
    r = check_molecule(np.array([[2.0, 0, 0]]), ["O"], Z, "ZN")
    assert r["coordination_number_total"] == 1
    assert r["coordination_rms_angle_dev"] is None


def test_distorted_sphere_scores_worse_than_ideal():
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    tet = tet / np.linalg.norm(tet, axis=1)[:, None] * 2.0
    ideal = check_molecule(tet[3:4], ["O"], Z, "ZN", protein_donors=tet[:3])
    bad_lig = np.array([[2.0, 0.15, 0.0]])  # oxygen crowded against a protein donor
    distorted = check_molecule(bad_lig, ["O"], Z, "ZN", protein_donors=tet[:3])
    assert distorted["coordination_rms_angle_dev"] > ideal["coordination_rms_angle_dev"]
