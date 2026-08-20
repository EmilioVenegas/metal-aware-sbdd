#!/usr/bin/env python
"""Arm D cluster-level analysis: seed-excluded endpoints, S1 attributability, S2 floor.

Produces the numbers the pre-registered Arm D decision rules require
(results/step2/ANALYSIS_PLAN_ARMD.md sections 4b and 5), which the pooled percentages
from score_arm_d.py cannot supply:

  PRIMARY     seed-excluded valid coordination, Arm D vs Arm C, paired at cluster level.
  CO-PRIMARY  V1 hard clash (as scored), Arm D vs Arm C. The seed sits at 2.05 A and
              cannot clash, so every clash is a model-placed secondary atom.
  S1          Arm D vs random-vector control, per endpoint. Separation below the
              registered 6.77 pp bound is reported UNRESOLVED, never as a null.
  S2          the seed-alone arithmetic floor (one-atom molecule, no model), recomputed
              here rather than quoted, so every endpoint is shown against it.

Statistical helpers are imported from run_arm_c_analysis so Arms C and D are scored on
identical machinery rather than a second implementation that could drift.

Runs on partial cohorts. When targets are missing it restricts every arm to the common
target set, reports the restriction, and labels the output INTERIM -- a partial cohort is
not a random subsample (targets are generated in cohort file order), so interim point
estimates are not the registered result and change no threshold.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from run_arm_c_analysis import (
    load_cohort_and_clusters,
    cluster_bootstrap_diff,
    cluster_bootstrap_single,
    fit_gee_contrast,
)
from coordination_checker import check_molecule
from generate_arm_d import compute_open_coordination_seed

# Registered in ANALYSIS_PLAN_ARMD.md section 5: control S1 runs at N=25, inflating the
# minimum detectable effect from 4.59 pp to 6.77 pp at m=21 clusters.
MDE_S1_PP = 6.77
MDE_ARMC_PP = 4.59

ARM_C_REF = {"seed_excluded_valid": 0.2405, "v1_clash": 0.1159}


def load_jsonl(path: Path, keep) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("unreadable"):
            continue
        if keep(r):
            out.append(r)
    return out


def s2_floor(targets, pdb_to_cluster, donors_path: Path, pdbs=None):
    """Seed-alone arithmetic floor: a one-atom 'molecule', no diffusion, no model."""
    donors = json.loads(donors_path.read_text())
    rows = []
    for t in targets:
        pdb = t["pdb_id"]
        if pdbs is not None and pdb not in pdbs:
            continue
        info = donors.get(pdb)
        if not info:
            continue
        zn = np.array(info["zn"], float)
        pdon = info.get("protein_donors", [])
        pxyz = np.array([d["xyz"] for d in pdon], float) if pdon else None
        seed = compute_open_coordination_seed(zn, pdon, 2.05)
        res = check_molecule(seed.reshape(1, 3), ["O"], zn, "ZN", pxyz, bonds=[])
        rows.append({
            "pdb_id": pdb,
            "cluster_id": pdb_to_cluster.get(pdb),
            "has_valid_coordination": bool(res["has_valid_coordination"]),
            "angular": res["coordination_rms_angle_dev"],
        })
    return pd.DataFrame(rows)


def cluster_means(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("cluster_id")[col].mean()


def paired_contrast(df_a: pd.DataFrame, df_b: pd.DataFrame, col_a: str,
                    col_b: str | None = None) -> Dict[str, Any]:
    """Cluster-level paired difference (a - b) with bootstrap CI and sigma_d."""
    col_b = col_b or col_a
    ma, mb = cluster_means(df_a, col_a), cluster_means(df_b, col_b)
    common = sorted(set(ma.index) & set(mb.index))
    if not common:
        return {"n_clusters": 0}
    diffs = (ma.loc[common] - mb.loc[common]).values
    boot = cluster_bootstrap_diff(diffs)
    boot["n_clusters"] = len(common)
    boot["mean_a"] = float(ma.loc[common].mean())
    boot["mean_b"] = float(mb.loc[common].mean())
    return boot


def verdict(delta_pp: float, bound_pp: float) -> str:
    if abs(delta_pp) < bound_pp:
        return f"UNRESOLVED (|{delta_pp:+.2f}| pp < {bound_pp:.2f} pp bound)"
    return "SEPARATES" if delta_pp > 0 else "SEPARATES (negative)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-d", default="results/step2/arm_d_generation/checker_results.jsonl")
    ap.add_argument("--control", default="results/step2/arm_d_control_random/checker_results.jsonl")
    ap.add_argument("--arm-c", default="results/step2/arm_c_generation/checker_results.jsonl")
    ap.add_argument("--targets", default="data/external_zn_test_clean.pt")
    ap.add_argument("--protein-donors", default="data/protein_donors.json")
    ap.add_argument("--stratum", default="xray", choices=["xray", "all"])
    ap.add_argument("--cluster-cap", type=int, default=3,
                    help="design depth per cluster (Amendment 1 registers 3). "
                         "0 = expect the full cohort.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    targets, clusters, tgt_by_pdb, pdb_to_cluster, cluster_meta = \
        load_cohort_and_clusters(args.targets)
    xray_clusters = {c for c, m in cluster_meta.items() if m["is_xray"]}

    def in_stratum(pdb):
        c = pdb_to_cluster.get(pdb)
        return c is not None and (args.stratum == "all" or c in xray_clusters)

    keep = lambda r: in_stratum(r["pdb_id"])
    d_recs = load_jsonl(Path(args.arm_d), keep)
    s1_recs = load_jsonl(Path(args.control), keep)
    c_recs = load_jsonl(Path(args.arm_c), keep)

    if not d_recs:
        sys.exit(f"no Arm D records in {args.arm_d}")

    df_d = pd.DataFrame(d_recs)
    df_c = pd.DataFrame(c_recs) if c_recs else pd.DataFrame()
    df_s1 = pd.DataFrame(s1_recs) if s1_recs else pd.DataFrame()
    for df in (df_d, df_c, df_s1):
        if len(df):
            df["cluster_id"] = df["pdb_id"].map(pdb_to_cluster)

    # Restrict every arm to the common target set: a partial cohort is not a random
    # subsample, so an unpaired comparison would confound arm with target composition.
    d_pdbs = set(df_d["pdb_id"])
    common = set(d_pdbs)
    if len(df_c):
        common &= set(df_c["pdb_id"])
    if len(df_s1):
        common &= set(df_s1["pdb_id"])

    # Completeness is judged against the DESIGN, not against raw target count. Under
    # Amendment 1 the registered design is cluster-capped: every cluster contributes
    # min(cap, cluster_size) targets. 36 targets covering 21/21 clusters is therefore the
    # complete registered analysis, not a partial look. Calling it "interim" would
    # misrepresent the result in the permanent record.
    stratum_pdbs = [t["pdb_id"] for t in targets if in_stratum(t["pdb_id"])]
    n_cohort = len(stratum_pdbs)
    cluster_sizes: Dict[str, int] = {}
    for p in stratum_pdbs:
        c = pdb_to_cluster[p]
        cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
    if args.cluster_cap:
        expected = {c: min(args.cluster_cap, n) for c, n in cluster_sizes.items()}
        design = f"cluster-cap {args.cluster_cap}"
    else:
        expected = dict(cluster_sizes)
        design = "full cohort"
    have: Dict[str, int] = {}
    for p in common:
        c = pdb_to_cluster.get(p)
        if c in expected:
            have[c] = have.get(c, 0) + 1
    short = {c: (have.get(c, 0), e) for c, e in expected.items() if have.get(c, 0) < e}
    interim = bool(short)

    df_d = df_d[df_d["pdb_id"].isin(common)]
    if len(df_c):
        df_c = df_c[df_c["pdb_id"].isin(common)]
    if len(df_s1):
        df_s1 = df_s1[df_s1["pdb_id"].isin(common)]

    df_s2 = s2_floor(targets, pdb_to_cluster, Path(args.protein_donors), pdbs=common)

    banner = (f"INTERIM -- {len(short)} CLUSTER(S) SHORT OF DESIGN, "
              f"NOT THE REGISTERED RESULT") if interim else \
        f"COMPLETE -- {design}, {len(expected)}/{len(cluster_sizes)} clusters at depth"
    print("=" * 74)
    print(f"  Arm D analysis [{banner}]")
    print("=" * 74)
    print(f"  stratum            : {args.stratum} ({n_cohort} targets in full stratum)")
    print(f"  design             : {design}")
    print(f"  paired targets     : {len(common)}")
    print(f"  clusters covered   : {df_d['cluster_id'].nunique()} / {len(cluster_sizes)}")
    print(f"  Arm D molecules    : {len(df_d)}")
    print(f"  control molecules  : {len(df_s1) if len(df_s1) else 'NOT RUN YET'}")
    print(f"  Arm C molecules    : {len(df_c) if len(df_c) else 'unavailable'}")
    if interim:
        print()
        print(f"  Clusters below design depth: "
              f"{', '.join(f'{c} {h}/{e}' for c, (h, e) in sorted(short.items()))}")
        print("  Point estimates below are not the registered result and change no threshold.")

    out: Dict[str, Any] = {"interim": interim, "design": design,
                           "n_paired_targets": len(common),
                           "n_clusters": int(df_d["cluster_id"].nunique()),
                           "clusters_short_of_design": short,
                           "stratum": args.stratum}

    # ---- S2 arithmetic floor -------------------------------------------------
    print()
    print("-" * 74)
    print("  S2 -- seed-alone arithmetic floor (one atom, no diffusion, no model)")
    print("-" * 74)
    if len(df_s2):
        f_valid = float(df_s2["has_valid_coordination"].mean())
        angs = df_s2["angular"].dropna()
        print(f"  as-scored valid coordination : {100*f_valid:.2f}%")
        print(f"  angular RMSD                 : {angs.mean():.2f} deg "
              f"(median {angs.median():.2f})")
        print("  => any as-scored endpoint at or below this floor is arithmetic, not a result.")
        out["s2_floor"] = {"valid_coordination": f_valid,
                           "angular_mean": float(angs.mean())}

    # ---- retired endpoint, shown for transparency ---------------------------
    print()
    print("-" * 74)
    print("  RETIRED endpoint (seed-determined -- reported only to show saturation)")
    print("-" * 74)
    as_valid = float(df_d["as_scored_has_valid_coordination"].mean())
    print(f"  Arm D as-scored valid coordination : {100*as_valid:.2f}%")
    if len(df_s1):
        s1_as = float(df_s1["as_scored_has_valid_coordination"].mean())
        print(f"  control  as-scored valid coordination : {100*s1_as:.2f}%")
        print(f"  => separation {100*(as_valid-s1_as):+.2f} pp : "
              f"{verdict(100*(as_valid-s1_as), MDE_S1_PP)}")
    out["as_scored_valid_coordination"] = as_valid

    # ---- primary + co-primary vs Arm C --------------------------------------
    endpoints = [
        ("PRIMARY   seed-excluded valid coordination",
         "seed_excluded_has_valid_coordination", "has_valid_coordination",
         ARM_C_REF["seed_excluded_valid"]),
        ("CO-PRIMARY V1 hard clash (as scored)",
         "v1_clash", "v1_clash", ARM_C_REF["v1_clash"]),
    ]
    print()
    print("-" * 74)
    print(f"  Arm D vs Arm C, paired at cluster level (MDE {MDE_ARMC_PP:.2f} pp)")
    print("-" * 74)
    out["vs_arm_c"] = {}
    for label, dcol, ccol, ref in endpoints:
        if dcol not in df_d.columns:
            continue
        sub = df_d.dropna(subset=[dcol])
        pooled = float(sub[dcol].mean())
        line = f"  {label}\n    Arm D pooled : {100*pooled:.2f}%"
        rec: Dict[str, Any] = {"arm_d_pooled": pooled, "arm_c_full_cohort_ref": ref}
        if len(df_c) and ccol in df_c.columns:
            con = paired_contrast(sub, df_c, dcol, ccol)
            if con.get("n_clusters"):
                dpp = 100 * con["mean"]
                line += (f"\n    Arm C paired : {100*con['mean_b']:.2f}%"
                         f"  (full-cohort published {100*ref:.2f}%)"
                         f"\n    paired diff  : {dpp:+.2f} pp  "
                         f"95% CI [{100*con['ci_95'][0]:+.2f}, {100*con['ci_95'][1]:+.2f}]"
                         f"  sigma_d={con['sigma_d']:.4f}  m={con['n_clusters']}"
                         f"\n    verdict      : {verdict(dpp, MDE_ARMC_PP)}")
                rec["paired_vs_arm_c"] = con
        print(line)
        out["vs_arm_c"][dcol] = rec

    # ---- S1 attributability -------------------------------------------------
    print()
    print("-" * 74)
    print(f"  S1 attributability -- Arm D vs random-vector control "
          f"(bound {MDE_S1_PP:.2f} pp)")
    print("-" * 74)
    if not len(df_s1):
        print("  control NOT RUN YET.")
        print("  Without S1, no Arm D endpoint is attributable to the open coordination")
        print("  vector rather than to placing an atom at 2.05 A. No claim is licensed.")
        out["s1"] = None
    else:
        out["s1"] = {}
        for label, col in [("seed-excluded valid coordination",
                            "seed_excluded_has_valid_coordination"),
                           ("V1 hard clash", "v1_clash"),
                           ("multi-fragment rate", "_multifrag")]:
            if col == "_multifrag":
                df_d["_multifrag"] = df_d["n_fragments"] > 1
                df_s1["_multifrag"] = df_s1["n_fragments"] > 1
            if col not in df_d.columns or col not in df_s1.columns:
                continue
            con = paired_contrast(df_d.dropna(subset=[col]),
                                  df_s1.dropna(subset=[col]), col)
            if not con.get("n_clusters"):
                continue
            dpp = 100 * con["mean"]
            print(f"  {label}")
            print(f"    Arm D {100*con['mean_a']:.2f}%  vs  control {100*con['mean_b']:.2f}%"
                  f"   diff {dpp:+.2f} pp"
                  f"  95% CI [{100*con['ci_95'][0]:+.2f}, {100*con['ci_95'][1]:+.2f}]")
            print(f"    {verdict(dpp, MDE_S1_PP)}")
            out["s1"][col] = con

    # ---- secondary ----------------------------------------------------------
    print()
    print("-" * 74)
    print("  Secondary diagnostics")
    print("-" * 74)
    seed_found = float(df_d["seed_found"].mean())
    multifrag = float((df_d["n_fragments"] > 1).mean())
    zbg = float(df_d["seed_zbg"].notna().mean()) if "seed_zbg" in df_d.columns else None
    print(f"  seed survived sanitization : {100*seed_found:.2f}%"
          f"   (seed-excluded denominator = {int(df_d['seed_found'].sum())})")
    print(f"  seed built into a ZBG      : {100*zbg:.2f}%" if zbg is not None else "")
    print(f"  multi-fragment molecules   : {100*multifrag:.2f}%")
    if multifrag > 0.5:
        print("  WARNING: majority of molecules are disconnected. Coordination can be")
        print("  satisfied by stray fragments rather than by a designed binding group;")
        print("  treat seed-excluded coordination as contaminated until fragment-resolved.")
    out["secondary"] = {"seed_found": seed_found, "multifragment": multifrag,
                        "seed_zbg": zbg}

    # ---- scenario mapping ---------------------------------------------------
    print()
    print("=" * 74)
    if "seed_excluded_has_valid_coordination" in out["vs_arm_c"]:
        v = out["vs_arm_c"]["seed_excluded_has_valid_coordination"]["arm_d_pooled"]
        c = out["vs_arm_c"].get("v1_clash", {}).get("arm_d_pooled")
        if c is not None:
            better_coord = v > ARM_C_REF["seed_excluded_valid"]
            better_clash = c < ARM_C_REF["v1_clash"]
            scen = {(True, True): "I  -- inpainting works",
                    (True, False): "II -- chemistry improves, sterics do not",
                    (False, False): "III -- pinning degrades (honest negative)",
                    (False, True): "IV -- trade, not fix"}[(better_coord, better_clash)]
            print(f"  Scenario reading: {scen}")
            if out["s1"] is None:
                print("  (provisional -- S1 not yet available, so not attributable)")
            out["scenario"] = scen
    print("=" * 74)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
