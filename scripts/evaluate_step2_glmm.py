#!/usr/bin/env python
import json
import os
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

def load_checker_results(jsonl_path, arm_label, pdb_to_cluster):
    records = []
    if not os.path.exists(jsonl_path):
        return records
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pdb = rec.get("pdb_id")
            if not pdb or pdb not in pdb_to_cluster:
                continue
            records.append({
                "pdb_id": pdb,
                "cluster_id": pdb_to_cluster[pdb],
                "arm": arm_label,
                "has_valid_coordination": int(rec.get("has_valid_coordination", False)),
                "primary_violation": int(rec.get("primary_violation_strict", rec.get("primary_violation", False)))
            })
    return records

def load_clusters(targets_path="data/external_zn_test_clean.pt"):
    import torch
    payload = torch.load(targets_path, map_location="cpu", weights_only=False)
    clusters = payload["clusters"]
    pdb_to_cluster = {}
    for c_idx, member_pdbs in enumerate(clusters):
        c_id = f"C{c_idx+1:02d}"
        for pdb in member_pdbs:
            pdb_to_cluster[pdb] = c_id
    return pdb_to_cluster

def fit_gee_4level(df: pd.DataFrame, outcome_col: str, arm_col: str, cluster_col: str, ref_level="ArmA"):
    """Fits GEE Binomial logistic regression with 4-level categorical predictor."""
    # We use the ifp conda env since it has statsmodels installed for this project
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_f:
        df[[outcome_col, arm_col, cluster_col]].to_csv(tmp_f.name, index=False)
        csv_path = tmp_f.name

    script = f"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import json

df = pd.read_csv('{csv_path}')
# Treat {arm_col} as categorical with reference level
df['{arm_col}'] = pd.Categorical(df['{arm_col}'])
# Set reference level
df['{arm_col}'] = df['{arm_col}'].cat.set_categories(df['{arm_col}'].unique().tolist())
if '{ref_level}' in df['{arm_col}'].cat.categories:
    df['{arm_col}'] = df['{arm_col}'].cat.reorder_categories(['{ref_level}'] + [c for c in df['{arm_col}'].cat.categories if c != '{ref_level}'])

try:
    formula = f"{{'{outcome_col}'}} ~ C({arm_col}, Treatment(reference='{ref_level}'))"
    gee = smf.gee(formula, '{cluster_col}', data=df,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    
    results = {{'fit_ok': True, 'params': {{}}, 'pvals': {{}}, 'ci': {{}}}}
    for param in gee.params.index:
        if param == 'Intercept': continue
        results['params'][param] = float(gee.params[param])
        results['pvals'][param] = float(gee.pvalues[param])
        ci_lower, ci_upper = gee.conf_int().loc[param]
        results['ci'][param] = [float(np.exp(ci_lower)), float(np.exp(ci_upper))]
        results[param + '_odds_ratio'] = float(np.exp(gee.params[param]))
        
    print(json.dumps(results))
except Exception as e:
    print(json.dumps({{'fit_ok': False, 'error': str(e)}}))
"""
    try:
        proc = subprocess.run(
            ["/home/emilio/.conda/envs/ifp/bin/python", "-c", script],
            capture_output=True, text=True, check=True
        )
        res = json.loads(proc.stdout.strip())
        return res
    except Exception as e:
        return {"fit_ok": False, "error": str(e), "stdout": proc.stdout if 'proc' in locals() else ""}
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

def main():
    pdb_to_cluster = load_clusters()
    
    # Paths to each arm's checker output
    sources = {
        "ArmA": "results/step1/checker/generated.jsonl",
        "ArmB": "results/step2/arm_b_generation/checker_results.jsonl",
        "ArmC": "results/step2/arm_c_generation/checker_results.jsonl",
        "Native": "results/step1/checker/native_c1.jsonl"
    }
    
    all_records = []
    for arm_label, path in sources.items():
        recs = load_checker_results(path, arm_label, pdb_to_cluster)
        if recs:
            all_records.extend(recs)
            print(f"Loaded {len(recs)} records for {arm_label}")
        else:
            print(f"Warning: No records found for {arm_label} at {path}")
            
    df = pd.DataFrame(all_records)
    if df.empty:
        print("No data loaded. Exiting.")
        return
        
    print(f"\n--- Valid Coordination Rate (Outcome: has_valid_coordination) ---")
    gee_valid = fit_gee_4level(df, "has_valid_coordination", "arm", "cluster_id", ref_level="ArmA")
    if gee_valid.get("fit_ok"):
        print(json.dumps(gee_valid, indent=2))
    else:
        print("GEE fit failed:", gee_valid.get("error"))

    print(f"\n--- Primary Violation Rate (Outcome: primary_violation) ---")
    gee_viol = fit_gee_4level(df, "primary_violation", "arm", "cluster_id", ref_level="ArmA")
    if gee_viol.get("fit_ok"):
        print(json.dumps(gee_viol, indent=2))
    else:
        print("GEE fit failed:", gee_viol.get("error"))

if __name__ == "__main__":
    main()
