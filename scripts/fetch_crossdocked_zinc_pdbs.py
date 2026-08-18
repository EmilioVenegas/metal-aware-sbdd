#!/usr/bin/env python3
"""Fetch source PDBs for CrossDocked zinc complexes directly from RCSB."""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def download_pdb(pdb_id, output_dir):
    pdb_id = pdb_id.lower()
    out_file = Path(output_dir) / f"{pdb_id}.pdb"
    if out_file.exists() and out_file.stat().st_size > 1000:
        return pdb_id, True, "already exists"
    
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) < 500:
                return pdb_id, False, f"response too small ({len(data)} bytes)"
            with open(out_file, 'wb') as f:
                f.write(data)
        return pdb_id, True, f"downloaded ({len(data)} bytes)"
    except Exception as e:
        return pdb_id, False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Fetch CrossDocked Zn PDBs from RCSB")
    parser.add_argument("--metals_map", default="data/pdb_metals_map.json")
    parser.add_argument("--output_dir", default="data/crossdocked_receptors")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(args.metals_map) as f:
        pdb_metals = json.load(f)

    zn_pdbs = sorted([k.lower() for k, v in pdb_metals.items() if 'ZN' in v])
    print(f"Total Zn PDBs to fetch: {len(zn_pdbs)}")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_pdb, pdb, out_dir): pdb for pdb in zn_pdbs}
        for future in as_completed(futures):
            pdb_id, success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Failed {pdb_id}: {msg}")

    elapsed = time.time() - start_time
    print(f"\nFetch completed in {elapsed:.1f}s.")
    print(f"Successfully fetched: {success_count}/{len(zn_pdbs)}")
    print(f"Failed: {fail_count}/{len(zn_pdbs)}")

if __name__ == "__main__":
    main()
