#!/usr/bin/env python3
"""
MIPLIB 2017 easy subset downloader.
Downloads a subset of representative MIPLIB 2017 instances into data/miplib/.
Provides manual curl/wget fallback instructions if network is unavailable.
"""

import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# Official MIPLIB 2017 instance archive URLs
MIPLIB_BASE_URL = "https://miplib.zib.de/WebData/instances"
MIPLIB_INSTANCES = [
    "flugpl",
    "markshare1",
    "pk1",
    "glass4",
    "dcmulti",
]

def download_miplib(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing MIPLIB 2017 benchmark files in {dest_dir}...")
    
    success_count = 0
    headers = {"User-Agent": "Mozilla/5.0"}
    for name in MIPLIB_INSTANCES:
        target = dest_dir / f"{name}.mps"
        if target.exists():
            print(f"  [EXISTS] {target.name}")
            success_count += 1
            continue
        
        gz_target = dest_dir / f"{name}.mps.gz"
        url = f"{MIPLIB_BASE_URL}/{name}.mps.gz"
        try:
            print(f"  Downloading {name} from {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp, open(gz_target, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
            with gzip.open(gz_target, "rb") as f_in, open(target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_target.unlink()
            success_count += 1
            print(f"  [OK] Saved {target.name}")
        except Exception as e:
            if gz_target.exists():
                gz_target.unlink()
            print(f"  [FAILED] Could not download {name}: {e}")

    print(f"\nMIPLIB download complete: {success_count}/{len(MIPLIB_INSTANCES)} available.")
    if success_count < len(MIPLIB_INSTANCES):
        print("\nManual download instructions (run from repository root):")
        for name in MIPLIB_INSTANCES:
            print(f"  curl -L https://miplib.zib.de/WebData/instances/{name}.mps.gz | gunzip > data/miplib/{name}.mps")

def main():
    root = Path(__file__).resolve().parent.parent
    dest_dir = root / "data" / "miplib"
    download_miplib(dest_dir)

if __name__ == "__main__":
    main()
