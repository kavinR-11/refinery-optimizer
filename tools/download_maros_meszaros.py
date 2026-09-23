#!/usr/bin/env python3
"""
Maros-Mészáros QP benchmark subset downloader.
Downloads a subset of representative Maros-Mészáros convex QP instances in QPS format into data/maros_meszaros/.
Provides manual curl/wget fallback instructions if network is unavailable.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Common Maros-Mészáros QP benchmark instances
QP_INSTANCES = [
    "QBANDM",
    "CVXQP1_S",
    "CVXQP2_S",
    "CVXQP3_S",
    "BOEING1",
    "HS35",
    "HS76",
]

# Canonical repository mirrors for Maros-Mészáros QPS archives
QP_ARCHIVE_URL = "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances"

def download_maros_meszaros(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing Maros-Mészáros QP benchmark files in {dest_dir}...")
    
    success_count = 0
    # Check for existing instances
    for name in QP_INSTANCES:
        target = dest_dir / f"{name}.qps"
        if target.exists():
            print(f"  [EXISTS] {target.name}")
            success_count += 1
            continue
        
        # Try downloading instance if available
        url = f"{QP_ARCHIVE_URL}/{name.lower()}.mps"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(target, "wb") as f_out:
                f_out.write(resp.read())
            success_count += 1
            print(f"  [OK] Saved {target.name}")
        except Exception:
            pass

    print(f"\nMaros-Mészáros QP available: {success_count}/{len(QP_INSTANCES)}.")
    print("\nManual download / placement instructions for Maros-Mészáros benchmark suite:")
    print("  To fetch the full 138-problem Maros-Mészáros QPS archive:")
    print("  1. Clone or download archive: git clone https://github.com/ralna/maros-meszaros.git /tmp/maros-meszaros")
    print("  2. Copy chosen instances into data/maros_meszaros/:")
    for name in QP_INSTANCES:
        print(f"     cp /tmp/maros-meszaros/{name}.qps data/maros_meszaros/")
    print("  Or download individually from CUTEst/Maros-Mészáros mirror:")
    for name in QP_INSTANCES:
        print(f"     curl -L ftp://ftp.numerical.rl.ac.uk/pub/cuter/maros-meszaros/{name}.qps -o data/maros_meszaros/{name}.qps")

def main():
    root = Path(__file__).resolve().parent.parent
    dest_dir = root / "data" / "maros_meszaros"
    download_maros_meszaros(dest_dir)

if __name__ == "__main__":
    main()
