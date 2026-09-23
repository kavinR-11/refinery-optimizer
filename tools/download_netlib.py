#!/usr/bin/env python3
"""
Netlib LP benchmark downloader.
Downloads a standard representative subset of Netlib LP instances into data/netlib/.
Provides manual curl/wget fallback instructions if network is unavailable.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Common Netlib LP instances (uncompressed or compressed)
NETLIB_BASE_URL = "https://netlib.org/lp/data"
NETLIB_INSTANCES = [
    "afiro",
    "adlittle",
    "beaconfd",
    "blend",
    "share2b",
    "sc50a",
    "sc50b",
    "lotfi",
    "stocfor1",
]

def download_netlib(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing Netlib LP benchmark files in {dest_dir}...")
    
    success_count = 0
    for name in NETLIB_INSTANCES:
        target = dest_dir / f"{name}.mps"
        if target.exists():
            print(f"  [EXISTS] {target.name}")
            success_count += 1
            continue
        
        url = f"{NETLIB_BASE_URL}/{name}"
        try:
            print(f"  Downloading {name} from {url}...")
            urllib.request.urlretrieve(url, target)
            success_count += 1
            print(f"  [OK] Saved {target.name}")
        except Exception as e:
            print(f"  [FAILED] Could not download {name}: {e}")

    print(f"\nNetlib download complete: {success_count}/{len(NETLIB_INSTANCES)} available.")
    if success_count < len(NETLIB_INSTANCES):
        print("\nManual download instructions (run from repository root):")
        for name in NETLIB_INSTANCES:
            print(f"  curl -L {NETLIB_BASE_URL}/{name} -o data/netlib/{name}.mps")

def main():
    root = Path(__file__).resolve().parent.parent
    dest_dir = root / "data" / "netlib"
    download_netlib(dest_dir)

if __name__ == "__main__":
    main()
