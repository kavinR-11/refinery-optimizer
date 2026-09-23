#!/usr/bin/env python3
"""
Unified dataset preparation script for Netlib, MIPLIB 2017, and Maros-Mészáros.
"""

from pathlib import Path
import sys

from download_netlib import download_netlib
from download_miplib import download_miplib
from download_maros_meszaros import download_maros_meszaros

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    
    print("=" * 60)
    print("SIH 26119 Optimization Solver: Benchmark Data Preparation")
    print("=" * 60)
    
    download_netlib(data_dir / "netlib")
    print("-" * 60)
    download_miplib(data_dir / "miplib")
    print("-" * 60)
    download_maros_meszaros(data_dir / "maros_meszaros")
    print("=" * 60)
    print("Dataset preparation process finished.")

if __name__ == "__main__":
    main()
