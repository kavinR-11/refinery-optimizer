# Changelog

All notable changes to the SIH 26119 Indigenous Optimization Solver will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Phase 0 Foundation

### Added
- **Developer Tools & Environment Reporting**:
  - `tools/env_report.py`: Inspects OS, CPU (hybrid P-core/E-core layout via `lscpu`), RAM, GPU/VRAM via `nvidia-smi`, CUDA toolkit (`nvcc`), and compiler versions (GCC 15.2, G++-13, CMake 4.2, Ninja, Python 3.14).
  - `/docs/environment.md`: Generated system environment report.
  - `tools/check_no_external_solvers.py`: Compliance verification tool enforcing Hard Rules 1-3. Scans for forbidden third-party libraries (Eigen, SuiteSparse, BLAS/LAPACK, cuBLAS/cuSPARSE outside `/bench`, external solvers like HiGHS/SciPy outside `/tests/oracle` and `/bench`).
  - Dataset preparation scripts: `tools/download_netlib.py`, `tools/download_miplib.py`, `tools/download_maros_meszaros.py`, and `tools/prepare_datasets.py`.
- **Documentation**:
  - `docs/tools.md`: Design doc for developer tooling, compliance scanning rules, and dataset acquisition.
