# Developer Tools & Compliance Architecture

This document describes the design, algorithm, data structures, and operational boundaries of the repository verification and dataset management tools in `/tools`.

## 1. Compliance Checker (`tools/check_no_external_solvers.py`)

### Purpose & Requirements
As mandated by Hard Rules 1, 2, and 3:
- Core solver code (`/core`, C++17) must use only the ISO C++ standard library and OpenMP (plus CUDA under `/core/gpu` with custom kernels). No Eigen, SuiteSparse, BLAS/LAPACK, cuBLAS, cuSPARSE, or external solver/factorization libraries.
- Python solver code (`/py`) may use NumPy, pandas, matplotlib, and pybind11. It must never import or call external solvers (`scipy.optimize`, `highspy`, `cvxpy`, `ortools`, `pulp`, `cbc`, `scip`, `gurobi`, `cplex`, etc.).
- External solvers (`highspy`, `scipy.optimize.linprog`) and baseline libraries (`cuSPARSE`, `cuBLAS`) are allowed **strictly and exclusively** under `/tests/oracle` and `/bench`.

### Algorithm & Scanning Rules
1. Recursively traverse all files in the repository.
2. Ignore benign directories: `.git`, `.venv`, `build`, `.pytest_cache`, `__pycache__`, `data`.
3. Allow exemptions only for files situated inside:
   - `tests/oracle/`
   - `bench/`
   - `docs/` (documentation text referencing library names)
   - `tools/check_no_external_solvers.py` (the scanner itself)
4. For all remaining Python and C++ source files, scan line by line for:
   - Python forbidden imports: `import scipy.optimize`, `from scipy import optimize`, `import highspy`, `from highspy import`, `import cvxpy`, `import ortools`, `import pulp`, `import cplex`, `import gurobipy`, `import cbc`, etc.
   - C++ forbidden headers: `<Eigen/`, `<suitesparse/`, `<cholmod.h>`, `<umfpack.h>`, `<cblas.h>`, `<lapacke.h>`, `<mkl.h>`, `<cusparse.h>`, `<cublas_v2.h>`, `<highs/`, etc.
5. Exit with code `0` on clean state, or code `1` with file, line number, and offending snippet when a violation is found.

---

## 2. Environment Reporting (`tools/env_report.py`)

### Purpose
To document the exact execution environment for benchmarks and reproducibility, saving hardware, compiler, and OS information to `/docs/environment.md`.

### Collected Attributes
1. **Operating System & WSL**: Kernel version, `/etc/os-release`, WSL2 build details.
2. **CPU Model & Core Layout**: Parsed from `lscpu`, detailing physical sockets, total cores, threads per core, and hybrid architecture (P-cores vs E-cores).
3. **Memory**: RAM total, used, free, and swap from `/proc/meminfo` or `free -h`.
4. **GPU & VRAM**: Driver version, CUDA version, GPU name, compute capability, total VRAM from `nvidia-smi`.
5. **Compilers & Toolchains**:
   - `gcc --version` and `g++ --version` (host compiler: GCC 15.2)
   - `g++-13 --version` (CUDA host compiler)
   - `nvcc --version` (CUDA toolkit: 12.4)
   - `cmake --version` (CMake 4.2+)
   - `ninja --version`
   - `python3 --version`

---

## 3. Dataset Download Tools (`tools/download_*.py`)

### Purpose
Automates downloading and extracting problem instances for LP, MILP, and QP benchmark suites into `/data`:
1. `download_netlib.py`: Downloads Netlib LP subset (e.g., `afiro`, `adlittle`, `beaconfd`, `blend`, `share2b`).
2. `download_miplib.py`: Downloads MIPLIB 2017 easy subset (e.g., `flugpl`, `bell5`, `markshare1`).
3. `download_maros_meszaros.py`: Downloads Maros-Mészáros QP benchmark subset (e.g., `QBANDM`, `CVXQP1_S`, `BOEING1`).
4. `prepare_datasets.py`: Unified entry point that downloads all datasets or prints exact curl/wget commands for offline environments.
