# Build System Architecture & Configuration

This document specifies the CMake build system, compiler configurations, Python extension packaging, and one-command workflows for the SIH 26119 Optimization Solver.

## 1. Toolchains & Compilers

- **CPU Code**: Compiled with ISO C++17 using the system host compiler `g++` (GCC 15.2). Strict warnings enabled (`-Wall -Wextra -Wpedantic -O3`).
- **Parallelism**: OpenMP enabled for embarrassingly parallel loops (matrix-vector multiplications, row scaling, checkers).
- **GPU Code (CUDA)**:
  - Governed by CMake option `ENABLE_CUDA` (Default: `OFF`).
  - When enabled, `CMAKE_CUDA_HOST_COMPILER` is explicitly set to `g++-13` (Ubuntu GCC 13.4), as CUDA 12.4 `nvcc` does not support GCC 15 host compiler.
  - `CMAKE_CUDA_ARCHITECTURES` is set to `89` (NVIDIA Ada Lovelace, RTX 4060 Laptop).
  - WSL2 dynamic library search path requires `/usr/lib/wsl/lib` first in `LD_LIBRARY_PATH`.
- **Build Generator**: Ninja with CMake 4.2+.

## 2. CMake Targets

1. `sih_core` (STATIC library):
   - Sources under `core/src/`:
     - `core/src/model/sparse_matrix.cpp`
     - `core/src/model/problem.cpp`
     - `core/src/model/solution.cpp`
     - `core/src/model/options.cpp`
     - `core/src/io/mps_io.cpp`
     - `core/src/telemetry/telemetry.cpp`
     - `core/src/checker/checker.cpp`
     - `core/src/utils/affinity.cpp`
   - Include directories: `core/include/`
   - Dependencies: OpenMP::OpenMP_CXX, `pthread`. Zero third-party linear algebra or solver libraries.
2. `_core` (Python C-Extension Module):
   - Built via `pybind11_add_module`.
   - Links `sih_core`.
   - Output directory: `py/sih_solver/`.
3. Unit Test Executables:
   - Built under `build/tests/unit/`.
   - Discovered and executed via CTest and `./test.sh`.

## 3. Workflow Commands

- **Build**: `./build.sh`
  Runs: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`
- **Test**: `./test.sh`
  Runs:
  1. `python3 tools/check_no_external_solvers.py` (Rule 1-3 compliance gate)
  2. `ctest --test-dir build --output-on-failure` (C++ unit tests)
  3. `pytest tests/` (Python tests, rational checker, HiGHS oracle suite)
