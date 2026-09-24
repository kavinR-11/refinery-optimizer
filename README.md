# SIH 26119: Indigenous Optimization Engine

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA 12](https://img.shields.io/badge/GPU-CUDA%2012%20(sm__89)-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Compliance](https://img.shields.io/badge/External%20Solvers-Zero%20(100%25%20Indigenous)-brightgreen.svg)](#from-scratch-guarantee)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade, mathematically complete optimization solver developed entirely from first principles for **Smart India Hackathon (SIH 26119)**. The engine is engineered to solve large-scale Linear Programming (LP), Mixed-Integer Linear Programming (MILP), and Convex Quadratic Programming (QP) problems, with specialized acceleration for oil refinery production scheduling and plant DCS integration.

---

## 1. Capabilities & Mathematical Features

The solver implements a complete, self-contained optimization stack with zero external linear programming or matrix algebra dependencies:

- **Linear Programming (LP)**:
  - **Dual Simplex**: Dual steepest-edge pricing, Devex pricing, Harris ratio test with bound flipping, Bland's anti-cycling rule, and cost perturbation.
  - **Primal Simplex**: Available as an alternative algorithm choice.
  - **Linear Algebra Core**: Custom sparse LU decomposition with Markowitz threshold partial pivoting, hypersparse FTRAN/BTRAN, and Forrest-Tomlin basis updates with numerical refactorization triggers.
  - **Presolve & Postsolve**: Reversible reduction stack (singleton rows/cols, bound tightening, fixed variable elimination, duplicate row removal, dominated columns).
  - **Scaling**: Power-of-two geometric mean equilibration preventing roundoff distortion.

- **Convex Quadratic Programming (QP)**:
  - **Primal-Dual Interior Point Method (IPM)**: Mehrotra predictor-corrector with Mehrotra centering parameter and strict interior-point step-length dampening ($\tau = 0.995$).
  - **Sparse Cholesky Factorization**: Custom $LL^T$ decomposition paired with Approximate Minimum Degree (AMD) ordering to minimize factorization fill-in.
  - **Simplex Crossover**: Automatic handoff from IPM solutions to Phase 1 Simplex to recover a basic vertex solution with exact shadow prices.

- **Mixed-Integer Linear Programming (MILP)**:
  - **Branch-and-Bound**: Node selection via Best-First Search (A*) and Depth-First Search (DFS).
  - **Branching Rules**: Pseudocost branching with history updates, alongside Most-Fractional branching.
  - **Warm Starts**: Exact basis warm-starting for every child node in the tree.
  - **Cutting Planes**: Root-node Gomory Mixed-Integer cuts.
  - **Primal Heuristics**: Fractional diving and rounding heuristics for early incumbent discovery.
  - **Multithreaded Tree Search**: Custom native thread pool (`std::thread` and task queue) optimized for multi-core Intel HX hybrid processors.

- **GPU Acceleration (FP32 PDHG + FP64 Hybrid Simplex)**:
  - **Custom CUDA Kernels**: Native sparse matrix-vector products (CSR and CSC), Chambolle-Pock primal-dual updates, adaptive step size scaling, and restart checks.
  - **Native Device Execution**: FP32 execution on NVIDIA RTX 4060 (Ada Lovelace, compute capability 8.9), delivering up to 36x speedup over CPU simplex on large sparse instances.
  - **Hybrid Crossover**: GPU iterates are passed to CPU Simplex for an exact FP64 cleanup pass.

- **Refinery Operational Planning Engine**:
  - Full industrial plant model: Crude scheduling (Arab Light, Brent, Maya Heavy), distillation units (ADU/CDU, VDU), conversion units (FCC, Reformer, Hydrotreater), and product blending pools (Gasoline, Jet Fuel, Diesel, Fuel Oil, LPG).
  - **Economic Explainability**: Automated shadow price interpretation, capacity bottleneck alerts, and basis sensitivity validity intervals ($[\Delta b_{down}, \Delta b_{up}]$).
  - **Warm-Start Re-Optimization**: Instantaneous re-optimization across crude spot price shocks and unit outages in 2–3 pivots (up to 11x faster than cold-start).

---

## 2. From-Scratch Guarantee & Compliance

> **HARD RULE COMPLIANCE**:
> The core optimization library (`core/`) and Python wrappers (`py/`) contain **zero calls to external solvers or linear algebra libraries**.
> Forbidden libraries strictly excluded from the codebase: HiGHS, GLPK, Gurobi, CPLEX, SCIP, CBC, Xpress, MOSEK, COIN-OR, scipy.optimize.linprog/milp/qap, cvxpy, PuLP, Pyomo, Eigen, SuiteSparse, Armadillo, LAPACK, BLAS, cuSPARSE, cuBLAS.

To mathematically audit and guarantee compliance across every source file in the repository, run the automated AST import-ban checker:

```bash
python3 tools/check_no_external_solvers.py
```

Expected output:
```text
[PASS] Compliance check successful: No forbidden external solver or linear algebra libraries detected.
```

*(Note: `highspy` is strictly quarantined under `tests/oracle/` solely as an independent verification oracle and is banned from production code).*

---

## 3. Building and Installation

### Prerequisites
- Linux or WSL2 (Ubuntu 22.04 / 24.04 / 26.04)
- CMake $\ge$ 3.20
- GCC / G++ $\ge$ 13 (with C++17 support)
- Python $\ge$ 3.10 with `pybind11`
- *(Optional for GPU acceleration)*: NVIDIA CUDA Toolkit $\ge$ 12.0 and NVIDIA GPU (Compute Capability 8.9 recommended)

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/your-org/sih26119-solver.git
cd sih26119-solver

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Installs pybind11, fastapi, uvicorn, httpx, matplotlib
```

### Step 2: Build C++ Core and Python Bindings

#### Option A: CPU-Only Build (Default)
```bash
mkdir -p build && cd build
cmake -DENABLE_CUDA=OFF ..
cmake --build . -j$(nproc)
cd ..
```

#### Option B: GPU-Accelerated Build (CUDA PDHG Enabled)
```bash
mkdir -p build && cd build
cmake -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_HOST_COMPILER=g++-13 ..
cmake --build . -j$(nproc)
cd ..
```

The compiled shared object (`sih_solver._core.cpython-*.so`) is automatically symlinked into `py/sih_solver/`.

---

## 4. Running Tests

### 4.1 C++ Unit Tests
```bash
cd build
ctest --output-on-failure
cd ..
```

### 4.2 Python Test Battery
Run the full regression test suite (LP, IPM, MILP, Refinery model, explainability, adaptive learning):
```bash
source .venv/bin/activate
export PYTHONPATH="./py:."

pytest tests/ -v
```

---

## 5. Benchmarking and Robustness

### 5.1 Full Benchmark Suite (Netlib, MIPLIB, Maros-Mészáros)
Compares our indigenous solver against the HiGHS reference oracle, computing Shifted Geometric Mean (SGM, $s=1.0\text{s}$) and Dolan-Moré performance profile curves:
```bash
python3 bench/run_full_benchmark_suite.py
```

Benchmark Summary:
| Problem Class | Benchmark Suite | Instances | Indigenous SGM | HiGHS SGM | Win Rate ($\tau=1.0$) | Solvability ($\tau \le 10$) |
|---|---|---|---|---|---|---|
| **LP** | Netlib Standard | 9 | **9.64 ms** | 1.78 ms | 22.2% | **88.9%** |
| **MILP** | MIPLIB Easy Subset | 5 | **984.35 ms** | 2366.91 ms | **80.0%** | **80.0%** |
| **QP** | Maros-Mészáros Convex | 10 | **7.86 ms** | 1.57 ms | **50.0%** | **100.0%** |

*Performance profile chart is generated at `docs/performance_profiles.png`.*

### 5.2 Robustness Hardening Battery
Tests solver stability on numerically pathological formulations (Beale cycling trap, badly scaled matrix with $10^{12}$ coefficient range, near-infeasible razor-thin polytope, and 100-seed Monte Carlo random sparse problems):
```bash
python3 bench/robustness_battery.py
```

Expected result:
```text
Total Seeds Tested : 100
Passed             : 100
Failed             : 0
Pass Rate          : 100.0% (Zero numerical breakdowns, zero memory faults)
```

---

## 6. Starting the Backend REST API

The FastAPI server provides asynchronous problem execution, job polling, telemetry, and structured refinery planning endpoints for web dashboards and plant control systems.

### Start the API Server
```bash
source .venv/bin/activate
export PYTHONPATH="./py:."

python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at:
`http://localhost:8000/docs`

### Key API Endpoints & Example Calls

#### 1. Submit Optimization Job (`POST /api/v1/solve`)
```bash
curl -X POST "http://localhost:8000/api/v1/solve?sync=false" \
  -H "Content-Type: application/json" \
  -d '{
    "mps_data": "NAME LP\nOBJSENSE\n MIN\nROWS\n N OBJ\n L R1\nCOLUMNS\n X1 OBJ 1.0 R1 1.0\nRHS\n RHS1 R1 5.0\nENDATA\n",
    "algorithm": "DualSimplex"
  }'
```
Response:
```json
{
  "job_id": "job_b8cb3b1b",
  "status": "RUNNING",
  "elapsed_ms": 0.0,
  "completed": false,
  "error_message": null
}
```

#### 2. Poll Status & Fetch Solution (`GET /api/v1/jobs/{job_id}/solution`)
```bash
curl "http://localhost:8000/api/v1/jobs/job_b8cb3b1b/solution"
```

#### 3. Solve Refinery Planning Scenario (`POST /api/v1/refinery/plan`)
```bash
curl -X POST "http://localhost:8000/api/v1/refinery/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "crude_prices": {"ArabLight": 75.0, "Brent": 82.0},
    "unit_capacities": {"ADU": 100000.0, "FCC": 35000.0},
    "is_milp": false
  }'
```

---

## 7. Architecture Overview

```text
sih26119-solver/
├── core/                       # Indigenous C++17 Solver Core (Zero external libraries)
│   ├── include/sih/
│   │   ├── algebra/            # Sparse Matrix, Sparse LU, Sparse Cholesky, AMD ordering
│   │   ├── simplex/            # Dual Simplex, Primal Simplex, Steepest-Edge, Harris ratio
│   │   ├── ipm/                # Primal-Dual Interior Point Method (Mehrotra), Crossover
│   │   ├── milp/               # Branch-and-Bound, Gomory cuts, Heuristics, Thread pool
│   │   ├── presolve/           # Bound tightening, singleton, duplicate row reduction
│   │   └── gpu/                # CUDA PDHG first-order solver & hybrid crossover
│   └── src/
├── py/                         # Python Interface & Industrial Applications
│   ├── sih_solver/             # Pybind11 wrapper module
│   ├── refinery/               # Refinery model, SLP nonlinear blender, Explainability, Re-opt
│   ├── learning/               # Feature extraction & adaptive strategy bandit
│   └── api/                    # FastAPI REST application & schemas
├── bench/                      # Benchmark suite & robustness battery
├── docs/                       # Comprehensive documentation & mathematical specifications
└── tools/                      # Compliance checker & environment reporting
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
