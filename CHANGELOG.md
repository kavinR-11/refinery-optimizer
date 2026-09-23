# Changelog

All notable changes to the SIH 26119 Indigenous Optimization Solver will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - Phase 0 Foundation Complete

### Added
- **Build System**:
  - `CMakeLists.txt`: Configured for ISO C++17 with system GCC 15.2, Ninja generator, OpenMP support, pybind11 integration, and conditional CUDA 12.4 (`ENABLE_CUDA` default OFF) with host compiler `g++-13` and architecture `sm_89`.
  - `build.sh`: Single command to configure and compile core library, Python bindings, and test executables.
  - `test.sh`: Single master test command enforcing compliance checks, CTest C++ unit tests, and Python test suite.
  - `docs/build_system.md`: Design doc for build system architecture.

- **Developer Tools & Compliance**:
  - `tools/check_no_external_solvers.py`: Enforces Hard Rules 1-3. Verified to pass on clean tree and fail on deliberate violations.
  - `tools/env_report.py`: System diagnostic script inspecting OS, CPU hybrid architecture (lscpu), memory, GPU/VRAM (nvidia-smi), CUDA toolkit (nvcc), and compiler toolchains.
  - `docs/environment.md`: Detailed baseline environment specification.
  - `tools/download_netlib.py`, `tools/download_miplib.py`, `tools/download_maros_meszaros.py`, `tools/prepare_datasets.py`: Benchmark dataset preparation scripts with offline fallback instructions.
  - `docs/tools.md`: Design doc for developer tooling.
  - `desingn/stitch_industrial_optimization_solver_dashboard/`: Full industrial UI dashboard design suite (HTML/CSS + mockups + design tokens) covering Solve Overview, Refinery Planning, Solver Internals, Benchmarks & Perf, Strategy Learning Log, and System Performance & Hardware Acceleration (GPU PDHG crossover analysis for RTX 4060).

- **Core Data Model & Thread Affinity**:
  - `sih::model::SparseMatrix`: Dual Compressed Sparse Column (CSC) and Compressed Sparse Row (CSR) matrix representation with $O(nnz)$ triplet conversion, matrix-vector multiplication, transpose multiplication, and $O(1)$ transposition.
  - `sih::model::Problem`: Mathematical programming problem model with linear objective $c$, constant $c_0$, sense (Min/Max), dual-representation $A$, row lower/upper bounds, column lower/upper bounds, variable integrality flags (`VariableType`), optional symmetric quadratic objective $Q$, and bidirectional name lookups.
  - `sih::model::Solution`: Comprehensive solution model capturing `SolutionStatus`, `BasisStatus`, primal variables $x$, row activity $Ax$, Lagrange multipliers $y$, reduced costs $s$, basis statuses, objective value, dual bound, MIP gap, iteration counts, and execution timings.
  - `sih::model::Options` & `sih::model::StrategyConfig`: Modular solver configuration governing resource limits, numerical tolerances, and adaptive strategy parameters (pricing rule, branching rule, node selection, presolve mode, cut rounds, GPU toggle).
  - `sih::utils::affinity`: Thread affinity utility wrapping Linux `sched_setaffinity` and `sched_getcpu` for reproducible microbenchmarks on hybrid P-core/E-core CPUs.
  - `docs/data_model.md`: Design doc for core mathematical data structures and affinity utilities.

- **MPS & QPS Reader and Writer**:
  - `sih::io::read_mps` & `sih::io::write_mps`: Full parser and serializer supporting fixed format and free format, `NAME`, `OBJSENSE`, `ROWS` (N, L, G, E), `COLUMNS` (with `INTORG`/`INTEND` integer markers), `RHS`, `RANGES`, `BOUNDS` (`LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `BV`, `UI`, `LI`), and `QUADOBJ`/`QMATRIX` extensions.
  - `docs/mps_io.md`: Design doc for MPS/QPS input-output engine.

- **Telemetry System**:
  - `sih::telemetry`: Appends dense, single-line JSON records on every solve event capturing problem features, strategy configuration, numerical tolerances, solution status, objective, iterations, runtime, and Linux resident memory (RSS).
  - `docs/telemetry.md`: Specification of the JSON lines telemetry schema.

- **Independent Solution Checker**:
  - C++ checker (`sih::checker::check_solution`) and Python checker (`FloatChecker` & `RationalChecker`): Computes primal constraint and bound residuals, integrality violations, objective value consistency, dual stationarity ($c + Qx - A^T y - s = 0$), and complementary slackness without relying on solver state.
  - `RationalChecker`: Employs Python `fractions.Fraction` for exact, zero-floating-point-error verification on rational test instances.
  - `docs/checker.md`: Design doc for independent verification formulations.

- **HiGHS Oracle Harness & Hand-Made Toy Suite**:
  - `tests/oracle/highs_oracle.py`: Reference solver oracle wrapping `highspy.Highs` (strictly confined to `/tests/oracle/` per Hard Rules 2 & 3).
  - `docs/oracle.md`: Design doc for oracle wrapper and status mapping.
  - 10 hand-crafted toy test cases in `data/toy/`:
    1. `toy01_lp_simple.mps`: 2-var LP with unique optimum ($x^* = (1.5, 2.5)$, obj = -6.5).
    2. `toy02_lp_infeasible.mps`: Contradictory row bounds ($x_1+x_2 \le 1$ and $x_1+x_2 \ge 2$).
    3. `toy03_lp_unbounded.mps`: Recession cone unbounded LP.
    4. `toy04_lp_equality_ranges.mps`: LP with equality constraint and `RANGES` section ($x^* = (0, 4)$, obj = 16.0).
    5. `toy05_lp_bounds.mps`: LP with `FX`, `LO`, `UP`, and negative variable bounds ($x^* = (2, 5, -2)$, obj = -14.0).
    6. `toy06_milp_knapsack.mps`: Binary 0-1 knapsack ($x^* = (1, 1, 0)$, obj = -11.0).
    7. `toy07_milp_mixed_integer.mps`: Mixed integer problem with continuous, general integer (`UI`), and binary (`BV`) variables ($x^* = (4, 0, 1)$, obj = 6.0).
    8. `toy08_qp_convex_diagonal.mps`: Strictly convex QP with diagonal $Q$ ($x^* = (1.5, 2.5)$, obj = -12.5).
    9. `toy09_qp_convex_general.mps`: Convex QP with non-diagonal symmetric positive-definite $Q$ ($x^* = (1/3, 4/3)$, obj = -7/3).
    10. `toy10_milp_infeasible_integer.mps`: MILP whose LP relaxation is feasible ($0.2 \le x \le 0.8$) but integer infeasible.
  - `tests/test_toy_suite.py`: Automated gate verification confirming MPS reader, HiGHS oracle, float checker, and rational checker agree across all 10 toy problems.
