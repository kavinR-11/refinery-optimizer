# Telemetry System & JSON Lines Schema

This document specifies the telemetry logging subsystem (`/core/telemetry` and `/py/sih_solver/telemetry.py`), defining the single-line JSON format appended on every solve event.

## 1. Overview & Motivation
To enable downstream offline analysis and train the adaptive strategy-learning layer, every solver invocation records a single, dense JSON line containing:
1. Static problem characteristics (dimensions, sparsity, integrality, quadratic terms).
2. Algorithmic configuration (strategy choices, tolerances, thread allocation).
3. Performance outcomes (solution status, final objective, dual bound, MIP gap).
4. Resource consumption (wall time, CPU time, iteration/node metrics, resident memory).

Telemetry is thread-safe and non-blocking, appending sequentially to `telemetry.jsonl` (or a path specified in `Options`).

---

## 2. Telemetry JSON Schema

Each entry is a single-line JSON string conforming to the following schema:

```json
{
  "timestamp": "2026-09-23T08:30:00.000Z",
  "run_id": "c1f7b9e0-82a1-43e9-a47f-8547382910ab",
  "problem": {
    "name": "blend",
    "sense": "minimize",
    "num_rows": 74,
    "num_cols": 83,
    "num_nonzeros": 522,
    "num_quad_nonzeros": 0,
    "num_integers": 0,
    "num_binaries": 0,
    "density": 0.085
  },
  "strategy": {
    "algorithm": "DualSimplex",
    "pricing_rule": "SteepestEdge",
    "branching_rule": "PseudoCost",
    "node_selection": "BestBound",
    "cut_rounds": 5,
    "presolve": "On",
    "enable_gpu": false,
    "threads": 4
  },
  "tolerances": {
    "primal_feasibility": 1e-6,
    "dual_feasibility": 1e-6,
    "integrality": 1e-5
  },
  "metrics": {
    "status": "Optimal",
    "primal_objective": -30.8125,
    "dual_bound": -30.8125,
    "mip_gap": 0.0,
    "simplex_iterations": 184,
    "barrier_iterations": 0,
    "nodes_explored": 0,
    "runtime_wall_ms": 14.28,
    "runtime_cpu_ms": 13.91,
    "memory_rss_mb": 24.5
  }
}
```

---

## 3. Metric Collection Details

- **Memory Sampling**: Under Linux/WSL2, resident set size (RSS) is queried via `/proc/self/statm` (reading page count multiplied by `sysconf(_SC_PAGESIZE)`), providing sub-microsecond overhead.
- **Timers**: `std::chrono::high_resolution_clock` for wall-clock tracking and `getrusage` / `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` for CPU utilization.
- **Schema Validation**: Unit tests verify that serialized lines parse cleanly with Python's standard `json` module.
