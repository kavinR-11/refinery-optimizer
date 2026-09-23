# Oracle Harness Architecture (HiGHS Integration)

This document describes the design, status mapping, and comparison protocols for the HiGHS reference oracle harness (`/tests/oracle`).

## 1. Role & Compliance Scope
In accordance with Hard Rules 2 and 3:
- HiGHS (`highspy`) is imported **strictly and exclusively** inside `/tests/oracle/` (and `/bench/` for comparison baselines).
- It serves as the trusted reference standard to cross-validate reader correctness, solution statuses, and optimal objective values on all benchmark and toy suites.

---

## 2. Oracle Wrapper Architecture (`highs_oracle.py`)

### Interface
```python
class HighsOracle:
    def solve_mps(self, mps_path: str, options: dict = None) -> OracleResult:
        ...
```

### Output Record (`OracleResult`)
- `status`: Unified status string (`Optimal`, `Infeasible`, `Unbounded`, etc.).
- `objective_value`: Real scalar or `None`.
- `primal_solution`: List of floats $x$.
- `row_duals`: List of floats $y$ (for LP/QP).
- `reduced_costs`: List of floats $s$ (for LP/QP).
- `simplex_iterations`: Pivot count.
- `wall_time`: Elapsed seconds.

### Status Translation
| HiGHS ModelStatus | Unified Solver Status |
| :--- | :--- |
| `kOptimal` | `Optimal` |
| `kInfeasible` | `Infeasible` |
| `kUnbounded` | `Unbounded` |
| `kTimeLimit` | `TimeLimit` |
| `kIterationLimit` | `IterationLimit` |
| Other | `NumericalFailure` / `Unknown` |

---

## 3. Comparison & Verification Protocol
1. Check status equivalence: `oracle.status == candidate.status`.
2. For bounded optimal instances:
   $$|obj_{candidate} - obj_{oracle}| \le 10^{-5} \cdot (1 + |obj_{oracle}|)$$
3. Cross-validate oracle candidate with the independent checker (`/tests/checker`).
