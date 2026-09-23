# Core Data Model & Thread Affinity Architecture

This document describes the algorithms, data structures, numerical conventions, and known limitations of the core mathematical optimization data model (`/core/model`) and thread affinity utilities (`/core/utils`).

## 1. Sparse Matrix Representation (`SparseMatrix`)

### Algorithms & Data Structures
To balance the algorithmic needs of dual simplex (which accesses columns of $A$ and basis columns $B$) and cutting planes / row checks / presolve (which iterate over rows), `SparseMatrix` provides:
- **Compressed Sparse Column (CSC)**:
  - `col_ptr`: Array of length $n + 1$, where `col_ptr[j]` points to the start of column $j$ in `row_ind` and `values`.
  - `row_ind`: Array of row indices of non-zero entries (0-indexed).
  - `values`: Non-zero numerical entries (`double`).
- **Compressed Sparse Row (CSR)**:
  - `row_ptr`: Array of length $m + 1$, where `row_ptr[i]` points to the start of row $i$ in `col_ind` and `values`.
  - `col_ind`: Array of column indices of non-zero entries.
  - `values`: Non-zero numerical entries (`double`).
- **Triplet / Coordinate (COO)** constructor and converter:
  - Accumulates $(i, j, v)$ triplets, sums duplicate entries, eliminates zeros below a numerical threshold $\epsilon_{zero} = 10^{-15}$, and constructs both sorted CSC and CSR representations in $O(nnz + m + n)$ time.
- **Matrix-Vector Operations**:
  - Primal forward multiplication: $y \leftarrow A x$ using CSR for cache-friendly row-wise accumulation.
  - Dual transpose multiplication: $z \leftarrow A^T y$ using CSC for column-wise dot products.
  - Transposition: Direct $O(nnz)$ transposition by swapping roles of CSR and CSC representations.

---

## 2. Optimization Problem Model (`Problem`)

### Data Structures
An instance of `Problem` represents:
$$
\begin{aligned}
\min \quad & c_0 + c^T x + \frac{1}{2} x^T Q x \\
\text{s.t.} \quad & row\_lower \le A x \le row\_upper \\
& col\_lower \le x \le col\_upper \\
& x_j \in \mathbb{Z} \quad \forall j \in \mathcal{I}
\end{aligned}
$$
- `name`: String identifier.
- `sense`: `ObjectiveSense::Minimize` or `ObjectiveSense::Maximize`.
- `obj_offset`: Constant scalar term $c_0$.
- `c`: Objective coefficients vector $\mathbb{R}^n$.
- `A`: Dual-representation `SparseMatrix` (CSC and CSR kept synchronized).
- `row_lower`, `row_upper`: Vectors of length $m$ in $[-\infty, +\infty]$. Equality constraints have $row\_lower_i = row\_upper_i$.
- `col_lower`, `col_upper`: Vectors of length $n$ in $[-\infty, +\infty]$.
- `var_types`: Vector of `VariableType` (`Continuous`, `Binary`, `Integer`, `SemiContinuous`).
- `Q`: Optional sparse symmetric matrix (CSC and CSR) representing $\frac{1}{2} x^T Q x$. If absent, $Q = 0$ (linear program).
- `row_names`, `col_names`: String names with bidirectional $O(1)$ hash maps for lookup.

---

## 3. Solution Model (`Solution`)

### Fields & Enums
- **`SolutionStatus`**:
  - `Optimal`: Solution satisfies primal, dual, and integrality tolerances.
  - `Infeasible`: Proven primal infeasibility (Farkas ray or bounding contradiction).
  - `Unbounded`: Proven dual infeasibility / primal ray.
  - `TimeLimit`: Exceeded configured wall-clock time limit.
  - `IterationLimit`: Exceeded simplex / barrier iteration count.
  - `NodeLimit`: Exceeded branch-and-bound node budget.
  - `NumericalFailure`: Factorization breakdown or unrecoverable numerical instability.
  - `Interrupted`: Terminated by external signal / user request.
  - `Unknown`: Unsolved or initialized state.
- **`BasisStatus`**:
  - `Basic`: Variable or slack is in the working simplex basis.
  - `AtLower`: Non-basic, held at its lower bound.
  - `AtUpper`: Non-basic, held at its upper bound.
  - `Free`: Non-basic free variable.
  - `Superbasic`: Non-basic variable between bounds (used in interior point / QP active set).
- **Vectors**:
  - `x`: Primal variable values $\mathbb{R}^n$.
  - `slack`: Row activity vector $A x \in \mathbb{R}^m$.
  - `row_duals`: Lagrange multipliers $y \in \mathbb{R}^m$.
  - `reduced_costs`: Dual slacks $s = c + Qx - A^T y \in \mathbb{R}^n$.
  - `col_basis`: Basis status for each column.
  - `row_basis`: Basis status for each row.
- **Scalars & Metrics**:
  - `primal_obj`: Evaluated primal objective value.
  - `dual_bound`: Best proven dual bound.
  - `mip_gap`: Relative gap $|primal\_obj - dual\_bound| / (1 + |primal\_obj|)$.
  - `simplex_iterations`: Total pivot count.
  - `barrier_iterations`: Total interior-point step count.
  - `nodes_explored`: Total branch-and-bound nodes processed.
  - `time_wall_sec`, `time_cpu_sec`, `time_presolve_sec`, `time_solver_sec`.

---

## 4. Solver Options & Adaptive Strategy Configuration

### Tolerances & Budgets (`Options`)
- `primal_feasibility_tol`: Maximum allowable constraint or bound violation (Default: $10^{-6}$).
- `dual_feasibility_tol`: Maximum allowable reduced cost / multiplier sign violation (Default: $10^{-6}$).
- `integrality_tol`: Maximum allowable distance from integer for integer variables (Default: $10^{-5}$).
- `zero_tol`: Numerical zero threshold (Default: $10^{-12}$).
- `time_limit_sec`: Wall-clock time budget in seconds (Default: $\infty$).
- `iteration_limit`: Maximum iterations (Default: $\infty$).
- `node_limit`: Maximum branch-and-bound nodes (Default: $\infty$).
- `threads`: Number of worker threads (Default: 0 $\rightarrow$ auto-detect).

### Adaptive Strategy (`StrategyConfig`)
All algorithmic choices are centralized in `StrategyConfig` to facilitate reinforcement learning / adaptive selection in later phases:
- `algorithm`: `Auto`, `PrimalSimplex`, `DualSimplex`, `Barrier`, `BranchAndBound`.
- `pricing_rule`: `Dantzig`, `SteepestEdge`, `Devex`, `BFRT`.
- `branching_rule`: `MostFractional`, `PseudoCost`, `StrongBranching`, `Reliability`.
- `node_selection`: `BestBound`, `DepthFirst`, `BestEstimate`.
- `presolve`: `Off`, `On`, `Aggressive`.
- `cut_rounds`: Integer (0 to $K$).
- `enable_gpu`: Boolean flag.

---

## 5. Thread Affinity Utilities (`affinity.hpp`)

On hybrid architectures (such as the 14th/13th Gen Intel Core HX series with 6 P-cores / 12 threads and 8 E-cores / 8 threads), scheduling variability between P-cores and E-cores can cause inconsistent benchmark timings.
- `sih::utils::pin_thread_to_core(int core_id)` wraps Linux `sched_setaffinity`.
- `sih::utils::get_current_core()` reads current thread CPU placement.
- Provides thread isolation for consistent micro-benchmarks.

---

## 6. Numerical Choices & Known Limitations
- Standard 64-bit IEEE 754 floats (`double`) are used throughout the C++ engine.
- 64-bit signed integers (`int64_t`) are used for dimension counts and indexing to avoid 32-bit overflow on large instances.
- Zero elements below `zero_tol` are explicitly flushed to zero to prevent catastrophic cancellation.
- In Phase 0, linear algebra operations are limited to sparse matrix-vector products; factorizations (LU, Cholesky) are scheduled for subsequent phases.
