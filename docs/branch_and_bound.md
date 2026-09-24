# Branch-and-Bound Engine for Mixed-Integer Linear Programming (MILP)

## 1. Overview & Mathematical Foundations

Mixed-Integer Linear Programming (MILP) seeks to solve:
$$\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^T x \\
\text{subject to} \quad & A x \le b \quad (\text{or } l \le A x \le u) \\
& l_x \le x \le u_x \\
& x_j \in \mathbb{Z} \quad \forall j \in \mathcal{I}
\end{aligned}$$
where $\mathcal{I} \subseteq \{0, \dots, n-1\}$ denotes the set of integer/binary variables.

The branch-and-bound (B&B) method systematically divides the feasible region by creating child subproblems (branching) and computing valid lower bounds via LP relaxations (bounding). When the LP relaxation at a node is infeasible, inferior to the current best known integer solution (the *incumbent*), or naturally integer-feasible, the node is *pruned*.

### 1.1 Dual Simplex Integration & Warm-Starting
Dual simplex is the optimal LP engine for branch-and-bound tree search because branching adds variable bound tightenings:
- Left branch: $x_j \le \lfloor x_j^* \rfloor$
- Right branch: $x_j \ge \lceil x_j^* \rceil$

A bound tightening retains **dual feasibility** ($c_N - A_N^T y \ge 0$). Consequently, the parent's optimal basis $\mathcal{B}$ is directly dual feasible for both children. The child node re-optimizes in only a few dual pivots (typically 1–10 pivots) rather than solving from scratch (cold start: hundreds of pivots).

---

## 2. Core Architecture & Data Structures

```
                       +----------------------------------+
                       |           MILP Solver            |
                       +----------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                |                               |
       v                                v                               v
+--------------+               +------------------+             +-----------------+
| MILP Presolve|               |  Gomory Root     |             | Rounding/Diving |
| (Tightening, |               |  Cut Generator   |             |   Heuristics    |
|   Probing)   |               +------------------+             +-----------------+
+--------------+                        |                                |
       |                                v                                v
       +---------------------> [ Root LP Relaxation ] <------------------+
                                        |
                                        v
                       +----------------------------------+
                       |      Node Queue (Work Pool)      |
                       |  (BestFirst / DepthFirst Order)  |
                       +----------------------------------+
                                        |
                         +--------------+---------------+
                         |                              |
                         v (Worker Thread 1)            v (Worker Thread k)
                 +----------------+             +----------------+
                 |  Dual Simplex  |             |  Dual Simplex  |
                 |   Warm-Start   |             |   Warm-Start   |
                 +----------------+             +----------------+
```

### 2.1 Node Representation (`BnBNode`)
```cpp
struct BnBNode {
    int64_t node_id;
    int64_t parent_id;
    int64_t depth;
    
    // Subproblem bound modifications relative to root problem
    std::vector<double> col_lower;
    std::vector<double> col_upper;
    
    // Warm-start basis inherited from parent
    std::vector<model::BasisStatus> col_basis;
    std::vector<model::BasisStatus> row_basis;
    
    // LP relaxation solution at this node
    double lp_objective;
    std::vector<double> lp_x;
    bool is_feasible;
    
    // Branching history
    int64_t branched_var;
    double branched_bound;
    bool is_lower_bound_branch;
};
```

### 2.2 Global State & Incumbent Management
```cpp
struct IncumbentTracker {
    std::mutex mtx;
    bool has_incumbent{false};
    double best_objective{model::SIH_INFINITY};
    std::vector<double> best_solution;
    
    // Dual bound (global lower bound for minimization)
    double global_dual_bound{-model::SIH_INFINITY};
    
    // Statistics
    std::atomic<int64_t> nodes_explored{0};
    std::atomic<int64_t> total_simplex_iterations{0};
    std::atomic<bool> time_limit_reached{false};
    std::atomic<bool> node_limit_reached{false};
};
```

---

## 3. Node Selection Strategies

Node selection determines which active subproblem to solve next:

1. **Best-First Search (`BestFirst`)**:
   - Order nodes by ascending LP objective: $\text{argmin}_{u \in \mathcal{U}} z_{LP}(u)$.
   - Priority queue with min-heap on `lp_objective`.
   - **Property**: Minimizes the total number of nodes required to prove optimality. Guarantees that the global dual bound advances monotonically.

2. **Depth-First Search (`DepthFirst`)**:
   - Order nodes by descending tree depth: $\text{argmax}_{u \in \mathcal{U}} \text{depth}(u)$ (LIFO stack).
   - **Property**: Conserves memory ($O(\text{depth})$ active nodes), traverses rapidly to leaf nodes to discover early feasible solutions, and maximizes basis similarity between parent and child for ultra-fast warm-starting.

Configured via `StrategyConfig::node_selection`:
```cpp
enum class NodeSelection {
    BestFirst = 0,
    DepthFirst = 1
};
```

---

## 4. Branching Rules

When an LP solution $x^*$ has fractional values for integer variables ($j \in \mathcal{I}$ such that $x_j^* \notin \mathbb{Z}$), a branching variable must be chosen:

### 4.1 Most-Fractional Branching (`MostFractional`)
- Computes fractionality: $f_j = x_j^* - \lfloor x_j^* \rfloor$.
- Selects variable closest to $0.5$:
  $$j^* = \text{argmin}_{j \in \mathcal{I}, f_j > \epsilon} |f_j - 0.5|$$
- Simple, zero-overhead heuristic that divides the feasible set near its midpoint.

### 4.2 Pseudocost Branching (`Pseudocost`)
Pseudocost branching tracks the historical rate of objective degradation per unit bound change:
- For down-branch ($x_j \le \lfloor x_j^* \rfloor$):
  $$\Delta z_j^- = z_{child}^- - z_{parent}, \quad \Delta x_j^- = x_j^* - \lfloor x_j^* \rfloor, \quad P_j^- = \frac{\Delta z_j^-}{\Delta x_j^-}$$
- For up-branch ($x_j \ge \lceil x_j^* \rceil$):
  $$\Delta z_j^+ = z_{child}^+ - z_{parent}, \quad \Delta x_j^+ = \lceil x_j^* \rceil - x_j^*, \quad P_j^+ = \frac{\Delta z_j^+}{\Delta x_j^+}$$

The pseudocosts are updated as exponentially smoothed running averages across all tree branches:
$$\bar{P}_j^- \leftarrow \frac{N_j^- \bar{P}_j^- + P_j^-}{N_j^- + 1}, \quad \bar{P}_j^+ \leftarrow \frac{N_j^+ \bar{P}_j^+ + P_j^+}{N_j^+ + 1}$$

The score for variable $j$ is computed using the standard MIP scoring function (Achterberg 2007):
$$\text{score}(j) = (1 - \mu) \min(D_j^-, D_j^+) + \mu \max(D_j^-, D_j^+)$$
where $D_j^- = \bar{P}_j^- (x_j^* - \lfloor x_j^* \rfloor)$, $D_j^+ = \bar{P}_j^+ (\lceil x_j^* \rceil - x_j^*)$, and $\mu = 1/6$.
- Uninitialized variables default to most-fractional branching until at least one up/down sample is recorded.

---

## 5. Primal Heuristics: Rounding & Diving

Finding an early integer incumbent allows immediate pruning of large subtrees:

### 5.1 Simple & Shift Rounding
- For each fractional variable $x_j^* \notin \mathbb{Z}$, round to nearest integer $\lfloor x_j^* + 0.5 \rfloor$.
- Check row feasibility $l \le A x_{rounded} \le u$.
- If feasible and $c^T x_{rounded} < z_{incumbent}$, accept as new incumbent.

### 5.2 Fractional Diving Heuristic
- Starting from the LP relaxation solution:
  1. Identify the fractional variable $x_j^*$ with fractionality closest to 0 or 1.
  2. Fix $x_j$ to its nearest integer: $l_j \leftarrow \text{round}(x_j^*)$, $u_j \leftarrow \text{round}(x_j^*)$.
  3. Re-optimize the LP with dual simplex warm-start.
  4. If LP becomes infeasible, abort dive.
  5. If all integer variables are integer, record candidate incumbent and return.
  6. Repeat up to a configurable maximum dive depth (e.g. 50 iterations).

---

## 6. MILP Presolve

Before tree search, MILP-specific reductions exploit integrality:

1. **Integer Bound Tightening**:
   - For all $j \in \mathcal{I}$:
     $$l_j \leftarrow \lceil l_j - \epsilon \rceil, \quad u_j \leftarrow \lfloor u_j + \epsilon \rfloor$$
   - If $l_j > u_j$, problem is immediately infeasible.

2. **Binary Variable Probing**:
   - For binary variables $x_j \in \{0, 1\}$:
     - Temporarily fix $x_j = 0$, evaluate feasibility via row activity bounds. If infeasible, fix $x_j = 1$ permanently.
     - Temporarily fix $x_j = 1$, evaluate feasibility. If infeasible, fix $x_j = 0$ permanently.
     - Detect implied bound tightenings: $x_j = 1 \implies x_k \le u_k'$.

---

## 7. Multithreaded Tree Search via Indigenous Thread Pool

Per the hard environment constraints, **no OpenMP** is used for tree search. A custom, indigenous thread pool is implemented with standard C++17 concurrency (`std::thread`, `std::mutex`, `std::condition_variable`):

### 7.1 Thread-Safe Priority Work Queue
- Synchronized queue guarded by `std::mutex`.
- Worker threads block on `std::condition_variable` when queue is empty, waking up when child nodes are pushed or termination is signaled.
- Active worker counter detects global termination: when queue is empty AND all workers are idle, the search tree is completely explored.

### 7.2 Deterministic Single-Threaded Mode
- When `threads == 1`, execution follows a deterministic traversal order, ensuring bitwise identical results and node sequences across runs.
- Multi-core mode supports 1, 4, 8, and all-core (14th-gen HX) execution.

---

## 8. Termination & Result Reporting

The search terminates upon:
1. **Optimality Proved**: Active node queue is empty.
2. **Time Limit**: `time_limit_sec` exceeded.
3. **Node Limit**: `node_limit` exceeded.
4. **MIP Gap Tolerance**: Relative gap $\le \text{integrality\_tol}$.

The solution returned contains:
- `sol.status`: `Optimal`, `TimeLimitReached`, `Infeasible`, or `Unbounded`.
- `sol.x`: Best integer feasible solution vector.
- `sol.primal_objective`: $c^T x_{incumbent}$.
- `sol.dual_bound`: Global lower bound from all active and pruned nodes.
- `sol.mip_gap`: $\frac{|z_{incumbent} - z_{dual}|}{10^{-10} + |z_{incumbent}|}$.
- `sol.nodes_explored`: Total branch-and-bound nodes processed.
