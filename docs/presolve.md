# Presolve and Postsolve Architecture (Phase 1)

This document specifies the design, mathematical foundations, data structures, and algorithms for the indigenous LP Presolve and Postsolve subsystem for Smart India Hackathon problem **SIH 26119**.

Per **Hard Rule 1**, the presolve and postsolve engine is implemented completely from scratch in standard C++17 (`/core/src/presolve/` and `/core/include/sih/presolve/`) without third-party dependencies or external solvers.

---

## 1. Mathematical Formulation & Invariants

Consider an optimization problem in general bounded form:

$$\min_{x} \quad c^T x + c_0$$
$$\text{subject to} \quad l \le A x \le u$$
$$lb \le x \le ub$$

where $A \in \mathbb{R}^{m \times n}$ is a sparse matrix, $c, lb, ub, x \in \mathbb{R}^n$, and $l, u \in \mathbb{R}^m$. The bounds $lb_j, l_i \in \mathbb{R} \cup \{-\infty\}$ and $ub_j, u_i \in \mathbb{R} \cup \{+\infty\}$.

### The Reversibility Invariant

Presolve transforms the original problem $\mathcal{P}_0 = (A_0, c_0, l_0, u_0, lb_0, ub_0)$ through a sequence of $K$ elementary reductions:

$$\mathcal{P}_0 \xrightarrow{\mathcal{R}_1} \mathcal{P}_1 \xrightarrow{\mathcal{R}_2} \cdots \xrightarrow{\mathcal{R}_K} \mathcal{P}_K$$

yielding a smaller, better-conditioned problem $\mathcal{P}_K$.

Every reduction $\mathcal{R}_k$ pushes an explicit, reversible transformation record onto a **LIFO Postsolve Stack**:

$$\mathcal{S} = [\mathcal{R}_1, \mathcal{R}_2, \ldots, \mathcal{R}_K]$$

When the reduced problem $\mathcal{P}_K$ is solved by the simplex engine, yielding optimal solution $\mathcal{S}_K = (x_K, y_K, s_K, \mathcal{B}_K)$, **Postsolve unwinds the stack in reverse order** ($\mathcal{R}_K \to \mathcal{R}_1$):

$$(x_K, y_K, s_K, \mathcal{B}_K) \xrightarrow{\mathcal{R}_K^{-1}} \cdots \xrightarrow{\mathcal{R}_1^{-1}} (x_0, y_0, s_0, \mathcal{B}_0)$$

**Exact Invariants Maintained Upon Full Reconstruction**:
1. **Primal Feasibility**: $l_0 - \epsilon_{\text{feas}} \le A_0 x_0 \le u_0 + \epsilon_{\text{feas}}$ and $lb_0 - \epsilon_{\text{feas}} \le x_0 \le ub_0 + \epsilon_{\text{feas}}$.
2. **Dual Stationarity**: $c_0 - A_0^T y_0 - s_0 = 0$ (within $\| \cdot \|_\infty \le \epsilon_{\text{dual}}$).
3. **Complementary Slackness**:
   - If $x_{0, j} > lb_{0, j} + \epsilon_{\text{zero}}$ and $x_{0, j} < ub_{0, j} - \epsilon_{\text{zero}}$, then $s_{0, j} = 0$.
   - If $s_{0, j} > \epsilon_{\text{zero}}$, then $x_{0, j} = lb_{0, j}$.
   - If $s_{0, j} < -\epsilon_{\text{zero}}$, then $x_{0, j} = ub_{0, j}$.
   - If $(A_0 x_0)_i > l_{0, i} + \epsilon_{\text{zero}}$ and $(A_0 x_0)_i < u_{0, i} - \epsilon_{\text{zero}}$, then $y_{0, i} = 0$.
4. **Valid Simplex Basis**: $\mathcal{B}_0$ contains exactly $m_0$ basic variables that form a non-singular basis matrix $B_0$.

---

## 2. Catalog of Presolve Reductions

Each reduction checks for feasibility, performs bound tightening or dimension removal, and pushes an undo operation.

### 2.1 Empty Rows
- **Detection**: Row $i$ has no nonzeros ($A_{i \cdot} = 0$).
- **Feasibility Check**: Must have $l_i \le 0 \le u_i$.
  - If $0 < l_i$ or $0 > u_i$: **Primal Infeasible**. Return Farkas ray $y = \pm e_i$.
- **Action**: Drop row $i$.
- **Postsolve Recovery**:
  - Primal: Row activity $(A x)_i = 0$.
  - Dual: Multiplier $y_i = 0.0$.
  - Basis: Row basis status is set to `AtLower` (if $l_i = 0$), `AtUpper` (if $u_i = 0$), or `Free`.

### 2.2 Empty Columns
- **Detection**: Column $j$ has no nonzeros ($A_{\cdot j} = 0$).
- **Objective Analysis**:
  - If $c_j > 0$: Set $x_j = lb_j$. If $lb_j = -\infty$, the problem is **Primal Unbounded** (dual infeasible).
  - If $c_j < 0$: Set $x_j = ub_j$. If $ub_j = +\infty$, the problem is **Primal Unbounded**.
  - If $c_j = 0$: Set $x_j = 0$ (or clamp to $[lb_j, ub_j]$).
- **Action**: Remove column $j$. Accumulate $c_j x_j$ into objective constant offset $c_0$.
- **Postsolve Recovery**:
  - Primal: Restore $x_j$ to its bound.
  - Dual: Reduced cost $s_j = c_j$.
  - Basis: `AtLower` (if $x_j = lb_j$), `AtUpper` (if $x_j = ub_j$), or `Free`.

### 2.3 Fixed Variables
- **Detection**: Column $j$ has $lb_j = ub_j = v$.
- **Action**:
  - For each nonzero $A_{ij} \ne 0$, update row bounds:
    $$l_i \leftarrow l_i - A_{ij} v, \quad u_i \leftarrow u_i - A_{ij} v$$
  - Update objective offset: $c_0 \leftarrow c_0 + c_j v$.
  - Remove column $j$ from matrix $A$.
- **Postsolve Recovery**:
  - Primal: $x_j = v$.
  - Dual: Compute reduced cost $s_j = c_j - \sum_{i} A_{ij} y_i$.
  - Basis: `AtLower` (or `AtUpper`).

### 2.4 Row Singletons
- **Detection**: Row $i$ has exactly one nonzero entry $A_{ij} \ne 0$.
- **Implied Variable Bounds**:
  $$l_i \le A_{ij} x_j \le u_i$$
  - If $A_{ij} > 0$:
    $$lb'_j = \frac{l_i}{A_{ij}}, \quad ub'_j = \frac{u_i}{A_{ij}}$$
  - If $A_{ij} < 0$:
    $$lb'_j = \frac{u_i}{A_{ij}}, \quad ub'_j = \frac{l_i}{A_{ij}}$$
- **Intersection**:
  $$lb_j^{\text{new}} = \max(lb_j, lb'_j), \quad ub_j^{\text{new}} = \min(ub_j, ub'_j)$$
  If $lb_j^{\text{new}} > ub_j^{\text{new}} + \epsilon_{\text{feas}}$: **Primal Infeasible**.
- **Action**: Tighten column $j$ bounds to $[lb_j^{\text{new}}, ub_j^{\text{new}}]$ and drop row $i$.
- **Postsolve Recovery**:
  - Variable $x_j$ and reduced cost $s'_j$ known from solved problem.
  - Dual multiplier $y_i$ is chosen to satisfy dual stationarity for column $j$:
    $$c_j - \sum_{k \ne i} A_{kj} y_k - A_{ij} y_i - s_j = 0 \implies y_i = \frac{c_j - \sum_{k \ne i} A_{kj} y_k - s_j}{A_{ij}}$$
  - If the new bound $lb'_j$ was active, $y_i$ carries the bound shadow price; $s_j$ is adjusted to reflect the original bounds $[lb_j, ub_j]$.

### 2.5 Column Singletons (Implied Slacks)
- **Detection**: Column $j$ appears in exactly one row $i$ with coefficient $A_{ij}$, and $lb_j = -\infty, ub_j = +\infty$ (free column singleton).
- **Transformation**: Variable $x_j$ can absorb all variations in row $i$:
  $$x_j = \frac{r_i - \sum_{k \ne j} A_{ik} x_k}{A_{ij}}$$
  where $r_i \in [l_i, u_i]$ is the row activity.
- **Substitution**: Substitute $x_j$ into objective $c^T x$. Row $i$ and column $j$ are eliminated.
- **Postsolve Recovery**:
  - Reconstruct $x_j$ from other optimal primal values.
  - Multiplier $y_i = c_j / A_{ij}$.
  - Basis: Variable $j$ enters the basis as `Basic`.

### 2.6 Forcing and Redundant Rows
For each row $i$, compute implied activity bounds from variable bounds:
$$L_i = \sum_{j: A_{ij} > 0} A_{ij} lb_j + \sum_{j: A_{ij} < 0} A_{ij} ub_j$$
$$U_i = \sum_{j: A_{ij} > 0} A_{ij} ub_j + \sum_{j: A_{ij} < 0} A_{ij} lb_j$$

1. **Infeasible Row**:
   If $L_i > u_i + \epsilon_{\text{feas}}$ or $U_i < l_i - \epsilon_{\text{feas}}$, the row can never be satisfied. Declare **Primal Infeasible** with Farkas certificate $y = e_i$ (or $-e_i$).
2. **Forcing Row**:
   - If $|L_i - u_i| \le \epsilon_{\text{feas}}$: To satisfy $\sum A_{ij} x_j \le u_i$, every variable must be fixed to its extreme bound:
     - For $A_{ij} > 0$, $x_j$ is fixed to $lb_j$.
     - For $A_{ij} < 0$, $x_j$ is fixed to $ub_j$.
   - If $|U_i - l_i| \le \epsilon_{\text{feas}}$: Symmetrically, every variable is fixed to its opposite bound.
   - All participating variables become fixed; row $i$ is dropped.
3. **Redundant Row**:
   If $L_i \ge l_i - \epsilon_{\text{feas}}$ and $U_i \le u_i + \epsilon_{\text{feas}}$, row $i$ is unconditionally satisfied by any bounded point.
   - **Action**: Drop row $i$.
   - **Postsolve**: $y_i = 0.0$, basis status `Free` or inactive bound.

### 2.7 Bound Tightening (Implied Bounds Propagation)
For constraint $l_i \le \sum_{k} A_{ik} x_k \le u_i$, isolate term for variable $j$:
- If $A_{ij} > 0$:
  $$x_j \le \frac{u_i - L_{i, \setminus j}}{A_{ij}}, \quad x_j \ge \frac{l_i - U_{i, \setminus j}}{A_{ij}}$$
- If $A_{ij} < 0$:
  $$x_j \le \frac{l_i - U_{i, \setminus j}}{A_{ij}}, \quad x_j \ge \frac{u_i - L_{i, \setminus j}}{A_{ij}}$$
where $L_{i, \setminus j} = L_i - \text{term}_j(L)$ and $U_{i, \setminus j} = U_i - \text{term}_j(U)$.
- Bounds are updated in-place; if tight bounds allow fixing $lb_j = ub_j$, the variable is queued for fixed-variable elimination.

### 2.8 Duplicate Rows
- **Detection**: Rows $p$ and $q$ are scalar multiples $A_{q \cdot} = \alpha A_{p \cdot}$.
- **Consistency**:
  Scale bounds: $l'_q = l_q / \alpha$, $u'_q = u_q / \alpha$ (swapping if $\alpha < 0$).
  Compute intersection $[l^*, u^*] = [\max(l_p, l'_q), \min(u_p, u'_q)]$.
  - If $l^* > u^* + \epsilon_{\text{feas}}$: **Primal Infeasible**.
- **Action**: Update row $p$ bounds to $[l^*, u^*]$; drop row $q$.
- **Postsolve Recovery**:
  Assign optimal dual multiplier $y^*$ of the merged row to whichever constraint was actively binding:
  - If $l_p$ or $u_p$ was active, $y_p = y^*, y_q = 0$.
  - If $l'_q$ or $u'_q$ was active, $y_p = 0, y_q = y^* / \alpha$.

### 2.9 Dominated and Duplicate Columns
- **Detection**: Columns $j_1, j_2$ have $A_{\cdot j_1} = \alpha A_{\cdot j_2}$.
- **Dominance Criterion**: If cost $c_{j_1} \ge \alpha c_{j_2}$ and variable bounds permit, one column dominates the other in the objective sense and can be fixed to its non-active bound.

---

## 3. Postsolve Stack & Data Structures

```cpp
namespace sih {
namespace presolve {

enum class ReductionType {
    EmptyRow,
    EmptyCol,
    FixedCol,
    RowSingleton,
    FreeColSingleton,
    ForcingRow,
    RedundantRow,
    TightenedBounds,
    DuplicateRow,
    DuplicateCol
};

struct PresolveUndo {
    ReductionType type;
    int64_t row_idx{-1};
    int64_t col_idx{-1};
    double old_lower{0.0};
    double old_upper{0.0};
    double old_col_lower{0.0};
    double old_col_upper{0.0};
    double scalar{1.0};
    double obj_coeff{0.0};
    std::vector<int64_t> affected_indices;
    std::vector<double> affected_values;
};

class PresolveStack {
public:
    void push(PresolveUndo undo) { m_stack.push_back(std::move(undo)); }
    bool empty() const noexcept { return m_stack.empty(); }
    size_t size() const noexcept { return m_stack.size(); }
    const std::vector<PresolveUndo>& entries() const noexcept { return m_stack; }

private:
    std::vector<PresolveUndo> m_stack;
};

} // namespace presolve
} // namespace sih
```

---

## 4. Postsolve Reconstruction Algorithm

```text
Algorithm Postsolve:
Input: Presolved solution (x_presolved, y_presolved, s_presolved, basis_presolved)
       Original problem P_orig
       PresolveStack stack
Output: Full original solution (x_orig, y_orig, s_orig, basis_orig)

1. Initialize x_orig, y_orig, s_orig, basis_orig to unreduced dimensions.
2. Scatter presolved solution vectors into active positions in original arrays.
3. For each undo record in REVERSE order from stack:
     Switch(undo.type):
       Case EmptyRow:
         y_orig[undo.row_idx] = 0.0
         basis_orig.row[undo.row_idx] = (l_i == 0 ? AtLower : Free)
       Case EmptyCol:
         x_orig[undo.col_idx] = undo.old_lower (or clamped bound)
         s_orig[undo.col_idx] = undo.obj_coeff
         basis_orig.col[undo.col_idx] = AtLower
       Case FixedCol:
         x_orig[undo.col_idx] = undo.old_col_lower
         s_orig[undo.col_idx] = c[j] - dot(A.col(j), y_orig)
         basis_orig.col[undo.col_idx] = AtLower
       Case RowSingleton:
         Recover y_orig[undo.row_idx] from dual stationarity relation of column j.
         Adjust s_orig[undo.col_idx] for original bounds.
       Case FreeColSingleton:
         x_orig[undo.col_idx] = (r_i - sum_{k != j} A_ik x_k) / A_ij
         y_orig[undo.row_idx] = c_j / A_ij
         basis_orig.col[undo.col_idx] = Basic
       Case DuplicateRow:
         Split dual y* between rows p and q based on which bound is active.
       ...
4. Compute final row slacks: slack_orig = A_orig * x_orig.
5. Verify primal residuals ||A_orig x_orig - slack|| <= eps_feas.
6. Verify dual residuals ||c_orig - A_orig^T y_orig - s_orig|| <= eps_dual.
7. Return completed Solution.
```

---

## 5. Termination & Convergence Loop

Presolve runs passes iteratively until:
1. No reductions are made in a complete pass (fixpoint reached).
2. Maximum passes reached (default: 10 passes, configurable in `StrategyConfig`).
3. Problem is determined to be Infeasible or Unbounded.

If the entire problem is eliminated by presolve (e.g., all variables fixed), postsolve reconstructs the unique optimal primal/dual solution directly without invoking the simplex engine!
