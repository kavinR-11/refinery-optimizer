# LP Core Solver Engine Architecture (Phase 1)

This document details the mathematical architecture, sparse algorithms, linear algebra implementations, and API designs for the Phase 1 LP solver core for Smart India Hackathon problem **SIH 26119**.

Per **Hard Rule 1**, all algorithms and data structures are developed from scratch in standard C++17 (`/core/src/simplex/`, `/core/src/factorization/`, `/core/src/scaling/`) with zero third-party linear algebra or solver libraries.

---

## 1. Problem Formulation & Simplex Standard Form

We consider the bounded general linear programming problem:

$$\min_{x} \quad c^T x + c_0$$
$$\text{subject to} \quad l \le A x \le u$$
$$lb \le x \le ub$$

where $A \in \mathbb{R}^{m \times n}$ has full row rank after presolve.

### 1.1 Transformation to Equality Form with Slacks

By introducing slack variables $s \in \mathbb{R}^m$, the system becomes:

$$\min_{x, s} \quad c^T x + 0^T s + c_0$$
$$\text{subject to} \quad A x - I s = 0$$
$$lb \le x \le ub, \quad l \le s \le u$$

Let $\bar{A} = [A \quad -I] \in \mathbb{R}^{m \times (n+m)}$, $\bar{x} = \begin{bmatrix} x \\ s \end{bmatrix}$, $\bar{c} = \begin{bmatrix} c \\ 0 \end{bmatrix}$, $\bar{lb} = \begin{bmatrix} lb \\ l \end{bmatrix}$, $\bar{ub} = \begin{bmatrix} ub \\ u \end{bmatrix}$.

The system is partitioned into:
- **Basic Variables** $\mathcal{B} \subset \{1, \ldots, n+m\}$ with $|\mathcal{B}| = m$ and non-singular basis matrix $B = \bar{A}_{\cdot \mathcal{B}} \in \mathbb{R}^{m \times m}$.
- **Non-Basic Variables** $\mathcal{N} = \mathcal{L} \cup \mathcal{U} \cup \mathcal{F}$, where:
  - $\mathcal{L}$: variables at lower bound ($\bar{x}_j = \bar{lb}_j$).
  - $\mathcal{U}$: variables at upper bound ($\bar{x}_j = \bar{ub}_j$).
  - $\mathcal{F}$: free variables ($\bar{lb}_j = -\infty, \bar{ub}_j = +\infty, \bar{x}_j = 0$).

### 1.2 Fundamental Simplex Equations
- **Primal Basic Solution**:
  $$\bar{x}_B = B^{-1} \left( -\sum_{j \in \mathcal{N}} \bar{A}_{\cdot j} \bar{x}_j \right)$$
- **Dual Vector (Lagrange Multipliers)**:
  $$y^T = \bar{c}_B^T B^{-1} \iff B^T y = \bar{c}_B \quad (\text{BTRAN})$$
- **Reduced Costs**:
  $$\bar{s}_N = \bar{c}_N - \bar{A}_{\cdot N}^T y$$
- **Dual Feasibility Condition**:
  $$\begin{cases}
  \bar{s}_j \ge -\epsilon_{\text{dual}} & \forall j \in \mathcal{L} \\
  \bar{s}_j \le \epsilon_{\text{dual}} & \forall j \in \mathcal{U} \\
  |\bar{s}_j| \le \epsilon_{\text{dual}} & \forall j \in \mathcal{F}
  \end{cases}$$

---

## 2. Matrix Scaling with Power-of-Two Quantization

To prevent numerical instability in ill-conditioned matrices without degrading precision, we implement geometric mean scaling followed by equilibration, with exact power-of-two quantization.

### 2.1 The Power-of-Two Theorem
Let $x \in \mathbb{R}$ be an IEEE-754 floating-point number with binary representation $x = (-1)^s \cdot 1.m \cdot 2^{e - 1023}$.
Multiplying $x$ by $2^k$ modifies only the exponent:

$$x \cdot 2^k = (-1)^s \cdot 1.m \cdot 2^{(e + k) - 1023}$$

**Mantissa bits $m$ are completely unaltered, generating exactly zero rounding noise.**

### 2.2 Scaling Procedure
1. **Geometric Mean Pass**:
   - For each row $i$:
     $$r_i = \left( \min_{j: A_{ij} \ne 0} |A_{ij}| \cdot \max_{j: A_{ij} \ne 0} |A_{ij}| \right)^{-1/2}$$
   - Quantize: $R_{ii} = 2^{\text{round}(\log_2(r_i))}$.
   - For each column $j$:
     $$c_j = \left( \min_{i: A_{ij} \ne 0} |R_{ii} A_{ij}| \cdot \max_{i: A_{ij} \ne 0} |R_{ii} A_{ij}| \right)^{-1/2}$$
   - Quantize: $C_{jj} = 2^{\text{round}(\log_2(c_j))}$.
2. **Equilibration Pass**:
   - Alternately scale rows $R_{ii} \leftarrow R_{ii} \cdot 2^{\text{round}(-\log_2(\max_j |A_{ij}|))}$ and columns $C_{jj} \leftarrow C_{jj} \cdot 2^{\text{round}(-\log_2(\max_i |A_{ij}|))}$.
3. **Scaled Problem**:
   $$\tilde{A} = R A C, \quad \tilde{c} = C c, \quad \tilde{l} = R l, \quad \tilde{u} = R u, \quad \tilde{lb} = C^{-1} lb, \quad \tilde{ub} = C^{-1} ub$$
4. **Exact Solution Unscaling**:
   $$x = C \tilde{x}, \quad Ax = R^{-1} (\tilde{A} \tilde{x}), \quad y = R \tilde{y}, \quad s = C^{-1} \tilde{s}$$

---

## 3. Sparse LU Factorization Engine

The basis matrix $B$ is factorized as:

$$P B Q = L U$$

where $P, Q$ are row and column permutation matrices, $L$ is unit lower triangular, and $U$ is upper triangular.

### 3.1 Singleton Elimination Phase
Before general factorization, row and column singletons are identified:
- **Row singleton**: A row with one nonzero entry $A_{ij}$. Since $B$ is square and non-singular, entry $A_{ij}$ must be a pivot in $U$. It is assigned to the current diagonal in $O(1)$ with zero fill-in.
- **Column singleton**: A column with one nonzero entry. Symmetrically assigned to $L$ with zero fill-in.

### 3.2 Markowitz Pivoting with Threshold Partial Pivoting
For the active submatrix at step $k$, candidate pivot $(i, j)$ is chosen to minimize Markowitz merit:

$$M_{ij} = (r_i - 1)(c_j - 1)$$

subject to the threshold partial pivoting numerical stability criterion:

$$|a_{ij}| \ge u \cdot \max_{k} |a_{kj}|, \quad u \in (0, 1] \quad (\text{default: } u = 0.1)$$

Candidates are searched in ascending order of row and column nonzero counts.

### 3.3 Hypersparse FTRAN and BTRAN
- **FTRAN (Forward Transformation)**: Solves $B d = a$.
  $$L (U (Q^{-1} d)) = P a$$
- **BTRAN (Backward Transformation)**: Solves $B^T \pi = c_B$.
  $$U^T (L^T (P \pi)) = Q c_B$$
- **Hypersparsity via Directed Acyclic Graph (DAG) Reachability**:
  When RHS $a$ or $c_B$ has few nonzeros ($k \ll m$), standard dense solve loops take $O(m)$ time.
  Instead, we perform a Depth-First Search (DFS) on the adjacency graph of $L$ / $U$ starting from the nonzero indices of the RHS. This identifies the exact topological order of the active variables in $O(nnz(\text{solution}))$, achieving hypersparse execution speed.

### 3.4 Basis Updates & Refactorization
When column $p$ of $B$ is replaced by entering column $a_q$:
- **Product-Form of the Inverse (PFI)**:
  $$B_{k+1}^{-1} = E_k B_k^{-1}, \quad E_k = I + (\eta_k - e_p) e_p^T, \quad \eta_k = B_k^{-1} a_q$$
- **Forrest-Tomlin (FT) Update**:
  Removes column $p$ from $U$, resulting in an upper Hessenberg matrix. Row rotations restore upper triangular form while controlling fill-in.
- **Refactorization Triggers**:
  The basis is fully refactorized from scratch when:
  1. Update count exceeds limit ($N_{\text{updates}} \ge 60$).
  2. Nonzero fill-in ratio exceeds threshold: $\frac{nnz(L) + nnz(U)}{nnz(B)} > 3.0$.
  3. Residual drift detected: $\|B x_B - \text{rhs}\|_\infty > 10^{-7}$.

---

## 4. Dual Simplex Algorithm with Bounded Variables

The Dual Simplex method is ideal for linear programming with bounded variables and is the backbone of branch-and-bound and warm-starting.

### 4.1 Algorithm Overview
```text
Dual Simplex Loop:
1. Compute basic primal values x_B = B^{-1} (b - N x_N).
2. Check Primal Feasibility:
   Find leaving variable p in B violating its bounds:
   violation = max(0, lb_p - x_p, x_p - ub_p).
   If max_violation <= eps_feas:
     OPTIMAL FOUND! Exit.
3. Dual Steepest-Edge Pricing (DSE):
   Select leaving row p = argmax_{i} (violation_i^2 / gamma_i).
4. BTRAN: Solve B^T v = e_p to get pivot row in tableau.
   Compute row tableau coefficients: alpha_{pj} = v^T A_{\cdot j}.
5. Harris Ratio Test with Bound Flipping (BFRT):
   Find entering column q minimizing dual step theta.
   If no candidate exists (all alpha_{pj} have wrong sign):
     DUAL UNBOUNDED => PRIMAL INFEASIBLE. Return Farkas Ray.
6. Pivot Update:
   Update basis: column q enters B, row p leaves to its bound.
   Update DSE weights gamma.
   Update LU factorization (PFI / FT).
```

### 4.2 Dual Steepest-Edge (DSE) Pricing
Standard Dantzig pricing chooses the row with the largest absolute violation $|x_p - \text{bound}_p|$, which is prone to degenerate cycling and poor convergence.
Dual Steepest-Edge chooses:

$$p = \arg\max_{i \in \mathcal{B}} \frac{(x_i - \text{bound}_i)^2}{\gamma_i}, \quad \gamma_i = \|(B^{-1})_{i \cdot}\|_2^2 = e_i^T B^{-1} B^{-T} e_i$$

**Exact Recursive Weight Update**:
When column $q$ replaces row $p$, weights $\gamma_i$ are updated in $O(m)$ using an auxiliary solve $w = B^{-T} \alpha_q$:

$$\gamma_i^{\text{new}} = \gamma_i - 2 \left( \frac{\alpha_{iq}}{\alpha_{pq}} \right) w_i + \left( \frac{\alpha_{iq}}{\alpha_{pq}} \right)^2 \gamma_p$$
$$\gamma_p^{\text{new}} = \frac{\gamma_p}{\alpha_{pq}^2}$$

*Devex Option*: Available in `StrategyConfig` as a lightweight approximation with periodic re-initialization.

### 4.3 Harris Two-Pass Ratio Test with Bound Flipping (BFRT)
- **Pass 1 (Step Length Determination)**:
  Compute maximum dual step $\theta_{\max}$ with relaxed tolerance $\delta = 10^{-6}$:
  $$\theta_{\max} = \min_{j \in \mathcal{C}} \frac{|s_j| + \delta}{|\alpha_{pj}|}$$
- **Pass 2 (Numerical Stability Selection)**:
  Among all candidates with step $\theta_j \le \theta_{\max}$, choose entering variable $q$ that maximizes pivot element magnitude:
  $$q = \arg\max_{j: \theta_j \le \theta_{\max}} |\alpha_{pj}|$$
- **Bound Flipping Expansion**:
  If a non-basic bounded variable $j$ would become dual infeasible during step $\theta$, rather than stopping immediately, $x_j$ is flipped from its current bound to its opposite bound ($lb_j \leftrightarrow ub_j$).
  Primal basic variables are adjusted:
  $$x_B \leftarrow x_B - \alpha_{\cdot j} (ub_j - lb_j)$$
  The step continues across multiple bound flips in a single iteration, dramatically reducing total simplex iterations.

### 4.4 Anti-Cycling & Perturbation
- **Deterministic Perturbation**:
  To break degeneracies, bounds or costs receive a tiny deterministic perturbation:
  $$c_j^{\text{pert}} = c_j + \epsilon \cdot (1 + \sin(j)) \cdot 10^{-11}$$
- **Bland's Smallest-Index Rule Fallback**:
  If more than 50 consecutive degenerate iterations (step length $\theta = 0$) are detected, the solver switches strictly to Bland's rule (choosing lowest index among ties) until progress resumes.

### 4.5 Primal Simplex Phase 2
If a warm-start basis is already primal-feasible but dual-infeasible (e.g., after an objective coefficient update), the primal simplex engine is executed directly:
- Pricing: Primal Steepest-Edge / Devex.
- Ratio Test: Harris primal ratio test with bounded variables.

---

## 5. Warm Start Architecture & Re-optimization API

In industrial refinery scheduling, successive LP models are solved with slight parameter adjustments. The warm-start engine avoids solving from scratch.

### 5.1 Re-optimization Scenarios
1. **Objective Coefficient Changes ($c \to c + \Delta c$)**:
   - Basis $\mathcal{B}$ and primal variables $x_B$ remain unchanged and **primal feasible**.
   - Dual multipliers update: $B^T y = (c_B + \Delta c_B)$.
   - Reduced costs update: $s_N = (c_N + \Delta c_N) - A_{\cdot N}^T y$.
   - If dual feasibility is preserved, current basis is still optimal (0 iterations!).
   - If dual infeasible, Primal Simplex Phase 2 restores optimality in very few pivots.
2. **RHS and Bound Changes ($b \to b + \Delta b, lb \to lb + \Delta lb, ub \to ub + \Delta ub$)**:
   - Dual vector $y$ and reduced costs $s_N$ depend only on $c$ and $B$, so **dual feasibility is completely preserved**.
   - Primal values $x_B$ are recomputed: $x_B = B^{-1} (b_{\text{new}} - N x_N)$.
   - If bounds violated, Dual Simplex restores primal feasibility in a few pivots.
3. **Constraint Additions (New Rows / Cuts)**:
   - New row $a_{m+1}^T x \le b_{m+1}$ added with slack variable $s_{m+1}$.
   - Existing basis $B$ extended with $s_{m+1}$ as a new basic variable:
     $$\tilde{B} = \begin{bmatrix} B & 0 \\ a_{m+1, B}^T & 1 \end{bmatrix}$$
   - Inversion is trivial via block substitution; Dual Simplex resolves immediately.

### 5.2 C++ and Python Warm-Start API
```cpp
// C++ API in sih::simplex::DualSimplex
Solution solve_from_basis(const Problem& problem,
                          const std::vector<BasisStatus>& col_basis,
                          const std::vector<BasisStatus>& row_basis,
                          const Options& options);
```
```python
# Python API in sih_solver.simplex
solution = solver.solve(problem, warm_start_basis=prev_solution.basis)
```

---

## 6. Certificates of Infeasibility and Unboundedness

When an optimal solution does not exist, the solver outputs a rigorous mathematical certificate.

### 6.1 Farkas Ray for Infeasible Problems
By the Farkas Lemma for bounded LP, if the problem is primal infeasible, there exists a certificate vector $y \in \mathbb{R}^m$ such that:

$$\sum_{i: y_i > 0} y_i l_i + \sum_{i: y_i < 0} y_i u_i > \sum_{j: (A^T y)_j > 0} (A^T y)_j lb_j + \sum_{j: (A^T y)_j < 0} (A^T y)_j ub_j$$

The vector $y$ is extracted directly from the row of $B^{-T}$ corresponding to the leaving row $p$ when no valid pivot exists.

### 6.2 Unbounded Ray for Unbounded Problems
If the problem is primal unbounded (dual infeasible), there exists an extreme direction $d \in \mathbb{R}^n$ such that:
$$A d = 0, \quad c^T d < 0$$
$$\begin{cases}
d_j \ge 0 & \text{if } ub_j = +\infty \\
d_j \le 0 & \text{if } lb_j = -\infty \\
d_j = 0 & \text{if } lb_j, ub_j \text{ finite}
\end{cases}$$
The direction $d$ is constructed from the FTRAN column $d_B = -B^{-1} A_{\cdot q}$ and $d_q = 1$.

### 6.3 Certificate Verification in Independent Checker
`sih::checker::Checker` and Python `FloatChecker` / `RationalChecker` evaluate:
- **Infeasibility**: Evaluates the Farkas inequality and verifies violation $> 10^{-6}$.
- **Unboundedness**: Verifies $\|A d\|_\infty \le \epsilon_{\text{feas}}$, $c^T d < -\epsilon_{\text{dual}}$, and non-negativity/non-positivity on unbounded coordinates.

---

## 7. Sensitivity Analysis & Ranges

The solver provides exact allowable perturbation ranges that maintain current basis optimality.

### 7.1 RHS Ranging
For each constraint $i$, determine $[\Delta l_i^{\min}, \Delta u_i^{\max}]$ such that $lb_B \le x_B(\Delta) \le ub_B$:
$$x_B(\Delta) = x_B + B^{-1} e_i \Delta b_i$$
$$\Delta b_i^{\max} = \min_{k \in \mathcal{B}: (B^{-1})_{ki} > 0} \frac{ub_k - x_k}{(B^{-1})_{ki}}$$
$$\Delta b_i^{\min} = \max_{k \in \mathcal{B}: (B^{-1})_{ki} < 0} \frac{ub_k - x_k}{(B^{-1})_{ki}}$$

### 7.2 Objective Coefficient Ranging
For non-basic variable $j \in \mathcal{N}$:
- If $j \in \mathcal{L}$, allowable decrease before entering basis: $\Delta c_j^{\min} = -s_j$.
- If $j \in \mathcal{U}$, allowable increase before entering basis: $\Delta c_j^{\max} = -s_j$.
For basic variable $k \in \mathcal{B}$, $s_N(\Delta c_k) = c_N - A_{\cdot N}^T (y + B^{-T} e_k \Delta c_k)$. The limits are determined by the first non-basic reduced cost that changes sign.

---

## 8. StrategyConfig Integration

All algorithmic parameters are exposed in `sih::model::StrategyConfig`:
```cpp
struct StrategyConfig {
    AlgorithmChoice algorithm{AlgorithmChoice::DualSimplex};
    PricingRule pricing_rule{PricingRule::SteepestEdge}; // Dantzig, SteepestEdge, Devex
    RatioTest ratio_test{RatioTest::HarrisBFRT};          // Textbook, Harris, HarrisBFRT
    bool enable_perturbation{true};
    double perturbation_magnitude{1e-11};
    int refactor_frequency{60};
    double refactor_fill_threshold{3.0};
    bool enable_scaling{true};
    bool power_of_two_scaling{true};
    PresolveMode presolve{PresolveMode::On};
    int max_presolve_passes{10};
};
```
