# Interior Point Method Engine Architecture (Phase 2)

This document details the mathematical theory, algorithmic formulations, sparse linear algebra, and software architecture for the Phase 2 Primal-Dual Interior Point Method (IPM) engine for Smart India Hackathon problem **SIH 26119**.

Per **Hard Rule 1**, all algorithms, sparse matrix factorizations (Cholesky / $LDL^T$), fill-reducing ordering heuristics (Approximate Minimum Degree - AMD), Mehrotra predictor-corrector systems, and crossover routines are developed completely from scratch in standard C++17 (`/core/src/ipm/`, `/core/include/sih/ipm/`, `/core/src/factorization/`) with zero external linear algebra or solver dependencies.

---

## 1. Unified Problem Formulation: LP & Convex QP

The interior point engine solves both general continuous linear programs (LP) and convex quadratic programs (QP) through a single unified primal-dual framework.

Consider the general convex quadratic optimization problem with bounded variables and two-sided general constraints:

$$\min_{x \in \mathbb{R}^n} \quad \frac{1}{2} x^T Q x + c^T x + c_0$$
$$\text{subject to} \quad l \le A x \le u$$
$$lb \le x \le ub$$

where:
- $Q \in \mathbb{R}^{n \times n}$ is a symmetric positive semi-definite sparse matrix ($x^T Q x \ge 0 \quad \forall x$). When $Q = 0$, the problem reduces exactly to a linear program.
- $A \in \mathbb{R}^{m \times n}$ is the constraint matrix in Compressed Sparse Column (CSC) format.
- $c \in \mathbb{R}^n$ is the linear objective vector, and $c_0 \in \mathbb{R}$ is the scalar objective offset.
- $l, u \in (\mathbb{R} \cup \{\pm \infty\})^m$ are the lower and upper row bounds.
- $lb, ub \in (\mathbb{R} \cup \{\pm \infty\})^n$ are the lower and upper variable bounds.

---

## 2. Standard Slack Equality Form & Barrier Penalty

To handle inequalities and two-sided bounds uniformly, we introduce slack variables $s \in \mathbb{R}^m$:

$$\min_{x, s} \quad \frac{1}{2} x^T Q x + c^T x$$
$$\text{subject to} \quad A x - s = 0$$
$$lb \le x \le ub$$
$$l \le s \le u$$

For variables with finite lower and/or upper bounds, we define bound gap vectors:
- For variable bounds:
  - $x - lb = x_l \ge 0$ (with dual multiplier / reduced cost $z_l \ge 0$)
  - $ub - x = x_u \ge 0$ (with dual multiplier / reduced cost $z_u \ge 0$)
- For constraint slacks:
  - $s - l = s_l \ge 0$ (with dual multiplier $w_l \ge 0$)
  - $u - s = s_u \ge 0$ (with dual multiplier $w_u \ge 0$)

The equality-constrained primal-dual barrier formulation with barrier parameter $\mu > 0$ introduces logarithmic penalties:

$$\min_{x, s} \quad \frac{1}{2} x^T Q x + c^T x - \mu \sum_{j \in \mathcal{L}_x} \ln(x_j - lb_j) - \mu \sum_{j \in \mathcal{U}_x} \ln(ub_j - x_j) - \mu \sum_{i \in \mathcal{L}_s} \ln(s_i - l_i) - \mu \sum_{i \in \mathcal{U}_s} \ln(u_i - s_i)$$
$$\text{subject to} \quad A x - s = 0$$

### 2.1 First-Order Optimality Conditions (Perturbed KKT System)

The Karush-Kuhn-Tucker (KKT) conditions for the perturbed barrier problem are:

1. **Primal Feasibility**:
   $$r_p = A x - s = 0$$
   $$x_l = x - lb \ge 0, \quad x_u = ub - x \ge 0$$
   $$s_l = s - l \ge 0, \quad s_u = u - s \ge 0$$

2. **Dual Feasibility / Stationarity**:
   $$r_d = c + Q x - A^T y - z_l + z_u = 0$$
   $$r_s = -y - w_l + w_u = 0 \iff y = w_u - w_l$$
   $$z_l, z_u \ge 0, \quad w_l, w_u \ge 0$$

3. **Complementary Slackness**:
   $$X_l Z_l e = \mu e \iff x_{l, j} z_{l, j} = \mu \quad \forall j \in \mathcal{L}_x$$
   $$X_u Z_u e = \mu e \iff x_{u, j} z_{u, j} = \mu \quad \forall j \in \mathcal{U}_x$$
   $$S_l W_l e = \mu e \iff s_{l, i} w_{l, i} = \mu \quad \forall i \in \mathcal{L}_s$$
   $$S_u W_u e = \mu e \iff s_{u, i} w_{u, i} = \mu \quad \forall i \in \mathcal{U}_s$$

where capital letters ($X_l, Z_l, \ldots$) denote diagonal matrices formed from the corresponding vectors, and $e = [1, \ldots, 1]^T$.

The average complementary barrier parameter $\mu$ is defined as:

$$\mu = \frac{x_l^T z_l + x_u^T z_u + s_l^T w_l + s_u^T w_u}{|\mathcal{L}_x| + |\mathcal{U}_x| + |\mathcal{L}_s| + |\mathcal{U}_s|}$$

As $\mu \to 0$, the sequence of interior solutions $(x(\mu), s(\mu), y(\mu), z(\mu), w(\mu))$ traces the **central path** and converges to the exact primal and dual optimal set $(x^*, s^*, y^*, z^*, w^*)$.

---

## 3. Mehrotra Predictor-Corrector Algorithm

The solver utilizes Mehrotra's celebrated predictor-corrector technique, which computes higher-order curvature corrections and dynamic centering without requiring additional matrix factorizations.

In each iteration, the non-linear KKT system is solved via Newton's method in two phases:
1. **Predictor (Affine Scaling) Phase**: Solves the linearized system with $\mu = 0$ to probe along the current affine direction and estimate how aggressively the barrier parameter $\mu$ can be reduced.
2. **Corrector & Centering Phase**: Uses the affine step to evaluate the nonlinear cross-terms ($\Delta x \cdot \Delta z$) and computes the centering parameter $\sigma \in [0, 1]$.
3. **Combined Step**: Solves the linear system with the combined RHS (affine + centering + cross-term) using the **same matrix factorization**.

### 3.1 Linearized Newton Step Equations

Applying Newton linearization to the KKT system around current point $(x, s, y, z_l, z_u, w_l, w_u)$:

$$\begin{bmatrix}
Q & 0 & -A^T & -I & I & 0 & 0 \\
0 & 0 & I & 0 & 0 & -I & I \\
A & -I & 0 & 0 & 0 & 0 & 0 \\
Z_l & 0 & 0 & X_l & 0 & 0 & 0 \\
-Z_u & 0 & 0 & 0 & X_u & 0 & 0 \\
0 & W_l & 0 & 0 & 0 & S_l & 0 \\
0 & -W_u & 0 & 0 & 0 & 0 & S_u
\end{bmatrix}
\begin{bmatrix}
\Delta x \\ \Delta s \\ \Delta y \\ \Delta z_l \\ \Delta z_u \\ \Delta w_l \\ \Delta w_u
\end{bmatrix}
=
\begin{bmatrix}
-r_d \\
-r_s \\
-r_p \\
-X_l Z_l e + \gamma_l \\
-X_u Z_u e + \gamma_u \\
-S_l W_l e + \eta_l \\
-S_u W_u e + \eta_u
\end{bmatrix}$$

where:
- In the **Predictor Phase**: $\gamma_l = \gamma_u = \eta_l = \eta_u = 0$.
- In the **Corrector Phase**:
  $$\gamma_l = \sigma \mu e - \Delta X_l^{\text{aff}} \Delta Z_l^{\text{aff}} e$$
  $$\gamma_u = \sigma \mu e - \Delta X_u^{\text{aff}} \Delta Z_u^{\text{aff}} e$$
  $$\eta_l = \sigma \mu e - \Delta S_l^{\text{aff}} \Delta W_l^{\text{aff}} e$$
  $$\eta_u = \sigma \mu e - \Delta S_u^{\text{aff}} \Delta W_u^{\text{aff}} e$$

### 3.2 Elimination to the Reduced Augmented and Normal Systems

From the complementary slackness rows, we eliminate $\Delta z_l, \Delta z_u, \Delta w_l, \Delta w_u$:

$$\Delta z_l = -z_l + X_l^{-1} (\gamma_l - Z_l \Delta x)$$
$$\Delta z_u = -z_u + X_u^{-1} (\gamma_u + Z_u \Delta x)$$
$$\Delta w_l = -w_l + S_l^{-1} (\eta_l - W_l \Delta s)$$
$$\Delta w_u = -w_u + S_u^{-1} (\eta_u + W_u \Delta s)$$

Substituting these into the dual stationarity equations gives diagonal diagonal scaling matrices:

$$\Theta_x = X_l^{-1} Z_l + X_u^{-1} Z_u \in \mathbb{R}^{n \times n} \quad (\text{strictly positive diagonal})$$
$$\Theta_s = S_l^{-1} W_l + S_u^{-1} W_u \in \mathbb{R}^{m \times m} \quad (\text{strictly positive diagonal})$$

Furthermore, since $\Delta s = A \Delta x + r_p$, eliminating $\Delta s$ yields the **Normal Equations**:

$$(A (Q + \Theta_x)^{-1} A^T + \Theta_s^{-1}) \Delta y = r_y$$

For LP ($Q = 0$), $Q + \Theta_x = \Theta_x$ is a purely diagonal matrix:

$$D_x = \Theta_x^{-1} = \left( X_l^{-1} Z_l + X_u^{-1} Z_u \right)^{-1}$$

Hence the Normal Equations matrix $M \in \mathbb{R}^{m \times m}$ is:

$$M = A D_x A^T + \Theta_s^{-1}$$

$M$ is symmetric positive definite (SPD) with the exact sparsity pattern of $A A^T$.
For convex QP with diagonal $Q$, $D_x = (Q + \Theta_x)^{-1}$ remains strictly diagonal and positive, preserving the exact same $M = A D_x A^T + \Theta_s^{-1}$ SPD structure.

For general sparse $Q$, we solve the **Symmetric Augmented System**:

$$\begin{bmatrix}
-(Q + \Theta_x) & A^T \\
A & \Theta_s^{-1}
\end{bmatrix}
\begin{bmatrix}
\Delta x \\ \Delta y
\end{bmatrix}
=
\begin{bmatrix}
\bar{r}_d \\
\bar{r}_p
\end{bmatrix}$$

using symmetric quasi-definite sparse factorization or diagonal decomposition.

---

## 4. Sparse Cholesky Factorization & AMD Ordering

Solving the Normal Equations $(A D A^T + \Theta_s^{-1}) \Delta y = r_y$ at each iteration is the primary computational cost of the interior point method.

### 4.1 Approximate Minimum Degree (AMD) Ordering

Before numerical factorization, an ordering permutation matrix $P \in \{0, 1\}^{m \times m}$ is computed to minimize fill-in during Cholesky factorization:

$$P M P^T = L L^T$$

We implement an indigenous Approximate Minimum Degree (AMD) ordering algorithm on the sparsity graph $G = (V, E)$ of $M$:
1. **Graph Construction**: Form the adjacency graph of $M = A D A^T + \Theta_s^{-1}$ directly from the row overlap structure of $A$:
   $$(i, k) \in E \iff \exists j \text{ s.t. } A_{ij} \ne 0 \text{ and } A_{kj} \ne 0$$
2. **Degree Approximations & Quotient Graph Elimination**:
   - Initialize node degrees $d_i = |Adj(i)|$.
   - Maintain supernodes / quotient graph to represent absorbed cliques formed by eliminated pivots.
   - At each step, select pivot node $p$ minimizing approximate degree:
     $$p = \arg\min_{i \in V_{\text{active}}} d_i$$
   - Update degrees of adjacent nodes using upper bounds without explicit element construction.
   - Record pivot order $P = [p_1, p_2, \ldots, p_m]$.

### 4.2 Symbolic Factorization & Elimination Tree (etree)

1. Compute the elimination tree of $P M P^T$:
   $$\text{parent}(j) = \min \{ i > j : L_{ij} \ne 0 \}$$
2. Compute nonzero counts per column of $L$ directly from the elimination tree.
3. Allocate exact compressed column pointers `col_ptr` and row index storage `row_ind` for $L$, avoiding dynamic memory reallocations during numeric factorization.

### 4.3 Numeric Cholesky Factorization ($L L^T$)

Given the permuted matrix $\bar{M} = P M P^T$, compute lower triangular $L$ such that $\bar{M} = L L^T$:

For $j = 1, \ldots, m$:
1. Compute diagonal element:
   $$L_{jj} = \sqrt{\bar{M}_{jj} - \sum_{k < j} L_{jk}^2}$$
   - **Numerical Safeguard (Dynamic Regularization)**: If $L_{jj}^2 \le \epsilon_{\text{piv}}$, add diagonal regularization $\delta = 10^{-12} \cdot (1 + |\bar{M}_{jj}|)$ to maintain strict positive definiteness even when constraints are degenerate or linearly dependent.
2. For each sub-diagonal entry $i > j$ with $\bar{M}_{ij} \ne 0$ or fill-in:
   $$L_{ij} = \frac{1}{L_{jj}} \left( \bar{M}_{ij} - \sum_{k < j} L_{ik} L_{jk} \right)$$

### 4.4 Forward and Back Substitution

Given $L L^T (P \Delta y) = P r_y$:
1. Permute RHS: $b' = P r_y$.
2. Forward solve: $L v = b'$.
3. Back solve: $L^T w = v$.
4. Un-permute: $\Delta y = P^T w$.

---

## 5. Step Length Selection & Iterate Update

To preserve strict interior feasibility ($x_l > 0, x_u > 0, s_l > 0, s_u > 0, z_l > 0, z_u > 0, w_l > 0, w_u > 0$):

### 5.1 Maximum Step Length to Boundary

Compute maximum step sizes $\alpha_p^{\max} \in (0, 1]$ and $\alpha_d^{\max} \in (0, 1]$ before hitting zero:

$$\alpha_p^{\max} = \min \left\{ 1, \min_{j: \Delta x_{l, j} < 0} \frac{-x_{l, j}}{\Delta x_{l, j}}, \min_{j: \Delta x_{u, j} < 0} \frac{-x_{u, j}}{\Delta x_{u, j}}, \min_{i: \Delta s_{l, i} < 0} \frac{-s_{l, i}}{\Delta s_{l, i}}, \min_{i: \Delta s_{u, i} < 0} \frac{-s_{u, i}}{\Delta s_{u, i}} \right\}$$

$$\alpha_d^{\max} = \min \left\{ 1, \min_{j: \Delta z_{l, j} < 0} \frac{-z_{l, j}}{\Delta z_{l, j}}, \min_{j: \Delta z_{u, j} < 0} \frac{-z_{u, j}}{\Delta z_{u, j}}, \min_{i: \Delta w_{l, i} < 0} \frac{-w_{l, i}}{\Delta w_{l, i}}, \min_{i: \Delta w_{u, i} < 0} \frac{-w_{u, i}}{\Delta w_{u, i}} \right\}$$

### 5.2 Step Safety Fraction-to-the-Boundary Rule

The actual steps taken scale by the safety parameter $\tau \in [0.99, 0.9995]$:

$$\alpha_p = \tau \cdot \alpha_p^{\max}$$
$$\alpha_d = \tau \cdot \alpha_d^{\max}$$

### 5.3 Mehrotra Centering Exponent

In the predictor step, affine step sizes $\alpha_p^{\text{aff}}, \alpha_d^{\text{aff}}$ are determined. The resulting affine duality measure is:

$$\mu_{\text{aff}} = \frac{(x_l + \alpha_p^{\text{aff}} \Delta x_l^{\text{aff}})^T (z_l + \alpha_d^{\text{aff}} \Delta z_l^{\text{aff}}) + \cdots}{\text{dim}}$$

The centering parameter $\sigma$ is computed dynamically via Mehrotra's cubic rule:

$$\sigma = \left( \frac{\mu_{\text{aff}}}{\mu} \right)^3$$

---

## 6. Stopping & Convergence Criteria

The interior point method evaluates three dimensionless, scale-invariant residual norms at each iteration:

1. **Primal Residual**:
   $$\epsilon_p = \frac{\|A x - s\|_\infty}{1 + \|b\|_\infty} \le \text{tol}_{\text{primal}}$$

2. **Dual Residual**:
   $$\epsilon_d = \frac{\|Q x + c - A^T y - z_l + z_u\|_\infty}{1 + \|c\|_\infty} \le \text{tol}_{\text{dual}}$$

3. **Relative Duality Gap**:
   $$\epsilon_g = \frac{|c^T x + \frac{1}{2} x^T Q x - (b^T y - \frac{1}{2} x^T Q x + lb^T z_l - ub^T z_u)|}{1 + |c^T x + \frac{1}{2} x^T Q x|} \le \text{tol}_{\text{gap}}$$

When $\max(\epsilon_p, \epsilon_d, \epsilon_g) \le \text{tolerance}$ (default $10^{-8}$), the iterate is declared **Optimal**.

---

## 7. Crossover: From Interior Point to Exact Vertex Basis

Interior point methods return solutions located in the relative interior of the optimal face ($x_j^* > 0$ and $s_j^* > 0$). Downstream algorithms, such as Branch-and-Bound for MILP (Phase 3) and refinery re-optimization warm-starts, require an **exact basic feasible solution (BFS)** with a non-singular basis matrix $B$ and zero reduced costs for non-basic variables.

### 7.1 Active Set Identification

From the interior solution $(x, z)$, we partition the variables:
- If $(x_j - lb_j) / z_{l, j} < \text{threshold}$, variable $x_j$ is strongly inclined to lower bound $\to$ candidate non-basic at lower.
- If $(ub_j - x_j) / z_{u, j} < \text{threshold}$, variable $x_j$ is strongly inclined to upper bound $\to$ candidate non-basic at upper.
- Otherwise, $x_j$ is a candidate basic variable.

### 7.2 Basis Selection & Factorization

1. Form candidate basic set $\mathcal{B} \subset \{1, \ldots, n + m\}$.
2. Factorize candidate basis $B$ using the Phase 1 sparse LU factorization.
3. If $B$ is structurally or numerically singular, replace dependent columns with identity slack columns from $A x - I s = 0$.

### 7.3 Simplex Cleanup Pass

Once a valid initial basis $(B, N)$ is formed:
1. Initialize Phase 1 Primal/Dual Simplex engine using `solve_from_basis`.
2. Execute a small number of cleanup pivots (typically $\le 5$ iterations) to achieve exact complementary slackness and vertex optimality.
3. Return the exact vertex `Solution` containing full basis statuses (`col_basis`, `row_basis`), row duals, and reduced costs.

---

## 8. StrategyConfig Tunable Options

All hyperparameters are integrated into `StrategyConfig`:

| Option Field | Type | Default | Description |
|---|---|---|---|
| `ipm_max_iterations` | `int64_t` | `100` | Maximum interior point iterations |
| `ipm_primal_tol` | `double` | `1e-8` | Relative primal feasibility tolerance |
| `ipm_dual_tol` | `double` | `1e-8` | Relative dual feasibility tolerance |
| `ipm_gap_tol` | `double` | `1e-8` | Relative duality gap tolerance |
| `ipm_step_safety` | `double` | `0.995` | Fraction-to-boundary safety coefficient ($\tau$) |
| `ipm_centering_exponent` | `double` | `3.0` | Mehrotra centering power ($\sigma = (\mu_{\text{aff}} / \mu)^p$) |
| `ipm_enable_crossover` | `bool` | `true` | Handoff to simplex cleanup for exact basic solution |
| `ipm_regularization` | `double` | `1e-12` | Levenberg-Marquardt diagonal regularization |
| `ipm_ordering` | `enum` | `AMD` | Ordering algorithm (`AMD`, `Natural`) |

---

## 9. Verification & Gate Plan

1. **Maros-Mészáros QP Benchmark Subset**:
   - `CVXQP1_S`, `CVXQP2_S`, `CVXQP3_S`, `DUAL1`, `DUAL2`, `HS21`, `HS35`, `HS76`.
   - Verified against HiGHS oracle within $10^{-6}$ relative objective difference.
2. **Netlib LP Subset**:
   - `adlittle`, `afiro`, `beaconfd`, `blend`, `lotfi`, `sc50a`, `sc50b`, `share2b`, `stocfor1`.
   - Checked against HiGHS oracle and compared with Phase 1 dual simplex.
3. **Independent Solution Checker**:
   - Checked for primal feasibility, dual feasibility, and complementarity on every returned solution.
4. **Crossover Validation**:
   - Checked that post-crossover solution has a non-singular basis valid for `solve_from_basis` warm starts.
