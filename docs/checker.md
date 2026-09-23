# Independent Solution Checker Architecture

This document describes the algorithms, mathematical formulations, numerical criteria, and exact rational arithmetic procedures used by the independent verification checker (`/tests/checker`).

## 1. Design Principles
The checker is strictly independent of the solver core code paths. It evaluates candidate solutions $(x, y, s)$ directly against raw problem constraints and bounds to guarantee ground-truth verification without relying on solver state or internal flags.

---

## 2. Floating-Point Residual Verification

Given optimization problem $\min c^T x + \frac{1}{2} x^T Q x + c_0$ s.t. $l \le A x \le u$ and $x_l \le x \le x_u$:

### 1. Primal Residuals
- **Row Violations**:
  $$v_{row, i} = \max\Big(0,\; l_i - (Ax)_i,\; (Ax)_i - u_i\Big)$$
- **Variable Bound Violations**:
  $$v_{col, j} = \max\Big(0,\; x_{l, j} - x_j,\; x_j - x_{u, j}\Big)$$
- **Max Primal Residual**:
  $$\|r_{primal}\|_\infty = \max\Big(\max_i v_{row, i},\; \max_j v_{col, j}\Big)$$
  Pass criterion: $\|r_{primal}\|_\infty \le \epsilon_{feas}$ (Default: $10^{-6}$).

### 2. Dual Residuals (LP / QP)
- **Lagrangian Stationarity**:
  $$r_{dual} = c + Q x - A^T y - s$$
  Pass criterion: $\|r_{dual}\|_\infty \le \epsilon_{dual}$ (Default: $10^{-6}$).
- **Multiplier Sign Consistency**:
  - For $A_i x \le u_i$ only: $y_i \le 0$ (for minimization).
  - For $A_i x \ge l_i$ only: $y_i \ge 0$.
  - For equality $l_i = u_i$: $y_i \in \mathbb{R}$.

### 3. Complementary Slackness
- **Primal-Dual Slacks**:
  $$\text{Slack}_{row, i} = \min(|(Ax)_i - l_i|,\; |u_i - (Ax)_i|) \cdot |y_i| \le \epsilon_{comp}$$
  $$\text{Slack}_{col, j} = \min(|x_j - x_{l, j}|,\; |x_{u, j} - x_j|) \cdot |s_j| \le \epsilon_{comp}$$

### 4. Integrality Violations (MIP)
- For each integer or binary variable $j \in \mathcal{I}$:
  $$v_{int, j} = |x_j - \text{round}(x_j)|$$
  Pass criterion: $\max_{j \in \mathcal{I}} v_{int, j} \le \epsilon_{int}$ (Default: $10^{-5}$).

### 5. Objective Value Check
- Evaluate $obj_{recalc} = c_0 + c^T x + \frac{1}{2} x^T Q x$.
- Verify $|obj_{recalc} - obj_{reported}| \le 10^{-6} \cdot (1 + |obj_{recalc}|)$.

---

## 3. Exact Rational Arithmetic Checker (`checker_rational.py`)

Floating-point roundoff can occasionally mask boundary violations or introduce numerical ambiguity on ill-conditioned problems. For small test instances (e.g. the 10-problem toy suite), `checker_rational.py` provides exact symbolic validation:
- All coefficients, bounds, and solution components are converted to `fractions.Fraction`.
- Matrix-vector products $\sum_j A_{ij} x_j$ are accumulated in arbitrary-precision integers with zero rounding error.
- Comparisons $l_i \le (Ax)_i \le u_i$ are performed as exact relational checks.
