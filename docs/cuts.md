# Cutting Planes: Gomory Mixed-Integer Cuts (GMIC)

## 1. Mathematical Foundations & Tableau Derivation

Cutting planes strengthen the linear programming relaxation of mixed-integer programs by removing fractional solutions without eliminating any integer feasible points.

Consider the optimal simplex tableau of the LP relaxation. Let $B$ be the optimal basis matrix, and let $x_{B_i}$ be a basic variable that is required to be integer ($B_i \in \mathcal{I}$), but whose current LP value is fractional:
$$x_{B_i}^* = \bar{b}_i \notin \mathbb{Z}$$

In the simplex tableau, row $i$ expresses the basic variable in terms of non-basic variables $j \in \mathcal{N}$:
$$x_{B_i} + \sum_{j \in \mathcal{N}} \bar{a}_{ij} x_j = \bar{b}_i$$
where $\bar{a}_{ij} = (B^{-1} A_j)_i = e_i^T B^{-1} A_j$, and $\bar{b}_i = (B^{-1} b)_i$.

### 1.1 Shift to Non-Basic Bounds
For variables with non-zero lower bounds $l_j$ or at their upper bounds $u_j$, we perform the standard change of variables:
$$\hat{x}_j = \begin{cases} x_j - l_j & \text{if } x_j \text{ is at lower bound} \\ u_j - x_j & \text{if } x_j \text{ is at upper bound} \end{cases}$$
such that $\hat{x}_j \ge 0$ with $\hat{x}_j^* = 0$ for all $j \in \mathcal{N}$.

The tableau row becomes:
$$x_{B_i} + \sum_{j \in \mathcal{N}} \hat{a}_{ij} \hat{x}_j = f_0 + \lfloor \bar{b}_i \rfloor$$
where $f_0 = \bar{b}_i - \lfloor \bar{b}_i \rfloor \in (0, 1)$ is the fractional part of the RHS.

### 1.2 Gomory Mixed-Integer Inequality
For mixed-integer sets where continuous variables can also appear non-basic, the Gomory Mixed-Integer Cut (GMIC) (Gomory 1960) states:
$$\sum_{j \in \mathcal{N} \cap \mathcal{I}} \alpha_j \hat{x}_j + \sum_{j \in \mathcal{N} \setminus \mathcal{I}} \gamma_j \hat{x}_j \ge f_0$$
where for integer non-basic variables $j \in \mathcal{N} \cap \mathcal{I}$:
$$f_j = \hat{a}_{ij} - \lfloor \hat{a}_{ij} \rfloor$$
$$\alpha_j = \begin{cases} f_j & \text{if } f_j \le f_0 \\ \frac{f_0}{1 - f_0} (1 - f_j) & \text{if } f_j > f_0 \end{cases}$$
and for continuous non-basic variables $j \in \mathcal{N} \setminus \mathcal{I}$:
$$\gamma_j = \begin{cases} \hat{a}_{ij} & \text{if } \hat{a}_{ij} \ge 0 \\ -\frac{f_0}{1 - f_0} \hat{a}_{ij} & \text{if } \hat{a}_{ij} < 0 \end{cases}$$

### 1.3 Transformation Back to Original Coordinates
Substituting back $\hat{x}_j = x_j - l_j$ (or $u_j - x_j$):
$$\sum_{j \in \mathcal{N}} a_j^{cut} x_j \ge b^{cut}$$
where:
$$a_j^{cut} = \begin{cases} \alpha_j & \text{if } j \in \mathcal{N} \cap \mathcal{I} \text{ (at lower bound)} \\ -\alpha_j & \text{if } j \in \mathcal{N} \cap \mathcal{I} \text{ (at upper bound)} \\ \gamma_j & \text{if } j \in \mathcal{N} \setminus \mathcal{I} \text{ (at lower bound)} \\ -\gamma_j & \text{if } j \in \mathcal{N} \setminus \mathcal{I} \text{ (at upper bound)} \end{cases}$$
and the right-hand side constant shifts accordingly:
$$b^{cut} = f_0 + \sum_{j \in \mathcal{N} \text{ at lower}} a_j^{cut} l_j + \sum_{j \in \mathcal{N} \text{ at upper}} (-a_j^{cut}) u_j$$

---

## 2. Cut Generation Pipeline

```
  +-------------------------------------------------------+
  |              Optimal Root LP Solution                 |
  |         Basis B, Tableau rows, x* fractional         |
  +-------------------------------------------------------+
                             |
                             v
  +-------------------------------------------------------+
  |       Select Basic Integer Variables with             |
  |           epsilon < f_0 < 1 - epsilon                 |
  +-------------------------------------------------------+
                             |
                             v
  +-------------------------------------------------------+
  |             BTRAN: pi = e_i^T B^{-1}                  |
  |       Generate tableau row: a_ij = pi^T A_j           |
  +-------------------------------------------------------+
                             |
                             v
  +-------------------------------------------------------+
  |          Compute GMIC coefficients & RHS              |
  +-------------------------------------------------------+
                             |
                             v
  +-------------------------------------------------------+
  |               Numerical Filtering:                    |
  |  Efficacy >= eps, Dynamism <= 1e6, Sparsity filter    |
  +-------------------------------------------------------+
                             |
                             v
  +-------------------------------------------------------+
  |        Append Valid Cuts as New Rows to LP            |
  |       Re-optimize with Dual Simplex Warm-Start        |
  +-------------------------------------------------------+
```

### 2.1 Generating a Tableau Row via BTRAN
To construct tableau row $i$ efficiently without forming the dense tableau $B^{-1} A$:
1. Solve $B^T \pi = e_i$ using Phase 1's hypersparse BTRAN (`SparseLU::btran`).
2. Compute $\bar{a}_{ij} = \pi^T A_j$ for each non-basic column $j \in \mathcal{N}$ via sparse dot product.

---

## 3. Numerical Stability & Cut Filtering

Uncontrolled cut generation can introduce severe ill-conditioning and numerical drift into the basis matrix $B$. The cut generator applies three mandatory filters:

1. **Fractionality Filter**:
   Only generate cuts from tableau rows where the fractional part is sufficiently bounded away from integer:
   $$0.05 \le f_0 \le 0.95$$

2. **Cut Efficacy (Violation)**:
   The Euclidean violation of candidate cut $a^T x \ge b$ at current LP solution $x^*$ is:
   $$\text{efficacy} = \frac{b - a^T x^*}{\|a\|_2}$$
   Cuts with $\text{efficacy} < 10^{-4}$ are discarded as numerically ineffective.

3. **Dynamism (Coefficient Ratio)**:
   $$\text{dynamism} = \frac{\max_{j, a_j \ne 0} |a_j|}{\min_{j, a_j \ne 0} |a_j|}$$
   Cuts with $\text{dynamism} > 10^6$ are rejected to prevent basis conditioning breakdown.

4. **Sparsity**:
   Cuts with excessively high fill-in ($> 0.8 n$) are discarded to maintain hypersparse linear algebra performance.

---

## 4. Root Node Cut Loop

The cut generator operates at the root node in rounds, configured via `StrategyConfig::cut_rounds` (default: 3 rounds):

```cpp
void apply_root_cuts(model::Problem& problem,
                     model::Solution& root_solution,
                     const model::Options& options) {
    int max_rounds = options.strategy.cut_rounds;
    
    for (int round = 0; round < max_rounds; ++round) {
        auto cuts = generate_gmic_cuts(problem, root_solution, options);
        if (cuts.empty()) break;
        
        // Add cuts as new inequality rows to problem: l_cut <= A_cut x <= SIH_INFINITY
        for (const auto& cut : cuts) {
            problem.add_row(cut.row_triplets, cut.rhs, model::SIH_INFINITY);
        }
        
        // Re-optimize with dual simplex warm-start
        DualSimplexEngine engine(problem, options);
        engine.init_warm_start(root_solution.col_basis, root_solution.row_basis);
        root_solution = engine.solve();
        
        if (!root_solution.is_optimal()) break;
        if (is_integer_feasible(root_solution, problem)) break;
    }
}
```

---

## 5. Performance Metrics & Gate Verification

Per Phase 3 GATE requirements:
- **Root Gap Improvement**:
  $$\text{gap\_closed} = \frac{z_{cut} - z_{LP}}{z_{oracle} - z_{LP}}$$
  Reported across representative MIPLIB instances to demonstrate cutting plane effectiveness before branching.
- **Warm-start efficiency**: Dual simplex takes minimal iterations to absorb new cuts because only row slacks enter the basis as negative pivots.
