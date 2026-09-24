# Adaptive Strategy Learning Layer: Architecture & Theory

## 1. Executive Summary

Mathematical programming solvers (Simplex, Interior Point, Branch-and-Bound) possess numerous algorithmic degrees of freedom. Choosing the optimal combination of:
- **LP Engine**: Primal Simplex vs Dual Simplex vs Barrier (IPM)
- **Pricing Rule**: Dual Steepest-Edge (DSE) vs Devex vs Textbook Dantzig
- **Branching Rule**: Most-Fractional vs Pseudocost Branching
- **Node Selection**: Best-Bound (min-heap) vs Depth-First Search (LIFO)
- **Cutting Planes**: Number of Gomory mixed-integer cut (GMIC) rounds (0 to 5)
- **Presolve & Scaling**: Presolve mode (Off, On, Aggressive) and equilibration scaling

traditionally relies on static, hand-tuned heuristics. However, refinery planning instances and industrial LPs exhibit distinctive structural topologies (e.g. sparse block-angular structures, tight blending constraints, high aspect ratios).

The **Adaptive Strategy Learning Layer** implements:
1. **Zero-Solve Structural Feature Extraction**: Extracting cheap ($< 1\text{ ms}$) topological and numerical features directly from problem data without running the solver.
2. **Nearest-Neighbor & Contextual Bandit Strategy Selection**: An empirical policy trained on historical solver telemetry logs that maps problem feature vectors $\phi(P)$ to optimal `StrategyConfig` configurations.
3. **In-Solve Adaptive Stagnation Monitoring**: Runtime monitoring for MILP tree searches that detects primal-dual gap stagnation and dynamically mutates node selection and branching strategies mid-solve.

---

## 2. Structural Feature Space $\Phi(P)$

To avoid incurring high preprocessing overhead, feature extraction is strictly $O(nnz + m + n)$ and runs in well under 5% of solve time.

### 2.1 Problem Dimensions & Density
- $m$: Number of constraints (rows)
- $n$: Number of decision variables (columns)
- $\rho = \frac{nnz}{m \cdot n}$: Matrix sparsity density
- Aspect ratio: $\log_{10}(m / n)$
- Variable types: Continuous fraction $f_{\text{cont}}$, Integer fraction $f_{\text{int}}$, Binary fraction $f_{\text{bin}}$

### 2.2 Numerical Dynamics & Condition Proxies
- Constraint bounds: Fraction of equality rows $f_{\text{eq}}$, upper-bounded $f_{\le}$, lower-bounded $f_{\ge}$, ranged $f_{\text{box}}$
- Matrix coefficient range: $\Delta_A = \log_{10}\left(\frac{\max |A_{ij}|}{\min_{|A_{ij}| > 0} |A_{ij}|}\right)$
- Objective coefficient range: $\Delta_c = \log_{10}\left(\frac{\max |c_j|}{\min_{|c_j| > 0} |c_j|}\right)$
- Right-hand side range: $\Delta_b = \log_{10}\left(\frac{\max |b_i|}{\min_{|b_i| > 0} |b_i|}\right)$
- Sparsity structure: Mean row degree $\bar{d}_{\text{row}} = nnz / m$, standard deviation $\sigma(d_{\text{row}})$, mean column degree $\bar{d}_{\text{col}} = nnz / n$

---

## 3. Strategy Selection Formulation

Let $\mathcal{A} = \{a_1, a_2, \dots, a_K\}$ denote the discrete set of discrete solver configuration profiles.

### 3.1 Profile Space
Each profile $a_k$ specifies:
```python
StrategyConfig(
    algorithm = AlgorithmChoice.[DualSimplex | PrimalSimplex | Barrier],
    pricing_rule = PricingRule.[SteepestEdge | Devex],
    branching_rule = BranchingRule.[MostFractional | PseudoCost],
    node_selection = NodeSelection.[BestBound | DepthFirst],
    cut_rounds = [0 | 1 | 3],
    presolve = PresolveMode.[Off | On],
    enable_scaling = [True | False]
)
```

### 3.2 Strategy Selector: $k$-Nearest Neighbors (KNN) & Bandit
Given an offline dataset of historical runs $\mathcal{D} = \{(\phi(P^{(i)}), a^{(i)}, \text{runtime}^{(i)}, \text{iter}^{(i)})\}_{i=1}^N$:
1. **Feature Standardization**: Features are normalized using z-score standardization:
   $$\tilde{\phi}(P) = \frac{\phi(P) - \mu_\phi}{\sigma_\phi + \epsilon}$$
2. **Distance Metric**: Euclidean distance in normalized feature space:
   $$D(P, P^{(i)}) = \|\tilde{\phi}(P) - \tilde{\phi}(P^{(i)})\|_2$$
3. **Nearest-Neighbor Retrieval**: Find $K$ closest historical instances and select the configuration that minimized runtime / iteration count on those neighbors.
4. **Contextual Bandit ($\epsilon$-Greedy)**:
   - With probability $1 - \epsilon$: exploit best-predicted configuration $\hat{a} = \arg\min_a \hat{Q}(\phi(P), a)$.
   - With probability $\epsilon$: explore a random configuration to continually expand telemetry coverage.

---

## 4. In-Solve MILP Stagnation Monitor

Branch-and-bound searches can suffer from "tailing off" or search stalls when:
1. Primal heuristics fail to find an early incumbent, forcing Best-Bound to swell the node queue without pruning.
2. Fractional branching choices fail to push objective bounds, causing the dual bound to plateau.

### 4.1 Stagnation Metric
Every $W = 50$ nodes (or after a fixed time window), the monitor computes the relative gap reduction rate:
$$\mathcal{V}_{\text{gap}} = \frac{|\text{gap}_{t} - \text{gap}_{t-W}|}{\max(1.0, |\text{gap}_{t-W}|)}$$

### 4.2 Dynamic Strategy Mutation
If $\mathcal{V}_{\text{gap}} < \tau_{\text{stall}}$ (e.g. $< 10^{-4}$ progress over $W$ nodes):
- **If currently in Best-Bound**: Temporarily mutate to **Depth-First Search (DFS)** to dive deep along fractional paths and force heuristic incumbent discovery.
- **If currently using Most-Fractional branching**: Mutate to **Pseudocost Branching** to exploit historical degradation history.
- **Telemetry Event**: An in-solve strategy shift is recorded in the run log with timestamps, node counts, and before/after node processing rates.
