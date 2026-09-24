# Explainability & Decision Support Layer

## 1. Overview & Business Context

Mathematical solvers return optimal coordinate vectors $x^*$, slack vectors $s^*$, and dual multipliers $y^*$. For refinery production planners, operations directors, and commercial traders, raw coordinate vectors are insufficient. The **Explainability Layer** translates low-level linear algebra into actionable business insights:
1. **Economic Bottleneck Valuation (Shadow Prices & Sensitivity Ranges)**: Identifying exactly which capacity or supply limits constrain profits, and quantifying how much additional capacity is worth.
2. **Infeasibility Diagnosis (Minimal Conflicting Sets / IIS)**: When operating plans are mathematically unfeasible (e.g. post-turnaround maintenance, pipeline outages, demand spikes), identifying the minimal conflicting subset of constraints in clear human language.
3. **Comparative Scenario Analysis (Plan Deltas)**: Comparing two operating plans (e.g., base case vs crude price shock) to quantify production shifts, economic impacts, and structural bottleneck migrations.

---

## 2. Economic Interpretations: Shadow Prices & Sensitivity Ranges

### 2.1 Shadow Price Interpretation
In a linear maximization problem $\max c^T x \text{ s.t. } A x \le b$, the dual variable $y_i \ge 0$ associated with constraint $i$ represents the **marginal shadow price**:
$$y_i = \frac{\partial z^*}{\partial b_i}$$

- **Zero Shadow Price ($y_i = 0$)**: The constraint is **slack** ($A_{i, :} x^* < b_i$). Increasing capacity has zero economic value because the unit is underutilized.
- **Positive Shadow Price ($y_i > 0$)**: The constraint is **binding** ($A_{i, :} x^* = b_i$). Relaxing the limit by 1 unit increases gross refining margin by exactly $\$y_i$.

### 2.2 Sensitivity Ranges (Allowable Increase & Decrease)
Using Phase 1's sensitivity analysis (`simplex::SensitivityReport`), for each binding constraint $i$, the engine computes the range $[b_i - \Delta^-, b_i + \Delta^+]$ over which the basis $\mathcal{B}$ remains primal feasible:
$$\Delta^- = \min_{k \in \mathcal{B}, d_k < 0} \frac{x_k^* - l_k}{|d_k|}, \quad \Delta^+ = \min_{k \in \mathcal{B}, d_k > 0} \frac{u_k - x_k^*}{d_k}$$
where $d = B^{-1} e_i$.

### 2.3 Plain-Language Explanation Engine
The explainability engine maps constraint names and dual values into structured operational narratives:
- **Binding Processing Unit**:
  > *"Atmospheric Distillation Unit (ADU) is operating at 100% capacity (100,000 bpd). An additional barrel of capacity would yield **$4.85/bbl** in margin. This valuation remains valid up to 108,500 bpd (+8,500 bpd allowable increase)."*
- **Slack Processing Unit**:
  > *"Fluid Catalytic Cracker (FCC) has surplus capacity. Utilization is 72.4% (36,200 bpd / 50,000 bpd). Shadow price is **$0.00/bbl**; throughput is limited by upstream Vacuum Gas Oil feed availability."*
- **Quality Specification**:
  > *"Diesel Ultra-Low Sulfur specification (10 ppm max) is actively binding with shadow price of **$0.35 per ppm-bbl**. Blending cheaper high-sulfur gas oil is constrained by hydrotreater desulfurization limits."*

---

## 3. Infeasibility Diagnosis: Irreducible Inconsistent Subsystems (IIS)

When operational targets or contractual demands exceed physical refinery limitations, the LP relaxation is infeasible ($r_p \ne 0$). Commercial solvers merely report `Status: Infeasible`, leaving planners blind to the underlying cause.

### 3.1 IIS Deletion Filtering Algorithm
An **Irreducible Inconsistent Subsystem (IIS)** is a subset of constraints $\mathcal{C}_{IIS} \subseteq \mathcal{C}$ such that:
1. $\mathcal{C}_{IIS}$ is infeasible.
2. Every proper subset $\mathcal{C}' \subset \mathcal{C}_{IIS}$ is feasible.

The deletion filtering algorithm isolates $\mathcal{C}_{IIS}$:
```
Algorithm: Deletion-Filter IIS Extraction
Input: Infeasible Problem P with constraints C = {c_1, ..., c_m}
Output: Minimal Conflicting Set C_IIS

1. For each constraint c_i in C:
2.    Temporarily deactivate c_i from P: P' = P \ {c_i}
3.    Solve LP relaxation of P' with Dual Simplex
4.    If P' is still Infeasible:
5.        c_i is redundant to infeasibility; remove c_i permanently from C
6.    Else (P' becomes Feasible):
7.        c_i is essential to infeasibility; retain c_i in C_IIS
8. Return C_IIS
```

### 3.2 Human-Readable Diagnosis Generation
Once the minimal conflict subset $\mathcal{C}_{IIS}$ is isolated, the diagnosis generator classifies the conflict pattern:
- **Capacity vs Demand Conflict**:
  $$\sum \text{Yield} \cdot \text{Cap}_{ADU} < D_{Diesel}^{\min}$$
  *Diagnosis*: *"Contractual commitment for Diesel ($D_{Diesel}^{\min} = 45,000 \text{ bpd}$) cannot be satisfied because maximum crude distillation capacity ($100,000 \text{ bpd}$) with maximal diesel cut yield ($35\%$) produces at most $35,000 \text{ bpd}$. Deficit: $10,000 \text{ bpd}$."*
- **Quality Specification Conflict**:
  *Diagnosis*: *"Gasoline Octane target (RON $\ge 95$) cannot be met with current crude slate: Reformer capacity is insufficient to upgrade light naphtha streams to the required aromaticity."*

---

## 4. Plan Comparison Utility: Multi-Scenario Delta Analysis

Refineries continually evaluate "what-if" scenarios: crude price spikes, product tariff revisions, or unplanned unit outages. The `PlanComparator` compares two solved plans $S_1$ and $S_2$:

1. **Economic Summary Delta**:
   - Total Gross Margin change: $\Delta \text{Profit} = \text{Profit}(S_2) - \text{Profit}(S_1)$.
   - Total Crude Purchasing Expenditure delta.
   - Total Product Revenue delta.
2. **Physical Throughput Deltas**:
   - $\Delta x_c = x_c^{(2)} - x_c^{(1)}$ (Crude slate shifts).
   - $\Delta y_p = y_p^{(2)} - y_p^{(1)}$ (Finished product mix shifts).
3. **Bottleneck Migration**:
   - **Newly Binding Constraints**: Constraints that were slack in $S_1$ ($y_i^{(1)} = 0$) but become binding bottlenecks in $S_2$ ($y_i^{(2)} > 0$).
   - **Newly Slack Constraints**: Constraints that were bottlenecks in $S_1$ but have surplus capacity in $S_2$.
