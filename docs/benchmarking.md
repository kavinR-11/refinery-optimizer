# Benchmark Methodology, Performance Profiles, and Robustness Battery

## 1. Overview and Objectives

This document establishes the benchmarking protocol, performance profiling methodology, and stress-testing framework for the indigenous solver across all supported mathematical programming classes:
1. **Linear Programming (LP)**: Netlib benchmark suite evaluated with Dual Simplex and Barrier (IPM).
2. **Mixed-Integer Linear Programming (MILP)**: MIPLIB benchmark subset evaluated with Branch-and-Bound and Gomory cuts.
3. **Quadratic Programming (Convex QP)**: Maros-Mészáros benchmark suite evaluated with Primal-Dual Interior Point (Mehrotra predictor-corrector).

All benchmark evaluations are conducted strictly against the open-source **HiGHS** oracle (accessed via `highspy` strictly under `/bench` and `/tests/oracle` in full compliance with Hard Rules 1–3).

---

## 2. Quantitative Metric Formulations

### 2.1 Shifted Geometric Mean
To aggregate execution runtimes across heterogeneous problem sizes without allowing negligible sub-millisecond variations or multi-minute timeouts to distort comparative ratios, we compute the **Shifted Geometric Mean** with shift parameter $s = 1.0\text{ second}$:

$$\gamma_s = \left( \prod_{i=1}^N (t_i + s) \right)^{1/N} - s$$

- For fast sub-second problems, the shift prevents near-zero runtimes from dominating ratios.
- For timeouts or limit-exceeded runs, a penalty cap of $t_{\text{cap}} = 120.0\text{ s}$ is applied.

### 2.2 Dolan-Moré Performance Profiles
To visualize algorithmic dominance across problem distributions, we construct **Dolan-Moré Performance Profiles**. 

Let $\mathcal{S}$ denote the set of solvers (our indigenous solver vs HiGHS oracle) and $\mathcal{P}$ denote the benchmark problem set. For each problem $p \in \mathcal{P}$ and solver $s \in \mathcal{S}$, let $t_{p, s}$ be the wall-clock solve time.

The performance ratio is defined as:
$$r_{p, s} = \frac{t_{p, s}}{\min_{s' \in \mathcal{S}} t_{p, s'}}$$

The cumulative probability distribution $\rho_s(\tau)$ is:
$$\rho_s(\tau) = \frac{1}{|\mathcal{P}|} \left| \left\{ p \in \mathcal{P} : r_{p, s} \le \tau \right\} \right|$$

- $\rho_s(1.0)$ indicates the fraction of problems on which solver $s$ was the fastest algorithm.
- $\lim_{\tau \to \infty} \rho_s(\tau)$ reflects the overall solvability (success rate) of solver $s$.

---

## 3. Benchmark Problem Suites

### 3.1 Netlib LP Suite
Representative standard-form linear programs exhibiting varied aspect ratios, ill-conditioned bases, and network structures:
- `adlittle` ($56 \times 97$, 383 nnz)
- `afiro` ($27 \times 32$, 83 nnz)
- `beaconfd` ($173 \times 262$, 3,375 nnz)
- `blend` ($74 \times 83$, 491 nnz)
- `lotfi` ($153 \times 308$, 1,078 nnz)
- `sc50a` ($50 \times 48$, 130 nnz)
- `sc50b` ($50 \times 48$, 118 nnz)
- `share2b` ($96 \times 79$, 694 nnz)
- `stocfor1` ($117 \times 111$, 447 nnz)

### 3.2 MIPLIB Mixed-Integer Suite
Standard combinatorial optimization problems with discrete variables, knapsack structures, and tree-search challenges:
- `dcmulti` ($290 \times 548$, 1,315 nnz)
- `flugpl` ($18 \times 18$, 46 nnz)
- `glass4` ($396 \times 322$, 1,815 nnz)
- `markshare1` ($6 \times 62$, 312 nnz)
- `pk1` ($45 \times 86$, 915 nnz)

### 3.3 Maros-Mészáros Convex QP Suite
Standard quadratic programs with dense/sparse Hessian matrices $Q \succeq 0$:
- `CVXQP1_S` ($10 \times 14$)
- `CVXQP2_S` ($25 \times 40$)
- `CVXQP3_S` ($50 \times 75$)
- `DUAL1` ($85 \times 1)
- `DUAL2` ($96 \times 1)
- `DUAL3` ($111 \times 1)
- `DUAL4` ($75 \times 1)
- `HS21` ($1 \times 2$)
- `HS35` ($1 \times 3$)
- `HS35MOD` ($1 \times 3$)

---

## 4. Robustness Hardening Battery

To ensure numerical stability in industrial environments, the robustness battery subjects the solver to edge cases:
1. **Degenerate LPs**: Over-determined bases with multiple zero basic variables, testing anti-cycling Bland's rule fallbacks and cost perturbation.
2. **Badly Scaled LPs**: Matrix coefficients spanning $10^{-6}$ to $10^{6}$, testing geometric mean equilibration and power-of-two scaling.
3. **Near-Infeasible LPs**: Feasible regions with volume $\epsilon < 10^{-7}$, testing Markowitz threshold pivoting drift detection.
4. **Randomized Monte Carlo Stress Test**: 100 distinct seeded random instances with variable dimensions and densities, testing memory safety and exception hygiene.
