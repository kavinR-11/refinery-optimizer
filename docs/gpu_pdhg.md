# GPU-Accelerated Primal-Dual Hybrid Gradient (PDHG) LP Solver

## 1. Executive Summary & Algorithmic Foundations

Linear Programming (LP) at massive scale ($10^6$ to $10^8$ nonzeros) challenges traditional Second-Order Interior Point Methods (IPM) and Simplex algorithms due to the memory and computation bottlenecks of sparse matrix factorizations ($B = LU$ or $A \Theta A^T = L L^T$). 

**Primal-Dual Hybrid Gradient (PDHG)** (also known as the Chambolle-Pock first-order method, popularized for industrial LP by Google's PDLP) replaces matrix factorization with matrix-vector multiplications ($A x$ and $A^T y$). This maps directly onto massively parallel GPU tensor and streaming multiprocessor (SM) architectures.

This module implements:
1. **Indigenous CUDA Kernels** (`core/src/gpu/cuda_pdhg.cu`):
   - Custom Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) SpMV kernels.
   - Vectorized projection onto box bounds $[l, u]$.
   - Fused primal-dual updates and Halpern / ergodic extrapolation.
   - Parallel reduction kernels for $\ell_2$ and $\ell_\infty$ residual computation.
2. **FP32 Native Execution**: Tailored to consumer hardware (RTX 4060 Laptop GPU, Ada Lovelace sm_89) where FP32 throughput exceeds FP64 by $64\times$.
3. **Hybrid CPU Crossover**: A GPU-to-CPU pipeline where GPU PDHG quickly computes an approximate solution ($10^{-3}$ to $10^{-4}$ tolerance), which is handed off to Phase 1's Dual Simplex or Phase 2's IPM for FP64 purification to $10^{-8}$ precision.

---

## 2. Mathematical Formulation of PDHG for Linear Programming

Consider standard-form LP:
$$\min c^T x \quad \text{s.t.} \quad A x = b, \quad l \le x \le u$$

The saddle-point Lagrangian is:
$$L(x, y) = c^T x + y^T (b - A x)$$
where $x \in [l, u]$ and $y \in \mathbb{R}^m$.

### 2.1 Iteration Updates
At iteration $k \ge 0$, with primal step $\tau > 0$ and dual step $\sigma > 0$:

1. **Primal Gradient Step & Bound Projection**:
   $$x^{k+1} = \text{proj}_{[l, u]}\left(x^k - \tau (c - A^T y^k)\right)$$
   where $\text{proj}_{[l_j, u_j]}(v_j) = \min(u_j, \max(l_j, v_j))$.

2. **Extrapolation Step**:
   $$\bar{x}^{k+1} = 2 x^{k+1} - x^k$$

3. **Dual Gradient Step**:
   $$y^{k+1} = y^k + \sigma (b - A \bar{x}^{k+1})$$

### 2.2 Step-Size Selection & Preconditioning
Convergence requires:
$$\tau \sigma \|A\|_2^2 < 1$$
We estimate the spectral norm $\|A\|_2$ using power iteration on the GPU at initialization:
$$\|A\|_2 \approx \sigma_{\max}(A)$$
and set $\tau = \frac{0.95}{\|A\|_2}, \sigma = \frac{0.95}{\|A\|_2}$. Diagonal Ruiz equilibration is applied prior to iteration.

### 2.3 Normalized Convergence Residuals
The solver tracks three termination criteria:
1. **Primal Infeasibility**:
   $$\epsilon_{\text{primal}} = \frac{\|A x^k - b\|_\infty}{1.0 + \|b\|_\infty} \le \text{tol}_{\text{feas}}$$
2. **Dual Infeasibility**:
   $$\epsilon_{\text{dual}} = \frac{\|A^T y^k + s^k - c\|_\infty}{1.0 + \|c\|_\infty} \le \text{tol}_{\text{feas}}$$
   where $s^k$ is the reduced cost vector.
3. **Relative Duality Gap**:
   $$\epsilon_{\text{gap}} = \frac{|c^T x^k - b^T y^k|}{1.0 + |c^T x^k| + |b^T y^k|} \le \text{tol}_{\text{gap}}$$

---

## 3. Hardware Architecture & Precision Strategy: FP32 on RTX 4060

### 3.1 Architectural Justification
The deployment target is an **NVIDIA GeForce RTX 4060 Laptop GPU** (Ada Lovelace, compute capability 8.9):
- **SM Count**: 24 Streaming Multiprocessors, 3072 CUDA Cores.
- **FP32 Peak Compute**: $\approx 15.1\text{ TFLOPS}$.
- **FP64 Peak Compute**: $\approx 0.236\text{ TFLOPS}$ (1:64 FP64-to-FP32 rate).
- **VRAM**: 8188 MiB (8 GB) GDDR6, 128-bit bus, $256\text{ GB/s}$ bandwidth.

Executing PDHG in double precision (FP64) on consumer Ada Lovelace is severely throttled by hardware ALU ratios. Executing in single precision (**FP32**):
- Delivers up to $60\times$ higher arithmetic throughput.
- Halves memory bandwidth footprint ($4\text{ bytes}$ vs $8\text{ bytes}$ per element).
- Doubles the maximum problem dimensions that can fit into 8 GB VRAM.

### 3.2 Numerical Tolerance Realism in FP32
In IEEE 754 single precision:
- Machine epsilon $\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$ (24-bit significand).
- Expecting a first-order solver in FP32 to reach $10^{-8}$ dual feasibility is numerically unachievable.
- PDHG in FP32 targets **$\text{tol} \in [10^{-3}, 10^{-4}]$**. Full precision ($10^{-8}$) is attained via the hybrid CPU crossover pass.

---

## 4. Hybrid GPU-to-CPU Crossover Architecture

```mermaid
graph LR
    P[Large LP Instance] --> G[GPU FP32 PDHG Engine]
    G -->|~1e-4 Solution| C[Basis Identifier]
    C -->|Warm-Start Basis| S[CPU Phase 1 Simplex / IPM]
    S -->|Exact 1e-8 Basis & Coordinates| O[Optimal Solution]
```

### 4.1 Basis Identification from PDHG
From the approximate GPU solution $x^*$:
1. If $x_j^* \approx l_j$ within $\delta$, assign $j \in \mathcal{N}_L$ (`BasisStatus::AtLower`).
2. If $x_j^* \approx u_j$ within $\delta$, assign $j \in \mathcal{N}_U$ (`BasisStatus::AtUpper`).
3. If $l_j < x_j^* < u_j$, variable $j$ is structurally candidate for the basic set $\mathcal{B}$.
4. Construct a triangular or crash basis from candidate columns and hand off to `sih_solver.solve_from_basis` for 5 to 50 simplex cleanup pivots in FP64.

---

## 5. Memory Footprint & 8 GB VRAM Capacity Limits

For a sparse matrix with $m$ rows, $n$ columns, and $nnz$ nonzeros in FP32:
- Matrix storage: CSR (values $4\text{B}$, column indices $4\text{B}$, row pointers $4\text{B}$) + CSC (row indices $4\text{B}$, column pointers $4\text{B}$) $\approx 16\text{B} \times nnz + 8\text{B} \times (m + n)$.
- Vectors: $x, \bar{x}, x_{\text{prev}}, y, \bar{y}, c, l, u, b, A x, A^T y \approx 12 \times 4\text{B} \times \max(m, n)$.
- Working buffers & reductions: $\approx 128\text{ MB}$.

| Nonzeros ($nnz$) | Rows $\times$ Cols | Est. GPU Memory | Fits in 8 GB VRAM? |
| :--- | :--- | :--- | :--- |
| **$1 \times 10^6$** (1M) | $50\text{k} \times 100\text{k}$ | $\approx 32\text{ MB}$ | Yes |
| **$10 \times 10^6$** (10M) | $200\text{k} \times 500\text{k}$ | $\approx 280\text{ MB}$ | Yes |
| **$50 \times 10^6$** (50M) | $1\text{M} \times 2\text{M}$ | $\approx 1.2\text{ GB}$ | Yes |
| **$150 \times 10^6$** (150M) | $2\text{M} \times 5\text{M}$ | $\approx 3.6\text{ GB}$ | Yes |
| **$300 \times 10^6$** (300M) | $5\text{M} \times 10\text{M}$ | $\approx 7.2\text{ GB}$ | **Maximum Theoretical Fit** |
| **$> 350 \times 10^6$** | - | $> 8.4\text{ GB}$ | Out of Memory (OOM) |
