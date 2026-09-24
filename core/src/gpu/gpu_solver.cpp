#include <sih/gpu/gpu_solver.hpp>
#include <sih/simplex/simplex_solver.hpp>
#include <sih/ipm/ipm_solver.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef SIH_CUDA_ENABLED
#include <cuda_runtime.h>

namespace sih {
namespace gpu {

// Declarations of kernel launchers from cuda_pdhg.cu
void launch_csr_spmv(int m, const int* d_row_ptr, const int* d_col_ind,
                     const float* d_values, const float* d_x, float* d_y,
                     cudaStream_t stream);

void launch_csc_spmv(int n, const int* d_col_ptr, const int* d_row_ind,
                     const float* d_values, const float* d_y, float* d_z,
                     cudaStream_t stream);

void launch_primal_update(int n, const float* d_tau, const float* d_c, const float* d_AT_y,
                          const float* d_l, const float* d_u, float* d_x, float* d_x_bar,
                          cudaStream_t stream);

void launch_dual_update(int m, const float* d_sigma, const float* d_A_xbar,
                        const float* d_row_l, const float* d_row_u, float* d_y,
                        cudaStream_t stream);

void launch_restart(int n, const float* d_x, float* d_x_bar, cudaStream_t stream);

void launch_residuals(int m, int n,
                      const float* d_Ax, const float* d_row_l, const float* d_row_u, float* d_row_viols,
                      const float* d_x, const float* d_c, const float* d_AT_y, const float* d_l, const float* d_u,
                      const float* d_tau, float* d_col_viols,
                      cudaStream_t stream);

bool GpuSolver::is_cuda_available() noexcept {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

GpuMemoryInfo GpuSolver::get_vram_info() {
    GpuMemoryInfo info;
    if (!is_cuda_available()) return info;

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    info.total_bytes = total_b;
    info.free_bytes = free_b;
    info.used_bytes = total_b - free_b;
    return info;
}

model::Solution GpuSolver::solve_pdhg(const model::Problem& problem,
                                     const GpuPdhgConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    model::Solution sol;
    sol.status = model::SolutionStatus::Unknown;

    if (!is_cuda_available()) {
        std::cerr << "[GpuSolver] No CUDA device found. Falling back to CPU.\n";
        model::Options opts;
        opts.strategy.algorithm = model::AlgorithmChoice::DualSimplex;
        return simplex::SimplexSolver::solve(problem, opts);
    }

    int m = static_cast<int>(problem.num_rows());
    int n = static_cast<int>(problem.num_cols());
    int nnz = static_cast<int>(problem.num_nonzeros());

    if (m == 0 || n == 0) {
        sol.status = model::SolutionStatus::Optimal;
        return sol;
    }

    // 1. Build FP32 CSR and CSC structures on Host
    const auto& csc_mat = problem.A();
    const auto& csc_col_ptr = csc_mat.csc_col_ptr();
    const auto& csc_row_ind = csc_mat.csc_row_ind();
    const auto& csc_vals_d = csc_mat.csc_values();

    std::vector<float> csc_vals_f(nnz);
    for (int k = 0; k < nnz; ++k) csc_vals_f[k] = static_cast<float>(csc_vals_d[k]);

    // Build CSR
    std::vector<int> csr_row_ptr(m + 1, 0);
    for (int idx : csc_row_ind) {
        if (idx < m) csr_row_ptr[idx + 1]++;
    }
    for (int i = 0; i < m; ++i) csr_row_ptr[i + 1] += csr_row_ptr[i];

    std::vector<int> csr_col_ind(nnz);
    std::vector<float> csr_vals_f(nnz);
    std::vector<int> row_fill = csr_row_ptr;

    for (int c = 0; c < n; ++c) {
        for (int idx = csc_col_ptr[c]; idx < csc_col_ptr[c + 1]; ++idx) {
            int r = csc_row_ind[idx];
            int dest = row_fill[r]++;
            csr_col_ind[dest] = c;
            csr_vals_f[dest] = csc_vals_f[idx];
        }
    }

    // Vectors in FP32
    std::vector<float> h_c(n), h_l(n), h_u(n), h_rl(m), h_ru(m);
    float max_abs_c = 1.0f;
    for (int j = 0; j < n; ++j) {
        h_c[j] = static_cast<float>(problem.c()[j]);
        h_l[j] = static_cast<float>(problem.col_lower()[j]);
        h_u[j] = static_cast<float>(problem.col_upper()[j]);
        max_abs_c = std::max(max_abs_c, std::abs(h_c[j]));
    }
    float max_abs_b = 1.0f;
    for (int i = 0; i < m; ++i) {
        h_rl[i] = static_cast<float>(problem.row_lower()[i]);
        h_ru[i] = static_cast<float>(problem.row_upper()[i]);
        if (h_ru[i] < 1e15f) max_abs_b = std::max(max_abs_b, std::abs(h_ru[i]));
        if (h_rl[i] > -1e15f) max_abs_b = std::max(max_abs_b, std::abs(h_rl[i]));
    }

    // Diagonal Ruiz / Pock-Chambolle step sizes
    std::vector<float> h_tau(n), h_sigma(m);
    for (int j = 0; j < n; ++j) {
        float col_sum = 0.0f;
        for (int k = csc_col_ptr[j]; k < csc_col_ptr[j + 1]; ++k) {
            col_sum += std::abs(csc_vals_f[k]);
        }
        if (config.step_tau > 0.0f) {
            h_tau[j] = config.step_tau;
        } else {
            h_tau[j] = 0.95f / std::max(1.0f, col_sum);
        }
    }
    for (int i = 0; i < m; ++i) {
        float row_sum = 0.0f;
        for (int k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k) {
            row_sum += std::abs(csr_vals_f[k]);
        }
        if (config.step_sigma > 0.0f) {
            h_sigma[i] = config.step_sigma;
        } else {
            h_sigma[i] = 0.95f / std::max(1.0f, row_sum);
        }
    }

    // 2. Allocate Device Buffers
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_ind = nullptr;
    float* d_csr_vals = nullptr;
    int* d_csc_col_ptr = nullptr;
    int* d_csc_row_ind = nullptr;
    float* d_csc_vals = nullptr;

    float *d_c = nullptr, *d_l = nullptr, *d_u = nullptr;
    float *d_rl = nullptr, *d_ru = nullptr;
    float *d_tau = nullptr, *d_sigma = nullptr;
    float *d_x = nullptr, *d_x_bar = nullptr, *d_y = nullptr;
    float *d_AT_y = nullptr, *d_A_xbar = nullptr, *d_Ax = nullptr;
    float *d_row_viols = nullptr, *d_col_viols = nullptr;

    cudaMalloc(&d_csr_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_ind, nnz * sizeof(int));
    cudaMalloc(&d_csr_vals, nnz * sizeof(float));
    cudaMalloc(&d_csc_col_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_csc_row_ind, nnz * sizeof(int));
    cudaMalloc(&d_csc_vals, nnz * sizeof(float));

    cudaMalloc(&d_c, n * sizeof(float));
    cudaMalloc(&d_l, n * sizeof(float));
    cudaMalloc(&d_u, n * sizeof(float));
    cudaMalloc(&d_rl, m * sizeof(float));
    cudaMalloc(&d_ru, m * sizeof(float));
    cudaMalloc(&d_tau, n * sizeof(float));
    cudaMalloc(&d_sigma, m * sizeof(float));

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_x_bar, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    cudaMalloc(&d_AT_y, n * sizeof(float));
    cudaMalloc(&d_A_xbar, m * sizeof(float));
    cudaMalloc(&d_Ax, m * sizeof(float));
    cudaMalloc(&d_row_viols, m * sizeof(float));
    cudaMalloc(&d_col_viols, n * sizeof(float));

    // Initialize x with projection of 0 onto [l, u], y with 0
    std::vector<float> h_x(n, 0.0f), h_y(m, 0.0f);
    for (int j = 0; j < n; ++j) {
        h_x[j] = std::max(h_l[j], std::min(h_u[j], 0.0f));
    }

    cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_vals, csr_vals_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csc_col_ptr, csc_col_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_row_ind, csc_row_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_vals, csc_vals_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_c, h_c.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl, h_rl.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ru, h_ru.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, h_tau.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, h_sigma.data(), m * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_bar, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), m * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Main PDHG Iteration Loop
    int iter = 0;
    bool converged = false;
    std::vector<float> host_row_viols(m, 0.0f);
    std::vector<float> host_col_viols(n, 0.0f);

    float best_combined_res = 1e30f;
    std::vector<float> best_x_cand = h_x;

    for (iter = 1; iter <= config.max_iterations; ++iter) {
        // Primal Step: A^T y, then x_new = proj(x - tau*(c + A^T y))
        launch_csc_spmv(n, d_csc_col_ptr, d_csc_row_ind, d_csc_vals, d_y, d_AT_y, nullptr);
        launch_primal_update(n, d_tau, d_c, d_AT_y, d_l, d_u, d_x, d_x_bar, nullptr);

        // Dual Step: A x_bar, then y_new = prox_sigma(A x_bar + y / sigma)
        launch_csr_spmv(m, d_csr_row_ptr, d_csr_col_ind, d_csr_vals, d_x_bar, d_A_xbar, nullptr);
        launch_dual_update(m, d_sigma, d_A_xbar, d_rl, d_ru, d_y, nullptr);

        // Restart check
        if (config.enable_restart && (iter % 250 == 0)) {
            launch_restart(n, d_x, d_x_bar, nullptr);
        }

        // Convergence Check
        if (iter % config.check_frequency == 0 || iter == config.max_iterations) {
            // Compute dedicated Ax and freshly evaluated A^T y for accurate residual checking
            launch_csr_spmv(m, d_csr_row_ptr, d_csr_col_ind, d_csr_vals, d_x, d_Ax, nullptr);
            launch_csc_spmv(n, d_csc_col_ptr, d_csc_row_ind, d_csc_vals, d_y, d_AT_y, nullptr);
            launch_residuals(m, n, d_Ax, d_rl, d_ru, d_row_viols,
                             d_x, d_c, d_AT_y, d_l, d_u, d_tau, d_col_viols, nullptr);

            cudaMemcpy(host_row_viols.data(), d_row_viols, m * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_col_viols.data(), d_col_viols, n * sizeof(float), cudaMemcpyDeviceToHost);

            float max_p_viol = 0.0f;
            for (float v : host_row_viols) max_p_viol = std::max(max_p_viol, v);
            float max_d_viol = 0.0f;
            for (float v : host_col_viols) max_d_viol = std::max(max_d_viol, v);

            float norm_p_res = max_p_viol / max_abs_b;
            float norm_d_res = max_d_viol / max_abs_c;
            float combined_res = std::max(norm_p_res, norm_d_res);

            if (combined_res < best_combined_res) {
                best_combined_res = combined_res;
                cudaMemcpy(best_x_cand.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

            if (norm_p_res <= config.tol_primal && norm_d_res <= config.tol_dual) {
                converged = true;
                break;
            }
        }
    }

    cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y.data(), d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device allocations
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_ind);
    cudaFree(d_csr_vals);
    cudaFree(d_csc_col_ptr);
    cudaFree(d_csc_row_ind);
    cudaFree(d_csc_vals);
    cudaFree(d_c); cudaFree(d_l); cudaFree(d_u);
    cudaFree(d_rl); cudaFree(d_ru);
    cudaFree(d_tau); cudaFree(d_sigma);
    cudaFree(d_x); cudaFree(d_x_bar); cudaFree(d_y);
    cudaFree(d_AT_y); cudaFree(d_A_xbar); cudaFree(d_Ax);
    cudaFree(d_row_viols); cudaFree(d_col_viols);

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    sol.status = converged ? model::SolutionStatus::Optimal : model::SolutionStatus::IterationLimit;
    sol.x.resize(n);
    for (int j = 0; j < n; ++j) sol.x[j] = static_cast<double>(best_x_cand[j]);

    sol.row_duals.resize(m);
    for (int i = 0; i < m; ++i) sol.row_duals[i] = static_cast<double>(h_y[i]);

    double obj = 0.0;
    for (int j = 0; j < n; ++j) obj += problem.c()[j] * sol.x[j];
    sol.primal_objective = obj;
    sol.time_wall_sec = elapsed;
    sol.simplex_iterations = iter;

    return sol;
}

model::Solution GpuSolver::solve_hybrid(const model::Problem& problem,
                                       const GpuPdhgConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Run GPU PDHG (FP32) to compute approximate solution
    model::Solution pdhg_sol = solve_pdhg(problem, config);

    int n = static_cast<int>(problem.num_cols());
    int m = static_cast<int>(problem.num_rows());

    // Compute Ax from PDHG solution to inspect row activities
    std::vector<double> Ax(m, 0.0);
    const auto& csc = problem.A();
    for (int c = 0; c < n; ++c) {
        double xc = pdhg_sol.x[c];
        if (std::abs(xc) > 1e-12) {
            for (int k = csc.csc_col_ptr()[c]; k < csc.csc_col_ptr()[c + 1]; ++k) {
                int r = csc.csc_row_ind()[k];
                Ax[r] += csc.csc_values()[k] * xc;
            }
        }
    }

    // 2. Identify candidate BasisStatus with EXACTLY m basic variables
    // Compute distance to bounds for all n variables and m slacks
    struct VarCandidate {
        int id; // 0..n-1: col, n..n+m-1: row slack
        double dist_from_bound;
        model::BasisStatus nonbasic_status;
    };

    std::vector<VarCandidate> candidates(n + m);

    for (int j = 0; j < n; ++j) {
        double val = pdhg_sol.x[j];
        double low = problem.col_lower()[j];
        double upp = problem.col_upper()[j];

        double d_low = (low > -1e15) ? std::max(0.0, val - low) : 1e30;
        double d_upp = (upp < 1e15) ? std::max(0.0, upp - val) : 1e30;
        double dist = std::min(d_low, d_upp);

        candidates[j].id = j;
        candidates[j].dist_from_bound = dist;
        candidates[j].nonbasic_status = (d_low <= d_upp) ? model::BasisStatus::AtLower : model::BasisStatus::AtUpper;
    }

    for (int i = 0; i < m; ++i) {
        double val = Ax[i];
        double low = problem.row_lower()[i];
        double upp = problem.row_upper()[i];

        double d_low = (low > -1e15) ? std::max(0.0, val - low) : 1e30;
        double d_upp = (upp < 1e15) ? std::max(0.0, upp - val) : 1e30;
        double dist = std::min(d_low, d_upp);

        candidates[n + i].id = n + i;
        candidates[n + i].dist_from_bound = dist;
        candidates[n + i].nonbasic_status = (d_low <= d_upp) ? model::BasisStatus::AtLower : model::BasisStatus::AtUpper;
    }

    // Sort descending by dist_from_bound (most interior first)
    std::sort(candidates.begin(), candidates.end(), [](const VarCandidate& a, const VarCandidate& b) {
        return a.dist_from_bound > b.dist_from_bound;
    });

    std::vector<model::BasisStatus> col_basis(n, model::BasisStatus::AtLower);
    std::vector<model::BasisStatus> row_basis(m, model::BasisStatus::AtLower);

    for (int k = 0; k < n + m; ++k) {
        int id = candidates[k].id;
        if (k < m) {
            // Top m candidates become Basic
            if (id < n) {
                col_basis[id] = model::BasisStatus::Basic;
            } else {
                row_basis[id - n] = model::BasisStatus::Basic;
            }
        } else {
            // Non-basic variables placed at their closer bound
            if (id < n) {
                col_basis[id] = candidates[k].nonbasic_status;
            } else {
                row_basis[id - n] = candidates[k].nonbasic_status;
            }
        }
    }

    // 3. CPU Polishing Pass (Phase 1 Dual Simplex solve_from_basis in FP64)
    model::Options polish_opts;
    polish_opts.strategy.algorithm = model::AlgorithmChoice::DualSimplex;
    polish_opts.strategy.presolve = model::PresolveMode::Off;
    polish_opts.strategy.enable_scaling = false;
    polish_opts.log_to_console = false;

    model::Solution polished_sol = simplex::SimplexSolver::solve_from_basis(
        problem, col_basis, row_basis, polish_opts
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    polished_sol.time_wall_sec = std::chrono::duration<double>(end_time - start_time).count();

    if (polished_sol.is_optimal()) {
        return polished_sol;
    }

    // Fallback: full CPU Simplex solve
    return simplex::SimplexSolver::solve(problem, polish_opts);
}

} // namespace gpu
} // namespace sih

#else // !SIH_CUDA_ENABLED

namespace sih {
namespace gpu {

bool GpuSolver::is_cuda_available() noexcept {
    return false;
}

GpuMemoryInfo GpuSolver::get_vram_info() {
    return GpuMemoryInfo{0, 0, 0};
}

model::Solution GpuSolver::solve_pdhg(const model::Problem& problem,
                                     const GpuPdhgConfig& config) {
    model::Options opts;
    opts.strategy.algorithm = model::AlgorithmChoice::DualSimplex;
    opts.log_to_console = false;
    return simplex::SimplexSolver::solve(problem, opts);
}

model::Solution GpuSolver::solve_hybrid(const model::Problem& problem,
                                       const GpuPdhgConfig& config) {
    model::Options opts;
    opts.strategy.algorithm = model::AlgorithmChoice::DualSimplex;
    opts.log_to_console = false;
    return simplex::SimplexSolver::solve(problem, opts);
}

} // namespace gpu
} // namespace sih

#endif // SIH_CUDA_ENABLED
