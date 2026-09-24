#include "sih/ipm/ipm_solver.hpp"
#include "sih/ipm/crossover.hpp"
#include "sih/factorization/sparse_cholesky.hpp"
#include "sih/factorization/sparse_ldl.hpp"
#include "sih/factorization/amd.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace sih {
namespace ipm {

model::Solution IpmSolver::solve(const model::Problem& problem,
                                 const model::Options& options) {
    auto start_time = std::chrono::high_resolution_clock::now();

    model::Solution sol;
    sol.status = model::SolutionStatus::Unknown;

    int64_t m = problem.num_rows();
    int64_t n = problem.num_cols();
    int64_t N = n + m; // Total variables: [x; s]

    if (n == 0) {
        sol.status = model::SolutionStatus::Optimal;
        sol.primal_objective = problem.obj_offset();
        return sol;
    }

    double sense_mult = (problem.sense() == model::ObjectiveSense::Maximize) ? -1.0 : 1.0;

    // Linear objective c_bar = [c * sense_mult; 0]
    std::vector<double> c_bar(N, 0.0);
    const auto& orig_c = problem.c();
    for (int64_t j = 0; j < n; ++j) {
        c_bar[j] = orig_c[j] * sense_mult;
    }

    // Combined bounds: lb_bar = [col_lower; row_lower], ub_bar = [col_upper; row_upper]
    const auto& orig_cl = problem.col_lower();
    const auto& orig_cu = problem.col_upper();
    const auto& orig_rl = problem.row_lower();
    const auto& orig_ru = problem.row_upper();

    std::vector<double> lb_bar(N), ub_bar(N);
    std::vector<bool> has_lb(N), has_ub(N);

    for (int64_t j = 0; j < n; ++j) {
        has_lb[j] = model::is_bounded_below(orig_cl[j]);
        has_ub[j] = model::is_bounded_above(orig_cu[j]);
        lb_bar[j] = has_lb[j] ? orig_cl[j] : -model::SIH_INFINITY;
        ub_bar[j] = has_ub[j] ? orig_cu[j] :  model::SIH_INFINITY;
    }
    for (int64_t i = 0; i < m; ++i) {
        int64_t k = n + i;
        has_lb[k] = model::is_bounded_below(orig_rl[i]);
        has_ub[k] = model::is_bounded_above(orig_ru[i]);
        lb_bar[k] = has_lb[k] ? orig_rl[i] : -model::SIH_INFINITY;
        ub_bar[k] = has_ub[k] ? orig_ru[i] :  model::SIH_INFINITY;
    }

    // Classify fixed variables and equality constraints (where lb == ub)
    std::vector<bool> is_fixed(N, false);
    for (int64_t k = 0; k < N; ++k) {
        if (has_lb[k] && has_ub[k] && std::abs(ub_bar[k] - lb_bar[k]) < 1e-12) {
            is_fixed[k] = true;
            has_lb[k] = false;
            has_ub[k] = false;
        }
    }

    // Initial point: strictly interior
    std::vector<double> x_bar(N, 0.0);
    std::vector<double> xl(N, 0.0), xu(N, 0.0);
    std::vector<double> zl(N, 0.0), zu(N, 0.0);

    // 1. Initial point for structural variables (0..n-1)
    for (int64_t j = 0; j < n; ++j) {
        if (is_fixed[j]) {
            x_bar[j] = lb_bar[j];
            xl[j] = 0.0; xu[j] = 0.0; zl[j] = 0.0; zu[j] = 0.0;
        } else if (has_lb[j] && has_ub[j]) {
            x_bar[j] = 0.5 * (lb_bar[j] + ub_bar[j]);
            xl[j] = x_bar[j] - lb_bar[j];
            xu[j] = ub_bar[j] - x_bar[j];
            zl[j] = 1.0; zu[j] = 1.0;
        } else if (has_lb[j]) {
            x_bar[j] = lb_bar[j] + 1.0;
            xl[j] = 1.0;
            zl[j] = std::max(1.0, std::abs(c_bar[j]));
        } else if (has_ub[j]) {
            x_bar[j] = ub_bar[j] - 1.0;
            xu[j] = 1.0;
            zu[j] = std::max(1.0, std::abs(c_bar[j]));
        } else {
            x_bar[j] = 0.0;
        }
    }

    // Compute Ax0 to project slacks close to feasibility
    std::vector<double> x0(x_bar.begin(), x_bar.begin() + n);
    auto Ax0 = problem.A().mat_vec(x0);

    // 2. Initial point for row slacks (n..N-1)
    for (int64_t i = 0; i < m; ++i) {
        int64_t k = n + i;
        if (is_fixed[k]) {
            x_bar[k] = lb_bar[k];
            xl[k] = 0.0; xu[k] = 0.0; zl[k] = 0.0; zu[k] = 0.0;
        } else if (has_lb[k] && has_ub[k]) {
            double span = ub_bar[k] - lb_bar[k];
            double v = std::clamp(Ax0[i], lb_bar[k] + 0.1 * span, ub_bar[k] - 0.1 * span);
            x_bar[k] = v;
            xl[k] = v - lb_bar[k];
            xu[k] = ub_bar[k] - v;
            zl[k] = 1.0; zu[k] = 1.0;
        } else if (has_lb[k]) {
            double v = std::max(Ax0[i], lb_bar[k] + 1.0);
            x_bar[k] = v;
            xl[k] = v - lb_bar[k];
            zl[k] = 1.0;
        } else if (has_ub[k]) {
            double v = std::min(Ax0[i], ub_bar[k] - 1.0);
            x_bar[k] = v;
            xu[k] = ub_bar[k] - v;
            zu[k] = 1.0;
        } else {
            x_bar[k] = Ax0[i];
        }
    }

    // Dual variables y initialized from slack duals
    std::vector<double> y(m, 0.0);
    for (int64_t i = 0; i < m; ++i) {
        int64_t k = n + i;
        y[i] = (has_lb[k] ? zl[k] : 0.0) - (has_ub[k] ? zu[k] : 0.0);
    }

    // Symbolic AMD ordering on normal equations
    auto perm = factorization::AmdOrder::order_aat(problem.A());

    const auto& cfg = options.strategy;
    int64_t max_iters = cfg.ipm_max_iterations;
    double p_tol = cfg.ipm_primal_tol;
    double d_tol = cfg.ipm_dual_tol;
    double g_tol = cfg.ipm_gap_tol;
    double tau   = cfg.ipm_step_safety;
    double reg   = cfg.ipm_regularization;

    double norm_b = 1.0;
    for (int64_t i = 0; i < m; ++i) {
        if (has_lb[n + i]) norm_b = std::max(norm_b, std::abs(lb_bar[n + i]));
        if (has_ub[n + i]) norm_b = std::max(norm_b, std::abs(ub_bar[n + i]));
    }
    double norm_c = 1.0;
    for (int64_t j = 0; j < n; ++j) norm_c = std::max(norm_c, std::abs(c_bar[j]));

    bool is_qp = problem.is_qp();
    const auto& Q = problem.Q();
    bool q_is_diagonal = true;
    if (is_qp) {
        const auto& col_ptr = Q.csc_col_ptr();
        const auto& row_ind = Q.csc_row_ind();
        for (int64_t j = 0; j < n && q_is_diagonal; ++j) {
            for (int64_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
                if (row_ind[p] != j) {
                    q_is_diagonal = false;
                    break;
                }
            }
        }
    }
    bool use_augmented = (is_qp && !q_is_diagonal);


    // Main Mehrotra Predictor-Corrector Loop
    int64_t iter = 0;
    for (; iter < max_iters; ++iter) {
        // Structural variables x = x_bar[0..n-1], slacks s = x_bar[n..N-1]
        std::vector<double> x(x_bar.begin(), x_bar.begin() + n);
        std::vector<double> s(x_bar.begin() + n, x_bar.end());

        // A * x
        std::vector<double> Ax = problem.A().mat_vec(x);

        // Primal residual: r_p = A * x - s
        std::vector<double> rp(m);
        double max_rp = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            rp[i] = Ax[i] - s[i];
            max_rp = std::max(max_rp, std::abs(rp[i]));
        }

        // Q * x (for QP)
        std::vector<double> Qx(n, 0.0);
        if (is_qp) {
            auto raw_Qx = Q.mat_vec(x);
            for (int64_t j = 0; j < n; ++j) Qx[j] = raw_Qx[j] * sense_mult;
        }

        // A^T * y
        std::vector<double> Aty = problem.A().mat_trans_vec(y);

        // Dual residual r_d for all N variables:
        // For j < n: r_d[j] = c[j] + Qx[j] - A^T y - zl[j] + zu[j]
        // For k = n + i: r_d[n+i] = 0 + 0 - (-y[i]) - zl[n+i] + zu[n+i] = y[i] - zl[n+i] + zu[n+i]
        std::vector<double> rd(N, 0.0);
        double max_rd = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            if (is_fixed[j]) continue;
            double z_net = (has_lb[j] ? zl[j] : 0.0) - (has_ub[j] ? zu[j] : 0.0);
            rd[j] = c_bar[j] + Qx[j] - Aty[j] - z_net;
            max_rd = std::max(max_rd, std::abs(rd[j]));
        }
        for (int64_t i = 0; i < m; ++i) {
            int64_t k = n + i;
            if (is_fixed[k]) continue;
            double z_net = (has_lb[k] ? zl[k] : 0.0) - (has_ub[k] ? zu[k] : 0.0);
            rd[k] = y[i] - z_net;
            max_rd = std::max(max_rd, std::abs(rd[k]));
        }

        // Complementarity barrier measure mu
        double comp_sum = 0.0;
        int64_t n_bounds = 0;
        for (int64_t k = 0; k < N; ++k) {
            if (is_fixed[k]) continue;
            if (has_lb[k]) { comp_sum += xl[k] * zl[k]; n_bounds++; }
            if (has_ub[k]) { comp_sum += xu[k] * zu[k]; n_bounds++; }
        }
        if (n_bounds == 0) n_bounds = 1;
        double mu = comp_sum / static_cast<double>(n_bounds);

        // Objectives and duality gap
        double p_obj = 0.0;
        for (int64_t j = 0; j < n; ++j) p_obj += c_bar[j] * x[j];
        if (is_qp) {
            for (int64_t j = 0; j < n; ++j) p_obj += 0.5 * x[j] * Qx[j];
        }

        double d_obj = 0.0;
        for (int64_t k = 0; k < N; ++k) {
            if (is_fixed[k]) continue;
            if (has_lb[k]) d_obj += lb_bar[k] * zl[k];
            if (has_ub[k]) d_obj -= ub_bar[k] * zu[k];
        }
        for (int64_t i = 0; i < m; ++i) {
            int64_t k = n + i;
            if (is_fixed[k]) {
                // For equality rows, y_i * b_i is the dual contribution
                d_obj += y[i] * lb_bar[k];
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            if (is_fixed[j]) {
                // For fixed variables, (c_j - A_j^T y) * b_j
                double rc_j = c_bar[j] + Qx[j] - Aty[j];
                d_obj += rc_j * lb_bar[j];
            }
        }
        if (is_qp) {
            for (int64_t j = 0; j < n; ++j) d_obj -= 0.5 * x[j] * Qx[j];
        }

        double norm_grad = std::max(1.0, norm_c);
        if (is_qp) {
            for (int64_t j = 0; j < n; ++j) {
                norm_grad = std::max(norm_grad, std::abs(c_bar[j] + Qx[j]));
            }
        }
        double rel_rp = max_rp / norm_b;
        double rel_rd = max_rd / norm_grad;
        double rel_gap = comp_sum / (1.0 + std::abs(p_obj));
        if (options.log_to_console) {
            std::cout << "[IPM Iter " << iter << "] p_obj=" << p_obj << " rel_rp=" << rel_rp
                      << " rel_rd=" << rel_rd << " rel_gap=" << rel_gap << " mu=" << mu << std::endl;
        }

        if (rel_rp <= p_tol && rel_rd <= d_tol && rel_gap <= g_tol) {
            sol.status = model::SolutionStatus::Optimal;
            break;
        }

        // Scaling diagonal Theta_k = zl[k]/xl[k] + zu[k]/xu[k]
        std::vector<double> Theta(N, 0.0);
        for (int64_t k = 0; k < N; ++k) {
            if (is_fixed[k]) continue;
            double diag = 0.0;
            if (has_lb[k]) diag += zl[k] / xl[k];
            if (has_ub[k]) diag += zu[k] / xu[k];
            Theta[k] = std::max(1e-12, diag);
        }

        std::vector<double> D_col(n, 0.0);
        for (int64_t j = 0; j < n; ++j) {
            if (is_fixed[j]) continue;
            double q_jj = 0.0;
            if (is_qp) {
                const auto& q_col_ptr = Q.csc_col_ptr();
                const auto& q_row_ind = Q.csc_row_ind();
                const auto& q_vals    = Q.csc_values();
                for (int64_t p = q_col_ptr[j]; p < q_col_ptr[j + 1]; ++p) {
                    if (q_row_ind[p] == j) {
                        q_jj = q_vals[p] * sense_mult;
                        break;
                    }
                }
            }
            D_col[j] = 1.0 / (q_jj + Theta[j]);
        }

        std::vector<double> D_slack(m, 0.0);
        for (int64_t i = 0; i < m; ++i) {
            int64_t k = n + i;
            if (is_fixed[k]) continue;
            D_slack[i] = 1.0 / Theta[k];
        }

        factorization::SparseCholesky chol;
        factorization::SparseLdl ldl;

        if (use_augmented) {
            std::vector<model::Triplet> kkt_triplets;
            if (is_qp) {
                const auto& q_col_ptr = Q.csc_col_ptr();
                const auto& q_row_ind = Q.csc_row_ind();
                const auto& q_vals    = Q.csc_values();
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t p = q_col_ptr[j]; p < q_col_ptr[j + 1]; ++p) {
                        kkt_triplets.emplace_back(q_row_ind[p], j, q_vals[p] * sense_mult);
                    }
                }
            }
            for (int64_t j = 0; j < n; ++j) {
                if (is_fixed[j]) {
                    kkt_triplets.emplace_back(j, j, 1e12);
                } else {
                    kkt_triplets.emplace_back(j, j, Theta[j]);
                }
            }
            const auto& a_col_ptr = problem.A().csc_col_ptr();
            const auto& a_row_ind = problem.A().csc_row_ind();
            const auto& a_vals    = problem.A().csc_values();
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t p = a_col_ptr[j]; p < a_col_ptr[j + 1]; ++p) {
                    int64_t i = a_row_ind[p];
                    double val = -a_vals[p];
                    kkt_triplets.emplace_back(j, n + i, val);
                    kkt_triplets.emplace_back(n + i, j, val);
                }
            }
            for (int64_t i = 0; i < m; ++i) {
                int64_t k = n + i;
                if (is_fixed[k]) {
                    kkt_triplets.emplace_back(n + i, n + i, -reg);
                } else {
                    kkt_triplets.emplace_back(n + i, n + i, -D_slack[i] - reg);
                }
            }
            auto K = model::SparseMatrix::from_triplets(n + m, n + m, kkt_triplets);
            auto kkt_perm = factorization::AmdOrder::order_matrix(n + m, K.csc_col_ptr(), K.csc_row_ind());
            bool ldl_ok = ldl.factorize(n + m, K, kkt_perm, reg);
            if (!ldl_ok) break;
        } else {
            bool chol_ok = chol.factorize_normal_equations(problem.A(), D_col, D_slack, perm, reg);
            if (!chol_ok) break;
        }

        // --- STEP 1: Predictor (Affine) Phase ---
        std::vector<double> rx_aff(N);
        for (int64_t k = 0; k < N; ++k) {
            double z_k = (has_lb[k] ? zl[k] : 0.0) - (has_ub[k] ? zu[k] : 0.0);
            rx_aff[k] = -rd[k] - z_k;
        }

        std::vector<double> dx_bar_aff(N, 0.0);
        std::vector<double> dy_aff(m, 0.0);

        if (use_augmented) {
            std::vector<double> rhs_aff(n + m);
            for (int64_t j = 0; j < n; ++j) rhs_aff[j] = is_fixed[j] ? 0.0 : rx_aff[j];
            for (int64_t i = 0; i < m; ++i) rhs_aff[n + i] = rp[i] - D_slack[i] * rx_aff[n + i];
            auto sol_aff = ldl.solve(rhs_aff);
            for (int64_t j = 0; j < n; ++j) dx_bar_aff[j] = is_fixed[j] ? 0.0 : sol_aff[j];
            for (int64_t i = 0; i < m; ++i) {
                dy_aff[i] = sol_aff[n + i];
                dx_bar_aff[n + i] = is_fixed[n + i] ? 0.0 : D_slack[i] * (rx_aff[n + i] - dy_aff[i]);
            }
        } else {
            std::vector<double> D_rx_aff_col(n);
            for (int64_t j = 0; j < n; ++j) D_rx_aff_col[j] = D_col[j] * rx_aff[j];
            std::vector<double> A_D_rx_aff = problem.A().mat_vec(D_rx_aff_col);

            std::vector<double> ry_aff(m);
            for (int64_t i = 0; i < m; ++i) {
                ry_aff[i] = -rp[i] - A_D_rx_aff[i] + D_slack[i] * rx_aff[n + i];
            }
            dy_aff = chol.solve(ry_aff);
            std::vector<double> Atdy_aff = problem.A().mat_trans_vec(dy_aff);
            for (int64_t j = 0; j < n; ++j) {
                dx_bar_aff[j] = D_col[j] * (rx_aff[j] + Atdy_aff[j]);
            }
            for (int64_t i = 0; i < m; ++i) {
                dx_bar_aff[n + i] = D_slack[i] * (rx_aff[n + i] - dy_aff[i]);
            }
        }

        // Affine step lengths to boundary
        double a_p_aff = 1.0;
        double a_d_aff = 1.0;

        std::vector<double> dzl_aff(N, 0.0), dzu_aff(N, 0.0);
        for (int64_t k = 0; k < N; ++k) {
            if (has_lb[k]) {
                if (dx_bar_aff[k] < 0.0) a_p_aff = std::min(a_p_aff, -xl[k] / dx_bar_aff[k]);
                dzl_aff[k] = -zl[k] - (zl[k] / xl[k]) * dx_bar_aff[k];
                if (dzl_aff[k] < 0.0) a_d_aff = std::min(a_d_aff, -zl[k] / dzl_aff[k]);
            }
            if (has_ub[k]) {
                if (-dx_bar_aff[k] < 0.0) a_p_aff = std::min(a_p_aff, -xu[k] / (-dx_bar_aff[k]));
                dzu_aff[k] = -zu[k] + (zu[k] / xu[k]) * dx_bar_aff[k];
                if (dzu_aff[k] < 0.0) a_d_aff = std::min(a_d_aff, -zu[k] / dzu_aff[k]);
            }
        }

        // Affine barrier measure mu_aff
        double comp_aff = 0.0;
        for (int64_t k = 0; k < N; ++k) {
            if (has_lb[k]) comp_aff += (xl[k] + a_p_aff * dx_bar_aff[k]) * (zl[k] + a_d_aff * dzl_aff[k]);
            if (has_ub[k]) comp_aff += (xu[k] - a_p_aff * dx_bar_aff[k]) * (zu[k] + a_d_aff * dzu_aff[k]);
        }
        double mu_aff = comp_aff / static_cast<double>(n_bounds);

        double ratio = (mu > 0.0) ? (mu_aff / mu) : 0.0;
        double sigma = std::clamp(std::pow(ratio, cfg.ipm_centering_exponent), 0.0, 1.0);

        // --- STEP 2: Corrector & Centering Phase ---
        std::vector<double> rx_cor(N);
        for (int64_t k = 0; k < N; ++k) {
            double z_k = (has_lb[k] ? zl[k] : 0.0) - (has_ub[k] ? zu[k] : 0.0);
            double corr_l = has_lb[k] ? (sigma * mu - dx_bar_aff[k] * dzl_aff[k]) / xl[k] : 0.0;
            double corr_u = has_ub[k] ? (sigma * mu + dx_bar_aff[k] * dzu_aff[k]) / xu[k] : 0.0;
            rx_cor[k] = -rd[k] - z_k + corr_l - corr_u;
        }

        std::vector<double> dx_bar(N, 0.0);
        std::vector<double> dy(m, 0.0);

        if (use_augmented) {
            std::vector<double> rhs_cor(n + m);
            for (int64_t j = 0; j < n; ++j) rhs_cor[j] = is_fixed[j] ? 0.0 : rx_cor[j];
            for (int64_t i = 0; i < m; ++i) rhs_cor[n + i] = rp[i] - D_slack[i] * rx_cor[n + i];
            auto sol_cor = ldl.solve(rhs_cor);
            for (int64_t j = 0; j < n; ++j) dx_bar[j] = is_fixed[j] ? 0.0 : sol_cor[j];
            for (int64_t i = 0; i < m; ++i) {
                dy[i] = sol_cor[n + i];
                dx_bar[n + i] = is_fixed[n + i] ? 0.0 : D_slack[i] * (rx_cor[n + i] - dy[i]);
            }
        } else {
            std::vector<double> D_rx_cor_col(n);
            for (int64_t j = 0; j < n; ++j) D_rx_cor_col[j] = D_col[j] * rx_cor[j];
            std::vector<double> A_D_rx_cor = problem.A().mat_vec(D_rx_cor_col);

            std::vector<double> ry_cor(m);
            for (int64_t i = 0; i < m; ++i) {
                ry_cor[i] = -rp[i] - A_D_rx_cor[i] + D_slack[i] * rx_cor[n + i];
            }
            dy = chol.solve(ry_cor);
            std::vector<double> Atdy = problem.A().mat_trans_vec(dy);
            for (int64_t j = 0; j < n; ++j) {
                dx_bar[j] = D_col[j] * (rx_cor[j] + Atdy[j]);
            }
            for (int64_t i = 0; i < m; ++i) {
                dx_bar[n + i] = D_slack[i] * (rx_cor[n + i] - dy[i]);
            }
        }



        // Combined dual directions
        std::vector<double> dzl(N, 0.0), dzu(N, 0.0);
        for (int64_t k = 0; k < N; ++k) {
            if (has_lb[k]) {
                dzl[k] = -zl[k] + (sigma * mu - dx_bar_aff[k] * dzl_aff[k] - zl[k] * dx_bar[k]) / xl[k];
            }
            if (has_ub[k]) {
                dzu[k] = -zu[k] + (sigma * mu - (-dx_bar_aff[k]) * dzu_aff[k] + zu[k] * dx_bar[k]) / xu[k];
            }
        }

        // Final step sizes
        double a_p = 1.0;
        double a_d = 1.0;
        for (int64_t k = 0; k < N; ++k) {
            if (has_lb[k]) {
                if (dx_bar[k] < 0.0) a_p = std::min(a_p, -xl[k] / dx_bar[k]);
                if (dzl[k] < 0.0)    a_d = std::min(a_d, -zl[k] / dzl[k]);
            }
            if (has_ub[k]) {
                if (-dx_bar[k] < 0.0) a_p = std::min(a_p, -xu[k] / (-dx_bar[k]));
                if (dzu[k] < 0.0)     a_d = std::min(a_d, -zu[k] / dzu[k]);
            }
        }

        a_p = std::min(1.0, tau * a_p);
        a_d = std::min(1.0, tau * a_d);

        // Update coordinates
        for (int64_t k = 0; k < N; ++k) {
            x_bar[k] += a_p * dx_bar[k];
            if (has_lb[k]) {
                xl[k] += a_p * dx_bar[k];
                zl[k] += a_d * dzl[k];
            }
            if (has_ub[k]) {
                xu[k] -= a_p * dx_bar[k];
                zu[k] += a_d * dzu[k];
            }
        }
        for (int64_t i = 0; i < m; ++i) {
            y[i] += a_d * dy[i];
        }
    }

    sol.simplex_iterations = iter;
    sol.barrier_iterations = iter;

    // Unpack solution
    sol.x.assign(x_bar.begin(), x_bar.begin() + n);
    sol.slack.assign(x_bar.begin() + n, x_bar.end());
    sol.row_duals = y;

    // Reduced costs
    std::vector<double> rc(n);
    std::vector<double> Aty_final = problem.A().mat_trans_vec(y);
    std::vector<double> Qx_final(n, 0.0);
    if (is_qp) {
        auto raw_Qx = Q.mat_vec(sol.x);
        for (int64_t j = 0; j < n; ++j) Qx_final[j] = raw_Qx[j] * sense_mult;
    }
    for (int64_t j = 0; j < n; ++j) {
        rc[j] = c_bar[j] + Qx_final[j] - Aty_final[j];
    }
    sol.reduced_costs = rc;

    // Objective value
    double final_obj = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        final_obj += orig_c[j] * sol.x[j];
    }
    if (is_qp) {
        auto raw_Qx = Q.mat_vec(sol.x);
        for (int64_t j = 0; j < n; ++j) {
            final_obj += 0.5 * sol.x[j] * raw_Qx[j];
        }
    }
    final_obj += problem.obj_offset();
    sol.primal_objective = final_obj;

    // Basis statuses
    sol.col_basis.assign(n, model::BasisStatus::Basic);
    for (int64_t j = 0; j < n; ++j) {
        if (has_lb[j] && std::abs(sol.x[j] - lb_bar[j]) < 1e-5) {
            sol.col_basis[j] = model::BasisStatus::AtLower;
        } else if (has_ub[j] && std::abs(sol.x[j] - ub_bar[j]) < 1e-5) {
            sol.col_basis[j] = model::BasisStatus::AtUpper;
        }
    }

    sol.row_basis.assign(m, model::BasisStatus::Basic);
    for (int64_t i = 0; i < m; ++i) {
        int64_t k = n + i;
        if (has_lb[k] && std::abs(sol.slack[i] - lb_bar[k]) < 1e-5) {
            sol.row_basis[i] = model::BasisStatus::AtLower;
        } else if (has_ub[k] && std::abs(sol.slack[i] - ub_bar[k]) < 1e-5) {
            sol.row_basis[i] = model::BasisStatus::AtUpper;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    sol.time_wall_sec = std::chrono::duration<double>(end_time - start_time).count();

    // 3. Optional Vertex Crossover for LPs
    if (cfg.ipm_enable_crossover && !is_qp && sol.is_optimal()) {
        sol = Crossover::crossover(problem, sol, options);
    }

    return sol;
}

} // namespace ipm
} // namespace sih
