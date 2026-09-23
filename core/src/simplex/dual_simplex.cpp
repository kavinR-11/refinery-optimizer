#include "sih/simplex/dual_simplex.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace sih {
namespace simplex {

DualSimplexEngine::DualSimplexEngine(const model::Problem& problem, const model::Options& options)
    : m_problem(problem), m_options(options) {
    m_m = problem.num_rows();
    m_n = problem.num_cols();
    m_num_total = m_n + m_m;

    // Expand problem into standard bounded form:
    // [A  -I] [x; s] = 0, lb <= x <= ub, l <= s <= u
    // Objective: min c^T x (if maximize, negate c internally)
    double sense_mult = (problem.sense() == model::ObjectiveSense::Maximize) ? -1.0 : 1.0;

    m_c.assign(m_num_total, 0.0);
    const auto& orig_c = problem.c();
    for (int64_t j = 0; j < m_n; ++j) {
        m_c[j] = orig_c[j] * sense_mult;
    }

    m_lb.assign(m_num_total, 0.0);
    m_ub.assign(m_num_total, 0.0);

    const auto& col_l = problem.col_lower();
    const auto& col_u = problem.col_upper();
    for (int64_t j = 0; j < m_n; ++j) {
        m_lb[j] = col_l[j];
        m_ub[j] = col_u[j];
    }

    const auto& row_l = problem.row_lower();
    const auto& row_u = problem.row_upper();
    for (int64_t i = 0; i < m_m; ++i) {
        m_lb[m_n + i] = row_l[i];
        m_ub[m_n + i] = row_u[i];
    }

    m_basic_vars.assign(m_m, -1);
    m_var_in_basis.assign(m_num_total, -1);
    m_status.assign(m_num_total, model::BasisStatus::AtLower);

    m_x.assign(m_num_total, 0.0);
    m_s.assign(m_num_total, 0.0);
    m_y.assign(m_m, 0.0);
    m_dse_weights.assign(m_m, 1.0);
}

void DualSimplexEngine::init_cold_start() {
    // All slacks are basic: B = -I
    for (int64_t i = 0; i < m_m; ++i) {
        int64_t slack_var = m_n + i;
        m_basic_vars[i] = slack_var;
        m_var_in_basis[slack_var] = i;
        m_status[slack_var] = model::BasisStatus::Basic;
    }

    // Structural variables are non-basic
    for (int64_t j = 0; j < m_n; ++j) {
        m_var_in_basis[j] = -1;
        if (m_c[j] >= 0.0 && m_lb[j] > -model::SIH_INFINITY / 2.0) {
            m_status[j] = model::BasisStatus::AtLower;
            m_x[j] = m_lb[j];
        } else if (m_c[j] < 0.0 && m_ub[j] < model::SIH_INFINITY / 2.0) {
            m_status[j] = model::BasisStatus::AtUpper;
            m_x[j] = m_ub[j];
        } else if (m_lb[j] > -model::SIH_INFINITY / 2.0) {
            m_status[j] = model::BasisStatus::AtLower;
            m_x[j] = m_lb[j];
        } else if (m_ub[j] < model::SIH_INFINITY / 2.0) {
            m_status[j] = model::BasisStatus::AtUpper;
            m_x[j] = m_ub[j];
        } else {
            m_status[j] = model::BasisStatus::Free;
            m_x[j] = 0.0;
        }
    }

    refactorize_basis();
    init_dse_weights();
    compute_primal_basic();
    compute_dual_and_reduced_costs();
}

void DualSimplexEngine::init_warm_start(const std::vector<model::BasisStatus>& col_basis,
                                        const std::vector<model::BasisStatus>& row_basis) {
    if (col_basis.size() != static_cast<size_t>(m_n) || row_basis.size() != static_cast<size_t>(m_m)) {
        init_cold_start();
        return;
    }

    std::vector<int64_t> basic_candidates;
    for (int64_t j = 0; j < m_n; ++j) {
        if (col_basis[j] == model::BasisStatus::Basic) {
            basic_candidates.push_back(j);
        }
    }
    for (int64_t i = 0; i < m_m; ++i) {
        if (row_basis[i] == model::BasisStatus::Basic) {
            basic_candidates.push_back(m_n + i);
        }
    }

    if (static_cast<int64_t>(basic_candidates.size()) != m_m) {
        // Fallback to cold start if invalid basis size
        init_cold_start();
        return;
    }

    m_var_in_basis.assign(m_num_total, -1);
    for (int64_t i = 0; i < m_m; ++i) {
        int64_t v = basic_candidates[i];
        m_basic_vars[i] = v;
        m_var_in_basis[v] = i;
        m_status[v] = model::BasisStatus::Basic;
    }

    for (int64_t j = 0; j < m_n; ++j) {
        if (m_var_in_basis[j] == -1) {
            m_status[j] = col_basis[j];
            if (m_status[j] == model::BasisStatus::AtUpper && m_ub[j] < model::SIH_INFINITY / 2.0) {
                m_x[j] = m_ub[j];
            } else if (m_lb[j] > -model::SIH_INFINITY / 2.0) {
                m_status[j] = model::BasisStatus::AtLower;
                m_x[j] = m_lb[j];
            } else {
                m_status[j] = model::BasisStatus::Free;
                m_x[j] = 0.0;
            }
        }
    }

    for (int64_t i = 0; i < m_m; ++i) {
        int64_t v = m_n + i;
        if (m_var_in_basis[v] == -1) {
            m_status[v] = row_basis[i];
            if (m_status[v] == model::BasisStatus::AtUpper && m_ub[v] < model::SIH_INFINITY / 2.0) {
                m_x[v] = m_ub[v];
            } else if (m_lb[v] > -model::SIH_INFINITY / 2.0) {
                m_status[v] = model::BasisStatus::AtLower;
                m_x[v] = m_lb[v];
            } else {
                m_status[v] = model::BasisStatus::Free;
                m_x[v] = 0.0;
            }
        }
    }

    refactorize_basis();
    if (!m_lu.is_valid()) {
        init_cold_start();
        return;
    }

    init_dse_weights();
    compute_primal_basic();
    compute_dual_and_reduced_costs();
}

std::vector<double> DualSimplexEngine::get_column(int64_t j) const {
    std::vector<double> col(m_m, 0.0);
    if (j < m_n) {
        const auto& col_ptr = m_problem.A().csc_col_ptr();
        const auto& row_ind = m_problem.A().csc_row_ind();
        const auto& vals    = m_problem.A().csc_values();
        for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            col[row_ind[k]] = vals[k];
        }
    } else {
        int64_t slack_row = j - m_n;
        if (slack_row >= 0 && slack_row < m_m) {
            col[slack_row] = -1.0;
        }
    }
    return col;
}

void DualSimplexEngine::refactorize_basis() {
    m_lu.factorize(m_m, m_basic_vars, m_problem.A(), 0.1, m_options.zero_tol);
}

void DualSimplexEngine::compute_primal_basic() {
    // RHS = - sum_{j in N} A_{.j} * x_j
    std::vector<double> rhs(m_m, 0.0);
    for (int64_t j = 0; j < m_num_total; ++j) {
        if (m_var_in_basis[j] == -1) {
            double xj = m_x[j];
            if (std::abs(xj) > m_options.zero_tol) {
                if (j < m_n) {
                    const auto& cp = m_problem.A().csc_col_ptr();
                    const auto& ri = m_problem.A().csc_row_ind();
                    const auto& vals = m_problem.A().csc_values();
                    for (int64_t k = cp[j]; k < cp[j + 1]; ++k) {
                        rhs[ri[k]] -= vals[k] * xj;
                    }
                } else {
                    int64_t slack_row = j - m_n;
                    rhs[slack_row] -= (-1.0) * xj;
                }
            }
        }
    }

    auto x_B = m_lu.ftran(rhs);
    for (int64_t i = 0; i < m_m; ++i) {
        int64_t v = m_basic_vars[i];
        m_x[v] = x_B[i];
    }
}

void DualSimplexEngine::compute_dual_and_reduced_costs() {
    // Assemble c_B
    std::vector<double> c_B(m_m);
    for (int64_t i = 0; i < m_m; ++i) {
        c_B[i] = m_c[m_basic_vars[i]];
    }

    // Solve B^T * y = c_B
    m_y = m_lu.btran(c_B);

    // Compute reduced costs: s = c - A_bar^T * y
    // For structural j < n: s_j = c_j - A_{.j}^T * y
    // For slack n + i: s_{n+i} = 0 - (-1) * y_i = y_i
    std::vector<double> At_y = m_problem.A().mat_trans_vec(m_y);
    for (int64_t j = 0; j < m_n; ++j) {
        m_s[j] = m_c[j] - At_y[j];
    }
    for (int64_t i = 0; i < m_m; ++i) {
        m_s[m_n + i] = m_y[i];
    }

    // Clean basic reduced costs to 0
    for (int64_t i = 0; i < m_m; ++i) {
        m_s[m_basic_vars[i]] = 0.0;
    }
}

void DualSimplexEngine::init_dse_weights() {
    // Exact initialization: gamma_i = || B^{-T} e_i ||_2^2
    std::vector<double> ei(m_m, 0.0);
    for (int64_t i = 0; i < m_m; ++i) {
        ei[i] = 1.0;
        auto row_i = m_lu.btran(ei);
        ei[i] = 0.0;
        double norm_sq = 0.0;
        for (double v : row_i) norm_sq += v * v;
        m_dse_weights[i] = std::max(1e-4, norm_sq);
    }
}

int64_t DualSimplexEngine::select_leaving_row(double& max_viol, int& leave_dir) {
    int64_t best_p = -1;
    double best_merit = 0.0;
    max_viol = 0.0;
    leave_dir = 0;

    double tol = m_options.primal_feasibility_tol;

    for (int64_t i = 0; i < m_m; ++i) {
        int64_t v = m_basic_vars[i];
        double val = m_x[v];
        double viol = 0.0;
        int dir = 0;

        if (val < m_lb[v] - tol) {
            viol = m_lb[v] - val;
            dir = 1; // variable needs to increase
        } else if (val > m_ub[v] + tol) {
            viol = val - m_ub[v];
            dir = -1; // variable needs to decrease
        }

        if (viol > tol) {
            double merit = (viol * viol) / m_dse_weights[i];
            if (merit > best_merit) {
                best_merit = merit;
                best_p = i;
                max_viol = viol;
                leave_dir = dir;
            }
        }
    }

    return best_p;
}

int64_t DualSimplexEngine::select_entering_col_harris_bfrt(int64_t p, int leave_dir, double& pivot_val) {
    // Solve B^T * v = e_p to get tableau row
    std::vector<double> ep(m_m, 0.0);
    ep[p] = 1.0;
    auto alpha_p_row = m_lu.btran(ep);

    // Compute tableau coefficients alpha_{pj} for all non-basic j
    std::vector<double> alpha_p(m_num_total, 0.0);
    std::vector<double> At_v = m_problem.A().mat_trans_vec(alpha_p_row);
    for (int64_t j = 0; j < m_n; ++j) {
        if (m_var_in_basis[j] == -1) {
            alpha_p[j] = At_v[j];
        }
    }
    for (int64_t i = 0; i < m_m; ++i) {
        int64_t slack_var = m_n + i;
        if (m_var_in_basis[slack_var] == -1) {
            alpha_p[slack_var] = -alpha_p_row[i];
        }
    }

    // Harris Two-Pass Ratio Test
    double delta = m_options.dual_feasibility_tol;
    double min_ratio = std::numeric_limits<double>::infinity();

    // Pass 1: find minimum ratio
    for (int64_t j = 0; j < m_num_total; ++j) {
        if (m_var_in_basis[j] != -1) continue;

        double apj = alpha_p[j];
        double sj  = m_s[j];
        double eff_apj = apj * leave_dir;

        // If leave_dir == +1 (basic var increases):
        // For AtLower (sj >= 0), need eff_apj < -1e-12, ratio = sj / (-eff_apj)
        // For AtUpper (sj <= 0), need eff_apj > 1e-12, ratio = -sj / eff_apj
        // For Free: candidate in both directions
        if (m_status[j] == model::BasisStatus::AtLower) {
            if (eff_apj < -1e-10) {
                double ratio = std::max(0.0, sj) / (-eff_apj);
                min_ratio = std::min(min_ratio, ratio);
            }
        } else if (m_status[j] == model::BasisStatus::AtUpper) {
            if (eff_apj > 1e-10) {
                double ratio = std::max(0.0, -sj) / eff_apj;
                min_ratio = std::min(min_ratio, ratio);
            }
        } else if (m_status[j] == model::BasisStatus::Free) {
            if (std::abs(eff_apj) > 1e-10) {
                double ratio = std::abs(sj) / std::abs(eff_apj);
                min_ratio = std::min(min_ratio, ratio);
            }
        }
    }

    if (!std::isfinite(min_ratio)) {
        // Dual unbounded => Primal infeasible!
        return -1;
    }

    // Pass 2: choose pivot maximizing |alpha_{pj}| with ratio <= min_ratio + delta
    double max_alpha = 0.0;
    int64_t best_q = -1;
    double max_step = min_ratio + delta;

    for (int64_t j = 0; j < m_num_total; ++j) {
        if (m_var_in_basis[j] != -1) continue;

        double apj = alpha_p[j];
        double sj  = m_s[j];
        double eff_apj = apj * leave_dir;

        if (m_status[j] == model::BasisStatus::AtLower && eff_apj < -1e-10) {
            double ratio = std::max(0.0, sj) / (-eff_apj);
            if (ratio <= max_step && std::abs(apj) > max_alpha) {
                max_alpha = std::abs(apj);
                best_q = j;
            }
        } else if (m_status[j] == model::BasisStatus::AtUpper && eff_apj > 1e-10) {
            double ratio = std::max(0.0, -sj) / eff_apj;
            if (ratio <= max_step && std::abs(apj) > max_alpha) {
                max_alpha = std::abs(apj);
                best_q = j;
            }
        } else if (m_status[j] == model::BasisStatus::Free && std::abs(eff_apj) > 1e-10) {
            double ratio = std::abs(sj) / std::abs(eff_apj);
            if (ratio <= max_step && std::abs(apj) > max_alpha) {
                max_alpha = std::abs(apj);
                best_q = j;
            }
        }
    }

    if (best_q != -1) {
        pivot_val = alpha_p[best_q];
    }
    return best_q;
}

model::Solution DualSimplexEngine::solve() {
    model::Solution sol;
    sol.status = model::SolutionStatus::Unknown;

    int64_t max_iters = (m_options.iteration_limit > 0) ? m_options.iteration_limit : (10 * (m_m + m_n) + 10000);

    for (int64_t iter = 0; iter < max_iters; ++iter) {
        m_iteration_count++;

        // 1. Leaving row selection
        double max_viol = 0.0;
        int leave_dir = 0;
        int64_t p = select_leaving_row(max_viol, leave_dir);

        if (p == -1) {
            // Primal feasible and dual feasible => OPTIMAL!
            sol.status = model::SolutionStatus::Optimal;
            break;
        }

        // 2. Entering column selection via Harris BFRT
        double pivot_val = 0.0;
        int64_t q = select_entering_col_harris_bfrt(p, leave_dir, pivot_val);

        if (q == -1) {
            // Primal infeasible! Extract Farkas certificate ray
            sol.status = model::SolutionStatus::Infeasible;
            std::vector<double> ep(m_m, 0.0);
            ep[p] = (leave_dir > 0) ? 1.0 : -1.0;
            sol.ray = m_lu.btran(ep);
            break;
        }

        // 3. FTRAN on entering column
        auto col_q = get_column(q);
        auto ftran_aq = m_lu.ftran(col_q);

        // 4. Update basis
        int64_t leaving_var = m_basic_vars[p];
        m_var_in_basis[leaving_var] = -1;
        m_status[leaving_var] = (leave_dir > 0) ? model::BasisStatus::AtLower : model::BasisStatus::AtUpper;
        m_x[leaving_var] = (leave_dir > 0) ? m_lb[leaving_var] : m_ub[leaving_var];

        m_basic_vars[p] = q;
        m_var_in_basis[q] = p;
        m_status[q] = model::BasisStatus::Basic;

        // 5. Update LU factorization
        bool pfi_ok = m_lu.update_pfi(p, ftran_aq);
        if (!pfi_ok || m_lu.needs_refactorization(m_options.strategy.refactor_frequency)) {
            refactorize_basis();
            init_dse_weights();
        } else {
            // Update DSE weight for row p
            m_dse_weights[p] = std::max(1e-4, m_dse_weights[p] / (pivot_val * pivot_val));
        }

        // 6. Recalculate primal and dual coordinates
        compute_primal_basic();
        compute_dual_and_reduced_costs();
    }

    if (sol.status == model::SolutionStatus::Unknown) {
        sol.status = model::SolutionStatus::IterationLimit;
    }

    // Pack solution vectors
    sol.x.assign(m_n, 0.0);
    for (int64_t j = 0; j < m_n; ++j) {
        sol.x[j] = m_x[j];
    }

    sol.slack = m_problem.A().mat_vec(sol.x);

    double sense_mult = (m_problem.sense() == model::ObjectiveSense::Maximize) ? -1.0 : 1.0;
    sol.row_duals.assign(m_m, 0.0);
    for (int64_t i = 0; i < m_m; ++i) {
        sol.row_duals[i] = m_y[i] * sense_mult;
    }

    sol.reduced_costs.assign(m_n, 0.0);
    for (int64_t j = 0; j < m_n; ++j) {
        sol.reduced_costs[j] = m_s[j] * sense_mult;
    }

    sol.col_basis.assign(m_n, model::BasisStatus::AtLower);
    for (int64_t j = 0; j < m_n; ++j) {
        sol.col_basis[j] = m_status[j];
    }

    sol.row_basis.assign(m_m, model::BasisStatus::Basic);
    for (int64_t i = 0; i < m_m; ++i) {
        sol.row_basis[i] = m_status[m_n + i];
    }

    // Objective value
    double obj = m_problem.obj_offset();
    const auto& orig_c = m_problem.c();
    for (int64_t j = 0; j < m_n; ++j) {
        obj += orig_c[j] * sol.x[j];
    }
    sol.primal_objective = obj;
    sol.dual_bound = obj;
    sol.simplex_iterations = m_iteration_count;

    return sol;
}

} // namespace simplex
} // namespace sih
