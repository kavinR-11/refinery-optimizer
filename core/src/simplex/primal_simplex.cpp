#include "sih/simplex/primal_simplex.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace sih {
namespace simplex {

PrimalSimplexEngine::PrimalSimplexEngine(const model::Problem& problem, const model::Options& options)
    : m_problem(problem), m_options(options) {
    m_m = problem.num_rows();
    m_n = problem.num_cols();
    m_num_total = m_n + m_m;

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
}

void PrimalSimplexEngine::init_cold_start() {
    for (int64_t i = 0; i < m_m; ++i) {
        int64_t slack_var = m_n + i;
        m_basic_vars[i] = slack_var;
        m_var_in_basis[slack_var] = i;
        m_status[slack_var] = model::BasisStatus::Basic;
    }

    for (int64_t j = 0; j < m_n; ++j) {
        m_var_in_basis[j] = -1;
        if (m_lb[j] > -model::SIH_INFINITY / 2.0) {
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
    compute_primal_basic();
    compute_dual_and_reduced_costs();
}

void PrimalSimplexEngine::init_warm_start(const std::vector<model::BasisStatus>& col_basis,
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

    compute_primal_basic();
    compute_dual_and_reduced_costs();
}

bool PrimalSimplexEngine::is_primal_feasible() const {
    double tol = m_options.primal_feasibility_tol;
    for (int64_t i = 0; i < m_m; ++i) {
        int64_t v = m_basic_vars[i];
        if (m_x[v] < m_lb[v] - tol || m_x[v] > m_ub[v] + tol) {
            return false;
        }
    }
    return true;
}

bool PrimalSimplexEngine::is_dual_feasible() const {
    double tol = m_options.dual_feasibility_tol;
    for (int64_t j = 0; j < m_num_total; ++j) {
        if (m_var_in_basis[j] == -1) {
            if (m_status[j] == model::BasisStatus::AtLower && m_s[j] < -tol) return false;
            if (m_status[j] == model::BasisStatus::AtUpper && m_s[j] > tol) return false;
            if (m_status[j] == model::BasisStatus::Free && std::abs(m_s[j]) > tol) return false;
        }
    }
    return true;
}

std::vector<double> PrimalSimplexEngine::get_column(int64_t j) const {
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

void PrimalSimplexEngine::refactorize_basis() {
    m_lu.factorize(m_m, m_basic_vars, m_problem.A(), 0.1, m_options.zero_tol);
}

void PrimalSimplexEngine::compute_primal_basic() {
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

void PrimalSimplexEngine::compute_dual_and_reduced_costs() {
    std::vector<double> c_B(m_m);
    for (int64_t i = 0; i < m_m; ++i) {
        c_B[i] = m_c[m_basic_vars[i]];
    }

    m_y = m_lu.btran(c_B);

    std::vector<double> At_y = m_problem.A().mat_trans_vec(m_y);
    for (int64_t j = 0; j < m_n; ++j) {
        m_s[j] = m_c[j] - At_y[j];
    }
    for (int64_t i = 0; i < m_m; ++i) {
        m_s[m_n + i] = m_y[i];
    }

    for (int64_t i = 0; i < m_m; ++i) {
        m_s[m_basic_vars[i]] = 0.0;
    }
}

model::Solution PrimalSimplexEngine::solve() {
    model::Solution sol;
    sol.status = model::SolutionStatus::Unknown;

    int64_t max_iters = (m_options.iteration_limit > 0) ? m_options.iteration_limit : (10 * (m_m + m_n) + 10000);
    double dual_tol = m_options.dual_feasibility_tol;

    for (int64_t iter = 0; iter < max_iters; ++iter) {
        m_iteration_count++;

        // 1. Pricing: find entering non-basic variable q
        int64_t best_q = -1;
        double max_viol = 0.0;
        int enter_dir = 0; // +1 if x_q increases, -1 if decreases

        for (int64_t j = 0; j < m_num_total; ++j) {
            if (m_var_in_basis[j] != -1) continue;

            double sj = m_s[j];
            double viol = 0.0;
            int dir = 0;

            if (m_status[j] == model::BasisStatus::AtLower) {
                if (sj < -dual_tol) {
                    viol = -sj;
                    dir = 1;
                }
            } else if (m_status[j] == model::BasisStatus::AtUpper) {
                if (sj > dual_tol) {
                    viol = sj;
                    dir = -1;
                }
            } else if (m_status[j] == model::BasisStatus::Free) {
                if (std::abs(sj) > dual_tol) {
                    viol = std::abs(sj);
                    dir = (sj < 0) ? 1 : -1;
                }
            }

            if (viol > max_viol) {
                max_viol = viol;
                best_q = j;
                enter_dir = dir;
            }
        }

        if (best_q == -1) {
            // Dual feasible and primal feasible => OPTIMAL!
            sol.status = model::SolutionStatus::Optimal;
            break;
        }

        // 2. FTRAN on entering column
        auto col_q = get_column(best_q);
        auto alpha_q = m_lu.ftran(col_q);

        // Direction of change for basic variables: delta_xB = -enter_dir * alpha_q
        std::vector<double> delta_xB(m_m);
        for (int64_t i = 0; i < m_m; ++i) {
            delta_xB[i] = -enter_dir * alpha_q[i];
        }

        // 3. Harris Ratio Test: find maximum step theta
        double max_step = std::numeric_limits<double>::infinity();
        // Variable best_q hitting opposite bound:
        if (enter_dir == 1 && m_ub[best_q] < model::SIH_INFINITY / 2.0) {
            max_step = std::min(max_step, m_ub[best_q] - m_x[best_q]);
        } else if (enter_dir == -1 && m_lb[best_q] > -model::SIH_INFINITY / 2.0) {
            max_step = std::min(max_step, m_x[best_q] - m_lb[best_q]);
        }

        int64_t best_p = -1; // leaving row
        double min_theta = max_step;

        for (int64_t i = 0; i < m_m; ++i) {
            int64_t v = m_basic_vars[i];
            double d = delta_xB[i];

            if (d < -1e-12) {
                // x_v is decreasing towards lower bound
                if (m_lb[v] > -model::SIH_INFINITY / 2.0) {
                    double theta = (m_lb[v] - m_x[v]) / d;
                    if (theta < min_theta) {
                        min_theta = theta;
                        best_p = i;
                    }
                }
            } else if (d > 1e-12) {
                // x_v is increasing towards upper bound
                if (m_ub[v] < model::SIH_INFINITY / 2.0) {
                    double theta = (m_ub[v] - m_x[v]) / d;
                    if (theta < min_theta) {
                        min_theta = theta;
                        best_p = i;
                    }
                }
            }
        }

        if (!std::isfinite(min_theta)) {
            // Primal unbounded! Construct unbounded ray
            sol.status = model::SolutionStatus::Unbounded;
            sol.ray.assign(m_n, 0.0);
            if (best_q < m_n) sol.ray[best_q] = enter_dir;
            for (int64_t i = 0; i < m_m; ++i) {
                int64_t v = m_basic_vars[i];
                if (v < m_n) sol.ray[v] = delta_xB[i];
            }
            break;
        }

        // Clamp theta >= 0
        min_theta = std::max(0.0, min_theta);

        if (best_p == -1) {
            // Variable best_q flipped to opposite bound without leaving basis
            m_x[best_q] += enter_dir * min_theta;
            m_status[best_q] = (enter_dir == 1) ? model::BasisStatus::AtUpper : model::BasisStatus::AtLower;
            for (int64_t i = 0; i < m_m; ++i) {
                int64_t v = m_basic_vars[i];
                m_x[v] += min_theta * delta_xB[i];
            }
            compute_dual_and_reduced_costs();
            continue;
        }

        // Step 4: Pivot execution
        // Leaving variable v leaves to bound
        int64_t leaving_var = m_basic_vars[best_p];
        m_var_in_basis[leaving_var] = -1;
        if (delta_xB[best_p] < 0.0) {
            m_status[leaving_var] = model::BasisStatus::AtLower;
            m_x[leaving_var] = m_lb[leaving_var];
        } else {
            m_status[leaving_var] = model::BasisStatus::AtUpper;
            m_x[leaving_var] = m_ub[leaving_var];
        }

        // Entering variable best_q enters basis
        m_basic_vars[best_p] = best_q;
        m_var_in_basis[best_q] = best_p;
        m_status[best_q] = model::BasisStatus::Basic;

        // Update LU
        bool pfi_ok = m_lu.update_pfi(best_p, alpha_q);
        if (!pfi_ok || m_lu.needs_refactorization(m_options.strategy.refactor_frequency)) {
            refactorize_basis();
        }

        compute_primal_basic();
        compute_dual_and_reduced_costs();
    }

    if (sol.status == model::SolutionStatus::Unknown) {
        sol.status = model::SolutionStatus::IterationLimit;
    }

    sol.x.assign(m_n, 0.0);
    for (int64_t j = 0; j < m_n; ++j) sol.x[j] = m_x[j];

    sol.slack = m_problem.A().mat_vec(sol.x);

    double sense_mult = (m_problem.sense() == model::ObjectiveSense::Maximize) ? -1.0 : 1.0;
    sol.row_duals.assign(m_m, 0.0);
    for (int64_t i = 0; i < m_m; ++i) sol.row_duals[i] = m_y[i] * sense_mult;

    sol.reduced_costs.assign(m_n, 0.0);
    for (int64_t j = 0; j < m_n; ++j) sol.reduced_costs[j] = m_s[j] * sense_mult;

    sol.col_basis.assign(m_n, model::BasisStatus::AtLower);
    for (int64_t j = 0; j < m_n; ++j) sol.col_basis[j] = m_status[j];

    sol.row_basis.assign(m_m, model::BasisStatus::Basic);
    for (int64_t i = 0; i < m_m; ++i) sol.row_basis[i] = m_status[m_n + i];

    double obj = m_problem.obj_offset();
    const auto& orig_c = m_problem.c();
    for (int64_t j = 0; j < m_n; ++j) obj += orig_c[j] * sol.x[j];

    sol.primal_objective = obj;
    sol.dual_bound = obj;
    sol.simplex_iterations = m_iteration_count;

    return sol;
}

} // namespace simplex
} // namespace sih
