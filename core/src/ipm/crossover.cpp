#include "sih/ipm/crossover.hpp"
#include "sih/simplex/simplex_solver.hpp"
#include "sih/factorization/sparse_lu.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace sih {
namespace ipm {

model::Solution Crossover::crossover(const model::Problem& problem,
                                     const model::Solution& ipm_sol,
                                     const model::Options& options) {
    if (!ipm_sol.is_optimal()) {
        return ipm_sol;
    }

    // For QP with quadratic terms, crossover to simplex BFS is not applicable
    if (problem.is_qp()) {
        return ipm_sol;
    }

    int64_t m = problem.num_rows();
    int64_t n = problem.num_cols();

    const auto& cl = problem.col_lower();
    const auto& cu = problem.col_upper();
    const auto& rl = problem.row_lower();
    const auto& ru = problem.row_upper();

    // 1. Initial basis: all slacks basic (B = -I), structural variables non-basic
    std::vector<model::BasisStatus> col_basis(n);
    std::vector<model::BasisStatus> row_basis(m, model::BasisStatus::Basic);
    std::vector<int64_t> basic_vars(m);

    for (int64_t j = 0; j < n; ++j) {
        double val = (j < static_cast<int64_t>(ipm_sol.x.size())) ? ipm_sol.x[j] : 0.0;
        double dist_l = model::is_bounded_below(cl[j]) ? std::abs(val - cl[j]) : 1e30;
        double dist_u = model::is_bounded_above(cu[j]) ? std::abs(val - cu[j]) : 1e30;
        col_basis[j] = (dist_l <= dist_u) ? model::BasisStatus::AtLower : model::BasisStatus::AtUpper;
    }

    for (int64_t i = 0; i < m; ++i) {
        basic_vars[i] = n + i;
    }

    // 2. Identify strictly interior structural variables and sort descending by distance to bound
    struct CandCol {
        int64_t j;
        double dist;
    };
    std::vector<CandCol> cand_cols;
    cand_cols.reserve(n);

    for (int64_t j = 0; j < n; ++j) {
        double val = (j < static_cast<int64_t>(ipm_sol.x.size())) ? ipm_sol.x[j] : 0.0;
        double dist_l = model::is_bounded_below(cl[j]) ? std::abs(val - cl[j]) : 1e30;
        double dist_u = model::is_bounded_above(cu[j]) ? std::abs(val - cu[j]) : 1e30;
        double dist = std::min(dist_l, dist_u);
        if (dist > 1e-5) {
            cand_cols.push_back({j, dist});
        }
    }

    std::sort(cand_cols.begin(), cand_cols.end(), [](const CandCol& a, const CandCol& b) {
        return a.dist > b.dist;
    });

    // 3. Incrementally pivot interior structural variables into basis, replacing slack variables
    factorization::SparseLU lu;
    auto lu_stat = lu.factorize(m, basic_vars, problem.A(), 0.1, options.zero_tol);

    if (lu_stat == factorization::FactorizationStatus::Success) {
        for (const auto& cand : cand_cols) {
            int64_t j = cand.j;

            // Form dense column of A
            std::vector<double> aj(m, 0.0);
            const auto& a_col_ptr = problem.A().csc_col_ptr();
            const auto& a_row_ind = problem.A().csc_row_ind();
            const auto& a_values  = problem.A().csc_values();
            for (int64_t p = a_col_ptr[j]; p < a_col_ptr[j + 1]; ++p) {
                aj[a_row_ind[p]] = a_values[p];
            }

            auto alpha = lu.ftran(aj);

            // Find best slack variable in basis to leave
            double best_pivot = 0.0;
            int64_t best_row = -1;
            for (int64_t i = 0; i < m; ++i) {
                if (basic_vars[i] >= n) { // still a slack
                    if (std::abs(alpha[i]) > best_pivot) {
                        best_pivot = std::abs(alpha[i]);
                        best_row = i;
                    }
                }
            }

            if (best_pivot > 1e-4 && best_row >= 0) {
                int64_t old_var = basic_vars[best_row];
                basic_vars[best_row] = j;

                factorization::SparseLU test_lu;
                auto test_stat = test_lu.factorize(m, basic_vars, problem.A(), 0.1, options.zero_tol);
                if (test_stat == factorization::FactorizationStatus::Success) {
                    lu = std::move(test_lu);
                    col_basis[j] = model::BasisStatus::Basic;

                    int64_t leaving_slack = old_var - n;
                    double s_val = (leaving_slack < static_cast<int64_t>(ipm_sol.slack.size())) ? ipm_sol.slack[leaving_slack] : 0.0;
                    double s_dl = model::is_bounded_below(rl[leaving_slack]) ? std::abs(s_val - rl[leaving_slack]) : 1e30;
                    double s_du = model::is_bounded_above(ru[leaving_slack]) ? std::abs(ru[leaving_slack] - s_val) : 1e30;
                    row_basis[leaving_slack] = (s_dl <= s_du) ? model::BasisStatus::AtLower : model::BasisStatus::AtUpper;
                } else {
                    // Revert swap
                    basic_vars[best_row] = old_var;
                }
            }
        }
    }

    // 4. Hand off clean, non-singular basis to Phase 1 Simplex cleanup pass
    auto simplex_sol = simplex::SimplexSolver::solve_from_basis(problem, col_basis, row_basis, options);

    double obj_diff = std::abs(simplex_sol.primal_objective - ipm_sol.primal_objective) /
                      (1.0 + std::abs(ipm_sol.primal_objective));

    if (simplex_sol.is_optimal() && obj_diff <= 1e-4) {
        simplex_sol.time_wall_sec += ipm_sol.time_wall_sec;
        return simplex_sol;
    }

    // Fallback to high-precision IPM solution, preserving valid basis
    auto fallback = ipm_sol;
    fallback.col_basis = col_basis;
    fallback.row_basis = row_basis;
    return fallback;
}

} // namespace ipm
} // namespace sih
