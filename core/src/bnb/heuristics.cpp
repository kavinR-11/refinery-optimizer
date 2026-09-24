#include "sih/bnb/heuristics.hpp"
#include "sih/simplex/simplex_solver.hpp"
#include "sih/checker/checker.hpp"
#include <cmath>
#include <algorithm>

namespace sih {
namespace bnb {

HeuristicResult PrimalHeuristics::simple_rounding(const model::Problem& problem,
                                                  const model::Solution& lp_sol,
                                                  double tol) {
    HeuristicResult res;
    if (!lp_sol.is_optimal()) return res;

    int64_t n = problem.num_cols();
    const auto& vt = problem.var_types();
    const auto& cl = problem.col_lower();
    const auto& cu = problem.col_upper();

    std::vector<double> x_round = lp_sol.x;

    for (int64_t j = 0; j < n; ++j) {
        if (vt[j] == model::VariableType::Integer || vt[j] == model::VariableType::Binary) {
            double r_val = std::round(x_round[j]);
            if (r_val < cl[j] - tol || r_val > cu[j] + tol) {
                return res; // Out of bounds
            }
            x_round[j] = r_val;
        }
    }

    // Verify row feasibility: l <= Ax <= u
    auto Ax = problem.A().mat_vec(x_round);
    const auto& rl = problem.row_lower();
    const auto& ru = problem.row_upper();

    for (int64_t i = 0; i < problem.num_rows(); ++i) {
        if (model::is_bounded_below(rl[i]) && Ax[i] < rl[i] - tol) return res;
        if (model::is_bounded_above(ru[i]) && Ax[i] > ru[i] + tol) return res;
    }

    // Feasible! Compute objective
    double obj = problem.obj_offset();
    for (int64_t j = 0; j < n; ++j) {
        obj += problem.c()[j] * x_round[j];
    }

    res.found_incumbent = true;
    res.objective = obj;
    res.solution = std::move(x_round);
    return res;
}

HeuristicResult PrimalHeuristics::fractional_diving(const model::Problem& problem,
                                                    const model::Solution& lp_sol,
                                                    const model::Options& options,
                                                    int64_t max_depth) {
    HeuristicResult res;
    if (!lp_sol.is_optimal()) return res;

    model::Problem sub_prob = problem;
    auto curr_sol = lp_sol;

    int64_t n = sub_prob.num_cols();
    const auto& vt = sub_prob.var_types();
    auto& cl = sub_prob.col_lower();
    auto& cu = sub_prob.col_upper();

    for (int64_t depth = 0; depth < max_depth; ++depth) {
        // Find fractional integer variable closest to an integer
        int64_t best_j = -1;
        double min_frac_dist = 1.0;

        for (int64_t j = 0; j < n; ++j) {
            if (vt[j] == model::VariableType::Integer || vt[j] == model::VariableType::Binary) {
                double val = curr_sol.x[j];
                double dist = std::abs(val - std::round(val));
                if (dist > 1e-5 && dist < min_frac_dist) {
                    min_frac_dist = dist;
                    best_j = j;
                }
            }
        }

        if (best_j == -1) {
            // All integer variables are integer!
            res.found_incumbent = true;
            res.objective = curr_sol.primal_objective;
            res.solution = curr_sol.x;
            return res;
        }

        // Fix best_j to nearest integer
        double fix_val = std::round(curr_sol.x[best_j]);
        cl[best_j] = fix_val;
        cu[best_j] = fix_val;

        // Re-optimize with warm-start
        curr_sol = simplex::SimplexSolver::solve_from_basis(sub_prob, curr_sol.col_basis, curr_sol.row_basis, options);
        if (!curr_sol.is_optimal()) {
            break; // Infeasible dive
        }
    }

    return res;
}

} // namespace bnb
} // namespace sih
