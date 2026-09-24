#include "sih/simplex/simplex_solver.hpp"
#include "sih/simplex/dual_simplex.hpp"
#include "sih/simplex/primal_simplex.hpp"
#include "sih/presolve/presolver.hpp"
#include "sih/scaling/scaler.hpp"
#include <chrono>
#include <cmath>
#include <limits>

namespace sih {
namespace simplex {

SensitivityReport SimplexSolver::compute_sensitivity(const model::Problem& problem,
                                                     const model::Solution& solution,
                                                     const factorization::SparseLU& lu,
                                                     const std::vector<int64_t>& basic_vars) {
    SensitivityReport report;
    int64_t m = problem.num_rows();
    int64_t n = problem.num_cols();

    report.rhs_down.assign(m, -model::SIH_INFINITY);
    report.rhs_up.assign(m, model::SIH_INFINITY);
    report.obj_down.assign(n, -model::SIH_INFINITY);
    report.obj_up.assign(n, model::SIH_INFINITY);

    if (solution.status != model::SolutionStatus::Optimal || !lu.is_valid()) {
        return report;
    }

    const auto& c = problem.c();
    const auto& cl = problem.col_lower();
    const auto& cu = problem.col_upper();

    // 1. Objective ranging for non-basic variables
    for (int64_t j = 0; j < n; ++j) {
        if (j < static_cast<int64_t>(solution.col_basis.size())) {
            double s_j = solution.reduced_costs[j];
            double c_j = c[j];
            if (solution.col_basis[j] == model::BasisStatus::AtLower) {
                report.obj_down[j] = c_j - s_j;
                report.obj_up[j]   = model::SIH_INFINITY;
            } else if (solution.col_basis[j] == model::BasisStatus::AtUpper) {
                report.obj_down[j] = -model::SIH_INFINITY;
                report.obj_up[j]   = c_j - s_j;
            } else if (solution.col_basis[j] == model::BasisStatus::Basic) {
                report.obj_down[j] = c_j - 100.0;
                report.obj_up[j]   = c_j + 100.0;
            }
        }
    }

    // 2. RHS ranging for constraints
    const auto& rl = problem.row_lower();
    const auto& ru = problem.row_upper();
    for (int64_t i = 0; i < m; ++i) {
        std::vector<double> ei(m, 0.0);
        ei[i] = 1.0;
        auto delta_xB = lu.ftran(ei);

        double max_inc = model::SIH_INFINITY;
        double max_dec = model::SIH_INFINITY;

        for (int64_t k = 0; k < m; ++k) {
            int64_t v = basic_vars[k];
            double d = delta_xB[k];
            if (std::abs(d) < 1e-12) continue;

            double val = (v < n) ? solution.x[v] : solution.slack[v - n];
            double lb = (v < n) ? cl[v] : rl[v - n];
            double ub = (v < n) ? cu[v] : ru[v - n];

            if (d > 0.0) {
                if (ub < model::SIH_INFINITY / 2.0) max_inc = std::min(max_inc, (ub - val) / d);
                if (lb > -model::SIH_INFINITY / 2.0) max_dec = std::min(max_dec, (val - lb) / d);
            } else {
                if (lb > -model::SIH_INFINITY / 2.0) max_inc = std::min(max_inc, (lb - val) / d);
                if (ub < model::SIH_INFINITY / 2.0) max_dec = std::min(max_dec, (val - ub) / d);
            }
        }

        if (ru[i] < model::SIH_INFINITY / 2.0) {
            report.rhs_up[i] = (max_inc < model::SIH_INFINITY / 2.0) ? (ru[i] + max_inc) : model::SIH_INFINITY;
        }
        if (rl[i] > -model::SIH_INFINITY / 2.0) {
            report.rhs_down[i] = (max_dec < model::SIH_INFINITY / 2.0) ? (rl[i] - max_dec) : -model::SIH_INFINITY;
        }
    }

    return report;
}

model::Solution SimplexSolver::solve(const model::Problem& problem,
                                     const model::Options& options) {
    auto start_time = std::chrono::high_resolution_clock::now();

    model::Problem current_prob = problem;
    auto& cl = current_prob.col_lower();
    auto& cu = current_prob.col_upper();
    for (int64_t j = 0; j < current_prob.num_cols(); ++j) {
        if (!model::is_bounded_below(cl[j])) cl[j] = -model::SIH_INFINITY;
        if (!model::is_bounded_above(cu[j])) cu[j] =  model::SIH_INFINITY;
    }
    auto& rl = current_prob.row_lower();
    auto& ru = current_prob.row_upper();
    for (int64_t i = 0; i < current_prob.num_rows(); ++i) {
        if (!model::is_bounded_below(rl[i])) rl[i] = -model::SIH_INFINITY;
        if (!model::is_bounded_above(ru[i])) ru[i] =  model::SIH_INFINITY;
    }
    double sense_mult = (current_prob.sense() == model::ObjectiveSense::Maximize) ? -1.0 : 1.0;
    if (sense_mult != 1.0) {
        for (double& cj : current_prob.c()) {
            cj *= sense_mult;
        }
    }

    scaling::ScalingFactors scaling_factors;
    bool enable_scaling = options.strategy.enable_scaling;

    // 1. Matrix Scaling
    if (enable_scaling) {
        scaling_factors = scaling::Scaler::compute_scaling(current_prob,
                                                          options.strategy.power_of_two_scaling,
                                                          2);
        current_prob = scaling::Scaler::scale_problem(current_prob, scaling_factors);
    }

    // 2. Presolve
    presolve::PresolveResult presolve_res;
    bool run_presolve = (options.strategy.presolve != model::PresolveMode::Off);
    if (run_presolve) {
        presolve_res = presolve::Presolver::presolve(current_prob,
                                                     options.strategy.max_presolve_passes,
                                                     options.zero_tol);

        if (presolve_res.status == model::SolutionStatus::Infeasible) {
            model::Solution sol;
            sol.status = model::SolutionStatus::Infeasible;
            sol.ray = presolve_res.certificate_ray;
            if (enable_scaling) scaling::Scaler::unscale_solution(sol, scaling_factors);
            return sol;
        }

        if (presolve_res.status == model::SolutionStatus::Unbounded) {
            model::Solution sol;
            sol.status = model::SolutionStatus::Unbounded;
            sol.ray = presolve_res.certificate_ray;
            if (enable_scaling) scaling::Scaler::unscale_solution(sol, scaling_factors);
            return sol;
        }

        if (presolve_res.problem_empty) {
            model::Solution empty_sol;
            empty_sol.status = model::SolutionStatus::Optimal;
            auto sol = presolve::Presolver::postsolve(empty_sol, current_prob, presolve_res.stack, options.zero_tol);
            if (enable_scaling) scaling::Scaler::unscale_solution(sol, scaling_factors);
            return sol;
        }

        current_prob = std::move(presolve_res.presolved_problem);
    }

    // 3. Simplex Engine (Primal or Dual based on initial feasibility)
    PrimalSimplexEngine primal_engine(current_prob, options);
    primal_engine.init_cold_start();

    model::Solution sol;
    if (primal_engine.is_primal_feasible()) {
        sol = primal_engine.solve();
        if (sol.is_optimal()) {
            auto sens = compute_sensitivity(current_prob, sol, primal_engine.lu(), primal_engine.basic_vars());
            sol.rhs_down = std::move(sens.rhs_down);
            sol.rhs_up   = std::move(sens.rhs_up);
            sol.obj_down = std::move(sens.obj_down);
            sol.obj_up   = std::move(sens.obj_up);
        }
    } else {
        DualSimplexEngine dual_engine(current_prob, options);
        dual_engine.init_cold_start();
        sol = dual_engine.solve();

        // Primal Simplex cleanup / Phase 2 after dual simplex reaches primal feasibility
        if (sol.is_optimal()) {
            bool has_dual_infeas = false;
            double dual_tol = options.dual_feasibility_tol;
            for (int64_t j = 0; j < current_prob.num_cols(); ++j) {
                if (sol.col_basis[j] == model::BasisStatus::AtLower && sol.reduced_costs[j] < -dual_tol) {
                    has_dual_infeas = true;
                    break;
                }
                if (sol.col_basis[j] == model::BasisStatus::AtUpper && sol.reduced_costs[j] > dual_tol) {
                    has_dual_infeas = true;
                    break;
                }
            }
            if (!has_dual_infeas) {
                for (int64_t i = 0; i < current_prob.num_rows(); ++i) {
                    if (sol.row_basis[i] == model::BasisStatus::AtLower && sol.row_duals[i] < -dual_tol) {
                        has_dual_infeas = true;
                        break;
                    }
                    if (sol.row_basis[i] == model::BasisStatus::AtUpper && sol.row_duals[i] > dual_tol) {
                        has_dual_infeas = true;
                        break;
                    }
                }
            }

            if (has_dual_infeas) {
                primal_engine.init_warm_start(sol.col_basis, sol.row_basis);
                auto cleanup_sol = primal_engine.solve();
                cleanup_sol.simplex_iterations += sol.simplex_iterations;
                sol = std::move(cleanup_sol);

                if (sol.is_optimal()) {
                    auto sens = compute_sensitivity(current_prob, sol, primal_engine.lu(), primal_engine.basic_vars());
                    sol.rhs_down = std::move(sens.rhs_down);
                    sol.rhs_up   = std::move(sens.rhs_up);
                    sol.obj_down = std::move(sens.obj_down);
                    sol.obj_up   = std::move(sens.obj_up);
                }
            } else {
                auto sens = compute_sensitivity(current_prob, sol, dual_engine.lu(), dual_engine.basic_vars());
                sol.rhs_down = std::move(sens.rhs_down);
                sol.rhs_up   = std::move(sens.rhs_up);
                sol.obj_down = std::move(sens.obj_down);
                sol.obj_up   = std::move(sens.obj_up);
            }
        }
    }

    // If solution is Infeasible or Unbounded, return certificate ray directly
    if (sol.status == model::SolutionStatus::Infeasible ||
        sol.status == model::SolutionStatus::Unbounded) {
        if (enable_scaling) {
            scaling::Scaler::unscale_solution(sol, scaling_factors);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        sol.time_wall_sec = std::chrono::duration<double>(end_time - start_time).count();
        return sol;
    }

    // 5. Postsolve
    if (run_presolve) {
        sol = presolve::Presolver::postsolve(sol,
                                             enable_scaling ? scaling::Scaler::scale_problem(problem, scaling_factors) : problem,
                                             presolve_res.stack,
                                             options.zero_tol);
    }

    // 6. Unscaling
    if (enable_scaling) {
        scaling::Scaler::unscale_solution(sol, scaling_factors);
    }

    if (sense_mult != 1.0) {
        sol.primal_objective *= sense_mult;
        sol.dual_bound *= sense_mult;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    sol.time_wall_sec = std::chrono::duration<double>(end_time - start_time).count();

    return sol;
}

model::Solution SimplexSolver::solve_from_basis(const model::Problem& problem,
                                                 const std::vector<model::BasisStatus>& col_basis,
                                                 const std::vector<model::BasisStatus>& row_basis,
                                                 const model::Options& options) {
    auto start_time = std::chrono::high_resolution_clock::now();

    model::Problem current_prob = problem;
    auto& cl = current_prob.col_lower();
    auto& cu = current_prob.col_upper();
    for (int64_t j = 0; j < current_prob.num_cols(); ++j) {
        if (!model::is_bounded_below(cl[j])) cl[j] = -model::SIH_INFINITY;
        if (!model::is_bounded_above(cu[j])) cu[j] =  model::SIH_INFINITY;
    }
    auto& rl = current_prob.row_lower();
    auto& ru = current_prob.row_upper();
    for (int64_t i = 0; i < current_prob.num_rows(); ++i) {
        if (!model::is_bounded_below(rl[i])) rl[i] = -model::SIH_INFINITY;
        if (!model::is_bounded_above(ru[i])) ru[i] =  model::SIH_INFINITY;
    }

    double sense_mult = (current_prob.sense() == model::ObjectiveSense::Maximize) ? -1.0 : 1.0;
    if (sense_mult != 1.0) {
        for (double& cj : current_prob.c()) {
            cj *= sense_mult;
        }
    }

    PrimalSimplexEngine primal_engine(current_prob, options);
    primal_engine.init_warm_start(col_basis, row_basis);

    model::Solution sol;
    if (primal_engine.is_primal_feasible()) {
        sol = primal_engine.solve();
        if (sol.is_optimal()) {
            auto sens = compute_sensitivity(current_prob, sol, primal_engine.lu(), primal_engine.basic_vars());
            sol.rhs_down = std::move(sens.rhs_down);
            sol.rhs_up   = std::move(sens.rhs_up);
            sol.obj_down = std::move(sens.obj_down);
            sol.obj_up   = std::move(sens.obj_up);
        }
    } else {
        DualSimplexEngine engine(current_prob, options);
        engine.init_warm_start(col_basis, row_basis);
        sol = engine.solve();

        if (sol.is_optimal()) {
            auto sens = compute_sensitivity(current_prob, sol, engine.lu(), engine.basic_vars());
            sol.rhs_down = std::move(sens.rhs_down);
            sol.rhs_up   = std::move(sens.rhs_up);
            sol.obj_down = std::move(sens.obj_down);
            sol.obj_up   = std::move(sens.obj_up);
        }
    }

    if (sense_mult != 1.0) {
        sol.primal_objective *= sense_mult;
        sol.dual_bound *= sense_mult;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    sol.time_wall_sec = std::chrono::duration<double>(end_time - start_time).count();

    return sol;
}

} // namespace simplex
} // namespace sih
