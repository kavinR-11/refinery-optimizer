#include "sih/checker/checker.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace sih {
namespace checker {

CheckResult check_solution(const model::Problem& problem,
                           const model::Solution& solution,
                           double tol_primal,
                           double tol_dual,
                           double tol_int) {
    CheckResult res;
    const int64_t m = problem.num_rows();
    const int64_t n = problem.num_cols();

    // 0. Certificate verification for Infeasible or Unbounded status
    if (solution.status == model::SolutionStatus::Infeasible) {
        if (solution.ray.empty() || static_cast<int64_t>(solution.ray.size()) != m) {
            res.summary = "Infeasible certificate error: Farkas ray dimension mismatch or empty";
            res.all_checks_passed = false;
            return res;
        }

        double ray_norm = 0.0;
        for (double val : solution.ray) ray_norm = std::max(ray_norm, std::abs(val));
        if (ray_norm < 1e-12) {
            res.summary = "Infeasible certificate error: Farkas ray is zero";
            res.all_checks_passed = false;
            return res;
        }

        bool valid_cert = false;
        double best_gap = -1.0;

        for (double sign : {1.0, -1.0}) {
            std::vector<double> y(m);
            for (int64_t i = 0; i < m; ++i) y[i] = solution.ray[i] * sign;

            std::vector<double> w(n, 0.0);
            problem.A().mat_trans_vec(y.data(), w.data());

            bool inf_lhs = false;
            double lhs = 0.0;
            for (int64_t i = 0; i < m; ++i) {
                if (y[i] > 1e-9) {
                    if (problem.row_lower()[i] <= -model::SIH_INFINITY / 2.0) { inf_lhs = true; break; }
                    lhs += y[i] * problem.row_lower()[i];
                } else if (y[i] < -1e-9) {
                    if (problem.row_upper()[i] >= model::SIH_INFINITY / 2.0) { inf_lhs = true; break; }
                    lhs += y[i] * problem.row_upper()[i];
                }
            }
            if (inf_lhs) continue;

            bool inf_rhs = false;
            double rhs = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                if (w[j] > 1e-9) {
                    if (problem.col_upper()[j] >= model::SIH_INFINITY / 2.0) { inf_rhs = true; break; }
                    rhs += w[j] * problem.col_upper()[j];
                } else if (w[j] < -1e-9) {
                    if (problem.col_lower()[j] <= -model::SIH_INFINITY / 2.0) { inf_rhs = true; break; }
                    rhs += w[j] * problem.col_lower()[j];
                }
            }
            if (inf_rhs) continue;

            double gap = lhs - rhs;
            if (gap > 1e-6) {
                valid_cert = true;
                best_gap = std::max(best_gap, gap);
            }
        }

        res.is_certificate_valid = valid_cert;
        res.certificate_violation = valid_cert ? 0.0 : 1.0;
        res.all_checks_passed = valid_cert;
        res.summary = valid_cert ? "Checker Summary: PASSED | Farkas certificate ray verified"
                                 : "Checker Summary: FAILED | Farkas certificate ray invalid";
        return res;
    }

    if (solution.status == model::SolutionStatus::Unbounded) {
        if (solution.ray.empty() || static_cast<int64_t>(solution.ray.size()) != n) {
            res.summary = "Unbounded certificate error: ray dimension mismatch or empty";
            res.all_checks_passed = false;
            return res;
        }

        double ray_norm = 0.0;
        for (double val : solution.ray) ray_norm = std::max(ray_norm, std::abs(val));
        if (ray_norm < 1e-12) {
            res.summary = "Unbounded certificate error: ray is zero";
            res.all_checks_passed = false;
            return res;
        }

        double c_dot_d = 0.0;
        for (int64_t j = 0; j < n; ++j) c_dot_d += problem.c()[j] * solution.ray[j];
        bool obj_improves = (problem.sense() == model::ObjectiveSense::Maximize) ? (c_dot_d > 1e-9) : (c_dot_d < -1e-9);

        bool bounds_feasible = true;
        for (int64_t j = 0; j < n; ++j) {
            if (solution.ray[j] > 1e-9 && problem.col_upper()[j] < model::SIH_INFINITY / 2.0) {
                bounds_feasible = false;
                break;
            }
            if (solution.ray[j] < -1e-9 && problem.col_lower()[j] > -model::SIH_INFINITY / 2.0) {
                bounds_feasible = false;
                break;
            }
        }

        std::vector<double> Ad(m, 0.0);
        problem.A().mat_vec(solution.ray.data(), Ad.data());
        bool rows_feasible = true;
        for (int64_t i = 0; i < m; ++i) {
            if (Ad[i] > 1e-9 && problem.row_upper()[i] < model::SIH_INFINITY / 2.0) {
                rows_feasible = false;
                break;
            }
            if (Ad[i] < -1e-9 && problem.row_lower()[i] > -model::SIH_INFINITY / 2.0) {
                rows_feasible = false;
                break;
            }
        }

        bool valid_cert = obj_improves && bounds_feasible && rows_feasible;
        res.is_certificate_valid = valid_cert;
        res.certificate_violation = valid_cert ? 0.0 : 1.0;
        res.all_checks_passed = valid_cert;
        res.summary = valid_cert ? "Checker Summary: PASSED | Unbounded ray certificate verified"
                                 : "Checker Summary: FAILED | Unbounded ray certificate invalid";
        return res;
    }

    if (static_cast<int64_t>(solution.x.size()) != n) {
        res.summary = "Primal vector x dimension mismatch";
        return res;
    }

    // 1. Primal feasibility checks
    // Calculate row activities Ax
    std::vector<double> Ax(m, 0.0);
    problem.A().mat_vec(solution.x.data(), Ax.data());

    double max_row_viol = 0.0;
    for (int64_t i = 0; i < m; ++i) {
        double lb = problem.row_lower()[i];
        double ub = problem.row_upper()[i];
        double val = Ax[i];

        if (lb > -model::SIH_INFINITY / 2.0 && val < lb) {
            max_row_viol = std::max(max_row_viol, lb - val);
        }
        if (ub < model::SIH_INFINITY / 2.0 && val > ub) {
            max_row_viol = std::max(max_row_viol, val - ub);
        }
    }
    res.max_row_violation = max_row_viol;

    double max_col_viol = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        double lb = problem.col_lower()[j];
        double ub = problem.col_upper()[j];
        double val = solution.x[j];

        if (lb > -model::SIH_INFINITY / 2.0 && val < lb) {
            max_col_viol = std::max(max_col_viol, lb - val);
        }
        if (ub < model::SIH_INFINITY / 2.0 && val > ub) {
            max_col_viol = std::max(max_col_viol, val - ub);
        }
    }
    res.max_col_bound_violation = max_col_viol;
    res.max_primal_residual = std::max(max_row_viol, max_col_viol);
    res.is_primal_feasible = (res.max_primal_residual <= tol_primal);

    // 2. Integrality checks
    double max_int_viol = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        auto vt = problem.var_types()[j];
        if (vt == model::VariableType::Integer || vt == model::VariableType::Binary) {
            double val = solution.x[j];
            double dist = std::abs(val - std::round(val));
            max_int_viol = std::max(max_int_viol, dist);
        }
    }
    res.max_integrality_violation = max_int_viol;
    res.is_integrality_satisfied = (max_int_viol <= tol_int);

    // 3. Objective consistency check
    double obj_eval = problem.obj_offset();
    for (int64_t j = 0; j < n; ++j) {
        obj_eval += problem.c()[j] * solution.x[j];
    }
    if (problem.has_quadratic() && problem.Q().num_nonzeros() > 0) {
        std::vector<double> Qx(n, 0.0);
        problem.Q().mat_vec(solution.x.data(), Qx.data());
        double quad_term = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            quad_term += solution.x[j] * Qx[j];
        }
        obj_eval += 0.5 * quad_term;
    }
    res.evaluated_objective = obj_eval;
    res.objective_discrepancy = std::abs(obj_eval - solution.primal_objective);

    // 4. Dual feasibility & complementary slackness (if continuous LP/QP and dual vectors provided)
    bool has_duals = (static_cast<int64_t>(solution.row_duals.size()) == m);
    bool has_red_costs = (static_cast<int64_t>(solution.reduced_costs.size()) == n);

    if (!problem.is_mip() && has_duals && has_red_costs) {
        // Lagrangian stationarity: r_dual = c + Qx - A^T y - s
        std::vector<double> AT_y(n, 0.0);
        problem.A().mat_trans_vec(solution.row_duals.data(), AT_y.data());

        std::vector<double> grad = problem.c();
        if (problem.has_quadratic() && problem.Q().num_nonzeros() > 0) {
            std::vector<double> Qx(n, 0.0);
            problem.Q().mat_vec(solution.x.data(), Qx.data());
            for (int64_t j = 0; j < n; ++j) {
                grad[j] += Qx[j];
            }
        }

        double max_d_viol = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double stat = grad[j] - AT_y[j] - solution.reduced_costs[j];
            max_d_viol = std::max(max_d_viol, std::abs(stat));
        }
        res.max_dual_residual = max_d_viol;
        res.is_dual_feasible = (max_d_viol <= tol_dual);

        // Complementary slackness
        double max_cs = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            double y_i = solution.row_duals[i];
            double act = Ax[i];
            double lb = problem.row_lower()[i];
            double ub = problem.row_upper()[i];
            double dist_to_bound = std::min(std::abs(act - lb), std::abs(ub - act));
            max_cs = std::max(max_cs, dist_to_bound * std::abs(y_i));
        }
        res.max_complementarity_slack = max_cs;
        res.is_complementary = (max_cs <= tol_dual * 10.0);
    } else {
        res.is_dual_feasible = true;
        res.is_complementary = true;
    }

    // Overall check summary
    bool obj_match = (res.objective_discrepancy <= 1e-4 * (1.0 + std::abs(obj_eval)));
    res.all_checks_passed = res.is_primal_feasible && res.is_integrality_satisfied && obj_match;

    std::ostringstream ss;
    ss << "Checker Summary: "
       << (res.all_checks_passed ? "PASSED" : "FAILED")
       << " | Max Primal Viol: " << res.max_primal_residual
       << " | Max Int Viol: " << res.max_integrality_violation
       << " | Obj Evaluated: " << res.evaluated_objective
       << " | Obj Discrepancy: " << res.objective_discrepancy;
    res.summary = ss.str();

    return res;
}

} // namespace checker
} // namespace sih
