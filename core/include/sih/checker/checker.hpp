#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"

namespace sih {
namespace checker {

struct CheckResult {
    bool is_primal_feasible{false};
    double max_row_violation{0.0};
    double max_col_bound_violation{0.0};
    double max_primal_residual{0.0};

    bool is_dual_feasible{false};
    double max_dual_residual{0.0};

    bool is_complementary{false};
    double max_complementarity_slack{0.0};

    bool is_integrality_satisfied{false};
    double max_integrality_violation{0.0};

    double evaluated_objective{0.0};
    double objective_discrepancy{0.0};

    // Certificate verification (for Infeasible / Unbounded solutions)
    bool is_certificate_valid{false};
    double certificate_violation{0.0};

    bool all_checks_passed{false};
    std::string summary;
};

// Independent verification of a candidate solution against problem definition
CheckResult check_solution(const model::Problem& problem,
                           const model::Solution& solution,
                           double tol_primal = 1e-6,
                           double tol_dual = 1e-6,
                           double tol_int = 1e-5);

} // namespace checker
} // namespace sih
