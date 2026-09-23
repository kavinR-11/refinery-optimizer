#include "sih/checker/checker.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_checker..." << std::endl;

    // min x1 + 2 x2
    // s.t. x1 + x2 >= 3
    //      x1 - x2 <= 1
    //      x1, x2 >= 0
    sih::model::Problem p("checker_toy");
    p.resize(2, 2);
    p.set_c({1.0, 2.0});
    p.row_lower()[0] = 3.0;
    p.row_upper()[0] = sih::model::SIH_INFINITY;

    p.row_lower()[1] = -sih::model::SIH_INFINITY;
    p.row_upper()[1] = 1.0;

    std::vector<sih::model::Triplet> A_trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, -1.0}
    };
    p.set_A(sih::model::SparseMatrix::from_triplets(2, 2, A_trips));

    // Test 1: Optimal solution x = (3, 0)
    // Ax = (3, 3) -> row 0: 3 >= 3 (ok), row 1: 3 <= 1 (VIOLATED! 3 > 1)
    // Wait, at (2, 1): x1 + x2 = 3 (ok), x1 - x2 = 1 (ok).
    // obj = 1*2 + 2*1 = 4.
    sih::model::Solution sol_feas;
    sol_feas.x = {2.0, 1.0};
    sol_feas.primal_objective = 4.0;

    auto res1 = sih::checker::check_solution(p, sol_feas);
    std::cout << "  Feasible test: " << res1.summary << std::endl;
    ASSERT(res1.is_primal_feasible);
    ASSERT(res1.is_integrality_satisfied);
    ASSERT(res1.all_checks_passed);
    ASSERT(std::abs(res1.evaluated_objective - 4.0) < 1e-9);

    // Test 2: Infeasible solution x = (0, 0)
    // row 0: 0 < 3 (violation = 3.0)
    sih::model::Solution sol_infeas;
    sol_infeas.x = {0.0, 0.0};
    sol_infeas.primal_objective = 0.0;

    auto res2 = sih::checker::check_solution(p, sol_infeas);
    std::cout << "  Infeasible test: " << res2.summary << std::endl;
    ASSERT(!res2.is_primal_feasible);
    ASSERT(!res2.all_checks_passed);
    ASSERT(std::abs(res2.max_row_violation - 3.0) < 1e-9);

    // Test 3: Integrality violation test
    p.var_types()[0] = sih::model::VariableType::Integer;
    sih::model::Solution sol_frac;
    sol_frac.x = {2.5, 0.5}; // x1 + x2 = 3, x1 - x2 = 2 (violates row 1: 2 > 1, and x1=2.5 fractional)
    sol_frac.primal_objective = 3.5;

    auto res3 = sih::checker::check_solution(p, sol_frac);
    std::cout << "  Fractional test: " << res3.summary << std::endl;
    ASSERT(!res3.is_integrality_satisfied);
    ASSERT(std::abs(res3.max_integrality_violation - 0.5) < 1e-9);

    std::cout << "test_checker: PASS" << std::endl;
    return 0;
}
