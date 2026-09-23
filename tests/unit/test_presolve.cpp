#include "sih/presolve/presolver.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " (" #cond ")" << std::endl; \
            std::abort(); \
        } \
    } while (0)

using namespace sih::model;
using namespace sih::presolve;

void test_empty_row_and_col() {
    std::cout << "[Test] test_empty_row_and_col running..." << std::endl;
    // 3 rows, 3 cols:
    // Row 0 is empty with 0 in bounds [-1, 2] -> should be dropped
    // Row 1 is normal
    // Col 2 is empty with c2 = 5 and bounds [1, 10] -> should be fixed to 1
    Problem prob("test_empty");
    prob.set_sense(ObjectiveSense::Minimize);
    prob.resize(2, 3);
    prob.set_c({1.0, 2.0, 5.0});

    std::vector<Triplet> triplets = {
        {1, 0, 1.0}, {1, 1, 1.0} // row 0 is empty
    };
    prob.set_A(SparseMatrix::from_triplets(2, 3, triplets));
    prob.row_lower() = {-1.0, 2.0};
    prob.row_upper() = {2.0, 10.0};
    prob.col_lower() = {0.0, 0.0, 1.0};
    prob.col_upper() = {5.0, 5.0, 10.0};

    auto res = Presolver::presolve(prob);
    TEST_ASSERT(res.status == SolutionStatus::Unknown);
    TEST_ASSERT(res.presolved_problem.num_rows() == 1); // Row 0 dropped
    TEST_ASSERT(res.presolved_problem.num_cols() == 2); // Col 2 dropped
    TEST_ASSERT(res.presolved_problem.obj_offset() == 5.0 * 1.0); // 5 * col2_min

    // Test postsolve
    Solution presolved_sol;
    presolved_sol.status = SolutionStatus::Optimal;
    presolved_sol.x = {1.0, 1.0};
    presolved_sol.row_duals = {1.0};
    presolved_sol.reduced_costs = {0.0, 1.0};

    auto full_sol = Presolver::postsolve(presolved_sol, prob, res.stack);
    TEST_ASSERT(static_cast<int64_t>(full_sol.x.size()) == 3);
    TEST_ASSERT(full_sol.x[0] == 1.0);
    TEST_ASSERT(full_sol.x[1] == 1.0);
    TEST_ASSERT(full_sol.x[2] == 1.0); // restored from empty col
    TEST_ASSERT(full_sol.row_duals[0] == 0.0); // empty row dual is 0
    TEST_ASSERT(full_sol.row_duals[1] == 1.0);

    std::cout << "[Test] test_empty_row_and_col passed." << std::endl;
}

void test_fixed_variable_and_row_singleton() {
    std::cout << "[Test] test_fixed_variable_and_row_singleton running..." << std::endl;
    // Problem with fixed variable x1 = 3 and row singleton: 2 * x0 <= 8
    Problem prob("test_fixed");
    prob.set_sense(ObjectiveSense::Minimize);
    prob.resize(2, 2);
    prob.set_c({2.0, 4.0});

    std::vector<Triplet> triplets = {
        {0, 0, 2.0},              // row 0 is singleton: 2 * x0 in [0, 8] => x0 in [0, 4]
        {1, 0, 1.0}, {1, 1, 3.0}  // row 1: x0 + 3 * x1 <= 15
    };
    prob.set_A(SparseMatrix::from_triplets(2, 2, triplets));
    prob.row_lower() = {0.0, -SIH_INFINITY};
    prob.row_upper() = {8.0, 15.0};
    prob.col_lower() = {0.0, 3.0};
    prob.col_upper() = {10.0, 3.0}; // x1 fixed to 3

    auto res = Presolver::presolve(prob, 1);
    // Fixed col x1 eliminated, row singleton row 0 eliminated after tightening x0 upper bound from 10 to 4
    TEST_ASSERT(res.presolved_problem.num_cols() == 1);
    TEST_ASSERT(res.presolved_problem.col_upper()[0] == 4.0);

    // Test postsolve reconstruction
    Solution presolved_sol;
    presolved_sol.status = SolutionStatus::Optimal;
    presolved_sol.x = {4.0};
    presolved_sol.row_duals = {0.0};
    presolved_sol.reduced_costs = {2.0};

    auto full_sol = Presolver::postsolve(presolved_sol, prob, res.stack);
    TEST_ASSERT(static_cast<int64_t>(full_sol.x.size()) == 2);
    TEST_ASSERT(full_sol.x[0] == 4.0);
    TEST_ASSERT(full_sol.x[1] == 3.0); // restored fixed var

    // Verify row activities match
    auto Ax = prob.A().mat_vec(full_sol.x);
    for (size_t i = 0; i < 2; ++i) {
        TEST_ASSERT(std::abs(full_sol.slack[i] - Ax[i]) < 1e-12);
    }

    std::cout << "[Test] test_fixed_variable_and_row_singleton passed." << std::endl;
}

void test_infeasible_detection() {
    std::cout << "[Test] test_infeasible_detection running..." << std::endl;
    // Empty row with 0 not in [5, 10]
    Problem prob("infeasible_empty_row");
    prob.resize(1, 1);
    prob.set_c({1.0});
    // A has 0 nonzeros
    prob.set_A(SparseMatrix(1, 1));
    prob.row_lower() = {5.0};
    prob.row_upper() = {10.0};
    prob.col_lower() = {0.0};
    prob.col_upper() = {1.0};

    auto res = Presolver::presolve(prob);
    TEST_ASSERT(res.status == SolutionStatus::Infeasible);
    TEST_ASSERT(!res.certificate_ray.empty());
    TEST_ASSERT(res.certificate_ray[0] != 0.0);
    std::cout << "[Test] test_infeasible_detection passed." << std::endl;
}

int main() {
    std::cout << "=== Running Presolver Unit Tests ===" << std::endl;
    test_empty_row_and_col();
    test_fixed_variable_and_row_singleton();
    test_infeasible_detection();
    std::cout << "=== All Presolver Unit Tests Passed! ===" << std::endl;
    return 0;
}
