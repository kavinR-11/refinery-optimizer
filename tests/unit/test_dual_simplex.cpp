#include "sih/simplex/simplex_solver.hpp"
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
using namespace sih::simplex;

void test_simple_lp() {
    std::cout << "[Test] test_simple_lp running..." << std::endl;
    // min -1.0 * x0 - 2.0 * x1
    // s.t. x0 + x1 <= 4
    //      x0 <= 2
    //      x0, x1 >= 0
    Problem prob("simple_lp");
    prob.set_sense(ObjectiveSense::Minimize);
    prob.resize(2, 2);
    prob.set_c({-1.0, -2.0});

    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}
    };
    prob.set_A(SparseMatrix::from_triplets(2, 2, triplets));
    prob.row_lower() = {-SIH_INFINITY, -SIH_INFINITY};
    prob.row_upper() = {4.0, 2.0};
    prob.col_lower() = {0.0, 0.0};
    prob.col_upper() = {SIH_INFINITY, SIH_INFINITY};

    Options opts;
    auto sol = SimplexSolver::solve(prob, opts);

    TEST_ASSERT(sol.is_optimal());
    TEST_ASSERT(std::abs(sol.x[0] - 0.0) < 1e-6);
    TEST_ASSERT(std::abs(sol.x[1] - 4.0) < 1e-6);
    TEST_ASSERT(std::abs(sol.primal_objective - (-8.0)) < 1e-6);

    // Verify sensitivity ranges exist
    TEST_ASSERT(!sol.rhs_up.empty());
    TEST_ASSERT(!sol.obj_down.empty());

    std::cout << "[Test] test_simple_lp passed. Obj: " << sol.primal_objective
              << ", Iters: " << sol.simplex_iterations << std::endl;
}

void test_infeasible_lp() {
    std::cout << "[Test] test_infeasible_lp running..." << std::endl;
    // Infeasible LP:
    // x0 + x1 <= 1
    // x0 + x1 >= 3
    // x0, x1 >= 0
    Problem prob("infeasible_lp");
    prob.set_sense(ObjectiveSense::Minimize);
    prob.resize(2, 2);
    prob.set_c({1.0, 1.0});

    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    prob.set_A(SparseMatrix::from_triplets(2, 2, triplets));
    prob.row_lower() = {-SIH_INFINITY, 3.0};
    prob.row_upper() = {1.0, SIH_INFINITY};
    prob.col_lower() = {0.0, 0.0};
    prob.col_upper() = {SIH_INFINITY, SIH_INFINITY};

    Options opts;
    auto sol = SimplexSolver::solve(prob, opts);

    TEST_ASSERT(sol.status == SolutionStatus::Infeasible);
    TEST_ASSERT(!sol.ray.empty());
    std::cout << "[Test] test_infeasible_lp passed. Ray size: " << sol.ray.size() << std::endl;
}

void test_warm_start() {
    std::cout << "[Test] test_warm_start running..." << std::endl;
    // 1. Solve base problem:
    // min 2 x0 + 3 x1 + x2
    // s.t. x0 + 2 x1 + 3 x2 >= 6
    //      2 x0 + x1 + x2 >= 4
    //      x >= 0
    Problem prob("warm_start_base");
    prob.set_sense(ObjectiveSense::Minimize);
    prob.resize(2, 3);
    prob.set_c({2.0, 3.0, 1.0});

    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 3.0},
        {1, 0, 2.0}, {1, 1, 1.0}, {1, 2, 1.0}
    };
    prob.set_A(SparseMatrix::from_triplets(2, 3, triplets));
    prob.row_lower() = {6.0, 4.0};
    prob.row_upper() = {SIH_INFINITY, SIH_INFINITY};
    prob.col_lower() = {0.0, 0.0, 0.0};
    prob.col_upper() = {SIH_INFINITY, SIH_INFINITY, SIH_INFINITY};

    Options opts;
    opts.strategy.presolve = PresolveMode::Off;
    opts.strategy.enable_scaling = false;

    auto cold_sol = SimplexSolver::solve(prob, opts);
    TEST_ASSERT(cold_sol.is_optimal());
    int64_t cold_iters = cold_sol.simplex_iterations;

    // 2. Perturb RHS slightly: 6 -> 6.1
    Problem perturbed = prob;
    perturbed.row_lower()[0] = 6.1;

    // Warm-start resolve using base basis
    auto warm_sol = SimplexSolver::solve_from_basis(perturbed, cold_sol.col_basis, cold_sol.row_basis, opts);
    TEST_ASSERT(warm_sol.is_optimal());
    int64_t warm_iters = warm_sol.simplex_iterations;

    std::cout << "Cold start iterations on base: " << cold_iters
              << ", Warm start iterations on perturbed: " << warm_iters << std::endl;
    TEST_ASSERT(warm_iters <= cold_iters);

    std::cout << "[Test] test_warm_start passed." << std::endl;
}

int main() {
    std::cout << "=== Running Dual Simplex Unit Tests ===" << std::endl;
    test_simple_lp();
    test_infeasible_lp();
    test_warm_start();
    std::cout << "=== All Dual Simplex Unit Tests Passed! ===" << std::endl;
    return 0;
}
