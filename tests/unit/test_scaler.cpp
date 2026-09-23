#include "sih/scaling/scaler.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace sih::model;
using namespace sih::scaling;

void test_quantize_pow2() {
    std::cout << "[Test] test_quantize_pow2 running..." << std::endl;
    assert(Scaler::quantize_pow2(1.0) == 1.0);
    assert(Scaler::quantize_pow2(2.0) == 2.0);
    assert(Scaler::quantize_pow2(4.0) == 4.0);
    assert(Scaler::quantize_pow2(0.25) == 0.25);
    assert(Scaler::quantize_pow2(0.125) == 0.125);

    // Verify power-of-two property: mantissa in frexp is exactly 0.5
    double values[] = {0.003, 0.7, 1.414, 99.5, 12345.67};
    for (double v : values) {
        double p2 = Scaler::quantize_pow2(v);
        int exp = 0;
        double mantissa = std::frexp(p2, &exp);
        assert(mantissa == 0.5);
        (void)mantissa;
        (void)exp;
    }
    std::cout << "[Test] test_quantize_pow2 passed." << std::endl;
}

void test_scaling_transformation_and_unscaling() {
    std::cout << "[Test] test_scaling_transformation_and_unscaling running..." << std::endl;
    // Build an ill-conditioned LP:
    // min 1000 x1 + 0.01 x2
    // s.t. 0.001 x1 + 100 x2 >= 5
    //      10000 x1 + 0.1 x2 <= 2000
    //      x1, x2 >= 0
    Problem prob("ill_conditioned");
    prob.set_sense(ObjectiveSense::Minimize);
    prob.resize(2, 2);
    prob.set_c({1000.0, 0.01});

    std::vector<Triplet> triplets = {
        {0, 0, 0.001},
        {0, 1, 100.0},
        {1, 0, 10000.0},
        {1, 1, 0.1}
    };
    prob.set_A(SparseMatrix::from_triplets(2, 2, triplets));
    prob.row_lower() = {5.0, -SIH_INFINITY};
    prob.row_upper() = {SIH_INFINITY, 2000.0};
    prob.col_lower() = {0.0, 0.0};
    prob.col_upper() = {SIH_INFINITY, SIH_INFINITY};

    auto factors = Scaler::compute_scaling(prob, true, 2);
    assert(factors.row_scale.size() == 2);
    assert(factors.col_scale.size() == 2);

    // Verify all factors are powers of 2
    for (double r : factors.row_scale) {
        int exp = 0;
        double mantissa = std::frexp(r, &exp);
        assert(mantissa == 0.5);
        (void)mantissa;
        (void)exp;
    }
    for (double c : factors.col_scale) {
        int exp = 0;
        double mantissa = std::frexp(c, &exp);
        assert(mantissa == 0.5);
        (void)mantissa;
        (void)exp;
    }

    Problem scaled = Scaler::scale_problem(prob, factors);
    assert(scaled.num_rows() == 2);
    assert(scaled.num_cols() == 2);

    // Check dynamic range in original vs scaled
    double orig_max = 10000.0;
    double orig_min = 0.001;
    double orig_ratio = orig_max / orig_min; // 1e7

    const auto& sc_vals = scaled.A().csc_values();
    double sc_max = 0.0;
    double sc_min = 1e9;
    for (double v : sc_vals) {
        double av = std::abs(v);
        sc_max = std::max(sc_max, av);
        sc_min = std::min(sc_min, av);
    }
    double sc_ratio = sc_max / sc_min;
    std::cout << "Original coefficient ratio: " << orig_ratio
              << ", Scaled ratio: " << sc_ratio << std::endl;
    assert(sc_ratio < orig_ratio);

    // Test solution unscaling
    Solution sol;
    sol.status = SolutionStatus::Optimal;
    sol.x = {2.0, 3.0};
    sol.slack = {scaled.A().get(0, 0) * 2.0 + scaled.A().get(0, 1) * 3.0,
                 scaled.A().get(1, 0) * 2.0 + scaled.A().get(1, 1) * 3.0};
    sol.row_duals = {0.5, 0.25};
    sol.reduced_costs = {0.1, 0.05};
    sol.primal_objective = scaled.c()[0] * sol.x[0] + scaled.c()[1] * sol.x[1];

    Solution unscaled_sol = sol;
    Scaler::unscale_solution(unscaled_sol, factors);

    // Verify x * C
    assert(unscaled_sol.x[0] == sol.x[0] * factors.col_scale[0]);
    assert(unscaled_sol.x[1] == sol.x[1] * factors.col_scale[1]);

    // Verify row activities Ax in unscaled space match prob.A() * unscaled_sol.x
    std::vector<double> Ax_direct = prob.A().mat_vec(unscaled_sol.x);
    for (size_t i = 0; i < 2; ++i) {
        assert(std::abs(unscaled_sol.slack[i] - Ax_direct[i]) < 1e-12);
    }

    std::cout << "[Test] test_scaling_transformation_and_unscaling passed." << std::endl;
}

int main() {
    std::cout << "=== Running Scaler Unit Tests ===" << std::endl;
    test_quantize_pow2();
    test_scaling_transformation_and_unscaling();
    std::cout << "=== All Scaler Unit Tests Passed! ===" << std::endl;
    return 0;
}
