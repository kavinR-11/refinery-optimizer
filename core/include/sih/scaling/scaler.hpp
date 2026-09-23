#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include <vector>
#include <cmath>

namespace sih {
namespace scaling {

struct ScalingFactors {
    std::vector<double> row_scale; // R: row multipliers
    std::vector<double> col_scale; // C: col multipliers
    bool is_scaled{false};
};

class Scaler {
public:
    Scaler() = default;

    // Scale a Problem in-place or return a new scaled Problem
    // If power_of_two is true, all factors R_i and C_j are quantized to 2^k,
    // ensuring zero mantissa truncation error.
    static ScalingFactors compute_scaling(const model::Problem& problem,
                                          bool power_of_two = true,
                                          int equilibration_passes = 1);

    static model::Problem scale_problem(const model::Problem& problem,
                                        const ScalingFactors& factors);

    // Unscale solution in-place from scaled coordinates to original coordinates
    static void unscale_solution(model::Solution& solution,
                                 const ScalingFactors& factors);

    // Power-of-two quantization helper
    static double quantize_pow2(double val);
};

} // namespace scaling
} // namespace sih
