#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include <vector>

namespace sih {
namespace bnb {

struct HeuristicResult {
    bool found_incumbent{false};
    double objective{model::SIH_INFINITY};
    std::vector<double> solution;
};

class PrimalHeuristics {
public:
    // Simple rounding heuristic
    static HeuristicResult simple_rounding(const model::Problem& problem,
                                           const model::Solution& lp_sol,
                                           double tol = 1e-5);

    // Fractional diving heuristic
    static HeuristicResult fractional_diving(const model::Problem& problem,
                                             const model::Solution& lp_sol,
                                             const model::Options& options,
                                             int64_t max_depth = 30);
};

} // namespace bnb
} // namespace sih
