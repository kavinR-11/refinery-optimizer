#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"

namespace sih {
namespace ipm {

class IpmSolver {
public:
    // Solve a linear or convex quadratic program using the Mehrotra predictor-corrector
    // primal-dual interior point method.
    static model::Solution solve(const model::Problem& problem,
                                 const model::Options& options = model::Options());
};

} // namespace ipm
} // namespace sih
