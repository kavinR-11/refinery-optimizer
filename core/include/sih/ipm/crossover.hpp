#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"

namespace sih {
namespace ipm {

class Crossover {
public:
    // Performs vertex crossover: converts an interior point solution into an exact
    // basic feasible solution (BFS) with a valid basis partition and simplex handoff.
    static model::Solution crossover(const model::Problem& problem,
                                     const model::Solution& ipm_sol,
                                     const model::Options& options);
};

} // namespace ipm
} // namespace sih
