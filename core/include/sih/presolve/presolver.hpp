#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/presolve/presolve_stack.hpp"

namespace sih {
namespace presolve {

struct PresolveResult {
    model::Problem presolved_problem;
    PresolveStack stack;
    model::SolutionStatus status{model::SolutionStatus::Unknown};
    std::vector<double> certificate_ray;
    bool problem_empty{false}; // true if entire problem was eliminated
};

class Presolver {
public:
    Presolver() = default;

    // Presolve Problem iteratively up to max_passes
    static PresolveResult presolve(const model::Problem& problem,
                                   int max_passes = 10,
                                   double tol = 1e-10);

    // Postsolve: unwinds stack in reverse order to recover original primal,
    // row activities, duals, reduced costs, and basis statuses
    static model::Solution postsolve(const model::Solution& presolved_sol,
                                     const model::Problem& orig_problem,
                                     const PresolveStack& stack,
                                     double tol = 1e-10);
};

} // namespace presolve
} // namespace sih
