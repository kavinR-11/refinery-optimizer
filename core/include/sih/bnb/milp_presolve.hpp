#pragma once

#include "sih/model/problem.hpp"
#include <vector>

namespace sih {
namespace bnb {

struct MilpPresolveResult {
    model::Problem problem;
    bool is_infeasible{false};
    int64_t fixed_integers_count{0};
    int64_t bounds_tightened_count{0};
};

class MilpPresolver {
public:
    static MilpPresolveResult presolve(const model::Problem& problem, double tol = 1e-8);
};

} // namespace bnb
} // namespace sih
