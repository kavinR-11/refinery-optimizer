#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include <vector>
#include <memory>

namespace sih {
namespace bnb {

struct BnBNode {
    int64_t id{0};
    int64_t depth{0};
    double lp_bound{0.0};

    // Subproblem column bounds
    std::vector<double> col_lower;
    std::vector<double> col_upper;

    // Basis inherited from parent for dual simplex warm-start
    std::vector<model::BasisStatus> col_basis;
    std::vector<model::BasisStatus> row_basis;
};

class BranchAndBound {
public:
    static model::Solution solve(const model::Problem& problem,
                                 const model::Options& options = model::Options{});
};

} // namespace bnb
} // namespace sih
