#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include "sih/factorization/sparse_lu.hpp"
#include <vector>

namespace sih {
namespace bnb {

struct Cut {
    std::vector<model::Triplet> row_entries; // Row entries with row index = 0, col index = j, value = a_j
    double rhs{0.0};                         // a^T x >= rhs
};

class GomoryCutGenerator {
public:
    // Generate Gomory Mixed-Integer cuts from optimal basis and LP solution
    static std::vector<Cut> generate_cuts(const model::Problem& problem,
                                          const model::Solution& solution,
                                          const factorization::SparseLU& lu,
                                          const std::vector<int64_t>& basic_vars,
                                          double min_viol = 1e-4,
                                          double max_dynamism = 1e6);

    // Apply multiple rounds of Gomory cuts at the root node
    static model::Solution apply_root_cuts(model::Problem& problem,
                                           model::Solution root_sol,
                                           const model::Options& options);
};

} // namespace bnb
} // namespace sih
