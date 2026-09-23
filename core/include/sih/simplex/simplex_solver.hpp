#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include "sih/factorization/sparse_lu.hpp"
#include <vector>
#include <cstdint>

namespace sih {
namespace simplex {

struct SensitivityReport {
    std::vector<double> rhs_down;
    std::vector<double> rhs_up;
    std::vector<double> obj_down;
    std::vector<double> obj_up;
};

class SimplexSolver {
public:
    SimplexSolver() = default;

    // Master solve entry point: handles Presolve, Scaling, Dual/Primal Simplex,
    // Sensitivity analysis, and Postsolve unscaling
    static model::Solution solve(const model::Problem& problem,
                                 const model::Options& options = model::Options{});

    // Warm-start solve entry point: accepts existing basis statuses
    static model::Solution solve_from_basis(const model::Problem& problem,
                                            const std::vector<model::BasisStatus>& col_basis,
                                            const std::vector<model::BasisStatus>& row_basis,
                                            const model::Options& options = model::Options{});

    // Compute sensitivity ranges for an optimal basis
    static SensitivityReport compute_sensitivity(const model::Problem& problem,
                                                 const model::Solution& solution,
                                                 const factorization::SparseLU& lu,
                                                 const std::vector<int64_t>& basic_vars);
};

} // namespace simplex
} // namespace sih
