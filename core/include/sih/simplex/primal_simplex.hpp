#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include "sih/factorization/sparse_lu.hpp"
#include <vector>
#include <cstdint>

namespace sih {
namespace simplex {

class PrimalSimplexEngine {
public:
    PrimalSimplexEngine(const model::Problem& problem, const model::Options& options);

    void init_cold_start();
    void init_warm_start(const std::vector<model::BasisStatus>& col_basis,
                         const std::vector<model::BasisStatus>& row_basis);

    model::Solution solve();

    const factorization::SparseLU& lu() const noexcept { return m_lu; }
    const std::vector<int64_t>& basic_vars() const noexcept { return m_basic_vars; }
    bool is_primal_feasible() const;
    bool is_dual_feasible() const;

private:
    const model::Problem& m_problem;
    const model::Options& m_options;

    int64_t m_m{0};
    int64_t m_n{0};
    int64_t m_num_total{0}; // n + m

    std::vector<double> m_c;
    std::vector<double> m_lb;
    std::vector<double> m_ub;

    std::vector<int64_t> m_basic_vars;
    std::vector<int64_t> m_var_in_basis;
    std::vector<model::BasisStatus> m_status;

    std::vector<double> m_x;
    std::vector<double> m_s;
    std::vector<double> m_y;

    factorization::SparseLU m_lu;
    int64_t m_iteration_count{0};

    void compute_primal_basic();
    void compute_dual_and_reduced_costs();
    void refactorize_basis();
    std::vector<double> get_column(int64_t j) const;
};

} // namespace simplex
} // namespace sih
