#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include "sih/factorization/sparse_lu.hpp"
#include <vector>
#include <cstdint>

namespace sih {
namespace simplex {

class DualSimplexEngine {
public:
    DualSimplexEngine(const model::Problem& problem, const model::Options& options);

    void init_cold_start();
    void init_warm_start(const std::vector<model::BasisStatus>& col_basis,
                         const std::vector<model::BasisStatus>& row_basis);

    model::Solution solve();

    const factorization::SparseLU& lu() const noexcept { return m_lu; }
    const std::vector<int64_t>& basic_vars() const noexcept { return m_basic_vars; }

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

    std::vector<double> m_dse_weights;
    factorization::SparseLU m_lu;

    int64_t m_iteration_count{0};
    int64_t m_updates_since_refactor{0};

    // Cached per-iteration workspace vectors (avoid repeated allocation)
    std::vector<double> m_ep;           // unit vector e_p (m x 1)
    std::vector<double> m_alpha_p_row;  // BTRAN result: B^{-T} e_p
    std::vector<double> m_alpha_p;      // tableau row coefficients
    std::vector<double> m_ftran_aq;     // FTRAN result: B^{-1} a_q
    std::vector<double> m_col_q;        // entering column raw data

    void compute_primal_basic();
    void compute_dual_and_reduced_costs();
    void init_dse_weights();
    int64_t select_leaving_row(double& max_viol, int& leave_dir);
    int64_t select_entering_col_harris_bfrt(int64_t p, int leave_dir, double& pivot_val);
    void refactorize_basis();
    std::vector<double> get_column(int64_t j) const;

    // Phase 8: Incremental update methods
    void update_primal_incremental(int64_t q, int64_t p, int leave_dir);
    void update_dual_incremental(int64_t q, int64_t p, int leave_dir, double pivot_val);
    void update_dse_weights_exact(int64_t p, double pivot_val);
};

} // namespace simplex
} // namespace sih
