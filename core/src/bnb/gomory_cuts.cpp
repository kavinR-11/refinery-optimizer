#include "sih/bnb/gomory_cuts.hpp"
#include "sih/simplex/simplex_solver.hpp"
#include <cmath>
#include <algorithm>

namespace sih {
namespace bnb {

std::vector<Cut> GomoryCutGenerator::generate_cuts(const model::Problem& problem,
                                                   const model::Solution& solution,
                                                   const factorization::SparseLU& lu,
                                                   const std::vector<int64_t>& basic_vars,
                                                   double min_viol,
                                                   double max_dynamism) {
    std::vector<Cut> cuts;
    if (!solution.is_optimal()) return cuts;

    int64_t m = problem.num_rows();
    int64_t n = problem.num_cols();
    const auto& vt = problem.var_types();
    const auto& cl = problem.col_lower();
    const auto& cu = problem.col_upper();
    const auto& rl = problem.row_lower();
    const auto& ru = problem.row_upper();

    // Map which variables are in basis
    std::vector<bool> in_basis(n + m, false);
    for (int64_t v : basic_vars) {
        if (v >= 0 && v < n + m) in_basis[v] = true;
    }

    // Inspect each basic variable
    for (int64_t i = 0; i < m; ++i) {
        int64_t bv = basic_vars[i];
        if (bv < 0 || bv >= n) continue; // Only cut on structural integer variables

        if (vt[bv] != model::VariableType::Integer && vt[bv] != model::VariableType::Binary) {
            continue;
        }

        double val = solution.x[bv];
        double f0 = val - std::floor(val);

        // Fractionality filter: 0.05 <= f0 <= 0.95
        if (f0 < 0.05 || f0 > 0.95) continue;

        // BTRAN: pi = e_i^T B^{-1}
        std::vector<double> ei(m, 0.0);
        ei[i] = 1.0;
        auto pi = lu.btran(ei);

        // Compute tableau row coefficients for non-basic structural variables
        std::vector<double> a_bar(n, 0.0);
        const auto& col_ptr = problem.A().csc_col_ptr();
        const auto& row_ind = problem.A().csc_row_ind();
        const auto& vals    = problem.A().csc_values();

        for (int64_t j = 0; j < n; ++j) {
            if (in_basis[j]) continue;
            double dot = 0.0;
            for (int64_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
                dot += pi[row_ind[p]] * vals[p];
            }
            a_bar[j] = dot;
        }

        // Form GMIC cut in terms of original variables: sum c_j x_j >= rhs
        std::vector<model::Triplet> cut_triplets;
        double cut_rhs = f0;
        double f0_inv = f0 / (1.0 - f0);

        double max_c = 0.0;
        double min_c = 1e30;
        double cut_viol = 0.0;

        for (int64_t j = 0; j < n; ++j) {
            if (in_basis[j]) continue;
            double a_val = a_bar[j];
            if (std::abs(a_val) < 1e-12) continue;

            bool is_at_upper = (j < static_cast<int64_t>(solution.col_basis.size())) && 
                               (solution.col_basis[j] == model::BasisStatus::AtUpper);

            // Shift coefficient if variable is at upper bound: a_hat = -a_val
            double a_hat = is_at_upper ? -a_val : a_val;
            double bound = is_at_upper ? cu[j] : cl[j];

            double cut_coeff = 0.0;
            bool is_int = (vt[j] == model::VariableType::Integer || vt[j] == model::VariableType::Binary);

            if (is_int) {
                double fj = a_hat - std::floor(a_hat);
                if (fj <= f0) {
                    cut_coeff = fj;
                } else {
                    cut_coeff = f0_inv * (1.0 - fj);
                }
            } else {
                if (a_hat >= 0.0) {
                    cut_coeff = a_hat;
                } else {
                    cut_coeff = -f0_inv * a_hat;
                }
            }

            if (std::abs(cut_coeff) < 1e-12) continue;

            // In original coordinates:
            if (is_at_upper) {
                // hat{x} = u_j - x_j => cut_coeff * (u_j - x_j) >= ... => -cut_coeff * x_j >= ... - cut_coeff * u_j
                cut_triplets.emplace_back(0, j, -cut_coeff);
                cut_rhs -= cut_coeff * bound;
                cut_viol += -cut_coeff * solution.x[j];
            } else {
                // hat{x} = x_j - l_j => cut_coeff * (x_j - l_j) >= ... => cut_coeff * x_j >= ... + cut_coeff * l_j
                cut_triplets.emplace_back(0, j, cut_coeff);
                cut_rhs += cut_coeff * bound;
                cut_viol += cut_coeff * solution.x[j];
            }

            max_c = std::max(max_c, std::abs(cut_coeff));
            min_c = std::min(min_c, std::abs(cut_coeff));
        }

        double viol = cut_rhs - cut_viol;
        if (viol < min_viol) continue; // Not sufficiently violated
        if (min_c > 0.0 && (max_c / min_c > max_dynamism)) continue; // Dynamism filter

        if (!cut_triplets.empty()) {
            Cut c;
            c.row_entries = std::move(cut_triplets);
            c.rhs = cut_rhs;
            cuts.push_back(std::move(c));
        }

        if (cuts.size() >= 10) break; // Limit cuts per round
    }

    return cuts;
}

model::Solution GomoryCutGenerator::apply_root_cuts(model::Problem& problem,
                                                    model::Solution root_sol,
                                                    const model::Options& options) {
    int64_t rounds = options.strategy.cut_rounds;
    if (rounds <= 0 || !problem.is_mip()) return root_sol;

    for (int64_t r = 0; r < rounds; ++r) {
        if (!root_sol.is_optimal()) break;

        // Re-construct basis LU
        int64_t m = problem.num_rows();
        int64_t n = problem.num_cols();
        std::vector<int64_t> basic_vars;
        basic_vars.reserve(m);
        for (int64_t j = 0; j < n; ++j) {
            if (j < static_cast<int64_t>(root_sol.col_basis.size()) && root_sol.col_basis[j] == model::BasisStatus::Basic) {
                basic_vars.push_back(j);
            }
        }
        for (int64_t i = 0; i < m; ++i) {
            if (i < static_cast<int64_t>(root_sol.row_basis.size()) && root_sol.row_basis[i] == model::BasisStatus::Basic) {
                basic_vars.push_back(n + i);
            }
        }

        if (static_cast<int64_t>(basic_vars.size()) != m) break;

        factorization::SparseLU lu;
        auto lu_stat = lu.factorize(m, basic_vars, problem.A(), 0.1, options.zero_tol);
        if (lu_stat != factorization::FactorizationStatus::Success) break;

        auto cuts = generate_cuts(problem, root_sol, lu, basic_vars);
        if (cuts.empty()) break;

        // Add cuts to problem
        std::vector<model::Triplet> a_triplets;
        a_triplets.reserve(problem.A().num_nonzeros() + 100);
        const auto& orig_cp = problem.A().csc_col_ptr();
        const auto& orig_ri = problem.A().csc_row_ind();
        const auto& orig_vl = problem.A().csc_values();
        for (int64_t j = 0; j < problem.num_cols(); ++j) {
            for (int64_t p = orig_cp[j]; p < orig_cp[j + 1]; ++p) {
                a_triplets.emplace_back(orig_ri[p], j, orig_vl[p]);
            }
        }

        int64_t curr_rows = problem.num_rows();

        for (size_t c_idx = 0; c_idx < cuts.size(); ++c_idx) {
            int64_t new_row = curr_rows + c_idx;
            for (const auto& trip : cuts[c_idx].row_entries) {
                a_triplets.emplace_back(new_row, trip.col, trip.value);
            }
            problem.row_lower().push_back(cuts[c_idx].rhs);
            problem.row_upper().push_back(model::SIH_INFINITY);
        }

        int64_t updated_rows = curr_rows + static_cast<int64_t>(cuts.size());
        problem.resize(updated_rows, problem.num_cols());
        problem.set_A(model::SparseMatrix::from_triplets(updated_rows, problem.num_cols(), a_triplets));

        // Re-optimize with dual simplex
        auto new_sol = simplex::SimplexSolver::solve(problem, options);
        if (new_sol.is_optimal()) {
            root_sol = new_sol;
        } else {
            break;
        }
    }

    return root_sol;
}

} // namespace bnb
} // namespace sih
