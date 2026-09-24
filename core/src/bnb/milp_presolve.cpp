#include "sih/bnb/milp_presolve.hpp"
#include <cmath>
#include <algorithm>

namespace sih {
namespace bnb {

MilpPresolveResult MilpPresolver::presolve(const model::Problem& problem, double tol) {
    MilpPresolveResult result;
    result.problem = problem;

    int64_t n = result.problem.num_cols();
    int64_t m = result.problem.num_rows();

    auto& cl = result.problem.col_lower();
    auto& cu = result.problem.col_upper();
    const auto& vt = result.problem.var_types();

    // 1. Integrality bound tightening on all integer/binary variables
    for (int64_t j = 0; j < n; ++j) {
        if (vt[j] == model::VariableType::Binary) {
            double new_l = std::max(0.0, cl[j]);
            double new_u = std::min(1.0, cu[j]);
            if (new_l > cl[j]) { cl[j] = new_l; result.bounds_tightened_count++; }
            if (new_u < cu[j]) { cu[j] = new_u; result.bounds_tightened_count++; }
        } else if (vt[j] == model::VariableType::Integer) {
            if (model::is_bounded_below(cl[j])) {
                double ceil_l = std::ceil(cl[j] - tol);
                if (ceil_l > cl[j]) { cl[j] = ceil_l; result.bounds_tightened_count++; }
            }
            if (model::is_bounded_above(cu[j])) {
                double floor_u = std::floor(cu[j] + tol);
                if (floor_u < cu[j]) { cu[j] = floor_u; result.bounds_tightened_count++; }
            }
        }

        if (cl[j] > cu[j] + tol) {
            result.is_infeasible = true;
            return result;
        }

        if (std::abs(cu[j] - cl[j]) < tol && (vt[j] == model::VariableType::Binary || vt[j] == model::VariableType::Integer)) {
            result.fixed_integers_count++;
        }
    }

    // 2. Binary Probing
    const auto& rl = result.problem.row_lower();
    const auto& ru = result.problem.row_upper();
    const auto& A = result.problem.A();
    const auto& col_ptr = A.csc_col_ptr();
    const auto& row_ind = A.csc_row_ind();
    const auto& vals    = A.csc_values();

    for (int64_t j = 0; j < n; ++j) {
        if (vt[j] != model::VariableType::Binary) continue;
        if (std::abs(cu[j] - cl[j]) < tol) continue; // Already fixed

        // Probe x_j = 0
        bool probe0_infeasible = false;
        for (int64_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int64_t r = row_ind[p];
            double a_rj = vals[p];

            // Evaluate min/max row activity when x_j = 0
            double min_act = 0.0;
            double max_act = 0.0;
            bool bounded = true;

            const auto& row_a_ptr = A.csr_row_ptr();
            const auto& row_a_ind = A.csr_col_ind();
            const auto& row_a_val = A.csr_values();

            for (int64_t rp = row_a_ptr[r]; rp < row_a_ptr[r + 1]; ++rp) {
                int64_t c = row_a_ind[rp];
                double coeff = row_a_val[rp];
                double c_l = (c == j) ? 0.0 : cl[c];
                double c_u = (c == j) ? 0.0 : cu[c];

                if (coeff > 0.0) {
                    if (model::is_bounded_below(c_l)) min_act += coeff * c_l; else bounded = false;
                    if (model::is_bounded_above(c_u)) max_act += coeff * c_u; else bounded = false;
                } else {
                    if (model::is_bounded_above(c_u)) min_act += coeff * c_u; else bounded = false;
                    if (model::is_bounded_below(c_l)) max_act += coeff * c_l; else bounded = false;
                }
            }

            if (bounded) {
                if (model::is_bounded_below(rl[r]) && max_act < rl[r] - tol) probe0_infeasible = true;
                if (model::is_bounded_above(ru[r]) && min_act > ru[r] + tol) probe0_infeasible = true;
            }
            if (probe0_infeasible) break;
        }

        if (probe0_infeasible) {
            // Fixing x_j = 0 is impossible -> must fix x_j = 1
            cl[j] = 1.0;
            cu[j] = 1.0;
            result.fixed_integers_count++;
            continue;
        }

        // Probe x_j = 1
        bool probe1_infeasible = false;
        for (int64_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            int64_t r = row_ind[p];
            double a_rj = vals[p];

            double min_act = 0.0;
            double max_act = 0.0;
            bool bounded = true;

            const auto& row_a_ptr = A.csr_row_ptr();
            const auto& row_a_ind = A.csr_col_ind();
            const auto& row_a_val = A.csr_values();

            for (int64_t rp = row_a_ptr[r]; rp < row_a_ptr[r + 1]; ++rp) {
                int64_t c = row_a_ind[rp];
                double coeff = row_a_val[rp];
                double c_l = (c == j) ? 1.0 : cl[c];
                double c_u = (c == j) ? 1.0 : cu[c];

                if (coeff > 0.0) {
                    if (model::is_bounded_below(c_l)) min_act += coeff * c_l; else bounded = false;
                    if (model::is_bounded_above(c_u)) max_act += coeff * c_u; else bounded = false;
                } else {
                    if (model::is_bounded_above(c_u)) min_act += coeff * c_u; else bounded = false;
                    if (model::is_bounded_below(c_l)) max_act += coeff * c_l; else bounded = false;
                }
            }

            if (bounded) {
                if (model::is_bounded_below(rl[r]) && max_act < rl[r] - tol) probe1_infeasible = true;
                if (model::is_bounded_above(ru[r]) && min_act > ru[r] + tol) probe1_infeasible = true;
            }
            if (probe1_infeasible) break;
        }

        if (probe1_infeasible) {
            // Fixing x_j = 1 is impossible -> must fix x_j = 0
            cl[j] = 0.0;
            cu[j] = 0.0;
            result.fixed_integers_count++;
        }
    }

    return result;
}

} // namespace bnb
} // namespace sih
