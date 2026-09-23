#include "sih/presolve/presolver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_set>

namespace sih {
namespace presolve {

PresolveResult Presolver::presolve(const model::Problem& problem,
                                   int max_passes,
                                   double tol) {
    PresolveResult result;
    result.presolved_problem = problem;
    result.status = model::SolutionStatus::Unknown;

    int64_t orig_m = problem.num_rows();
    int64_t orig_n = problem.num_cols();

    // Mapping from current row/col index to original index
    std::vector<int64_t> row_map(orig_m);
    for (int64_t i = 0; i < orig_m; ++i) row_map[i] = i;
    std::vector<int64_t> col_map(orig_n);
    for (int64_t j = 0; j < orig_n; ++j) col_map[j] = j;

    bool changed = true;
    int pass = 0;

    while (changed && pass < max_passes) {
        changed = false;
        pass++;

        int64_t m = result.presolved_problem.num_rows();
        int64_t n = result.presolved_problem.num_cols();

        if (m == 0 || n == 0) {
            result.problem_empty = true;
            result.status = model::SolutionStatus::Optimal;
            return result;
        }

        const auto& A = result.presolved_problem.A();
        const auto& c = result.presolved_problem.c();
        auto& row_lower = result.presolved_problem.row_lower();
        auto& row_upper = result.presolved_problem.row_upper();
        auto& col_lower = result.presolved_problem.col_lower();
        auto& col_upper = result.presolved_problem.col_upper();

        std::vector<int64_t> row_degrees(m, 0);
        std::vector<int64_t> col_degrees(n, 0);

        const auto& col_ptr = A.csc_col_ptr();
        const auto& row_ind = A.csc_row_ind();
        const auto& values  = A.csc_values();

        if (col_ptr.size() >= static_cast<size_t>(n + 1)) {
            for (int64_t j = 0; j < n; ++j) {
                col_degrees[j] = col_ptr[j + 1] - col_ptr[j];
                for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                    row_degrees[row_ind[k]]++;
                }
            }
        }

        std::unordered_set<int64_t> rows_to_remove;
        std::unordered_set<int64_t> cols_to_remove;

        // 1. Empty rows detection
        for (int64_t i = 0; i < m; ++i) {
            if (row_degrees[i] == 0) {
                if (0.0 < row_lower[i] - tol || 0.0 > row_upper[i] + tol) {
                    // Primal Infeasible
                    result.status = model::SolutionStatus::Infeasible;
                    result.certificate_ray.assign(orig_m, 0.0);
                    result.certificate_ray[row_map[i]] = (0.0 < row_lower[i]) ? 1.0 : -1.0;
                    return result;
                }
                PresolveUndo undo;
                undo.type = ReductionType::EmptyRow;
                undo.row_idx = row_map[i];
                undo.old_row_lower = row_lower[i];
                undo.old_row_upper = row_upper[i];
                result.stack.push(std::move(undo));
                rows_to_remove.insert(i);
                changed = true;
            }
        }

        // 2. Empty columns detection
        for (int64_t j = 0; j < n; ++j) {
            if (col_degrees[j] == 0 && cols_to_remove.find(j) == cols_to_remove.end()) {
                double c_j = c[j];
                double fixed_val = 0.0;
                if (c_j > tol) {
                    if (col_lower[j] <= -model::SIH_INFINITY / 2.0) {
                        result.status = model::SolutionStatus::Unbounded;
                        result.certificate_ray.assign(orig_n, 0.0);
                        result.certificate_ray[col_map[j]] = -1.0;
                        return result;
                    }
                    fixed_val = col_lower[j];
                } else if (c_j < -tol) {
                    if (col_upper[j] >= model::SIH_INFINITY / 2.0) {
                        result.status = model::SolutionStatus::Unbounded;
                        result.certificate_ray.assign(orig_n, 0.0);
                        result.certificate_ray[col_map[j]] = 1.0;
                        return result;
                    }
                    fixed_val = col_upper[j];
                } else {
                    fixed_val = std::max(col_lower[j], std::min(0.0, col_upper[j]));
                }

                PresolveUndo undo;
                undo.type = ReductionType::EmptyCol;
                undo.col_idx = col_map[j];
                undo.old_col_lower = col_lower[j];
                undo.old_col_upper = col_upper[j];
                undo.coeff = fixed_val;
                undo.obj_coeff = c_j;
                result.stack.push(std::move(undo));

                result.presolved_problem.set_obj_offset(
                    result.presolved_problem.obj_offset() + c_j * fixed_val);
                cols_to_remove.insert(j);
                changed = true;
            }
        }

        // 3. Fixed columns (col_lower == col_upper)
        for (int64_t j = 0; j < n; ++j) {
            if (cols_to_remove.find(j) == cols_to_remove.end() &&
                std::abs(col_upper[j] - col_lower[j]) <= tol) {
                double v = col_lower[j];
                PresolveUndo undo;
                undo.type = ReductionType::FixedCol;
                undo.col_idx = col_map[j];
                undo.old_col_lower = col_lower[j];
                undo.old_col_upper = col_upper[j];
                undo.coeff = v;
                undo.obj_coeff = c[j];

                // Adjust row bounds
                for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                    int64_t i = row_ind[k];
                    double a_ij = values[k];
                    if (row_lower[i] > -model::SIH_INFINITY / 2.0) row_lower[i] -= a_ij * v;
                    if (row_upper[i] < model::SIH_INFINITY / 2.0)  row_upper[i] -= a_ij * v;
                    undo.sparse_indices.push_back(row_map[i]);
                    undo.sparse_values.push_back(a_ij);
                }

                result.presolved_problem.set_obj_offset(
                    result.presolved_problem.obj_offset() + c[j] * v);
                result.stack.push(std::move(undo));
                cols_to_remove.insert(j);
                changed = true;
            }
        }

        // 4. Row singletons (row with 1 nonzero)
        for (int64_t i = 0; i < m; ++i) {
            if (rows_to_remove.find(i) == rows_to_remove.end() && row_degrees[i] == 1) {
                // Find the single entry
                int64_t target_j = -1;
                double a_ij = 0.0;
                for (int64_t j = 0; j < n; ++j) {
                    if (cols_to_remove.find(j) != cols_to_remove.end()) continue;
                    for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                        if (row_ind[k] == i) {
                            target_j = j;
                            a_ij = values[k];
                            break;
                        }
                    }
                    if (target_j != -1) break;
                }

                if (target_j != -1 && std::abs(a_ij) > tol) {
                    double implied_lb = (a_ij > 0) ? (row_lower[i] / a_ij) : (row_upper[i] / a_ij);
                    double implied_ub = (a_ij > 0) ? (row_upper[i] / a_ij) : (row_lower[i] / a_ij);

                    double new_lb = std::max(col_lower[target_j], implied_lb);
                    double new_ub = std::min(col_upper[target_j], implied_ub);

                    if (new_lb > new_ub + tol) {
                        // Infeasible
                        result.status = model::SolutionStatus::Infeasible;
                        result.certificate_ray.assign(orig_m, 0.0);
                        result.certificate_ray[row_map[i]] = (a_ij > 0) ? 1.0 : -1.0;
                        return result;
                    }

                    PresolveUndo undo;
                    undo.type = ReductionType::RowSingleton;
                    undo.row_idx = row_map[i];
                    undo.col_idx = col_map[target_j];
                    undo.old_row_lower = row_lower[i];
                    undo.old_row_upper = row_upper[i];
                    undo.old_col_lower = col_lower[target_j];
                    undo.old_col_upper = col_upper[target_j];
                    undo.coeff = a_ij;
                    result.stack.push(std::move(undo));

                    col_lower[target_j] = new_lb;
                    col_upper[target_j] = new_ub;
                    rows_to_remove.insert(i);
                    changed = true;
                }
            }
        }

        // 5. Forcing and redundant rows
        for (int64_t i = 0; i < m; ++i) {
            if (rows_to_remove.find(i) != rows_to_remove.end()) continue;

            double L_i = 0.0;
            double U_i = 0.0;
            bool bounded_L = true;
            bool bounded_U = true;

            for (int64_t j = 0; j < n; ++j) {
                if (cols_to_remove.find(j) != cols_to_remove.end()) continue;
                for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                    if (row_ind[k] == i) {
                        double a_ij = values[k];
                        if (a_ij > 0) {
                            if (col_lower[j] <= -model::SIH_INFINITY / 2.0) bounded_L = false;
                            else L_i += a_ij * col_lower[j];
                            if (col_upper[j] >= model::SIH_INFINITY / 2.0) bounded_U = false;
                            else U_i += a_ij * col_upper[j];
                        } else {
                            if (col_upper[j] >= model::SIH_INFINITY / 2.0) bounded_L = false;
                            else L_i += a_ij * col_upper[j];
                            if (col_lower[j] <= -model::SIH_INFINITY / 2.0) bounded_U = false;
                            else U_i += a_ij * col_lower[j];
                        }
                        break;
                    }
                }
            }

            // Infeasible check
            if ((bounded_L && L_i > row_upper[i] + tol) ||
                (bounded_U && U_i < row_lower[i] - tol)) {
                result.status = model::SolutionStatus::Infeasible;
                result.certificate_ray.assign(orig_m, 0.0);
                result.certificate_ray[row_map[i]] = 1.0;
                return result;
            }

            // Redundant check: row is completely covered by variable bounds
            if (bounded_L && bounded_U &&
                L_i >= row_lower[i] - tol && U_i <= row_upper[i] + tol) {
                PresolveUndo undo;
                undo.type = ReductionType::RedundantRow;
                undo.row_idx = row_map[i];
                undo.old_row_lower = row_lower[i];
                undo.old_row_upper = row_upper[i];
                result.stack.push(std::move(undo));
                rows_to_remove.insert(i);
                changed = true;
            }
        }

        // Apply row & col eliminations to assemble smaller problem for next pass
        if (!rows_to_remove.empty() || !cols_to_remove.empty()) {
            int64_t new_m = m - static_cast<int64_t>(rows_to_remove.size());
            int64_t new_n = n - static_cast<int64_t>(cols_to_remove.size());

            std::vector<int64_t> new_row_map(new_m);
            std::vector<int64_t> new_col_map(new_n);
            std::vector<int64_t> old_to_new_row(m, -1);
            std::vector<int64_t> old_to_new_col(n, -1);

            int64_t r_count = 0;
            for (int64_t i = 0; i < m; ++i) {
                if (rows_to_remove.find(i) == rows_to_remove.end()) {
                    new_row_map[r_count] = row_map[i];
                    old_to_new_row[i] = r_count++;
                }
            }

            int64_t c_count = 0;
            for (int64_t j = 0; j < n; ++j) {
                if (cols_to_remove.find(j) == cols_to_remove.end()) {
                    new_col_map[c_count] = col_map[j];
                    old_to_new_col[j] = c_count++;
                }
            }

            row_map = std::move(new_row_map);
            col_map = std::move(new_col_map);

            model::Problem new_prob(result.presolved_problem.name());
            new_prob.set_sense(result.presolved_problem.sense());
            new_prob.set_obj_offset(result.presolved_problem.obj_offset());
            new_prob.resize(new_m, new_n);

            auto& new_c = new_prob.c();
            auto& new_cl = new_prob.col_lower();
            auto& new_cu = new_prob.col_upper();
            for (int64_t j = 0; j < n; ++j) {
                if (old_to_new_col[j] != -1) {
                    int64_t nj = old_to_new_col[j];
                    new_c[nj]  = c[j];
                    new_cl[nj] = col_lower[j];
                    new_cu[nj] = col_upper[j];
                }
            }

            auto& new_rl = new_prob.row_lower();
            auto& new_ru = new_prob.row_upper();
            for (int64_t i = 0; i < m; ++i) {
                if (old_to_new_row[i] != -1) {
                    int64_t ni = old_to_new_row[i];
                    new_rl[ni] = row_lower[i];
                    new_ru[ni] = row_upper[i];
                }
            }

            std::vector<model::Triplet> new_triplets;
            for (int64_t j = 0; j < n; ++j) {
                if (old_to_new_col[j] == -1) continue;
                int64_t nj = old_to_new_col[j];
                for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                    int64_t i = row_ind[k];
                    if (old_to_new_row[i] != -1) {
                        int64_t ni = old_to_new_row[i];
                        new_triplets.emplace_back(ni, nj, values[k]);
                    }
                }
            }
            new_prob.set_A(model::SparseMatrix::from_triplets(new_m, new_n, new_triplets, tol));
            result.presolved_problem = std::move(new_prob);
        }
    }

    if (result.presolved_problem.num_rows() == 0 || result.presolved_problem.num_cols() == 0) {
        result.problem_empty = true;
        result.status = model::SolutionStatus::Optimal;
    }

    return result;
}

model::Solution Presolver::postsolve(const model::Solution& presolved_sol,
                                     const model::Problem& orig_problem,
                                     const PresolveStack& stack,
                                     double tol) {
    model::Solution full_sol;
    full_sol.status = presolved_sol.status;
    int64_t orig_m = orig_problem.num_rows();
    int64_t orig_n = orig_problem.num_cols();

    full_sol.x.assign(orig_n, 0.0);
    full_sol.slack.assign(orig_m, 0.0);
    full_sol.row_duals.assign(orig_m, 0.0);
    full_sol.reduced_costs.assign(orig_n, 0.0);
    full_sol.col_basis.assign(orig_n, model::BasisStatus::AtLower);
    full_sol.row_basis.assign(orig_m, model::BasisStatus::Basic);

    // Initialize bounds-based status
    const auto& cl = orig_problem.col_lower();
    const auto& cu = orig_problem.col_upper();
    for (int64_t j = 0; j < orig_n; ++j) {
        if (cl[j] <= -model::SIH_INFINITY / 2.0 && cu[j] >= model::SIH_INFINITY / 2.0) {
            full_sol.col_basis[j] = model::BasisStatus::Free;
        } else {
            full_sol.col_basis[j] = model::BasisStatus::AtLower;
        }
    }

    // Copy forward any presolved values for active variables
    // In our single-pass or multi-pass model, unreduced variables are mapped back
    const auto& entries = stack.entries();

    // Set of eliminated rows and columns
    std::unordered_set<int64_t> elim_rows;
    std::unordered_set<int64_t> elim_cols;
    for (const auto& u : entries) {
        if (u.type == ReductionType::EmptyRow || u.type == ReductionType::RowSingleton ||
            u.type == ReductionType::RedundantRow || u.type == ReductionType::DuplicateRow) {
            elim_rows.insert(u.row_idx);
        }
        if (u.type == ReductionType::EmptyCol || u.type == ReductionType::FixedCol ||
            u.type == ReductionType::DominatedCol) {
            elim_cols.insert(u.col_idx);
        }
    }

    int64_t p_col = 0;
    for (int64_t j = 0; j < orig_n; ++j) {
        if (elim_cols.find(j) == elim_cols.end() && p_col < static_cast<int64_t>(presolved_sol.x.size())) {
            full_sol.x[j] = presolved_sol.x[p_col];
            full_sol.reduced_costs[j] = presolved_sol.reduced_costs[p_col];
            if (p_col < static_cast<int64_t>(presolved_sol.col_basis.size())) {
                full_sol.col_basis[j] = presolved_sol.col_basis[p_col];
            }
            p_col++;
        }
    }

    int64_t p_row = 0;
    for (int64_t i = 0; i < orig_m; ++i) {
        if (elim_rows.find(i) == elim_rows.end() && p_row < static_cast<int64_t>(presolved_sol.row_duals.size())) {
            full_sol.row_duals[i] = presolved_sol.row_duals[p_row];
            if (p_row < static_cast<int64_t>(presolved_sol.row_basis.size())) {
                full_sol.row_basis[i] = presolved_sol.row_basis[p_row];
            }
            p_row++;
        }
    }

    // Walk backwards through stack
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        const auto& u = *it;
        switch (u.type) {
            case ReductionType::EmptyRow: {
                full_sol.row_duals[u.row_idx] = 0.0;
                full_sol.row_basis[u.row_idx] = model::BasisStatus::Free;
                break;
            }
            case ReductionType::EmptyCol: {
                full_sol.x[u.col_idx] = u.coeff;
                full_sol.reduced_costs[u.col_idx] = u.obj_coeff;
                if (std::abs(full_sol.x[u.col_idx] - u.old_col_lower) <= tol) {
                    full_sol.col_basis[u.col_idx] = model::BasisStatus::AtLower;
                } else if (std::abs(full_sol.x[u.col_idx] - u.old_col_upper) <= tol) {
                    full_sol.col_basis[u.col_idx] = model::BasisStatus::AtUpper;
                } else {
                    full_sol.col_basis[u.col_idx] = model::BasisStatus::Free;
                }
                break;
            }
            case ReductionType::FixedCol: {
                full_sol.x[u.col_idx] = u.coeff;
                full_sol.col_basis[u.col_idx] = model::BasisStatus::AtLower;
                // Dual reduced cost: s_j = c_j - sum_i A_ij * y_i
                double a_dot_y = 0.0;
                for (size_t k = 0; k < u.sparse_indices.size(); ++k) {
                    a_dot_y += u.sparse_values[k] * full_sol.row_duals[u.sparse_indices[k]];
                }
                full_sol.reduced_costs[u.col_idx] = u.obj_coeff - a_dot_y;
                break;
            }
            case ReductionType::RowSingleton: {
                // Recover dual multiplier y_i to satisfy dual stationarity:
                // c_j - sum_{k != i} A_kj y_k - A_ij y_i - s_j = 0
                int64_t i = u.row_idx;
                int64_t j = u.col_idx;
                double a_ij = u.coeff;

                // Check if row singleton bound was active
                double xj = full_sol.x[j];
                bool active_lower = (std::abs(a_ij * xj - u.old_row_lower) <= tol);
                bool active_upper = (std::abs(a_ij * xj - u.old_row_upper) <= tol);

                if (active_lower || active_upper) {
                    double col_slack = full_sol.reduced_costs[j];
                    full_sol.row_duals[i] = col_slack / a_ij;
                    full_sol.reduced_costs[j] = 0.0;
                    full_sol.row_basis[i] = active_lower ? model::BasisStatus::AtLower : model::BasisStatus::AtUpper;
                } else {
                    full_sol.row_duals[i] = 0.0;
                    full_sol.row_basis[i] = model::BasisStatus::Free;
                }
                break;
            }
            case ReductionType::RedundantRow: {
                full_sol.row_duals[u.row_idx] = 0.0;
                full_sol.row_basis[u.row_idx] = model::BasisStatus::Free;
                break;
            }
            default:
                break;
        }
    }

    // Recompute row activities: Ax = A * x
    full_sol.slack = orig_problem.A().mat_vec(full_sol.x);

    // Compute primal objective
    const auto& orig_c = orig_problem.c();
    double obj = orig_problem.obj_offset();
    for (int64_t j = 0; j < orig_n; ++j) {
        obj += orig_c[j] * full_sol.x[j];
    }
    full_sol.primal_objective = obj;
    full_sol.dual_bound = obj;

    return full_sol;
}

} // namespace presolve
} // namespace sih
