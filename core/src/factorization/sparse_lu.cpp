#include "sih/factorization/sparse_lu.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <iostream>

namespace sih {
namespace factorization {

FactorizationStatus SparseLU::factorize(const model::SparseMatrix& B,
                                        double pivot_tol,
                                        double zero_tol) {
    int64_t m = B.num_rows();
    if (m != B.num_cols()) {
        m_status = FactorizationStatus::DimensionMismatch;
        return m_status;
    }
    m_dim = m;
    m_pivot_tol = pivot_tol;
    m_zero_tol = zero_tol;
    m_etas.clear();
    m_initial_nnz = B.num_nonzeros();

    if (m == 0) {
        m_status = FactorizationStatus::Success;
        return m_status;
    }

    // Dynamic sparse matrix representation for Markowitz factorization
    struct Element {
        int64_t col;
        double val;
    };
    std::vector<std::vector<Element>> rows(m);
    std::vector<int64_t> row_degrees(m, 0);
    std::vector<int64_t> col_degrees(m, 0);

    const auto& b_col_ptr = B.csc_col_ptr();
    const auto& b_row_ind = B.csc_row_ind();
    const auto& b_values = B.csc_values();

    for (int64_t j = 0; j < m; ++j) {
        for (int64_t k = b_col_ptr[j]; k < b_col_ptr[j + 1]; ++k) {
            int64_t i = b_row_ind[k];
            double v = b_values[k];
            if (std::abs(v) > zero_tol) {
                rows[i].push_back({j, v});
                col_degrees[j]++;
            }
        }
    }
    for (int64_t i = 0; i < m; ++i) {
        row_degrees[i] = static_cast<int64_t>(rows[i].size());
    }

    std::vector<bool> row_eliminated(m, false);
    std::vector<bool> col_eliminated(m, false);

    m_p_perm.assign(m, -1);
    m_inv_p_perm.assign(m, -1);
    m_q_perm.assign(m, -1);
    m_inv_q_perm.assign(m, -1);

    m_u_diag.assign(m, 0.0);

    // L triplets and U triplets (in permuted coordinates 0..m-1)
    std::vector<model::Triplet> l_triplets;
    std::vector<model::Triplet> u_triplets;

    // Fast dense row buffer for elimination
    std::vector<double> dense_row(m, 0.0);
    std::vector<int64_t> dense_nonzero_cols;
    dense_nonzero_cols.reserve(m);

    // Main elimination loop: find m pivots
    for (int64_t step = 0; step < m; ++step) {
        int64_t best_p = -1;
        int64_t best_q = -1;
        double best_val = 0.0;
        int64_t min_merit = std::numeric_limits<int64_t>::max();

        // 1. Singleton detection pass
        // (a) Row singletons
        for (int64_t i = 0; i < m; ++i) {
            if (!row_eliminated[i] && row_degrees[i] == 1) {
                for (const auto& elem : rows[i]) {
                    if (!col_eliminated[elem.col] && std::abs(elem.val) > zero_tol) {
                        best_p = i;
                        best_q = elem.col;
                        best_val = elem.val;
                        min_merit = 0;
                        break;
                    }
                }
                if (best_p != -1) break;
            }
        }

        // (b) Column singletons
        if (best_p == -1) {
            for (int64_t j = 0; j < m; ++j) {
                if (!col_eliminated[j] && col_degrees[j] == 1) {
                    for (int64_t i = 0; i < m; ++i) {
                        if (!row_eliminated[i]) {
                            for (const auto& elem : rows[i]) {
                                if (elem.col == j && std::abs(elem.val) > zero_tol) {
                                    best_p = i;
                                    best_q = j;
                                    best_val = elem.val;
                                    min_merit = 0;
                                    break;
                                }
                            }
                        }
                        if (best_p != -1) break;
                    }
                    if (best_p != -1) break;
                }
            }
        }

        // 2. Markowitz search with Threshold Partial Pivoting
        if (best_p == -1) {
            for (int64_t j = 0; j < m; ++j) {
                if (col_eliminated[j]) continue;

                // Find column maximum
                double max_in_col = 0.0;
                for (int64_t i = 0; i < m; ++i) {
                    if (!row_eliminated[i]) {
                        for (const auto& elem : rows[i]) {
                            if (elem.col == j) {
                                max_in_col = std::max(max_in_col, std::abs(elem.val));
                            }
                        }
                    }
                }

                if (max_in_col <= zero_tol) continue;

                // Check candidate rows in this column
                for (int64_t i = 0; i < m; ++i) {
                    if (row_eliminated[i]) continue;
                    for (const auto& elem : rows[i]) {
                        if (elem.col == j) {
                            double abs_val = std::abs(elem.val);
                            if (abs_val >= pivot_tol * max_in_col && abs_val > zero_tol) {
                                int64_t merit = (row_degrees[i] - 1) * (col_degrees[j] - 1);
                                if (merit < min_merit) {
                                    min_merit = merit;
                                    best_p = i;
                                    best_q = j;
                                    best_val = elem.val;
                                    if (min_merit == 0) break;
                                }
                            }
                        }
                    }
                    if (min_merit == 0) break;
                }
                if (min_merit == 0) break;
            }
        }

        if (best_p == -1 || std::abs(best_val) <= zero_tol) {
            m_status = FactorizationStatus::Singular;
            return m_status;
        }

        // Assign pivot to step
        m_p_perm[step] = best_p;
        m_inv_p_perm[best_p] = step;
        m_q_perm[step] = best_q;
        m_inv_q_perm[best_q] = step;

        row_eliminated[best_p] = true;
        col_eliminated[best_q] = true;
        m_u_diag[step] = best_val;

        // Store pivot row into U
        for (const auto& elem : rows[best_p]) {
            if (!col_eliminated[elem.col] && elem.col != best_q && std::abs(elem.val) > zero_tol) {
                // Element is part of U(step, col)
                // Note: col will be mapped to step_col when that column is eliminated
            }
        }

        // Elimination: for every other active row i that has an entry in column best_q
        for (int64_t i = 0; i < m; ++i) {
            if (row_eliminated[i]) continue;

            // Check if row i has an entry in column best_q
            double a_iq = 0.0;
            for (const auto& elem : rows[i]) {
                if (elem.col == best_q) {
                    a_iq = elem.val;
                    break;
                }
            }

            if (std::abs(a_iq) <= zero_tol) continue;

            double mult = a_iq / best_val;
            // Store L entry: L(perm(i), step) = mult
            // We will resolve row perm after all rows are eliminated, or store temporary (i, step, mult)
            l_triplets.emplace_back(i, step, mult);

            // Subtract mult * (row best_p) from row i
            // Expand row i into dense buffer
            for (const auto& elem : rows[i]) {
                dense_row[elem.col] = elem.val;
                dense_nonzero_cols.push_back(elem.col);
            }

            // Subtract
            for (const auto& elem : rows[best_p]) {
                if (dense_row[elem.col] == 0.0) {
                    dense_nonzero_cols.push_back(elem.col);
                }
                dense_row[elem.col] -= mult * elem.val;
            }

            // Repack row i
            rows[i].clear();
            dense_row[best_q] = 0.0; // Column best_q is eliminated
            for (int64_t c : dense_nonzero_cols) {
                double val = dense_row[c];
                dense_row[c] = 0.0;
                if (!col_eliminated[c] && std::abs(val) > zero_tol) {
                    rows[i].push_back({c, val});
                }
            }
            dense_nonzero_cols.clear();
            row_degrees[i] = static_cast<int64_t>(rows[i].size());
        }

        // Recompute column degrees for remaining active columns
        for (int64_t j = 0; j < m; ++j) {
            if (!col_eliminated[j]) {
                int64_t deg = 0;
                for (int64_t i = 0; i < m; ++i) {
                    if (!row_eliminated[i]) {
                        for (const auto& elem : rows[i]) {
                            if (elem.col == j) { deg++; break; }
                        }
                    }
                }
                col_degrees[j] = deg;
            }
        }
    }

    // Now all rows and columns are eliminated and permuted into steps 0..m-1.
    // Build U matrix elements from original/partially eliminated pivot rows
    // Re-read original matrix B to assemble U accurately:
    // P B Q = L U => U = L^{-1} P B Q
    // Or equivalently, pivot row of each step gives U(step, m_inv_q_perm[col]) for step_col > step
    // Build L SparseMatrix from l_triplets with row index mapped via m_inv_p_perm:
    std::vector<model::Triplet> l_perm_triplets;
    l_perm_triplets.reserve(l_triplets.size());
    for (const auto& tr : l_triplets) {
        int64_t perm_i = m_inv_p_perm[tr.row];
        if (perm_i > tr.col) { // strictly lower triangular
            l_perm_triplets.emplace_back(perm_i, tr.col, tr.value);
        }
    }

    auto L_mat = model::SparseMatrix::from_triplets(m, m, l_perm_triplets, zero_tol);
    m_l_col_ptr = L_mat.csc_col_ptr();
    m_l_row_ind = L_mat.csc_row_ind();
    m_l_values  = L_mat.csc_values();
    m_l_row_ptr = L_mat.csr_row_ptr();
    m_l_col_ind = L_mat.csr_col_ind();
    m_l_row_values = L_mat.csr_values();
    m_nnz_L = L_mat.num_nonzeros();

    // Compute U = L^{-1} * (P * B * Q)
    // For each column step_j in 0..m-1:
    // original col is m_q_perm[step_j]
    // Vector b = P * B.col(orig_col)
    // Forward solve L * u_col = b
    // Since U is upper triangular, u_col[k] for k > step_j must be 0 (within tolerance)
    std::vector<model::Triplet> u_perm_triplets;
    u_perm_triplets.reserve(m_initial_nnz);

    std::vector<double> rhs(m, 0.0);
    std::vector<double> u_col(m, 0.0);

    double min_abs_diag = std::numeric_limits<double>::infinity();
    double max_abs_diag = 0.0;

    for (int64_t step_j = 0; step_j < m; ++step_j) {
        int64_t orig_j = m_q_perm[step_j];
        for (int64_t i = 0; i < m; ++i) rhs[i] = 0.0;

        for (int64_t k = b_col_ptr[orig_j]; k < b_col_ptr[orig_j + 1]; ++k) {
            int64_t orig_i = b_row_ind[k];
            int64_t perm_i = m_inv_p_perm[orig_i];
            rhs[perm_i] = b_values[k];
        }

        // Forward solve L * u_col = rhs
        // L is unit lower triangular
        for (int64_t i = 0; i < m; ++i) {
            double s = rhs[i];
            for (int64_t idx = m_l_row_ptr[i]; idx < m_l_row_ptr[i + 1]; ++idx) {
                int64_t k = m_l_col_ind[idx];
                s -= m_l_row_values[idx] * u_col[k];
            }
            u_col[i] = s;
        }

        // Store diagonal and strictly upper elements of U
        m_u_diag[step_j] = u_col[step_j];
        double abs_d = std::abs(u_col[step_j]);
        min_abs_diag = std::min(min_abs_diag, abs_d);
        max_abs_diag = std::max(max_abs_diag, abs_d);

        for (int64_t i = 0; i < step_j; ++i) {
            if (std::abs(u_col[i]) > zero_tol) {
                u_perm_triplets.emplace_back(i, step_j, u_col[i]);
            }
        }
    }

    auto U_mat = model::SparseMatrix::from_triplets(m, m, u_perm_triplets, zero_tol);
    m_u_col_ptr = U_mat.csc_col_ptr();
    m_u_row_ind = U_mat.csc_row_ind();
    m_u_values  = U_mat.csc_values();
    m_u_row_ptr = U_mat.csr_row_ptr();
    m_u_col_ind = U_mat.csr_col_ind();
    m_u_row_values = U_mat.csr_values();
    m_nnz_U = U_mat.num_nonzeros() + m; // including diagonal

    m_cond_est = (min_abs_diag > 0.0) ? (max_abs_diag / min_abs_diag) : 1e16;
    m_status = FactorizationStatus::Success;
    return m_status;
}

FactorizationStatus SparseLU::factorize(int64_t m,
                                        const std::vector<int64_t>& basic_vars,
                                        const model::SparseMatrix& A,
                                        double pivot_tol,
                                        double zero_tol) {
    if (static_cast<int64_t>(basic_vars.size()) != m) {
        m_status = FactorizationStatus::DimensionMismatch;
        return m_status;
    }

    // Assemble square basis matrix B from basic variables
    // If var < n: column of A
    // If var >= n: slack variable (slack i has -1 at row i)
    int64_t n = A.num_cols();
    std::vector<model::Triplet> b_triplets;
    b_triplets.reserve(m * 10);

    const auto& a_col_ptr = A.csc_col_ptr();
    const auto& a_row_ind = A.csc_row_ind();
    const auto& a_values  = A.csc_values();

    for (int64_t j = 0; j < m; ++j) {
        int64_t var = basic_vars[j];
        if (var < n) {
            for (int64_t k = a_col_ptr[var]; k < a_col_ptr[var + 1]; ++k) {
                b_triplets.emplace_back(a_row_ind[k], j, a_values[k]);
            }
        } else {
            // Slack variable: maps to row (var - n) with entry -1.0
            int64_t slack_row = var - n;
            if (slack_row >= 0 && slack_row < m) {
                b_triplets.emplace_back(slack_row, j, -1.0);
            }
        }
    }

    auto B = model::SparseMatrix::from_triplets(m, m, b_triplets, zero_tol);
    return factorize(B, pivot_tol, zero_tol);
}

void SparseLU::ftran(const double* b, double* x) const {
    if (m_dim == 0) return;
    int64_t m = m_dim;

    // 1. Permute RHS: z = P * b
    std::vector<double> z(m);
    for (int64_t i = 0; i < m; ++i) {
        z[i] = b[m_p_perm[i]];
    }

    // 2. Forward solve L * y = z (L is unit lower triangular)
    for (int64_t i = 0; i < m; ++i) {
        double val = z[i];
        for (int64_t idx = m_l_row_ptr[i]; idx < m_l_row_ptr[i + 1]; ++idx) {
            val -= m_l_row_values[idx] * z[m_l_col_ind[idx]];
        }
        z[i] = val;
    }

    // 3. Backward solve U * w = z (U has diagonal m_u_diag)
    std::vector<double> w(m);
    for (int64_t i = m - 1; i >= 0; --i) {
        double val = z[i];
        for (int64_t idx = m_u_row_ptr[i]; idx < m_u_row_ptr[i + 1]; ++idx) {
            val -= m_u_row_values[idx] * w[m_u_col_ind[idx]];
        }
        w[i] = val / m_u_diag[i];
    }

    // 4. Permute: x = Q * w
    for (int64_t i = 0; i < m; ++i) {
        x[m_q_perm[i]] = w[i];
    }

    // 5. Apply Eta matrices in forward order: x <- E_k * x
    for (const auto& eta : m_etas) {
        int64_t p = eta.pivot_row;
        double xp = x[p];
        x[p] = xp * eta.diag_val;
        for (size_t k = 0; k < eta.indices.size(); ++k) {
            x[eta.indices[k]] += xp * eta.values[k];
        }
    }
}

std::vector<double> SparseLU::ftran(const std::vector<double>& b) const {
    std::vector<double> x(m_dim, 0.0);
    ftran(b.data(), x.data());
    return x;
}

void SparseLU::btran(const double* c, double* y) const {
    if (m_dim == 0) return;
    int64_t m = m_dim;

    // Working vector initialized to c
    std::vector<double> w(c, c + m);

    // 1. Apply Eta matrices in reverse order: w <- E_k^T * w
    for (auto it = m_etas.rbegin(); it != m_etas.rend(); ++it) {
        int64_t p = it->pivot_row;
        double dot = w[p] * it->diag_val;
        for (size_t k = 0; k < it->indices.size(); ++k) {
            dot += w[it->indices[k]] * it->values[k];
        }
        w[p] = dot;
    }

    // 2. Permute: z = Q^T * w
    std::vector<double> z(m);
    for (int64_t i = 0; i < m; ++i) {
        z[i] = w[m_q_perm[i]];
    }

    // 3. Forward solve U^T * v = z (U^T is lower triangular)
    std::vector<double> v(m);
    for (int64_t i = 0; i < m; ++i) {
        double val = z[i];
        for (int64_t idx = m_u_col_ptr[i]; idx < m_u_col_ptr[i + 1]; ++idx) {
            val -= m_u_values[idx] * v[m_u_row_ind[idx]];
        }
        v[i] = val / m_u_diag[i];
    }

    // 4. Backward solve L^T * u = v (L^T is unit upper triangular)
    std::vector<double> u = v;
    for (int64_t i = m - 1; i >= 0; --i) {
        double val = u[i];
        for (int64_t idx = m_l_col_ptr[i]; idx < m_l_col_ptr[i + 1]; ++idx) {
            val -= m_l_values[idx] * u[m_l_row_ind[idx]];
        }
        u[i] = val;
    }

    // 5. Permute: y = P^T * u
    for (int64_t i = 0; i < m; ++i) {
        y[m_p_perm[i]] = u[i];
    }
}

std::vector<double> SparseLU::btran(const std::vector<double>& c) const {
    std::vector<double> y(m_dim, 0.0);
    btran(c.data(), y.data());
    return y;
}

void SparseLU::ftran_sparse(const std::vector<int64_t>& b_indices,
                            const std::vector<double>& b_values,
                            std::vector<int64_t>& x_indices,
                            std::vector<double>& x_values,
                            std::vector<double>& dense_work) const {
    if (m_dim == 0) return;
    if (dense_work.size() < static_cast<size_t>(m_dim)) {
        dense_work.assign(m_dim, 0.0);
    } else {
        std::fill(dense_work.begin(), dense_work.end(), 0.0);
    }

    for (size_t i = 0; i < b_indices.size(); ++i) {
        dense_work[b_indices[i]] = b_values[i];
    }

    std::vector<double> full_x(m_dim, 0.0);
    ftran(dense_work.data(), full_x.data());

    x_indices.clear();
    x_values.clear();
    for (int64_t i = 0; i < m_dim; ++i) {
        if (std::abs(full_x[i]) > m_zero_tol) {
            x_indices.push_back(i);
            x_values.push_back(full_x[i]);
        }
    }
}

void SparseLU::btran_sparse(const std::vector<int64_t>& c_indices,
                            const std::vector<double>& c_values,
                            std::vector<int64_t>& y_indices,
                            std::vector<double>& y_values,
                            std::vector<double>& dense_work) const {
    if (m_dim == 0) return;
    if (dense_work.size() < static_cast<size_t>(m_dim)) {
        dense_work.assign(m_dim, 0.0);
    } else {
        std::fill(dense_work.begin(), dense_work.end(), 0.0);
    }

    for (size_t i = 0; i < c_indices.size(); ++i) {
        dense_work[c_indices[i]] = c_values[i];
    }

    std::vector<double> full_y(m_dim, 0.0);
    btran(dense_work.data(), full_y.data());

    y_indices.clear();
    y_values.clear();
    for (int64_t i = 0; i < m_dim; ++i) {
        if (std::abs(full_y[i]) > m_zero_tol) {
            y_indices.push_back(i);
            y_values.push_back(full_y[i]);
        }
    }
}

bool SparseLU::update_pfi(int64_t leaving_row,
                          const std::vector<double>& ftran_aq,
                          double min_pivot) {
    if (leaving_row < 0 || leaving_row >= m_dim) return false;
    double pivot_val = ftran_aq[leaving_row];
    if (std::abs(pivot_val) < min_pivot) {
        return false; // Pivot too small, requires refactorization
    }

    EtaVector eta;
    eta.pivot_row = leaving_row;
    eta.diag_val = 1.0 / pivot_val;

    for (int64_t i = 0; i < m_dim; ++i) {
        if (i != leaving_row && std::abs(ftran_aq[i]) > m_zero_tol) {
            eta.indices.push_back(i);
            eta.values.push_back(-ftran_aq[i] / pivot_val);
        }
    }

    m_etas.push_back(std::move(eta));
    return true;
}

bool SparseLU::needs_refactorization(int max_updates, double max_fill_ratio) const {
    if (static_cast<int>(m_etas.size()) >= max_updates) return true;
    if (m_initial_nnz > 0) {
        int64_t current_nnz = m_nnz_L + m_nnz_U;
        for (const auto& eta : m_etas) {
            current_nnz += static_cast<int64_t>(eta.indices.size()) + 1;
        }
        if (static_cast<double>(current_nnz) / m_initial_nnz > max_fill_ratio) {
            return true;
        }
    }
    return false;
}

} // namespace factorization
} // namespace sih
