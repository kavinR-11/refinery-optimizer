#include "sih/factorization/sparse_cholesky.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace sih {
namespace factorization {

bool SparseCholesky::factorize(int64_t m,
                               const model::SparseMatrix& M,
                               const std::vector<int64_t>& perm,
                               double reg) {
    m_m = m;
    m_valid = false;
    m_col_ptr.clear();
    m_row_ind.clear();
    m_values.clear();
    m_diag_inv.clear();

    if (m <= 0) return false;

    // 1. Setup permutation
    if (perm.empty() || static_cast<int64_t>(perm.size()) != m) {
        m_perm.resize(m);
        for (int64_t i = 0; i < m; ++i) m_perm[i] = i;
    } else {
        m_perm = perm;
    }

    m_inv_perm.resize(m);
    for (int64_t k = 0; k < m; ++k) {
        m_inv_perm[m_perm[k]] = k;
    }

    // 2. Permute M into M_bar = P * M * P^T in CSC format (lower triangular part only)
    std::vector<std::vector<std::pair<int64_t, double>>> m_bar_cols(m);

    const auto& mat_col_ptr = M.csc_col_ptr();
    const auto& mat_row_ind = M.csc_row_ind();
    const auto& mat_vals    = M.csc_values();

    for (int64_t orig_j = 0; orig_j < M.num_cols() && orig_j < m; ++orig_j) {
        int64_t j = m_inv_perm[orig_j];
        for (int64_t k = mat_col_ptr[orig_j]; k < mat_col_ptr[orig_j + 1]; ++k) {
            int64_t orig_i = mat_row_ind[k];
            if (orig_i >= m) continue;
            int64_t i = m_inv_perm[orig_i];
            double val = mat_vals[k];

            if (i >= j) {
                m_bar_cols[j].emplace_back(i, val);
            }
        }
    }

    // 3. Left-Looking Column-by-Column Sparse Cholesky
    m_col_ptr.assign(m + 1, 0);
    m_diag_inv.assign(m, 0.0);

    // Dense workspace accumulator for current column
    std::vector<double> w(m, 0.0);
    std::vector<bool> in_w(m, false);
    std::vector<int64_t> pattern;
    pattern.reserve(m);

    // List of nonzeros in L for each row to accelerate left-looking access
    // row_to_cols[i] = list of columns k where L_ik != 0 (with k < i)
    std::vector<std::vector<int64_t>> row_to_cols(m);

    for (int64_t j = 0; j < m; ++j) {
        pattern.clear();

        // Load column j of M_bar into accumulator w
        for (const auto& item : m_bar_cols[j]) {
            int64_t i = item.first;
            double val = item.second;
            if (!in_w[i]) {
                in_w[i] = true;
                pattern.push_back(i);
            }
            w[i] += val;
        }

        // Left-looking updates from prior columns k < j where L_jk != 0
        for (int64_t k : row_to_cols[j]) {
            // Find L_jk
            double l_jk = 0.0;
            for (int64_t p = m_col_ptr[k]; p < m_col_ptr[k + 1]; ++p) {
                if (m_row_ind[p] == j) {
                    l_jk = m_values[p];
                    break;
                }
            }

            if (std::abs(l_jk) < 1e-15) continue;

            // Subtract L_ik * L_jk for all i >= j in column k
            for (int64_t p = m_col_ptr[k]; p < m_col_ptr[k + 1]; ++p) {
                int64_t i = m_row_ind[p];
                if (i >= j) {
                    double l_ik = m_values[p];
                    if (!in_w[i]) {
                        in_w[i] = true;
                        pattern.push_back(i);
                    }
                    w[i] -= l_ik * l_jk;
                }
            }
        }

        // Sort pattern indices for deterministic CSC structure
        std::sort(pattern.begin(), pattern.end());

        // Diagonal entry L_jj
        double d_jj = w[j];
        if (d_jj <= reg) {
            d_jj += reg * (1.0 + std::abs(d_jj));
        }

        if (d_jj <= 0.0) {
            d_jj = 1e-8; // absolute positive safety
        }

        double l_jj = std::sqrt(d_jj);
        double l_jj_inv = 1.0 / l_jj;
        m_diag_inv[j] = l_jj_inv;

        // Store diagonal
        m_row_ind.push_back(j);
        m_values.push_back(l_jj);

        // Sub-diagonal entries L_ij = w[i] / L_jj
        for (int64_t i : pattern) {
            if (i > j) {
                double val = w[i] * l_jj_inv;
                if (std::abs(val) > 1e-15) {
                    m_row_ind.push_back(i);
                    m_values.push_back(val);
                    row_to_cols[i].push_back(j);
                }
            }
            w[i] = 0.0;
            in_w[i] = false;
        }

        m_col_ptr[j + 1] = static_cast<int64_t>(m_row_ind.size());
    }

    m_nnz_L = static_cast<int64_t>(m_values.size());
    m_valid = true;
    return true;
}

bool SparseCholesky::factorize_normal_equations(const model::SparseMatrix& A,
                                                const std::vector<double>& D,
                                                const std::vector<double>& reg_diag,
                                                const std::vector<int64_t>& perm,
                                                double reg) {
    int64_t m = A.num_rows();
    int64_t n = A.num_cols();

    // Form M = A * diag(D) * A^T + diag(reg_diag)
    const auto& col_ptr = A.csc_col_ptr();
    const auto& row_ind = A.csc_row_ind();
    const auto& vals    = A.csc_values();

    std::vector<model::Triplet> triplets;

    // Diagonal elements from reg_diag + reg
    for (int64_t i = 0; i < m; ++i) {
        double d_i = (i < static_cast<int64_t>(reg_diag.size())) ? reg_diag[i] : 0.0;
        triplets.emplace_back(i, i, d_i + reg);
    }

    // Outer products of columns of A scaled by D[j]
    for (int64_t j = 0; j < n; ++j) {
        double dj = (j < static_cast<int64_t>(D.size())) ? D[j] : 1.0;
        if (std::abs(dj) < 1e-20) continue;

        int64_t start = col_ptr[j];
        int64_t end   = col_ptr[j + 1];

        for (int64_t k1 = start; k1 < end; ++k1) {
            int64_t r1 = row_ind[k1];
            double v1 = vals[k1];

            for (int64_t k2 = start; k2 < end; ++k2) {
                int64_t r2 = row_ind[k2];
                double v2 = vals[k2];
                triplets.emplace_back(r1, r2, v1 * v2 * dj);
            }
        }
    }

    auto M = model::SparseMatrix::from_triplets(m, m, triplets);
    return factorize(m, M, perm, reg);
}

std::vector<double> SparseCholesky::solve(const std::vector<double>& b) const {
    if (!m_valid || static_cast<int64_t>(b.size()) != m_m) {
        return std::vector<double>(m_m, 0.0);
    }

    // 1. Permute b' = P * b
    std::vector<double> b_prime(m_m);
    for (int64_t k = 0; k < m_m; ++k) {
        b_prime[k] = b[m_perm[k]];
    }

    // 2. Forward solve L * v = b'
    // L is lower triangular with diagonal m_values[m_col_ptr[j]]
    std::vector<double> v(m_m, 0.0);
    for (int64_t j = 0; j < m_m; ++j) {
        int64_t start = m_col_ptr[j];
        int64_t end   = m_col_ptr[j + 1];

        // Diagonal is first entry in column j
        double l_jj = m_values[start];
        double vj = b_prime[j] / l_jj;
        v[j] = vj;

        for (int64_t p = start + 1; p < end; ++p) {
            int64_t i = m_row_ind[p];
            double l_ij = m_values[p];
            b_prime[i] -= l_ij * vj;
        }
    }

    // 3. Back solve L^T * w = v
    std::vector<double> w = v;
    for (int64_t j = m_m - 1; j >= 0; --j) {
        int64_t start = m_col_ptr[j];
        int64_t end   = m_col_ptr[j + 1];

        double sum = w[j];
        for (int64_t p = start + 1; p < end; ++p) {
            int64_t i = m_row_ind[p];
            double l_ij = m_values[p];
            sum -= l_ij * w[i];
        }

        double l_jj = m_values[start];
        w[j] = sum / l_jj;
    }

    // 4. Unpermute x = P^T * w
    std::vector<double> x(m_m);
    for (int64_t k = 0; k < m_m; ++k) {
        x[m_perm[k]] = w[k];
    }

    return x;
}

} // namespace factorization
} // namespace sih
