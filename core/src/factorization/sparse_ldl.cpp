#include "sih/factorization/sparse_ldl.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace sih {
namespace factorization {

bool SparseLdl::factorize(int64_t n,
                          const model::SparseMatrix& K,
                          const std::vector<int64_t>& perm,
                          double reg) {
    m_n = n;
    m_valid = false;
    m_col_ptr.clear();
    m_row_ind.clear();
    m_values.clear();
    m_D.clear();
    m_D_inv.clear();

    if (n <= 0) return false;

    // 1. Permutation setup
    if (perm.empty() || static_cast<int64_t>(perm.size()) != n) {
        m_perm.resize(n);
        for (int64_t i = 0; i < n; ++i) m_perm[i] = i;
    } else {
        m_perm = perm;
    }

    m_inv_perm.resize(n);
    for (int64_t k = 0; k < n; ++k) {
        m_inv_perm[m_perm[k]] = k;
    }

    // 2. Permute K into K_bar = P * K * P^T in lower triangular format
    std::vector<std::vector<std::pair<int64_t, double>>> k_bar_cols(n);

    const auto& mat_col_ptr = K.csc_col_ptr();
    const auto& mat_row_ind = K.csc_row_ind();
    const auto& mat_vals    = K.csc_values();

    for (int64_t orig_j = 0; orig_j < K.num_cols() && orig_j < n; ++orig_j) {
        int64_t j = m_inv_perm[orig_j];
        for (int64_t k = mat_col_ptr[orig_j]; k < mat_col_ptr[orig_j + 1]; ++k) {
            int64_t orig_i = mat_row_ind[k];
            if (orig_i >= n) continue;
            int64_t i = m_inv_perm[orig_i];
            double val = mat_vals[k];

            if (i >= j) {
                k_bar_cols[j].emplace_back(i, val);
            }
        }
    }

    // 3. Left-Looking Column-by-Column Sparse LDL^T
    m_col_ptr.assign(n + 1, 0);
    m_D.assign(n, 0.0);
    m_D_inv.assign(n, 0.0);

    std::vector<double> w(n, 0.0);
    std::vector<bool> in_w(n, false);
    std::vector<int64_t> pattern;
    pattern.reserve(n);

    std::vector<std::vector<int64_t>> row_to_cols(n);

    for (int64_t j = 0; j < n; ++j) {
        pattern.clear();

        // Load column j of K_bar into accumulator w
        for (const auto& item : k_bar_cols[j]) {
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
            double t = l_jk * m_D[k];

            // w[j] -= L_jk * t = L_jk^2 * D_k
            // And for all i > j with L_ik != 0: w[i] -= L_ik * t
            for (int64_t p = m_col_ptr[k]; p < m_col_ptr[k + 1]; ++p) {
                int64_t i = m_row_ind[p];
                if (i >= j) {
                    double l_ik = m_values[p];
                    if (!in_w[i]) {
                        in_w[i] = true;
                        pattern.push_back(i);
                    }
                    w[i] -= l_ik * t;
                }
            }
        }

        std::sort(pattern.begin(), pattern.end());

        // Pivot D_jj = w[j]
        double d_jj = w[j];
        if (std::abs(d_jj) <= reg) {
            d_jj = (d_jj >= 0.0 ? 1.0 : -1.0) * reg * (1.0 + std::abs(w[j]));
        }
        if (std::abs(d_jj) < 1e-14) {
            d_jj = (d_jj >= 0.0 ? 1e-10 : -1e-10);
        }

        m_D[j] = d_jj;
        double d_jj_inv = 1.0 / d_jj;
        m_D_inv[j] = d_jj_inv;

        // Subdiagonal entries L_ij = w[i] / D_jj
        for (int64_t i : pattern) {
            if (i > j) {
                double val = w[i] * d_jj_inv;
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

std::vector<double> SparseLdl::solve(const std::vector<double>& b) const {
    if (!m_valid || static_cast<int64_t>(b.size()) != m_n) {
        return std::vector<double>(m_n, 0.0);
    }

    // 1. Permute b' = P * b
    std::vector<double> b_prime(m_n);
    for (int64_t k = 0; k < m_n; ++k) {
        b_prime[k] = b[m_perm[k]];
    }

    // 2. Forward solve unit lower triangular L * v = b'
    // L_jj = 1.0 implicitly
    std::vector<double> v(m_n, 0.0);
    for (int64_t j = 0; j < m_n; ++j) {
        double vj = b_prime[j];
        v[j] = vj;

        int64_t start = m_col_ptr[j];
        int64_t end   = m_col_ptr[j + 1];
        for (int64_t p = start; p < end; ++p) {
            int64_t i = m_row_ind[p];
            double l_ij = m_values[p];
            b_prime[i] -= l_ij * vj;
        }
    }

    // 3. Diagonal solve D * u = v
    std::vector<double> u(m_n, 0.0);
    for (int64_t j = 0; j < m_n; ++j) {
        u[j] = v[j] * m_D_inv[j];
    }

    // 4. Back solve unit upper triangular L^T * w = u
    std::vector<double> w = u;
    for (int64_t j = m_n - 1; j >= 0; --j) {
        int64_t start = m_col_ptr[j];
        int64_t end   = m_col_ptr[j + 1];

        double sum = w[j];
        for (int64_t p = start; p < end; ++p) {
            int64_t i = m_row_ind[p];
            double l_ij = m_values[p];
            sum -= l_ij * w[i];
        }
        w[j] = sum;
    }

    // 5. Unpermute x = P^T * w
    std::vector<double> x(m_n);
    for (int64_t k = 0; k < m_n; ++k) {
        x[m_perm[k]] = w[k];
    }

    return x;
}

} // namespace factorization
} // namespace sih
