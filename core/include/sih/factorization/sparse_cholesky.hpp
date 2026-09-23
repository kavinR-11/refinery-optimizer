#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "sih/model/sparse_matrix.hpp"

namespace sih {
namespace factorization {

class SparseCholesky {
public:
    SparseCholesky() = default;

    // Factorize a general symmetric positive definite matrix M (m x m)
    // Permutation perm can be supplied (e.g. from AMD). If empty, natural order is used.
    bool factorize(int64_t m,
                   const model::SparseMatrix& M,
                   const std::vector<int64_t>& perm = {},
                   double reg = 1e-12);

    // Form and factorize normal equations M = A * diag(D) * A^T + diag(reg_diag)
    bool factorize_normal_equations(const model::SparseMatrix& A,
                                    const std::vector<double>& D,
                                    const std::vector<double>& reg_diag,
                                    const std::vector<int64_t>& perm = {},
                                    double reg = 1e-12);

    // Solve M * x = b using L * L^T factorization and permutation
    std::vector<double> solve(const std::vector<double>& b) const;

    // Check if factorization is valid
    bool is_valid() const noexcept { return m_valid; }

    int64_t dim() const noexcept { return m_m; }
    int64_t num_nonzeros_L() const noexcept { return m_nnz_L; }
    const std::vector<int64_t>& perm() const noexcept { return m_perm; }

private:
    int64_t m_m = 0;
    int64_t m_nnz_L = 0;
    bool m_valid = false;

    std::vector<int64_t> m_perm;
    std::vector<int64_t> m_inv_perm;

    // Lower triangular factor L in CSC format
    std::vector<int64_t> m_col_ptr;
    std::vector<int64_t> m_row_ind;
    std::vector<double>  m_values;
    std::vector<double>  m_diag_inv; // 1.0 / L_jj for fast solve
};

} // namespace factorization
} // namespace sih
