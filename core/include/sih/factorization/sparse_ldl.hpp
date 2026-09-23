#pragma once

#include <vector>
#include <cstdint>
#include "sih/model/sparse_matrix.hpp"

namespace sih {
namespace factorization {

class SparseLdl {
public:
    SparseLdl() = default;

    // Factorize a symmetric indefinite matrix K (N x N) as P * K * P^T = L * D * L^T
    // where L is unit lower triangular and D is diagonal.
    bool factorize(int64_t n,
                   const model::SparseMatrix& K,
                   const std::vector<int64_t>& perm = {},
                   double reg = 1e-12);

    // Solve K * x = b
    std::vector<double> solve(const std::vector<double>& b) const;

    bool is_valid() const noexcept { return m_valid; }
    int64_t dim() const noexcept { return m_n; }
    int64_t num_nonzeros_L() const noexcept { return m_nnz_L; }
    const std::vector<int64_t>& perm() const noexcept { return m_perm; }

private:
    int64_t m_n = 0;
    int64_t m_nnz_L = 0;
    bool m_valid = false;

    std::vector<int64_t> m_perm;
    std::vector<int64_t> m_inv_perm;

    // Unit lower triangular factor L in CSC format (excluding diagonal 1.0)
    std::vector<int64_t> m_col_ptr;
    std::vector<int64_t> m_row_ind;
    std::vector<double>  m_values;

    // Diagonal matrix D
    std::vector<double> m_D;
    std::vector<double> m_D_inv;
};

} // namespace factorization
} // namespace sih
