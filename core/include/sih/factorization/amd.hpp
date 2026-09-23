#pragma once

#include <vector>
#include <cstdint>
#include "sih/model/sparse_matrix.hpp"

namespace sih {
namespace factorization {

class AmdOrder {
public:
    // Computes fill-reducing permutation from a general symmetric matrix pattern
    static std::vector<int64_t> order_matrix(int64_t n,
                                             const std::vector<int64_t>& col_ptr,
                                             const std::vector<int64_t>& row_ind);

    // Computes fill-reducing permutation for A * A^T directly from matrix A (m x n)
    static std::vector<int64_t> order_aat(const model::SparseMatrix& A);

    // Helper: Invert a permutation vector
    static std::vector<int64_t> invert_permutation(const std::vector<int64_t>& perm);
};

} // namespace factorization
} // namespace sih
