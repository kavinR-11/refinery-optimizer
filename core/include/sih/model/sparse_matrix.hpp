#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

namespace sih {
namespace model {

struct Triplet {
    int64_t row{0};
    int64_t col{0};
    double value{0.0};

    Triplet() = default;
    Triplet(int64_t r, int64_t c, double v) : row(r), col(c), value(v) {}
};

class SparseMatrix {
public:
    SparseMatrix() : m_rows(0), m_cols(0) {}
    SparseMatrix(int64_t rows, int64_t cols) : m_rows(rows), m_cols(cols) {}

    // Factory method to build dual CSC & CSR sparse matrix from triplets
    static SparseMatrix from_triplets(int64_t m, int64_t n,
                                      const std::vector<Triplet>& triplets,
                                      double zero_tol = 1e-15);

    int64_t num_rows() const noexcept { return m_rows; }
    int64_t num_cols() const noexcept { return m_cols; }
    int64_t num_nonzeros() const noexcept { return static_cast<int64_t>(m_csc_values.size()); }

    // CSC format getters
    const std::vector<int64_t>& csc_col_ptr() const noexcept { return m_csc_col_ptr; }
    const std::vector<int64_t>& csc_row_ind() const noexcept { return m_csc_row_ind; }
    const std::vector<double>& csc_values() const noexcept { return m_csc_values; }

    // CSR format getters
    const std::vector<int64_t>& csr_row_ptr() const noexcept { return m_csr_row_ptr; }
    const std::vector<int64_t>& csr_col_ind() const noexcept { return m_csr_col_ind; }
    const std::vector<double>& csr_values() const noexcept { return m_csr_values; }

    // Matrix-vector operations: y = A * x (using CSR)
    void mat_vec(const double* x, double* y) const;
    std::vector<double> mat_vec(const std::vector<double>& x) const;

    // Transpose matrix-vector operations: z = A^T * y (using CSC)
    void mat_trans_vec(const double* y, double* z) const;
    std::vector<double> mat_trans_vec(const std::vector<double>& y) const;

    // Transpose matrix: returns a new SparseMatrix with swapped dimensions and reversed CSC/CSR roles
    SparseMatrix transpose() const;

    // Element access: returns A(i, j)
    double get(int64_t row, int64_t col) const;

    // Check consistency of CSC and CSR internal structures
    bool is_valid() const;

private:
    int64_t m_rows{0};
    int64_t m_cols{0};

    // Compressed Sparse Column (CSC)
    std::vector<int64_t> m_csc_col_ptr;
    std::vector<int64_t> m_csc_row_ind;
    std::vector<double> m_csc_values;

    // Compressed Sparse Row (CSR)
    std::vector<int64_t> m_csr_row_ptr;
    std::vector<int64_t> m_csr_col_ind;
    std::vector<double> m_csr_values;
};

} // namespace model
} // namespace sih
