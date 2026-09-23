#include "sih/model/sparse_matrix.hpp"
#include <algorithm>
#include <cmath>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sih {
namespace model {

SparseMatrix SparseMatrix::from_triplets(int64_t m, int64_t n,
                                         const std::vector<Triplet>& triplets,
                                         double zero_tol) {
    if (m < 0 || n < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }

    SparseMatrix mat(m, n);

    // Sum duplicate triplets and remove entries below zero_tol
    // Using a coordinate map to accumulate duplicates
    std::map<std::pair<int64_t, int64_t>, double> coord_map;
    for (const auto& trip : triplets) {
        if (trip.row < 0 || trip.row >= m || trip.col < 0 || trip.col >= n) {
            throw std::out_of_range("Triplet indices out of matrix bounds");
        }
        coord_map[{trip.row, trip.col}] += trip.value;
    }

    // Filter out zeros below zero_tol
    std::vector<Triplet> clean_triplets;
    clean_triplets.reserve(coord_map.size());
    for (const auto& kv : coord_map) {
        if (std::abs(kv.second) > zero_tol) {
            clean_triplets.emplace_back(kv.first.first, kv.first.second, kv.second);
        }
    }

    const int64_t nnz = static_cast<int64_t>(clean_triplets.size());

    // 1. Build CSC
    // Sort by (col ascending, then row ascending)
    std::vector<Triplet> csc_triplets = clean_triplets;
    std::sort(csc_triplets.begin(), csc_triplets.end(),
              [](const Triplet& a, const Triplet& b) {
                  if (a.col != b.col) return a.col < b.col;
                  return a.row < b.row;
              });

    mat.m_csc_col_ptr.assign(n + 1, 0);
    mat.m_csc_row_ind.resize(nnz);
    mat.m_csc_values.resize(nnz);

    for (int64_t k = 0; k < nnz; ++k) {
        mat.m_csc_col_ptr[csc_triplets[k].col + 1]++;
        mat.m_csc_row_ind[k] = csc_triplets[k].row;
        mat.m_csc_values[k] = csc_triplets[k].value;
    }
    for (int64_t j = 0; j < n; ++j) {
        mat.m_csc_col_ptr[j + 1] += mat.m_csc_col_ptr[j];
    }

    // 2. Build CSR
    // Sort by (row ascending, then col ascending)
    std::vector<Triplet> csr_triplets = clean_triplets;
    std::sort(csr_triplets.begin(), csr_triplets.end(),
              [](const Triplet& a, const Triplet& b) {
                  if (a.row != b.row) return a.row < b.row;
                  return a.col < b.col;
              });

    mat.m_csr_row_ptr.assign(m + 1, 0);
    mat.m_csr_col_ind.resize(nnz);
    mat.m_csr_values.resize(nnz);

    for (int64_t k = 0; k < nnz; ++k) {
        mat.m_csr_row_ptr[csr_triplets[k].row + 1]++;
        mat.m_csr_col_ind[k] = csr_triplets[k].col;
        mat.m_csr_values[k] = csr_triplets[k].value;
    }
    for (int64_t i = 0; i < m; ++i) {
        mat.m_csr_row_ptr[i + 1] += mat.m_csr_row_ptr[i];
    }

    return mat;
}

void SparseMatrix::mat_vec(const double* x, double* y) const {
    if (!x || !y) return;
    const int64_t m = m_rows;

#pragma omp parallel for schedule(static) if (m > 1000)
    for (int64_t i = 0; i < m; ++i) {
        double sum = 0.0;
        const int64_t start = m_csr_row_ptr[i];
        const int64_t end = m_csr_row_ptr[i + 1];
        for (int64_t k = start; k < end; ++k) {
            sum += m_csr_values[k] * x[m_csr_col_ind[k]];
        }
        y[i] = sum;
    }
}

std::vector<double> SparseMatrix::mat_vec(const std::vector<double>& x) const {
    if (static_cast<int64_t>(x.size()) != m_cols) {
        throw std::invalid_argument("Vector length does not match matrix column count in mat_vec");
    }
    std::vector<double> y(m_rows, 0.0);
    mat_vec(x.data(), y.data());
    return y;
}

void SparseMatrix::mat_trans_vec(const double* y, double* z) const {
    if (!y || !z) return;
    const int64_t n = m_cols;

#pragma omp parallel for schedule(static) if (n > 1000)
    for (int64_t j = 0; j < n; ++j) {
        double sum = 0.0;
        const int64_t start = m_csc_col_ptr[j];
        const int64_t end = m_csc_col_ptr[j + 1];
        for (int64_t k = start; k < end; ++k) {
            sum += m_csc_values[k] * y[m_csc_row_ind[k]];
        }
        z[j] = sum;
    }
}

std::vector<double> SparseMatrix::mat_trans_vec(const std::vector<double>& y) const {
    if (static_cast<int64_t>(y.size()) != m_rows) {
        throw std::invalid_argument("Vector length does not match matrix row count in mat_trans_vec");
    }
    std::vector<double> z(m_cols, 0.0);
    mat_trans_vec(y.data(), z.data());
    return z;
}

SparseMatrix SparseMatrix::transpose() const {
    SparseMatrix trans(m_cols, m_rows);

    // CSR of A becomes CSC of A^T
    trans.m_csc_col_ptr = m_csr_row_ptr;
    trans.m_csc_row_ind = m_csr_col_ind;
    trans.m_csc_values = m_csr_values;

    // CSC of A becomes CSR of A^T
    trans.m_csr_row_ptr = m_csc_col_ptr;
    trans.m_csr_col_ind = m_csc_row_ind;
    trans.m_csr_values = m_csc_values;

    return trans;
}

double SparseMatrix::get(int64_t row, int64_t col) const {
    if (row < 0 || row >= m_rows || col < 0 || col >= m_cols) {
        throw std::out_of_range("Indices out of range in SparseMatrix::get");
    }

    const int64_t start = m_csr_row_ptr[row];
    const int64_t end = m_csr_row_ptr[row + 1];

    auto it = std::lower_bound(m_csr_col_ind.begin() + start,
                               m_csr_col_ind.begin() + end,
                               col);

    if (it != m_csr_col_ind.begin() + end && *it == col) {
        const int64_t idx = std::distance(m_csr_col_ind.begin(), it);
        return m_csr_values[idx];
    }
    return 0.0;
}

bool SparseMatrix::is_valid() const {
    if (static_cast<int64_t>(m_csc_col_ptr.size()) != m_cols + 1) return false;
    if (static_cast<int64_t>(m_csr_row_ptr.size()) != m_rows + 1) return false;
    if (m_csc_values.size() != m_csr_values.size()) return false;
    if (m_csc_row_ind.size() != m_csc_values.size()) return false;
    if (m_csr_col_ind.size() != m_csr_values.size()) return false;
    return true;
}

} // namespace model
} // namespace sih
