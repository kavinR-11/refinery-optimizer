#pragma once

#include "sih/model/sparse_matrix.hpp"
#include <vector>
#include <cstdint>
#include <memory>

namespace sih {
namespace factorization {

enum class FactorizationStatus {
    Success = 0,
    Singular = 1,
    NumericalInstability = 2,
    DimensionMismatch = 3
};

struct EtaVector {
    int64_t pivot_row{0};
    std::vector<int64_t> indices;
    std::vector<double> values;
    double diag_val{1.0};
};

class SparseLU {
public:
    SparseLU() = default;

    // Factorize basis matrix B from triplets or column vectors of constraint matrix
    FactorizationStatus factorize(int64_t m,
                                  const std::vector<int64_t>& basic_vars,
                                  const model::SparseMatrix& A,
                                  double pivot_tol = 0.1,
                                  double zero_tol = 1e-13);

    // Factorize a standalone square SparseMatrix
    FactorizationStatus factorize(const model::SparseMatrix& B,
                                  double pivot_tol = 0.1,
                                  double zero_tol = 1e-13);

    // FTRAN: Solves B * x = b (Forward transformation, primal step)
    // Applies L, U, and all accumulated Eta updates
    void ftran(const double* b, double* x) const;
    std::vector<double> ftran(const std::vector<double>& b) const;

    // Hypersparse FTRAN: solves when b is sparse, returns touched nonzero indices
    void ftran_sparse(const std::vector<int64_t>& b_indices,
                      const std::vector<double>& b_values,
                      std::vector<int64_t>& x_indices,
                      std::vector<double>& x_values,
                      std::vector<double>& dense_work) const;

    // BTRAN: Solves B^T * y = c (Backward transformation, dual step)
    // Applies inverse Eta updates in reverse, then U^T, L^T
    void btran(const double* c, double* y) const;
    std::vector<double> btran(const std::vector<double>& c) const;

    // Hypersparse BTRAN: solves when c is sparse, returns touched nonzero indices
    void btran_sparse(const std::vector<int64_t>& c_indices,
                      const std::vector<double>& c_values,
                      std::vector<int64_t>& y_indices,
                      std::vector<double>& y_values,
                      std::vector<double>& dense_work) const;

    // Update factorization with new column replacing column in basis (PFI update)
    // ftran_aq is the solved vector d = B^{-1} * a_q, and leaving_row is pivot row p
    bool update_pfi(int64_t leaving_row, const std::vector<double>& ftran_aq, double min_pivot = 1e-11);

    // Refactorization policy checks
    bool needs_refactorization(int max_updates = 60, double max_fill_ratio = 3.0) const;

    // Getters & diagnostics
    int64_t dimension() const noexcept { return m_dim; }
    int64_t num_updates() const noexcept { return static_cast<int64_t>(m_etas.size()); }
    int64_t nnz_L() const noexcept { return m_nnz_L; }
    int64_t nnz_U() const noexcept { return m_nnz_U; }
    double condition_estimate() const noexcept { return m_cond_est; }
    bool is_valid() const noexcept { return m_status == FactorizationStatus::Success; }
    FactorizationStatus status() const noexcept { return m_status; }

private:
    int64_t m_dim{0};
    FactorizationStatus m_status{FactorizationStatus::DimensionMismatch};
    double m_pivot_tol{0.1};
    double m_zero_tol{1e-13};

    // Permutation vectors
    std::vector<int64_t> m_p_perm;     // row perm: P[i] = orig_row
    std::vector<int64_t> m_inv_p_perm; // orig_row -> perm_row
    std::vector<int64_t> m_q_perm;     // col perm: Q[j] = orig_col
    std::vector<int64_t> m_inv_q_perm; // orig_col -> perm_col

    // L factor (unit lower triangular: diag is 1.0, not stored)
    // Stored column-wise (for FTRAN) and row-wise (for BTRAN)
    std::vector<int64_t> m_l_col_ptr;
    std::vector<int64_t> m_l_row_ind;
    std::vector<double>  m_l_values;

    std::vector<int64_t> m_l_row_ptr;
    std::vector<int64_t> m_l_col_ind;
    std::vector<double>  m_l_row_values;

    // U factor (upper triangular: diag stored separately for O(1) division)
    std::vector<double>  m_u_diag;     // U(k, k)
    std::vector<int64_t> m_u_col_ptr;
    std::vector<int64_t> m_u_row_ind;
    std::vector<double>  m_u_values;

    std::vector<int64_t> m_u_row_ptr;
    std::vector<int64_t> m_u_col_ind;
    std::vector<double>  m_u_row_values;

    // Product-form of the inverse updates (Etas)
    std::vector<EtaVector> m_etas;

    int64_t m_nnz_L{0};
    int64_t m_nnz_U{0};
    int64_t m_initial_nnz{0};
    double m_cond_est{1.0};

    // Internal sparse build helpers
    void build_transposes();
};

} // namespace factorization
} // namespace sih
