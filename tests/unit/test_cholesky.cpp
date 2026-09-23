#include "sih/factorization/sparse_cholesky.hpp"
#include "sih/factorization/amd.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " (" #cond ")" << std::endl; \
            std::abort(); \
        } \
    } while (0)

using namespace sih::model;
using namespace sih::factorization;

void test_cholesky_identity() {
    std::cout << "[Test] test_cholesky_identity running..." << std::endl;
    int64_t m = 5;
    std::vector<Triplet> triplets;
    for (int64_t i = 0; i < m; ++i) {
        triplets.emplace_back(i, i, 4.0); // Diag = 4.0
    }
    auto M = SparseMatrix::from_triplets(m, m, triplets);

    SparseCholesky chol;
    bool ok = chol.factorize(m, M);
    TEST_ASSERT(ok);
    TEST_ASSERT(chol.is_valid());

    std::vector<double> b = {8.0, 12.0, 16.0, 20.0, 24.0};
    auto x = chol.solve(b);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(x[i] - b[i] / 4.0) < 1e-12);
    }
    std::cout << "[Test] test_cholesky_identity passed." << std::endl;
}

void test_cholesky_spd_tridiagonal() {
    std::cout << "[Test] test_cholesky_spd_tridiagonal running..." << std::endl;
    // 4x4 tridiagonal SPD matrix:
    // [  4  -1   0   0 ]
    // [ -1   4  -1   0 ]
    // [  0  -1   4  -1 ]
    // [  0   0  -1   4 ]
    int64_t m = 4;
    std::vector<Triplet> triplets = {
        {0, 0, 4.0}, {0, 1, -1.0},
        {1, 0, -1.0}, {1, 1, 4.0}, {1, 2, -1.0},
        {2, 1, -1.0}, {2, 2, 4.0}, {2, 3, -1.0},
        {3, 2, -1.0}, {3, 3, 4.0}
    };
    auto M = SparseMatrix::from_triplets(m, m, triplets);

    // Natural order
    SparseCholesky chol;
    bool ok = chol.factorize(m, M);
    TEST_ASSERT(ok);

    std::vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = M.mat_vec(x_true);

    auto x_calc = chol.solve(b);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(x_calc[i] - x_true[i]) < 1e-12);
    }

    // With AMD order
    auto perm = AmdOrder::order_matrix(m, M.csc_col_ptr(), M.csc_row_ind());
    SparseCholesky chol_amd;
    ok = chol_amd.factorize(m, M, perm);
    TEST_ASSERT(ok);

    auto x_calc_amd = chol_amd.solve(b);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(x_calc_amd[i] - x_true[i]) < 1e-12);
    }

    std::cout << "[Test] test_cholesky_spd_tridiagonal passed." << std::endl;
}

void test_cholesky_normal_equations() {
    std::cout << "[Test] test_cholesky_normal_equations running..." << std::endl;
    int64_t m = 3;
    int64_t n = 4;
    // A: 3x4
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 2.0},
        {1, 1, 1.0}, {1, 2, 3.0},
        {2, 2, 1.0}, {2, 3, 2.0}
    };
    auto A = SparseMatrix::from_triplets(m, n, triplets);
    std::vector<double> D = {1.0, 2.0, 0.5, 3.0};
    std::vector<double> reg_diag = {0.1, 0.1, 0.1};

    SparseCholesky chol;
    bool ok = chol.factorize_normal_equations(A, D, reg_diag);
    TEST_ASSERT(ok);

    std::vector<double> b = {10.0, 20.0, 30.0};
    auto x = chol.solve(b);

    // Verify M * x == b
    // Compute M explicitly to check residual:
    // M = A * D * A^T + diag(reg_diag)
    std::vector<double> ADAt_x(m, 0.0);
    // At * x
    auto At_x = A.mat_trans_vec(x);
    // D * (At * x)
    for (int64_t j = 0; j < n; ++j) At_x[j] *= D[j];
    // A * (D * At * x)
    auto M_x = A.mat_vec(At_x);
    for (int64_t i = 0; i < m; ++i) {
        M_x[i] += reg_diag[i] * x[i];
        TEST_ASSERT(std::abs(M_x[i] - b[i]) < 1e-10);
    }

    std::cout << "[Test] test_cholesky_normal_equations passed with residual < 1e-10." << std::endl;
}

int main() {
    test_cholesky_identity();
    test_cholesky_spd_tridiagonal();
    test_cholesky_normal_equations();
    std::cout << "All Sparse Cholesky tests passed!" << std::endl;
    return 0;
}
