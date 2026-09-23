#include "sih/factorization/sparse_lu.hpp"
#include <iostream>
#include <cstdlib>
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

void test_identity() {
    std::cout << "[Test] test_identity running..." << std::endl;
    int64_t m = 5;
    std::vector<Triplet> triplets;
    for (int64_t i = 0; i < m; ++i) {
        triplets.emplace_back(i, i, 1.0);
    }
    auto B = SparseMatrix::from_triplets(m, m, triplets);

    SparseLU lu;
    auto status = lu.factorize(B);
    TEST_ASSERT(status == FactorizationStatus::Success);
    TEST_ASSERT(lu.is_valid());

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto x = lu.ftran(b);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(x[i] - b[i]) < 1e-12);
    }

    auto y = lu.btran(b);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(y[i] - b[i]) < 1e-12);
    }
    std::cout << "[Test] test_identity passed." << std::endl;
}

void test_general_matrix() {
    std::cout << "[Test] test_general_matrix running..." << std::endl;
    // 4x4 matrix:
    // [ 2  1  0  0 ]
    // [ 1  2  1  0 ]
    // [ 0  1  2  1 ]
    // [ 0  0  1  2 ]
    int64_t m = 4;
    std::vector<Triplet> triplets = {
        {0, 0, 2.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 2.0}, {1, 2, 1.0},
        {2, 1, 1.0}, {2, 2, 2.0}, {2, 3, 1.0},
        {3, 2, 1.0}, {3, 3, 2.0}
    };
    auto B = SparseMatrix::from_triplets(m, m, triplets);

    SparseLU lu;
    auto status = lu.factorize(B);
    TEST_ASSERT(status == FactorizationStatus::Success);

    // Test FTRAN: B * x = b
    std::vector<double> b = {4.0, 7.0, 8.0, 5.0};
    auto x = lu.ftran(b);

    // Verify B * x == b
    auto b_rec = B.mat_vec(x);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(b_rec[i] - b[i]) < 1e-11);
    }

    // Test BTRAN: B^T * y = c
    std::vector<double> c = {3.0, 6.0, 9.0, 12.0};
    auto y = lu.btran(c);

    auto c_rec = B.mat_trans_vec(y);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(c_rec[i] - c[i]) < 1e-11);
    }

    std::cout << "[Test] test_general_matrix passed." << std::endl;
}

void test_pfi_update() {
    std::cout << "[Test] test_pfi_update running..." << std::endl;
    int64_t m = 3;
    // Matrix B:
    // [ 1  2  0 ]
    // [ 0  1  1 ]
    // [ 1  0  2 ]
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 2.0},
        {1, 1, 1.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 2, 2.0}
    };
    auto B = SparseMatrix::from_triplets(m, m, triplets);

    SparseLU lu;
    auto status = lu.factorize(B);
    TEST_ASSERT(status == FactorizationStatus::Success);

    // Replace column 1 of B with entering column aq = [3, 4, 1]^T
    std::vector<double> aq = {3.0, 4.0, 1.0};
    auto ftran_aq = lu.ftran(aq);

    int64_t leaving_col = 1;
    bool ok = lu.update_pfi(leaving_col, ftran_aq);
    TEST_ASSERT(ok);
    TEST_ASSERT(lu.num_updates() == 1);

    // Construct expected updated matrix B_new:
    // [ 1  3  0 ]
    // [ 0  4  1 ]
    // [ 1  1  2 ]
    std::vector<Triplet> new_triplets = {
        {0, 0, 1.0}, {0, 1, 3.0},
        {1, 1, 4.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 1, 1.0}, {2, 2, 2.0}
    };
    auto B_new = SparseMatrix::from_triplets(m, m, new_triplets);

    // Check that FTRAN solves B_new * x = rhs
    std::vector<double> rhs = {5.0, 9.0, 4.0};
    auto x = lu.ftran(rhs);
    TEST_ASSERT(static_cast<int64_t>(x.size()) == m);
    auto rhs_check = B_new.mat_vec(x);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(rhs_check[i] - rhs[i]) < 1e-10);
    }

    // Check that BTRAN solves B_new^T * y = c
    std::vector<double> c = {2.0, 5.0, 1.0};
    auto y = lu.btran(c);
    TEST_ASSERT(static_cast<int64_t>(y.size()) == m);
    auto c_check = B_new.mat_trans_vec(y);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(std::abs(c_check[i] - c[i]) < 1e-10);
    }

    std::cout << "[Test] test_pfi_update passed." << std::endl;
}

void test_singular_matrix() {
    std::cout << "[Test] test_singular_matrix running..." << std::endl;
    // Singular matrix with duplicate row
    int64_t m = 3;
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 2.0}, {0, 2, 3.0},
        {1, 0, 2.0}, {1, 1, 4.0}, {1, 2, 6.0}, // row 1 = 2 * row 0
        {2, 0, 0.0}, {2, 1, 1.0}, {2, 2, 1.0}
    };
    auto B = SparseMatrix::from_triplets(m, m, triplets);

    SparseLU lu;
    auto status = lu.factorize(B);
    TEST_ASSERT(status == FactorizationStatus::Singular);
    TEST_ASSERT(!lu.is_valid());
    std::cout << "[Test] test_singular_matrix passed." << std::endl;
}

int main() {
    std::cout << "=== Running SparseLU Unit Tests ===" << std::endl;
    test_identity();
    test_general_matrix();
    test_pfi_update();
    test_singular_matrix();
    std::cout << "=== All SparseLU Unit Tests Passed! ===" << std::endl;
    return 0;
}
