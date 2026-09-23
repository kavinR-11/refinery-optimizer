#include "sih/factorization/sparse_ldl.hpp"
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

void test_ldl_indefinite_2x2() {
    std::cout << "[Test] test_ldl_indefinite_2x2 running..." << std::endl;
    // [ 2   1 ]
    // [ 1  -2 ]
    int64_t n = 2;
    std::vector<Triplet> triplets = {
        {0, 0, 2.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, -2.0}
    };
    auto K = SparseMatrix::from_triplets(n, n, triplets);

    SparseLdl ldl;
    bool ok = ldl.factorize(n, K);
    TEST_ASSERT(ok);
    TEST_ASSERT(ldl.is_valid());

    std::vector<double> x_true = {3.0, -1.0};
    auto b = K.mat_vec(x_true);

    auto x_calc = ldl.solve(b);
    for (int64_t i = 0; i < n; ++i) {
        TEST_ASSERT(std::abs(x_calc[i] - x_true[i]) < 1e-12);
    }
    std::cout << "[Test] test_ldl_indefinite_2x2 passed." << std::endl;
}

void test_ldl_augmented_kkt() {
    std::cout << "[Test] test_ldl_augmented_kkt running..." << std::endl;
    // Augmented KKT system:
    // [ H   A^T ]
    // [ A  -G   ]
    // H: 2x2 SPD, G: 2x2 SPD
    int64_t n = 4;
    std::vector<Triplet> triplets = {
        // H
        {0, 0, 4.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 3.0},
        // A and A^T
        {2, 0, 1.0}, {0, 2, 1.0},
        {2, 1, 2.0}, {1, 2, 2.0},
        {3, 0, 3.0}, {0, 3, 3.0},
        {3, 1, -1.0}, {1, 3, -1.0},
        // -G
        {2, 2, -1.0},
        {3, 3, -2.0}
    };
    auto K = SparseMatrix::from_triplets(n, n, triplets);

    SparseLdl ldl;
    bool ok = ldl.factorize(n, K);
    TEST_ASSERT(ok);

    std::vector<double> x_true = {1.5, -2.5, 0.5, 3.0};
    auto b = K.mat_vec(x_true);

    auto x_calc = ldl.solve(b);
    for (int64_t i = 0; i < n; ++i) {
        TEST_ASSERT(std::abs(x_calc[i] - x_true[i]) < 1e-11);
    }
    std::cout << "[Test] test_ldl_augmented_kkt passed with residual < 1e-11." << std::endl;
}

int main() {
    test_ldl_indefinite_2x2();
    test_ldl_augmented_kkt();
    std::cout << "All Sparse LDL^T tests passed!" << std::endl;
    return 0;
}
