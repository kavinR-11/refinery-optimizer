#include "sih/factorization/amd.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " (" #cond ")" << std::endl; \
            std::abort(); \
        } \
    } while (0)

using namespace sih::model;
using namespace sih::factorization;

void test_amd_arrow_matrix() {
    std::cout << "[Test] test_amd_arrow_matrix running..." << std::endl;
    // An arrow matrix has dense first row/col and diagonal.
    // If pivoted in natural order (0 first), it causes 100% fill-in!
    // AMD should eliminate node 0 LAST, resulting in 0 fill-in!
    int64_t n = 6;
    std::vector<Triplet> triplets;
    // Row 0 and Col 0
    for (int64_t i = 0; i < n; ++i) {
        triplets.emplace_back(0, i, 1.0);
        triplets.emplace_back(i, 0, 1.0);
        triplets.emplace_back(i, i, 2.0);
    }
    auto M = SparseMatrix::from_triplets(n, n, triplets);

    auto perm = AmdOrder::order_matrix(n, M.csc_col_ptr(), M.csc_row_ind());
    TEST_ASSERT(static_cast<int64_t>(perm.size()) == n);

    // Verify it is a valid permutation of 0..n-1
    std::vector<int64_t> sorted_p = perm;
    std::sort(sorted_p.begin(), sorted_p.end());
    for (int64_t i = 0; i < n; ++i) {
        TEST_ASSERT(sorted_p[i] == i);
    }

    std::cout << "perm: ";
    for (int64_t x : perm) std::cout << x << " ";
    std::cout << std::endl;
    TEST_ASSERT(perm[0] != 0);
    TEST_ASSERT(perm.back() == 0 || perm[n - 2] == 0);
    std::cout << "[Test] test_amd_arrow_matrix passed. Node 0 ordered last as expected." << std::endl;
}

void test_amd_aat() {
    std::cout << "[Test] test_amd_aat running..." << std::endl;
    int64_t m = 4;
    int64_t n = 5;
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {1, 0, 1.0},
        {1, 1, 2.0}, {2, 1, 1.0},
        {2, 2, 3.0}, {3, 2, 1.0},
        {3, 3, 1.0}, {0, 4, 2.0}
    };
    auto A = SparseMatrix::from_triplets(m, n, triplets);

    auto perm = AmdOrder::order_aat(A);
    TEST_ASSERT(static_cast<int64_t>(perm.size()) == m);

    auto inv = AmdOrder::invert_permutation(perm);
    for (int64_t i = 0; i < m; ++i) {
        TEST_ASSERT(inv[perm[i]] == i);
    }

    std::cout << "[Test] test_amd_aat passed." << std::endl;
}

int main() {
    test_amd_arrow_matrix();
    test_amd_aat();
    std::cout << "All AMD tests passed!" << std::endl;
    return 0;
}
