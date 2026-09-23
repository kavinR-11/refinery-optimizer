#include "sih/model/sparse_matrix.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_sparse_matrix..." << std::endl;

    // Test matrix:
    // [ 1.0  0.0  2.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 4.0  5.0  6.0 ]
    std::vector<sih::model::Triplet> trips = {
        {0, 0, 1.0},
        {0, 2, 2.0},
        {1, 1, 3.0},
        {2, 0, 4.0},
        {2, 1, 5.0},
        {2, 2, 6.0}
    };

    auto mat = sih::model::SparseMatrix::from_triplets(3, 3, trips);
    ASSERT(mat.is_valid());
    ASSERT(mat.num_rows() == 3);
    ASSERT(mat.num_cols() == 3);
    ASSERT(mat.num_nonzeros() == 6);

    // Test get()
    ASSERT(std::abs(mat.get(0, 0) - 1.0) < 1e-9);
    ASSERT(std::abs(mat.get(0, 1) - 0.0) < 1e-9);
    ASSERT(std::abs(mat.get(0, 2) - 2.0) < 1e-9);
    ASSERT(std::abs(mat.get(1, 1) - 3.0) < 1e-9);
    ASSERT(std::abs(mat.get(2, 2) - 6.0) < 1e-9);

    // Test mat_vec: y = A * x
    // x = [1, 2, 3]^T
    // y = [1*1 + 2*3, 3*2, 4*1 + 5*2 + 6*3]^T = [7, 6, 32]^T
    std::vector<double> x = {1.0, 2.0, 3.0};
    auto y = mat.mat_vec(x);
    ASSERT(y.size() == 3);
    ASSERT(std::abs(y[0] - 7.0) < 1e-9);
    ASSERT(std::abs(y[1] - 6.0) < 1e-9);
    ASSERT(std::abs(y[2] - 32.0) < 1e-9);

    // Test mat_trans_vec: z = A^T * y_vec
    // y_vec = [1, 0, 1]^T
    // A^T =
    // [ 1.0  0.0  4.0 ]
    // [ 0.0  3.0  5.0 ]
    // [ 2.0  0.0  6.0 ]
    // z = [1*1 + 4*1, 5*1, 2*1 + 6*1]^T = [5, 5, 8]^T
    std::vector<double> y_vec = {1.0, 0.0, 1.0};
    auto z = mat.mat_trans_vec(y_vec);
    ASSERT(z.size() == 3);
    ASSERT(std::abs(z[0] - 5.0) < 1e-9);
    ASSERT(std::abs(z[1] - 5.0) < 1e-9);
    ASSERT(std::abs(z[2] - 8.0) < 1e-9);

    // Test transpose
    auto trans = mat.transpose();
    ASSERT(trans.is_valid());
    ASSERT(trans.num_rows() == 3);
    ASSERT(trans.num_cols() == 3);
    ASSERT(trans.num_nonzeros() == 6);
    ASSERT(std::abs(trans.get(0, 2) - 4.0) < 1e-9);
    ASSERT(std::abs(trans.get(2, 0) - 2.0) < 1e-9);

    std::cout << "test_sparse_matrix: PASS" << std::endl;
    return 0;
}
