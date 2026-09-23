#include "sih/model/problem.hpp"
#include <iostream>
#include <cassert>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_problem_model..." << std::endl;

    sih::model::Problem p("test_lp");
    p.resize(2, 3);
    p.set_sense(sih::model::ObjectiveSense::Minimize);
    p.set_c({1.0, 2.0, 3.0});
    p.set_row_names({"r1", "r2"});
    p.set_col_names({"x1", "x2", "x3"});

    p.row_lower()[0] = 0.0;
    p.row_upper()[0] = 10.0;
    p.row_lower()[1] = 5.0;
    p.row_upper()[1] = 5.0; // equality

    p.col_lower()[0] = 0.0;
    p.col_upper()[0] = 1.0;
    p.var_types()[0] = sih::model::VariableType::Binary;

    p.col_lower()[1] = 0.0;
    p.col_upper()[1] = 10.0;
    p.var_types()[1] = sih::model::VariableType::Integer;

    std::vector<sih::model::Triplet> A_trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 2.0}, {1, 2, 1.0}
    };
    p.set_A(sih::model::SparseMatrix::from_triplets(2, 3, A_trips));

    ASSERT(p.name() == "test_lp");
    ASSERT(p.num_rows() == 2);
    ASSERT(p.num_cols() == 3);
    ASSERT(p.num_nonzeros() == 4);
    ASSERT(p.num_integers() == 2);
    ASSERT(p.num_binaries() == 1);
    ASSERT(p.is_mip());
    ASSERT(!p.is_qp());

    // Check name lookup
    ASSERT(p.get_row_index("r1") == 0);
    ASSERT(p.get_row_index("r2") == 1);
    ASSERT(p.get_col_index("x1") == 0);
    ASSERT(p.get_col_index("x3") == 2);

    // Validate passes
    p.validate();

    // Test quadratic term
    std::vector<sih::model::Triplet> Q_trips = {
        {0, 0, 2.0}, {1, 1, 4.0}, {2, 2, 6.0}
    };
    p.set_quadratic_objective(sih::model::SparseMatrix::from_triplets(3, 3, Q_trips));
    ASSERT(p.is_qp());
    ASSERT(p.num_quad_nonzeros() == 3);

    p.validate();

    std::cout << "test_problem_model: PASS" << std::endl;
    return 0;
}
