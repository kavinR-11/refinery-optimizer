#include "sih/io/mps_io.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_mps_io..." << std::endl;

    // 1. Create a synthetic problem with LP, bounds, integer, and quadratic term
    sih::model::Problem p("roundtrip_test");
    p.resize(2, 3);
    p.set_sense(sih::model::ObjectiveSense::Minimize);
    p.set_c({1.5, -2.0, 3.25});
    p.set_row_names({"con1", "con2"});
    p.set_col_names({"col1", "col2", "col3"});

    p.row_lower()[0] = -sih::model::SIH_INFINITY;
    p.row_upper()[0] = 10.0; // L row

    p.row_lower()[1] = 5.0;  // E row
    p.row_upper()[1] = 5.0;

    p.col_lower()[0] = 0.0;
    p.col_upper()[0] = 1.0;
    p.var_types()[0] = sih::model::VariableType::Binary;

    p.col_lower()[1] = 1.0;
    p.col_upper()[1] = 20.0;
    p.var_types()[1] = sih::model::VariableType::Integer;

    p.col_lower()[2] = -sih::model::SIH_INFINITY;
    p.col_upper()[2] = sih::model::SIH_INFINITY; // FR

    std::vector<sih::model::Triplet> A_trips = {
        {0, 0, 1.0}, {0, 1, 2.5},
        {1, 1, -1.0}, {1, 2, 4.0}
    };
    p.set_A(sih::model::SparseMatrix::from_triplets(2, 3, A_trips));

    std::vector<sih::model::Triplet> Q_trips = {
        {0, 0, 2.0}, {1, 1, 4.0}
    };
    p.set_quadratic_objective(sih::model::SparseMatrix::from_triplets(3, 3, Q_trips));

    // Write to temporary file
    std::string tmp_file = "/tmp/test_roundtrip.mps";
    sih::io::write_mps(p, tmp_file);

    // Read back
    sih::model::Problem p_read = sih::io::read_mps(tmp_file);
    ASSERT(p_read.name() == "roundtrip_test");
    ASSERT(p_read.num_rows() == 2);
    ASSERT(p_read.num_cols() == 3);
    ASSERT(p_read.num_nonzeros() == 4);
    ASSERT(p_read.is_mip());
    ASSERT(p_read.is_qp());

    ASSERT(std::abs(p_read.c()[0] - 1.5) < 1e-6);
    ASSERT(std::abs(p_read.c()[1] - (-2.0)) < 1e-6);
    ASSERT(std::abs(p_read.c()[2] - 3.25) < 1e-6);

    // Check quadratic
    ASSERT(std::abs(p_read.Q().get(0, 0) - 2.0) < 1e-6);
    ASSERT(std::abs(p_read.Q().get(1, 1) - 4.0) < 1e-6);

    // 2. Test reading real Netlib instance afiro if available
    std::string netlib_afiro = "data/netlib/afiro.mps";
    if (std::filesystem::exists(netlib_afiro)) {
        std::cout << "Testing parse of Netlib afiro.mps..." << std::endl;
        sih::model::Problem afiro = sih::io::read_mps(netlib_afiro);
        std::cout << "  Afiro rows: " << afiro.num_rows()
                  << ", cols: " << afiro.num_cols()
                  << ", nonzeros: " << afiro.num_nonzeros() << std::endl;
        ASSERT(afiro.num_rows() == 27);
        ASSERT(afiro.num_cols() == 32);
        ASSERT(afiro.num_nonzeros() == 83);
    }

    std::cout << "test_mps_io: PASS" << std::endl;
    return 0;
}
