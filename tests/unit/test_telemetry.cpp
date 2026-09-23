#include "sih/telemetry/telemetry.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_telemetry..." << std::endl;

    sih::model::Problem p("telemetry_test");
    p.resize(1, 2);
    p.set_c({1.0, 2.0});
    std::vector<sih::model::Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    p.set_A(sih::model::SparseMatrix::from_triplets(1, 2, trips));

    sih::model::Solution sol;
    sol.status = sih::model::SolutionStatus::Optimal;
    sol.x = {1.0, 1.0};
    sol.primal_objective = 3.0;
    sol.simplex_iterations = 10;
    sol.time_wall_sec = 0.005;

    sih::model::Options opt;
    opt.strategy.algorithm = sih::model::AlgorithmChoice::PrimalSimplex;

    std::string json = sih::telemetry::serialize_solve_telemetry(p, sol, opt, "test-run-12345");
    std::cout << "Serialized telemetry JSON:" << std::endl << json << std::endl;

    ASSERT(json.find("\"run_id\":\"test-run-12345\"") != std::string::npos);
    ASSERT(json.find("\"name\":\"telemetry_test\"") != std::string::npos);
    ASSERT(json.find("\"status\":\"Optimal\"") != std::string::npos);
    ASSERT(json.find("\"primal_objective\":3") != std::string::npos);
    ASSERT(json.find("\"algorithm\":\"PrimalSimplex\"") != std::string::npos);

    // Test file logging
    std::string log_file = "/tmp/test_telemetry.jsonl";
    bool logged = sih::telemetry::log_solve_telemetry(p, sol, opt, log_file, "run-file-999");
    ASSERT(logged);

    std::ifstream in(log_file);
    ASSERT(in.is_open());
    std::string line;
    std::getline(in, line);
    ASSERT(line.find("run-file-999") != std::string::npos);

    std::cout << "test_telemetry: PASS" << std::endl;
    return 0;
}
