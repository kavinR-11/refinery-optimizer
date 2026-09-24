#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <limits>

namespace sih {
namespace model {

enum class SolutionStatus {
    Optimal = 0,
    Infeasible = 1,
    Unbounded = 2,
    TimeLimit = 3,
    IterationLimit = 4,
    NodeLimit = 5,
    NumericalFailure = 6,
    Interrupted = 7,
    Unknown = 8
};

enum class BasisStatus {
    Basic = 0,
    AtLower = 1,
    AtUpper = 2,
    Free = 3,
    Superbasic = 4
};

std::string status_to_string(SolutionStatus status);
SolutionStatus string_to_status(const std::string& str);

std::string basis_to_string(BasisStatus status);
BasisStatus string_to_basis(const std::string& str);

struct Solution {
    SolutionStatus status{SolutionStatus::Unknown};

    // Primal vectors
    std::vector<double> x;
    std::vector<double> slack; // row activity Ax

    // Dual vectors
    std::vector<double> row_duals;     // Lagrange multipliers y
    std::vector<double> reduced_costs; // Dual slacks s = c + Qx - A^T y

    // Basis status
    std::vector<BasisStatus> col_basis;
    std::vector<BasisStatus> row_basis;

    // Certificate ray (Farkas ray for Infeasible, Unbounded ray for Unbounded)
    std::vector<double> ray;

    // Sensitivity ranges
    std::vector<double> rhs_down;
    std::vector<double> rhs_up;
    std::vector<double> obj_down;
    std::vector<double> obj_up;

    // Objectives and bounds
    double primal_objective{0.0};
    double dual_bound{0.0};
    double mip_gap{0.0};

    // Counters
    int64_t simplex_iterations{0};
    int64_t barrier_iterations{0};
    int64_t nodes_explored{0};

    // Timings in seconds
    double time_wall_sec{0.0};
    double time_cpu_sec{0.0};
    double time_presolve_sec{0.0};
    double time_solver_sec{0.0};

    // Phase 5: Adaptive in-solve monitoring metrics
    int strategy_switches{0};
    std::string in_solve_log{""};

    // Check if solution claims optimality
    bool is_optimal() const noexcept { return status == SolutionStatus::Optimal; }
    bool is_feasible() const noexcept { return status == SolutionStatus::Optimal || status == SolutionStatus::TimeLimit || status == SolutionStatus::IterationLimit || status == SolutionStatus::NodeLimit; }
};

} // namespace model
} // namespace sih
