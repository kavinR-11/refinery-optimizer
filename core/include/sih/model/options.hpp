#pragma once

#include <cstdint>
#include <string>
#include <limits>

namespace sih {
namespace model {

enum class AlgorithmChoice {
    Auto = 0,
    PrimalSimplex = 1,
    DualSimplex = 2,
    Barrier = 3,
    BranchAndBound = 4
};

enum class PricingRule {
    Dantzig = 0,
    SteepestEdge = 1,
    Devex = 2,
    BFRT = 3
};

enum class BranchingRule {
    MostFractional = 0,
    PseudoCost = 1,
    StrongBranching = 2,
    Reliability = 3
};

enum class NodeSelection {
    BestBound = 0,
    DepthFirst = 1,
    BestEstimate = 2
};

enum class PresolveMode {
    Off = 0,
    On = 1,
    Aggressive = 2
};

enum class RatioTest {
    Textbook = 0,
    Harris = 1,
    HarrisBFRT = 2
};

struct StrategyConfig {
    AlgorithmChoice algorithm{AlgorithmChoice::Auto};
    PricingRule pricing_rule{PricingRule::SteepestEdge};
    BranchingRule branching_rule{BranchingRule::PseudoCost};
    NodeSelection node_selection{NodeSelection::BestBound};
    PresolveMode presolve{PresolveMode::On};
    int max_presolve_passes{10};
    RatioTest ratio_test{RatioTest::HarrisBFRT};
    bool enable_scaling{true};
    bool power_of_two_scaling{true};
    int refactor_frequency{60};
    bool enable_perturbation{true};
    double perturbation_magnitude{1e-11};
    int cut_rounds{5};
    bool enable_gpu{false};
};

struct Options {
    // Resource limits
    double time_limit_sec{std::numeric_limits<double>::infinity()};
    int64_t iteration_limit{-1}; // -1 means no limit
    int64_t node_limit{-1};      // -1 means no limit
    int threads{0};              // 0 means auto-detect

    // Numerical tolerances
    double primal_feasibility_tol{1e-6};
    double dual_feasibility_tol{1e-6};
    double integrality_tol{1e-5};
    double zero_tol{1e-12};

    // Logging & Telemetry
    bool log_to_console{true};
    std::string telemetry_log_path{"telemetry.jsonl"};

    // Adaptive strategy configuration
    StrategyConfig strategy;
};

// String conversion helpers
std::string algorithm_to_string(AlgorithmChoice alg);
AlgorithmChoice string_to_algorithm(const std::string& str);

std::string pricing_to_string(PricingRule rule);
PricingRule string_to_pricing(const std::string& str);

std::string branching_to_string(BranchingRule rule);
BranchingRule string_to_branching(const std::string& str);

std::string node_selection_to_string(NodeSelection node);
NodeSelection string_to_node_selection(const std::string& str);

std::string presolve_to_string(PresolveMode mode);
PresolveMode string_to_presolve(const std::string& str);

} // namespace model
} // namespace sih
