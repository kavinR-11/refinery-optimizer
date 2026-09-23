#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include <iostream>
#include <cassert>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_solution_options..." << std::endl;

    // Test Solution
    sih::model::Solution sol;
    sol.status = sih::model::SolutionStatus::Optimal;
    sol.x = {1.0, 2.0, 3.0};
    sol.primal_objective = 14.0;
    sol.dual_bound = 14.0;
    sol.mip_gap = 0.0;
    sol.simplex_iterations = 25;
    sol.time_wall_sec = 0.012;

    ASSERT(sol.is_optimal());
    ASSERT(sol.is_feasible());
    ASSERT(sih::model::status_to_string(sol.status) == "Optimal");
    ASSERT(sih::model::string_to_status("Optimal") == sih::model::SolutionStatus::Optimal);
    ASSERT(sih::model::string_to_status("Infeasible") == sih::model::SolutionStatus::Infeasible);

    // Test Options & StrategyConfig
    sih::model::Options opt;
    opt.time_limit_sec = 60.0;
    opt.threads = 4;
    opt.strategy.algorithm = sih::model::AlgorithmChoice::DualSimplex;
    opt.strategy.pricing_rule = sih::model::PricingRule::Devex;
    opt.strategy.branching_rule = sih::model::BranchingRule::StrongBranching;
    opt.strategy.node_selection = sih::model::NodeSelection::BestBound;
    opt.strategy.presolve = sih::model::PresolveMode::Aggressive;
    opt.strategy.cut_rounds = 10;
    opt.strategy.enable_gpu = false;

    ASSERT(sih::model::algorithm_to_string(opt.strategy.algorithm) == "DualSimplex");
    ASSERT(sih::model::pricing_to_string(opt.strategy.pricing_rule) == "Devex");
    ASSERT(sih::model::branching_to_string(opt.strategy.branching_rule) == "StrongBranching");
    ASSERT(sih::model::node_selection_to_string(opt.strategy.node_selection) == "BestBound");
    ASSERT(sih::model::presolve_to_string(opt.strategy.presolve) == "Aggressive");

    ASSERT(sih::model::string_to_algorithm("DualSimplex") == sih::model::AlgorithmChoice::DualSimplex);
    ASSERT(sih::model::string_to_pricing("Devex") == sih::model::PricingRule::Devex);

    std::cout << "test_solution_options: PASS" << std::endl;
    return 0;
}
