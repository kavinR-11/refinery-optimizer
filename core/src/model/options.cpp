#include "sih/model/options.hpp"
#include <stdexcept>

namespace sih {
namespace model {

std::string algorithm_to_string(AlgorithmChoice alg) {
    switch (alg) {
        case AlgorithmChoice::Auto: return "Auto";
        case AlgorithmChoice::PrimalSimplex: return "PrimalSimplex";
        case AlgorithmChoice::DualSimplex: return "DualSimplex";
        case AlgorithmChoice::Barrier: return "Barrier";
        case AlgorithmChoice::BranchAndBound: return "BranchAndBound";
        default: return "Auto";
    }
}

AlgorithmChoice string_to_algorithm(const std::string& str) {
    if (str == "Auto") return AlgorithmChoice::Auto;
    if (str == "PrimalSimplex") return AlgorithmChoice::PrimalSimplex;
    if (str == "DualSimplex") return AlgorithmChoice::DualSimplex;
    if (str == "Barrier") return AlgorithmChoice::Barrier;
    if (str == "BranchAndBound") return AlgorithmChoice::BranchAndBound;
    throw std::invalid_argument("Unknown algorithm choice string: " + str);
}

std::string pricing_to_string(PricingRule rule) {
    switch (rule) {
        case PricingRule::Dantzig: return "Dantzig";
        case PricingRule::SteepestEdge: return "SteepestEdge";
        case PricingRule::Devex: return "Devex";
        case PricingRule::BFRT: return "BFRT";
        default: return "SteepestEdge";
    }
}

PricingRule string_to_pricing(const std::string& str) {
    if (str == "Dantzig") return PricingRule::Dantzig;
    if (str == "SteepestEdge") return PricingRule::SteepestEdge;
    if (str == "Devex") return PricingRule::Devex;
    if (str == "BFRT") return PricingRule::BFRT;
    throw std::invalid_argument("Unknown pricing rule string: " + str);
}

std::string branching_to_string(BranchingRule rule) {
    switch (rule) {
        case BranchingRule::MostFractional: return "MostFractional";
        case BranchingRule::PseudoCost: return "PseudoCost";
        case BranchingRule::StrongBranching: return "StrongBranching";
        case BranchingRule::Reliability: return "Reliability";
        default: return "PseudoCost";
    }
}

BranchingRule string_to_branching(const std::string& str) {
    if (str == "MostFractional") return BranchingRule::MostFractional;
    if (str == "PseudoCost") return BranchingRule::PseudoCost;
    if (str == "StrongBranching") return BranchingRule::StrongBranching;
    if (str == "Reliability") return BranchingRule::Reliability;
    throw std::invalid_argument("Unknown branching rule string: " + str);
}

std::string node_selection_to_string(NodeSelection node) {
    switch (node) {
        case NodeSelection::BestBound: return "BestBound";
        case NodeSelection::DepthFirst: return "DepthFirst";
        case NodeSelection::BestEstimate: return "BestEstimate";
        default: return "BestBound";
    }
}

NodeSelection string_to_node_selection(const std::string& str) {
    if (str == "BestBound") return NodeSelection::BestBound;
    if (str == "DepthFirst") return NodeSelection::DepthFirst;
    if (str == "BestEstimate") return NodeSelection::BestEstimate;
    throw std::invalid_argument("Unknown node selection string: " + str);
}

std::string presolve_to_string(PresolveMode mode) {
    switch (mode) {
        case PresolveMode::Off: return "Off";
        case PresolveMode::On: return "On";
        case PresolveMode::Aggressive: return "Aggressive";
        default: return "On";
    }
}

PresolveMode string_to_presolve(const std::string& str) {
    if (str == "Off") return PresolveMode::Off;
    if (str == "On") return PresolveMode::On;
    if (str == "Aggressive") return PresolveMode::Aggressive;
    throw std::invalid_argument("Unknown presolve mode string: " + str);
}

} // namespace model
} // namespace sih
