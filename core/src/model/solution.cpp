#include "sih/model/solution.hpp"
#include <stdexcept>

namespace sih {
namespace model {

std::string status_to_string(SolutionStatus status) {
    switch (status) {
        case SolutionStatus::Optimal: return "Optimal";
        case SolutionStatus::Infeasible: return "Infeasible";
        case SolutionStatus::Unbounded: return "Unbounded";
        case SolutionStatus::TimeLimit: return "TimeLimit";
        case SolutionStatus::IterationLimit: return "IterationLimit";
        case SolutionStatus::NodeLimit: return "NodeLimit";
        case SolutionStatus::NumericalFailure: return "NumericalFailure";
        case SolutionStatus::Interrupted: return "Interrupted";
        case SolutionStatus::Unknown: default: return "Unknown";
    }
}

SolutionStatus string_to_status(const std::string& str) {
    if (str == "Optimal") return SolutionStatus::Optimal;
    if (str == "Infeasible") return SolutionStatus::Infeasible;
    if (str == "Unbounded") return SolutionStatus::Unbounded;
    if (str == "TimeLimit") return SolutionStatus::TimeLimit;
    if (str == "IterationLimit") return SolutionStatus::IterationLimit;
    if (str == "NodeLimit") return SolutionStatus::NodeLimit;
    if (str == "NumericalFailure") return SolutionStatus::NumericalFailure;
    if (str == "Interrupted") return SolutionStatus::Interrupted;
    return SolutionStatus::Unknown;
}

std::string basis_to_string(BasisStatus status) {
    switch (status) {
        case BasisStatus::Basic: return "Basic";
        case BasisStatus::AtLower: return "AtLower";
        case BasisStatus::AtUpper: return "AtUpper";
        case BasisStatus::Free: return "Free";
        case BasisStatus::Superbasic: return "Superbasic";
        default: return "Unknown";
    }
}

BasisStatus string_to_basis(const std::string& str) {
    if (str == "Basic") return BasisStatus::Basic;
    if (str == "AtLower") return BasisStatus::AtLower;
    if (str == "AtUpper") return BasisStatus::AtUpper;
    if (str == "Free") return BasisStatus::Free;
    if (str == "Superbasic") return BasisStatus::Superbasic;
    throw std::invalid_argument("Unknown basis status string: " + str);
}

} // namespace model
} // namespace sih
