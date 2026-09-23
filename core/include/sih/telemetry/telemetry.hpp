#pragma once

#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include <string>

namespace sih {
namespace telemetry {

// Memory usage helper: queries resident set size (RSS) in megabytes
double get_resident_memory_mb();

// Generate single-line JSON string representing the solve event
std::string serialize_solve_telemetry(const model::Problem& problem,
                                     const model::Solution& solution,
                                     const model::Options& options,
                                     const std::string& run_id = "");

// Append single-line JSON to the specified log file
bool log_solve_telemetry(const model::Problem& problem,
                         const model::Solution& solution,
                         const model::Options& options,
                         const std::string& file_path = "",
                         const std::string& run_id = "");

} // namespace telemetry
} // namespace sih
