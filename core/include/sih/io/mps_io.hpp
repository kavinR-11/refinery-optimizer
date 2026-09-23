#pragma once

#include "sih/model/problem.hpp"
#include <string>

namespace sih {
namespace io {

// Read MPS or QPS file (auto-detects fixed/free format, bounds, ranges, quadobj/qmatrix, integer markers)
model::Problem read_mps(const std::string& filepath);

// Write problem to MPS / QPS file
void write_mps(const model::Problem& problem, const std::string& filepath, bool free_format = true);

} // namespace io
} // namespace sih
