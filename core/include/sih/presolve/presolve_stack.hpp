#pragma once

#include "sih/model/solution.hpp"
#include <vector>
#include <cstdint>

namespace sih {
namespace presolve {

enum class ReductionType {
    EmptyRow,
    EmptyCol,
    FixedCol,
    RowSingleton,
    FreeColSingleton,
    ForcingRow,
    RedundantRow,
    TightenedColBounds,
    DuplicateRow,
    DominatedCol
};

struct PresolveUndo {
    ReductionType type;
    int64_t row_idx{-1};
    int64_t col_idx{-1};
    double old_row_lower{0.0};
    double old_row_upper{0.0};
    double old_col_lower{0.0};
    double old_col_upper{0.0};
    double coeff{0.0};
    double obj_coeff{0.0};

    // For rows or substitutions with multiple participating terms
    std::vector<int64_t> sparse_indices;
    std::vector<double> sparse_values;
};

class PresolveStack {
public:
    void push(PresolveUndo undo) { m_stack.push_back(std::move(undo)); }
    bool empty() const noexcept { return m_stack.empty(); }
    size_t size() const noexcept { return m_stack.size(); }
    const std::vector<PresolveUndo>& entries() const noexcept { return m_stack; }
    void clear() { m_stack.clear(); }

private:
    std::vector<PresolveUndo> m_stack;
};

} // namespace presolve
} // namespace sih
