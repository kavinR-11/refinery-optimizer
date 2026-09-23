#include "sih/model/problem.hpp"
#include <algorithm>
#include <sstream>

namespace sih {
namespace model {

void Problem::resize(int64_t num_rows, int64_t num_cols) {
    m_num_rows = num_rows;
    m_num_cols = num_cols;

    m_c.assign(num_cols, 0.0);
    m_row_lower.assign(num_rows, -SIH_INFINITY);
    m_row_upper.assign(num_rows, SIH_INFINITY);
    m_col_lower.assign(num_cols, 0.0); // Standard LP default: x >= 0
    m_col_upper.assign(num_cols, SIH_INFINITY);
    m_var_types.assign(num_cols, VariableType::Continuous);

    m_row_names.clear();
    m_col_names.clear();
    m_row_name_to_idx.clear();
    m_col_name_to_idx.clear();
}

void Problem::set_A(SparseMatrix A) {
    if (A.num_rows() != m_num_rows || A.num_cols() != m_num_cols) {
        throw std::invalid_argument("Constraint matrix A dimensions do not match problem dimensions");
    }
    m_A = std::move(A);
}

void Problem::set_quadratic_objective(SparseMatrix Q) {
    if (Q.num_rows() != m_num_cols || Q.num_cols() != m_num_cols) {
        throw std::invalid_argument("Quadratic matrix Q dimensions must be n x n");
    }
    m_Q = std::move(Q);
    m_has_quadratic = true;
}

void Problem::clear_quadratic_objective() {
    m_has_quadratic = false;
    m_Q = SparseMatrix(0, 0);
}

void Problem::set_row_names(std::vector<std::string> names) {
    if (static_cast<int64_t>(names.size()) != m_num_rows) {
        throw std::invalid_argument("Row names count does not match number of rows");
    }
    m_row_names = std::move(names);
    m_row_name_to_idx.clear();
    m_row_name_to_idx.reserve(m_num_rows);
    for (int64_t i = 0; i < m_num_rows; ++i) {
        m_row_name_to_idx[m_row_names[i]] = i;
    }
}

void Problem::set_col_names(std::vector<std::string> names) {
    if (static_cast<int64_t>(names.size()) != m_num_cols) {
        throw std::invalid_argument("Column names count does not match number of columns");
    }
    m_col_names = std::move(names);
    m_col_name_to_idx.clear();
    m_col_name_to_idx.reserve(m_num_cols);
    for (int64_t j = 0; j < m_num_cols; ++j) {
        m_col_name_to_idx[m_col_names[j]] = j;
    }
}

int64_t Problem::get_row_index(const std::string& name) const {
    auto it = m_row_name_to_idx.find(name);
    if (it == m_row_name_to_idx.end()) {
        throw std::out_of_range("Row name not found: " + name);
    }
    return it->second;
}

int64_t Problem::get_col_index(const std::string& name) const {
    auto it = m_col_name_to_idx.find(name);
    if (it == m_col_name_to_idx.end()) {
        throw std::out_of_range("Column name not found: " + name);
    }
    return it->second;
}

bool Problem::has_row(const std::string& name) const {
    return m_row_name_to_idx.find(name) != m_row_name_to_idx.end();
}

bool Problem::has_col(const std::string& name) const {
    return m_col_name_to_idx.find(name) != m_col_name_to_idx.end();
}

int64_t Problem::num_integers() const noexcept {
    int64_t cnt = 0;
    for (auto vt : m_var_types) {
        if (vt == VariableType::Integer || vt == VariableType::Binary) {
            cnt++;
        }
    }
    return cnt;
}

int64_t Problem::num_binaries() const noexcept {
    int64_t cnt = 0;
    for (auto vt : m_var_types) {
        if (vt == VariableType::Binary) {
            cnt++;
        }
    }
    return cnt;
}

bool Problem::is_mip() const noexcept {
    return num_integers() > 0;
}

void Problem::validate() const {
    if (m_num_rows < 0 || m_num_cols < 0) {
        throw std::runtime_error("Problem has negative dimensions");
    }
    if (static_cast<int64_t>(m_c.size()) != m_num_cols) {
        throw std::runtime_error("Objective vector c size mismatch");
    }
    if (static_cast<int64_t>(m_row_lower.size()) != m_num_rows ||
        static_cast<int64_t>(m_row_upper.size()) != m_num_rows) {
        throw std::runtime_error("Row bounds size mismatch");
    }
    if (static_cast<int64_t>(m_col_lower.size()) != m_num_cols ||
        static_cast<int64_t>(m_col_upper.size()) != m_num_cols) {
        throw std::runtime_error("Column bounds size mismatch");
    }
    if (static_cast<int64_t>(m_var_types.size()) != m_num_cols) {
        throw std::runtime_error("Variable types size mismatch");
    }

    for (int64_t i = 0; i < m_num_rows; ++i) {
        if (m_row_lower[i] > m_row_upper[i]) {
            std::ostringstream ss;
            ss << "Contradictory row bounds at row " << i << ": lower=" << m_row_lower[i] << " > upper=" << m_row_upper[i];
            throw std::runtime_error(ss.str());
        }
    }

    for (int64_t j = 0; j < m_num_cols; ++j) {
        if (m_col_lower[j] > m_col_upper[j]) {
            std::ostringstream ss;
            ss << "Contradictory column bounds at column " << j << ": lower=" << m_col_lower[j] << " > upper=" << m_col_upper[j];
            throw std::runtime_error(ss.str());
        }
    }

    if (m_A.num_rows() != m_num_rows || m_A.num_cols() != m_num_cols) {
        throw std::runtime_error("Constraint matrix A dimensions mismatch");
    }

    if (m_has_quadratic) {
        if (m_Q.num_rows() != m_num_cols || m_Q.num_cols() != m_num_cols) {
            throw std::runtime_error("Quadratic matrix Q dimensions mismatch");
        }
    }
}

} // namespace model
} // namespace sih
