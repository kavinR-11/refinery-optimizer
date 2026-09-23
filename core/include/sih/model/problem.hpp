#pragma once

#include "sih/model/sparse_matrix.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <optional>

namespace sih {
namespace model {

enum class ObjectiveSense {
    Minimize = 1,
    Maximize = -1
};

enum class VariableType {
    Continuous = 0,
    Binary = 1,
    Integer = 2,
    SemiContinuous = 3
};

inline constexpr double SIH_INFINITY = std::numeric_limits<double>::infinity();

class Problem {
public:
    Problem() = default;
    explicit Problem(std::string name) : m_name(std::move(name)) {}

    // Dimensions
    int64_t num_rows() const noexcept { return m_num_rows; }
    int64_t num_cols() const noexcept { return m_num_cols; }
    int64_t num_nonzeros() const noexcept { return m_A.num_nonzeros(); }
    int64_t num_quad_nonzeros() const noexcept { return m_has_quadratic ? m_Q.num_nonzeros() : 0; }

    int64_t num_integers() const noexcept;
    int64_t num_binaries() const noexcept;
    bool is_mip() const noexcept;
    bool is_qp() const noexcept { return m_has_quadratic && m_Q.num_nonzeros() > 0; }

    // Problem attributes
    const std::string& name() const noexcept { return m_name; }
    void set_name(std::string name) { m_name = std::move(name); }

    ObjectiveSense sense() const noexcept { return m_sense; }
    void set_sense(ObjectiveSense sense) noexcept { m_sense = sense; }

    double obj_offset() const noexcept { return m_obj_offset; }
    void set_obj_offset(double offset) noexcept { m_obj_offset = offset; }

    // Objective vector c
    const std::vector<double>& c() const noexcept { return m_c; }
    std::vector<double>& c() noexcept { return m_c; }
    void set_c(std::vector<double> c) { m_c = std::move(c); }

    // Constraint matrix A
    const SparseMatrix& A() const noexcept { return m_A; }
    void set_A(SparseMatrix A);

    // Bounds
    const std::vector<double>& row_lower() const noexcept { return m_row_lower; }
    std::vector<double>& row_lower() noexcept { return m_row_lower; }

    const std::vector<double>& row_upper() const noexcept { return m_row_upper; }
    std::vector<double>& row_upper() noexcept { return m_row_upper; }

    const std::vector<double>& col_lower() const noexcept { return m_col_lower; }
    std::vector<double>& col_lower() noexcept { return m_col_lower; }

    const std::vector<double>& col_upper() const noexcept { return m_col_upper; }
    std::vector<double>& col_upper() noexcept { return m_col_upper; }

    // Variable types
    const std::vector<VariableType>& var_types() const noexcept { return m_var_types; }
    std::vector<VariableType>& var_types() noexcept { return m_var_types; }

    // Quadratic objective Q (symmetric, 0.5 * x^T Q x)
    bool has_quadratic() const noexcept { return m_has_quadratic; }
    const SparseMatrix& Q() const noexcept { return m_Q; }
    void set_quadratic_objective(SparseMatrix Q);
    void clear_quadratic_objective();

    // Names
    const std::vector<std::string>& row_names() const noexcept { return m_row_names; }
    const std::vector<std::string>& col_names() const noexcept { return m_col_names; }
    void set_row_names(std::vector<std::string> names);
    void set_col_names(std::vector<std::string> names);

    int64_t get_row_index(const std::string& name) const;
    int64_t get_col_index(const std::string& name) const;
    bool has_row(const std::string& name) const;
    bool has_col(const std::string& name) const;

    // Resizing & Initialization
    void resize(int64_t num_rows, int64_t num_cols);

    // Validation
    void validate() const;

private:
    std::string m_name{"problem"};
    ObjectiveSense m_sense{ObjectiveSense::Minimize};
    double m_obj_offset{0.0};

    int64_t m_num_rows{0};
    int64_t m_num_cols{0};

    std::vector<double> m_c;
    SparseMatrix m_A;

    std::vector<double> m_row_lower;
    std::vector<double> m_row_upper;
    std::vector<double> m_col_lower;
    std::vector<double> m_col_upper;

    std::vector<VariableType> m_var_types;

    bool m_has_quadratic{false};
    SparseMatrix m_Q;

    std::vector<std::string> m_row_names;
    std::vector<std::string> m_col_names;
    std::unordered_map<std::string, int64_t> m_row_name_to_idx;
    std::unordered_map<std::string, int64_t> m_col_name_to_idx;
};

} // namespace model
} // namespace sih
