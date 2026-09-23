#include "sih/io/mps_io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <iomanip>

namespace sih {
namespace io {

namespace {

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::string cur;
    bool in_quote = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\'') {
            in_quote = !in_quote;
            cur.push_back(c);
        } else if (std::isspace(static_cast<unsigned char>(c)) && !in_quote) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        tokens.push_back(cur);
    }
    return tokens;
}

enum class Section {
    NONE,
    NAME,
    OBJSENSE,
    ROWS,
    COLUMNS,
    RHS,
    RANGES,
    BOUNDS,
    QUADOBJ,
    QMATRIX,
    ENDATA
};

} // anonymous namespace

model::Problem read_mps(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MPS file: " + filepath);
    }

    model::Problem problem;
    Section current_section = Section::NONE;

    std::string line;
    std::string obj_row_name = "";
    model::ObjectiveSense sense = model::ObjectiveSense::Minimize;

    struct RowDef {
        char type{'E'}; // N, G, L, E
        int64_t idx{-1};
        double rhs{0.0};
        double range{0.0};
        bool has_range{false};
        bool has_rhs{false};
    };

    std::unordered_map<std::string, RowDef> rows_map;
    std::vector<std::string> row_names_order;

    std::unordered_map<std::string, int64_t> cols_map;
    std::vector<std::string> col_names_order;

    std::vector<model::Triplet> A_triplets;
    std::vector<double> obj_coeffs;

    std::vector<model::VariableType> col_types;
    std::vector<double> col_lower;
    std::vector<double> col_upper;

    std::vector<model::Triplet> Q_triplets;
    double obj_offset = 0.0;

    bool integer_marker_active = false;

    auto get_or_create_col = [&](const std::string& cname) -> int64_t {
        auto it = cols_map.find(cname);
        if (it != cols_map.end()) {
            return it->second;
        }
        int64_t idx = static_cast<int64_t>(col_names_order.size());
        cols_map[cname] = idx;
        col_names_order.push_back(cname);
        col_types.push_back(integer_marker_active ? model::VariableType::Integer : model::VariableType::Continuous);
        col_lower.push_back(0.0); // Standard MPS default
        col_upper.push_back(model::SIH_INFINITY);
        obj_coeffs.push_back(0.0);
        return idx;
    };

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '*') continue; // Comment

        // Check if line starts with section keyword (non-space at column 0)
        if (!std::isspace(static_cast<unsigned char>(line[0]))) {
            std::vector<std::string> tokens = tokenize(line);
            if (tokens.empty()) continue;

            std::string header = to_upper(tokens[0]);
            if (header == "NAME") {
                current_section = Section::NAME;
                if (tokens.size() > 1) {
                    problem.set_name(tokens[1]);
                }
            } else if (header == "OBJSENSE") {
                current_section = Section::OBJSENSE;
            } else if (header == "ROWS") {
                current_section = Section::ROWS;
            } else if (header == "COLUMNS") {
                current_section = Section::COLUMNS;
            } else if (header == "RHS") {
                current_section = Section::RHS;
            } else if (header == "RANGES") {
                current_section = Section::RANGES;
            } else if (header == "BOUNDS") {
                current_section = Section::BOUNDS;
            } else if (header == "QUADOBJ") {
                current_section = Section::QUADOBJ;
            } else if (header == "QMATRIX") {
                current_section = Section::QMATRIX;
            } else if (header == "ENDATA") {
                current_section = Section::ENDATA;
                break;
            }
            continue;
        }

        std::vector<std::string> tokens = tokenize(line);
        if (tokens.empty()) continue;

        switch (current_section) {
            case Section::OBJSENSE: {
                std::string s = to_upper(tokens[0]);
                if (s == "MAX" || s == "MAXIMIZE") {
                    sense = model::ObjectiveSense::Maximize;
                } else if (s == "MIN" || s == "MINIMIZE") {
                    sense = model::ObjectiveSense::Minimize;
                }
                break;
            }
            case Section::ROWS: {
                // tokens: [type, name]
                if (tokens.size() >= 2) {
                    char type = std::toupper(tokens[0][0]);
                    std::string rname = tokens[1];
                    if (type == 'N') {
                        if (obj_row_name.empty()) {
                            obj_row_name = rname;
                        }
                    } else {
                        if (rows_map.find(rname) == rows_map.end()) {
                            int64_t idx = static_cast<int64_t>(row_names_order.size());
                            rows_map[rname] = {type, idx, 0.0, 0.0, false, false};
                            row_names_order.push_back(rname);
                        }
                    }
                }
                break;
            }
            case Section::COLUMNS: {
                // Check for marker: 'MARKER', 'INTORG' / 'INTEND'
                if (tokens.size() >= 3 && (tokens[1] == "'MARKER'" || tokens[1] == "MARKER")) {
                    std::string marker_type = to_upper(tokens[2]);
                    if (marker_type == "'INTORG'" || marker_type == "INTORG") {
                        integer_marker_active = true;
                    } else if (marker_type == "'INTEND'" || marker_type == "INTEND") {
                        integer_marker_active = false;
                    }
                    continue;
                }

                // Normal column: col_name, row1, val1, [row2, val2]
                if (tokens.size() >= 3) {
                    std::string cname = tokens[0];
                    int64_t col_idx = get_or_create_col(cname);

                    for (size_t k = 1; k + 1 < tokens.size(); k += 2) {
                        std::string rname = tokens[k];
                        double val = std::stod(tokens[k + 1]);

                        if (rname == obj_row_name) {
                            obj_coeffs[col_idx] += val;
                        } else {
                            auto r_it = rows_map.find(rname);
                            if (r_it != rows_map.end()) {
                                A_triplets.emplace_back(r_it->second.idx, col_idx, val);
                            }
                        }
                    }
                }
                break;
            }
            case Section::RHS: {
                // tokens: [rhs_name, row1, val1, [row2, val2]]
                size_t start = (tokens.size() % 2 == 1) ? 1 : 0;
                for (size_t k = start; k + 1 < tokens.size(); k += 2) {
                    std::string rname = tokens[k];
                    double val = std::stod(tokens[k + 1]);
                    if (rname == obj_row_name) {
                        obj_offset = -val;
                    } else {
                        auto r_it = rows_map.find(rname);
                        if (r_it != rows_map.end()) {
                            r_it->second.rhs = val;
                            r_it->second.has_rhs = true;
                        }
                    }
                }
                break;
            }
            case Section::RANGES: {
                // tokens: [range_name, row1, val1, [row2, val2]]
                size_t start = (tokens.size() % 2 == 1) ? 1 : 0;
                for (size_t k = start; k + 1 < tokens.size(); k += 2) {
                    std::string rname = tokens[k];
                    double val = std::stod(tokens[k + 1]);
                    auto r_it = rows_map.find(rname);
                    if (r_it != rows_map.end()) {
                        r_it->second.range = val;
                        r_it->second.has_range = true;
                    }
                }
                break;
            }
            case Section::BOUNDS: {
                // tokens: [bound_type, bound_name, col_name, (value)]
                if (tokens.size() >= 3) {
                    std::string btype = to_upper(tokens[0]);
                    std::string cname = tokens[2];
                    double val = (tokens.size() >= 4) ? std::stod(tokens[3]) : 0.0;

                    int64_t col_idx = get_or_create_col(cname);

                    if (btype == "UP") {
                        col_upper[col_idx] = val;
                    } else if (btype == "LO") {
                        col_lower[col_idx] = val;
                    } else if (btype == "FX") {
                        col_lower[col_idx] = val;
                        col_upper[col_idx] = val;
                    } else if (btype == "FR") {
                        col_lower[col_idx] = -model::SIH_INFINITY;
                        col_upper[col_idx] = model::SIH_INFINITY;
                    } else if (btype == "MI") {
                        col_lower[col_idx] = -model::SIH_INFINITY;
                        col_upper[col_idx] = 0.0;
                    } else if (btype == "PL") {
                        col_lower[col_idx] = 0.0;
                        col_upper[col_idx] = model::SIH_INFINITY;
                    } else if (btype == "BV") {
                        col_types[col_idx] = model::VariableType::Binary;
                        col_lower[col_idx] = 0.0;
                        col_upper[col_idx] = 1.0;
                    } else if (btype == "UI") {
                        col_types[col_idx] = model::VariableType::Integer;
                        col_upper[col_idx] = val;
                    } else if (btype == "LI") {
                        col_types[col_idx] = model::VariableType::Integer;
                        col_lower[col_idx] = val;
                    }
                }
                break;
            }
            case Section::QUADOBJ:
            case Section::QMATRIX: {
                // tokens: [col1, col2, val]
                if (tokens.size() >= 3) {
                    std::string c1 = tokens[0];
                    std::string c2 = tokens[1];
                    double val = std::stod(tokens[2]);

                    int64_t idx1 = get_or_create_col(c1);
                    int64_t idx2 = get_or_create_col(c2);

                    Q_triplets.emplace_back(idx1, idx2, val);
                    if (idx1 != idx2) {
                        // Symmetrize if off-diagonal
                        Q_triplets.emplace_back(idx2, idx1, val);
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    const int64_t m = static_cast<int64_t>(row_names_order.size());
    const int64_t n = static_cast<int64_t>(col_names_order.size());

    problem.resize(m, n);
    problem.set_sense(sense);
    problem.set_obj_offset(obj_offset);
    problem.set_c(obj_coeffs);
    problem.set_row_names(row_names_order);
    problem.set_col_names(col_names_order);

    // Compute row bounds from type, rhs, ranges
    std::vector<double>& row_lower = problem.row_lower();
    std::vector<double>& row_upper = problem.row_upper();

    for (int64_t i = 0; i < m; ++i) {
        const auto& rdef = rows_map[row_names_order[i]];
        double rhs = rdef.rhs;
        char type = rdef.type;

        if (!rdef.has_range) {
            if (type == 'G') {
                row_lower[i] = rhs;
                row_upper[i] = model::SIH_INFINITY;
            } else if (type == 'L') {
                row_lower[i] = -model::SIH_INFINITY;
                row_upper[i] = rhs;
            } else if (type == 'E') {
                row_lower[i] = rhs;
                row_upper[i] = rhs;
            }
        } else {
            double r = rdef.range;
            double abs_r = std::abs(r);
            if (type == 'G') {
                row_lower[i] = rhs;
                row_upper[i] = rhs + abs_r;
            } else if (type == 'L') {
                row_lower[i] = rhs - abs_r;
                row_upper[i] = rhs;
            } else if (type == 'E') {
                if (r >= 0.0) {
                    row_lower[i] = rhs;
                    row_upper[i] = rhs + r;
                } else {
                    row_lower[i] = rhs + r;
                    row_upper[i] = rhs;
                }
            }
        }
    }

    // Assign column bounds and types
    for (int64_t j = 0; j < n; ++j) {
        problem.col_lower()[j] = col_lower[j];
        problem.col_upper()[j] = col_upper[j];
        problem.var_types()[j] = col_types[j];
    }

    // Build sparse constraint matrix A
    problem.set_A(model::SparseMatrix::from_triplets(m, n, A_triplets));

    // Build quadratic matrix Q if present
    if (!Q_triplets.empty()) {
        problem.set_quadratic_objective(model::SparseMatrix::from_triplets(n, n, Q_triplets));
    }

    problem.validate();
    return problem;
}

void write_mps(const model::Problem& problem, const std::string& filepath, bool free_format) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open file for writing MPS: " + filepath);
    }

    out << std::setprecision(12);

    // Ensure row and col names exist
    std::vector<std::string> row_names = problem.row_names();
    while (static_cast<int64_t>(row_names.size()) < problem.num_rows()) {
        row_names.push_back("R" + std::to_string(row_names.size()));
    }
    std::vector<std::string> col_names = problem.col_names();
    while (static_cast<int64_t>(col_names.size()) < problem.num_cols()) {
        col_names.push_back("C" + std::to_string(col_names.size()));
    }

    // NAME
    out << "NAME          " << problem.name() << "\n";

    // OBJSENSE
    out << "OBJSENSE\n";
    out << "  " << (problem.sense() == model::ObjectiveSense::Maximize ? "MAX" : "MIN") << "\n";

    auto is_neg_inf = [](double v) { return std::isinf(v) ? (v < 0.0) : (v <= -1e20); };
    auto is_pos_inf = [](double v) { return std::isinf(v) ? (v > 0.0) : (v >= 1e20); };

    // ROWS
    out << "ROWS\n";
    out << " N  OBJ\n";
    for (int64_t i = 0; i < problem.num_rows(); ++i) {
        const std::string& rname = row_names[i];
        double lb = problem.row_lower()[i];
        double ub = problem.row_upper()[i];

        char type = 'E';
        if (is_neg_inf(lb) && !is_pos_inf(ub)) {
            type = 'L';
        } else if (!is_neg_inf(lb) && is_pos_inf(ub)) {
            type = 'G';
        } else if (lb == ub) {
            type = 'E';
        } else {
            type = 'L'; // Boxed/ranged row written as L with RANGES
        }
        out << " " << type << "  " << rname << "\n";
    }

    // COLUMNS
    out << "COLUMNS\n";
    bool in_int_marker = false;
    const auto& csc_col_ptr = problem.A().csc_col_ptr();
    const auto& csc_row_ind = problem.A().csc_row_ind();
    const auto& csc_values = problem.A().csc_values();

    for (int64_t j = 0; j < problem.num_cols(); ++j) {
        const std::string& cname = col_names[j];
        bool is_int = (problem.var_types()[j] == model::VariableType::Integer ||
                       problem.var_types()[j] == model::VariableType::Binary);

        if (is_int && !in_int_marker) {
            out << "    MARK0000  'MARKER'                 'INTORG'\n";
            in_int_marker = true;
        } else if (!is_int && in_int_marker) {
            out << "    MARK0001  'MARKER'                 'INTEND'\n";
            in_int_marker = false;
        }

        // Objective coeff
        if (std::abs(problem.c()[j]) > 1e-15) {
            out << "    " << std::left << std::setw(8) << cname
                << "  " << std::setw(8) << "OBJ"
                << "  " << problem.c()[j] << "\n";
        }

        // Matrix entries in col j
        const int64_t start = csc_col_ptr[j];
        const int64_t end = csc_col_ptr[j + 1];
        for (int64_t k = start; k < end; ++k) {
            int64_t r_idx = csc_row_ind[k];
            double val = csc_values[k];
            out << "    " << std::left << std::setw(8) << cname
                << "  " << std::setw(8) << row_names[r_idx]
                << "  " << val << "\n";
        }
    }
    if (in_int_marker) {
        out << "    MARK9999  'MARKER'                 'INTEND'\n";
    }

    // RHS
    out << "RHS\n";
    if (std::abs(problem.obj_offset()) > 1e-15) {
        out << "    RHS1      OBJ       " << -problem.obj_offset() << "\n";
    }
    for (int64_t i = 0; i < problem.num_rows(); ++i) {
        double lb = problem.row_lower()[i];
        double ub = problem.row_upper()[i];
        const std::string& rname = row_names[i];

        if (is_neg_inf(lb) && !is_pos_inf(ub)) {
            if (std::abs(ub) > 1e-15) {
                out << "    RHS1      " << std::setw(8) << rname << "  " << ub << "\n";
            }
        } else if (!is_neg_inf(lb) && is_pos_inf(ub)) {
            if (std::abs(lb) > 1e-15) {
                out << "    RHS1      " << std::setw(8) << rname << "  " << lb << "\n";
            }
        } else if (lb == ub) {
            if (std::abs(lb) > 1e-15) {
                out << "    RHS1      " << std::setw(8) << rname << "  " << lb << "\n";
            }
        } else {
            // Ranged row: write ub as rhs
            out << "    RHS1      " << std::setw(8) << rname << "  " << ub << "\n";
        }
    }

    // RANGES
    bool has_ranges = false;
    for (int64_t i = 0; i < problem.num_rows(); ++i) {
        double lb = problem.row_lower()[i];
        double ub = problem.row_upper()[i];
        if (!is_neg_inf(lb) && !is_pos_inf(ub) && lb != ub) {
            has_ranges = true;
            break;
        }
    }
    if (has_ranges) {
        out << "RANGES\n";
        for (int64_t i = 0; i < problem.num_rows(); ++i) {
            double lb = problem.row_lower()[i];
            double ub = problem.row_upper()[i];
            if (!is_neg_inf(lb) && !is_pos_inf(ub) && lb != ub) {
                out << "    RNG1      " << std::setw(8) << row_names[i] << "  " << (ub - lb) << "\n";
            }
        }
    }

    // BOUNDS
    out << "BOUNDS\n";
    for (int64_t j = 0; j < problem.num_cols(); ++j) {
        const std::string& cname = col_names[j];
        double lb = problem.col_lower()[j];
        double ub = problem.col_upper()[j];
        auto vt = problem.var_types()[j];

        if (vt == model::VariableType::Binary) {
            out << " BV BND1      " << cname << "\n";
        } else if (lb == ub) {
            out << " FX BND1      " << std::setw(8) << cname << "  " << lb << "\n";
        } else {
            if (is_neg_inf(lb) && is_pos_inf(ub)) {
                out << " FR BND1      " << cname << "\n";
            } else {
                if (!is_neg_inf(lb) && lb != 0.0) {
                    out << " LO BND1      " << std::setw(8) << cname << "  " << lb << "\n";
                }
                if (!is_pos_inf(ub)) {
                    if (vt == model::VariableType::Integer) {
                        out << " UI BND1      " << std::setw(8) << cname << "  " << ub << "\n";
                    } else {
                        out << " UP BND1      " << std::setw(8) << cname << "  " << ub << "\n";
                    }
                }
            }
        }
    }

    // QUADOBJ / QMATRIX
    if (problem.has_quadratic() && problem.Q().num_nonzeros() > 0) {
        out << "QUADOBJ\n";
        const auto& q_col_ptr = problem.Q().csc_col_ptr();
        const auto& q_row_ind = problem.Q().csc_row_ind();
        const auto& q_values = problem.Q().csc_values();

        for (int64_t j = 0; j < problem.num_cols(); ++j) {
            const std::string& cname2 = col_names[j];
            const int64_t start = q_col_ptr[j];
            const int64_t end = q_col_ptr[j + 1];
            for (int64_t k = start; k < end; ++k) {
                int64_t i = q_row_ind[k];
                if (i <= j) {
                    out << "    " << std::left << std::setw(8) << col_names[i]
                        << "  " << std::setw(8) << cname2
                        << "  " << q_values[k] << "\n";
                }
            }
        }
    }

    out << "ENDATA\n";
}

} // namespace io
} // namespace sih
