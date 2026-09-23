#include "sih/scaling/scaler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sih {
namespace scaling {

double Scaler::quantize_pow2(double val) {
    if (val <= 0.0 || !std::isfinite(val)) return 1.0;
    int k = static_cast<int>(std::round(std::log2(val)));
    // Clamp exponent to [-30, 30] to safeguard against numerical extremes
    k = std::max(-30, std::min(30, k));
    return std::ldexp(1.0, k);
}

ScalingFactors Scaler::compute_scaling(const model::Problem& problem,
                                       bool power_of_two,
                                       int equilibration_passes) {
    ScalingFactors factors;
    int64_t m = problem.num_rows();
    int64_t n = problem.num_cols();
    factors.row_scale.assign(m, 1.0);
    factors.col_scale.assign(n, 1.0);
    factors.is_scaled = true;

    if (m == 0 || n == 0 || problem.num_nonzeros() == 0) {
        return factors;
    }

    const auto& A = problem.A();
    const auto& col_ptr = A.csc_col_ptr();
    const auto& row_ind = A.csc_row_ind();
    const auto& values = A.csc_values();

    // 1. Geometric mean scaling pass
    // (a) Row geometric mean
    std::vector<double> min_row(m, std::numeric_limits<double>::infinity());
    std::vector<double> max_row(m, 0.0);

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int64_t i = row_ind[k];
            double abs_v = std::abs(values[k]);
            if (abs_v > 0.0) {
                min_row[i] = std::min(min_row[i], abs_v);
                max_row[i] = std::max(max_row[i], abs_v);
            }
        }
    }

    for (int64_t i = 0; i < m; ++i) {
        if (max_row[i] > 0.0 && std::isfinite(min_row[i])) {
            double r = 1.0 / std::sqrt(min_row[i] * max_row[i]);
            factors.row_scale[i] = power_of_two ? quantize_pow2(r) : r;
        }
    }

    // (b) Column geometric mean (accounting for row scaling)
    std::vector<double> min_col(n, std::numeric_limits<double>::infinity());
    std::vector<double> max_col(n, 0.0);

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int64_t i = row_ind[k];
            double abs_v = std::abs(values[k] * factors.row_scale[i]);
            if (abs_v > 0.0) {
                min_col[j] = std::min(min_col[j], abs_v);
                max_col[j] = std::max(max_col[j], abs_v);
            }
        }
        if (max_col[j] > 0.0 && std::isfinite(min_col[j])) {
            double c = 1.0 / std::sqrt(min_col[j] * max_col[j]);
            factors.col_scale[j] = power_of_two ? quantize_pow2(c) : c;
        }
    }

    // 2. Equilibration passes
    for (int pass = 0; pass < equilibration_passes; ++pass) {
        // Row equilibration
        std::vector<double> row_norm(m, 0.0);
        for (int64_t j = 0; j < n; ++j) {
            double c_scale = factors.col_scale[j];
            for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                int64_t i = row_ind[k];
                double abs_v = std::abs(values[k] * factors.row_scale[i] * c_scale);
                row_norm[i] = std::max(row_norm[i], abs_v);
            }
        }
        for (int64_t i = 0; i < m; ++i) {
            if (row_norm[i] > 0.0) {
                double r = 1.0 / row_norm[i];
                double mult = power_of_two ? quantize_pow2(r) : r;
                factors.row_scale[i] *= mult;
            }
        }

        // Column equilibration
        std::vector<double> col_norm(n, 0.0);
        for (int64_t j = 0; j < n; ++j) {
            double c_scale = factors.col_scale[j];
            for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                int64_t i = row_ind[k];
                double abs_v = std::abs(values[k] * factors.row_scale[i] * c_scale);
                col_norm[j] = std::max(col_norm[j], abs_v);
            }
            if (col_norm[j] > 0.0) {
                double c = 1.0 / col_norm[j];
                double mult = power_of_two ? quantize_pow2(c) : c;
                factors.col_scale[j] *= mult;
            }
        }
    }

    return factors;
}

model::Problem Scaler::scale_problem(const model::Problem& problem,
                                     const ScalingFactors& factors) {
    if (!factors.is_scaled) {
        return problem;
    }

    model::Problem scaled = problem;
    int64_t m = scaled.num_rows();
    int64_t n = scaled.num_cols();

    // Scale constraint matrix A: A_scaled = R * A * C
    const auto& A = problem.A();
    const auto& col_ptr = A.csc_col_ptr();
    const auto& row_ind = A.csc_row_ind();
    const auto& values = A.csc_values();

    std::vector<model::Triplet> triplets;
    triplets.reserve(A.num_nonzeros());

    for (int64_t j = 0; j < n; ++j) {
        double c_j = factors.col_scale[j];
        for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int64_t i = row_ind[k];
            double r_i = factors.row_scale[i];
            double val = values[k] * r_i * c_j;
            triplets.emplace_back(i, j, val);
        }
    }
    scaled.set_A(model::SparseMatrix::from_triplets(m, n, triplets));

    // Scale objective c: c_scaled = C * c
    auto& c = scaled.c();
    for (int64_t j = 0; j < n; ++j) {
        c[j] *= factors.col_scale[j];
    }

    // Scale row bounds: l_scaled = R * l, u_scaled = R * u
    auto& rl = scaled.row_lower();
    auto& ru = scaled.row_upper();
    for (int64_t i = 0; i < m; ++i) {
        double r_i = factors.row_scale[i];
        if (rl[i] > -model::SIH_INFINITY / 2.0) rl[i] *= r_i;
        if (ru[i] < model::SIH_INFINITY / 2.0) ru[i] *= r_i;
    }

    // Scale column bounds: lb_scaled = C^{-1} * lb, ub_scaled = C^{-1} * ub
    auto& cl = scaled.col_lower();
    auto& cu = scaled.col_upper();
    for (int64_t j = 0; j < n; ++j) {
        double inv_c_j = 1.0 / factors.col_scale[j];
        if (cl[j] > -model::SIH_INFINITY / 2.0) cl[j] *= inv_c_j;
        if (cu[j] < model::SIH_INFINITY / 2.0) cu[j] *= inv_c_j;
    }

    return scaled;
}

void Scaler::unscale_solution(model::Solution& solution,
                             const ScalingFactors& factors) {
    if (!factors.is_scaled) {
        return;
    }

    int64_t m = static_cast<int64_t>(factors.row_scale.size());
    int64_t n = static_cast<int64_t>(factors.col_scale.size());

    // Primal variables: x = C * x_scaled
    if (solution.x.size() == static_cast<size_t>(n)) {
        for (int64_t j = 0; j < n; ++j) {
            solution.x[j] *= factors.col_scale[j];
        }
    }

    // Row activities: Ax = R^{-1} * (Ax_scaled)
    if (solution.slack.size() == static_cast<size_t>(m)) {
        for (int64_t i = 0; i < m; ++i) {
            solution.slack[i] /= factors.row_scale[i];
        }
    }

    // Dual multipliers: y = R * y_scaled
    if (solution.row_duals.size() == static_cast<size_t>(m)) {
        for (int64_t i = 0; i < m; ++i) {
            solution.row_duals[i] *= factors.row_scale[i];
        }
    }

    // Reduced costs: s = C^{-1} * s_scaled
    if (solution.reduced_costs.size() == static_cast<size_t>(n)) {
        for (int64_t j = 0; j < n; ++j) {
            solution.reduced_costs[j] /= factors.col_scale[j];
        }
    }

    // Certificate ray unscaling:
    // If infeasible, ray is dual Farkas ray y: y = R * y_scaled
    // If unbounded, ray is primal direction d: d = C * d_scaled
    if (!solution.ray.empty()) {
        if (solution.status == model::SolutionStatus::Infeasible &&
            solution.ray.size() == static_cast<size_t>(m)) {
            for (int64_t i = 0; i < m; ++i) {
                solution.ray[i] *= factors.row_scale[i];
            }
        } else if (solution.status == model::SolutionStatus::Unbounded &&
                   solution.ray.size() == static_cast<size_t>(n)) {
            for (int64_t j = 0; j < n; ++j) {
                solution.ray[j] *= factors.col_scale[j];
            }
        }
    }
}

} // namespace scaling
} // namespace sih
