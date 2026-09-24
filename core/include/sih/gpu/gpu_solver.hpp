#pragma once

#include <sih/model/problem.hpp>
#include <sih/model/solution.hpp>
#include <sih/model/options.hpp>
#include <string>
#include <vector>

namespace sih {
namespace gpu {

struct GpuPdhgConfig {
    int max_iterations{4000};
    float tol_primal{1e-4f};
    float tol_dual{1e-4f};
    float tol_gap{1e-4f};
    int check_frequency{25};
    float step_tau{0.0f}; // 0 = auto-estimate via spectral norm
    float step_sigma{0.0f};
    bool enable_restart{true};
    bool enable_hybrid_polish{true};
};

struct GpuMemoryInfo {
    size_t total_bytes{0};
    size_t free_bytes{0};
    size_t used_bytes{0};
};

class GpuSolver {
public:
    // Query whether solver was compiled with CUDA support and GPU device is detected
    static bool is_cuda_available() noexcept;

    // Query VRAM memory statistics (total, free, used bytes)
    static GpuMemoryInfo get_vram_info();

    // First-order Primal-Dual Hybrid Gradient (PDHG) solve on GPU (FP32)
    static model::Solution solve_pdhg(const model::Problem& problem,
                                     const GpuPdhgConfig& config = GpuPdhgConfig{});

    // Hybrid GPU PDHG + CPU Simplex/IPM crossover polish (FP32 -> FP64)
    static model::Solution solve_hybrid(const model::Problem& problem,
                                       const GpuPdhgConfig& config = GpuPdhgConfig{});
};

} // namespace gpu
} // namespace sih
