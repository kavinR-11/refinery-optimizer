"""
Basic GPU Solver Smoke Test.
Verifies CUDA device availability, VRAM memory query, and GPU PDHG kernel execution.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver


def test_gpu_smoke():
    print("\n=== GPU PDHG Smoke Test ===")
    cuda_avail = sih_solver.GpuSolver.is_cuda_available()
    print(f"CUDA Device Available: {cuda_avail}")
    assert cuda_avail, "CUDA device should be available on WSL RTX 4060"

    vram = sih_solver.GpuSolver.get_vram_info()
    total_mb = vram.total_bytes / (1024 * 1024)
    free_mb = vram.free_bytes / (1024 * 1024)
    used_mb = vram.used_bytes / (1024 * 1024)
    print(f"VRAM Info: Total={total_mb:.1f} MB, Free={free_mb:.1f} MB, Used={used_mb:.1f} MB")
    assert total_mb > 1000.0, "Expected > 1GB VRAM detected"

    # Test on toy problem
    prob = sih_solver.read_mps("data/toy/toy01_lp_simple.mps")
    print(f"Loaded {prob.name}: Rows={prob.num_rows()}, Cols={prob.num_cols()}")

    # 1. CPU Oracle
    opts_cpu = sih_solver.Options()
    opts_cpu.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
    sol_cpu = sih_solver.solve(prob, opts_cpu)
    print(f"CPU Oracle: Status={sol_cpu.status}, Obj={sol_cpu.primal_objective:.6f}")
    assert sol_cpu.is_optimal()

    # 2. GPU PDHG
    cfg = sih_solver.GpuPdhgConfig()
    cfg.max_iterations = 2000
    cfg.tol_primal = 1e-4
    sol_gpu = sih_solver.solve_gpu_pdhg(prob, cfg)
    print(f"GPU PDHG: Status={sol_gpu.status}, Obj={sol_gpu.primal_objective:.6f}, Iters={sol_gpu.simplex_iterations}, Time={sol_gpu.time_wall_sec*1000:.3f} ms")

    # 3. Hybrid GPU PDHG + CPU Polish
    sol_hybrid = sih_solver.solve_gpu_hybrid(prob, cfg)
    print(f"GPU Hybrid: Status={sol_hybrid.status}, Obj={sol_hybrid.primal_objective:.6f}, Time={sol_hybrid.time_wall_sec*1000:.3f} ms")
    assert sol_hybrid.is_optimal()

    diff = abs(sol_hybrid.primal_objective - sol_cpu.primal_objective)
    print(f"Absolute diff between Hybrid and CPU Oracle: {diff:.2e}")
    assert diff < 1e-3, f"Hybrid solution mismatch: diff={diff}"
    print(">>> GPU smoke test PASSED.")


if __name__ == "__main__":
    test_gpu_smoke()
