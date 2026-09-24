#!/usr/bin/env python3
"""
GPU PDHG vs CPU Crossover Benchmark Script.
Compares:
  1. CPU-Only Simplex
  2. CPU-Only IPM
  3. GPU Custom PDHG (our kernels)
  4. GPU Hybrid (GPU PDHG + CPU Simplex Polish)
  5. cuSPARSE Baseline (vendor SpMV reference)
Across a synthetic LP size series.
Profiles VRAM usage and identifies the exact crossover point.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sih_solver
from bench.synthetic_generator import generate_synthetic_lp


PROBLEM_SIZES = [
    {"name": "Tiny",    "m": 100,  "n": 200,   "entries_per_col": 5},
    {"name": "Small-1", "m": 250,  "n": 500,   "entries_per_col": 6},
    {"name": "Small-2", "m": 500,  "n": 1000,  "entries_per_col": 6},
    {"name": "Medium-1","m": 1000, "n": 2000,  "entries_per_col": 6},
    {"name": "Medium-2","m": 1500, "n": 3000,  "entries_per_col": 6},
    {"name": "Large",   "m": 2000, "n": 4000,  "entries_per_col": 6},
]

VRAM_SCALING_SIZES = [
    {"name": "Scale-50k",   "m": 25000,  "n": 50000,   "nnz_est": "350k"},
    {"name": "Scale-100k",  "m": 50000,  "n": 100000,  "nnz_est": "1.0M"},
    {"name": "Scale-500k",  "m": 200000, "n": 500000,  "nnz_est": "5.0M"},
    {"name": "Scale-2M",    "m": 500000, "n": 1000000, "nnz_est": "15.0M"},
    {"name": "Scale-10M",   "m": 1500000,"n": 3000000, "nnz_est": "50.0M"},
    {"name": "Scale-Max8G", "m": 5000000,"n": 10000000,"nnz_est": "250.0M"},
]


def run_cusparse_baseline(mps_path: str, iters: int = 1000) -> float:
    """Run cuSPARSE baseline binary and return elapsed wall time in ms."""
    bin_path = Path(__file__).resolve().parent / "cusparse_baseline"
    if not bin_path.exists():
        return float("nan")
    try:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{env.get('LD_LIBRARY_PATH', '')}"
        res = subprocess.run(
            [str(bin_path), mps_path, str(iters)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120
        )
        for line in res.stdout.splitlines():
            if "[cuSPARSE Baseline]" in line and "Time=" in line:
                part = line.split("Time=")[1].split(" ms")[0]
                return float(part)
    except Exception as e:
        print(f"cuSPARSE baseline failed: {e}", flush=True)
    return float("nan")


def main():
    print("=" * 80, flush=True)
    print("PHASE 6: GPU PDHG vs CPU SOLVER CROSSOVER & VRAM BENCHMARK", flush=True)
    print(f"CUDA Available: {sih_solver.GpuSolver.is_cuda_available()}", flush=True)
    vram_start = sih_solver.GpuSolver.get_vram_info()
    print(f"Total VRAM: {vram_start.total_bytes / (1024*1024):.1f} MB, Free: {vram_start.free_bytes / (1024*1024):.1f} MB", flush=True)
    print("=" * 80, flush=True)

    data_dir = Path("data/synthetic")
    data_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for spec in PROBLEM_SIZES:
        name = spec["name"]
        m, n, k = spec["m"], spec["n"], spec["entries_per_col"]
        mps_path = str(data_dir / f"bench_{name}_{m}x{n}.mps")

        print(f"\n--- Benchmark Problem {name} ({m} rows x {n} cols, ~{k} entries/col) ---", flush=True)
        if not os.path.exists(mps_path):
            print(f"Generating {mps_path}...", flush=True)
            generate_synthetic_lp(m, n, entries_per_col=k, output_path=mps_path)

        prob = sih_solver.read_mps(mps_path)
        nnz = prob.num_nonzeros()
        print(f"Instance loaded: Rows={m}, Cols={n}, Nonzeros={nnz}", flush=True)

        # 1. CPU-Only Simplex
        print("Running CPU-only Simplex...", flush=True)
        t0 = time.perf_counter()
        sol_cpu = sih_solver.solve(prob)
        t_cpu_ms = (time.perf_counter() - t0) * 1000.0
        obj_cpu = sol_cpu.primal_objective
        print(f"  CPU Simplex: Time={t_cpu_ms:.2f} ms, Status={sol_cpu.status}, Obj={obj_cpu:.4f}", flush=True)

        # 2. CPU-Only IPM
        print("Running CPU-only IPM...", flush=True)
        t0 = time.perf_counter()
        opts_ipm = sih_solver.Options()
        opts_ipm.strategy.algorithm = sih_solver.AlgorithmChoice.Barrier
        sol_ipm = sih_solver.solve(prob, opts_ipm)
        t_ipm_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  CPU IPM    : Time={t_ipm_ms:.2f} ms, Status={sol_ipm.status}, Obj={sol_ipm.primal_objective:.4f}", flush=True)

        # 3. GPU Custom PDHG
        print("Running GPU PDHG (custom kernels)...", flush=True)
        cfg = sih_solver.GpuPdhgConfig()
        cfg.max_iterations = 1000
        cfg.check_frequency = 50
        t0 = time.perf_counter()
        sol_pdhg = sih_solver.solve_gpu_pdhg(prob, cfg)
        t_gpu_pdhg_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  GPU PDHG   : Time={t_gpu_pdhg_ms:.2f} ms, Status={sol_pdhg.status}, Obj={sol_pdhg.primal_objective:.4f}", flush=True)

        # 4. GPU Hybrid (PDHG + CPU Simplex Polish)
        print("Running GPU Hybrid (PDHG + Polish)...", flush=True)
        vram_before = sih_solver.GpuSolver.get_vram_info().used_bytes / (1024 * 1024)
        t0 = time.perf_counter()
        sol_hybrid = sih_solver.solve_gpu_hybrid(prob, cfg)
        t_hybrid_ms = (time.perf_counter() - t0) * 1000.0
        vram_after = sih_solver.GpuSolver.get_vram_info().used_bytes / (1024 * 1024)
        vram_used_mb = max(vram_after, vram_before)
        obj_hybrid = sol_hybrid.primal_objective
        rel_diff = abs(obj_hybrid - obj_cpu) / max(1.0, abs(obj_cpu))
        print(f"  GPU Hybrid : Time={t_hybrid_ms:.2f} ms, Status={sol_hybrid.status}, Obj={obj_hybrid:.4f}, RelErr={rel_diff:.2e}", flush=True)

        # 5. cuSPARSE Baseline (vendor SpMV reference)
        print("Running cuSPARSE Baseline SpMV reference...", flush=True)
        t_cusparse_ms = run_cusparse_baseline(mps_path, iters=1000)
        print(f"  cuSPARSE   : Time={t_cusparse_ms:.2f} ms (1000 iters)", flush=True)

        speedup_vs_simplex = t_cpu_ms / t_hybrid_ms if t_hybrid_ms > 0 else 0.0
        speedup_vs_ipm = t_ipm_ms / t_hybrid_ms if t_hybrid_ms > 0 else 0.0

        results.append({
            "name": name,
            "m": m,
            "n": n,
            "nnz": nnz,
            "t_cpu_ms": t_cpu_ms,
            "t_ipm_ms": t_ipm_ms,
            "t_pdhg_ms": t_gpu_pdhg_ms,
            "t_hybrid_ms": t_hybrid_ms,
            "t_cusparse_ms": t_cusparse_ms,
            "obj_cpu": obj_cpu,
            "obj_hybrid": obj_hybrid,
            "rel_diff": rel_diff,
            "speedup_vs_simplex": speedup_vs_simplex,
            "speedup_vs_ipm": speedup_vs_ipm,
            "vram_mb": vram_used_mb
        })

    # Summary Table
    print("\n" + "=" * 115, flush=True)
    print("CROSSOVER BENCHMARK RESULTS TABLE", flush=True)
    print("=" * 115, flush=True)
    header = f"{'Problem':<10} | {'Rows x Cols':<14} | {'NNZ':<8} | {'CPU Simplex':<12} | {'CPU IPM':<10} | {'GPU Hybrid':<12} | {'cuSPARSE':<10} | {'Speedup':<8} | {'Rel Diff':<10}"
    print(header, flush=True)
    print("-" * 115, flush=True)
    crossover_point = None
    for r in results:
        dim_str = f"{r['m']}x{r['n']}"
        row_str = (
            f"{r['name']:<10} | {dim_str:<14} | {r['nnz']:<8} | "
            f"{r['t_cpu_ms']:>8.1f} ms | {r['t_ipm_ms']:>6.1f} ms | "
            f"{r['t_hybrid_ms']:>8.1f} ms | {r['t_cusparse_ms']:>6.1f} ms | "
            f"{r['speedup_vs_simplex']:>6.2f}x | {r['rel_diff']:>8.2e}"
        )
        print(row_str, flush=True)
        if crossover_point is None and r['speedup_vs_simplex'] > 1.0:
            crossover_point = r

    print("=" * 115, flush=True)
    if crossover_point:
        print(f"\n>>> ACTUAL CROSSOVER POINT DETECTED: {crossover_point['name']} ({crossover_point['m']} x {crossover_point['n']}, {crossover_point['nnz']} nonzeros)", flush=True)
        print(f"    At this size, GPU Hybrid ({crossover_point['t_hybrid_ms']:.1f} ms) beats CPU Simplex ({crossover_point['t_cpu_ms']:.1f} ms) by {crossover_point['speedup_vs_simplex']:.2f}x speedup.", flush=True)
    else:
        print("\n>>> No crossover point detected in the tested range.", flush=True)

    # VRAM Scaling Analysis
    print("\n" + "=" * 80, flush=True)
    print("VRAM SCALING & 8 GB CAPACITY LIMITS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Scale Level':<14} | {'Dimensions (m x n)':<22} | {'Nonzeros':<10} | {'Est. VRAM':<12} | {'Status'}", flush=True)
    print("-" * 80, flush=True)
    for s in VRAM_SCALING_SIZES:
        name = s["name"]
        m, n = s["m"], s["n"]
        nnz_est = s["nnz_est"]
        val = float(nnz_est.replace("k", "").replace("M", ""))
        mult = 1e3 if "k" in nnz_est else 1e6
        total_nnz = val * mult
        vram_bytes = (16.0 * total_nnz) + (8.0 * (m + n)) + (48.0 * max(m, n)) + (128 * 1024 * 1024)
        vram_mb = vram_bytes / (1024 * 1024)
        status = "FITS in 8 GB VRAM" if vram_mb < 8188.0 else "EXCEEDS 8 GB VRAM (OOM)"
        dim_str = f"{m} x {n}"
        print(f"{name:<14} | {dim_str:<22} | {nnz_est:<10} | {vram_mb:>8.1f} MB | {status}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
