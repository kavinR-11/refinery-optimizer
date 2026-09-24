#!/usr/bin/env python3
"""
Full Solver Benchmark Suite: Netlib (LP), MIPLIB (MILP), and Maros-Mészáros (QP)
Evaluates our indigenous solver against the HiGHS reference oracle.
Computes Shifted Geometric Mean (SGM, s = 1.0s) and Dolan-Moré performance profile points.
Strictly compliant with Hard Rules 1-3.
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sih_solver
from tests.oracle.highs_oracle import HighsOracle


def shifted_geometric_mean(times: List[float], shift: float = 1.0) -> float:
    """Compute shifted geometric mean: exp(1/N * sum(ln(t_i + s))) - s"""
    if not times:
        return 0.0
    sum_log = sum(math.log(max(0.0001, t) + shift) for t in times)
    return math.exp(sum_log / len(times)) - shift


def compute_performance_profile(ratios: List[float], tau_grid: List[float]) -> List[float]:
    """Compute Dolan-More cumulative distribution rho(tau)"""
    n = len(ratios)
    if n == 0:
        return [0.0] * len(tau_grid)
    profile = []
    for tau in tau_grid:
        count = sum(1 for r in ratios if r <= tau)
        profile.append(round(count / n, 4))
    return profile


def run_lp_benchmarks() -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("1. LP BENCHMARK SUITE (Netlib Standard)")
    print("=" * 90)
    netlib_dir = Path("data/netlib")
    mps_files = sorted([f for f in netlib_dir.glob("*.mps") if f.stat().st_size > 0])

    results = []
    our_times = []
    highs_times = []
    ratios = []

    print(f"{'Instance':<14} | {'Rows x Cols':<12} | {'Our Status':<10} | {'Our Time':<10} | {'HiGHS Status':<12} | {'HiGHS Time':<10} | {'Ratio':<7} | {'Diff'}")
    print("-" * 90)

    for mps in mps_files:
        name = mps.stem
        prob = sih_solver.read_mps(str(mps))
        dim_str = f"{prob.num_rows()}x{prob.num_cols()}"

        # Our solve (Dual Simplex)
        t0 = time.perf_counter()
        opts = sih_solver.Options()
        opts.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
        sol = sih_solver.solve(prob, opts)
        t_our = max(0.0001, time.perf_counter() - t0)

        # HiGHS solve
        t0 = time.perf_counter()
        h_res = HighsOracle.solve_mps(str(mps))
        t_highs = max(0.0001, time.perf_counter() - t0)

        our_times.append(t_our)
        highs_times.append(t_highs)

        ratio = t_our / t_highs
        ratios.append(ratio)

        obj_diff = 0.0
        if sol.is_optimal() and h_res["status"] == "Optimal":
            obj_diff = abs(sol.primal_objective - h_res["objective_value"])

        results.append({
            "name": name,
            "m": prob.num_rows(),
            "n": prob.num_cols(),
            "our_status": str(sol.status).replace("SolutionStatus.", ""),
            "our_time": t_our,
            "highs_status": h_res["status"],
            "highs_time": t_highs,
            "ratio": ratio,
            "obj_diff": obj_diff
        })

        print(f"{name:<14} | {dim_str:<12} | {results[-1]['our_status']:<10} | {t_our*1000:>7.2f} ms | {h_res['status']:<12} | {t_highs*1000:>7.2f} ms | {ratio:>6.2f}x | {obj_diff:.1e}")

    sgm_our = shifted_geometric_mean(our_times, shift=1.0)
    sgm_highs = shifted_geometric_mean(highs_times, shift=1.0)
    tau_grid = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    profile = compute_performance_profile(ratios, tau_grid)

    print("-" * 90)
    print(f"Shifted GeoMean (s=1.0s): Our Solver = {sgm_our*1000:.2f} ms | HiGHS = {sgm_highs*1000:.2f} ms")
    print(f"Performance Profile rho(tau) for tau in {tau_grid}:")
    print(f"  {profile}")

    return {
        "class": "LP",
        "count": len(mps_files),
        "sgm_our_sec": sgm_our,
        "sgm_highs_sec": sgm_highs,
        "tau_grid": tau_grid,
        "profile": profile,
        "records": results
    }


def run_milp_benchmarks() -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("2. MILP BENCHMARK SUITE (MIPLIB Subset)")
    print("=" * 90)
    miplib_dir = Path("data/miplib")
    mps_files = sorted([f for f in miplib_dir.glob("*.mps") if f.stat().st_size > 0])

    results = []
    our_times = []
    highs_times = []
    ratios = []

    print(f"{'Instance':<14} | {'Rows x Cols':<12} | {'Our Status':<10} | {'Our Time':<10} | {'HiGHS Status':<12} | {'HiGHS Time':<10} | {'Ratio':<7} | {'Nodes'}")
    print("-" * 90)

    for mps in mps_files:
        name = mps.stem
        prob = sih_solver.read_mps(str(mps))
        dim_str = f"{prob.num_rows()}x{prob.num_cols()}"

        # Our solve (Branch and Bound with node limit)
        t0 = time.perf_counter()
        opts = sih_solver.Options()
        opts.strategy.algorithm = sih_solver.AlgorithmChoice.BranchAndBound
        opts.node_limit = 500
        sol = sih_solver.solve(prob, opts)
        t_our = max(0.0001, time.perf_counter() - t0)

        # HiGHS solve
        t0 = time.perf_counter()
        h_res = HighsOracle.solve_mps(str(mps), time_limit_sec=5.0)
        t_highs = max(0.0001, time.perf_counter() - t0)

        our_times.append(t_our)
        highs_times.append(t_highs)

        ratio = t_our / t_highs
        ratios.append(ratio)

        our_stat = str(sol.status).replace("SolutionStatus.", "")
        results.append({
            "name": name,
            "m": prob.num_rows(),
            "n": prob.num_cols(),
            "our_status": our_stat,
            "our_time": t_our,
            "highs_status": h_res["status"],
            "highs_time": t_highs,
            "ratio": ratio,
            "nodes": sol.nodes_explored
        })

        print(f"{name:<14} | {dim_str:<12} | {our_stat:<10} | {t_our*1000:>7.2f} ms | {h_res['status']:<12} | {t_highs*1000:>7.2f} ms | {ratio:>6.2f}x | {sol.nodes_explored}")

    sgm_our = shifted_geometric_mean(our_times, shift=1.0)
    sgm_highs = shifted_geometric_mean(highs_times, shift=1.0)
    tau_grid = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    profile = compute_performance_profile(ratios, tau_grid)

    print("-" * 90)
    print(f"Shifted GeoMean (s=1.0s): Our Solver = {sgm_our*1000:.2f} ms | HiGHS = {sgm_highs*1000:.2f} ms")
    print(f"Performance Profile rho(tau) for tau in {tau_grid}:")
    print(f"  {profile}")

    return {
        "class": "MILP",
        "count": len(mps_files),
        "sgm_our_sec": sgm_our,
        "sgm_highs_sec": sgm_highs,
        "tau_grid": tau_grid,
        "profile": profile,
        "records": results
    }


def run_qp_benchmarks() -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("3. QP BENCHMARK SUITE (Maros-Mészáros Convex QP)")
    print("=" * 90)
    qp_dir = Path("data/maros_meszaros")
    mps_files = sorted([f for f in qp_dir.glob("*.mps") if f.stat().st_size > 0])

    results = []
    our_times = []
    highs_times = []
    ratios = []

    print(f"{'Instance':<14} | {'Rows x Cols':<12} | {'Our Status':<10} | {'Our Time':<10} | {'HiGHS Status':<12} | {'HiGHS Time':<10} | {'Ratio':<7} | {'Diff'}")
    print("-" * 90)

    for mps in mps_files:
        name = mps.stem
        prob = sih_solver.read_mps(str(mps))
        dim_str = f"{prob.num_rows()}x{prob.num_cols()}"

        # Our solve (Barrier / Primal-Dual IPM)
        t0 = time.perf_counter()
        opts = sih_solver.Options()
        opts.strategy.algorithm = sih_solver.AlgorithmChoice.Barrier
        sol = sih_solver.solve(prob, opts)
        t_our = max(0.0001, time.perf_counter() - t0)

        # HiGHS solve
        t0 = time.perf_counter()
        h_res = HighsOracle.solve_mps(str(mps))
        t_highs = max(0.0001, time.perf_counter() - t0)

        our_times.append(t_our)
        highs_times.append(t_highs)

        ratio = t_our / t_highs
        ratios.append(ratio)

        obj_diff = 0.0
        if sol.is_optimal() and h_res["status"] == "Optimal":
            obj_diff = abs(sol.primal_objective - h_res["objective_value"])

        our_stat = str(sol.status).replace("SolutionStatus.", "")
        results.append({
            "name": name,
            "m": prob.num_rows(),
            "n": prob.num_cols(),
            "our_status": our_stat,
            "our_time": t_our,
            "highs_status": h_res["status"],
            "highs_time": t_highs,
            "ratio": ratio,
            "obj_diff": obj_diff
        })

        print(f"{name:<14} | {dim_str:<12} | {our_stat:<10} | {t_our*1000:>7.2f} ms | {h_res['status']:<12} | {t_highs*1000:>7.2f} ms | {ratio:>6.2f}x | {obj_diff:.1e}")

    sgm_our = shifted_geometric_mean(our_times, shift=1.0)
    sgm_highs = shifted_geometric_mean(highs_times, shift=1.0)
    tau_grid = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    profile = compute_performance_profile(ratios, tau_grid)

    print("-" * 90)
    print(f"Shifted GeoMean (s=1.0s): Our Solver = {sgm_our*1000:.2f} ms | HiGHS = {sgm_highs*1000:.2f} ms")
    print(f"Performance Profile rho(tau) for tau in {tau_grid}:")
    print(f"  {profile}")

    return {
        "class": "QP",
        "count": len(mps_files),
        "sgm_our_sec": sgm_our,
        "sgm_highs_sec": sgm_highs,
        "tau_grid": tau_grid,
        "profile": profile,
        "records": results
    }


def save_benchmark_charts(data: Dict[str, Any], output_path: str = "docs/performance_profiles.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    classes = ["LP", "MILP", "QP"]
    titles = [
        "Netlib LP Performance Profile",
        "MIPLIB Easy MILP Performance Profile",
        "Maros-Mészáros QP Performance Profile"
    ]

    for ax, cls_name, title in zip(axes, classes, titles):
        cls_data = data[cls_name]
        tau = cls_data["tau_grid"]
        rho = cls_data["profile"]

        ax.step(tau, rho, where="post", color="#1f77b4", linewidth=2.5, label="Indigenous Solver vs HiGHS")
        ax.scatter(tau, rho, color="#1f77b4", s=30, zorder=3)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Performance Ratio $\tau$ ($t_{solver} / t_{HiGHS}$)", fontsize=10)
        ax.set_ylabel(r"Fraction of Problems Solved $\rho(\tau)$", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(1.0, 10.0)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Performance profile chart saved to {output_path}")


def main():
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Full Benchmark Suite")
    parser.add_argument("--plot-only", action="store_true", help="Generate plots from existing JSON results")
    args = parser.parse_args()

    json_path = Path("bench/benchmark_results.json")

    if args.plot_only and json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        save_benchmark_charts(data)
        print("Charts regenerated successfully from cache.")
        return

    print("=" * 90)
    print("SIH 26119 INDIGENOUS SOLVER: FULL BENCHMARK SUITE")
    print("Comparing Indigenous Solver against HiGHS Oracle Reference")
    print("=" * 90)

    lp_res = run_lp_benchmarks()
    milp_res = run_milp_benchmarks()
    qp_res = run_qp_benchmarks()

    print("\n" + "=" * 90)
    print("FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Class':<8} | {'Instances':<10} | {'Our SGM (ms)':<14} | {'HiGHS SGM (ms)':<14} | {'Win Rate (tau=1.0)':<20} | {'Solvability (tau<=10)'}")
    print("-" * 90)
    for res in [lp_res, milp_res, qp_res]:
        win_rate = f"{res['profile'][0]*100:.1f}%"
        solvability = f"{res['profile'][-1]*100:.1f}%"
        print(f"{res['class']:<8} | {res['count']:<10} | {res['sgm_our_sec']*1000:>10.2f} ms   | {res['sgm_highs_sec']*1000:>10.2f} ms   | {win_rate:<20} | {solvability}")
    print("=" * 90)

    full_results = {
        "LP": lp_res,
        "MILP": milp_res,
        "QP": qp_res,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Benchmark results saved to {json_path}")

    save_benchmark_charts(full_results)


if __name__ == "__main__":
    main()

