import sys
import time
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from oracle import HighsOracle

NETLIB_PROBLEMS = [
    "afiro",
    "sc50a",
    "sc50b",
    "adlittle",
    "blend",
    "share2b",
    "lotfi",
    "stocfor1",
    "beaconfd",
]

def run_netlib_comparison():
    print(f"\n{'Problem':<12} | {'Rows':>4} | {'Cols':>4} | {'HiGHS Obj':>14} | {'Simplex Obj':>14} | {'IPM Obj':>14} | {'Smplx It':>8} | {'IPM It':>6} | {'Smplx (s)':>9} | {'IPM (s)':>8} | {'Faster'}")
    print("-" * 125)
    
    results = []
    
    for name in NETLIB_PROBLEMS:
        mps_path = root / "data" / "netlib" / f"{name}.mps"
        if not mps_path.exists():
            continue
            
        p = sih_solver.read_mps(str(mps_path))
        
        # Oracle solve
        h_res = HighsOracle.solve_mps(str(mps_path))
        h_obj = h_res["objective_value"]
        
        # Simplex solve (Phase 1)
        opts_simplex = sih_solver.Options()
        opts_simplex.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
        opts_simplex.strategy.presolve = sih_solver.PresolveMode.Off
        t0 = time.perf_counter()
        sol_simplex = sih_solver.solve(p, opts_simplex)
        t_simplex = time.perf_counter() - t0
        
        # IPM solve with Crossover (Phase 2)
        opts_ipm = sih_solver.Options()
        opts_ipm.strategy.algorithm = sih_solver.AlgorithmChoice.Barrier
        opts_ipm.strategy.ipm_enable_crossover = True
        t0 = time.perf_counter()
        sol_ipm = sih_solver.solve_ipm(p, opts_ipm)
        t_ipm = time.perf_counter() - t0
        
        # Independent checker
        chk = sih_solver.check_solution(p, sol_ipm)
        
        diff_ipm = abs(sol_ipm.primal_objective - h_obj) / max(1.0, abs(h_obj))
        faster = "IPM" if t_ipm < t_simplex else "Simplex"
        
        print(f"{name:<12} | {p.num_rows():4d} | {p.num_cols():4d} | {h_obj:>14.6f} | {sol_simplex.primal_objective:>14.6f} | {sol_ipm.primal_objective:>14.6f} | {sol_simplex.simplex_iterations:8d} | {sol_ipm.simplex_iterations:6d} | {t_simplex:9.4f} | {t_ipm:8.4f} | {faster}")
        
        results.append({
            "name": name,
            "rows": p.num_rows(),
            "cols": p.num_cols(),
            "highs_obj": h_obj,
            "simplex_obj": sol_simplex.primal_objective,
            "ipm_obj": sol_ipm.primal_objective,
            "simplex_iters": sol_simplex.simplex_iterations,
            "ipm_iters": sol_ipm.simplex_iterations,
            "simplex_time_sec": t_simplex,
            "ipm_time_sec": t_ipm,
            "diff_ipm_vs_oracle": diff_ipm,
            "checker_passed": chk.all_checks_passed
        })
        
    return results

@pytest.mark.parametrize("name", NETLIB_PROBLEMS)
def test_netlib_ipm_instance(name):
    mps_path = root / "data" / "netlib" / f"{name}.mps"
    assert mps_path.exists()
    
    h_res = HighsOracle.solve_mps(str(mps_path))
    h_obj = h_res["objective_value"]
    
    p = sih_solver.read_mps(str(mps_path))
    opts = sih_solver.Options()
    opts.strategy.ipm_enable_crossover = True
    sol = sih_solver.solve_ipm(p, opts)
    
    assert sol.is_optimal()
    rel_diff = abs(sol.primal_objective - h_obj) / max(1.0, abs(h_obj))
    assert rel_diff < 1e-6, f"{name}: rel diff {rel_diff} exceeds 1e-6"
    
    chk = sih_solver.check_solution(p, sol)
    assert chk.all_checks_passed, f"{name}: checker failed: {chk.summary}"

if __name__ == "__main__":
    run_netlib_comparison()
