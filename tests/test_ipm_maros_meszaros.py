import sys
import time
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from oracle import HighsOracle

QP_PROBLEMS = [
    "HS21",
    "HS35",
    "HS51",
    "HS52",
    "HS76",
    "CVXQP1_S",
    "CVXQP2_S",
    "CVXQP3_S",
    "DUAL1",
    "DUAL2",
]

def run_qp_benchmark():
    print(f"\n{'Problem':<15} | {'HiGHS Obj':>14} | {'IPM Obj':>14} | {'Rel Diff':>10} | {'Iters':<5} | {'Time (s)':>8} | {'Status':<10} | {'Checker'}")
    print("-" * 95)
    
    results = []
    
    for name in QP_PROBLEMS:
        mps_path = root / "data" / "maros_meszaros" / f"{name}.mps"
        if not mps_path.exists():
            print(f"Skipping {name}: file not found")
            continue
            
        # Oracle solve
        h_res = HighsOracle.solve_mps(str(mps_path))
        h_obj = h_res["objective_value"]
        h_status = h_res["status"]
        
        # SIH IPM solve
        p = sih_solver.read_mps(str(mps_path))
        
        opts = sih_solver.Options()
        opts.strategy.ipm_max_iterations = 100

        
        t0 = time.perf_counter()
        sol = sih_solver.solve_ipm(p, opts)
        t_elapsed = time.perf_counter() - t0
        
        chk = sih_solver.check_solution(p, sol)
        
        rel_diff = abs(sol.primal_objective - h_obj) / max(1.0, abs(h_obj))
        
        print(f"{name:<15} | {h_obj:>14.6f} | {sol.primal_objective:>14.6f} | {rel_diff:>10.2e} | {sol.simplex_iterations:<5} | {t_elapsed:>8.4f} | {sol.status.name:<10} | {chk.all_checks_passed}")
        
        results.append({
            "name": name,
            "highs_obj": h_obj,
            "ipm_obj": sol.primal_objective,
            "rel_diff": rel_diff,
            "iters": sol.simplex_iterations,
            "time_sec": t_elapsed,
            "status": sol.status.name,
            "checker_passed": chk.all_checks_passed
        })
        
    return results

@pytest.mark.parametrize("name", QP_PROBLEMS)
def test_maros_meszaros_instance(name):
    mps_path = root / "data" / "maros_meszaros" / f"{name}.mps"
    assert mps_path.exists()
    
    h_res = HighsOracle.solve_mps(str(mps_path))
    h_obj = h_res["objective_value"]
    
    p = sih_solver.read_mps(str(mps_path))
    opts = sih_solver.Options()
    sol = sih_solver.solve_ipm(p, opts)
    
    assert sol.is_optimal()
    rel_diff = abs(sol.primal_objective - h_obj) / max(1.0, abs(h_obj))
    assert rel_diff < 1e-6, f"{name}: rel diff {rel_diff} exceeds 1e-6"
    
    chk = sih_solver.check_solution(p, sol)
    assert chk.all_checks_passed, f"{name}: checker failed: {chk.summary}"

if __name__ == "__main__":
    run_qp_benchmark()

