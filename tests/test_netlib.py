"""
Netlib LP Benchmark and Validation Suite.
Compares indigenous SimplexSolver against HiGHS oracle across Netlib benchmark instances.
Gate Criterion: Status and objective must match HiGHS within 1e-6 relative on every problem.
"""

import sys
import time
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from oracle import HighsOracle
from checker import FloatChecker

NETLIB_DIR = root / "data" / "netlib"
NETLIB_FILES = sorted([f.name for f in NETLIB_DIR.glob("*.mps")])

@pytest.mark.parametrize("mps_name", NETLIB_FILES)
def test_netlib_instance(mps_name):
    mps_path = str(NETLIB_DIR / mps_name)

    # 1. Load problem with indigenous reader
    p = sih_solver.read_mps(mps_path)
    assert p.num_rows() > 0
    assert p.num_cols() > 0

    # 2. Solve with HiGHS oracle
    t0_oracle = time.perf_counter()
    oracle_res = HighsOracle.solve_mps(mps_path)
    t1_oracle = time.perf_counter()
    highs_time = t1_oracle - t0_oracle

    assert oracle_res["status"] in ("Optimal", "Infeasible", "Unbounded"), (
        f"HiGHS returned unexpected status {oracle_res['status']} on {mps_name}"
    )

    # 3. Solve with indigenous SimplexSolver
    opts = sih_solver.Options()
    opts.strategy.presolve = sih_solver.PresolveMode.On
    opts.strategy.enable_scaling = True

    t0_indig = time.perf_counter()
    sol = sih_solver.solve(p, opts)
    t1_indig = time.perf_counter()
    indig_time = t1_indig - t0_indig

    # Check status match
    assert sol.status.name == oracle_res["status"], (
        f"Status mismatch on {mps_name}: indigenous={sol.status.name}, HiGHS={oracle_res['status']}"
    )

    # If Optimal: Check objective match within 1e-6 relative tolerance
    if oracle_res["status"] == "Optimal":
        highs_obj = oracle_res["objective_value"]
        indig_obj = sol.primal_objective
        rel_diff = abs(indig_obj - highs_obj) / (1.0 + abs(highs_obj))

        print(f"\n[Netlib: {mps_name}]")
        print(f"  Rows: {p.num_rows()}, Cols: {p.num_cols()}")
        print(f"  HiGHS Obj:      {highs_obj:.8e} (iters: {oracle_res['simplex_iterations']}, time: {highs_time*1000:.2f}ms)")
        print(f"  Indigenous Obj: {indig_obj:.8e} (iters: {sol.simplex_iterations}, time: {indig_time*1000:.2f}ms)")
        print(f"  Rel Diff:       {rel_diff:.4e}")

        assert rel_diff <= 1e-6, (
            f"Objective mismatch on {mps_name}: indigenous={indig_obj}, HiGHS={highs_obj}, rel_diff={rel_diff}"
        )

        # 4. Independent checker verification
        chk = sih_solver.check_solution(p, sol)
        assert chk.all_checks_passed, f"C++ check_solution failed on {mps_name}: {chk.summary}"

        fc = FloatChecker.check_solution(p, sol.x, sol.row_duals, sol.reduced_costs, sol.primal_objective)
        assert fc["valid"], f"FloatChecker error on {mps_name}: {fc.get('error')}"
        assert fc["is_primal_feasible"], f"Primal infeasible on {mps_name}: max_viol={fc['max_primal_residual']}"
        assert fc["all_passed"], f"FloatChecker checks failed on {mps_name}: {fc}"
