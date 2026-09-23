"""
Comprehensive Gate Test: Toy Suite Oracle & Independent Checker Cross-Validation.
Loads all 10 hand-crafted toy problems through sih_solver.read_mps, solves with HiGHS oracle,
and validates solution feasibility and objective with the independent float and rational checkers.
"""

import json
from pathlib import Path
import pytest
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "tests"))

import sih_solver
from oracle import HighsOracle
from checker import FloatChecker, RationalChecker

# Load suite metadata
META_PATH = ROOT / "data" / "toy" / "toy_suite_metadata.json"
with open(META_PATH) as f:
    TOY_META = json.load(f)["problems"]

@pytest.mark.parametrize("prob_def", TOY_META, ids=[p["name"] for p in TOY_META])
def test_toy_problem(prob_def):
    mps_path = str(ROOT / "data" / "toy" / prob_def["file"])
    expected_status = prob_def["expected_status"]
    expected_obj = prob_def["expected_objective"]

    # 1. Load problem through indigenous MPS reader
    p = sih_solver.read_mps(mps_path)
    assert p.num_rows() >= 0
    assert p.num_cols() > 0

    # 2. Solve with HiGHS oracle
    oracle_res = HighsOracle.solve_mps(mps_path)
    assert oracle_res["status"] == expected_status, (
        f"Status mismatch on {prob_def['name']}: got {oracle_res['status']}, expected {expected_status}"
    )

    # 3. If optimal, cross-validate with independent checkers
    if expected_status == "Optimal":
        sol_x = oracle_res["primal_solution"]
        sol_obj = oracle_res["objective_value"]
        sol_y = oracle_res["row_duals"]
        sol_s = oracle_res["reduced_costs"]

        assert sol_x is not None and len(sol_x) == p.num_cols()
        assert abs(sol_obj - expected_obj) < 1e-4 * (1.0 + abs(expected_obj)), (
            f"Objective mismatch on {prob_def['name']}: got {sol_obj}, expected {expected_obj}"
        )

        # Run Independent Float Checker
        chk_res = FloatChecker.check_solution(
            problem=p,
            x=sol_x,
            row_duals=sol_y,
            reduced_costs=sol_s,
            reported_obj=sol_obj
        )
        assert chk_res["valid"], f"FloatChecker error: {chk_res}"
        assert chk_res["is_primal_feasible"], (
            f"Primal infeasible on {prob_def['name']}: viol={chk_res['max_primal_residual']}"
        )
        assert chk_res["is_integrality_satisfied"], (
            f"Integrality violated on {prob_def['name']}: viol={chk_res['max_integrality_violation']}"
        )
        assert chk_res["all_passed"], f"Checker checks failed: {chk_res}"

        # Run Exact Rational Arithmetic Checker
        rat_res = RationalChecker.check_solution(
            problem=p,
            x_float=sol_x,
            reported_obj=sol_obj
        )
        assert rat_res["valid"], f"RationalChecker error: {rat_res}"
        assert rat_res["all_passed"], (
            f"Rational checker failed on {prob_def['name']}: "
            f"row_viols={rat_res['row_violations']}, col_viols={rat_res['col_violations']}, "
            f"int_viols={rat_res['int_violations']}"
        )
