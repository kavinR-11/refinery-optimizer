import sys
from pathlib import Path
import pytest

# Ensure py/ and tests/ are in sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from checker import FloatChecker, RationalChecker

def test_checker_lp_feasible():
    p = sih_solver.Problem("checker_py_test")
    p.resize(2, 2)
    p.c = [1.0, 2.0]
    p.row_lower = [3.0, -1e30]
    p.row_upper = [1e30, 1.0]

    trips = [
        sih_solver.Triplet(0, 0, 1.0), sih_solver.Triplet(0, 1, 1.0),
        sih_solver.Triplet(1, 0, 1.0), sih_solver.Triplet(1, 1, -1.0)
    ]
    p.A = sih_solver.SparseMatrix.from_triplets(2, 2, trips)

    # Feasible solution: x = [2, 1], obj = 4
    x = [2.0, 1.0]
    res_float = FloatChecker.check_solution(p, x, reported_obj=4.0)
    assert res_float["valid"]
    assert res_float["all_passed"]
    assert res_float["is_primal_feasible"]
    assert abs(res_float["evaluated_objective"] - 4.0) < 1e-9

    res_rat = RationalChecker.check_solution(p, x, reported_obj=4.0)
    assert res_rat["valid"]
    assert res_rat["all_passed"]
    assert res_rat["exact_objective_fraction"] == "4"

def test_checker_lp_infeasible():
    p = sih_solver.Problem("checker_py_infeas")
    p.resize(2, 2)
    p.c = [1.0, 2.0]
    p.row_lower = [3.0, -1e30]
    p.row_upper = [1e30, 1.0]

    trips = [
        sih_solver.Triplet(0, 0, 1.0), sih_solver.Triplet(0, 1, 1.0),
        sih_solver.Triplet(1, 0, 1.0), sih_solver.Triplet(1, 1, -1.0)
    ]
    p.A = sih_solver.SparseMatrix.from_triplets(2, 2, trips)

    # Infeasible: x = [0, 0]
    x = [0.0, 0.0]
    res_float = FloatChecker.check_solution(p, x, reported_obj=0.0)
    assert res_float["valid"]
    assert not res_float["is_primal_feasible"]
    assert not res_float["all_passed"]

    res_rat = RationalChecker.check_solution(p, x, reported_obj=0.0)
    assert res_rat["valid"]
    assert not res_rat["all_passed"]
    assert res_rat["row_violations_count"] > 0
