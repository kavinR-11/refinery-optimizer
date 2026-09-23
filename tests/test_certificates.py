"""
Phase 1 Certificate & Sensitivity Range Verification.
Tests Farkas rays for infeasible LPs, unbounded rays for unbounded LPs,
and sensitivity ranges for optimal bases.
"""

import sys
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from checker import FloatChecker

def test_infeasible_farkas_certificate():
    mps_path = str(root / "data" / "toy" / "toy02_lp_infeasible.mps")
    p = sih_solver.read_mps(mps_path)

    sol = sih_solver.solve(p)
    assert sol.status == sih_solver.SolutionStatus.Infeasible, f"Expected Infeasible, got {sol.status}"
    assert len(sol.ray) == p.num_rows(), f"Ray size {len(sol.ray)} != rows {p.num_rows()}"

    # 1. Independent C++ Checker verification
    chk = sih_solver.check_solution(p, sol)
    assert chk.all_checks_passed, f"C++ check_solution failed on Farkas ray: {chk.summary}"
    assert chk.is_certificate_valid, "C++ checker failed to certify Farkas ray"

    # 2. Independent Python FloatChecker verification
    fc_res = FloatChecker.check_farkas_ray(p, sol.ray)
    assert fc_res["valid"], f"FloatChecker Farkas error: {fc_res.get('error')}"
    assert fc_res["all_passed"], f"FloatChecker Farkas certificate check failed: {fc_res}"

def test_unbounded_ray_certificate():
    mps_path = str(root / "data" / "toy" / "toy03_lp_unbounded.mps")
    p = sih_solver.read_mps(mps_path)

    sol = sih_solver.solve(p)
    assert sol.status == sih_solver.SolutionStatus.Unbounded, f"Expected Unbounded, got {sol.status}"
    assert len(sol.ray) == p.num_cols(), f"Ray size {len(sol.ray)} != cols {p.num_cols()}"

    # 1. Independent C++ Checker verification
    chk = sih_solver.check_solution(p, sol)
    assert chk.all_checks_passed, f"C++ check_solution failed on Unbounded ray: {chk.summary}"
    assert chk.is_certificate_valid, "C++ checker failed to certify Unbounded ray"

    # 2. Independent Python FloatChecker verification
    fc_res = FloatChecker.check_unbounded_ray(p, sol.ray)
    assert fc_res["valid"], f"FloatChecker Unbounded error: {fc_res.get('error')}"
    assert fc_res["all_passed"], f"FloatChecker Unbounded certificate check failed: {fc_res}"

def test_optimal_sensitivity_ranges():
    mps_path = str(root / "data" / "toy" / "toy01_lp_simple.mps")
    p = sih_solver.read_mps(mps_path)

    sol = sih_solver.solve(p)
    assert sol.status == sih_solver.SolutionStatus.Optimal, f"Expected Optimal, got {sol.status}"

    assert len(sol.rhs_down) == p.num_rows()
    assert len(sol.rhs_up) == p.num_rows()
    assert len(sol.obj_down) == p.num_cols()
    assert len(sol.obj_up) == p.num_cols()

    # Verify that ranges bracket current values or infinities
    for i in range(p.num_rows()):
        assert sol.rhs_down[i] <= sol.rhs_up[i], f"Row {i} inverted range [{sol.rhs_down[i]}, {sol.rhs_up[i]}]"

    for j in range(p.num_cols()):
        assert sol.obj_down[j] <= sol.obj_up[j], f"Col {j} inverted range [{sol.obj_down[j]}, {sol.obj_up[j]}]"
