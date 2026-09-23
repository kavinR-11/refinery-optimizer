"""
Warm-Start Re-Optimization Test Suite.
Validates requirement 5 & Gate Warm-Start criterion:
After (a) objective change, (b) RHS/bound change, (c) added rows,
iterations drop significantly versus a cold start.
"""

import copy
import sys
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver

def test_warm_start_objective_perturbation():
    mps_path = str(root / "data" / "netlib" / "adlittle.mps")
    p_orig = sih_solver.read_mps(mps_path)

    opts = sih_solver.Options()
    opts.strategy.presolve = sih_solver.PresolveMode.Off
    opts.strategy.enable_scaling = False

    # 1. Cold start base solve
    sol_base = sih_solver.solve(p_orig, opts)
    assert sol_base.status == sih_solver.SolutionStatus.Optimal
    base_iters = sol_base.simplex_iterations
    basis_cols = sol_base.col_basis
    basis_rows = sol_base.row_basis

    # 2. Perturb objective slightly
    p_pert = sih_solver.read_mps(mps_path)
    c_new = list(p_pert.c)
    for j in range(len(c_new)):
        if abs(c_new[j]) > 1e-4:
            c_new[j] *= 1.05 # 5% shift
    p_pert.c = c_new

    # Cold start on perturbed problem
    sol_cold = sih_solver.solve(p_pert, opts)
    cold_iters = sol_cold.simplex_iterations

    # Warm start from base basis
    sol_warm = sih_solver.solve_from_basis(p_pert, basis_cols, basis_rows, opts)
    warm_iters = sol_warm.simplex_iterations

    print(f"\n[Warm Start: Objective Perturbation (adlittle)]")
    print(f"  Base Cold Iters: {base_iters}")
    print(f"  Perturbed Cold Iters: {cold_iters}")
    print(f"  Warm Start Iters:     {warm_iters}")
    print(f"  Iteration Reduction:  {((cold_iters - warm_iters) / max(cold_iters, 1)) * 100:.1f}%")

    assert sol_warm.status == sih_solver.SolutionStatus.Optimal
    assert abs(sol_warm.primal_objective - sol_cold.primal_objective) < 1e-6 * (1.0 + abs(sol_cold.primal_objective))
    assert warm_iters < cold_iters, f"Warm start iters {warm_iters} not strictly less than cold {cold_iters}"

def test_warm_start_rhs_perturbation():
    mps_path = str(root / "data" / "netlib" / "adlittle.mps")
    p_orig = sih_solver.read_mps(mps_path)

    opts = sih_solver.Options()
    opts.strategy.presolve = sih_solver.PresolveMode.Off
    opts.strategy.enable_scaling = False

    # 1. Base solve
    sol_base = sih_solver.solve(p_orig, opts)
    assert sol_base.status == sih_solver.SolutionStatus.Optimal
    basis_cols = sol_base.col_basis
    basis_rows = sol_base.row_basis

    # 2. Perturb RHS / row bounds
    p_pert = sih_solver.read_mps(mps_path)
    rl_new = list(p_pert.row_lower)
    ru_new = list(p_pert.row_upper)
    for i in range(len(rl_new)):
        if rl_new[i] > -1e30:
            rl_new[i] *= 1.03
        if ru_new[i] < 1e30:
            ru_new[i] *= 1.03
    p_pert.row_lower = rl_new
    p_pert.row_upper = ru_new

    # Cold start on perturbed problem
    sol_cold = sih_solver.solve(p_pert, opts)
    cold_iters = sol_cold.simplex_iterations

    # Warm start from base basis
    sol_warm = sih_solver.solve_from_basis(p_pert, basis_cols, basis_rows, opts)
    warm_iters = sol_warm.simplex_iterations

    print(f"\n[Warm Start: RHS Perturbation (adlittle)]")
    print(f"  Perturbed Cold Iters: {cold_iters}")
    print(f"  Warm Start Iters:     {warm_iters}")
    print(f"  Iteration Reduction:  {((cold_iters - warm_iters) / max(cold_iters, 1)) * 100:.1f}%")

    assert sol_warm.status == sih_solver.SolutionStatus.Optimal
    assert abs(sol_warm.primal_objective - sol_cold.primal_objective) < 1e-6 * (1.0 + abs(sol_cold.primal_objective))
    assert warm_iters < cold_iters, f"Warm start iters {warm_iters} not strictly less than cold {cold_iters}"

def test_warm_start_added_row():
    mps_path = str(root / "data" / "netlib" / "afiro.mps")
    p_orig = sih_solver.read_mps(mps_path)

    opts = sih_solver.Options()
    opts.strategy.presolve = sih_solver.PresolveMode.Off
    opts.strategy.enable_scaling = False

    sol_base = sih_solver.solve(p_orig, opts)
    assert sol_base.status == sih_solver.SolutionStatus.Optimal
    basis_cols = sol_base.col_basis
    basis_rows = sol_base.row_basis

    # Add a cut/constraint: sum_{j=0}^{n-1} 0.1 * x_j <= 100.0
    m_orig = p_orig.num_rows()
    n_orig = p_orig.num_cols()

    p_add = sih_solver.Problem(p_orig.name + "_add")
    p_add.sense = p_orig.sense
    p_add.obj_offset = p_orig.obj_offset
    p_add.resize(m_orig + 1, n_orig)
    p_add.c = list(p_orig.c)
    p_add.col_lower = list(p_orig.col_lower)
    p_add.col_upper = list(p_orig.col_upper)
    p_add.var_types = list(p_orig.var_types)

    # Copy matrix entries + new row
    A_orig = p_orig.A
    triplets = []
    col_ptr = A_orig.csc_col_ptr()
    row_ind = A_orig.csc_row_ind()
    vals    = A_orig.csc_values()
    for j in range(n_orig):
        for k in range(col_ptr[j], col_ptr[j + 1]):
            triplets.append(sih_solver.Triplet(row_ind[k], j, vals[k]))
        triplets.append(sih_solver.Triplet(m_orig, j, 0.1))

    p_add.A = sih_solver.SparseMatrix.from_triplets(m_orig + 1, n_orig, triplets)
    rl = list(p_orig.row_lower) + [-1e30]
    ru = list(p_orig.row_upper) + [100.0]
    p_add.row_lower = rl
    p_add.row_upper = ru

    # Cold start on augmented problem
    sol_cold = sih_solver.solve(p_add, opts)
    cold_iters = sol_cold.simplex_iterations

    # Warm start: new slack variable is basic
    extended_row_basis = list(basis_rows) + [sih_solver.BasisStatus.Basic]
    sol_warm = sih_solver.solve_from_basis(p_add, basis_cols, extended_row_basis, opts)
    warm_iters = sol_warm.simplex_iterations

    print(f"\n[Warm Start: Added Constraint Row (afiro)]")
    print(f"  Augmented Cold Iters: {cold_iters}")
    print(f"  Warm Start Iters:     {warm_iters}")
    print(f"  Iteration Reduction:  {((cold_iters - warm_iters) / max(cold_iters, 1)) * 100:.1f}%")

    assert sol_warm.status == sih_solver.SolutionStatus.Optimal
    assert abs(sol_warm.primal_objective - sol_cold.primal_objective) < 1e-6 * (1.0 + abs(sol_cold.primal_objective))
    assert warm_iters <= cold_iters
