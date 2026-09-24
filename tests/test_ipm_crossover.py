import sys
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from oracle import HighsOracle

def test_ipm_crossover_basis_validity():
    mps_path = root / "data" / "netlib" / "afiro.mps"
    p = sih_solver.read_mps(str(mps_path))
    
    opts = sih_solver.Options()
    opts.strategy.ipm_enable_crossover = True
    
    sol = sih_solver.solve_ipm(p, opts)
    assert sol.is_optimal()
    
    # Verify basis size
    assert len(sol.col_basis) == p.num_cols()
    assert len(sol.row_basis) == p.num_rows()
    
    # Count basic variables
    basic_cols = sum(1 for b in sol.col_basis if b == sih_solver.BasisStatus.Basic)
    basic_rows = sum(1 for b in sol.row_basis if b == sih_solver.BasisStatus.Basic)
    assert basic_cols + basic_rows == p.num_rows(), f"Basis must have exactly m={p.num_rows()} basic variables, got {basic_cols + basic_rows}"
    
    # Warm-start Phase 1 Simplex directly with this crossover basis
    opts_warm = sih_solver.Options()
    sol_warm = sih_solver.SimplexSolver.solve_from_basis(p, sol.col_basis, sol.row_basis, opts_warm)
    assert sol_warm.is_optimal()
    
    # Simplex should take very few iterations from the clean crossover basis
    assert sol_warm.simplex_iterations <= 10
    
    # Check objective matches oracle
    h_res = HighsOracle.solve_mps(str(mps_path))
    rel_diff = abs(sol_warm.primal_objective - h_res["objective_value"]) / max(1.0, abs(h_res["objective_value"]))
    assert rel_diff < 1e-6

if __name__ == "__main__":
    test_ipm_crossover_basis_validity()
    print("Crossover basis validity test PASSED!")
