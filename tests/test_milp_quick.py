import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
import sih_solver

def test_milp_simple():
    # Minimize -x0 - 2*x1 s.t. x0 + x1 <= 3.5, x0, x1 >= 0, integer
    # LP relaxation gives x0 = 0, x1 = 3.5 (obj = -7.0)
    # Integer optimal gives x0 = 0, x1 = 3.0 (obj = -6.0)
    p = sih_solver.Problem()
    p.resize(1, 2)
    p.c = [-1.0, -2.0]
    p.row_lower = [-float("inf")]
    p.row_upper = [3.5]
    p.col_lower = [0.0, 0.0]
    p.col_upper = [10.0, 10.0]
    p.var_types = [sih_solver.VariableType.Integer, sih_solver.VariableType.Integer]

    trips = [sih_solver.Triplet(0, 0, 1.0), sih_solver.Triplet(0, 1, 1.0)]
    p.A = sih_solver.SparseMatrix.from_triplets(1, 2, trips)

    sol = sih_solver.solve_milp(p)
    print("Status:", sol.status.name, "obj:", sol.primal_objective, "x:", list(sol.x), "nodes:", sol.nodes_explored)
    assert sol.status == sih_solver.SolutionStatus.Optimal
    assert abs(sol.primal_objective - (-6.0)) < 1e-5
    assert abs(sol.x[0] - 0.0) < 1e-5
    assert abs(sol.x[1] - 3.0) < 1e-5
    print("test_milp_simple PASSED!")

if __name__ == "__main__":
    test_milp_simple()
