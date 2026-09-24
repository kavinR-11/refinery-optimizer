import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))

import sih_solver
from refinery.model_builder import RefineryModelBuilder

def test_refinery_milp():
    builder = RefineryModelBuilder(is_milp=True)
    p, meta = builder.build_problem()
    
    print(f"Refinery MILP Model: Rows={p.num_rows()}, Cols={p.num_cols()}, Integers={p.num_integers()}")
    assert p.is_mip()
    
    sol = sih_solver.solve(p)
    print(f"MILP Solve status: {sol.status.name}")
    print(f"Gross margin: ${sol.primal_objective:,.2f}/day")
    print(f"Nodes explored: {sol.nodes_explored}")
    
    assert sol.status == sih_solver.SolutionStatus.Optimal
    assert sol.primal_objective > 0.0
    
    # Check binary decisions
    print("\nUnit On/Off Decisions & Turndown:")
    for u_name in builder.units:
        z_idx = meta["var_map"][f"z_{u_name}"]
        u_idx = meta["var_map"][f"unit_{u_name}"]
        z_val = round(sol.x[z_idx])
        u_val = sol.x[u_idx]
        min_td = builder.units[u_name]["min_turndown"]
        cap = builder.units[u_name]["capacity"]
        print(f"  Unit {u_name:<5}: Active={z_val} (z={sol.x[z_idx]:.2f}), Throughput={u_val:10.1f} bpd (min_turndown={min_td:8.1f}, cap={cap:8.1f})")
        assert z_val in (0, 1)
        if z_val == 1:
            assert u_val >= min_td - 1e-4
            assert u_val <= cap + 1e-4
        else:
            assert u_val <= 1e-4

if __name__ == "__main__":
    test_refinery_milp()
