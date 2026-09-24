import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))

import sih_solver
from refinery.model_builder import RefineryModelBuilder

def test_base_refinery_lp():
    builder = RefineryModelBuilder(is_milp=False)
    p, meta = builder.build_problem()
    
    print(f"Refinery LP Model: Rows={p.num_rows()}, Cols={p.num_cols()}, Nonzeros={p.num_nonzeros()}")
    
    sol = sih_solver.solve(p)
    print(f"Solve status: {sol.status.name}")
    print(f"Gross margin: ${sol.primal_objective:,.2f}/day")
    
    assert sol.status == sih_solver.SolutionStatus.Optimal
    assert sol.primal_objective > 0.0
    
    # Check that crude intakes do not exceed supply
    for c_name in builder.crudes:
        idx = meta["var_map"][f"crude_{c_name}"]
        val = sol.x[idx]
        max_s = builder.crudes[c_name]["max_supply"]
        print(f"  Crude {c_name:<10}: {val:10.1f} bpd (max {max_s:10.1f})")
        assert val <= max_s + 1e-4
        
    # Check product outputs
    print("\nProduct Deliveries:")
    for p_name in builder.products:
        idx = meta["var_map"][f"prod_{p_name}"]
        val = sol.x[idx]
        min_d = builder.products[p_name]["min_demand"]
        max_d = builder.products[p_name]["max_demand"]
        print(f"  Product {p_name:<18}: {val:10.1f} bpd (range [{min_d}, {max_d}])")
        assert val >= min_d - 1e-4
        assert val <= max_d + 1e-4

    # Check unit throughputs
    print("\nUnit Operations:")
    for u_name in builder.units:
        idx = meta["var_map"][f"unit_{u_name}"]
        val = sol.x[idx]
        cap = builder.units[u_name]["capacity"]
        print(f"  Unit {u_name:<5}: {val:10.1f} bpd / {cap:10.1f} bpd ({val/cap*100:5.1f}% util)")
        assert val <= cap + 1e-4

if __name__ == "__main__":
    test_base_refinery_lp()
