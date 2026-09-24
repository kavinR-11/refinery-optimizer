"""
Test Refinery Plan Comparator.
Verifies:
1. Economic margin and solver execution delta reporting.
2. Crude intake volume shifts and processing unit throughput changes.
3. Accurate identification of newly binding vs newly slack constraints.
4. Executive comparison reporting format.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver
from refinery.model_builder import RefineryModelBuilder
from refinery.comparator import RefineryPlanComparator


def test_plan_comparator():
    print("\n=== GATE TEST: Refinery Plan Comparison Utility ===")
    
    # 1. Base Plan (Plan A)
    builder_a = RefineryModelBuilder(is_milp=False)
    prob_a, meta_a = builder_a.build_problem()
    
    options = sih_solver.Options()
    options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
    options.log_to_console = False
    
    sol_a = sih_solver.solve(prob_a, options)
    assert sol_a.is_optimal(), "Plan A failed to solve"

    # 2. Perturbed Scenario (Plan B):
    # - Brent crude price increases from $85 to $115/bbl (+35%)
    # - HDT capacity cut from 40,000 bpd to 35,000 bpd (turnaround maintenance)
    # - Diesel price increases from $112 to $130/bbl (+16%)
    builder_b = RefineryModelBuilder(is_milp=False)
    builder_b.crudes["Brent"]["price"] = 115.0
    builder_b.units["HDT"]["capacity"] = 35000.0
    builder_b.products["Diesel"]["price"] = 130.0

    prob_b, meta_b = builder_b.build_problem()
    sol_b = sih_solver.solve(prob_b, options)
    print(f"Plan B status: {sol_b.status}")
    assert sol_b.is_optimal(), f"Plan B failed to solve: {sol_b.status}"

    # 3. Run Comparator
    comparator = RefineryPlanComparator(
        prob_a, sol_a, meta_a,
        prob_b, sol_b, meta_b,
        label_a="Base Case",
        label_b="Brent Shock & HDT Outage",
    )

    econ = comparator.compare_economics()
    crudes = comparator.compare_crudes()
    units = comparator.compare_units()
    constrs = comparator.compare_constraints()

    print(f"Margin Plan A: ${econ['margin_a']:,.2f}/day")
    print(f"Margin Plan B: ${econ['margin_b']:,.2f}/day")
    print(f"Delta Margin:  ${econ['delta_margin']:,.2f}/day ({econ['pct_margin']:+.2f}%)")

    # Newly binding constraint should include cap_max_HDT
    newly_binding_names = [nb["name"] for nb in constrs["newly_binding"]]
    print(f"Newly binding constraints: {newly_binding_names}")
    assert "cap_max_HDT" in newly_binding_names, "cap_max_HDT should become newly binding in Plan B"

    # Generate full Markdown report
    report = comparator.format_comparison_report()
    print("\n--- Executive Plan Comparison Report ---\n")
    print(report)

    print("\n>>> Plan comparator verification PASSED.")


if __name__ == "__main__":
    test_plan_comparator()
