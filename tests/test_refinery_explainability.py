"""
Test Refinery Explainability: Shadow Prices and Infeasibility Diagnosis (IIS).
Verifies:
1. Shadow price on binding capacity is non-zero.
2. Shadow price on slack capacity is zero.
3. Infeasibility diagnosis correctly identifies minimal conflicting set in plain English.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver
from refinery.model_builder import RefineryModelBuilder
from refinery.explainability import RefineryExplainer, InfeasibilityDiagnoser


def test_shadow_prices():
    print("\n=== GATE TEST: Shadow Price Economic Explainability ===")
    builder = RefineryModelBuilder(is_milp=False)
    prob, meta = builder.build_problem()

    options = sih_solver.Options()
    options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
    options.log_to_console = False
    sol = sih_solver.solve(prob, options)

    assert sol.is_optimal(), "Refinery LP failed to solve to optimality"

    explainer = RefineryExplainer(prob, sol, meta)
    explanations = explainer.explain_shadow_prices()

    # Find ADU capacity constraint (known to be binding at 100,000 bpd)
    adu_exp = next((e for e in explanations if e["name"] == "cap_max_ADU"), None)
    assert adu_exp is not None, "cap_max_ADU not found"
    print(f"ADU Constraint: slack={adu_exp['slack']:.2f}, shadow_price={adu_exp['shadow_price']:.2f}")
    assert adu_exp["is_binding"], "ADU should be binding at 100,000 bpd capacity"
    assert adu_exp["shadow_price"] > 0.0, f"Binding ADU capacity must have positive shadow price, got {adu_exp['shadow_price']}"

    # Find VDU or FCC capacity constraint (known to have slack in baseline)
    fcc_exp = next((e for e in explanations if e["name"] == "cap_max_FCC"), None)
    assert fcc_exp is not None, "cap_max_FCC not found"
    print(f"FCC Constraint: slack={fcc_exp['slack']:.2f}, shadow_price={fcc_exp['shadow_price']:.2f}")
    assert not fcc_exp["is_binding"], "FCC should be slack"
    assert abs(fcc_exp["shadow_price"]) < 1e-4, f"Slack FCC capacity must have zero shadow price, got {fcc_exp['shadow_price']}"

    print("\n--- Formatted Bottlenecks Table ---")
    print(explainer.format_bottlenecks_table())

    print("\n--- Formatted Executive Summary ---")
    print(explainer.format_executive_summary())
    print("\n>>> Shadow price verification PASSED.")


def test_infeasibility_diagnosis():
    print("\n=== GATE TEST: Infeasibility Diagnosis (IIS Deletion Filtering) ===")
    builder = RefineryModelBuilder(is_milp=False)
    
    # Deliberately construct an impossible scenario:
    # Set Diesel minimum demand to 90,000 bpd (exceeds total possible distillate yield of 32,000 bpd from 100k ADU)
    builder.products["Diesel"]["min_demand"] = 90000.0
    builder.products["Diesel"]["max_demand"] = 120000.0
    
    prob, meta = builder.build_problem()

    options = sih_solver.Options()
    options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
    options.log_to_console = False
    sol = sih_solver.solve(prob, options)

    print(f"Solver status on impossible model: {sol.status}")
    assert not sol.is_feasible(), "Model was expected to be infeasible"

    # Diagnose infeasibility using IIS deletion filtering
    diagnoser = InfeasibilityDiagnoser()
    diagnosis = diagnoser.diagnose(prob, meta)

    assert diagnosis["is_infeasible"], "Diagnoser should detect infeasibility"
    assert diagnosis["conflict_size"] > 0, "Conflict set should not be empty"

    print("\n--- Infeasibility Diagnosis Output ---")
    print(f"Category: {diagnosis['category']}")
    print(f"Minimal Conflicting Constraints ({diagnosis['conflict_size']} rows):")
    for r_name in diagnosis["conflicting_row_names"]:
        print(f"  - {r_name}")
    print("\nPlain-Language Explanation:")
    print(diagnosis["plain_language_narrative"])
    print("\nActionable Managerial Remedies:")
    for remedy in diagnosis["actionable_remedies"]:
        print(f"  * {remedy}")

    print("\n>>> Infeasibility diagnosis PASSED.")


if __name__ == "__main__":
    test_shadow_prices()
    test_infeasibility_diagnosis()
