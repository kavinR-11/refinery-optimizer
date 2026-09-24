"""
Test Parameterized Refinery Model Generator.
Verifies:
1. Scenario generation across diverse market regimes (normal, high crude, diesel surge, turnaround, sour crude).
2. Reproducibility using seeds.
3. Dataset generation with train/test partitioning for learning layer and benchmarking.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver
from refinery.parameterized_generator import ParameterizedRefineryGenerator


def test_parameterized_generator():
    print("\n=== GATE TEST: Parameterized Refinery Generator ===")
    gen = ParameterizedRefineryGenerator()

    # 1. Test Regime Generation
    regimes = ["normal", "high_crude", "diesel_surge", "turnaround", "sour_crude"]
    options = sih_solver.Options()
    options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
    options.log_to_console = False

    for regime in regimes:
        prob, meta, params = gen.generate_instance(seed=42, market_regime=regime, is_milp=False)
        assert prob.num_rows() == 29
        assert prob.num_cols() == 34
        
        sol = sih_solver.solve(prob, options)
        print(f"Regime '{regime:15s}': Status={sol.status}, Margin=${sol.primal_objective:,.2f}/day")
        assert sol.is_optimal(), f"Regime {regime} failed to solve"

    # 2. Test Reproducibility
    prob1, _, _ = gen.generate_instance(seed=777, market_regime="normal")
    prob2, _, _ = gen.generate_instance(seed=777, market_regime="normal")
    assert prob1.c == prob2.c, "Seed reproducibility failed on objective coefficients"
    assert prob1.col_upper == prob2.col_upper, "Seed reproducibility failed on column bounds"

    # 3. Test Dataset Generation (Train/Test Partitioning)
    dataset = gen.generate_dataset(num_samples=10, train_split=0.8, base_seed=5000, is_milp=False)
    assert len(dataset["train"]) == 8, f"Expected 8 train samples, got {len(dataset['train'])}"
    assert len(dataset["test"]) == 2, f"Expected 2 test samples, got {len(dataset['test'])}"
    print(f"\nGenerated dataset: {len(dataset['train'])} train instances, {len(dataset['test'])} test instances")
    
    # 4. Test Parameterized MILP Generation
    prob_milp, meta_milp, _ = gen.generate_instance(seed=999, market_regime="normal", is_milp=True)
    assert prob_milp.is_mip(), "MILP instance should be flagged as MIP"
    assert prob_milp.num_binaries() == 5, f"Expected 5 binaries, got {prob_milp.num_binaries()}"
    
    sol_milp = sih_solver.solve(prob_milp, options)
    print(f"Parameterized MILP solve: Status={sol_milp.status}, Margin=${sol_milp.primal_objective:,.2f}/day, Nodes={sol_milp.nodes_explored}")
    assert sol_milp.is_optimal(), "Parameterized MILP failed to solve"

    print("\n>>> Parameterized generator verification PASSED.")


if __name__ == "__main__":
    test_parameterized_generator()
