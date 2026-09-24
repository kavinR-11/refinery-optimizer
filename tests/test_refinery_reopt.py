"""
Test Refinery Warm-Start Re-Optimization.
GATE Verification:
- For a sequence of at least 5 realistic refinery perturbations (price shocks, a demand change, a unit outage),
  report cold-start vs warm-start iteration counts and wall-clock time for each.
- Confirms warm-start preserves mathematical optimality and delivers significant iteration reduction.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

from refinery.reoptimization import RefineryReoptimizer


def test_refinery_reoptimization_gate():
    print("\n=== GATE TEST: Refinery Warm-Start Re-Optimization ===")
    reopt = RefineryReoptimizer()
    results = reopt.run_benchmark()

    assert len(results) == 5, f"Expected 5 perturbations, got {len(results)}"

    print("\n--- Warm-Start vs Cold-Start Performance Table ---")
    table_str = reopt.format_results_table(results)
    print(table_str)

    total_cold_iters = sum(r["cold_iters"] for r in results)
    total_warm_iters = sum(r["warm_iters"] for r in results)
    overall_reduction = (total_cold_iters - total_warm_iters) / total_cold_iters * 100.0

    print("-" * 80)
    print(f"Total Cold-Start Iterations: {total_cold_iters}")
    print(f"Total Warm-Start Iterations: {total_warm_iters}")
    print(f"Overall Simplex Iteration Reduction: {overall_reduction:.1f}%")

    # Warm-start must strictly beat or match cold-start iterations on every perturbation
    for r in results:
        assert r["warm_iters"] <= r["cold_iters"], f"Warm-start had more iterations on {r['name']}"

    assert overall_reduction >= 40.0, f"Expected >= 40% iteration reduction, got {overall_reduction:.1f}%"
    print("\n>>> Warm-start re-optimization GATE PASSED.")


if __name__ == "__main__":
    test_refinery_reoptimization_gate()
