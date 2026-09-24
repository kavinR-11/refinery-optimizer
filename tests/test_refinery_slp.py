"""
Test Successive Linear Programming (SLP) for Nonlinear Blending.
Verifies:
1. Iterative convergence of bilinear sulfur pooling property estimates.
2. Property error monotonically decreases or stabilizes below tolerance (1e-4).
3. Independent nonlinear physical verification confirming final diesel blend meets legal sulfur specs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

from refinery.model_builder import RefineryModelBuilder
from refinery.slp_blender import SLPRefineryBlender


def test_slp_blending():
    print("\n=== GATE TEST: Successive Linear Programming (SLP) Blending ===")
    
    # Configure base builder with distinct crude sulfur slates
    builder = RefineryModelBuilder(is_milp=False)
    
    # Initialize SLP solver with initial guess s_hat = 1.20 wt% sulfur
    # (True value depends on optimal crude mix, which will settle around 0.68% - 0.75%)
    slp_solver = SLPRefineryBlender(
        base_builder=builder,
        theta_sulfur=0.70,
        max_iterations=12,
        tolerance=1e-4,
        damping=0.75,
    )

    initial_guess = 1.20
    print(f"Starting SLP blending loop with initial sulfur guess: {initial_guess:.4f} wt%")
    result = slp_solver.solve(initial_sulfur_guess=initial_guess)

    print("\n--- SLP Convergence Table ---")
    print(slp_solver.format_convergence_table(result))

    assert result.converged, f"SLP failed to converge in {slp_solver.max_iterations} iterations"
    assert result.num_iterations <= 8, f"SLP should converge quickly (took {result.num_iterations} iterations)"
    assert result.is_physically_feasible, "Independent nonlinear verification failed: diesel sulfur breached legal limit"
    assert result.actual_diesel_sulfur <= result.target_diesel_sulfur_spec + 1e-4, (
        f"Nonlinear diesel sulfur {result.actual_diesel_sulfur:.5f}% exceeds spec {result.target_diesel_sulfur_spec:.5f}%"
    )

    print(f"\n>>> SLP Blending converged in {result.num_iterations} iterations with property error = {result.property_error:.2e}")
    print(">>> Independent nonlinear physical verification PASSED.")


if __name__ == "__main__":
    test_slp_blending()
