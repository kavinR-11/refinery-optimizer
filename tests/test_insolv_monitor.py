"""
Test In-Solve Stagnation Monitor for MILP.
GATE Verification:
- Demonstrate at least one MILP instance where the monitor detects a stall and switches strategy,
  with before/after node-processing-rate numbers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver


def test_insolv_monitor_gate():
    print("\n=== GATE TEST: In-Solve MILP Stagnation Monitor ===")
    
    # Try MIPLIB problem or parameterized MILP
    # pk1 or flugpl or markshare1
    test_files = ["data/miplib/flugpl.mps", "data/miplib/markshare1.mps", "data/miplib/pk1.mps"]
    
    selected_prob = None
    for f in test_files:
        if os.path.exists(f):
            p = sih_solver.read_mps(f)
            if p.num_rows() > 0:
                selected_prob = p
                print(f"Selected benchmark instance: {f} (Rows={p.num_rows()}, Cols={p.num_cols()})")
                break

    assert selected_prob is not None, "No benchmark MIP instance found"

    opts = sih_solver.Options()
    opts.node_limit = 300
    opts.time_limit_sec = 5.0
    opts.strategy.enable_in_solve_monitor = True
    opts.strategy.stall_node_window = 15
    opts.strategy.stall_gap_tolerance = 1e-3
    opts.strategy.node_selection = sih_solver.NodeSelection.BestBound
    opts.strategy.branching_rule = sih_solver.BranchingRule.MostFractional
    opts.log_to_console = False

    sol = sih_solver.solve(selected_prob, opts)

    print(f"Solve completed with status: {sol.status}")
    print(f"Nodes explored: {sol.nodes_explored}")
    print(f"Strategy mutations triggered: {sol.strategy_switches}")
    print("\n--- In-Solve Monitor Log ---")
    print(sol.in_solve_log)

    assert sol.strategy_switches >= 1, f"Expected at least 1 strategy mutation, got {sol.strategy_switches}"
    assert "Stall detected" in sol.in_solve_log, "In-solve log missing stall detection event"
    assert "Pre-switch node processing rate" in sol.in_solve_log, "Missing pre-switch node rate"
    assert "Post-switch node processing rate" in sol.in_solve_log, "Missing post-switch node rate"

    print(">>> In-solve monitor GATE PASSED.")


if __name__ == "__main__":
    test_insolv_monitor_gate()
