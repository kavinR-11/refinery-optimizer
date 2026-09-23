"""
Stress Testing Suite for Indigenous LP Solver.
Validates:
1. Highly degenerate LPs (anti-cycling Bland fallback & cost perturbation).
2. Badly scaled LPs (power-of-two geometric mean scaling & equilibration).
3. 1000 seeded random LPs cross-validated against HiGHS oracle and independent checker.
"""

import sys
import random
import math
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from oracle import HighsOracle

def test_degenerate_lp():
    """Test solver on highly degenerate problem (multiple basic variables zero at bounds)."""
    # Classic Beale cycling problem or highly degenerate LP:
    # min -0.75 x1 + 20 x2 - 0.5 x3 + 6 x4
    # s.t. 0.25 x1 - 8 x2 - x3 + 9 x4 <= 0
    #      0.5  x1 - 12 x2 - 0.5 x3 + 3 x4 <= 0
    #      x3 <= 1
    #      x1, x2, x3, x4 >= 0
    p = sih_solver.Problem("beale_degenerate")
    p.resize(3, 4)
    p.c = [-0.75, 20.0, -0.5, 6.0]
    p.row_lower = [-1e30, -1e30, -1e30]
    p.row_upper = [0.0, 0.0, 1.0]
    p.col_lower = [0.0, 0.0, 0.0, 0.0]
    p.col_upper = [1e30, 1e30, 1e30, 1e30]

    triplets = [
        sih_solver.Triplet(0, 0, 0.25), sih_solver.Triplet(0, 1, -8.0), sih_solver.Triplet(0, 2, -1.0), sih_solver.Triplet(0, 3, 9.0),
        sih_solver.Triplet(1, 0, 0.5),  sih_solver.Triplet(1, 1, -12.0), sih_solver.Triplet(1, 2, -0.5), sih_solver.Triplet(1, 3, 3.0),
        sih_solver.Triplet(2, 2, 1.0)
    ]
    p.A = sih_solver.SparseMatrix.from_triplets(3, 4, triplets)

    opts = sih_solver.Options()
    opts.strategy.enable_perturbation = True
    sol = sih_solver.solve(p, opts)

    assert sol.status == sih_solver.SolutionStatus.Optimal
    # Exact optimum: x = [1.0, 0.0, 1.0, 0.0], obj = -1.25
    assert abs(sol.primal_objective - (-1.25)) < 1e-5
    chk = sih_solver.check_solution(p, sol)
    assert chk.all_checks_passed

def test_badly_scaled_lp():
    """Test solver with coefficients spanning 1e-6 to 1e6 to verify scaling."""
    p = sih_solver.Problem("badly_scaled")
    p.resize(3, 3)
    p.c = [1e-4, 1e4, 1.0]
    p.row_lower = [1.0, 1e3, -1e30]
    p.row_upper = [1e30, 1e30, 1e6]
    p.col_lower = [0.0, 0.0, 0.0]
    p.col_upper = [1e6, 1e6, 1e6]

    triplets = [
        sih_solver.Triplet(0, 0, 1e-5), sih_solver.Triplet(0, 1, 1e3),
        sih_solver.Triplet(1, 1, 1e-2), sih_solver.Triplet(1, 2, 1e5),
        sih_solver.Triplet(2, 0, 1e4),  sih_solver.Triplet(2, 2, 1e-3)
    ]
    p.A = sih_solver.SparseMatrix.from_triplets(3, 3, triplets)

    opts = sih_solver.Options()
    opts.strategy.enable_scaling = True
    opts.strategy.power_of_two_scaling = True

    sol = sih_solver.solve(p, opts)
    assert sol.status == sih_solver.SolutionStatus.Optimal
    chk = sih_solver.check_solution(p, sol)
    assert chk.all_checks_passed

def test_random_1000_seeds():
    """Run 1000 seeded random LPs and compare status & objective against HiGHS oracle."""
    total_seeds = 1000
    matched_count = 0

    tmp_mps = root / "data" / "toy" / "_tmp_stress.mps"

    for seed in range(total_seeds):
        rng = random.Random(seed)

        m = rng.randint(3, 6)
        n = rng.randint(4, 8)

        p = sih_solver.Problem(f"rand_{seed}")
        p.resize(m, n)

        # Objective
        c = [round(rng.uniform(-10.0, 20.0), 2) for _ in range(n)]
        p.c = c

        # Matrix: ~50% density
        triplets = []
        for i in range(m):
            for j in range(n):
                if rng.random() < 0.5:
                    val = round(rng.uniform(-5.0, 10.0), 2)
                    if abs(val) > 0.1:
                        triplets.append(sih_solver.Triplet(i, j, val))
        # Ensure at least 1 entry per row and col
        for i in range(m):
            triplets.append(sih_solver.Triplet(i, rng.randint(0, n - 1), round(rng.uniform(1.0, 5.0), 2)))
        for j in range(n):
            triplets.append(sih_solver.Triplet(rng.randint(0, m - 1), j, round(rng.uniform(1.0, 5.0), 2)))

        p.A = sih_solver.SparseMatrix.from_triplets(m, n, triplets)

        # Row bounds
        rl = []
        ru = []
        for _ in range(m):
            r_type = rng.choice(["<=", ">=", "=="])
            rhs = round(rng.uniform(5.0, 50.0), 2)
            if r_type == "<=":
                rl.append(-1e30)
                ru.append(rhs)
            elif r_type == ">=":
                rl.append(rhs)
                ru.append(1e30)
            else:
                rl.append(rhs)
                ru.append(rhs)
        p.row_lower = rl
        p.row_upper = ru

        # Col bounds
        cl = [0.0 if rng.random() < 0.8 else -round(rng.uniform(1.0, 10.0), 2) for _ in range(n)]
        cu = [round(rng.uniform(10.0, 100.0), 2) if rng.random() < 0.7 else 1e30 for _ in range(n)]
        p.col_lower = cl
        p.col_upper = cu

        # Write out to temporary MPS for HiGHS oracle
        sih_solver.write_mps(p, str(tmp_mps), free_format=True)

        oracle = HighsOracle.solve_mps(str(tmp_mps))
        oracle_status = oracle["status"]

        opts = sih_solver.Options()
        opts.strategy.presolve = sih_solver.PresolveMode.On
        opts.strategy.enable_scaling = True

        sol = sih_solver.solve(p, opts)
        sol_status = sol.status.name

        assert sol_status == oracle_status, (
            f"Seed {seed} status mismatch: indigenous={sol_status}, oracle={oracle_status}"
        )

        if oracle_status == "Optimal":
            highs_obj = oracle["objective_value"]
            indig_obj = sol.primal_objective
            rel_diff = abs(indig_obj - highs_obj) / (1.0 + abs(highs_obj))
            assert rel_diff < 1e-4, f"Seed {seed} obj mismatch: indig={indig_obj}, highs={highs_obj}"

            chk = sih_solver.check_solution(p, sol)
            assert chk.all_checks_passed, f"Seed {seed} checker failed: {chk.summary}"

        matched_count += 1

    if tmp_mps.exists():
        tmp_mps.unlink()

    print(f"\n[Stress Test: 1000 Random LP Seeds]")
    print(f"  Total seeds tested: {total_seeds}")
    print(f"  Exact matches with oracle: {matched_count}/{total_seeds} (100.0%)")
