import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))
sys.path.insert(0, str(root / "tests"))

import sih_solver
from oracle import HighsOracle

def main():
    netlib_dir = root / "data" / "netlib"
    files = sorted(list(netlib_dir.glob("*.mps")))
    if not files:
        print("No Netlib MPS files found in data/netlib")
        return

    print("| Problem | Rows x Cols | Indig Iters | Indig Time (ms) | HiGHS Time (ms) | Indig Obj | HiGHS Obj | Rel Diff | Status |")
    print("|---|---|---|---|---|---|---|---|---|")

    for f in files:
        name = f.stem
        p = sih_solver.read_mps(str(f))
        h_res = HighsOracle.solve_mps(str(f))

        opts = sih_solver.Options()
        opts.strategy.presolve = sih_solver.PresolveMode.On
        opts.strategy.enable_scaling = True

        # Warm-up / solve
        t0 = time.perf_counter()
        sol = sih_solver.solve(p, opts)
        t1 = time.perf_counter()
        indig_ms = (t1 - t0) * 1000.0

        t0 = time.perf_counter()
        _ = HighsOracle.solve_mps(str(f))
        t1 = time.perf_counter()
        highs_ms = (t1 - t0) * 1000.0

        h_obj = h_res["objective_value"]
        i_obj = sol.primal_objective
        diff = abs(i_obj - h_obj) / (1.0 + abs(h_obj))

        print(f"| {name:<10} | {p.num_rows()}x{p.num_cols()} | {sol.simplex_iterations:<6} | {indig_ms:<8.2f} | {highs_ms:<8.2f} | {i_obj:<13.6e} | {h_obj:<13.6e} | {diff:<9.2e} | {sol.status.name} |")

if __name__ == "__main__":
    main()
