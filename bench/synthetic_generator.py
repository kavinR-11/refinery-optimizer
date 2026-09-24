#!/usr/bin/env python3
"""
Synthetic Sparse LP Generator for GPU PDHG Crossover Benchmarks.
Generates large, sparse linear programs with guaranteed primal and dual feasibility.
Format: standard MPS.
"""

import os
import random
from pathlib import Path


def generate_synthetic_lp(
    m: int,
    n: int,
    entries_per_col: int = 6,
    seed: int = 42,
    output_path: str = None
) -> str:
    """
    Generate a sparse LP:
      min c^T x
      s.t. A x <= b
           l <= x <= u
    where:
      - A is m x n sparse with ~entries_per_col nonzeros per col
      - x0 in [l, u] is a feasible interior point
      - b = A x0 + slack, slack > 0 (primal feasible)
      - y0 >= 0, r0 >= 0, c = A^T y0 + r0 (dual feasible)
    """
    rng = random.Random(seed)

    # 1. Bounds
    l_bounds = [0.0] * n
    u_bounds = [rng.uniform(5.0, 20.0) for _ in range(n)]

    # Strictly feasible x0
    x0 = [rng.uniform(0.1 * u_bounds[j], 0.6 * u_bounds[j]) for j in range(n)]

    # 2. Sparse Matrix A (in CSC: columns list of (row, val))
    cols = []
    # Row activities Ax0
    Ax0 = [0.0] * m

    for j in range(n):
        k = min(m, max(2, entries_per_col))
        rows = sorted(rng.sample(range(m), k))
        col_entries = []
        for r in rows:
            val = round(rng.uniform(0.1, 2.0), 3)
            col_entries.append((r, val))
            Ax0[r] += val * x0[j]
        cols.append(col_entries)

    # 3. b = Ax0 + positive slack
    b = [round(Ax0[i] + rng.uniform(2.0, 10.0), 3) for i in range(m)]

    # 4. Dual variables y0 >= 0 and cost vector c = A^T y0 + r0
    y0 = [rng.uniform(0.1, 1.0) for _ in range(m)]
    c = [0.0] * n
    for j in range(n):
        dot = sum(val * y0[r] for r, val in cols[j])
        r0 = rng.uniform(0.01, 1.0)
        c[j] = round(dot + r0, 3)

    # 5. Format as MPS text
    lines = []
    prob_name = f"synth_{m}x{n}"
    lines.append(f"NAME          {prob_name}")
    lines.append("OBJSENSE")
    lines.append("  MIN")
    lines.append("ROWS")
    lines.append(" N  OBJ")
    for i in range(m):
        lines.append(f" L  R{i:07d}")

    lines.append("COLUMNS")
    for j in range(n):
        col_name = f"C{j:07d}"
        # Obj coeff
        lines.append(f"    {col_name:<10}  OBJ       {c[j]:12.4f}")
        for r, val in cols[j]:
            row_name = f"R{r:07d}"
            lines.append(f"    {col_name:<10}  {row_name:<8}  {val:12.4f}")

    lines.append("RHS")
    for i in range(m):
        row_name = f"R{i:07d}"
        lines.append(f"    RHS1        {row_name:<8}  {b[i]:12.4f}")

    lines.append("BOUNDS")
    for j in range(n):
        col_name = f"C{j:07d}"
        lines.append(f" UP BND1        {col_name:<10}  {u_bounds[j]:12.4f}")
        lines.append(f" LO BND1        {col_name:<10}  {l_bounds[j]:12.4f}")

    lines.append("ENDATA")
    mps_text = "\n".join(lines) + "\n"

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mps_text)

    return mps_text


if __name__ == "__main__":
    import sys
    out = "data/synthetic/synth_500x1000.mps"
    print(f"Generating sample synthetic LP: 500 rows, 1000 cols -> {out}")
    generate_synthetic_lp(500, 1000, entries_per_col=6, output_path=out)
    print("Done. File size:", os.path.getsize(out), "bytes")
