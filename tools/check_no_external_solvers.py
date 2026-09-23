#!/usr/bin/env python3
"""
Compliance Verification Tool: check_no_external_solvers.py
Enforces Hard Rules 1, 2, and 3:
- /core (C++17) must use ONLY the C++ standard library, OpenMP, and (for /core/gpu) CUDA runtime with custom kernels.
- No Eigen, SuiteSparse, BLAS/LAPACK, cuSPARSE, cuBLAS, or 3rd-party solver code anywhere in /core.
- /py may use NumPy, pandas, matplotlib, and pybind11. It must NOT import external solvers.
- highspy and scipy.optimize are allowed ONLY under /tests/oracle and /bench.
"""

import os
import re
import sys
from pathlib import Path

# Directories completely exempted from solver checks
EXEMPT_DIRS = {
    "tests/oracle",
    "bench",
    "docs",
    ".git",
    ".venv",
    "build",
    "__pycache__",
    ".pytest_cache",
    "data",
}

# Python forbidden import patterns
PY_FORBIDDEN_PATTERNS = [
    (r"\bimport\s+scipy\.optimize\b", "scipy.optimize import"),
    (r"\bfrom\s+scipy\s+import\s+optimize\b", "scipy.optimize import"),
    (r"\bfrom\s+scipy\.optimize\b", "scipy.optimize import"),
    (r"\bimport\s+highspy\b", "highspy import"),
    (r"\bfrom\s+highspy\b", "highspy import"),
    (r"\bimport\s+cvxpy\b", "cvxpy import"),
    (r"\bfrom\s+cvxpy\b", "cvxpy import"),
    (r"\bimport\s+ortools\b", "ortools import"),
    (r"\bfrom\s+ortools\b", "ortools import"),
    (r"\bimport\s+pulp\b", "pulp import"),
    (r"\bfrom\s+pulp\b", "pulp import"),
    (r"\bimport\s+gurobipy\b", "gurobipy import"),
    (r"\bfrom\s+gurobipy\b", "gurobipy import"),
    (r"\bimport\s+cplex\b", "cplex import"),
    (r"\bfrom\s+cplex\b", "cplex import"),
    (r"\bimport\s+pyscipopt\b", "scip import"),
    (r"\bfrom\s+pyscipopt\b", "scip import"),
    (r"\bimport\s+cylp\b", "cbc/cylp import"),
]

# C++ forbidden includes/headers
CPP_FORBIDDEN_PATTERNS = [
    (r"#\s*include\s*<[Ee]igen", "Eigen linear algebra library"),
    (r"#\s*include\s*<suitesparse", "SuiteSparse factorization library"),
    (r"#\s*include\s*[<\"]cholmod\.h[>\"]", "CHOLMOD library"),
    (r"#\s*include\s*[<\"]umfpack\.h[>\"]", "UMFPACK library"),
    (r"#\s*include\s*[<\"]klu\.h[>\"]", "KLU library"),
    (r"#\s*include\s*[<\"]cblas\.h[>\"]", "BLAS/CBLAS library"),
    (r"#\s*include\s*[<\"]lapacke?\.h[>\"]", "LAPACK library"),
    (r"#\s*include\s*<openblas", "OpenBLAS library"),
    (r"#\s*include\s*<mkl", "Intel MKL library"),
    (r"#\s*include\s*[<\"]Highs\.h[>\"]", "HiGHS C++ library"),
    (r"#\s*include\s*[<\"]gurobi_c\+\+\.h[>\"]", "Gurobi library"),
    (r"#\s*include\s*[<\"]cusparse\.h[>\"]", "cuSPARSE library (forbidden outside /bench)"),
    (r"#\s*include\s*[<\"]cublas.*\.h[>\"]", "cuBLAS library (forbidden outside /bench)"),
]

def is_exempt(rel_path: str) -> bool:
    posix_path = rel_path.replace("\\", "/")
    # Exact exemptions
    if posix_path == "tools/check_no_external_solvers.py":
        return True
    for exempt in EXEMPT_DIRS:
        if posix_path == exempt or posix_path.startswith(exempt + "/"):
            return True
    return False

def scan_file(file_path: Path, rel_path: str) -> list[tuple[int, str, str]]:
    violations = []
    suffix = file_path.suffix.lower()
    
    # Check if file is python or C/C++
    is_py = suffix == ".py"
    is_cpp = suffix in {".cpp", ".hpp", ".h", ".c", ".cu", ".cuh"}
    
    if not (is_py or is_cpp):
        return violations

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        print(f"Warning: could not read {file_path}: {e}")
        return violations

    patterns = PY_FORBIDDEN_PATTERNS if is_py else CPP_FORBIDDEN_PATTERNS

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Ignore comments
        if is_py and stripped.startswith("#"):
            continue
        if is_cpp and (stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")):
            continue

        for regex, desc in patterns:
            if re.search(regex, line):
                violations.append((line_num, line.strip(), desc))

    return violations

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    all_violations = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = str(file_path.relative_to(root)).replace("\\", "/")
        if is_exempt(rel_path):
            continue

        file_violations = scan_file(file_path, rel_path)
        for line_num, line_content, desc in file_violations:
            all_violations.append((rel_path, line_num, line_content, desc))

    if all_violations:
        print("=" * 70)
        print("COMPLIANCE VIOLATION DETECTED: FORBIDDEN EXTERNAL SOLVER / LIBRARY")
        print("=" * 70)
        for path, line_num, line_content, desc in all_violations:
            print(f"  [FAIL] {path}:{line_num}")
            print(f"         Reason: {desc}")
            print(f"         Line:   {line_content}")
            print()
        print("Rules Violated:")
        print("- Rule 1: /core may only use ISO C++ stdlib, OpenMP, and custom CUDA kernels.")
        print("- Rule 2: /py may NOT import external solvers (scipy.optimize, highspy, etc.).")
        print("- Rule 3: External solvers and cuBLAS/cuSPARSE are permitted ONLY in /tests/oracle and /bench.")
        print("=" * 70)
        return 1

    print("[PASS] Compliance check successful: No forbidden external solver or linear algebra libraries detected.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
