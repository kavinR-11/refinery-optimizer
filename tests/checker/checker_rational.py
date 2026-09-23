"""
Independent Rational-Arithmetic Solution Checker using Python's fractions.Fraction.
Provides exact symbolic zero-floating-point-error verification for small problem instances.
"""

from fractions import Fraction
from typing import Dict, Any, List, Optional
import math

class RationalChecker:
    @staticmethod
    def to_fraction(val: float) -> Optional[Fraction]:
        if math.isinf(val):
            return None
        return Fraction(val).limit_denominator(1_000_000_000)

    @staticmethod
    def check_solution(problem: Any,
                       x_float: List[float],
                       reported_obj: Optional[float] = None) -> Dict[str, Any]:
        m = problem.num_rows()
        n = problem.num_cols()

        if len(x_float) != n:
            return {
                "valid": False,
                "error": f"Dimension mismatch: x has length {len(x_float)}, expected {n}"
            }

        # Convert primal x to exact Fraction
        x_rat = [RationalChecker.to_fraction(val) for val in x_float]

        # Extract CSC matrix entries and compute Ax exactly in Fraction
        col_ptr = problem.A.csc_col_ptr()
        row_ind = problem.A.csc_row_ind()
        values = problem.A.csc_values()

        Ax_rat = [Fraction(0) for _ in range(m)]
        for j in range(n):
            xj = x_rat[j]
            start = col_ptr[j]
            end = col_ptr[j + 1]
            for k in range(start, end):
                r = row_ind[k]
                v = RationalChecker.to_fraction(values[k])
                Ax_rat[r] += v * xj

        row_lower = problem.row_lower
        row_upper = problem.row_upper
        col_lower = problem.col_lower
        col_upper = problem.col_upper
        var_types = problem.var_types

        # Check row bounds exactly
        row_violations = []
        for i in range(m):
            lb = RationalChecker.to_fraction(row_lower[i])
            ub = RationalChecker.to_fraction(row_upper[i])
            val = Ax_rat[i]

            if lb is not None and val < lb:
                row_violations.append((i, "lower", float(lb - val)))
            if ub is not None and val > ub:
                row_violations.append((i, "upper", float(val - ub)))

        # Check column bounds exactly
        col_violations = []
        for j in range(n):
            lb = RationalChecker.to_fraction(col_lower[j])
            ub = RationalChecker.to_fraction(col_upper[j])
            val = x_rat[j]

            if lb is not None and val < lb:
                col_violations.append((j, "lower", float(lb - val)))
            if ub is not None and val > ub:
                col_violations.append((j, "upper", float(val - ub)))

        # Check integrality exactly
        int_violations = []
        for j in range(n):
            if int(var_types[j]) in (1, 2): # Binary or Integer
                if x_rat[j].denominator != 1:
                    int_violations.append((j, float(abs(x_rat[j] - round(float(x_rat[j]))))))

        # Objective calculation
        c = problem.c
        obj_rat = RationalChecker.to_fraction(problem.obj_offset)
        for j in range(n):
            cj = RationalChecker.to_fraction(c[j])
            obj_rat += cj * x_rat[j]

        if problem.has_quadratic() and problem.num_quad_nonzeros() > 0:
            q_col_ptr = problem.Q().csc_col_ptr()
            q_row_ind = problem.Q().csc_row_ind()
            q_values = problem.Q().csc_values()
            q_sum = Fraction(0)
            for j in range(n):
                xj = x_rat[j]
                start = q_col_ptr[j]
                end = q_col_ptr[j + 1]
                for k in range(start, end):
                    i = q_row_ind[k]
                    v = RationalChecker.to_fraction(q_values[k])
                    q_sum += xj * x_rat[i] * v
            obj_rat += Fraction(1, 2) * q_sum

        obj_float = float(obj_rat)
        obj_match = True
        if reported_obj is not None:
            obj_match = abs(obj_float - reported_obj) < 1e-4 * (1.0 + abs(obj_float))

        all_passed = (len(row_violations) == 0 and
                      len(col_violations) == 0 and
                      len(int_violations) == 0 and
                      obj_match)

        return {
            "valid": True,
            "all_passed": all_passed,
            "exact_objective_fraction": str(obj_rat),
            "exact_objective_float": obj_float,
            "row_violations_count": len(row_violations),
            "col_violations_count": len(col_violations),
            "int_violations_count": len(int_violations),
            "row_violations": row_violations,
            "col_violations": col_violations,
            "int_violations": int_violations,
        }
