"""
Independent Floating-Point Solution Checker.
Evaluates primal residuals, dual residuals, complementary slackness,
integrality violations, and objective value consistency without relying on solver internals.
"""

from typing import Dict, Any, List, Optional
import math

class FloatChecker:
    @staticmethod
    def check_solution(problem: Any,
                       x: List[float],
                       row_duals: Optional[List[float]] = None,
                       reduced_costs: Optional[List[float]] = None,
                       reported_obj: Optional[float] = None,
                       tol_primal: float = 1e-6,
                       tol_dual: float = 1e-6,
                       tol_int: float = 1e-5) -> Dict[str, Any]:
        m = problem.num_rows()
        n = problem.num_cols()
        
        if len(x) != n:
            return {
                "valid": False,
                "error": f"Dimension mismatch: x has length {len(x)}, expected {n}"
            }

        # 1. Primal row activities Ax
        Ax = problem.A.mat_vec(x)
        
        row_lower = problem.row_lower
        row_upper = problem.row_upper
        col_lower = problem.col_lower
        col_upper = problem.col_upper
        var_types = problem.var_types

        # Check row bounds
        max_row_viol = 0.0
        for i in range(m):
            lb = row_lower[i]
            ub = row_upper[i]
            val = Ax[i]
            if lb > -1e30 and val < lb:
                max_row_viol = max(max_row_viol, lb - val)
            if ub < 1e30 and val > ub:
                max_row_viol = max(max_row_viol, val - ub)

        # Check variable bounds
        max_col_viol = 0.0
        for j in range(n):
            lb = col_lower[j]
            ub = col_upper[j]
            val = x[j]
            if lb > -1e30 and val < lb:
                max_col_viol = max(max_col_viol, lb - val)
            if ub < 1e30 and val > ub:
                max_col_viol = max(max_col_viol, val - ub)

        max_primal_residual = max(max_row_viol, max_col_viol)
        is_primal_feasible = max_primal_residual <= tol_primal

        # 2. Integrality check
        max_int_viol = 0.0
        for j in range(n):
            # 1 = Binary, 2 = Integer
            if int(var_types[j]) in (1, 2):
                val = x[j]
                dist = abs(val - round(val))
                max_int_viol = max(max_int_viol, dist)
        is_integrality_satisfied = max_int_viol <= tol_int

        # 3. Objective evaluation
        c = problem.c
        obj_eval = problem.obj_offset + sum(c[j] * x[j] for j in range(n))
        if problem.has_quadratic() and problem.num_quad_nonzeros() > 0:
            Qx = problem.Q().mat_vec(x)
            obj_eval += 0.5 * sum(x[j] * Qx[j] for j in range(n))

        obj_discrepancy = 0.0
        obj_match = True
        if reported_obj is not None:
            obj_discrepancy = abs(obj_eval - reported_obj)
            obj_match = obj_discrepancy <= 1e-4 * (1.0 + abs(obj_eval))

        # 4. Dual feasibility & complementary slackness (if provided)
        is_dual_feasible = True
        max_dual_residual = 0.0
        max_comp_slack = 0.0
        
        if row_duals is not None and reduced_costs is not None and len(row_duals) == m and len(reduced_costs) == n:
            AT_y = problem.A.mat_trans_vec(row_duals)
            grad = list(c)
            if problem.has_quadratic() and problem.num_quad_nonzeros() > 0:
                Qx = problem.Q().mat_vec(x)
                for j in range(n):
                    grad[j] += Qx[j]

            for j in range(n):
                stat = grad[j] - AT_y[j] - reduced_costs[j]
                max_dual_residual = max(max_dual_residual, abs(stat))
            is_dual_feasible = max_dual_residual <= tol_dual

            for i in range(m):
                lb = row_lower[i]
                ub = row_upper[i]
                dist = min(abs(Ax[i] - lb), abs(ub - Ax[i]))
                max_comp_slack = max(max_comp_slack, dist * abs(row_duals[i]))

        all_passed = is_primal_feasible and is_integrality_satisfied and obj_match and is_dual_feasible

        return {
            "valid": True,
            "all_passed": all_passed,
            "is_primal_feasible": is_primal_feasible,
            "max_primal_residual": max_primal_residual,
            "max_row_violation": max_row_viol,
            "max_col_bound_violation": max_col_viol,
            "is_integrality_satisfied": is_integrality_satisfied,
            "max_integrality_violation": max_int_viol,
            "evaluated_objective": obj_eval,
            "objective_discrepancy": obj_discrepancy,
            "is_dual_feasible": is_dual_feasible,
            "max_dual_residual": max_dual_residual,
            "max_complementary_slack": max_comp_slack,
        }
