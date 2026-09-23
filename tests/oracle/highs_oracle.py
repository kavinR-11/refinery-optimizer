"""
HiGHS Oracle Harness for Reference Comparison.
Strictly confined to tests/oracle/ and bench/ per Hard Rules 2 and 3.
"""

from typing import Dict, Any, List, Optional
import os
import highspy

class HighsOracle:
    @staticmethod
    def solve_mps(mps_path: str, time_limit_sec: Optional[float] = None) -> Dict[str, Any]:
        if not os.path.exists(mps_path):
            raise FileNotFoundError(f"MPS file not found: {mps_path}")

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        if time_limit_sec is not None:
            h.setOptionValue("time_limit", float(time_limit_sec))

        # Read model
        read_status = h.readModel(mps_path)
        if read_status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS failed to read model from {mps_path}")

        # Solve model
        run_status = h.run()
        model_status = h.getModelStatus()
        info = h.getInfo()
        solution = h.getSolution()

        # Map HiGHS status enum to unified status string
        status_map = {
            highspy.HighsModelStatus.kOptimal: "Optimal",
            highspy.HighsModelStatus.kInfeasible: "Infeasible",
            highspy.HighsModelStatus.kUnbounded: "Unbounded",
            highspy.HighsModelStatus.kUnboundedOrInfeasible: "Unbounded",
            highspy.HighsModelStatus.kTimeLimit: "TimeLimit",
            highspy.HighsModelStatus.kIterationLimit: "IterationLimit",
        }
        status_str = status_map.get(model_status, "Unknown")

        obj_val = None
        col_value = None
        row_dual = None
        col_dual = None

        if status_str == "Optimal":
            obj_val = info.objective_function_value
            col_value = list(solution.col_value) if solution.col_value else []
            row_dual = list(solution.row_dual) if solution.row_dual else []
            col_dual = list(solution.col_dual) if solution.col_dual else []

        return {
            "status": status_str,
            "raw_status": str(model_status),
            "objective_value": obj_val,
            "primal_solution": col_value,
            "row_duals": row_dual,
            "reduced_costs": col_dual,
            "simplex_iterations": info.simplex_iteration_count,
            "ipm_iterations": info.ipm_iteration_count,
            "mip_nodes": info.mip_node_count,
            "wall_time_sec": info.primal_dual_integral,
        }
