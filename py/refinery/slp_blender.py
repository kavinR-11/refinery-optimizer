"""
Successive Linear Programming (SLP) for Nonlinear Refinery Blending.
Models bilinear crude-to-stream quality pooling (sulfur content) and solves iteratively
via successive linearization until property estimates converge within tolerance.
Includes independent nonlinear physical verification.
"""

from typing import Dict, List, Tuple, Optional, Any
import copy
import sih_solver
from refinery.model_builder import RefineryModelBuilder


class SLPBlendingResult:
    """
    Holds the complete convergence trajectory and physical verification metrics of an SLP run.
    """

    def __init__(self):
        self.converged: bool = False
        self.num_iterations: int = 0
        self.history: List[Dict[str, Any]] = []
        self.final_solution: Optional[sih_solver.Solution] = None
        self.final_problem: Optional[sih_solver.Problem] = None
        self.final_metadata: Optional[Dict[str, Any]] = None
        
        # Independent physical verification
        self.actual_feed_sulfur: float = 0.0
        self.actual_gasoil_sulfur: float = 0.0
        self.actual_diesel_sulfur: float = 0.0
        self.target_diesel_sulfur_spec: float = 0.05
        self.is_physically_feasible: bool = False
        self.property_error: float = 0.0


class SLPRefineryBlender:
    """
    Successive Linear Programming solver for nonlinear crude sulfur pooling and blending.
    
    Bilinear Relationship:
      Crude feed sulfur: S_feed = sum(s_c * x_c) / sum(x_c)
      Straight-run distillate sulfur: S_dist = theta * S_feed
      Finished diesel sulfur spec: sum((S_stream - S_max) * w_stream) <= 0
      where S_dist is endogenous to crude slate selection.
    """

    def __init__(
        self,
        base_builder: Optional[RefineryModelBuilder] = None,
        theta_sulfur: float = 0.70,
        max_iterations: int = 15,
        tolerance: float = 1e-4,
        damping: float = 0.80,
    ):
        self.base_builder = base_builder if base_builder is not None else RefineryModelBuilder(is_milp=False)
        self.theta_sulfur = theta_sulfur
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

        self.options = sih_solver.Options()
        self.options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
        self.options.log_to_console = False

    def solve(self, initial_sulfur_guess: float = 1.00) -> SLPBlendingResult:
        """
        Execute Successive Linear Programming (SLP) iterations.

        Parameters:
            initial_sulfur_guess: Initial estimate for gasoil sulfur content (wt%).

        Returns:
            SLPBlendingResult with iteration logs and independent physical check.
        """
        result = SLPBlendingResult()
        s_hat = initial_sulfur_guess
        result.target_diesel_sulfur_spec = self.base_builder.products["Diesel"]["max_sulfur"]

        for it in range(1, self.max_iterations + 1):
            # 1. Build refinery LP with linearized gasoil sulfur = s_hat
            builder = copy.deepcopy(self.base_builder)
            # Update sulfur property for Distillate_Raw in stream properties
            builder.stream_props["Distillate_Raw"]["sulfur"] = s_hat

            prob, meta = builder.build_problem()

            # 2. Update the diesel sulfur constraint with current s_hat
            # The constraint row is 'spec_sulfur_Diesel'
            row_names = meta["row_names"]
            if "spec_sulfur_Diesel" in row_names:
                sulf_row_idx = row_names.index("spec_sulfur_Diesel")
                var_map = meta["var_map"]
                raw_var_idx = var_map["flow_Distillate_Raw_to_Diesel"]
                
                # Coeff in (S_stream - S_spec) * w <= 0 is (s_hat - S_max)
                spec_max = builder.products["Diesel"]["max_sulfur"]
                new_coeff = s_hat - spec_max
                
                # Rebuild matrix with updated coefficient
                triplets = []
                csc_col_ptr = prob.A.csc_col_ptr()
                csc_row_ind = prob.A.csc_row_ind()
                csc_vals = prob.A.csc_values()
                
                for c in range(prob.num_cols()):
                    for idx in range(csc_col_ptr[c], csc_col_ptr[c + 1]):
                        r = csc_row_ind[idx]
                        v = csc_vals[idx]
                        if r == sulf_row_idx and c == raw_var_idx:
                            v = new_coeff
                        triplets.append(sih_solver.Triplet(r, c, v))
                prob.A = sih_solver.SparseMatrix.from_triplets(prob.num_rows(), prob.num_cols(), triplets)

            # 3. Solve the linearized LP
            sol = sih_solver.solve(prob, self.options)
            if not sol.is_feasible():
                break

            # 4. Extract crude purchase decisions and compute true physical pooling
            var_map = meta["var_map"]
            total_crude = 0.0
            total_crude_sulfur = 0.0

            for c_name, c_info in builder.crudes.items():
                x_val = sol.x[var_map[f"crude_{c_name}"]]
                total_crude += x_val
                total_crude_sulfur += c_info["sulfur"] * x_val

            actual_feed_sulfur = (total_crude_sulfur / total_crude) if total_crude > 1e-6 else 0.0
            actual_gasoil_sulfur = self.theta_sulfur * actual_feed_sulfur

            # Check error between estimate and true nonlinear property
            error = abs(actual_gasoil_sulfur - s_hat)
            step_delta = actual_gasoil_sulfur - s_hat

            record = {
                "iteration": it,
                "s_hat_estimate": s_hat,
                "actual_feed_sulfur": actual_feed_sulfur,
                "actual_gasoil_sulfur": actual_gasoil_sulfur,
                "error": error,
                "objective": sol.primal_objective,
                "total_crude_bpd": total_crude,
            }
            result.history.append(record)

            result.final_solution = sol
            result.final_problem = prob
            result.final_metadata = meta

            if error < self.tolerance:
                result.converged = True
                result.num_iterations = it
                break

            # Damped update for next iteration
            s_hat = s_hat + self.damping * step_delta

        if not result.converged:
            result.num_iterations = len(result.history)

        # 5. Independent Physical Verification
        self._verify_nonlinear_physics(result)
        return result

    def _verify_nonlinear_physics(self, result: SLPBlendingResult):
        """
        Perform an independent post-solve nonlinear physical audit.
        Evaluates true bilinearly-blended diesel sulfur from optimal streams.
        """
        if result.final_solution is None or result.final_metadata is None:
            return

        sol = result.final_solution
        meta = result.final_metadata
        builder = meta["builder"]
        var_map = meta["var_map"]

        # Recalculate true crude mix sulfur
        tot_crude = sum(sol.x[var_map[f"crude_{c}"]] for c in builder.crudes)
        tot_sulfur = sum(builder.crudes[c]["sulfur"] * sol.x[var_map[f"crude_{c}"]] for c in builder.crudes)
        true_feed_sulfur = tot_sulfur / tot_crude if tot_crude > 0 else 0.0
        true_gasoil_sulfur = self.theta_sulfur * true_feed_sulfur

        # Calculate true blended finished diesel sulfur
        hdt_flow = sol.x[var_map.get("flow_Distillate_HDT_to_Diesel", 0)]
        raw_flow = sol.x[var_map.get("flow_Distillate_Raw_to_Diesel", 0)]
        cycle_hdt_flow = sol.x[var_map.get("flow_CycleOil_HDT_to_Diesel", 0)]
        cycle_raw_flow = sol.x[var_map.get("flow_CycleOil_Raw_to_Diesel", 0)]
        
        total_diesel_vol = hdt_flow + raw_flow + cycle_hdt_flow + cycle_raw_flow

        # Sulfur contributions
        s_hdt = builder.stream_props["Distillate_HDT"]["sulfur"]
        s_raw = true_gasoil_sulfur  # true nonlinear pooled quality!
        s_cycle_hdt = builder.stream_props["CycleOil_HDT"]["sulfur"]
        s_cycle_raw = builder.stream_props["CycleOil_Raw"]["sulfur"]

        total_diesel_sulfur_wt = (
            s_hdt * hdt_flow +
            s_raw * raw_flow +
            s_cycle_hdt * cycle_hdt_flow +
            s_cycle_raw * cycle_raw_flow
        )

        true_diesel_sulfur = (total_diesel_sulfur_wt / total_diesel_vol) if total_diesel_vol > 0 else 0.0

        result.actual_feed_sulfur = true_feed_sulfur
        result.actual_gasoil_sulfur = true_gasoil_sulfur
        result.actual_diesel_sulfur = true_diesel_sulfur
        result.property_error = abs(true_gasoil_sulfur - result.history[-1]["s_hat_estimate"])
        
        # Independent check against legal limit
        spec_limit = result.target_diesel_sulfur_spec
        # Accept if within 1e-4 tolerance
        result.is_physically_feasible = (true_diesel_sulfur <= spec_limit + 1e-4)

    @staticmethod
    def format_convergence_table(result: SLPBlendingResult) -> str:
        """
        Format a Markdown convergence table showing property iterations.
        """
        lines = [
            "| Iter | Sulfur Guess (s_hat) | Actual Feed S (%) | Actual Gasoil S (%) | Abs Error | Daily Margin ($) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for h in result.history:
            lines.append(
                f"| {h['iteration']:2d}   | {h['s_hat_estimate']:.6f}%        "
                f"| {h['actual_feed_sulfur']:.4f}%           | {h['actual_gasoil_sulfur']:.6f}%         "
                f"| {h['error']:.2e}  | ${h['objective']:,.2f} |"
            )

        lines.extend([
            "",
            "### Independent Nonlinear Physical Verification",
            f"- **Convergence**: `{'CONVERGED' if result.converged else 'MAX_ITERATIONS'}` in {result.num_iterations} iterations",
            f"- **True Crude Feed Sulfur**: **{result.actual_feed_sulfur:.4f} wt%**",
            f"- **True Gasoil Sulfur (Nonlinear Pooled)**: **{result.actual_gasoil_sulfur:.6f} wt%**",
            f"- **True Finished Diesel Sulfur**: **{result.actual_diesel_sulfur:.6f} wt%** (Legal Spec: $\\le$ {result.target_diesel_sulfur_spec:.4f} wt%)",
            f"- **Independent Physical Feasibility**: `{'VERIFIED PASS' if result.is_physically_feasible else 'FAIL'}`",
        ])

        return "\n".join(lines)
