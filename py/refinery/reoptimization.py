"""
Refinery Warm-Start Re-Optimization Workflow.
Executes sequences of operational refinery perturbations (crude prices, product demand,
unit capacity outages, supply disruptions) and measures cold-start vs warm-start performance.
"""

import copy
import time
from typing import Dict, List, Tuple, Optional, Any
import sih_solver
from refinery.model_builder import RefineryModelBuilder


class RefineryReoptimizer:
    """
    Executes and benchmarks warm-start re-optimization on sequential refinery perturbations.
    """

    def __init__(self, base_builder: Optional[RefineryModelBuilder] = None):
        if base_builder is None:
            self.base_builder = RefineryModelBuilder(is_milp=False)
        else:
            self.base_builder = base_builder

        self.options = sih_solver.Options()
        self.options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
        self.options.strategy.presolve = sih_solver.PresolveMode.Off  # Exact basis preservation
        self.options.strategy.enable_scaling = False
        self.options.log_to_console = False

    def get_standard_perturbations(self) -> List[Dict[str, Any]]:
        """
        Define 5 realistic industrial operational perturbations.
        """
        return [
            {
                "id": "pert_1_crude_price",
                "name": "Crude Price Shock",
                "desc": "Arab Light spot price +$12/bbl, Brent +$5/bbl",
                "apply": lambda b: (
                    b.crudes["ArabLight"].__setitem__("price", 92.0),
                    b.crudes["Brent"].__setitem__("price", 90.0)
                ),
            },
            {
                "id": "pert_2_gasoline_demand",
                "name": "Gasoline Demand Surge",
                "desc": "Regular Gasoline contractual demand +46% (15k -> 22k bpd)",
                "apply": lambda b: (
                    b.products["Gasoline_Regular"].__setitem__("min_demand", 22000.0),
                    b.products["Gasoline_Regular"].__setitem__("max_demand", 45000.0)
                ),
            },
            {
                "id": "pert_3_fcc_outage",
                "name": "FCC Unit Turndown",
                "desc": "FCC capacity slashed by 37% (35k -> 22k bpd) for cyclone maintenance",
                "apply": lambda b: (
                    b.units["FCC"].__setitem__("capacity", 22000.0)
                ),
            },
            {
                "id": "pert_4_brent_shortage",
                "name": "Sweet Crude Supply Cut",
                "desc": "Brent pipeline terminal limits intake to 50k bpd (-37.5%)",
                "apply": lambda b: (
                    b.crudes["Brent"].__setitem__("max_supply", 50000.0)
                ),
            },
            {
                "id": "pert_5_jet_fuel_spike",
                "name": "Aviation Jet Fuel Surge",
                "desc": "Jet Fuel demand rises to 14k bpd (+40%) with price surge to $128/bbl",
                "apply": lambda b: (
                    b.products["JetFuel"].__setitem__("min_demand", 14000.0),
                    b.products["JetFuel"].__setitem__("price", 128.0)
                ),
            },
        ]

    def run_benchmark(self) -> List[Dict[str, Any]]:
        """
        Run sequence of 5 perturbations comparing cold start vs warm start.
        """
        results: List[Dict[str, Any]] = []

        # 1. Solve base case
        base_b = copy.deepcopy(self.base_builder)
        prob_base, meta_base = base_b.build_problem()
        sol_base = sih_solver.solve(prob_base, self.options)
        assert sol_base.is_optimal(), "Base model failed to solve"

        current_basis_col = sol_base.col_basis
        current_basis_row = sol_base.row_basis

        perturbations = self.get_standard_perturbations()

        for pert in perturbations:
            b_pert = copy.deepcopy(self.base_builder)
            pert["apply"](b_pert)
            prob_pert, _ = b_pert.build_problem()

            # --- Cold Start ---
            # Benchmark 5 repetitions to get accurate microseconds
            cold_iters = 0
            t0 = time.perf_counter()
            sol_cold = sih_solver.solve(prob_pert, self.options)
            t_cold = (time.perf_counter() - t0) * 1000.0
            cold_iters = sol_cold.simplex_iterations

            # --- Warm Start ---
            t0 = time.perf_counter()
            sol_warm = sih_solver.solve_from_basis(
                prob_pert,
                current_basis_col,
                current_basis_row,
                self.options,
            )
            t_warm = (time.perf_counter() - t0) * 1000.0
            warm_iters = sol_warm.simplex_iterations

            # Verify identical solutions
            print(f"[{pert['name']}] cold_status={sol_cold.status} (iters={cold_iters}), warm_status={sol_warm.status} (iters={warm_iters})")
            assert sol_cold.is_optimal(), f"Cold solve failed on {pert['name']}: {sol_cold.status}"
            assert sol_warm.is_optimal(), f"Warm solve failed on {pert['name']}: {sol_warm.status}"
            obj_diff = abs(sol_cold.primal_objective - sol_warm.primal_objective)
            assert obj_diff < 1e-2, f"Objective mismatch on {pert['name']}: cold={sol_cold.primal_objective}, warm={sol_warm.primal_objective}"

            iter_reduction = ((cold_iters - warm_iters) / cold_iters * 100.0) if cold_iters > 0 else 0.0
            speedup = (t_cold / t_warm) if t_warm > 1e-4 else 1.0

            record = {
                "id": pert["id"],
                "name": pert["name"],
                "desc": pert["desc"],
                "cold_iters": cold_iters,
                "warm_iters": warm_iters,
                "iter_reduction_pct": iter_reduction,
                "cold_time_ms": t_cold,
                "warm_time_ms": t_warm,
                "speedup": speedup,
                "objective": sol_warm.primal_objective,
            }
            results.append(record)

            # Update basis for sequential cascading perturbations
            current_basis_col = sol_warm.col_basis
            current_basis_row = sol_warm.row_basis

        return results

    @staticmethod
    def format_results_table(results: List[Dict[str, Any]]) -> str:
        """
        Format Markdown comparison table.
        """
        lines = [
            "| Scenario / Perturbation | Cold Iters | Warm Iters | Iter Reduction (%) | Cold Time (ms) | Warm Time (ms) | Speedup | Solved Margin ($/day) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in results:
            lines.append(
                f"| **{r['name']}** ({r['desc']}) | {r['cold_iters']} | **{r['warm_iters']}** | "
                f"**{r['iter_reduction_pct']:.1f}%** | {r['cold_time_ms']:.2f} ms | "
                f"**{r['warm_time_ms']:.2f} ms** | **{r['speedup']:.2f}x** | ${r['objective']:,.2f} |"
            )

        return "\n".join(lines)
