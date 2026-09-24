"""
Refinery Explainability and Infeasibility Diagnosis Layer.
Translates mathematical solver outputs (shadow prices, sensitivity ranges, slack)
into plain-language business insights for refinery planners and traders,
and isolates Irreducible Inconsistent Subsystems (IIS) for infeasible scenarios.
"""

from typing import Dict, List, Tuple, Optional, Any
import sih_solver
import copy


class RefineryExplainer:
    """
    Explainability engine for solved refinery optimization models.
    Maps mathematical coordinates, row duals (shadow prices), and basis sensitivity ranges
    into intuitive operational narratives.
    """

    def __init__(self, problem: sih_solver.Problem, solution: sih_solver.Solution, metadata: Dict[str, Any]):
        self.problem = problem
        self.solution = solution
        self.metadata = metadata
        self.var_map = metadata.get("var_map", {})
        self.row_names = metadata.get("row_names", [])
        self.builder = metadata.get("builder")

    def explain_shadow_prices(self, tolerance: float = 1e-4) -> List[Dict[str, Any]]:
        """
        Analyze all constraints and produce plain-language interpretations of shadow prices
        and sensitivity validity intervals.
        """
        explanations: List[Dict[str, Any]] = []
        m = self.problem.num_rows()

        row_duals = self.solution.row_duals if len(self.solution.row_duals) == m else [0.0] * m
        slack = self.solution.slack if len(self.solution.slack) == m else [0.0] * m
        rhs_down = self.solution.rhs_down if len(self.solution.rhs_down) == m else [-float("inf")] * m
        rhs_up = self.solution.rhs_up if len(self.solution.rhs_up) == m else [float("inf")] * m

        for i in range(m):
            name = self.row_names[i] if i < len(self.row_names) else f"row_{i}"
            dual = row_duals[i]
            slk = slack[i]
            r_down = rhs_down[i]
            r_up = rhs_up[i]
            is_binding = abs(dual) > tolerance

            category = "Other"
            plain_desc = ""
            current_limit = self.problem.row_upper[i]

            # Categorize constraint and generate human explanation
            if name.startswith("cap_max_"):
                unit_name = name.replace("cap_max_", "")
                category = "Unit Capacity"
                cap_val = current_limit
                if is_binding:
                    val_str = f"${dual:.2f}/bbl"
                    range_str = f"[{r_down:.1f}, {r_up:.1f}] bpd" if r_up < 1e15 else f">= {r_down:.1f} bpd"
                    plain_desc = (
                        f"Processing unit {unit_name} is operating at 100% capacity ({cap_val:,.1f} bpd). "
                        f"Shadow price: {val_str}. Expanding {unit_name} capacity by 1 bpd generates {val_str}/day "
                        f"in additional gross margin (valid over {range_str})."
                    )
                else:
                    throughput = cap_val - slk if slk < 1e15 else 0.0
                    util_pct = (throughput / cap_val * 100.0) if cap_val > 0 else 0.0
                    plain_desc = (
                        f"Processing unit {unit_name} has spare capacity ({throughput:,.1f}/{cap_val:,.1f} bpd, "
                        f"{util_pct:.1f}% utilization). Shadow price: $0.00/bbl (adding capacity provides zero economic benefit)."
                    )

            elif name.startswith("cap_min_"):
                unit_name = name.replace("cap_min_", "")
                category = "Unit Turndown"
                if is_binding:
                    plain_desc = (
                        f"Processing unit {unit_name} is operating at its minimum technical turndown limit. "
                        f"Dual penalty: ${abs(dual):.2f}/bbl."
                    )
                else:
                    plain_desc = f"Processing unit {unit_name} operates comfortably above minimum turndown limit."

            elif name.startswith("spec_ron_"):
                prod_name = name.replace("spec_ron_", "")
                category = "Octane Specification"
                if is_binding:
                    plain_desc = (
                        f"Octane (RON) specification for {prod_name} is strictly binding. "
                        f"Marginal cost of octane: ${abs(dual):.2f}/RON-bbl. Increasing reformer or FCC severity would add value."
                    )
                else:
                    plain_desc = f"Octane specification for {prod_name} has positive giveaway (spec is slack)."

            elif name.startswith("spec_sulfur_"):
                prod_name = name.replace("spec_sulfur_", "")
                category = "Sulfur Specification"
                if is_binding:
                    plain_desc = (
                        f"Sulfur specification for {prod_name} is strictly binding at maximum legal limit. "
                        f"Marginal cost of sulfur compliance: ${abs(dual):.2f}/wt%-bbl. Hydrotreating capacity constrains blend quality."
                    )
                else:
                    plain_desc = f"Sulfur specification for {prod_name} is safely within clean product limits."

            elif name.startswith("bal_"):
                category = "Material Balance"
                plain_desc = f"Mass balance conservation constraint for stream {name.replace('bal_', '')}."

            elif name.startswith("feed_") or name.startswith("yield_"):
                category = "Process Flow / Yield"
                plain_desc = f"Unit yield and conversion balance for {name}."

            elif name.startswith("blend_"):
                category = "Product Blending"
                plain_desc = f"Volumetric blending summation for {name.replace('blend_', '')}."

            explanations.append({
                "row_index": i,
                "name": name,
                "category": category,
                "is_binding": is_binding,
                "shadow_price": dual,
                "slack": slk,
                "rhs_down": r_down,
                "rhs_up": r_up,
                "limit": current_limit,
                "explanation": plain_desc,
            })

        return explanations

    def format_bottlenecks_table(self) -> str:
        """
        Format a Markdown table of active refinery bottlenecks sorted by economic impact (shadow price).
        """
        explanations = self.explain_shadow_prices()
        binding_bottlenecks = [e for e in explanations if e["is_binding"]]
        binding_bottlenecks.sort(key=lambda x: abs(x["shadow_price"]), reverse=True)

        lines = [
            "| Constraint Name | Category | Limit | Shadow Price | Status | Operational Impact |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        if not binding_bottlenecks:
            lines.append("| (None) | - | - | $0.00 | Slack | All constraints have slack |")
        else:
            for b in binding_bottlenecks:
                limit_str = f"{b['limit']:,.1f}" if b['limit'] < 1e15 else "N/A"
                price_str = f"${b['shadow_price']:.2f}/unit"
                lines.append(
                    f"| `{b['name']}` | {b['category']} | {limit_str} | **{price_str}** | **BINDING** | {b['explanation']} |"
                )

        return "\n".join(lines)

    def format_executive_summary(self) -> str:
        """
        Format a high-level executive briefing summarizing solved plant throughput, margin, and debottlenecking.
        """
        obj = self.solution.primal_objective
        explanations = self.explain_shadow_prices()
        binding_units = [e for e in explanations if e["is_binding"] and e["category"] == "Unit Capacity"]
        slack_units = [e for e in explanations if not e["is_binding"] and e["category"] == "Unit Capacity"]

        lines = [
            "### Refinery Operational & Economic Executive Summary",
            f"- **Gross Refining Margin**: **${obj:,.2f}/day**",
            f"- **Solver Status**: `{str(self.solution.status)}` (Wall time: {self.solution.time_wall_sec * 1000.0:.2f} ms)",
            "",
            "#### Active Physical Bottlenecks (Top Value Opportunities):",
        ]

        if binding_units:
            for bu in sorted(binding_units, key=lambda x: abs(x["shadow_price"]), reverse=True):
                lines.append(f"  - **{bu['name']}**: Shadow price = **${bu['shadow_price']:.2f}/bbl**. {bu['explanation']}")
        else:
            lines.append("  - *No processing unit capacity constraints are currently binding.*")

        lines.append("")
        lines.append("#### Underutilized Assets (Available Spare Capacity):")
        for su in slack_units:
            lines.append(f"  - **{su['name']}**: {su['explanation']}")

        return "\n".join(lines)


class InfeasibilityDiagnoser:
    """
    Infeasibility diagnosis engine using Irreducible Inconsistent Subsystem (IIS) deletion filtering.
    Identifies the minimal conflicting subset of refinery constraints causing infeasibility,
    and articulates the physical conflict in plain English.
    """

    def __init__(self, solver_options: Optional[sih_solver.Options] = None):
        if solver_options is None:
            self.options = sih_solver.Options()
            self.options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
            self.options.strategy.presolve = sih_solver.PresolveMode.Off  # Keep raw matrix for direct IIS isolation
            self.options.log_to_console = False
        else:
            self.options = solver_options

    def diagnose(self, problem: sih_solver.Problem, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute minimal conflicting constraint subset (IIS) via systematic deletion filtering.
        """
        # Step 1: Verify problem is infeasible
        initial_sol = sih_solver.solve(problem, self.options)
        if initial_sol.is_feasible():
            return {
                "is_infeasible": False,
                "message": "Problem is feasible. No conflicting constraints exist.",
                "conflicting_constraints": [],
            }

        m = problem.num_rows()
        row_names = metadata.get("row_names", [f"row_{i}" for i in range(m)])
        
        # Clone problem
        work_prob = sih_solver.Problem(problem.name)
        work_prob.sense = problem.sense
        work_prob.resize(m, problem.num_cols())
        work_prob.c = list(problem.c)
        work_prob.col_lower = list(problem.col_lower)
        work_prob.col_upper = list(problem.col_upper)
        work_prob.var_types = list(problem.var_types)
        work_prob.row_lower = list(problem.row_lower)
        work_prob.row_upper = list(problem.row_upper)
        work_prob.set_row_names(row_names)
        work_prob.set_col_names(problem.col_names())
        work_prob.A = problem.A

        cur_row_lower = list(problem.row_lower)
        cur_row_upper = list(problem.row_upper)

        # Set of active constraints initially
        candidate_rows = list(range(m))
        essential_rows: List[int] = []

        # Step 2: IIS Deletion Filtering on Row Constraints
        for row_idx in candidate_rows:
            orig_lower = cur_row_lower[row_idx]
            orig_upper = cur_row_upper[row_idx]
            
            # Temporarily relax row_idx
            cur_row_lower[row_idx] = -float("inf")
            cur_row_upper[row_idx] = float("inf")
            work_prob.row_lower = cur_row_lower
            work_prob.row_upper = cur_row_upper

            # Test if system remains infeasible
            test_sol = sih_solver.solve(work_prob, self.options)

            if test_sol.is_feasible():
                # Removing row_idx made the problem feasible!
                # Therefore, row_idx is ESSENTIAL to the conflict.
                essential_rows.append(row_idx)
                # Restore original bounds
                cur_row_lower[row_idx] = orig_lower
                cur_row_upper[row_idx] = orig_upper
                work_prob.row_lower = cur_row_lower
                work_prob.row_upper = cur_row_upper
            else:
                # Still infeasible without row_idx; row_idx was redundant for the conflict.
                # Keep row_idx relaxed!
                pass

        # Step 3: Analyze Essential Rows and Generate Plain-Language Narrative
        iis_names = [row_names[r] for r in essential_rows]
        narrative, remedies, category = self._explain_conflict(iis_names, problem, essential_rows, metadata)

        return {
            "is_infeasible": True,
            "category": category,
            "conflict_size": len(essential_rows),
            "conflicting_rows": essential_rows,
            "conflicting_row_names": iis_names,
            "plain_language_narrative": narrative,
            "actionable_remedies": remedies,
        }

    def _explain_conflict(
        self,
        iis_names: List[str],
        problem: sih_solver.Problem,
        essential_rows: List[int],
        metadata: Dict[str, Any],
    ) -> Tuple[str, List[str], str]:
        """
        Synthesize plain-language diagnosis and actionable operational remedies from IIS constraint names.
        """
        has_adu_cap = any("cap_max_ADU" in name for name in iis_names)
        has_diesel_blend = any("blend_Diesel" in name or "prod_Diesel" in name for name in iis_names)
        has_sulfur_spec = any("spec_sulfur" in name for name in iis_names)
        has_octane_spec = any("spec_ron" in name for name in iis_names)
        has_hdt_cap = any("cap_max_HDT" in name or "feed_HDT" in name for name in iis_names)

        category = "General Operational Infeasibility"
        remedies: List[str] = []

        if has_adu_cap and has_diesel_blend:
            category = "Throughput Deficit / Excessive Product Demand"
            narrative = (
                f"Minimal Conflicting Constraints ({len(iis_names)} items: {', '.join(iis_names)}):\n"
                f"A fundamental physical deficit exists between contracted finished product commitments "
                f"and total primary distillation capacity. The mandatory minimum delivery requirements for "
                f"distillate fuels exceed the maximum possible intermediate stream yields that can be produced "
                f"from atmospheric distillation (ADU) operating at 100% capacity."
            )
            remedies = [
                "Negotiate contractual reduction in mandatory Diesel/Jet delivery quotas.",
                "Expand ADU distillation throughput limit or schedule debottlenecking revamps.",
                "Procure third-party straight-run atmospheric gasoil (AGO) or finished blendstock on spot markets.",
            ]
        elif has_sulfur_spec and (has_hdt_cap or "yield_HDT" in iis_names or "feed_HDT" in iis_names or has_diesel_blend):
            category = "Hydroprocessing Quality & Distillate Capacity Deficit"
            narrative = (
                f"Minimal Conflicting Constraints ({len(iis_names)} items: {', '.join(iis_names)}):\n"
                f"Severe clean fuels conflict. Contracted finished Diesel production volume requires clean low-sulfur "
                f"blendstocks to meet the 500 ppm sulfur specification (spec_sulfur_Diesel). However, total hydrotreated "
                f"distillate volume from HDT (yield_HDT) is physically capped and cannot supply the required volume for "
                f"Diesel blending (blend_Diesel). Untreated straight-run distillate carries excessive sulfur (8,000 ppm) "
                f"and cannot be blended without violating legal clean air regulations."
            )
            remedies = [
                "Increase HDT unit capacity allocation or utilize catalyst bypass options.",
                "Shift crude slate to sweet crudes (e.g. Brent / Arab Light) with lower native sulfur fractions.",
                "Downgrade high-sulfur stream volumes to marine fuel oil blending where sulfur limits are more permissive.",
            ]
        elif has_octane_spec:
            category = "Octane Quality Deficit"
            narrative = (
                f"Minimal Conflicting Constraints ({len(iis_names)} items: {', '.join(iis_names)}):\n"
                f"Octane specification conflict. Finished premium gasoline minimum RON requirement cannot be satisfied "
                f"because catalytic reforming (REF) and FCC gasoline production cannot supply sufficient high-octane blend components."
            )
            remedies = [
                "Increase catalytic reformer severity or feed rate.",
                "Import high-octane blendstocks (e.g. Alkylate or Ethanol).",
                "Reduce premium gasoline production quota in favor of regular 87 RON gasoline.",
            ]
        else:
            narrative = (
                f"Minimal Conflicting Constraints ({len(iis_names)} items: {', '.join(iis_names)}):\n"
                f"The solver isolated a minimal conflicting subsystem comprising the constraints listed above. "
                f"Simultaneously enforcing these mass balance, capacity, and quality boundaries results in an empty primal feasible region."
            )
            remedies = [
                "Relax non-essential demand upper/lower bounds.",
                "Verify whether simultaneous turnaround maintenance on multiple conversion units is feasible.",
            ]

        return narrative, remedies, category
