"""
Refinery Plan Comparator Utility.
Quantifies economic, operational, and structural shifts between two solved refinery plans
(e.g., baseline vs crude price shock, turnaround outage, or demand surge).
Reports deltas in throughputs, margins, and identifies newly binding vs newly slack constraints.
"""

from typing import Dict, List, Tuple, Optional, Any
import sih_solver


class RefineryPlanComparator:
    """
    Compares two refinery solutions, reporting quantitative deltas and constraint transitions.
    """

    def __init__(
        self,
        prob_a: sih_solver.Problem,
        sol_a: sih_solver.Solution,
        meta_a: Dict[str, Any],
        prob_b: sih_solver.Problem,
        sol_b: sih_solver.Solution,
        meta_b: Dict[str, Any],
        label_a: str = "Plan A (Baseline)",
        label_b: str = "Plan B (Scenario)",
        binding_tol: float = 1e-4,
    ):
        self.prob_a = prob_a
        self.sol_a = sol_a
        self.meta_a = meta_a
        self.prob_b = prob_b
        self.sol_b = sol_b
        self.meta_b = meta_b
        self.label_a = label_a
        self.label_b = label_b
        self.binding_tol = binding_tol

    def compare_economics(self) -> Dict[str, Any]:
        """
        Compare top-level objective value (gross margin) and solver performance.
        """
        margin_a = self.sol_a.primal_objective
        margin_b = self.sol_b.primal_objective
        delta_margin = margin_b - margin_a
        pct_margin = (delta_margin / abs(margin_a) * 100.0) if abs(margin_a) > 1e-6 else 0.0

        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "margin_a": margin_a,
            "margin_b": margin_b,
            "delta_margin": delta_margin,
            "pct_margin": pct_margin,
            "time_a_sec": self.sol_a.time_wall_sec,
            "time_b_sec": self.sol_b.time_wall_sec,
        }

    def compare_crudes(self) -> List[Dict[str, Any]]:
        """
        Compare crude intake quantities between Plan A and Plan B.
        """
        crude_deltas: List[Dict[str, Any]] = []
        builder_a = self.meta_a.get("builder")
        var_map_a = self.meta_a.get("var_map", {})
        var_map_b = self.meta_b.get("var_map", {})

        if not builder_a:
            return crude_deltas

        for c_name in builder_a.crudes.keys():
            var_key = f"crude_{c_name}"
            val_a = self.sol_a.x[var_map_a[var_key]] if var_key in var_map_a else 0.0
            val_b = self.sol_b.x[var_map_b[var_key]] if var_key in var_map_b else 0.0
            delta = val_b - val_a
            pct = (delta / val_a * 100.0) if val_a > 1e-6 else (100.0 if val_b > 1e-6 else 0.0)

            crude_deltas.append({
                "crude": c_name,
                "val_a": val_a,
                "val_b": val_b,
                "delta": delta,
                "pct": pct,
            })

        return crude_deltas

    def compare_units(self) -> List[Dict[str, Any]]:
        """
        Compare processing unit throughput and utilization between Plan A and Plan B.
        """
        unit_deltas: List[Dict[str, Any]] = []
        builder_a = self.meta_a.get("builder")
        builder_b = self.meta_b.get("builder")
        var_map_a = self.meta_a.get("var_map", {})
        var_map_b = self.meta_b.get("var_map", {})

        if not builder_a:
            return unit_deltas

        for u_name in builder_a.units.keys():
            var_key = f"unit_{u_name}"
            val_a = self.sol_a.x[var_map_a[var_key]] if var_key in var_map_a else 0.0
            val_b = self.sol_b.x[var_map_b[var_key]] if var_key in var_map_b else 0.0
            cap_a = builder_a.units[u_name]["capacity"]
            cap_b = builder_b.units[u_name]["capacity"] if builder_b else cap_a
            
            util_a = (val_a / cap_a * 100.0) if cap_a > 0 else 0.0
            util_b = (val_b / cap_b * 100.0) if cap_b > 0 else 0.0

            unit_deltas.append({
                "unit": u_name,
                "throughput_a": val_a,
                "throughput_b": val_b,
                "delta_bpd": val_b - val_a,
                "util_a_pct": util_a,
                "util_b_pct": util_b,
                "delta_util_pct": util_b - util_a,
            })

        return unit_deltas

    def compare_products(self) -> List[Dict[str, Any]]:
        """
        Compare finished product deliveries between Plan A and Plan B.
        """
        prod_deltas: List[Dict[str, Any]] = []
        builder_a = self.meta_a.get("builder")
        var_map_a = self.meta_a.get("var_map", {})
        var_map_b = self.meta_b.get("var_map", {})

        if not builder_a:
            return prod_deltas

        for p_name in builder_a.products.keys():
            var_key = f"prod_{p_name}"
            val_a = self.sol_a.x[var_map_a[var_key]] if var_key in var_map_a else 0.0
            val_b = self.sol_b.x[var_map_b[var_key]] if var_key in var_map_b else 0.0
            delta = val_b - val_a
            pct = (delta / val_a * 100.0) if val_a > 1e-6 else (100.0 if val_b > 1e-6 else 0.0)

            prod_deltas.append({
                "product": p_name,
                "val_a": val_a,
                "val_b": val_b,
                "delta": delta,
                "pct": pct,
            })

        return prod_deltas

    def compare_constraints(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify constraint status migrations:
        - newly_binding: slack in A, binding in B
        - newly_slack: binding in A, slack in B
        - persistent_binding: binding in both A and B
        - persistent_slack: slack in both A and B
        """
        row_names_a = self.meta_a.get("row_names", [])
        row_names_b = self.meta_b.get("row_names", [])
        
        # Build map name -> (dual, slack)
        duals_a = {
            row_names_a[i]: self.sol_a.row_duals[i]
            for i in range(min(len(row_names_a), len(self.sol_a.row_duals)))
        }
        duals_b = {
            row_names_b[i]: self.sol_b.row_duals[i]
            for i in range(min(len(row_names_b), len(self.sol_b.row_duals)))
        }

        newly_binding: List[Dict[str, Any]] = []
        newly_slack: List[Dict[str, Any]] = []
        persistent_binding: List[Dict[str, Any]] = []

        all_names = set(duals_a.keys()).intersection(duals_b.keys())

        for name in sorted(all_names):
            ya = duals_a[name]
            yb = duals_b[name]
            is_bind_a = abs(ya) > self.binding_tol
            is_bind_b = abs(yb) > self.binding_tol

            record = {
                "name": name,
                "shadow_price_a": ya,
                "shadow_price_b": yb,
                "delta_shadow_price": yb - ya,
            }

            if not is_bind_a and is_bind_b:
                newly_binding.append(record)
            elif is_bind_a and not is_bind_b:
                newly_slack.append(record)
            elif is_bind_a and is_bind_b:
                persistent_binding.append(record)

        return {
            "newly_binding": newly_binding,
            "newly_slack": newly_slack,
            "persistent_binding": persistent_binding,
        }

    def format_comparison_report(self) -> str:
        """
        Generate a comprehensive executive comparison report in Markdown.
        """
        econ = self.compare_economics()
        crudes = self.compare_crudes()
        units = self.compare_units()
        prods = self.compare_products()
        constrs = self.compare_constraints()

        lines = [
            f"# Refinery Plan Comparison: {self.label_a} vs {self.label_b}",
            "",
            "## 1. Executive Economic Summary",
            f"- **{self.label_a} Margin**: ${econ['margin_a']:,.2f}/day",
            f"- **{self.label_b} Margin**: ${econ['margin_b']:,.2f}/day",
            f"- **Margin Delta**: **{'+' if econ['delta_margin'] >= 0 else ''}${econ['delta_margin']:,.2f}/day ({econ['pct_margin']:+.2f}%)**",
            "",
            "## 2. Crude Slate Allocation Delta",
            "| Crude Name | Plan A (bpd) | Plan B (bpd) | Delta (bpd) | Delta (%) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for c in crudes:
            lines.append(f"| {c['crude']} | {c['val_a']:,.1f} | {c['val_b']:,.1f} | {c['delta']:+,.1f} | {c['pct']:+.1f}% |")

        lines.extend([
            "",
            "## 3. Processing Unit Throughput & Utilization Delta",
            "| Unit | Plan A (bpd) | Util A (%) | Plan B (bpd) | Util B (%) | Delta (bpd) | Delta Util (%) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for u in units:
            lines.append(
                f"| {u['unit']} | {u['throughput_a']:,.1f} | {u['util_a_pct']:.1f}% | "
                f"{u['throughput_b']:,.1f} | {u['util_b_pct']:.1f}% | {u['delta_bpd']:+,.1f} | {u['delta_util_pct']:+.1f}% |"
            )

        lines.extend([
            "",
            "## 4. Finished Product Delivery Delta",
            "| Product | Plan A (bpd) | Plan B (bpd) | Delta (bpd) | Delta (%) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for p in prods:
            lines.append(f"| {p['product']} | {p['val_a']:,.1f} | {p['val_b']:,.1f} | {p['delta']:+,.1f} | {p['pct']:+.1f}% |")

        lines.extend([
            "",
            "## 5. Constraint Bottleneck Migration Analysis",
            f"### Newly Binding Constraints ({len(constrs['newly_binding'])} items):",
        ])
        if constrs["newly_binding"]:
            for nb in constrs["newly_binding"]:
                lines.append(f"- `{nb['name']}`: Previously SLACK -> Now **BINDING** (Shadow Price = ${nb['shadow_price_b']:.2f}/unit)")
        else:
            lines.append("- *(No newly binding constraints)*")

        lines.append(f"\n### Newly Slack Constraints ({len(constrs['newly_slack'])} items):")
        if constrs["newly_slack"]:
            for ns in constrs["newly_slack"]:
                lines.append(f"- `{ns['name']}`: Previously BINDING (${ns['shadow_price_a']:.2f}/unit) -> Now **SLACK** ($0.00/unit)")
        else:
            lines.append("- *(No newly slack constraints)*")

        lines.append(f"\n### Persistent Bottlenecks ({len(constrs['persistent_binding'])} items):")
        if constrs["persistent_binding"]:
            for pb in constrs["persistent_binding"]:
                lines.append(f"- `{pb['name']}`: Remained binding (Shadow price: ${pb['shadow_price_a']:.2f} -> ${pb['shadow_price_b']:.2f}/unit)")
        else:
            lines.append("- *(No persistent bottlenecks)*")

        return "\n".join(lines)
