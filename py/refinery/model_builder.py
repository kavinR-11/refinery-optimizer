"""
Refinery Planning Model Generator (LP and MILP).
Builds operational crude selection, unit processing, and product blending optimization models.
"""

from typing import Dict, List, Tuple, Optional, Any
import sih_solver

class RefineryModelBuilder:
    def __init__(self, is_milp: bool = False):
        self.is_milp = is_milp
        
        # Crude slate definitions
        self.crudes = {
            "ArabLight": {"price": 80.0, "max_supply": 100000.0, "api": 33.4, "sulfur": 1.8},
            "Brent":     {"price": 85.0, "max_supply": 80000.0,  "api": 38.3, "sulfur": 0.4},
            "MayaHeavy": {"price": 70.0, "max_supply": 60000.0,  "api": 21.8, "sulfur": 3.5},
        }
        
        # Distillation yields from ADU per crude (sum <= 1.0)
        self.adu_yields = {
            "ArabLight": {"LPG": 0.03, "LightNaphtha": 0.18, "HeavyNaphtha": 0.15, "Distillate": 0.30, "Residue": 0.34},
            "Brent":     {"LPG": 0.04, "LightNaphtha": 0.22, "HeavyNaphtha": 0.16, "Distillate": 0.32, "Residue": 0.26},
            "MayaHeavy": {"LPG": 0.02, "LightNaphtha": 0.10, "HeavyNaphtha": 0.12, "Distillate": 0.24, "Residue": 0.52},
        }
        
        # Processing unit capacities, costs, and turndown
        self.units = {
            "ADU": {"capacity": 100000.0, "min_turndown": 30000.0, "var_cost": 1.50, "fixed_cost": 15000.0},
            "VDU": {"capacity": 45000.0,  "min_turndown": 15000.0, "var_cost": 1.80, "fixed_cost": 8000.0},
            "FCC": {"capacity": 35000.0,  "min_turndown": 10000.0, "var_cost": 3.50, "fixed_cost": 12000.0},
            "REF": {"capacity": 25000.0,  "min_turndown": 8000.0,  "var_cost": 4.00, "fixed_cost": 10000.0},
            "HDT": {"capacity": 40000.0,  "min_turndown": 12000.0, "var_cost": 2.20, "fixed_cost": 7000.0},
        }
        
        # Finished product market demands and prices ($/bbl)
        self.products = {
            "Gasoline_Regular": {"price": 110.0, "min_demand": 15000.0, "max_demand": 45000.0, "min_ron": 87.0, "max_sulfur": 0.05},
            "Gasoline_Premium": {"price": 125.0, "min_demand": 5000.0,  "max_demand": 20000.0, "min_ron": 93.0, "max_sulfur": 0.03},
            "JetFuel":          {"price": 118.0, "min_demand": 10000.0, "max_demand": 30000.0, "max_sulfur": 0.04},
            "Diesel":           {"price": 112.0, "min_demand": 20000.0, "max_demand": 50000.0, "max_sulfur": 0.05, "min_cetane": 48.0},
            "FuelOil":          {"price": 65.0,  "min_demand": 5000.0,  "max_demand": 30000.0, "max_sulfur": 2.5},
            "LPG":              {"price": 75.0,  "min_demand": 2000.0,  "max_demand": 15000.0},
        }
        
        # Stream blending properties (Octane RON, Sulfur wt%, Cetane)
        self.stream_props = {
            "LightNaphtha":   {"ron": 70.0, "sulfur": 0.05, "cetane": 20.0},
            "Reformate":      {"ron": 98.0, "sulfur": 0.005, "cetane": 15.0},
            "FCC_Gasoline":   {"ron": 92.0, "sulfur": 0.02, "cetane": 25.0},
            "Distillate_Raw": {"ron": 30.0, "sulfur": 0.8, "cetane": 46.0},
            "Distillate_HDT": {"ron": 32.0, "sulfur": 0.01, "cetane": 52.0},
            "VGO":            {"ron": 25.0, "sulfur": 1.5, "cetane": 35.0},
            "CycleOil_Raw":   {"ron": 40.0, "sulfur": 1.2, "cetane": 38.0},
            "CycleOil_HDT":   {"ron": 42.0, "sulfur": 0.02, "cetane": 45.0},
            "VacuumResidue":  {"ron": 10.0, "sulfur": 3.2, "cetane": 20.0},
        }

    def build_problem(self) -> Tuple[sih_solver.Problem, Dict[str, Any]]:
        """
        Construct and return the sih_solver.Problem and a metadata dictionary
        containing variable and constraint mappings.
        """
        var_names: List[str] = []
        c_obj: List[double] = []
        col_lower: List[double] = []
        col_upper: List[double] = []
        var_types: List[sih_solver.VariableType] = []
        
        def add_var(name: str, cost: float, lb: float, ub: float, vt: sih_solver.VariableType) -> int:
            idx = len(var_names)
            var_names.append(name)
            c_obj.append(cost)
            col_lower.append(lb)
            col_upper.append(ub)
            var_types.append(vt)
            return idx

        var_map: Dict[str, int] = {}
        
        # 1. Crude purchase variables x_c
        for c_name, c_info in self.crudes.items():
            # In maximization: profit += Revenue - Cost, so objective coeff = -price
            var_map[f"crude_{c_name}"] = add_var(
                f"crude_{c_name}", -c_info["price"], 0.0, c_info["max_supply"], sih_solver.VariableType.Continuous
            )

        # 2. Unit throughput variables f_u
        for u_name, u_info in self.units.items():
            var_map[f"unit_{u_name}"] = add_var(
                f"unit_{u_name}", -u_info["var_cost"], 0.0, u_info["capacity"], sih_solver.VariableType.Continuous
            )
            
        # 3. MILP Unit on/off binary variables z_u
        if self.is_milp:
            for u_name, u_info in self.units.items():
                var_map[f"z_{u_name}"] = add_var(
                    f"z_{u_name}", -u_info["fixed_cost"], 0.0, 1.0, sih_solver.VariableType.Binary
                )

        # 4. Finished product sale variables y_p
        for p_name, p_info in self.products.items():
            var_map[f"prod_{p_name}"] = add_var(
                f"prod_{p_name}", p_info["price"], p_info["min_demand"], p_info["max_demand"], sih_solver.VariableType.Continuous
            )

        # 5. Intermediate blending flow variables w_{stream, product/unit}
        blend_links = [
            # Naphtha / Reformate to Gasolines
            ("LightNaphtha", "Gasoline_Regular"),
            ("LightNaphtha", "Gasoline_Premium"),
            ("FCC_Gasoline", "Gasoline_Regular"),
            ("FCC_Gasoline", "Gasoline_Premium"),
            ("Reformate", "Gasoline_Regular"),
            ("Reformate", "Gasoline_Premium"),
            # Reforming feed
            ("HeavyNaphtha", "unit_REF"),
            # VDU feed
            ("Residue", "unit_VDU"),
            # FCC feed
            ("VGO", "unit_FCC"),
            # Hydrotreating feed
            ("Distillate", "unit_HDT"),
            ("CycleOil", "unit_HDT"),
            # Diesel & Jet blending
            ("Distillate_HDT", "Diesel"),
            ("Distillate_HDT", "JetFuel"),
            ("Distillate_Raw", "Diesel"),
            ("CycleOil_HDT", "Diesel"),
            ("CycleOil_Raw", "Diesel"),
            ("CycleOil_Raw", "FuelOil"),
            # Fuel oil blending
            ("VacuumResidue", "FuelOil"),
            ("VGO", "FuelOil"),
            ("Distillate_Raw", "FuelOil"),
        ]
        
        for s_from, target in blend_links:
            var_map[f"flow_{s_from}_to_{target}"] = add_var(
                f"flow_{s_from}_to_{target}", 0.0, 0.0, float("inf"), sih_solver.VariableType.Continuous
            )

        # Constraints
        triplets: List[sih_solver.Triplet] = []
        row_names: List[str] = []
        row_lower: List[double] = []
        row_upper: List[double] = []

        def add_constraint(name: str, terms: List[Tuple[str, float]], lb: float, ub: float):
            row_idx = len(row_names)
            row_names.append(name)
            row_lower.append(lb)
            row_upper.append(ub)
            for var_name, coeff in terms:
                triplets.append(sih_solver.Triplet(row_idx, var_map[var_name], coeff))

        # Constraint 1: ADU throughput = sum of crudes
        adu_terms = [("unit_ADU", -1.0)] + [(f"crude_{c}", 1.0) for c in self.crudes]
        add_constraint("bal_ADU_feed", adu_terms, 0.0, 0.0)

        # Constraint 2: ADU capacity & semi-continuous turndown (if MILP)
        if self.is_milp:
            # f_u - Cap_max * z_u <= 0
            # f_u - Cap_min * z_u >= 0
            for u_name, u_info in self.units.items():
                add_constraint(f"cap_max_{u_name}", [(f"unit_{u_name}", 1.0), (f"z_{u_name}", -u_info["capacity"])], -float("inf"), 0.0)
                add_constraint(f"cap_min_{u_name}", [(f"unit_{u_name}", 1.0), (f"z_{u_name}", -u_info["min_turndown"])], 0.0, float("inf"))
        else:
            for u_name, u_info in self.units.items():
                add_constraint(f"cap_max_{u_name}", [(f"unit_{u_name}", 1.0)], 0.0, u_info["capacity"])

        # Constraint 3: ADU Cut Balances: stream_yield = sum(yield_c * crude_c)
        # Light Naphtha balance
        ln_terms = [(f"flow_LightNaphtha_to_Gasoline_Regular", 1.0), (f"flow_LightNaphtha_to_Gasoline_Premium", 1.0)]
        for c in self.crudes:
            ln_terms.append((f"crude_{c}", -self.adu_yields[c]["LightNaphtha"]))
        add_constraint("bal_LightNaphtha", ln_terms, -float("inf"), 0.0)

        # Heavy Naphtha to Reformer
        hn_terms = [(f"flow_HeavyNaphtha_to_unit_REF", 1.0)]
        for c in self.crudes:
            hn_terms.append((f"crude_{c}", -self.adu_yields[c]["HeavyNaphtha"]))
        add_constraint("bal_HeavyNaphtha", hn_terms, -float("inf"), 0.0)

        # Distillate balance
        dist_terms = [(f"flow_Distillate_to_unit_HDT", 1.0)]
        for c in self.crudes:
            dist_terms.append((f"crude_{c}", -self.adu_yields[c]["Distillate"]))
        add_constraint("bal_Distillate", dist_terms, -float("inf"), 0.0)

        # Atmospheric Residue to VDU
        res_terms = [(f"flow_Residue_to_unit_VDU", 1.0)]
        for c in self.crudes:
            res_terms.append((f"crude_{c}", -self.adu_yields[c]["Residue"]))
        add_constraint("bal_Residue", res_terms, -float("inf"), 0.0)

        # ADU LPG directly to LPG product
        lpg_terms = [("prod_LPG", 1.0)]
        for c in self.crudes:
            lpg_terms.append((f"crude_{c}", -self.adu_yields[c]["LPG"]))
        add_constraint("bal_LPG_prod", lpg_terms, -float("inf"), 0.0)

        # Constraint 4: Unit Feed Balances
        # VDU: feed = flow_Residue_to_unit_VDU
        add_constraint("feed_VDU", [("unit_VDU", 1.0), ("flow_Residue_to_unit_VDU", -1.0)], 0.0, 0.0)
        # REF: feed = flow_HeavyNaphtha_to_unit_REF
        add_constraint("feed_REF", [("unit_REF", 1.0), ("flow_HeavyNaphtha_to_unit_REF", -1.0)], 0.0, 0.0)
        # HDT: feed = Distillate + CycleOil
        add_constraint("feed_HDT", [("unit_HDT", 1.0), ("flow_Distillate_to_unit_HDT", -1.0), ("flow_CycleOil_to_unit_HDT", -1.0)], 0.0, 0.0)
        # FCC: feed = flow_VGO_to_unit_FCC
        add_constraint("feed_FCC", [("unit_FCC", 1.0), ("flow_VGO_to_unit_FCC", -1.0)], 0.0, 0.0)

        # Constraint 5: Unit Product Yields
        # VDU yields 75% VGO, 25% Vacuum Residue
        add_constraint("yield_VGO", [("flow_VGO_to_unit_FCC", 1.0), ("flow_VGO_to_FuelOil", 1.0), ("unit_VDU", -0.75)], -float("inf"), 0.0)
        add_constraint("yield_VacRes", [("flow_VacuumResidue_to_FuelOil", 1.0), ("unit_VDU", -0.25)], -float("inf"), 0.0)

        # REF yields 85% Reformate
        add_constraint("yield_Reformate", [
            ("flow_Reformate_to_Gasoline_Regular", 1.0),
            ("flow_Reformate_to_Gasoline_Premium", 1.0),
            ("unit_REF", -0.85)
        ], -float("inf"), 0.0)

        # FCC yields 55% FCC Gasoline, 25% Cycle Oil
        add_constraint("yield_FCC_Gas", [
            ("flow_FCC_Gasoline_to_Gasoline_Regular", 1.0),
            ("flow_FCC_Gasoline_to_Gasoline_Premium", 1.0),
            ("unit_FCC", -0.55)
        ], -float("inf"), 0.0)
        add_constraint("yield_CycleOil", [
            ("flow_CycleOil_to_unit_HDT", 1.0),
            ("flow_CycleOil_Raw_to_Diesel", 1.0),
            ("flow_CycleOil_Raw_to_FuelOil", 1.0),
            ("unit_FCC", -0.25)
        ], -float("inf"), 0.0)

        # HDT yields 98% hydrotreated distillate
        add_constraint("yield_HDT", [
            ("flow_Distillate_HDT_to_Diesel", 1.0),
            ("flow_Distillate_HDT_to_JetFuel", 1.0),
            ("flow_CycleOil_HDT_to_Diesel", 1.0),
            ("unit_HDT", -0.98)
        ], -float("inf"), 0.0)

        # Constraint 6: Product Blending Sums
        # Gasoline Regular
        add_constraint("blend_Gas_Reg", [
            ("prod_Gasoline_Regular", -1.0),
            ("flow_LightNaphtha_to_Gasoline_Regular", 1.0),
            ("flow_FCC_Gasoline_to_Gasoline_Regular", 1.0),
            ("flow_Reformate_to_Gasoline_Regular", 1.0)
        ], 0.0, 0.0)

        # Gasoline Premium
        add_constraint("blend_Gas_Prem", [
            ("prod_Gasoline_Premium", -1.0),
            ("flow_LightNaphtha_to_Gasoline_Premium", 1.0),
            ("flow_FCC_Gasoline_to_Gasoline_Premium", 1.0),
            ("flow_Reformate_to_Gasoline_Premium", 1.0)
        ], 0.0, 0.0)

        # Jet Fuel
        add_constraint("blend_JetFuel", [
            ("prod_JetFuel", -1.0),
            ("flow_Distillate_HDT_to_JetFuel", 1.0)
        ], 0.0, 0.0)

        # Diesel
        add_constraint("blend_Diesel", [
            ("prod_Diesel", -1.0),
            ("flow_Distillate_HDT_to_Diesel", 1.0),
            ("flow_Distillate_Raw_to_Diesel", 1.0),
            ("flow_CycleOil_HDT_to_Diesel", 1.0),
            ("flow_CycleOil_Raw_to_Diesel", 1.0)
        ], 0.0, 0.0)

        # Fuel Oil
        add_constraint("blend_FuelOil", [
            ("prod_FuelOil", -1.0),
            ("flow_VacuumResidue_to_FuelOil", 1.0),
            ("flow_VGO_to_FuelOil", 1.0),
            ("flow_Distillate_Raw_to_FuelOil", 1.0),
            ("flow_CycleOil_Raw_to_FuelOil", 1.0)
        ], 0.0, 0.0)

        # Constraint 7: Quality Specifications (Octane RON for Gasoline)
        # Regular: sum (RON_s - 87) * flow_s >= 0
        ron_reg_terms = [
            ("flow_LightNaphtha_to_Gasoline_Regular", 70.0 - 87.0),
            ("flow_FCC_Gasoline_to_Gasoline_Regular", 92.0 - 87.0),
            ("flow_Reformate_to_Gasoline_Regular",    98.0 - 87.0),
        ]
        add_constraint("spec_ron_Gas_Reg", ron_reg_terms, 0.0, float("inf"))

        # Premium: sum (RON_s - 93) * flow_s >= 0
        ron_prem_terms = [
            ("flow_LightNaphtha_to_Gasoline_Premium", 70.0 - 93.0),
            ("flow_FCC_Gasoline_to_Gasoline_Premium", 92.0 - 93.0),
            ("flow_Reformate_to_Gasoline_Premium",    98.0 - 93.0),
        ]
        add_constraint("spec_ron_Gas_Prem", ron_prem_terms, 0.0, float("inf"))

        # Constraint 8: Sulfur Specification on Diesel (Max 0.05% = 500 ppm)
        # sum (Sulfur_s - 0.05) * flow_s <= 0
        sulf_diesel_terms = [
            ("flow_Distillate_HDT_to_Diesel", 0.01 - 0.05),
            ("flow_Distillate_Raw_to_Diesel", 0.80 - 0.05),
            ("flow_CycleOil_HDT_to_Diesel",   0.02 - 0.05),
            ("flow_CycleOil_Raw_to_Diesel",   1.20 - 0.05),
        ]
        add_constraint("spec_sulfur_Diesel", sulf_diesel_terms, -float("inf"), 0.0)

        # Assemble Problem
        p = sih_solver.Problem("refinery_planning")
        p.sense = sih_solver.ObjectiveSense.Maximize
        p.resize(len(row_names), len(var_names))
        p.c = c_obj
        p.col_lower = col_lower
        p.col_upper = col_upper
        p.var_types = var_types
        p.row_lower = row_lower
        p.row_upper = row_upper
        p.set_row_names(row_names)
        p.set_col_names(var_names)
        p.A = sih_solver.SparseMatrix.from_triplets(len(row_names), len(var_names), triplets)

        metadata = {
            "var_names": var_names,
            "row_names": row_names,
            "var_map": var_map,
            "builder": self,
        }
        return p, metadata
