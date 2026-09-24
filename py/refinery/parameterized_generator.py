"""
Parameterized Refinery Model Generator.
Generates diverse refinery LP and MILP model instances by perturbing prices,
demands, capacities, and crude slates under varied market regimes.
Used for building training/testing benchmark datasets and stress-testing solvers.
"""

import copy
import random
import os
from typing import Dict, List, Tuple, Optional, Any
import sih_solver
from refinery.model_builder import RefineryModelBuilder


class ParameterizedRefineryGenerator:
    """
    Generator for creating randomized or regime-specific refinery planning problems.
    Supports continuous price/demand/capacity perturbations and macro-economic market regimes.
    """

    def __init__(self, base_builder: Optional[RefineryModelBuilder] = None):
        if base_builder is None:
            self.base_builder = RefineryModelBuilder(is_milp=False)
        else:
            self.base_builder = base_builder

    def generate_instance(
        self,
        seed: Optional[int] = None,
        price_std: float = 0.10,
        demand_std: float = 0.15,
        capacity_std: float = 0.08,
        market_regime: str = "normal",
        is_milp: bool = False,
    ) -> Tuple[sih_solver.Problem, Dict[str, Any], Dict[str, Any]]:
        """
        Generate a single perturbed refinery instance.

        Parameters:
            seed: Random seed for reproducibility.
            price_std: Standard deviation of price perturbations (relative).
            demand_std: Standard deviation of demand perturbations (relative).
            capacity_std: Standard deviation of unit capacity perturbations.
            market_regime: Macro scenario ('normal', 'high_crude', 'diesel_surge', 'turnaround', 'sour_crude').
            is_milp: Whether to build an LP or a MILP with unit commitment binaries.

        Returns:
            (problem, metadata, scenario_params)
        """
        rng = random.Random(seed)
        builder = copy.deepcopy(self.base_builder)
        builder.is_milp = is_milp

        scenario_params: Dict[str, Any] = {
            "seed": seed,
            "market_regime": market_regime,
            "is_milp": is_milp,
            "price_std": price_std,
            "demand_std": demand_std,
            "capacity_std": capacity_std,
        }

        # 1. Perturb Crude Prices and Availability
        crude_price_mult = 1.0
        if market_regime == "high_crude":
            crude_price_mult = 1.30
        elif market_regime == "sour_crude":
            # Sour crudes discounted, sweet crudes premium
            crude_price_mult = 1.10

        for c_name, c_info in builder.crudes.items():
            perturb = rng.gauss(0.0, price_std)
            perturb = max(-0.30, min(0.40, perturb))
            
            regime_factor = crude_price_mult
            if market_regime == "sour_crude" and c_name == "MayaHeavy":
                regime_factor = 0.85
            elif market_regime == "sour_crude" and c_name == "Brent":
                regime_factor = 1.25

            c_info["price"] = round(c_info["price"] * regime_factor * (1.0 + perturb), 2)

            # Supply perturbation
            supply_perturb = rng.gauss(0.0, demand_std)
            supply_factor = 1.0 + max(-0.40, min(0.50, supply_perturb))
            if market_regime == "sweet_crude_shortage" and c_name == "Brent":
                supply_factor *= 0.35
            c_info["max_supply"] = round(c_info["max_supply"] * supply_factor, 1)

        # 2. Perturb Processing Unit Capacities
        for u_name, u_info in builder.units.items():
            cap_perturb = rng.gauss(0.0, capacity_std)
            cap_factor = 1.0 + max(-0.25, min(0.30, cap_perturb))
            
            # Regime: Turnaround / maintenance shutdown on selected units
            if market_regime == "turnaround":
                if u_name == "FCC":
                    cap_factor *= 0.65  # 35% FCC reduction puts it right at bottleneck
            
            u_info["capacity"] = round(u_info["capacity"] * cap_factor, 1)
            u_info["min_turndown"] = round(u_info["min_turndown"] * cap_factor, 1)

        # 3. Perturb Finished Product Demands and Prices
        prod_price_mult = 1.0
        if market_regime == "diesel_surge":
            prod_price_mult = 1.30

        for p_name, p_info in builder.products.items():
            price_perturb = rng.gauss(0.0, price_std)
            p_factor = 1.0 + max(-0.25, min(0.35, price_perturb))
            if market_regime == "diesel_surge" and "Diesel" in p_name:
                p_factor *= 1.25
            p_info["price"] = round(p_info["price"] * prod_price_mult * p_factor, 2)

            # Demand bounds perturbation
            d_perturb = rng.gauss(0.0, demand_std)
            d_factor = 1.0 + max(-0.20, min(0.25, d_perturb))
            
            p_info["min_demand"] = round(p_info["min_demand"] * d_factor, 1)
            
            max_mult = 1.60 if (market_regime == "diesel_surge" and "Diesel" in p_name) else 1.25
            p_info["max_demand"] = round(max(p_info["min_demand"] * 1.2, p_info["max_demand"] * d_factor * max_mult), 1)

        # Store perturbed definitions in scenario_params
        scenario_params["crudes"] = copy.deepcopy(builder.crudes)
        scenario_params["units"] = copy.deepcopy(builder.units)
        scenario_params["products"] = copy.deepcopy(builder.products)

        # Build problem instance
        prob, meta = builder.build_problem()
        meta["scenario_params"] = scenario_params
        return prob, meta, scenario_params

    def generate_dataset(
        self,
        num_samples: int = 50,
        train_split: float = 0.8,
        base_seed: int = 1000,
        is_milp: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a train/test benchmark dataset of refinery planning problems.

        Parameters:
            num_samples: Total number of scenario instances.
            train_split: Fraction allocated to training split.
            base_seed: Starting seed.
            is_milp: Whether instances are MILP.
            output_dir: If provided, exports each instance to an MPS file in output_dir.

        Returns:
            Dictionary with 'train' and 'test' lists of scenario instance dictionaries.
        """
        regimes = ["normal", "normal", "high_crude", "diesel_surge", "turnaround", "sour_crude"]
        dataset: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}

        if output_dir:
            os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

        num_train = int(num_samples * train_split)

        for i in range(num_samples):
            seed = base_seed + i
            split = "train" if i < num_train else "test"
            regime = regimes[i % len(regimes)]

            prob, meta, params = self.generate_instance(
                seed=seed,
                market_regime=regime,
                is_milp=is_milp,
            )

            record = {
                "id": f"refinery_{split}_{i:04d}",
                "split": split,
                "regime": regime,
                "seed": seed,
                "problem": prob,
                "metadata": meta,
                "params": params,
                "num_rows": prob.num_rows(),
                "num_cols": prob.num_cols(),
                "num_nonzeros": prob.num_nonzeros(),
            }

            if output_dir:
                mps_path = os.path.join(output_dir, split, f"{record['id']}.mps")
                sih_solver.MpsReader.write(prob, mps_path)
                record["mps_path"] = mps_path

            dataset[split].append(record)

        return dataset
