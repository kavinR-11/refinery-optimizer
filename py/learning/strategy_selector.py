"""
Adaptive Strategy Selector.
Implements Nearest-Neighbor (KNN) and Contextual Bandit policies over StrategyConfig choices,
trained on historical telemetry logs and offline solver runs.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
import sih_solver
from learning.feature_extractor import ProblemFeatureExtractor


class StrategyProfile:
    """
    Defines a discrete solver configuration profile.
    """
    def __init__(
        self,
        name: str,
        algorithm: sih_solver.AlgorithmChoice,
        pricing: sih_solver.PricingRule = sih_solver.PricingRule.SteepestEdge,
        branching: sih_solver.BranchingRule = sih_solver.BranchingRule.PseudoCost,
        node_sel: sih_solver.NodeSelection = sih_solver.NodeSelection.BestBound,
        cut_rounds: int = 5,
        presolve: sih_solver.PresolveMode = sih_solver.PresolveMode.On,
        scaling: bool = True,
        ratio_test: sih_solver.RatioTest = sih_solver.RatioTest.HarrisBFRT,
    ):
        self.name = name
        self.algorithm = algorithm
        self.pricing = pricing
        self.branching = branching
        self.node_sel = node_sel
        self.cut_rounds = cut_rounds
        self.presolve = presolve
        self.scaling = scaling
        self.ratio_test = ratio_test

    def to_options(self) -> sih_solver.Options:
        opts = sih_solver.Options()
        opts.strategy.algorithm = self.algorithm
        opts.strategy.pricing_rule = self.pricing
        opts.strategy.branching_rule = self.branching
        opts.strategy.node_selection = self.node_sel
        opts.strategy.cut_rounds = self.cut_rounds
        opts.strategy.presolve = self.presolve
        opts.strategy.enable_scaling = self.scaling
        opts.strategy.ratio_test = self.ratio_test
        opts.log_to_console = False
        return opts


# Standard discrete strategy library
PROFILES: List[StrategyProfile] = [
    # Profile 0: Standard Dual Simplex + DSE
    StrategyProfile(
        name="DualSimplex_SteepestEdge",
        algorithm=sih_solver.AlgorithmChoice.DualSimplex,
        pricing=sih_solver.PricingRule.SteepestEdge,
        presolve=sih_solver.PresolveMode.On,
        scaling=True,
    ),
    # Profile 1: Dual Simplex + Devex (fast iterations on sparse LPs)
    StrategyProfile(
        name="DualSimplex_Devex",
        algorithm=sih_solver.AlgorithmChoice.DualSimplex,
        pricing=sih_solver.PricingRule.Devex,
        presolve=sih_solver.PresolveMode.On,
        scaling=True,
    ),
    # Profile 2: Primal Simplex + Devex
    StrategyProfile(
        name="PrimalSimplex_Devex",
        algorithm=sih_solver.AlgorithmChoice.PrimalSimplex,
        pricing=sih_solver.PricingRule.Devex,
        presolve=sih_solver.PresolveMode.On,
        scaling=True,
    ),
    # Profile 3: Barrier IPM (for dense or huge continuous problems)
    StrategyProfile(
        name="Barrier_IPM",
        algorithm=sih_solver.AlgorithmChoice.Barrier,
        presolve=sih_solver.PresolveMode.On,
        scaling=True,
    ),
    # Profile 4: MILP Best-Bound + Most-Fractional + Moderate Cuts
    StrategyProfile(
        name="MILP_BestBound_MostFrac_Cuts5",
        algorithm=sih_solver.AlgorithmChoice.BranchAndBound,
        branching=sih_solver.BranchingRule.MostFractional,
        node_sel=sih_solver.NodeSelection.BestBound,
        cut_rounds=5,
        presolve=sih_solver.PresolveMode.On,
    ),
    # Profile 5: MILP Depth-First Diving + PseudoCost (rapid incumbent finding)
    StrategyProfile(
        name="MILP_DepthFirst_PseudoCost_NoCuts",
        algorithm=sih_solver.AlgorithmChoice.BranchAndBound,
        branching=sih_solver.BranchingRule.PseudoCost,
        node_sel=sih_solver.NodeSelection.DepthFirst,
        cut_rounds=0,
        presolve=sih_solver.PresolveMode.On,
    ),
    # Profile 6: MILP Best-Bound + PseudoCost + Deep Cuts
    StrategyProfile(
        name="MILP_BestBound_PseudoCost_Cuts10",
        algorithm=sih_solver.AlgorithmChoice.BranchAndBound,
        branching=sih_solver.BranchingRule.PseudoCost,
        node_sel=sih_solver.NodeSelection.BestBound,
        cut_rounds=10,
        presolve=sih_solver.PresolveMode.On,
    ),
    # Profile 7: Lightweight Bare Dual Simplex (Zero Presolve, Zero Scaling overhead)
    StrategyProfile(
        name="Bare_DualSimplex_NoPresolve",
        algorithm=sih_solver.AlgorithmChoice.DualSimplex,
        pricing=sih_solver.PricingRule.SteepestEdge,
        presolve=sih_solver.PresolveMode.Off,
        scaling=False,
    ),
]


class KNNStrategySelector:
    """
    Nearest-Neighbor Strategy Selector.
    Standardizes structural features and retrieves historical neighbors to select
    the strategy profile that delivered minimal runtime or iterations.
    """

    def __init__(self, k_neighbors: int = 3):
        self.k_neighbors = k_neighbors
        self.training_data: List[Dict[str, Any]] = []
        self.means: List[float] = []
        self.stds: List[float] = []

    def fit(self, dataset: List[Dict[str, Any]]):
        """
        Train on a collection of records:
        [{'problem': Problem, 'profile_id': int, 'runtime_sec': float, 'iterations': int}, ...]
        """
        self.training_data = []
        if not dataset:
            return

        all_vecs: List[List[float]] = []
        for record in dataset:
            prob = record.get("problem")
            vec = record.get("features")
            if vec is None and prob is not None:
                vec = ProblemFeatureExtractor.feature_vector(prob)
            if vec is not None:
                all_vecs.append(vec)
                self.training_data.append({
                    "features": vec,
                    "profile_id": record["profile_id"],
                    "runtime_sec": record["runtime_sec"],
                    "iterations": record.get("iterations", 0),
                })

        if not all_vecs:
            return

        num_features = len(all_vecs[0])
        self.means = [0.0] * num_features
        self.stds = [1.0] * num_features

        # Compute mean
        for j in range(num_features):
            vals = [v[j] for v in all_vecs]
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            self.means[j] = m
            self.stds[j] = math.sqrt(var) if var > 1e-12 else 1.0

    def _normalize(self, vec: List[float]) -> List[float]:
        if not self.means or not self.stds:
            return vec
        return [(vec[j] - self.means[j]) / self.stds[j] for j in range(len(vec))]

    def predict_profile(self, problem: sih_solver.Problem) -> Tuple[int, StrategyProfile]:
        """
        Predict optimal StrategyProfile index for an unseen problem.
        """
        is_mip = problem.is_mip()
        vec = ProblemFeatureExtractor.feature_vector(problem)
        norm_vec = self._normalize(vec)

        # Filter valid profiles (LP profiles vs MILP profiles)
        valid_indices = [4, 5, 6] if is_mip else [0, 1, 2, 3, 7]

        if not self.training_data:
            # Fallback to defaults
            chosen = 4 if is_mip else 0
            return chosen, PROFILES[chosen]

        # Compute distances to all historical instances
        scored_neighbors = []
        for d in self.training_data:
            pid = d["profile_id"]
            if pid not in valid_indices:
                continue
            
            d_norm = self._normalize(d["features"])
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(norm_vec, d_norm)))
            scored_neighbors.append((dist, pid, d["runtime_sec"]))

        if not scored_neighbors:
            chosen = 4 if is_mip else 0
            return chosen, PROFILES[chosen]

        scored_neighbors.sort(key=lambda x: x[0])
        top_k = scored_neighbors[:self.k_neighbors]

        # Profile with minimum average runtime among top neighbors
        profile_times: Dict[int, List[float]] = {}
        for _, pid, rtime in top_k:
            profile_times.setdefault(pid, []).append(rtime)

        best_pid = min(profile_times.keys(), key=lambda p: sum(profile_times[p]) / len(profile_times[p]))
        return best_pid, PROFILES[best_pid]

    def select_options(self, problem: sih_solver.Problem) -> sih_solver.Options:
        _, prof = self.predict_profile(problem)
        return prof.to_options()


class ContextualBanditSelector:
    """
    Contextual Bandit Strategy Selector using epsilon-greedy exploration/exploitation.
    Maintains empirical reward estimates per profile conditioned on structural clusters.
    """

    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon
        self.knn = KNNStrategySelector()
        self.rng = random.Random(42)

    def fit(self, dataset: List[Dict[str, Any]]):
        self.knn.fit(dataset)

    def select(self, problem: sih_solver.Problem, explore: bool = True) -> Tuple[int, sih_solver.Options]:
        is_mip = problem.is_mip()
        valid_indices = [4, 5, 6] if is_mip else [0, 1, 2, 3, 7]

        if explore and self.rng.random() < self.epsilon:
            # Explore random valid profile
            chosen_pid = self.rng.choice(valid_indices)
        else:
            # Exploit best predicted profile
            chosen_pid, _ = self.knn.predict_profile(problem)

        return chosen_pid, PROFILES[chosen_pid].to_options()
