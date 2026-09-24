"""
Adaptive Strategy Learning Package (Phase 5).
Includes zero-solve structural feature extraction, Nearest-Neighbor strategy selector,
contextual bandit selector, and in-solve monitoring.
"""

from learning.feature_extractor import ProblemFeatureExtractor
from learning.strategy_selector import (
    StrategyProfile,
    PROFILES,
    KNNStrategySelector,
    ContextualBanditSelector,
)

__all__ = [
    "ProblemFeatureExtractor",
    "StrategyProfile",
    "PROFILES",
    "KNNStrategySelector",
    "ContextualBanditSelector",
]
