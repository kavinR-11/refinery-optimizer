"""
Reference Oracle Harness Package.
Permitted to import highspy and scipy.optimize per Hard Rule 3.
"""

from .highs_oracle import HighsOracle

__all__ = ["HighsOracle"]
