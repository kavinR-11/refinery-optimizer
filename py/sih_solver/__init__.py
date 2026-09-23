"""
SIH 26119 Indigenous Optimization Solver Python Interface.
"""

from typing import Optional, List, Dict, Any

try:
    from sih_solver._core import (
        SparseMatrix,
        Triplet,
        Problem,
        Solution,
        Options,
        StrategyConfig,
        ObjectiveSense,
        VariableType,
        SolutionStatus,
        BasisStatus,
        AlgorithmChoice,
        PricingRule,
        BranchingRule,
        NodeSelection,
        PresolveMode,
        read_mps,
        write_mps,
        check_solution,
        CheckResult,
        pin_thread_to_core,
        pin_thread_to_cores,
        get_current_core,
        get_num_procs,
        log_solve_telemetry,
        serialize_solve_telemetry,
        get_resident_memory_mb,
    )
except ImportError:
    # Extension module not yet built in current path
    pass

__version__ = "0.1.0"
__all__ = [
    "SparseMatrix",
    "Triplet",
    "Problem",
    "Solution",
    "Options",
    "StrategyConfig",
    "ObjectiveSense",
    "VariableType",
    "SolutionStatus",
    "BasisStatus",
    "AlgorithmChoice",
    "PricingRule",
    "BranchingRule",
    "NodeSelection",
    "PresolveMode",
    "read_mps",
    "write_mps",
    "check_solution",
    "CheckResult",
    "pin_thread_to_core",
    "pin_thread_to_cores",
    "get_current_core",
    "get_num_procs",
    "log_solve_telemetry",
    "serialize_solve_telemetry",
    "get_resident_memory_mb",
]
