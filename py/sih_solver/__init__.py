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
        RatioTest,
        SimplexSolver,
        SensitivityReport,
        IpmSolver,
        Crossover,
        BranchAndBoundSolver,
        GpuSolver,
        GpuPdhgConfig,
        GpuMemoryInfo,
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
    BranchAndBound = BranchAndBoundSolver
except ImportError as e:
    # Re-raise so any import failure is visible
    raise e

def solve_milp(problem: "Problem", options: Optional["Options"] = None) -> "Solution":
    """Solve an MILP problem using the indigenous branch-and-bound engine."""
    if options is None:
        options = Options()
    return BranchAndBoundSolver.solve(problem, options)

def solve_ipm(problem: "Problem", options: Optional["Options"] = None) -> "Solution":
    """Solve an LP or convex QP using the indigenous primal-dual interior point method."""
    if options is None:
        options = Options()
    return IpmSolver.solve(problem, options)

def solve(problem: "Problem", options: Optional["Options"] = None) -> "Solution":
    """Solve an optimization problem using the appropriate indigenous engine."""
    if options is None:
        options = Options()
    if problem.is_mip():
        return BranchAndBound.solve(problem, options)
    if options.strategy.algorithm == AlgorithmChoice.Barrier or problem.is_qp():
        return IpmSolver.solve(problem, options)
    return SimplexSolver.solve(problem, options)

def solve_from_basis(problem: "Problem",
                     col_basis: List["BasisStatus"],
                     row_basis: List["BasisStatus"],
                     options: Optional["Options"] = None) -> "Solution":
    """Warm-start an optimization solve from an existing basis configuration."""
    if options is None:
        options = Options()
    return SimplexSolver.solve_from_basis(problem, col_basis, row_basis, options)

def solve_gpu_pdhg(problem: "Problem", config: Optional["GpuPdhgConfig"] = None) -> "Solution":
    """Solve an LP using the GPU-accelerated first-order PDHG engine."""
    if config is None:
        config = GpuPdhgConfig()
    return GpuSolver.solve_pdhg(problem, config)

def solve_gpu_hybrid(problem: "Problem", config: Optional["GpuPdhgConfig"] = None) -> "Solution":
    """Solve an LP using hybrid GPU PDHG + CPU Simplex crossover polish."""
    if config is None:
        config = GpuPdhgConfig()
    return GpuSolver.solve_hybrid(problem, config)

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
    "RatioTest",
    "SimplexSolver",
    "SensitivityReport",
    "IpmSolver",
    "Crossover",
    "solve",
    "solve_ipm",
    "solve_from_basis",
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
    "GpuSolver",
    "GpuPdhgConfig",
    "GpuMemoryInfo",
    "solve_gpu_pdhg",
    "solve_gpu_hybrid",
]
