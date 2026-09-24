"""
Pydantic Schemas for Solver REST API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    mps_data: str = Field(..., description="Raw MPS format string representing the optimization problem")
    algorithm: Optional[str] = Field("DualSimplex", description="DualSimplex, PrimalSimplex, Barrier, BranchAndBound")
    enable_gpu: Optional[bool] = Field(False, description="Run with GPU PDHG acceleration if available")
    time_limit_sec: Optional[float] = Field(None, description="Max solve time in seconds")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    elapsed_ms: float
    completed: bool
    error_message: Optional[str] = None


class VariableValue(BaseModel):
    name: str
    value: float
    reduced_cost: float = 0.0
    status: Optional[str] = None


class ConstraintActivity(BaseModel):
    name: str
    activity: float
    dual_value: float = 0.0
    status: Optional[str] = None


class SolutionResponse(BaseModel):
    job_id: str
    status: str
    is_optimal: bool
    primal_objective: float
    dual_bound: float
    relative_gap: float
    solve_time_sec: float
    simplex_iterations: int
    nodes_explored: int
    variables: List[VariableValue]
    constraints: List[ConstraintActivity]
    diagnostics: Optional[Dict[str, Any]] = None


class RefineryPlanRequest(BaseModel):
    crude_prices: Optional[Dict[str, float]] = Field(None, description="Price per barrel by crude type ($/bbl)")
    product_demands: Optional[Dict[str, float]] = Field(None, description="Minimum product demand targets (bpd)")
    unit_capacities: Optional[Dict[str, float]] = Field(None, description="Maximum throughput by unit (bpd)")
    is_milp: Optional[bool] = Field(False, description="Include binary on/off decisions for units")


class RefineryPlanResponse(BaseModel):
    status: str
    is_optimal: bool
    profit_daily_usd: float
    crude_intake_bpd: Dict[str, float]
    unit_utilization_pct: Dict[str, float]
    product_yields_bpd: Dict[str, float]
    bottlenecks: List[Dict[str, Any]]
    solve_time_ms: float
    iterations: int


class RefineryReoptRequest(BaseModel):
    perturbation_type: str = Field(..., description="price_shock, demand_surge, unit_outage")
    perturbation_data: Dict[str, Any] = Field(..., description="Specific parameter overrides")
