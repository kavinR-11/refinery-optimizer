"""
FastAPI Backend Application for Indigenous Optimization Solver (SIH 26119).
Exposes REST endpoints for:
- Problem submission (MPS format) and asynchronous solve job management
- Job status polling and comprehensive solution retrieval (shadow prices, sensitivity)
- Operational telemetry and solver benchmarking statistics
- Industrial refinery planning and warm-start re-optimization
Strictly compliant with Hard Rules 1-3 (Zero external solver / linprog libraries).
"""

import os
import sys
import uuid
import time
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

# Insert paths to import py modules and sih_solver
py_path = Path(__file__).resolve().parent.parent
root_path = py_path.parent
if str(py_path) not in sys.path:
    sys.path.insert(0, str(py_path))
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import sih_solver
from refinery.model_builder import RefineryModelBuilder
from refinery.explainability import RefineryExplainer
from refinery.reoptimization import RefineryReoptimizer
from api.schemas import (
    SolveRequest,
    JobStatusResponse,
    SolutionResponse,
    VariableValue,
    ConstraintActivity,
    RefineryPlanRequest,
    RefineryPlanResponse,
    RefineryReoptRequest,
)

app = FastAPI(
    title="Indigenous Optimization Engine API",
    version="1.0.0",
    description="High-performance indigenous LP/MILP/QP solver REST backend for plant DCS and web dashboards.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry: job_id -> dict
JOBS: Dict[str, Dict[str, Any]] = {}


def _run_solve_job(job_id: str, request: SolveRequest):
    t_start = time.perf_counter()
    tmp_path = None
    try:
        # Write temporary MPS file
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False, mode="w") as f:
            f.write(request.mps_data)
            tmp_path = f.name

        prob = sih_solver.read_mps(tmp_path)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        options = sih_solver.Options()
        algo_str = (request.algorithm or "DualSimplex").lower()
        if "primal" in algo_str:
            options.strategy.algorithm = sih_solver.AlgorithmChoice.PrimalSimplex
        elif "barrier" in algo_str or "ipm" in algo_str:
            options.strategy.algorithm = sih_solver.AlgorithmChoice.Barrier
        elif "milp" in algo_str or "branch" in algo_str:
            options.strategy.algorithm = sih_solver.AlgorithmChoice.BranchAndBound
        else:
            options.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex

        if request.time_limit_sec is not None:
            options.time_limit_sec = request.time_limit_sec

        sol = sih_solver.solve(prob, options)
        elapsed_sec = time.perf_counter() - t_start

        # Extract variables
        vars_list = []
        for j in range(prob.num_cols()):
            val = sol.x[j] if j < len(sol.x) else 0.0
            rc = sol.reduced_costs[j] if j < len(sol.reduced_costs) else 0.0
            stat_name = str(sol.col_basis[j]).replace("BasisStatus.", "") if j < len(sol.col_basis) else "Nonbasic"
            vars_list.append(VariableValue(
                name=f"C{j:04d}",
                value=round(val, 8),
                reduced_cost=round(rc, 8),
                status=stat_name
            ))

        # Extract constraints
        cons_list = []
        for i in range(prob.num_rows()):
            act = sol.slack[i] if i < len(sol.slack) else 0.0
            dual = sol.row_duals[i] if i < len(sol.row_duals) else 0.0
            stat_name = str(sol.row_basis[i]).replace("BasisStatus.", "") if i < len(sol.row_basis) else "Basic"
            cons_list.append(ConstraintActivity(
                name=f"R{i:04d}",
                activity=round(act, 8),
                dual_value=round(dual, 8),
                status=stat_name
            ))

        diag = {
            "rhs_down": [round(v, 6) for v in list(sol.rhs_down)[:20]] if hasattr(sol, "rhs_down") else [],
            "rhs_up": [round(v, 6) for v in list(sol.rhs_up)[:20]] if hasattr(sol, "rhs_up") else [],
            "obj_down": [round(v, 6) for v in list(sol.obj_down)[:20]] if hasattr(sol, "obj_down") else [],
            "obj_up": [round(v, 6) for v in list(sol.obj_up)[:20]] if hasattr(sol, "obj_up") else [],
        }

        sol_resp = SolutionResponse(
            job_id=job_id,
            status=str(sol.status).replace("SolutionStatus.", ""),
            is_optimal=sol.is_optimal(),
            primal_objective=round(sol.primal_objective, 8),
            dual_bound=round(sol.dual_bound, 8),
            relative_gap=round(sol.mip_gap, 8),
            solve_time_sec=round(elapsed_sec, 6),
            simplex_iterations=sol.simplex_iterations,
            nodes_explored=sol.nodes_explored,
            variables=vars_list,
            constraints=cons_list,
            diagnostics=diag
        )

        JOBS[job_id]["status"] = sol_resp.status
        JOBS[job_id]["completed"] = True
        JOBS[job_id]["elapsed_ms"] = round(elapsed_sec * 1000.0, 3)
        JOBS[job_id]["solution"] = sol_resp

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        JOBS[job_id]["status"] = "ERROR"
        JOBS[job_id]["completed"] = True
        JOBS[job_id]["error_message"] = str(e)


@app.get("/")
def health_check():
    return {
        "status": "ONLINE",
        "service": "Indigenous Optimization Engine API",
        "version": "1.0.0",
        "engine": "C++17 Indigenous Core (No external solvers)",
    }


@app.post("/api/v1/solve", response_model=JobStatusResponse, status_code=202)
def submit_solve(request: SolveRequest, background_tasks: BackgroundTasks, sync: bool = Query(False)):
    """
    Submit an optimization problem in MPS format.
    Runs asynchronously in background by default, or synchronously if sync=True.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "RUNNING",
        "elapsed_ms": 0.0,
        "completed": False,
        "error_message": None,
        "solution": None,
    }

    if sync:
        _run_solve_job(job_id, request)
        rec = JOBS[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=rec["status"],
            elapsed_ms=rec["elapsed_ms"],
            completed=rec["completed"],
            error_message=rec["error_message"]
        )
    else:
        background_tasks.add_task(_run_solve_job, job_id, request)
        return JobStatusResponse(
            job_id=job_id,
            status="RUNNING",
            elapsed_ms=0.0,
            completed=False,
            error_message=None
        )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Poll the execution status of a submitted optimization job.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    rec = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=rec["status"],
        elapsed_ms=rec["elapsed_ms"],
        completed=rec["completed"],
        error_message=rec["error_message"]
    )


@app.get("/api/v1/jobs/{job_id}/solution", response_model=SolutionResponse)
def get_job_solution(job_id: str):
    """
    Fetch the complete numerical solution, basis states, and sensitivity intervals.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    rec = JOBS[job_id]
    if not rec["completed"]:
        raise HTTPException(status_code=202, detail=f"Job '{job_id}' is still in progress")
    if rec["error_message"]:
        raise HTTPException(status_code=500, detail=f"Job '{job_id}' encountered error: {rec['error_message']}")
    return rec["solution"]


@app.get("/api/v1/telemetry")
def get_telemetry(limit: int = Query(50, ge=1, le=500)):
    """
    Fetch historical execution telemetry records across previous solver runs.
    """
    possible_paths = [
        root_path / "telemetry.csv",
        py_path / "telemetry.csv",
    ]
    records = []
    for p in possible_paths:
        if p.exists():
            import csv
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            break
    return {
        "count": len(records[-limit:]),
        "total_recorded": len(records),
        "records": records[-limit:]
    }


@app.get("/api/v1/benchmarks")
def get_benchmarks():
    """
    Retrieve benchmark summary metrics across Netlib (LP), MIPLIB (MILP), and Maros-Mészáros (QP).
    """
    json_path = root_path / "bench" / "benchmark_results.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found. Run bench/run_full_benchmark_suite.py first.")
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/api/v1/refinery/plan", response_model=RefineryPlanResponse)
def solve_refinery_plan(request: RefineryPlanRequest):
    """
    Refinery planning model endpoint tailored for the Refinery Planning dashboard screen.
    Returns optimal profit, crude diet, unit utilizations, product yields, and economic bottlenecks.
    """
    t0 = time.perf_counter()
    builder = RefineryModelBuilder(is_milp=request.is_milp or False)

    # Apply request overrides
    if request.crude_prices:
        for c, p in request.crude_prices.items():
            if c in builder.crudes:
                builder.crudes[c]["price"] = p
    if request.product_demands:
        for p, d in request.product_demands.items():
            if p in builder.products:
                builder.products[p]["min_demand"] = d
    if request.unit_capacities:
        for u, cap in request.unit_capacities.items():
            if u in builder.units:
                builder.units[u]["capacity"] = cap

    prob, meta = builder.build_problem()
    opts = sih_solver.Options()
    if request.is_milp:
        opts.strategy.algorithm = sih_solver.AlgorithmChoice.BranchAndBound
    else:
        opts.strategy.algorithm = sih_solver.AlgorithmChoice.DualSimplex
    opts.log_to_console = False

    sol = sih_solver.solve(prob, opts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    var_map = meta.get("var_map", {})

    # Crude intakes
    crude_intake = {}
    for c in builder.crudes:
        idx = var_map.get(f"crude_{c}")
        crude_intake[c] = round(sol.x[idx], 2) if idx is not None and idx < len(sol.x) else 0.0

    # Unit utilization
    unit_util = {}
    for u in builder.units:
        idx = var_map.get(f"unit_{u}")
        cap = builder.units[u]["capacity"]
        tp = sol.x[idx] if idx is not None and idx < len(sol.x) else 0.0
        pct = (tp / cap * 100.0) if cap > 0 else 0.0
        unit_util[u] = round(pct, 2)

    # Product yields
    prod_yields = {}
    for p in builder.products:
        idx = var_map.get(f"prod_{p}")
        prod_yields[p] = round(sol.x[idx], 2) if idx is not None and idx < len(sol.x) else 0.0

    # Bottlenecks via RefineryExplainer
    bottlenecks = []
    if sol.is_optimal():
        explainer = RefineryExplainer(prob, sol, meta)
        shadow_exps = explainer.explain_shadow_prices()
        for exp in shadow_exps:
            if exp.get("is_binding") and exp.get("shadow_price", 0.0) > 0.01:
                bottlenecks.append({
                    "constraint": exp["name"],
                    "shadow_price_usd": round(exp["shadow_price"], 2),
                    "interpretation": exp.get("plain_desc", "")
                })

    return RefineryPlanResponse(
        status=str(sol.status).replace("SolutionStatus.", ""),
        is_optimal=sol.is_optimal(),
        profit_daily_usd=round(sol.primal_objective, 2),
        crude_intake_bpd=crude_intake,
        unit_utilization_pct=unit_util,
        product_yields_bpd=prod_yields,
        bottlenecks=bottlenecks,
        solve_time_ms=round(elapsed_ms, 2),
        iterations=sol.simplex_iterations
    )


@app.post("/api/v1/refinery/reoptimize")
def reoptimize_refinery(request: RefineryReoptRequest):
    """
    Refinery warm-start re-optimization endpoint.
    Benchmarked across operational perturbations showing cold vs warm iteration speedups.
    """
    reoptimizer = RefineryReoptimizer()
    results = reoptimizer.run_benchmark()
    return {
        "perturbation_requested": request.perturbation_type,
        "reoptimization_benchmarks": results
    }
