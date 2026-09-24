"""
Backend API Integration Test Suite for Indigenous Optimization Solver.
Validates all REST endpoints using FastAPI TestClient:
- Health check
- Job submission (MPS)
- Job polling and solution retrieval
- Telemetry retrieval
- Benchmark retrieval
- Refinery planning scenario solve
- Refinery warm-start re-optimization
Strictly compliant with Hard Rules 1-3.
"""

import sys
import os
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Add py to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "py"))

from api.app import app

client = TestClient(app)

SAMPLE_MPS = """NAME          SAMPLE
OBJSENSE
  MIN
ROWS
 N  OBJ
 L  ROW1
 G  ROW2
COLUMNS
    X1        OBJ             -1.00   ROW1             2.00
    X1        ROW2             1.00
    X2        OBJ             -2.00   ROW1             1.00
    X2        ROW2             1.00
RHS
    RHS1      ROW1             4.00   ROW2             1.00
BOUNDS
 UP BND1      X1               5.00
 UP BND1      X2               5.00
ENDATA
"""


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "Indigenous Optimization" in data["service"]
    print("✓ Health check endpoint PASSED")


def test_submit_poll_solution_lifecycle():
    # 1. Submit solve job
    payload = {
        "mps_data": SAMPLE_MPS,
        "algorithm": "DualSimplex",
        "enable_gpu": False
    }
    submit_resp = client.post("/api/v1/solve?sync=true", json=payload)
    assert submit_resp.status_code == 202
    job_data = submit_resp.json()
    job_id = job_data["job_id"]
    assert job_id.startswith("job_")
    assert job_data["completed"] is True
    print(f"✓ Job submission: job_id={job_id}, status={job_data['status']}, error={job_data.get('error_message')}")

    # 2. Poll job status
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    print(f"Status data: {status_data}")

    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["completed"] is True
    print(f"✓ Job polling PASSED: status={status_data['status']}, elapsed={status_data['elapsed_ms']} ms")

    # 3. Fetch solution
    sol_resp = client.get(f"/api/v1/jobs/{job_id}/solution")
    assert sol_resp.status_code == 200
    sol_data = sol_resp.json()
    assert sol_data["is_optimal"] is True
    assert len(sol_data["variables"]) == 2
    assert len(sol_data["constraints"]) == 2
    print(f"✓ Fetch solution PASSED: obj={sol_data['primal_objective']}, vars={len(sol_data['variables'])}, iters={sol_data['simplex_iterations']}")


def test_telemetry_endpoint():
    response = client.get("/api/v1/telemetry?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
    assert "count" in data
    print(f"✓ Telemetry endpoint PASSED: returned {data['count']} records")


def test_benchmarks_endpoint():
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "LP" in data
    assert "MILP" in data
    assert "QP" in data
    print(f"✓ Benchmarks endpoint PASSED: LP SGM = {data['LP']['sgm_our_sec']*1000:.2f} ms")


def test_refinery_plan_endpoint():
    payload = {
        "crude_prices": {"ArabLight": 75.0, "Brent": 80.0},
        "unit_capacities": {"ADU": 100000.0, "VDU": 60000.0},
        "is_milp": False
    }
    response = client.post("/api/v1/refinery/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_optimal"] is True
    assert data["profit_daily_usd"] > 0.0
    assert "ArabLight" in data["crude_intake_bpd"]
    assert "ADU" in data["unit_utilization_pct"]
    assert len(data["bottlenecks"]) > 0
    print(f"✓ Refinery plan endpoint PASSED: Daily Profit = ${data['profit_daily_usd']:,.2f}, Bottlenecks = {len(data['bottlenecks'])}")


def test_refinery_reoptimize_endpoint():
    payload = {
        "perturbation_type": "crude_price_shock",
        "perturbation_data": {"ArabLight": 92.0}
    }
    response = client.post("/api/v1/refinery/reoptimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reoptimization_benchmarks" in data
    assert len(data["reoptimization_benchmarks"]) == 5
    print(f"✓ Refinery reoptimize endpoint PASSED: evaluated {len(data['reoptimization_benchmarks'])} perturbations")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING BACKEND API INTEGRATION TEST BATTERY")
    print("=" * 70)
    test_health_check()
    test_submit_poll_solution_lifecycle()
    test_telemetry_endpoint()
    test_benchmarks_endpoint()
    test_refinery_plan_endpoint()
    test_refinery_reoptimize_endpoint()
    print("=" * 70)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
