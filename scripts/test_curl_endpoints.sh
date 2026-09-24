#!/usr/bin/env bash
set -e

echo "=== 1. Starting FastAPI Server in Background ==="
cd /home/notanracistfr/projects/sih26119-solver
source .venv/bin/activate
export PYTHONPATH="/home/notanracistfr/projects/sih26119-solver/py:/home/notanracistfr/projects/sih26119-solver"

python3 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 > /tmp/uvicorn.log 2>&1 &
UVICORN_PID=$!

trap "kill $UVICORN_PID 2>/dev/null || true" EXIT

# Wait for server to start
sleep 2

echo -e "\n=== 2. GET / (Health Check) ==="
curl -s -X GET http://127.0.0.1:8000/ | jq . || curl -s -X GET http://127.0.0.1:8000/

echo -e "\n=== 3. POST /api/v1/solve (Submit LP Job Async) ==="
SUBMIT_RESP=$(curl -s -X POST "http://127.0.0.1:8000/api/v1/solve?sync=false" \
  -H "Content-Type: application/json" \
  -d '{
    "mps_data": "NAME          SAMPLE\nOBJSENSE\n  MIN\nROWS\n N  OBJ\n L  ROW1\n G  ROW2\nCOLUMNS\n    X1        OBJ             -1.00   ROW1             2.00\n    X1        ROW2             1.00\n    X2        OBJ             -2.00   ROW1             1.00\n    X2        ROW2             1.00\nRHS\n    RHS1      ROW1             4.00   ROW2             1.00\nBOUNDS\n UP BND1      X1               5.00\n UP BND1      X2               5.00\nENDATA\n",
    "algorithm": "DualSimplex"
  }')
echo "$SUBMIT_RESP" | jq . || echo "$SUBMIT_RESP"
JOB_ID=$(echo "$SUBMIT_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

# Poll for completion
sleep 0.5
echo -e "\n=== 4. GET /api/v1/jobs/$JOB_ID (Poll Status) ==="
curl -s -X GET "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID" | jq . || curl -s -X GET "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID"

echo -e "\n=== 5. GET /api/v1/jobs/$JOB_ID/solution (Fetch Solution) ==="
curl -s -X GET "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID/solution" | jq . || curl -s -X GET "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID/solution"

echo -e "\n=== 6. GET /api/v1/telemetry (Execution Telemetry) ==="
curl -s -X GET "http://127.0.0.1:8000/api/v1/telemetry?limit=3" | jq . || curl -s -X GET "http://127.0.0.1:8000/api/v1/telemetry?limit=3"

echo -e "\n=== 7. GET /api/v1/benchmarks (Benchmark SGM & Profiles) ==="
curl -s -X GET "http://127.0.0.1:8000/api/v1/benchmarks" | python3 -c "import sys, json; d=json.load(sys.stdin); print({k: {'SGM_ms': d[k]['sgm_our_sec']*1000, 'count': d[k]['count']} for k in ['LP', 'MILP', 'QP']})"

echo -e "\n=== 8. POST /api/v1/refinery/plan (Refinery Planning Dashboard) ==="
curl -s -X POST "http://127.0.0.1:8000/api/v1/refinery/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "crude_prices": {"ArabLight": 76.5, "Brent": 81.2},
    "unit_capacities": {"ADU": 100000.0, "FCC": 35000.0},
    "is_milp": false
  }' | jq . || curl -s -X POST "http://127.0.0.1:8000/api/v1/refinery/plan" -H "Content-Type: application/json" -d '{"is_milp": false}'

echo -e "\n=== 9. Completed Live REST API Verification ==="
