#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONPATH="${SCRIPT_DIR}/py:${SCRIPT_DIR}/tests:${PYTHONPATH:-}"

echo "======================================================================"
echo "SIH 26119 Phase 0 Verification & Test Suite"
echo "======================================================================"

echo ""
echo "[Step 1/3] Enforcing Hard Rules 1-3 (No External Solvers / Libraries)..."
python3 tools/check_no_external_solvers.py

echo ""
echo "[Step 2/3] Executing C++ Unit Tests via CTest..."
ctest --test-dir build --output-on-failure

echo ""
echo "[Step 3/3] Executing Python Test Suite (Checker & HiGHS Oracle Harness)..."
pytest -v tests/

echo ""
echo "======================================================================"
echo "ALL PHASE 0 GATES PASSED CLEANLY"
echo "======================================================================"
