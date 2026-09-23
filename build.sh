#!/usr/bin/env bash
set -euo pipefail

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building SIH 26119 Indigenous Solver ==="

# Activate virtual environment if present
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Configure with CMake and Ninja
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(which python3)"

# Build core library, python module, and unit tests
cmake --build build

echo "=== Build Completed Successfully ==="
