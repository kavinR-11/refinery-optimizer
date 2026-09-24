#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cmake -B build -G Ninja \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(which python3)"

cmake --build build
