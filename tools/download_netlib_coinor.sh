#!/usr/bin/env bash
set -e
cd /home/notanracistfr/projects/sih26119-solver

INSTANCES=("afiro" "adlittle" "beaconfd" "blend" "share2b" "sc50a" "sc50b" "lotfi" "stocfor1")

for name in "${INSTANCES[@]}"; do
    echo "Downloading ${name}..."
    curl -sL "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/${name}.mps.gz" | gzip -d > "data/netlib/${name}.mps"
    lines=$(wc -l < "data/netlib/${name}.mps")
    echo "  ${name}.mps: ${lines} lines"
done
echo "All Netlib instances downloaded and uncompressed successfully."
