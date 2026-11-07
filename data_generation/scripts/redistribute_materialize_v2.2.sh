#!/usr/bin/env bash
set -euo pipefail

# Materialize redistributed datasets for symbolic and code domains (v2.2).
# Usage: redistribute_materialize_v2.2.sh <manifest_timestamp>

manifest_timestamp="${1:?Usage: $0 <manifest_timestamp>}"
timestamp=$(date +%Y%m%d_%H%M%S)

log_dir="./data_generation/generated_datasets/v2.2/.logs"
mkdir -p "${log_dir}"

echo "=== v2.2 Materialization ==="
echo "Manifest timestamp: ${manifest_timestamp}"
echo "Run timestamp: ${timestamp}"

python3 -m data_generation.redistribute_materialize_v2 \
    --run_timestamp "${timestamp}" \
    --manifest_timestamp "${manifest_timestamp}" \
    2>&1 | tee "${log_dir}/${timestamp}-materialize.log"

echo ""
echo "=== Materialization complete ==="
