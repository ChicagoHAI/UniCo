#!/usr/bin/env bash
set -euo pipefail

# Generate redistribution manifests for v2.2 cross-domain partitioning.

timestamp=$(date +%Y%m%d_%H%M%S)
base_seed=42

log_dir="./data_generation/generated_datasets/v2.2/.logs"
mkdir -p "${log_dir}"

echo "=== v2.2 Manifest Generation ==="
echo "Timestamp: ${timestamp}"
echo "Base seed: ${base_seed}"

python3 -m data_generation.redistribute_manifest_v2 \
    --run_timestamp "${timestamp}" \
    --base_seed "${base_seed}" \
    2>&1 | tee "${log_dir}/${timestamp}-manifest.log"

echo ""
echo "=== Manifest generation complete ==="
echo "Manifest timestamp for downstream steps: ${timestamp}"
