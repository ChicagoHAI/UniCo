#!/usr/bin/env bash
set -euo pipefail

# Generate redistribution manifests for v1.4 cross-domain partitioning.

timestamp=$(date +%Y%m%d_%H%M%S)
base_seed=42
query_types=(ate cate jate)

log_dir="./data_generation/generated_datasets/v1.4/.logs"
mkdir -p "${log_dir}"

echo "=== v1.4 Manifest Generation ==="
echo "Timestamp: ${timestamp}"
echo "Base seed: ${base_seed}"

# The released v1.4 eval split is the half that the default selection leaves out.
python3 -m data_generation.redistribute_manifest \
    --run_timestamp "${timestamp}" \
    --base_seed "${base_seed}" \
    --query_types "${query_types[@]}" \
    --invert_eval_selection \
    2>&1 | tee "${log_dir}/${timestamp}-manifest.log"

echo ""
echo "=== Manifest generation complete ==="
echo "Manifest timestamp for downstream steps: ${timestamp}"
