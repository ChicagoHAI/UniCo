#!/usr/bin/env bash
set -euo pipefail

# Extract NL-domain triplet sub-corpora from v2.x triplet corpus (v2.2).
# Usage: redistribute_nl_subcorpus_v2.2.sh <manifest_timestamp>

manifest_timestamp="${1:?Usage: $0 <manifest_timestamp>}"

log_dir="./data_generation/generated_datasets/v2.2/.logs"
mkdir -p "${log_dir}"

echo "=== v2.2 NL Sub-Corpus Extraction ==="
echo "Manifest timestamp: ${manifest_timestamp}"

python3 -m data_generation.redistribute_nl_subcorpus_v2 \
    --manifest_timestamp "${manifest_timestamp}" \
    2>&1 | tee "${log_dir}/${manifest_timestamp}-nl-subcorpus.log"

echo ""
echo "=== NL sub-corpus extraction complete ==="
