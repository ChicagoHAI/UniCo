#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# Cross-Domain Dataset Redistribution Pipeline (v1.4) — Orchestration
#
# Runs all redistribution steps via their dedicated runners:
#   1. Manifest generation (redistribute_manifest_v1.4.sh)
#   2. Symbolic/code materialization (redistribute_materialize_v1.4.sh)
#   3. NL sub-corpus extraction (redistribute_nl_subcorpus_v1.4.sh)
#
# Each step can also be run independently with its own runner script.
# ==========================================================================

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== v1.4 Redistribution Pipeline ==="
echo ""

# --- Step 1: Generate manifests ---
echo "========== Step 1: Manifest Generation =========="
bash "${script_dir}/redistribute_manifest_v1.4.sh"

# Extract the manifest timestamp from the generated files
manifest_timestamp=$(ls -1t ./data_generation/generated_datasets/v1.4/manifests/*-ate-train-manifest.jsonl 2>/dev/null \
    | head -1 | xargs basename | cut -d'-' -f1)
echo "Manifest timestamp: ${manifest_timestamp}"
echo ""

# --- Step 2: Materialize symbolic and code datasets ---
echo "========== Step 2: Symbolic/Code Materialization =========="
bash "${script_dir}/redistribute_materialize_v1.4.sh" "${manifest_timestamp}"
echo ""

# --- Step 3: Extract NL sub-corpora ---
echo "========== Step 3: NL Sub-Corpus Extraction =========="
bash "${script_dir}/redistribute_nl_subcorpus_v1.4.sh" "${manifest_timestamp}"
echo ""

echo "=== v1.4 Redistribution Pipeline Complete ==="
echo "Outputs: ./data_generation/generated_datasets/v1.4/"
