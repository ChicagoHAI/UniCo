#!/usr/bin/env bash
set -euo pipefail

# Integrate v2.2 NL train/eval outputs for the 10 association/counterfactual query types.

version="v2.2"
manifest_timestamp="${1:?Usage: $0 <manifest_timestamp>}"
manifest_dir="./data_generation/generated_datasets/${version}/manifests"
variant_assignment="manifest"

query_types=(
    "conditional_prob"
    "joint_prob"
    "marginal_prob"
    "observed_difference"
    "att"
    "cf"
    "nie"
    "nde"
    "pn"
    "ps"
)

split_list=(
    "train"
    "eval"
)

num_nodes_args="3 4 5 6 7 8 9 10"
prompting_version_args="v2 v2 v2 v2 v3 v3 v3 v3"

timestamp=$(date +%Y%m%d_%H%M%S)

for qt in "${query_types[@]}"; do
for split in "${split_list[@]}"; do
    base_dir="./data_generation/generated_datasets/${version}/${qt}"

    echo ""
    echo "=== Integrating ${qt} ${split} NL ==="

    python3 -m data_generation.integration_pipeline_v2 \
        --query_type "${qt}" \
        --split "${split}" \
        --base_dir "${base_dir}" \
        --manifest_dir "${manifest_dir}" \
        --manifest_timestamp "${manifest_timestamp}" \
        --run_timestamp "${timestamp}" \
        --mode both \
        --num_nodes_list ${num_nodes_args} \
        --prompting_version_list ${prompting_version_args} \
        --variant_assignment "${variant_assignment}"

done
done

echo ""
echo "=== v2.2 NL integration complete ==="
