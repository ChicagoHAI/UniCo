#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# Integration pipeline for NL domain (v1.4)
# Integrates per-node and cross-node outputs from redistributed NL
# sub-corpora (v1.4 = 1/3 train, 1/2 eval of v1.1 symbolic totals).
#
# Supports mixed prompting versions: v2 for nodes <= 6, v3 for nodes >= 7.
# ==========================================================================

version="v1.4"

# --- Domain ---
domains=("nl")

# --- Variant assignment (must match the generation runner setting) ---
variant_assignment="split"

# --- Query types ---
query_types=("ate" "cate" "jate")

# --- Splits ---
split_list=(
    "train"
    "eval"
)

# --- ATE: nodes 3-10 ---
# num_examples use ORIGINAL v1.1 counts (not NL-reduced) because v1.4 NL
# output filenames preserve original per-DP idx offsets for integration
# compatibility. The integration pipeline uses these for file discovery
# (start_idx/end_idx patterns) and idx-based slicing.
# Actual NL records are 1/3 train, 1/2 eval of these counts.
ate_num_nodes=(3 4 5 6 7 8 9 10)
ate_train_examples=(120 600 960 1200 1200 960 600 360)
ate_eval_examples=(16 80 128 160 160 128 80 48)
ate_prompting_versions=("v2" "v2" "v2" "v2" "v3" "v3" "v3" "v3")

# --- CATE/JATE: nodes 4-10 ---
cate_jate_num_nodes=(4 5 6 7 8 9 10)
cate_jate_train_examples=(600 960 1200 1200 960 600 360)
cate_jate_eval_examples=(80 128 160 160 128 80 48)
cate_jate_prompting_versions=("v2" "v2" "v2" "v3" "v3" "v3" "v3")

for domain in "${domains[@]}"; do
    timestamp=$(date +%Y%m%d_%H%M%S)
for qt in "${query_types[@]}"; do
for split in "${split_list[@]}"; do

    if [[ "$qt" == "ate" ]]; then
        base_dir="./data_generation/generated_datasets/${version}/ate"
        num_nodes_args="${ate_num_nodes[*]}"
        pv_args="${ate_prompting_versions[*]}"
        if [[ "$split" == "train" ]]; then
            examples_args="${ate_train_examples[*]}"
        else
            examples_args="${ate_eval_examples[*]}"
        fi
    else
        base_dir="./data_generation/generated_datasets/${version}/cate_jate"
        num_nodes_args="${cate_jate_num_nodes[*]}"
        pv_args="${cate_jate_prompting_versions[*]}"
        if [[ "$split" == "train" ]]; then
            examples_args="${cate_jate_train_examples[*]}"
        else
            examples_args="${cate_jate_eval_examples[*]}"
        fi
    fi

    va_args=()
    if [[ -n "${variant_assignment}" ]]; then
        va_args+=(--variant_assignment "${variant_assignment}")
    fi

    echo "=== Integrating ${qt} ${domain} ${split} ==="

    python3 -m data_generation.integration_pipeline \
        --query_type "${qt}" \
        --split "${split}" \
        --base_dir "${base_dir}" \
        --run_timestamp "${timestamp}" \
        --domain "${domain}" \
        --prompting_version_list ${pv_args} \
        --num_nodes_list ${num_nodes_args} \
        --num_examples_list ${examples_args} \
        --mode both \
        "${va_args[@]}"

done
done
done

echo ""
echo "=== Integration v1.4 complete ==="
