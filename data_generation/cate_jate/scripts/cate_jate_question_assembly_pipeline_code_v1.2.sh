#!/usr/bin/env bash
set -euo pipefail

# --- Distribution: parallel arrays (#nodes <-> per_combo per split) ---
# CATE/JATE: nodes 4-10 only. per_combo = num_examples / 8.
num_nodes_list=(
    4
    5
    6
    7
    8
    9
    10
)
train_per_combo_list=(
    75     # 600/8
    120    # 960/8
    150    # 1200/8
    150    # 1200/8
    120    # 960/8
    75     # 600/8
    45     # 360/8
)
eval_per_combo_list=(
    10     # 80/8
    16     # 128/8
    20     # 160/8
    20     # 160/8
    16     # 128/8
    10     # 80/8
    6      # 48/8
)

dp_list=(1 2 3 4)
query_type_list=(
    "cate"
    "jate"
)
train_seed=0
eval_seed=1000000
split_list=(
    "train"
    "eval"
)
version="v1.2"
triplet_version="v1.1"
output_base="./data_generation/generated_datasets/${version}/cate_jate"
triplet_base="./data_generation/generated_datasets/${triplet_version}/cate_jate"

# --- Nested loops ---
for query_type in "${query_type_list[@]}"; do
for split in "${split_list[@]}"; do
for i in "${!num_nodes_list[@]}"; do
  num_nodes="${num_nodes_list[$i]}"
  if [[ "$split" == "train" ]]; then
    per_combo="${train_per_combo_list[$i]}"
    seed=$train_seed
  else
    per_combo="${eval_per_combo_list[$i]}"
    seed=$eval_seed
  fi

  output_dir="${output_base}/nodes${num_nodes}"
  block_size=$((per_combo * 2))

  for dp_idx in 0 1 2 3; do
    dp="${dp_list[$dp_idx]}"
    start_idx=$((dp_idx * block_size))
    end_idx=$(((dp_idx + 1) * block_size))

    # Discover v1.1 corpus file for this (query_type, nodes, dp, seed)
    corpus_path=$(find "${triplet_base}/nodes${num_nodes}/triplets/jsonls" \
        -name "*-${query_type}-*-dp${dp}-seed${seed}*.jsonl" -print -quit)

    if [[ -z "$corpus_path" ]]; then
        echo "ERROR: No corpus found for query_type=${query_type}, nodes=${num_nodes}, dp=${dp}, seed=${seed}"
        exit 1
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    corpus_basename=$(basename "${corpus_path}" .jsonl)
    filename="${timestamp}-start${start_idx}-end${end_idx}-from-${corpus_basename}"
    log_file="${output_dir}/.logs/${filename}.log"

    # Create log directory (variant directories are created by Python)
    mkdir -p "${output_dir}/.logs"

    echo "=== Run: ${corpus_basename} ==="
    echo "Source: ${corpus_path}"
    echo "Log: ${log_file}"

    python3 -m data_generation.cate_jate.cate_jate_question_assembly_pipeline_code \
        --triplet_corpus_path "${corpus_path}" \
        --start_idx "${start_idx}" \
        --end_idx "${end_idx}" \
        --output_dir "${output_dir}" \
        --run_timestamp "${timestamp}" \
    2>&1 | tee -a "${log_file}"

  done
done
done
done
