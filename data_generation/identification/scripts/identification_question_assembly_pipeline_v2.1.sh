#!/usr/bin/env bash
set -euo pipefail

num_nodes_list=(
    4
    5
    6
    7
    8
    9
    10
)
train_num_examples_list=(
    153
    246
    306
    306
    246
    150
    93
)
eval_num_examples_list=(
    30
    48
    63
    63
    48
    30
    18
)
query_format_list=(
    "binary"
    "multi_choice"
    "multi_answer"
)
train_seed=0
eval_seed=1000000
split_list=(
    "train"
    "eval"
)
version="v2.1"
output_base="./data_generation/generated_datasets/${version}/identification"

# --- Nested loops ---
for split in "${split_list[@]}"; do
for i in "${!num_nodes_list[@]}"; do
  num_nodes="${num_nodes_list[$i]}"
  if [[ "$split" == "train" ]]; then
    num_examples="${train_num_examples_list[$i]}"
    seed=$train_seed
  else
    num_examples="${eval_num_examples_list[$i]}"
    seed=$eval_seed
  fi

  output_dir="${output_base}/nodes${num_nodes}"
  # Each query format renders its own block of records. The integration step finds them by these indices.
  num_formats=${#query_format_list[@]}
  base=$((num_examples / num_formats))
  remainder=$((num_examples % num_formats))
  format_idx=0
  start_idx=0

  for query_format in "${query_format_list[@]}"; do
    block_size=$base
    if (( format_idx < remainder )); then
      block_size=$((block_size + 1))
    fi
    end_idx=$((start_idx + block_size))
    corpus_path=$(find "${output_base}/nodes${num_nodes}/triplets/jsonls"         -name "*-fmt${query_format}-seed${seed}*.jsonl" -print -quit)

    if [[ -z "$corpus_path" ]]; then
        echo "ERROR: No corpus found for nodes=${num_nodes}, query_format=${query_format}, seed=${seed}"
        exit 1
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    corpus_basename=$(basename "${corpus_path}" .jsonl)
    filename="${timestamp}-start${start_idx}-end${end_idx}-from-${corpus_basename}"
    log_file="${output_dir}/.logs/${filename}.log"

    mkdir -p "${output_dir}/.logs"

    echo "=== Run: ${filename} ==="
    echo "Source: ${corpus_path}"
    echo "Log: ${log_file}"

    python3 -m data_generation.identification.identification_question_assembly_pipeline \
        --triplet_corpus_path "${corpus_path}" \
        --start_idx "${start_idx}" \
        --end_idx "${end_idx}" \
        --output_dir "${output_dir}" \
        --run_timestamp "${timestamp}" \
    2>&1 | tee -a "${log_file}"

    start_idx=$end_idx
    format_idx=$((format_idx + 1))

  done
done
done
