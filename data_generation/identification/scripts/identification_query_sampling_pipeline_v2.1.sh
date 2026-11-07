#!/usr/bin/env bash
set -euo pipefail

# --- Distribution: parallel arrays (#nodes <-> num_examples per split) ---
# Identification: default wrappers target 4-option formats, so nodes 4-10.
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

# --- Fixed hyperparameters (array style maintained) ---
max_unique_dags_list=(
    5000
)
max_abnormal_ratio_list=(
    0
)
max_naive_conditioning_ratio_list=(
    0.3
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

  for query_format in "${query_format_list[@]}"; do
  for max_unique_dags in "${max_unique_dags_list[@]}"; do
  for max_abnormal_ratio in "${max_abnormal_ratio_list[@]}"; do
  for max_naive_conditioning_ratio in "${max_naive_conditioning_ratio_list[@]}"; do

    timestamp=$(date +%Y%m%d_%H%M%S)

    filename="${timestamp}-nodes${num_nodes}-examples${num_examples}-maxdags${max_unique_dags}-max_abnormal${max_abnormal_ratio}-fmt${query_format}-seed${seed}"
    if [[ "${max_naive_conditioning_ratio}" != "1.0" ]]; then
      filename="${filename}-max_naive_cond${max_naive_conditioning_ratio}"
    fi
    log_file="${output_dir}/triplets/.logs/${filename}.log"

    mkdir -p "${output_dir}/triplets/.logs"

    echo "=== Run: ${filename} ==="
    echo "Log: ${log_file}"

    python3 -m data_generation.identification.identification_query_sampling_pipeline         --num_nodes "${num_nodes}"         --num_examples "${num_examples}"         --max_unique_dags "${max_unique_dags}"         --seed "${seed}"         --query_format "${query_format}"         --max_abnormal_ratio "${max_abnormal_ratio}"         --max_naive_conditioning_ratio "${max_naive_conditioning_ratio}"         --output_dir "${output_dir}"         --run_timestamp "${timestamp}"     2>&1 | tee -a "${log_file}"

  done
  done
  done
  done
done
done
