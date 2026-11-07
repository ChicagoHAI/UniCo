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
max_unique_dags_list=(
    5000
)
query_format_list=(
    binary
    multi_choice
    multi_answer
)
max_abnormal_ratio_list=(
    0.05
)
train_seed=0
eval_seed=1000000
split_list=(
    "train"
    "eval"
)
version="v2.0"
output_base="./data_generation/generated_datasets/${version}/markov_blanket"

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

  for max_unique_dags in "${max_unique_dags_list[@]}"; do
  for query_format in "${query_format_list[@]}"; do
  for max_abnormal_ratio in "${max_abnormal_ratio_list[@]}"; do

    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="${output_dir}/triplets/.logs/${timestamp}.log"

    mkdir -p "${output_dir}/triplets/.logs"

    python3 -m data_generation.markov_blanket.markov_blanket_query_sampling_pipeline         --num_nodes "${num_nodes}"         --num_examples "${num_examples}"         --max_unique_dags "${max_unique_dags}"         --seed "${seed}"         --query_format "${query_format}"         --max_abnormal_ratio "${max_abnormal_ratio}"         --output_dir "${output_dir}"         --run_timestamp "${timestamp}"     2>&1 | tee -a "${log_file}"

  done
  done
  done
done
done
