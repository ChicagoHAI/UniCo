#!/usr/bin/env bash
set -euo pipefail

num_nodes_list=(
    3
    4
    5
    6
    7
    8
    9
    10
)
train_num_examples_list=(
    30
    150
    240
    300
    300
    240
    150
    90
)
eval_num_examples_list=(
    6
    30
    48
    60
    60
    48
    30
    18
)
dp_list=(
    1
    2
    3
    4
)
train_seed=0
eval_seed=1000000
split_list=(
    "train"
    "eval"
)
version="v2.0"
output_base="./data_generation/generated_datasets/${version}/conditional_prob"

for split in "${split_list[@]}"; do
for i in "${!num_nodes_list[@]}"; do
  num_nodes="${num_nodes_list[$i]}"
  if [[ "${split}" == "train" ]]; then
    num_examples="${train_num_examples_list[$i]}"
    seed="${train_seed}"
  else
    num_examples="${eval_num_examples_list[$i]}"
    seed="${eval_seed}"
  fi

  output_dir="${output_base}/nodes${num_nodes}"
  half_total=$((num_examples / 2))
  base_per_combo=$((half_total / 4))
  remainder=$((half_total % 4))
  start_idx=0

  for dp_idx in 0 1 2 3; do
    dp="${dp_list[$dp_idx]}"
    per_combo="${base_per_combo}"
    if (( dp_idx < remainder )); then
      per_combo=$((per_combo + 1))
    fi
    block_size=$((per_combo * 2))
    end_idx=$((start_idx + block_size))

    corpus_path=$(find "${output_dir}/triplets/jsonls" \
      -name "*-conditional_prob-*-dp${dp}-seed${seed}.jsonl" \
      -print \
      -quit)

    if [[ -z "${corpus_path}" ]]; then
      echo "ERROR: No corpus found for nodes=${num_nodes}, dp=${dp}, seed=${seed}"
      exit 1
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="${output_dir}/.logs/${timestamp}-code.log"

    mkdir -p "${output_dir}/.logs"

    python3 -m data_generation.conditional_prob.conditional_prob_question_assembly_pipeline_code \
      --triplet_corpus_path "${corpus_path}" \
      --start_idx "${start_idx}" \
      --end_idx "${end_idx}" \
      --output_dir "${output_dir}" \
      --run_timestamp "${timestamp}" \
    2>&1 | tee -a "${log_file}"

    start_idx="${end_idx}"
  done
done
done
