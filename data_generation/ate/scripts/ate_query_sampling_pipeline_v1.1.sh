#!/usr/bin/env bash
set -euo pipefail

# --- Distribution: parallel arrays (#nodes <-> num_examples per split) ---
# ATE: nodes 3-10. Train samples 3V triplets, eval samples V triplets.
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
    120    # 3*40
    600    # 3*200
    960    # 3*320
    1200   # 3*400
    1200   # 3*400
    960    # 3*320
    600    # 3*200
    360    # 3*120
)
eval_num_examples_list=(
    16
    80
    128
    160
    160
    128
    80
    48
)

# --- Fixed hyperparameters (array style maintained) ---
max_unique_dags_list=(
    5000
)
atol_list=(
    1e-6
)
decimal_places_list=(
    1
    2
    3
    4
)
max_abnormal_ratio_list=(
    0.05
)
max_naive_conditioning_ratio_list=(
    0.3
)

train_seed=0
eval_seed=1000000
split_list=(
    "train"
    "eval"
)
version="v1.1"
output_base="./data_generation/generated_datasets/${version}/ate"

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

  for max_unique_dags in "${max_unique_dags_list[@]}"; do
  for atol in "${atol_list[@]}"; do
  for decimal_places in "${decimal_places_list[@]}"; do
  for max_abnormal_ratio in "${max_abnormal_ratio_list[@]}"; do
  for max_naive_conditioning_ratio in "${max_naive_conditioning_ratio_list[@]}"; do

    timestamp=$(date +%Y%m%d_%H%M%S)

    filename="${timestamp}-nodes${num_nodes}-examples${num_examples}-maxdags${max_unique_dags}-max_abnormal${max_abnormal_ratio}-atol${atol}-dp${decimal_places}-seed${seed}-max_naive_cond${max_naive_conditioning_ratio}"
    log_file="${output_dir}/triplets/.logs/${filename}.log"

    # Create log directory (variant directories are created by Python)
    mkdir -p "${output_dir}/triplets/.logs"

    echo "=== Run: ${filename} ==="
    echo "Log: ${log_file}"

    python3 -m data_generation.ate.ate_query_sampling_pipeline \
        --num_nodes "${num_nodes}" \
        --num_examples "${num_examples}" \
        --max_unique_dags "${max_unique_dags}" \
        --seed "${seed}" \
        --atol "${atol}" \
        --decimal_places "${decimal_places}" \
        --max_abnormal_ratio "${max_abnormal_ratio}" \
        --max_naive_conditioning_ratio "${max_naive_conditioning_ratio}" \
        --output_dir "${output_dir}" \
        --run_timestamp "${timestamp}" \
    2>&1 | tee -a "${log_file}"

  done
  done
  done
  done
  done
done
done
